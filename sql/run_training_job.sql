-- run_training_job.sql
-- One-time environment setup + compute pool creation for the DeepSet training and evaluation pipelines.
-- Run steps 0-2, 2b, 3a, and 4 in Snowsight or SnowSQL.
-- Steps 3 and 3b (PUT) must be run in SnowSQL; PUT is not supported in Snowsight.
-- Training uses the Snowflake Container Runtime for ML.
-- For SnowSQL/connectors, YOUR_ACCOUNT_NAME means the Snowflake account identifier
-- from <account_identifier>.snowflakecomputing.com, preferably organization-account_name.

-- Step 0: Create database and schema
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
USE DATABASE TABPFN_DB;
CREATE SCHEMA IF NOT EXISTS TABPFN_SCHEMA;
USE SCHEMA TABPFN_SCHEMA;

-- Step 1: Create stages
-- Dedicated database and schema own project stages, compute pools, and model registry entries.
-- META_DATASET_STAGE: train/val/test synthetic parquet and staged benchmark datasets.
-- MLJobs materialize this stage into ephemeral container-local /tmp/data.
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- MODEL_STAGE: scripts, HPO config, and model checkpoints only.
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- EVALUATION_RESULTS_STAGE: all synthetic reports, benchmark part CSVs, and
-- the canonical final comparison file model_comparison.csv.
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- MLJOB_PAYLOAD_STAGE: bare stage name passed to submit_from_stage(stage_name=...).
-- It is separate from @MODEL_STAGE/scripts/, which is passed as source.
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- EPOCH_STAGE: output-only stage for epoch calibration JSON.
-- hpo_timing.json contains baseline, marginal sweep runs, and HPO wall-clock
-- estimates; train_timing.json contains production-topology epoch timing.
-- Runnable code is loaded from @MODEL_STAGE/scripts/.
CREATE STAGE IF NOT EXISTS EPOCH_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Step 2: Create compute pools
-- GPU_NV_M: 4 A10G GPUs per node. MAX_NODES=10 supports 5-node HPO
-- (20 concurrent one-GPU trials, 1 round of 20 trials) and 10-node DDP training
-- (40 DDP workers via num_workers_per_node=4).
-- CPU_X64_M: MAX_NODES=3 supports the three combined CPU baseline shard jobs.
-- Each shard owns a deterministic dataset subset and evaluates all configured
-- baseline methods across all configured seeds. Oversized CPU rows emit NaN
-- skip rows instead of exhausting container memory.
-- AUTOGLUON_CPU_POOL uses CPU_X64_M for AutoGluon shard jobs:
-- 30 independent single-node shard jobs, each loading one owned dataset at a time.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY; drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = GPU_NV_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;
CREATE COMPUTE POOL DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 3
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 30
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

-- Verify pools reach ACTIVE before submitting the job:
SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';
SHOW COMPUTE POOLS LIKE 'DEEPSET_CPU_POOL';
SHOW COMPUTE POOLS LIKE 'AUTOGLUON_CPU_POOL';

-- Step 2b: Allow benchmark jobs to fetch datasets from inside Snowflake
-- Run with a role that can create network rules in TABPFN_SCHEMA.
-- OpenML hosts are required for evaluation. Kaggle API/www and Google storage hosts are
-- required for the one-off Kaggle download job. Benchmark shard jobs consume only
-- prepared .npz files; only the prep MLJob may use this network route and its
-- OpenML pip dependency.
USE ROLE SYSADMIN;
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

CREATE OR REPLACE NETWORK RULE openml_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'www.openml.org',
    'openml.org',
    'api.openml.org',
    'pypi.org',
    'files.pythonhosted.org'
  );

-- If CALL download_kaggle_to_stage() fails with host='api.kaggle.com', recreate
-- this network rule and the benchmark_external_access integration below.

CREATE OR REPLACE NETWORK RULE kaggle_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'api.kaggle.com',
    'www.kaggle.com',
    'kaggle.com',
    'storage.googleapis.com',
    'pypi.org',
    'files.pythonhosted.org'
  );

-- Kaggle API authentication requires the Kaggle account username/handle,
-- not the email address used to log in. A 401 Unauthorized from
-- DownloadDataFiles means Kaggle rejected these secret values.
-- If that happens, generate a fresh Kaggle API token and recreate this secret.
-- Keep this statement in Snowsight/SnowSQL history only; do not commit real keys.
CREATE OR REPLACE SECRET KAGGLE_API_SECRET
  TYPE = PASSWORD
  USERNAME = '<kaggle_account_username_not_email>'
  PASSWORD = '<kaggle_api_key>';

-- Run with ACCOUNTADMIN or another role granted CREATE EXTERNAL ACCESS INTEGRATION.
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION benchmark_external_access
  ALLOWED_NETWORK_RULES = (openml_network_rule, kaggle_network_rule)
  ALLOWED_AUTHENTICATION_SECRETS = (KAGGLE_API_SECRET)
  ENABLED = TRUE;

-- If a non-admin role submits the procedure or MLJobs, grant it least-privilege access:
-- GRANT READ ON SECRET KAGGLE_API_SECRET TO ROLE <job_submitter_role>;
-- GRANT USAGE ON INTEGRATION benchmark_external_access TO ROLE <job_submitter_role>;

-- =====================================================================
-- Step 2c: General PyPI External Access Integration for Snowflake-managed Container Runtime
--
--   Allows Snowflake ML Jobs to install approved third-party PyPI packages
--   (CatBoost, OpenML, AutoGluon, etc.) that are NOT preinstalled in the
--   2.5.0-py311 managed runtime image. Used for all pip_requirements jobs.
--
-- Required privilege: ACCOUNTADMIN (or role with CREATE INTEGRATION and
--   access to SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE).
-- Run this block once. After creation, grant USAGE to the role that
--   executes stored procedures and submits ML Jobs.
-- =====================================================================

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TABPFN_PYPI_EAI
  ALLOWED_NETWORK_RULES = (SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE)
  ENABLED = TRUE
  COMMENT = 'Allows TabPFN ML Jobs to install approved PyPI dependencies in Snowflake-managed Container Runtime.';

GRANT USAGE ON INTEGRATION TABPFN_PYPI_EAI TO ROLE ACCOUNTADMIN;

-- Verification:
-- SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'TABPFN_PYPI_EAI';
-- DESC EXTERNAL ACCESS INTEGRATION TABPFN_PYPI_EAI;
-- SHOW GRANTS ON INTEGRATION TABPFN_PYPI_EAI;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- Step 3: Upload training and benchmark data (SnowSQL only)
-- Run the three REMOVE lines first to clear stale files before re-uploading:
-- REMOVE @META_DATASET_STAGE/train/;
-- REMOVE @META_DATASET_STAGE/val/;
-- REMOVE @META_DATASET_STAGE/test/;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/kaggle/*.npz    @META_DATASET_STAGE/kaggle/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
-- Verify:
-- LIST @META_DATASET_STAGE/train/;
-- LIST @META_DATASET_STAGE/val/;
-- LIST @META_DATASET_STAGE/test/;
-- LIST @META_DATASET_STAGE/kaggle/;

-- Step 3a: Create and verify the metadata pruning index for HPO
-- META_DATASET_INDEX is a table-backed pruning layer over parquet payloads in
-- @META_DATASET_STAGE. It is not a replacement for staged parquet storage.
-- HPO should query this table to choose a deterministic balanced subset, then
-- materialize only those selected stage_path payloads into the MLJob container.
-- One-time migration: DROP TABLE IF EXISTS META_DATASET_INDEX;
-- (Drops the permanent version so it can be recreated as TRANSIENT)
CREATE TRANSIENT TABLE IF NOT EXISTS META_DATASET_INDEX (
  split STRING NOT NULL,
  task_id STRING NOT NULL,
  stage_path STRING NOT NULL,
  n NUMBER,
  p NUMBER,
  n_train NUMBER,
  n_test NUMBER,
  prior_regime STRING,
  hpo_bucket NUMBER
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p, n_train);

-- Step 3b: Benchmark manifest metadata index
-- BENCHMARK_DATASET_INDEX is rebuilt by prepare_benchmark_datasets() from
-- benchmark_manifest.json metadata. It supports manifest completeness checks
-- and optional BENCHMARK_SHARD_STRATEGY=balanced assignment by benchmark_weight.
-- Prepared .npz files remain opaque staged payloads. Do not split, cluster, or
-- transform them for Snowflake query performance; benchmark jobs download exact
-- stage_path files only.
-- Snowflake automatic micro-partitioning is sufficient for this small metadata
-- table. Add a clustering key only after measured query benefit on a large
-- manifest, and keep it to a few useful columns such as (source, dataset_index).
CREATE TRANSIENT TABLE IF NOT EXISTS BENCHMARK_DATASET_INDEX (
  dataset_index NUMBER NOT NULL,
  source STRING,
  task_id STRING,
  dataset_id STRING,
  name STRING,
  stage_path STRING,
  n_samples NUMBER,
  n_features NUMBER,
  estimated_bytes NUMBER,
  benchmark_weight NUMBER,
  created_at_utc TIMESTAMP_NTZ
)
DATA_RETENTION_TIME_IN_DAYS = 0;

-- Populate/rebuild guidance:
-- 1. Upload staged parquet to @META_DATASET_STAGE/{train,val,test}/.
-- 2. Upload scripts to @MODEL_STAGE/scripts/.
-- 3. Run CALL build_meta_dataset_index(); to rebuild from staged parquet
--    metadata inside Snowflake. Re-run it whenever synthetic parquet is
--    regenerated or restaged.

-- Expected full split counts before HPO/training:
SELECT split, COUNT(*) AS task_count
FROM META_DATASET_INDEX
GROUP BY split
ORDER BY split;

-- Copy-paste assertions for the current generated dataset:
SELECT COUNT(*) AS train_tasks FROM META_DATASET_INDEX WHERE split = 'train'; -- expected 800
SELECT COUNT(*) AS val_tasks   FROM META_DATASET_INDEX WHERE split = 'val';   -- expected 100
SELECT COUNT(*) AS test_tasks  FROM META_DATASET_INDEX WHERE split = 'test';  -- expected 100

-- Inspect clustering/pruning health after loading enough rows:
SELECT SYSTEM$CLUSTERING_INFORMATION(
  'META_DATASET_INDEX',
  '(split, hpo_bucket, prior_regime, p, n_train)'
);

-- Example deterministic HPO subset checks. These mirror the intended default:
-- 200 train tasks and 40 validation tasks, balanced by hpo_bucket and stable
-- within buckets by task_id. HPO should use equivalent table-backed selection
-- before materializing payloads from @META_DATASET_STAGE.
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY split, hpo_bucket
      ORDER BY prior_regime, p, n_train, task_id
    ) AS bucket_rank
  FROM META_DATASET_INDEX
  WHERE split IN ('train', 'val')
),
selected AS (
  SELECT *
  FROM ranked
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY split
    ORDER BY bucket_rank, hpo_bucket, prior_regime, p, n_train, task_id
  ) <= IFF(split = 'train', 200, 40)
)
SELECT split, COUNT(*) AS selected_rows
FROM selected
GROUP BY split
ORDER BY split;

-- Confirm selected payload paths are concrete staged parquet references:
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY split, hpo_bucket
      ORDER BY prior_regime, p, n_train, task_id
    ) AS bucket_rank
  FROM META_DATASET_INDEX
  WHERE split IN ('train', 'val')
),
selected AS (
  SELECT *
  FROM ranked
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY split
    ORDER BY bucket_rank, hpo_bucket, prior_regime, p, n_train, task_id
  ) <= IFF(split = 'train', 200, 40)
)
SELECT split, stage_path, n, p, n_train, n_test, prior_regime, hpo_bucket
FROM selected
ORDER BY split, hpo_bucket, task_id
LIMIT 20;

-- Step 3b: Upload Python scripts (SnowSQL only)
-- Re-run whenever any script changes:
-- PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- After refactor: run_epoch_tests.py is now its own handler file (was in run_training_job.py).
-- PUT file://C:/Documents/TabPFN_DemandModel/scripts/run_epoch_tests.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- After refactor: run_evaluation_test.py is the evaluation handler (was in run_training_job.py).
-- PUT file://C:/Documents/TabPFN_DemandModel/scripts/run_evaluation_test.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- New: prepare_benchmark_datasets.py is the benchmark dataset prep handler (covered by src/*.py wildcard above).
-- PUT file://C:/Documents/TabPFN_DemandModel/src/prepare_benchmark_datasets.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- Evaluation runtime preflight uses runtime_probe.py (covered by src/*.py wildcard above).
-- PUT file://C:/Documents/TabPFN_DemandModel/src/runtime_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
-- Verify:
-- LIST @MODEL_STAGE/scripts/;
-- Epoch calibration jobs also run from @MODEL_STAGE/scripts/. @EPOCH_STAGE is
-- output-only for hpo_timing.json, hpo_epoch_error.json, train_timing.json, and
-- train_epoch_error.json.
-- Before epoch calibration, verify the shared modules and entrypoints:
-- LIST @MODEL_STAGE/scripts/ PATTERN='.*(hpo|hpo_epoch_test|train_epoch_test|train|model|snowflake_io)[.]py';

-- Step 4: Create and call the orchestrator stored procedure
-- download_kaggle_to_stage() is a separate setup job. Run it after scripts are
-- uploaded and before the first benchmark/training run that needs Kaggle data.
CREATE OR REPLACE PROCEDURE download_kaggle_to_stage()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_kaggle_download';

-- build_meta_dataset_index() launches build_meta_dataset_index.py on the CPU
-- pool. It lists @META_DATASET_STAGE/{train,val,test}/, reads scalar parquet
-- metadata, truncates/rebuilds META_DATASET_INDEX, and validates 800/100/100.
CREATE OR REPLACE PROCEDURE build_meta_dataset_index()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_dataset_index';

-- run_pretrain_pipeline() trains with default hyperparameters and writes
-- @MODEL_STAGE/checkpoints/pretrain.pt. Run before run_hpo_pipeline() so
-- HPO trials warm-start from the pre-trained weights.
CREATE OR REPLACE PROCEDURE run_pretrain_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_pretrain_job.py')
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline';

-- run_hpo_pipeline() launches hpo.py on the GPU pool using Ray Tune for distributed HPO.
-- Produces @MODEL_STAGE/hpo/best_config.json on success or hpo_failure.json
-- if the Python driver starts and then fails. Per-trial errors appear in Ray
-- worker logs (visible in Snowsight container logs), not in hpo_failure.json.
CREATE OR REPLACE PROCEDURE run_hpo_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline';

-- run_model_training() reads @MODEL_STAGE/hpo/best_config.json, passes it to
-- train.py as BEST_CONFIG, and produces @MODEL_STAGE/checkpoints/best.pt.
CREATE OR REPLACE PROCEDURE run_model_training()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_training';

CREATE OR REPLACE PROCEDURE run_training_runtime_probe(target_instances INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_training_runtime_probe';

-- Usage:
--   CALL run_training_runtime_probe(1);   -- single-node probe
--   CALL run_training_runtime_probe(2);   -- 2-node probe (optional)
--   CALL run_training_runtime_probe(5);   -- 5-node probe (optional)
--   CALL run_training_runtime_probe(10);  -- full-topology probe
--
-- Expected success markers in job logs:
--   [runtime_probe] entered Python
--   [runtime_probe] completed
--
-- If logs show Prometheus mmap panic before '[runtime_probe] entered Python',
-- escalate to Snowflake Support as a managed MLJob/Ray/Prometheus runtime issue.

-- run_training_pipeline() runs the full 3-phase pipeline in sequence:
--   Phase 1 (Pretrain)       → @MODEL_STAGE/checkpoints/pretrain.pt
--   Phase 2 (HPO fine-tune)  → @MODEL_STAGE/hpo/best_config.json
--   Phase 3 (Final training) → @MODEL_STAGE/checkpoints/best.pt
-- HPO trials and final training both warm-start from pretrain.pt when
-- architecture matches; otherwise fall back to random init.
CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';

-- run_hpo_epoch_test() runs baseline and marginal HPO epoch timing sweeps.
-- Result at @EPOCH_STAGE/hpo_timing.json; read with SELECT below.
CREATE OR REPLACE PROCEDURE run_hpo_epoch_test()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_epoch_tests.py')
  HANDLER = 'run_epoch_tests.run_hpo_epoch_test';

-- run_train_epoch_test() runs one DDP training epoch with the production
-- GPU_NV_M topology: 10 nodes x 4 workers/node = world_size 40.
-- Result at @EPOCH_STAGE/train_timing.json; read with SELECT below.
CREATE OR REPLACE PROCEDURE run_train_epoch_test()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_epoch_tests.py')
  HANDLER = 'run_epoch_tests.run_train_epoch_test';

-- prepare_benchmark_datasets() fetches OpenML/Kaggle benchmark datasets once,
-- stages them to @META_DATASET_STAGE/benchmark_prepared/, and writes
-- benchmark_manifest.json. Run before run_evaluation_pipeline(), or let
-- run_evaluation_pipeline() call it automatically.
-- @META_DATASET_STAGE/benchmark_prepared/ is created by this procedure.
-- It contains benchmark_manifest.json and prepared .npz files for all benchmark datasets.
-- Benchmark shard jobs read exact prepared files from this prefix; they do not
-- call OpenML directly and they load only one owned dataset at a time.
-- The procedure also refreshes BENCHMARK_DATASET_INDEX from manifest metadata.
-- To rebuild: set BENCHMARK_FORCE_REBUILD=true env var before calling this procedure.
CREATE OR REPLACE PROCEDURE prepare_benchmark_datasets()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python', 'openml==0.15.1')
  EXTERNAL_ACCESS_INTEGRATIONS = (BENCHMARK_EXTERNAL_ACCESS)
  IMPORTS = ('@MODEL_STAGE/scripts/prepare_benchmark_datasets.py')
  HANDLER = 'prepare_benchmark_datasets.prepare_datasets';

-- run_evaluation_pipeline() requires @MODEL_STAGE/checkpoints/best.pt and
-- @MODEL_STAGE/scripts/runtime_probe.py. It preflights compute pools and
-- configured evaluation runtime images with runtime-specific REQUIRED_IMPORTS,
-- then runs synthetic evaluation and benchmark dataset preparation. The prep
-- job always runs as a lightweight manifest/BENCHMARK_DATASET_INDEX validation
-- step before shard submission, even when benchmark_manifest.json already exists.
CREATE OR REPLACE PROCEDURE run_evaluation_pipeline(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_pipeline';

-- Drop old zero-argument overload if it exists:
DROP PROCEDURE IF EXISTS run_evaluation_pipeline();

-- Call:
-- CALL run_evaluation_pipeline('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');

-- run_evaluation_runtime_probes() runs all preflight checks without submitting
-- evaluation jobs. Use this to validate runtime environments before a full run.
CREATE OR REPLACE PROCEDURE run_evaluation_runtime_probes(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_runtime_probes';

-- Call:
-- CALL run_evaluation_runtime_probes('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');

-- run_evaluation_capacity_probe() is a lightweight quota/capacity check. It submits
-- capacity_probe.py in 3 non-overlapping phases matching the fixed evaluation pipeline
-- envelope (GPU=10, CPU=3, AutoGluon=30). Run between runtime probes and the full pipeline.
CREATE OR REPLACE PROCEDURE run_evaluation_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_capacity_probe';

-- Call:
-- CALL run_evaluation_capacity_probe('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');

-- Split-phase evaluation: run each phase independently to release quota between pools.
-- Recommended run sequence under tight node quota:
--   CALL run_evaluation_runtime_probes('<prep>', '<bench>', '<ag>');
--   CALL run_evaluation_prep('<prep>', '<bench>', '<ag>');
--   CALL run_deepset_evaluation('<prep>', '<bench>', '<ag>');
--   ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
--   CALL run_baseline_evaluation('<prep>', '<bench>', '<ag>');
--   ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
--   CALL run_autogluon_evaluation('<prep>', '<bench>', '<ag>');
--   ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
--   CALL run_evaluation_aggregation('<prep>', '<bench>', '<ag>');
--
-- run_evaluation_prep() fetches/validates benchmark manifest and index on DEEPSET_CPU_POOL.
CREATE OR REPLACE PROCEDURE run_evaluation_prep(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_prep';

-- run_deepset_evaluation() runs synthetic eval and 10 DeepSet GPU shards on DEEPSET_GPU_POOL.
-- Requires @MODEL_STAGE/checkpoints/best.pt.
CREATE OR REPLACE PROCEDURE run_deepset_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_deepset_evaluation';

-- run_baseline_evaluation() runs 3 CPU baseline benchmark shards on DEEPSET_CPU_POOL.
CREATE OR REPLACE PROCEDURE run_baseline_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_baseline_evaluation';

-- run_autogluon_evaluation() runs 30 AutoGluon shards (max 30 concurrent) on AUTOGLUON_CPU_POOL.
CREATE OR REPLACE PROCEDURE run_autogluon_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_autogluon_evaluation';

-- run_evaluation_aggregation() runs the benchmark aggregation job on DEEPSET_CPU_POOL
-- and returns a listing of @EVALUATION_RESULTS_STAGE. Can be re-run without re-running
-- prior phases if benchmark_parts/ files already exist on stage.
CREATE OR REPLACE PROCEDURE run_evaluation_aggregation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_aggregation';

CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;

-- Repair for Kaggle 401 Unauthorized:
-- USE ROLE SYSADMIN;
-- USE DATABASE TABPFN_DB;
-- USE SCHEMA TABPFN_SCHEMA;
-- CREATE OR REPLACE SECRET KAGGLE_API_SECRET
--   TYPE = PASSWORD
--   USERNAME = '<kaggle_account_username_not_email>'
--   PASSWORD = '<new_kaggle_api_key>';
-- Then re-run:
-- CALL download_kaggle_to_stage();
-- LIST @META_DATASET_STAGE/kaggle/;

-- Step 4b: Epoch calibration (run before upgrading compute pool)
-- Requires @MODEL_STAGE/scripts/ populated with src/*.py and scripts/*.py (Step 3b above).
LIST @MODEL_STAGE/scripts/ PATTERN='.*(hpo_epoch_test|train_epoch_test|train|model|snowflake_io)[.]py';
CALL run_hpo_epoch_test();
SELECT $1 FROM @EPOCH_STAGE/hpo_timing.json (FILE_FORMAT => (TYPE = JSON));
-- Inspect HPO timing by phase, not only epoch_time_s. Query wall time includes
-- MLJob startup, Ray Tune scheduling, metadata selection, stage materialization,
-- and trial epoch compute.
SELECT
  $1:metadata:metadata_selection_time_s::FLOAT AS metadata_selection_time_s,
  $1:metadata:materialization_time_s::FLOAT AS materialization_time_s,
  $1:summary:mean_epoch_time_s::FLOAT AS mean_epoch_time_s,
  $1:summary:max_epoch_time_s::FLOAT AS max_epoch_time_s,
  $1:summary:parallel_trials::NUMBER AS parallel_trials,
  $1:summary:hpo_rounds::NUMBER AS hpo_rounds,
  $1:summary:estimated_hpo_wall_time_s_mean::FLOAT AS estimated_hpo_wall_time_s_mean,
  $1:summary:estimated_hpo_wall_time_s_conservative::FLOAT AS estimated_hpo_wall_time_s_conservative
FROM @EPOCH_STAGE/hpo_timing.json (FILE_FORMAT => (TYPE = JSON));

CALL run_train_epoch_test();
SELECT $1 FROM @EPOCH_STAGE/train_timing.json (FILE_FORMAT => (TYPE = JSON));
-- Decision gate:
--   summary.parallel_trials = 20 and summary.hpo_rounds = 1
--       -> 5 nodes GPU_NV_M (20 concurrent, ~30-60 min HPO including overhead)
--   high materialization time
--       -> inspect metadata selection/materialization before changing topology
--   summary.max_epoch_time_s > 30 s    -> re-evaluate; consider GPU_NV_L or reducing num_trials

-- Preferred staged training: HPO writes best_config.json; training consumes it and writes best.pt.
-- Rebuild META_DATASET_INDEX before HPO/training whenever staged synthetic parquet changes.
-- Pre-warm: issue RESUME so the GPU pool transitions SUSPENDED→ACTIVE while the
-- CPU index job runs. With AUTO_RESUME=TRUE the pool also starts on job submission,
-- but this moves the ~3-5 min startup wait off the critical path of run_hpo_pipeline().
ALTER COMPUTE POOL DEEPSET_GPU_POOL RESUME;

-- This runs on DEEPSET_CPU_POOL (GPU pool starts warming in background):
CALL build_meta_dataset_index();
-- Verify full split counts are 800/100/100 and deterministic HPO subset query
-- above returns 200 train rows and 40 val rows.

-- GPU pool should be ACTIVE by now; no startup wait inside the jobs:
-- Runtime probe (run before final training if MLJob startup failures are suspected):
CALL run_training_runtime_probe(1);   -- single-node probe
CALL run_training_runtime_probe(2);   -- 2-node probe (optional)
CALL run_training_runtime_probe(5);   -- 5-node probe (optional)
CALL run_training_runtime_probe(10);  -- full-topology probe
CALL run_pretrain_pipeline();
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain[.]pt';
CALL run_hpo_pipeline();
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));
CALL run_model_training();
LIST @MODEL_STAGE/checkpoints/;

-- Optional one-call training convenience wrapper:
-- CALL run_training_pipeline();

-- Evaluation only: requires @MODEL_STAGE/checkpoints/best.pt and does not read best_config.json.
-- Step 1: Validate runtime images (serialized probes, no model/data loaded).
CALL run_evaluation_runtime_probes(
  '<prep_runtime_image_name>',
  '<benchmark_runtime_image_name>',
  '<autogluon_runtime_image_name>'
);

-- Step 2: Validate node quota (capacity_probe.py, 3 phases: GPU=10, CPU=3, AutoGluon=30).
-- If this fails with a node limit error: SHOW COMPUTE POOLS; suspend idle pools;
-- wait for active jobs to finish; or request higher Snowflake account node quota.
CALL run_evaluation_capacity_probe(
  '<prep_runtime_image_name>',
  '<benchmark_runtime_image_name>',
  '<autogluon_runtime_image_name>'
);

-- Step 3 (recommended under tight quota): Split-phase evaluation.
-- Run each phase independently and suspend its pool to release quota before the next.
CALL run_evaluation_prep('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');

CALL run_deepset_evaluation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;

CALL run_baseline_evaluation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;

CALL run_autogluon_evaluation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

CALL run_evaluation_aggregation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');

-- Step 3 (legacy convenience, holds all 3 pools simultaneously):
-- CALL run_evaluation_pipeline(
--   '<prep_runtime_image_name>',
--   '<benchmark_runtime_image_name>',
--   '<autogluon_runtime_image_name>'
-- );

-- Step 5: Verify output
LIST @MODEL_STAGE/hpo/;
LIST @MODEL_STAGE/checkpoints/;
LIST @EVALUATION_RESULTS_STAGE/;

-- Step 6: Download outputs (SnowSQL only)
-- GET @MODEL_STAGE/checkpoints/best.pt 'file://C:/Documents/TabPFN_DemandModel/results/';
-- GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
