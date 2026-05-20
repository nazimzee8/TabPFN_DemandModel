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

-- EVALUATION_DATASET_STAGE: staged synthetic regression and OOD evaluation
-- input datasets only. Production training parquet stays in META_DATASET_STAGE.
CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

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
-- CPU_X64_M: MAX_NODES=6 supports the six combined CPU baseline shard jobs.
-- Each shard owns a deterministic dataset subset and evaluates all configured
-- baseline methods across all configured seeds. Oversized CPU rows emit NaN
-- skip rows instead of exhausting container memory.
-- AUTOGLUON_CPU_POOL uses CPU_X64_M for AutoGluon shard jobs (combined suite):
-- Default: 6 logical AutoGluon work-item clusters, 4 target instances per cluster.
-- Each cluster runs evaluate_synthetic_regression_autogluon_ray.py with Ray for
-- distributed independent work items. Maximum concurrent nodes = 6 * 4 = 24.
-- Runtime concurrency can be tuned via SYNREG_AUTOGLUON_CLUSTER_SHARDS /
-- SYNREG_AUTOGLUON_WORKERS_PER_SHARD / SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS
-- or procedure overload arguments when quota is temporarily constrained.
-- Run run_synthetic_regression_combined_autogluon_capacity_probe before evaluation.
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
  MAX_NODES = 6
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 60
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

-- Step 3c: Synthetic regression dataset index
-- SYNTHETIC_REGRESSION_DATASET_INDEX is shared across all synthetic evaluation suites,
-- differentiated by suite_id. It is rebuilt by prepare_synthetic_regression.py (in-distribution)
-- and prepare_ood_regression.py (OOD pilot and full suite). Each prep job deletes only its own
-- suite_id rows; the table is never dropped unless SYNTHETIC_REGRESSION_DROP_INDEX_TABLE=true.
-- logical_dataset_key has zero-padded format: {suite_id}:{prior_regime}:{dataset_id:04d}
-- source_suite_id: populated for combined suites (e.g. linear_all_v1) to record
--   which source suite contributed each row; NULL for primary suites.
CREATE TRANSIENT TABLE IF NOT EXISTS SYNTHETIC_REGRESSION_DATASET_INDEX (
  suite_id             STRING,
  suite_family         STRING,
  dataset_id           NUMBER,
  dataset_seed         NUMBER,
  stage_path           STRING,
  prior_name           STRING,
  prior_version        STRING,
  prior_regime         STRING,
  split_seeds          ARRAY,
  n_total              NUMBER,
  n_train_default      NUMBER,
  n_holdout_default    NUMBER,
  p_signal             NUMBER,
  p_noise              NUMBER,
  p_total              NUMBER,
  target_noise_scale   FLOAT,
  training_size_anchor BOOLEAN,
  feature_noise_level  NUMBER,
  eval_weight          FLOAT,
  payload_bytes        NUMBER,
  created_at           TIMESTAMP_NTZ,
  logical_dataset_key  STRING,
  source_suite_id      STRING
) DATA_RETENTION_TIME_IN_DAYS = 0;

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
-- Zero-arg form uses env-var defaults:
--   CALL run_pretrain_pipeline();
-- Parameterized form:
--   CALL run_pretrain_pipeline(
--       'market_exchangeable_icl',       -- MODEL_FAMILY
--       'synthetic_regression_combined', -- TRAINING_DATA_FAMILY
--       'inductive_forecasting'          -- MODEL_DESIGN_PATTERN
--   );
CREATE OR REPLACE PROCEDURE run_pretrain_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_pretrain_job.py')
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline';

CREATE OR REPLACE PROCEDURE run_pretrain_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_pretrain_job.py')
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline_model';

-- run_hpo_pipeline() launches hpo.py on the GPU pool using Ray Tune for distributed HPO.
-- Produces @MODEL_STAGE/hpo/best_config.json on success or hpo_failure.json on failure.
-- best_config.json keys: lr, weight_decay, dropout, d_phi, d_rho, pool, n_sab_feat,
--   use_ridge_expert, ridge_lambda, gate_hidden_dim, use_huber, huber_delta, lambda_l1,
--   model_family, model_arch_version, model_design_pattern, hpo_sweep_mode.
-- Note: HPO only supports inductive_forecasting; transductive_completion raises in hpo.py.
--
-- HPO_SWEEP_MODE controls the search space (default: ridge_residual):
--   ridge_residual — fixed architecture (d_phi=128, n_sab_feat=1); tunes optimizer,
--                    Ridge Expert (ridge_lambda, gate_hidden_dim), and robust loss params.
--   architecture   — tunes d_phi (from [64,128,192,256]) and n_sab_feat (from [1,2]);
--                    Ridge Expert params also tuned; cold-start allowed on mismatch.
--
-- These are two separate HPO executions using the SAME procedure with different HPO_SWEEP_MODE.
-- Always run ridge_residual first; run architecture only after memory probes confirm safety.
--
-- Zero-arg (uses env-var defaults including HPO_SWEEP_MODE):
--   CALL run_hpo_pipeline();
-- Three-arg (explicit model selectors; uses HPO_SWEEP_MODE env-var default):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting'
--   );
-- Four-arg (explicit model selectors + explicit HPO_SWEEP_MODE):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual'
--   );
CREATE OR REPLACE PROCEDURE run_hpo_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline';

CREATE OR REPLACE PROCEDURE run_hpo_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline_model';

-- Four-arg overload: explicit HPO_SWEEP_MODE selector.
-- Use 'ridge_residual' for production-safe tuning; 'architecture' for d_phi/n_sab_feat exploration.
--
-- Sweep 1 — Ridge Expert / residual / optimizer tuning (run first):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual'
--   );
-- Sweep 2 — memory-gated architecture exploration (run after ridge_residual is stable):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'architecture'
--   );
CREATE OR REPLACE PROCEDURE run_hpo_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  HPO_SWEEP_MODE STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline_model_sweep';

-- Five-arg overload: explicit HPO_SWEEP_MODE + baseline config stage path.
-- Use for architecture sweep to freeze optimizer/regularization from sweep 1.
-- HPO_BASELINE_CONFIG_STAGE_PATH must point to best_config_ridge_residual.json
-- written by sweep 1 (four-arg or five-arg call with HPO_SWEEP_MODE='ridge_residual').
-- Pass '' (empty string) for HPO_BASELINE_CONFIG_STAGE_PATH when HPO_SWEEP_MODE='ridge_residual'.
--
-- Two-sweep HPO (recommended):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual', ''
--   );
--   CALL run_model_ddp_memory_probe(
--       'inductive_forecasting', 'market_exchangeable_icl',
--       200, 128, 128, 256, 2, TRUE
--   );
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'architecture',
--       '@MODEL_STAGE/hpo/best_config_ridge_residual.json'
--   );
CREATE OR REPLACE PROCEDURE run_hpo_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  HPO_SWEEP_MODE STRING,
  HPO_BASELINE_CONFIG_STAGE_PATH STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline_model_sweep_with_baseline';

-- run_model_training() reads @MODEL_STAGE/hpo/best_config.json, passes it to
-- train.py as BEST_CONFIG, and produces @MODEL_STAGE/checkpoints/best.pt.
-- Checkpoint metadata includes: model_family, task_type, training_data_family,
--   best_val_mse, train_mse_at_best, best_epoch, pytorch_version.
-- Zero-arg form uses env-var defaults (MODEL_FAMILY, TRAINING_DATA_FAMILY, MODEL_DESIGN_PATTERN):
CREATE OR REPLACE PROCEDURE run_model_training()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_training';

-- Parameterized form — same explicit runtime lineage variables as pretrain and HPO:
--   CALL run_model_training(
--       'market_exchangeable_icl',       -- MODEL_FAMILY
--       'synthetic_regression_combined', -- TRAINING_DATA_FAMILY
--       'inductive_forecasting'          -- MODEL_DESIGN_PATTERN
--   );
CREATE OR REPLACE PROCEDURE run_model_training(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_training_model';

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

-- run_training_pipeline() runs the full 7-step two-sweep pipeline in sequence:
--   Step 1 (Validate)         META_DATASET_INDEX counts, columns, stage access
--   Step 2 (Pretrain)       → @MODEL_STAGE/checkpoints/pretrain.pt
--   Step 3 (HPO sweep 1)    → @MODEL_STAGE/hpo/best_config_ridge_residual.json
--                              + @MODEL_STAGE/hpo/best_config.json (same content)
--   Step 4 (Memory probe)     Worst-case probe (d_phi=256, n_blocks=2) — mandatory gate
--   Step 5 (HPO sweep 2)    → @MODEL_STAGE/hpo/best_config_architecture.json
--                              + @MODEL_STAGE/hpo/best_config.json (merged)
--   Step 6 (Load config)      Downloads and parses merged best_config.json
--   Step 7 (Final training) → @MODEL_STAGE/checkpoints/best.pt
-- HPO trials warm-start from pretrain.pt; architecture mismatch policy is
-- set automatically from best_config.hpo_sweep_mode (require_match for
-- ridge_residual; allow_cold_start_on_arch_mismatch for architecture).
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

-- run_model_ddp_memory_probe() measures peak CUDA memory per DDP worker for a
-- representative MODEL3 ICL shape before pretrain / HPO / final training.
--
-- Run RUN_BACKWARD=TRUE (always) because MODEL3 meta-training uses back-propagation;
-- the probe with backward gives a faithful peak-memory measurement for training-regime
-- validation that covers gradient tensors and activation storage during backprop.
--
-- Launches model_ddp_memory_probe.py on DEEPSET_GPU_POOL with the same topology as
-- training (TRAIN_NUM_NODES=10 nodes x 4 workers = world_size 40).
--
-- Diagnostic JSON uploaded to @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json.
--
-- Usage:
--   CALL run_training_runtime_probe(1);
--   CALL run_training_runtime_probe(10);
--
--   CALL run_model_ddp_memory_probe(
--       'inductive_forecasting',
--       'market_exchangeable_icl',
--       200,    -- N_CONTEXT
--       128,    -- P_FEATURES
--       128,    -- M_QUERY
--       128,    -- D_PHI
--       1,      -- N_BLOCKS
--       TRUE    -- RUN_BACKWARD (always TRUE for training-regime validation)
--   );
--
--   LIST @MODEL_STAGE/diagnostics/;
--   SELECT $1
--   FROM @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json
--     (FILE_FORMAT => (TYPE = JSON));
CREATE OR REPLACE PROCEDURE run_model_ddp_memory_probe(
  MODEL_DESIGN_PATTERN STRING,
  MODEL_FAMILY STRING,
  N_CONTEXT INTEGER,
  P_FEATURES INTEGER,
  M_QUERY INTEGER,
  D_PHI INTEGER,
  N_BLOCKS INTEGER,
  RUN_BACKWARD BOOLEAN
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_ddp_memory_probe';

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

-- ============================================================
-- Main Synthetic Regression Pipeline — Split-Phase Stored Procedures
-- (linear_poisson_v1_recommended, 200 datasets, all methods)
-- ============================================================

CREATE OR REPLACE PROCEDURE run_synthetic_regression_runtime_probes(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_runtime_probes';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_baseline_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_baseline_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_prep(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_prep';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_deepset_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_deepset_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_baseline_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_baseline_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'F
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_aggregation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_aggregation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_pipeline(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_pipeline';

-- Split-phase execution (recommended under tight quota):
CALL run_synthetic_regression_prep('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_deepset_evaluation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_baseline_evaluation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_synthetic_regression_autogluon_evaluation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
CALL run_synthetic_regression_aggregation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
-- All-in-one (convenience):
-- CALL run_synthetic_regression_pipeline('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');

-- ============================================================
-- OOD Full Suite Evaluation (ood_linear_full_v1, 200 datasets)
-- ============================================================
-- Runbook:
--   1. Generate 200 OOD parquet files locally:
--      python scripts/ood_regression/generate_ood_eval_data.py --n_datasets 200
--   2. Stage all 200 OOD parquet files:
--      PUT file://data/ood_regression/E/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/E/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/ood_regression/F/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/F/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/ood_regression/G/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/G/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/ood_regression/H/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/H/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/ood_regression/ood_manifest.json @EVALUATION_DATASET_STAGE/ood_parity/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   3. Stage updated Python scripts:
--      PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/ood_regression/prepare_ood_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/prepare_synthetic_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   4. Call procedure:
--      CALL run_synthetic_regression_ood_full_evaluation('2.5.0-py311', '2.5.0-py311');
--   5. Verify index:
--      SELECT prior_regime, COUNT(*) AS n
--      FROM SYNTHETIC_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'ood_linear_full_v1'
--      GROUP BY prior_regime ORDER BY prior_regime;
--      -- Expected: E/F/G/H each with 50 rows
--   6. Verify outputs:
--      LIST @EVALUATION_RESULTS_STAGE/ood_full/;
--      -- Expected: synthetic_regression_model_comparison.csv,
--      --           synthetic_regression_model_comparison_summary.csv,
--      --           synthetic_regression_summary_by_regime.csv,
--      --           synthetic_regression_chart_data_model_rank.csv
CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_ood_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_ood_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_evaluation';

-- OOD Full Suite — Split-Phase Procedures
CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_ood_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_prep';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_deepset_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_aggregation';

-- OOD full split-phase execution (recommended under tight quota):
CALL run_synthetic_regression_ood_full_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_ood_full_deepset_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_ood_full_baseline_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_synthetic_regression_ood_full_autogluon_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
CALL run_synthetic_regression_ood_full_aggregation('2.5.0-py311', '2.5.0-py311');

-- ============================================================
-- Combined Suite Evaluation (linear_all_v1, 400 datasets)
-- ============================================================
-- Runbook:
--   Prerequisites (both source suites must already be indexed):
--     CALL run_synthetic_regression_prep('<bench_rt>', '<bench_rt>', '<ag_rt>');
--     CALL run_synthetic_regression_ood_full_evaluation('<bench_rt>', '<ag_rt>');
--   1. Stage updated Python scripts:
--      PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/prepare_synthetic_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   2. Call procedure:
--      CALL run_synthetic_regression_combined_evaluation('2.5.0-py311', '2.5.0-py311');
--   3. Verify index (expect A/B/C/D/E/F/G/H each with 50 rows):
--      SELECT prior_regime, COUNT(*) AS n
--      FROM SYNTHETIC_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'linear_all_v1'
--      GROUP BY prior_regime ORDER BY prior_regime;
--   4. Verify source lineage:
--      SELECT source_suite_id, COUNT(*) AS n
--      FROM SYNTHETIC_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'linear_all_v1'
--      GROUP BY source_suite_id ORDER BY source_suite_id;
--      -- Expected: linear_poisson_v1_recommended=200, ood_linear_full_v1=200
--   5. Verify outputs:
--      LIST @EVALUATION_RESULTS_STAGE/combined/;
--   6. Migration note — if SYNTHETIC_REGRESSION_DATASET_INDEX exists without source_suite_id:
--      ALTER TABLE SYNTHETIC_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS source_suite_id STRING;
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation';

-- Combined Suite — Split-Phase Procedures
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_prep';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_deepset_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_evaluation';

-- Combined AutoGluon evaluation — two-argument form uses all defaults from env:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_evaluation';

-- Combined AutoGluon evaluation — full dynamic form:
-- AUTOGLUON_CLUSTER_SHARDS:      number of logical shard files to write (default 6)
-- AUTOGLUON_WORKERS_PER_SHARD:   target_instances per MLJob cluster   (default 4)
-- AUTOGLUON_TASK_CPUS:           CPUs per individual AutoGluon fit     (default 1)
-- AUTOGLUON_CONCURRENT_CLUSTERS: max simultaneous MLJob clusters       (default 6)
-- AUTOGLUON_TIME_LIMIT_SECONDS:  per-fit time limit in seconds         (default 300)
-- AUTOGLUON_PRESETS:             AutoGluon presets string              (default best_quality)
-- AUTOGLUON_ENTRYPOINT:          Ray entrypoint filename               (default evaluate_synthetic_regression_autogluon_ray.py)
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING,
  AUTOGLUON_ENTRYPOINT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_evaluation';

-- Combined aggregation — two-argument form defaults to SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT=6:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation';

-- Combined aggregation — three-argument form with explicit expected AutoGluon shard count:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  EXPECTED_AUTOGLUON_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation';

-- Combined baseline capacity probe — verify DEEPSET_CPU_POOL can scale before evaluation:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_capacity_probe';

-- Combined AutoGluon capacity probe — verify AUTOGLUON_CPU_POOL can satisfy the distributed
-- envelope (concurrent_clusters * workers_per_shard nodes) before evaluation:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_capacity_probe';

-- Combined split-phase execution with distributed AutoGluon (recommended under tight quota):
-- Step 0: Verify node quota before committing to the evaluation runs.
-- CALL run_synthetic_regression_combined_baseline_capacity_probe('2.5.0-py311', '2.5.0-py311', 6);
-- ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
-- ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
-- Step 1: Run combined split phases.
CALL run_synthetic_regression_combined_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_combined_deepset_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_baseline_evaluation('2.5.0-py311', '2.5.0-py311', 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- Recommended: 6 clusters x 4 workers = 24 concurrent CPU_X64_M nodes, 1 CPU per fit task
CALL run_synthetic_regression_combined_autogluon_evaluation(
  '2.5.0-py311', '2.5.0-py311', 6, 4, 1, 6, 300, 'best_quality',
  'evaluate_synthetic_regression_autogluon_ray.py'
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
-- Aggregation expects 6 AutoGluon shard files (N=6 matches cluster_shards above)
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', '2.5.0-py311', 6);

-- Recommended capacity probe calls for reference:
-- CALL run_synthetic_regression_combined_baseline_capacity_probe(
--   '2.5.0-py311', '2.5.0-py311', 6
-- );
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe(
--   '2.5.0-py311', '2.5.0-py311', 6, 4, 6
-- );
-- CALL run_synthetic_regression_combined_autogluon_evaluation(
--   '2.5.0-py311', '2.5.0-py311', 6, 4, 1, 6, 300, 'best_quality',
--   'evaluate_synthetic_regression_autogluon_ray.py'
-- );
-- CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', '2.5.0-py311', 6);
-- Final training with explicit runtime lineage variables:
-- CALL run_model_training(
--   'market_exchangeable_icl', 'synthetic_regression_combined', 'inductive_forecasting'
-- );

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

-- HPO Sweep 1 (recommended first): ridge_residual — fixed architecture, tune optimizer/Ridge Expert.
CALL run_hpo_pipeline(
  'market_exchangeable_icl',
  'synthetic_regression_combined',
  'inductive_forecasting',
  'ridge_residual'
);
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
-- Inspect best_config.json; expected keys include: lr, weight_decay, dropout, ridge_lambda,
--   gate_hidden_dim, use_huber, n_sab_feat, d_phi, hpo_sweep_mode, use_ridge_expert, ...
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));

-- HPO Sweep 2 (optional, run after ridge_residual is stable): architecture — tunes d_phi/n_sab_feat.
-- Run DDP memory probe if d_phi > 128 before using architecture sweep results for final training.
-- CALL run_hpo_pipeline(
--   'market_exchangeable_icl',
--   'synthetic_regression_combined',
--   'inductive_forecasting',
--   'architecture'
-- );
-- LIST @MODEL_STAGE/hpo/;
-- SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));

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
