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
-- AUTOGLUON_CPU_POOL uses CPU_X64_M for AutoGluon shard jobs. MAX_NODES=60
-- supports legacy single-wave AutoGluon (60 single-node shards). The combined
-- suite default uses only 24 of those nodes: 6 logical AutoGluon work-item
-- clusters with 4 target instances per cluster.
-- Each combined cluster runs autogluon_ray.py
-- with Ray for distributed independent work items.
-- Single-wave execution is enforced. Lower concurrency arguments are rejected;
-- they are not used as multi-wave throttles. For combined AutoGluon, reduce
-- SYNREG_AUTOGLUON_CLUSTER_SHARDS explicitly if fewer output shard files are
-- desired, or request quota for cluster_shards * workers_per_shard nodes.
-- Run run_synthetic_regression_combined_autogluon_capacity_probe before evaluation.
-- Run run_synthetic_regression_combined_autogluon_worker_access_probe after the
-- capacity probe to validate driver-to-worker item metadata transfer and worker
-- dataset access before full AutoGluon evaluation.
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
-- Distributed AutoGluon uses autogluon_ray.py, ray_capacity_probe.py, and
-- autogluon_worker_access_probe.py (covered by scripts/*.py wildcard above).
--
-- Verify:
-- LIST @MODEL_STAGE/scripts/;
-- Epoch calibration jobs also run from @MODEL_STAGE/scripts/. @EPOCH_STAGE is
-- output-only for hpo_timing.json, hpo_epoch_error.json, train_timing.json, and
-- train_epoch_error.json.
-- Before epoch calibration, verify the shared modules and entrypoints:
-- LIST @MODEL_STAGE/scripts/ PATTERN='.*(hpo|hpo_epoch_test|train_epoch_test|train|model|snowflake_io)[.]py';
-- Before synthetic regression evaluation, verify the orchestration and Ray entrypoints:
-- LIST @MODEL_STAGE/scripts/ PATTERN='.*(run_synthetic_regression_evaluation|evaluate_synthetic_regression|autogluon_ray|ray_capacity_probe|autogluon_worker_access_probe)[.]py';

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

-- run_pretrain_pipeline() — pretrain entrypoints.
--
-- Production path: use the 4-arg gate-specific overload for all three gate candidates.
-- HPO requires pretrain_gate32.pt, pretrain_gate64.pt, and pretrain_gate128.pt to exist
-- before starting. HPO fails hard if any are missing.
--
-- 4-arg gate-specific form (production):
--   CALL run_pretrain_pipeline(MODEL_FAMILY, TRAINING_DATA_FAMILY, MODEL_DESIGN_PATTERN, GATE_DIM);
--   Example: CALL run_pretrain_pipeline('market_exchangeable_icl',
--                'synthetic_regression_combined', 'inductive_forecasting', 64);
--
-- 0-arg form (env-var defaults; writes pretrain.pt — legacy, not used by HPO flow):
--   CALL run_pretrain_pipeline();
-- 3-arg form (explicit selectors; writes pretrain.pt — legacy, not used by HPO flow):
--   CALL run_pretrain_pipeline(
--       'market_exchangeable_icl',
--       'synthetic_regression_combined',
--       'inductive_forecasting'
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

-- Gate-specific pretrain: writes pretrain_gate<N>.pt for one HPO gate candidate.
-- Must be called for all three candidates (32, 64, 128) before run_hpo_pipeline().
-- The ridge_residual HPO tunes gate_hidden_dim; each trial requires a matching
-- pretrain checkpoint to warm-start from.
--   CALL run_pretrain_pipeline(
--       'market_exchangeable_icl',       -- MODEL_FAMILY
--       'synthetic_regression_combined', -- TRAINING_DATA_FAMILY
--       'inductive_forecasting',         -- MODEL_DESIGN_PATTERN
--       64                               -- GATE_HIDDEN_DIM (32, 64, or 128)
--   );
CREATE OR REPLACE PROCEDURE run_pretrain_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  GATE_HIDDEN_DIM INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_pretrain_job.py')
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline_model_gate';

-- run_hpo_pipeline() launches hpo.py on the GPU pool using Ray Tune for distributed HPO.
-- Sweep-specific outputs: best_config_ridge_residual.json, best_config_architecture.json.
-- Merged output (written by architecture sweep): best_config.json.
-- Keys: lr, weight_decay, dropout, d_phi, d_rho, pool, n_sab_feat,
--   use_ridge_expert, ridge_lambda, gate_hidden_dim, use_huber, huber_delta, lambda_l1,
--   model_family, model_arch_version, model_design_pattern, hpo_sweep_mode, _meta.
-- Note: HPO only supports inductive_forecasting; transductive_completion raises in hpo.py.
--
-- Two-sweep strategy (recommended):
--   Sweep 1 (ridge_residual): Fixed architecture (d_phi=128, n_sab_feat=1). Tunes
--     optimizer, Ridge Expert (ridge_lambda, gate_hidden_dim ∈ {32,64,128}), and
--     robust loss params. Requires gate-specific pretrain checkpoints to exist.
--     Writes best_config_ridge_residual.json and best_config.json.
--   Run model_ddp_memory_probe for worst-case d_phi=256, n_blocks=2 before sweep 2.
--   Sweep 2 (architecture): Tunes d_phi and n_sab_feat. Freezes optimizer params from
--     sweep 1 via HPO_BASELINE_CONFIG_STAGE_PATH. Allows cold-start on arch mismatch.
--     Writes best_config_architecture.json and merged best_config.json.
--
-- Zero-arg (uses env-var defaults; HPO_SWEEP_MODE defaults to ridge_residual):
--   CALL run_hpo_pipeline();
-- Three-arg (explicit model selectors; HPO_SWEEP_MODE from env-var, defaults ridge_residual):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting'
--   );
-- Four-arg (production-recommended explicit form):
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
--
-- Sweep 1 (ridge_residual — tunes optimizer/regularization):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual'
--   );
-- Sweep 2 (architecture — tunes d_phi/n_sab_feat; run after ridge_residual + memory probe):
-- Use five-arg overload below to pass HPO_BASELINE_CONFIG_STAGE_PATH for architecture sweep.
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
-- HPO_BASELINE_CONFIG_STAGE_PATH is required when HPO_SWEEP_MODE='architecture';
-- pass '' for ridge_residual.
--
-- Two-sweep HPO (recommended):
--   -- Sweep 1: ridge_residual
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual', ''
--   );
--   -- Run memory probe before architecture sweep:
--   CALL run_model_ddp_memory_probe(
--       'model3', 'inductive_forecasting', 'market_exchangeable_icl',
--       200, 128, 128, 256, 2, TRUE
--   );
--   -- Sweep 2: architecture (requires best_config_ridge_residual.json from sweep 1)
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
--   Step 1  (Validate)            META_DATASET_INDEX counts, columns, stage access
--   Step 2  (Pretrain)          → @MODEL_STAGE/checkpoints/pretrain.pt
--   Step 3  (HPO ridge_residual)→ @MODEL_STAGE/hpo/best_config_ridge_residual.json
--   Step 4  (Memory probe)        Worst-case d_phi=256, n_blocks=2; guards against OOM
--   Step 5  (HPO architecture)  → @MODEL_STAGE/hpo/best_config.json (merged)
--   Step 6  (Load config)         Reads merged best_config.json
--   Step 7  (Final training)    → @MODEL_STAGE/checkpoints/best.pt
--                                  (with _meta.pretrain_checkpoint_stage_path)
--   Step 6  (Load config)        Downloads and parses best_config.json
--   Step 7  (Final training)  → @MODEL_STAGE/checkpoints/best.pt
-- Architecture HPO trials freeze optimizer params from best_config_ridge_residual.json
-- via HPO_BASELINE_CONFIG_STAGE_PATH. Cold-start on arch mismatch is allowed.
-- Final training uses PRETRAIN_LOAD_POLICY=allow_cold_start_on_arch_mismatch with
-- PRETRAIN_CHECKPOINT_PATH=@MODEL_STAGE/checkpoints/pretrain.pt.
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_capacity_probe_default';

-- Single-wave capacity probes: BASELINE_CONCURRENT_NODES must equal 6
-- (SYNREG_CPU_SHARDS) and AUTOGLUON_CONCURRENT_NODES must equal 60
-- (SYNREG_AUTOGLUON_SHARDS). Lower values fail fast instead of batching.
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

-- Baseline capacity probe: BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS
-- (default SYNREG_CPU_SHARDS=6). Lower values are rejected; request quota or
-- increase BASELINE_SHARDS through SYNREG_BASELINE_SHARDS or the BASELINE_SHARDS arg.
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

-- Legacy AutoGluon capacity probe: AUTOGLUON_CONCURRENT_NODES must equal
-- SYNREG_AUTOGLUON_SHARDS (60 by default). Lower values are rejected.
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

-- Baseline evaluation is single-wave: BASELINE_CONCURRENT_NODES must equal
-- SYNREG_CPU_SHARDS (6 by default). Lower values are rejected, not batched.
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_baseline_evaluation_default';

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

-- Baseline evaluation with explicit shard count: BASELINE_SHARDS controls the number
-- of shard files written and BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS.
-- Use this overload to run more than 6 baseline shards (e.g. BASELINE_SHARDS=10).
CREATE OR REPLACE PROCEDURE run_synthetic_regression_baseline_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_baseline_evaluation_with_shards';

-- Legacy AutoGluon evaluation is single-wave: AUTOGLUON_CONCURRENT_NODES must
-- equal SYNREG_AUTOGLUON_SHARDS (60 by default). Lower values are rejected.
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_evaluation_default';

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
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_evaluation_default';

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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_baseline_evaluation_default';

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

-- OOD full baseline with explicit shard count: BASELINE_SHARDS and BASELINE_CONCURRENT_NODES
-- must match. Use to write more than 6 OOD baseline shard files.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_baseline_evaluation_with_shards';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_autogluon_evaluation_default';

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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation_legacy_concurrency';

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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_evaluation_default';

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

-- Combined baseline evaluation with explicit shard count: BASELINE_SHARDS controls the
-- number of shard files written; BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS.
-- 1 baseline shard = 1 single-node MLJob = 1 output shard file.
-- Increasing BASELINE_SHARDS increases required concurrent CPU nodes and output file count.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_evaluation_with_shards';

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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_evaluation_default';

-- Combined AutoGluon evaluation — full dynamic form.
-- Supports two execution modes controlled by AUTOGLUON_CLUSTER_SHARDS:
--
-- Ray distributed cluster-shard mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--   AUTOGLUON_CLUSTER_SHARDS:      number of Ray cluster shards / output shard files (e.g. 6)
--   AUTOGLUON_WORKERS_PER_SHARD:   target_instances per MLJob cluster (e.g. 4)
--   AUTOGLUON_TASK_CPUS:           CPUs per individual AutoGluon fit  (default 1)
--   AUTOGLUON_CONCURRENT_CLUSTERS: must equal AUTOGLUON_CLUSTER_SHARDS; lower values fail fast
--   AUTOGLUON_TIME_LIMIT_SECONDS:  per-fit time limit in seconds      (default 300)
--   AUTOGLUON_PRESETS:             AutoGluon presets string           (default best_quality)
--   RAY_READY_TIMEOUT_SECONDS:     optional Ray readiness timeout     (default 600)
--   RAY_READY_POLL_SECONDS:        optional Ray readiness poll period  (default 10)
--   These map to MLJob env vars SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS and
--   SYNREG_RAY_CLUSTER_READY_POLL_SECONDS.
--   entrypoint:                    always autogluon_ray.py (derived internally)
--   total containers = AUTOGLUON_CLUSTER_SHARDS x AUTOGLUON_WORKERS_PER_SHARD
--   aggregation expects N = AUTOGLUON_CLUSTER_SHARDS output shard files
--
-- Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS = 0):
--   AUTOGLUON_CLUSTER_SHARDS:      0 (selects single-node mode)
--   AUTOGLUON_WORKERS_PER_SHARD:   must be 1 (single-node; no multi-instance MLJob)
--   AUTOGLUON_CONCURRENT_CLUSTERS: concurrent single-node shard jobs (e.g. 30)
--   entrypoint:                    always evaluate_synthetic_regression.py (derived internally)
--   total containers = AUTOGLUON_CONCURRENT_CLUSTERS
--   aggregation expects N = AUTOGLUON_CONCURRENT_CLUSTERS output shard files
--   no Ray, no Ray object store, no ray_capacity_probe needed
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING,
  RAY_READY_TIMEOUT_SECONDS INTEGER,
  RAY_READY_POLL_SECONDS INTEGER
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation_default';

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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation_ag';

-- Combined aggregation - full explicit form matching the Python implementation
-- signature after session:
--   (bench_rt, ag_rt, expected_ag_shards, expected_baseline_shards, expected_deepset_shards)
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  EXPECTED_AUTOGLUON_SHARDS INTEGER,
  EXPECTED_BASELINE_SHARDS INTEGER,
  EXPECTED_DEEPSET_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation';

-- Combined baseline capacity probe: verify DEEPSET_CPU_POOL can scale to all
-- BASELINE_SHARDS (default SYNREG_CPU_SHARDS=6) in one wave. Single-wave execution
-- is enforced. Lower BASELINE_CONCURRENT_NODES values fail fast. Increase BASELINE_SHARDS
-- through SYNREG_BASELINE_SHARDS env var or the explicit BASELINE_SHARDS SQL arg.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_capacity_probe_default';

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

-- Combined baseline capacity probe with explicit shard count: BASELINE_SHARDS sets the
-- number of probes submitted; BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS.
-- Use when intentionally running with a non-default baseline shard count.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_capacity_probe_with_shards';

-- Combined AutoGluon capacity probe — verify AUTOGLUON_CPU_POOL node availability
-- before evaluation. Mirrors the two execution modes of the evaluation procedure:
--
-- Ray distributed mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--   Submits one ray_capacity_probe.py job per cluster shard, each with
--   target_instances=AUTOGLUON_WORKERS_PER_SHARD. Verifies the pool can satisfy
--   AUTOGLUON_CLUSTER_SHARDS * AUTOGLUON_WORKERS_PER_SHARD nodes simultaneously.
--   AUTOGLUON_CONCURRENT_CLUSTERS must equal AUTOGLUON_CLUSTER_SHARDS.
--   RAY_READY_TIMEOUT_SECONDS defaults to 300 seconds; RAY_READY_POLL_SECONDS
--   defaults to 10 seconds. Use the extended overload when diagnosing cold-start
--   or startup-convergence behavior. These map to MLJob env vars
--   SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS and
--   SYNREG_RAY_CLUSTER_READY_POLL_SECONDS.
--
-- Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS = 0):
--   Submits AUTOGLUON_CONCURRENT_CLUSTERS single-node capacity_probe.py jobs
--   (target_instances=1). No Ray. AUTOGLUON_WORKERS_PER_SHARD must be 1.
-- This probe verifies infrastructure startup/resource visibility only; it does not
-- validate the worker dataset-access descriptor. Run the worker-access probe below
-- after this succeeds and before distributed AutoGluon evaluation.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_capacity_probe_default';

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

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  RAY_READY_TIMEOUT_SECONDS INTEGER,
  RAY_READY_POLL_SECONDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_capacity_probe';

-- Combined AutoGluon worker-access probe — verify the worker data-access path
-- after the capacity probe and before full AutoGluon evaluation. Uses the same
-- runtime topology parameters as the capacity probe.
--
-- Ray distributed mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--   Submits one autogluon_worker_access_probe.py job per cluster shard, each with
--   target_instances=AUTOGLUON_WORKERS_PER_SHARD. The Ray driver loads metadata
--   from SYNTHETIC_REGRESSION_DATASET_INDEX, builds compact item dicts with
--   dataset_access.mode='scoped_file_url', and sends only those dicts to workers
--   as Ray task arguments. The driver derives dataset_access.scoped_url with
--   BUILD_SCOPED_FILE_URL. Workers use SnowflakeFile.open(scoped_url) and do not
--   create Snowpark sessions or query SYNTHETIC_REGRESSION_DATASET_INDEX.
--   No AutoGluon training and no full-suite dataset fan-out.
--
-- Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS = 0):
--   Submits AUTOGLUON_CONCURRENT_CLUSTERS single-node access probes
--   (target_instances=1). No Ray. AUTOGLUON_WORKERS_PER_SHARD must be 1.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_worker_access_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_worker_access_probe_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_worker_access_probe(
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_worker_access_probe';

-- AutoGluon import timing probe — measures dependency bootstrap latency.
--
-- Stage the probe script before calling this procedure:
--   PUT file:///path/to/scripts/autogluon_import_timing_probe.py
--     @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
-- With-pip mode (WITH_PIP=TRUE, default):
--   pip_requirements=autogluon.tabular==1.3.0 is installed at container startup.
--   Time from MLJob submission to the python_entrypoint_started JSON log line
--   approximates Snowflake scheduling + image startup + AutoGluon pip bootstrap.
--   autogluon_import_complete.import_seconds measures pure import overhead after
--   the environment is ready.
--
-- No-pip baseline mode (WITH_PIP=FALSE):
--   No pip install at startup and no AutoGluon/Ray import is attempted. The probe
--   should succeed even when AutoGluon is not installed, giving a clean scheduling
--   + image startup baseline.
--   Compare pip vs no-pip probe waves to estimate bootstrap overhead under concurrency.
--
-- PROBE_COUNT: number of independent single-instance MLJobs to submit concurrently.
--   Use PROBE_COUNT=8 to simulate the concurrency of a full AutoGluon evaluation wave.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_import_timing_probe(
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_import_timing_probe_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_import_timing_probe(
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  WITH_PIP BOOLEAN,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_import_timing_probe';

-- ===========================================================================
-- SPCS custom-image AutoGluon backend procedures
-- ===========================================================================
-- AUTOGLUON_IMAGE_REPOSITORY: Snowflake Image Repository for the custom AutoGluon SPCS image.
-- See sql/create_autogluon_spcs_image_repository.sql for setup commands.
-- No runtime_environment or pip_requirements are used by SPCS procedures.
-- AutoGluon, Ray, and all dependencies are preinstalled in the custom OCI image.
-- Self-managed Ray: head + workers start in SPCS containers; driver connects explicitly.

-- Stage scripts before calling SPCS procedures:
--   PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/spcs_snowpark_session_probe.py         @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/spcs_ray_head.py                       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/spcs_ray_worker.py                     @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/autogluon_ray.py                       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/autogluon_import_timing_probe.py       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- SPCS preflight runbook (must be run in order before full evaluation):
--   Step 1. Import probe — verify image starts and AutoGluon/Ray imports succeed.
--   Step 2. Session probe — verify Snowflake OAuth token injection (snowflakeService.enabled=true).
--   Step 3. Capacity probe — verify compute pool can host the requested Ray cluster topology.
--   Step 4. Worker-access probe — verify presigned-URL dataset loading from workers.
--   Step 5. One-shard mini evaluation — verify end-to-end with 1 shard before full run.
--   Step 6. Full distributed evaluation.
--   Step 7. Aggregation.
-- Default separated Ray topology: 6 heads + 24 workers + 6 drivers = 36 containers.
-- Only the 24 worker containers advertise Ray CPUs for AutoGluon tasks.
-- Heads and drivers are intentionally downsized: head 0.5 CPU, 2Gi/4Gi memory;
-- driver 0.5/1 CPU, 2Gi/4Gi memory; worker/single-node 4 CPU, 16Gi memory.
-- Override with SYNREG_SPCS_RAY_HEAD_*, SYNREG_SPCS_RAY_DRIVER_*,
-- SYNREG_SPCS_RAY_WORKER_*, SYNREG_SPCS_SINGLE_NODE_*, or explicit
-- *_CPU_REQUEST/*_CPU_LIMIT/*_MEMORY_REQUEST/*_MEMORY_LIMIT suffixes.
-- Pass the full pushed image reference as AUTOGLUON_SPCS_IMAGE when calling.
-- Legacy callers that still pass 'spcs_job' may optionally set:
--   ALTER SESSION SET SYNREG_AUTOGLUON_SPCS_IMAGE = '<account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY/tabpfn-autogluon-ray:1.0.0';
-- For multi-shard Ray mode, also set SPCS_RAY_HEAD_DNS_SUFFIX:
--   ALTER SESSION SET SPCS_RAY_HEAD_DNS_SUFFIX = '<schema_lower>.<db_lower>.snowflakecomputing.internal';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_spcs_import_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_spcs_import_probe';

-- SPCS Snowpark session probe: validates OAuth token injection (snowflakeService.enabled=true)
-- and Snowpark session creation inside SPCS containers.
-- The SPCS job spec builder (run_synthetic_regression_evaluation._build_spcs_job_spec) sets
-- snowflakeService.enabled=true so the container receives the OAuth token at /snowflake/session/token.
-- Must be run after the import probe and before the capacity probe.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_spcs_session_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_spcs_session_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_worker_access_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_worker_access_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_evaluation(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT INTEGER,
  AUTOGLUON_PRESETS STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_evaluation';

-- Combined all-in-one evaluation with explicit baseline shard count.
-- BASELINE_SHARDS: number of baseline shard files to write (default 6); must equal BASELINE_CONCURRENT_NODES.
-- BASELINE_CONCURRENT_NODES: required single-wave CPU nodes; must equal BASELINE_SHARDS.
-- Aggregation automatically uses the resolved AutoGluon execution plan output_shards.
-- 1 baseline shard = 1 single-node MLJob = 1 output shard file.
-- AutoGluon mode is selected by AUTOGLUON_CLUSTER_SHARDS (see evaluation procedure above).
-- Set AUTOGLUON_CLUSTER_SHARDS=0 to use single-node shard mode for AutoGluon.
-- Entrypoints are derived internally from the mode and are not accepted as arguments.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation_with_baseline_shards';

-- ===========================================================================
-- AutoGluon execution mode examples
-- ===========================================================================
-- Required operational sequence:
--   1. run_synthetic_regression_runtime_probes
--   2. run_synthetic_regression_combined_autogluon_capacity_probe
--   3. run_synthetic_regression_combined_autogluon_worker_access_probe
--   4. run_synthetic_regression_combined_autogluon_evaluation
--   5. run_synthetic_regression_combined_aggregation

-- A. Ray distributed cluster-shard mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--    6 clusters x 4 workers = 24 containers; 6 AutoGluon shard files.
--    Use when Ray memory is sufficient and dynamic work-item scheduling is desired.
--
-- Capacity probe (verifies 24 CPU_X64_M nodes across 6 Ray clusters):
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   6, 4, 6,
--   300, 10  -- RAY_READY_TIMEOUT_SECONDS, RAY_READY_POLL_SECONDS
-- );
-- Worker-access probe (verifies compact item dict transfer and scoped_file_url access):
-- CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   6, 4, 6
-- );
-- Evaluation (6 cluster shards x 4 workers each; aggregation expects N=6):
-- CALL run_synthetic_regression_combined_autogluon_evaluation(
--   '2.5.0-py311', '2.5.0-py311',
--   6, 4, 1, 6, 300, 'best_quality',
--   600, 10  -- RAY_READY_TIMEOUT_SECONDS, RAY_READY_POLL_SECONDS
-- );

-- B. Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS = 0):
--    30 containers; 30 AutoGluon shard files. No Ray.
--    Use when memory is constrained or Ray/object-store issues are unreliable.
--    AUTOGLUON_WORKERS_PER_SHARD must be 1.
--    AUTOGLUON_CONCURRENT_CLUSTERS is the concurrent single-node shard count.
--
-- Capacity probe (verifies 30 single-node CPU_X64_M containers):
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   0, 1, 30
-- );
-- Worker-access probe (verifies single-node metadata/dataset access path):
-- CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   0, 1, 30
-- );
-- Evaluation (30 single-node shards; aggregation expects N=30):
-- CALL run_synthetic_regression_combined_autogluon_evaluation(
--   '2.5.0-py311', '2.5.0-py311',
--   0, 1, 1, 30, 300, 'best_quality'
-- );

-- ===========================================================================
-- Combined split-phase execution — Ray distributed AutoGluon (default):
-- ===========================================================================
-- Step 0: Verify node quota before committing to the evaluation runs.
-- CALL run_synthetic_regression_combined_baseline_capacity_probe('2.5.0-py311', '2.5.0-py311', 6);
-- ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
-- CALL run_synthetic_regression_combined_autogluon_worker_access_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
-- ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
-- Step 1: Run combined split phases.
CALL run_synthetic_regression_combined_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_combined_deepset_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_baseline_evaluation('2.5.0-py311', '2.5.0-py311', 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- Ray mode: 6 clusters x 4 workers = 24 concurrent CPU_X64_M nodes, 1 CPU per fit task
CALL run_synthetic_regression_combined_autogluon_evaluation(
  '2.5.0-py311', '2.5.0-py311', 6, 4, 1, 6, 300, 'best_quality'
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
-- Aggregation expects 6 AutoGluon shard files (N=6 matches cluster_shards above)
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', '2.5.0-py311', 6);

-- ===========================================================================
-- Combined split-phase execution — single-node AutoGluon (memory-constrained):
-- ===========================================================================
-- Step 0: Capacity probe (30 single-node containers, no Ray):
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe(
--   '2.5.0-py311', '2.5.0-py311', 0, 1, 30
-- );
-- CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
--   '2.5.0-py311', '2.5.0-py311', 0, 1, 30
-- );
-- Step 1: Single-node AutoGluon evaluation (30 shards; aggregation expects N=30):
-- CALL run_synthetic_regression_combined_autogluon_evaluation(
--   '2.5.0-py311', '2.5.0-py311', 0, 1, 1, 30, 300, 'best_quality'
-- );
-- CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', '2.5.0-py311', 30);
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
-- Two-sweep HPO strategy (recommended):
--
-- Option A: Gate-specific pretrain + ridge_residual sweep only (single-sweep):
--   Gate-specific pretrains are required for ridge_residual HPO.
--   HPO tunes gate_hidden_dim across all three candidates.
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_combined',
  'inductive_forecasting', 32
);
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_combined',
  'inductive_forecasting', 64
);
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_combined',
  'inductive_forecasting', 128
);
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_gate.*[.]pt';

-- HPO sweep 1: ridge_residual — fixed d_phi/n_sab_feat, tune optimizer/Ridge Expert/gate.
-- Writes best_config_ridge_residual.json and best_config.json.
CALL run_hpo_pipeline(
  'market_exchangeable_icl',
  'synthetic_regression_combined',
  'inductive_forecasting',
  'ridge_residual'
);
LIST @MODEL_STAGE/hpo/;
-- Inspect best_config_ridge_residual.json:
SELECT $1 FROM @MODEL_STAGE/hpo/best_config_ridge_residual.json (FILE_FORMAT => (TYPE = JSON));

-- Option B (recommended): Continue with architecture sweep after ridge_residual:
-- Mandatory memory probe before architecture HPO (worst-case: d_phi=256, n_blocks=2).
CALL run_model_ddp_memory_probe(
  'model3', 'inductive_forecasting', 'market_exchangeable_icl',
  200, 128, 128, 256, 2, TRUE
);

-- HPO sweep 2: architecture — tunes d_phi and n_sab_feat; freezes optimizer from sweep 1.
-- HPO_BASELINE_CONFIG_STAGE_PATH must point to best_config_ridge_residual.json from sweep 1.
-- Writes best_config_architecture.json and merged best_config.json.
CALL run_hpo_pipeline(
  'market_exchangeable_icl',
  'synthetic_regression_combined',
  'inductive_forecasting',
  'architecture',
  '@MODEL_STAGE/hpo/best_config_ridge_residual.json'
);
LIST @MODEL_STAGE/hpo/;
-- Inspect merged best_config.json; expected: _meta.sweeps.ridge_residual + .architecture:
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
