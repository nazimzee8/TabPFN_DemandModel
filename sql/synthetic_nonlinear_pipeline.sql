SELECT CURRENT_USER();
SELECT
    CURRENT_ORGANIZATION_NAME() AS orgname,
    CURRENT_ACCOUNT_NAME() AS acctname;

CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
CREATE SCHEMA  IF NOT EXISTS TABPFN_DB.TABPFN_SCHEMA;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- ============================================================
-- TRAINING DATA STAGES
-- Each stage uses numeric/ and mixed/ subdirectories so both
-- numeric-only and mixed-categorical families can share the
-- same stage object.  Separate mixed stages are NOT created.
--
-- Layout:
--   @META_NONLINEAR_REGRESSION_DATASET_STAGE/numeric/{train,val,test}/
--   @META_NONLINEAR_REGRESSION_DATASET_STAGE/mixed/{train,val,test}/
--   @META_NONLINEAR_CLASSIFICATION_DATASET_STAGE/numeric/{train,val,test}/
--   @META_NONLINEAR_CLASSIFICATION_DATASET_STAGE/mixed/{train,val,test}/
-- ============================================================
CREATE STAGE IF NOT EXISTS META_NONLINEAR_REGRESSION_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

CREATE STAGE IF NOT EXISTS META_NONLINEAR_CLASSIFICATION_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Shared infrastructure stages — idempotent alongside linear pipeline
CREATE STAGE IF NOT EXISTS MODEL_STAGE              ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE      ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ============================================================
-- STAGE PATH CONTRACT
-- Training data (one-row-per-task index):
--   @META_NONLINEAR_REGRESSION_DATASET_STAGE/numeric/{split}/{task_id}.parquet
--   @META_NONLINEAR_REGRESSION_DATASET_STAGE/mixed/{split}/{task_id}.parquet
--   @META_NONLINEAR_CLASSIFICATION_DATASET_STAGE/numeric/{split}/{task_id}.parquet
--   @META_NONLINEAR_CLASSIFICATION_DATASET_STAGE/mixed/{split}/{task_id}.parquet
-- Eval data (per-sample, separate schema):
--   @EVALUATION_DATASET_STAGE/nonlinear_v2/{family}/
--   @EVALUATION_DATASET_STAGE/nonlinear_v2/nonlinear_v2_manifest.json
-- Results:
--   @EVALUATION_RESULTS_STAGE/nonlinear/<suite_id>/
--   @EVALUATION_RESULTS_STAGE/nonlinear_mixed/<suite_id>/
--   @EVALUATION_RESULTS_STAGE/nonlinear_classification/<suite_id>/
--   @EVALUATION_RESULTS_STAGE/nonlinear_classification_mixed/<suite_id>/
-- ============================================================

-- ============================================================
-- COMPUTE POOLS — IF NOT EXISTS; already provisioned by linear pipeline
-- ============================================================
CREATE COMPUTE POOL IF NOT EXISTS DEEPSET_GPU_POOL
  MIN_NODES = 1 MAX_NODES = 10
  INSTANCE_FAMILY = GPU_NV_M
  AUTO_SUSPEND_SECS = 300 INITIALLY_SUSPENDED = TRUE;

CREATE COMPUTE POOL IF NOT EXISTS DEEPSET_CPU_POOL
  MIN_NODES = 1 MAX_NODES = 10
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300 INITIALLY_SUSPENDED = TRUE;

CREATE COMPUTE POOL IF NOT EXISTS AUTOGLUON_CPU_POOL
  MIN_NODES = 1 MAX_NODES = 60
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300 INITIALLY_SUSPENDED = TRUE;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TABPFN_PYPI_EAI
  ALLOWED_NETWORK_RULES = (SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE)
  ENABLED = TRUE
  COMMENT = 'Allows TabPFN ML Jobs to install approved PyPI dependencies.';
GRANT USAGE ON INTEGRATION TABPFN_PYPI_EAI TO ROLE ACCOUNTADMIN;
GRANT DATABASE ROLE SNOWFLAKE.PYPI_REPOSITORY_USER TO ROLE ACCOUNTADMIN;

-- ============================================================
-- TRAINING INDEX TABLES
-- ============================================================

-- Numeric-only nonlinear regression training tasks
CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_REGRESSION_DATASET_INDEX (
  split        STRING NOT NULL,
  task_id      STRING NOT NULL,
  stage_path   STRING NOT NULL,   -- numeric/{split}/{task_id}.parquet
  n            NUMBER,
  p            NUMBER,
  n_train      NUMBER,
  n_test       NUMBER,
  prior_regime STRING,
  hpo_bucket   NUMBER
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p, n_train);

-- Numeric-only nonlinear classification training tasks
CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_CLASSIFICATION_DATASET_INDEX (
  split                  STRING NOT NULL,
  task_id                STRING NOT NULL,
  stage_path             STRING NOT NULL,  -- numeric/{split}/{task_id}.parquet
  n                      NUMBER,
  p                      NUMBER,
  n_train                NUMBER,
  n_test                 NUMBER,
  prior_regime           STRING,
  hpo_bucket             NUMBER,
  num_classes            NUMBER,
  classification_regime  STRING,
  task_objective         STRING,
  p_signal               NUMBER,
  p_noise                NUMBER,
  p_total                NUMBER,
  class_imbalance_type   STRING,
  label_noise_rate       FLOAT,
  feature_noise_level    FLOAT
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, num_classes, classification_regime, p, n_train);

-- Mixed-categorical nonlinear regression training tasks
-- (stored under @META_NONLINEAR_REGRESSION_DATASET_STAGE/mixed/)
CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX (
  split                STRING  NOT NULL,
  task_id              STRING  NOT NULL,
  stage_path           STRING  NOT NULL,  -- mixed/{split}/{task_id}.parquet
  n                    NUMBER,
  p                    NUMBER,            -- p_num + p_cat
  p_num                NUMBER,
  p_cat                NUMBER,
  n_train              NUMBER,
  n_test               NUMBER,
  prior_regime         STRING,
  hpo_bucket           NUMBER,
  schema_version       STRING,
  task_family          STRING,
  training_data_family STRING,
  task_objective       STRING,
  categorical_cardinalities VARIANT
) DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p_num, p_cat, n_train);

-- Mixed-categorical nonlinear classification training tasks
-- (stored under @META_NONLINEAR_CLASSIFICATION_DATASET_STAGE/mixed/)
CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX (
  split                STRING  NOT NULL,
  task_id              STRING  NOT NULL,
  stage_path           STRING  NOT NULL,  -- mixed/{split}/{task_id}.parquet
  n                    NUMBER,
  p                    NUMBER,            -- p_num + p_cat
  p_num                NUMBER,
  p_cat                NUMBER,
  n_train              NUMBER,
  n_test               NUMBER,
  prior_regime         STRING,
  hpo_bucket           NUMBER,
  num_classes          NUMBER,
  classification_regime STRING,
  task_objective       STRING,
  class_imbalance_type STRING,
  label_noise_rate     FLOAT,
  feature_noise_level  FLOAT,
  temperature          FLOAT,
  schema_version       STRING,
  task_family          STRING,
  training_data_family STRING,
  categorical_cardinalities VARIANT
) DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, num_classes, classification_regime, p_num, p_cat, n_train);

-- ============================================================
-- TRAINING INDEX BUILDER PROCEDURES
-- IS_MIXED_CATEGORICAL = FALSE -> numeric/ subdir + numeric index table
-- IS_MIXED_CATEGORICAL = TRUE  -> mixed/   subdir + mixed   index table
-- ============================================================

-- Nonlinear regression index builders
CREATE OR REPLACE PROCEDURE build_meta_nonlinear_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_nonlinear_dataset_index_with_flag';

CREATE OR REPLACE PROCEDURE build_meta_nonlinear_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_nonlinear_dataset_index_with_flag_and_total';

-- Nonlinear classification index builders
CREATE OR REPLACE PROCEDURE build_meta_nonlinear_classification_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_nonlinear_classification_dataset_index_with_flag';

CREATE OR REPLACE PROCEDURE build_meta_nonlinear_classification_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_nonlinear_classification_dataset_index_with_flag_and_total';

-- ============================================================
-- EVALUATION INDEX TABLES
-- Nonlinear regression: NONLINEAR_REGRESSION_DATASET_INDEX (numeric)
--                       NONLINEAR_MIXED_REGRESSION_DATASET_INDEX (mixed-categorical)
-- Nonlinear classification: NONLINEAR_CLASSIFICATION_DATASET_INDEX (numeric)
--                           NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX (mixed-categorical)
-- NEVER DROP these tables. Prep handler uses DELETE WHERE suite_id only.
-- Schema migrations below add new columns if absent (idempotent).
-- ============================================================

CREATE TRANSIENT TABLE IF NOT EXISTS NONLINEAR_REGRESSION_DATASET_INDEX (
  suite_id               STRING,
  suite_family           STRING,
  dataset_id             NUMBER,
  dataset_seed           NUMBER,
  stage_path             STRING,
  prior_name             STRING,
  prior_version          STRING,
  prior_regime           STRING,
  split_seeds            ARRAY,
  n_total                NUMBER,
  n_train_default        NUMBER,
  n_holdout_default      NUMBER,
  p_signal               NUMBER,
  p_noise                NUMBER,
  p_total                NUMBER,
  target_noise_scale     FLOAT,
  training_size_anchor   BOOLEAN,
  feature_noise_level    NUMBER,
  eval_weight            FLOAT,
  payload_bytes          NUMBER,
  created_at             TIMESTAMP_NTZ,
  logical_dataset_key    STRING,
  source_suite_id        STRING,
  feature_regime         STRING,
  covariance_type        STRING,
  rho                    FLOAT,
  active_fraction        FLOAT,
  noise_feature_fraction FLOAT,
  feature_noise_sigma    FLOAT,
  suite_component        STRING,
  target_noise_type      STRING,
  snr_target             FLOAT,
  condition_id           STRING,
  teacher_seed           NUMBER,
  sample_seed            NUMBER,
  normalization_constant FLOAT
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- Schema migration -- add v2 metadata columns if absent (idempotent, safe to re-run)
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS feature_regime         STRING;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS covariance_type        STRING;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS rho                    FLOAT;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS active_fraction        FLOAT;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS noise_feature_fraction FLOAT;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS feature_noise_sigma    FLOAT;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS suite_component        STRING;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS target_noise_type      STRING;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS snr_target             FLOAT;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS condition_id           STRING;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS teacher_seed           NUMBER;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS sample_seed            NUMBER;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS normalization_constant FLOAT;

CREATE TRANSIENT TABLE IF NOT EXISTS NONLINEAR_MIXED_REGRESSION_DATASET_INDEX (
  suite_id               STRING,
  suite_family           STRING,
  dataset_id             NUMBER,
  dataset_seed           NUMBER,
  stage_path             STRING,
  prior_name             STRING,
  prior_version          STRING,
  prior_regime           STRING,
  split_seeds            ARRAY,
  n_total                NUMBER,
  n_train_default        NUMBER,
  n_holdout_default      NUMBER,
  p_signal               NUMBER,
  p_noise                NUMBER,
  p_total                NUMBER,
  target_noise_scale     FLOAT,
  training_size_anchor   BOOLEAN,
  feature_noise_level    NUMBER,
  eval_weight            FLOAT,
  payload_bytes          NUMBER,
  created_at             TIMESTAMP_NTZ,
  logical_dataset_key    STRING,
  source_suite_id        STRING,
  feature_regime         STRING,
  covariance_type        STRING,
  rho                    FLOAT,
  active_fraction        FLOAT,
  noise_feature_fraction FLOAT,
  feature_noise_sigma    FLOAT,
  suite_component        STRING,
  target_noise_type      STRING,
  snr_target             FLOAT,
  condition_id           STRING,
  teacher_seed           NUMBER,
  sample_seed            NUMBER,
  normalization_constant FLOAT,
  p_num                  NUMBER,
  p_cat                  NUMBER,
  categorical_cardinalities VARIANT,
  cat_effect_scale       FLOAT,
  max_cardinality        NUMBER,
  missing_rate           FLOAT
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- Schema migration -- idempotent, safe to re-run
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS feature_regime         STRING;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS covariance_type        STRING;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS rho                    FLOAT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS active_fraction        FLOAT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS noise_feature_fraction FLOAT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS feature_noise_sigma    FLOAT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS suite_component        STRING;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS target_noise_type      STRING;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS snr_target             FLOAT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS condition_id           STRING;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS teacher_seed           NUMBER;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS sample_seed            NUMBER;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS normalization_constant FLOAT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS p_num                  NUMBER;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS p_cat                  NUMBER;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS categorical_cardinalities VARIANT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS cat_effect_scale       FLOAT;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS max_cardinality        NUMBER;
ALTER TABLE NONLINEAR_MIXED_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS missing_rate           FLOAT;

CREATE TRANSIENT TABLE IF NOT EXISTS NONLINEAR_CLASSIFICATION_DATASET_INDEX (
  suite_id               STRING,
  suite_family           STRING,
  dataset_id             NUMBER,
  dataset_seed           NUMBER,
  stage_path             STRING,
  prior_regime           STRING,
  classification_regime  STRING,
  split_seeds            ARRAY,
  n_total                NUMBER,
  n_train_default        NUMBER,
  n_holdout_default      NUMBER,
  p_signal               NUMBER,
  p_noise                NUMBER,
  p_total                NUMBER,
  num_classes            NUMBER,
  feature_noise_level    FLOAT,
  training_size_anchor   BOOLEAN,
  eval_weight            FLOAT,
  payload_bytes          NUMBER,
  created_at             TIMESTAMP_NTZ,
  logical_dataset_key    STRING,
  source_suite_id        STRING,
  task_objective         STRING,
  label_noise_rate       FLOAT,
  class_imbalance_type   STRING,
  feature_regime         STRING,
  covariance_type        STRING,
  rho                    FLOAT,
  active_fraction        FLOAT,
  noise_feature_fraction FLOAT,
  feature_noise_sigma    FLOAT,
  suite_component        STRING,
  condition_id           STRING,
  teacher_seed           NUMBER,
  sample_seed            NUMBER
) DATA_RETENTION_TIME_IN_DAYS = 0;

CREATE TRANSIENT TABLE IF NOT EXISTS NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX (
  suite_id               STRING,
  suite_family           STRING,
  dataset_id             NUMBER,
  dataset_seed           NUMBER,
  stage_path             STRING,
  prior_regime           STRING,
  classification_regime  STRING,
  split_seeds            ARRAY,
  n_total                NUMBER,
  n_train_default        NUMBER,
  n_holdout_default      NUMBER,
  p_signal               NUMBER,
  p_noise                NUMBER,
  p_total                NUMBER,
  num_classes            NUMBER,
  feature_noise_level    FLOAT,
  training_size_anchor   BOOLEAN,
  eval_weight            FLOAT,
  payload_bytes          NUMBER,
  created_at             TIMESTAMP_NTZ,
  logical_dataset_key    STRING,
  source_suite_id        STRING,
  task_objective         STRING,
  label_noise_rate       FLOAT,
  class_imbalance_type   STRING,
  -- Mixed-categorical columns
  p_num                  NUMBER,
  p_cat                  NUMBER,
  categorical_cardinalities VARIANT,
  cat_effect_scale       FLOAT,
  max_cardinality        NUMBER,
  missing_rate           FLOAT,
  -- Feature distribution
  feature_regime         STRING,
  covariance_type        STRING,
  rho                    FLOAT,
  active_fraction        FLOAT,
  noise_feature_fraction FLOAT,
  feature_noise_sigma    FLOAT,
  suite_component        STRING,
  condition_id           STRING,
  teacher_seed           NUMBER,
  sample_seed            NUMBER
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- ============================================================
-- RUNTIME VARIABLES -- set before CALL or injected by procedure
-- ============================================================
-- REGRESSION NONLINEAR
--   SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH  -> @MODEL_STAGE/checkpoints/best_nonlinear.pt
--   SYNREG_RESULTS_STAGE                  -> @EVALUATION_RESULTS_STAGE/nonlinear/<suite_id>
--   NONLINEAR_INDEX_TABLE                 -> NONLINEAR_REGRESSION_DATASET_INDEX (numeric)
--                                           or NONLINEAR_MIXED_REGRESSION_DATASET_INDEX (mixed)
--
-- CLASSIFICATION NONLINEAR
--   SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH  -> @MODEL_STAGE/checkpoints/best_nonlinear_cls.pt
--   SYNCLS_RESULTS_STAGE                  -> @EVALUATION_RESULTS_STAGE/nonlinear_classification/<suite_id>
--   NONLINEAR_CLS_INDEX_TABLE             -> NONLINEAR_CLASSIFICATION_DATASET_INDEX (numeric)
--                                           or NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX (mixed)
-- ============================================================

-- ============================================================
-- NONLINEAR REGRESSION EVALUATION PROCEDURES
-- Phases: Prep -> DeepSet -> Baseline -> AutoGluon SPCS -> Aggregation
-- ============================================================

-- Phase 1: Prep -- index nonlinear_v2 datasets
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_prep';

-- Phase 2: DeepSet GPU evaluation
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_deepset_evaluation';

-- Phase 3: Baseline CPU evaluation -- 1-arg (default shard count from env)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_baseline_evaluation_default';

-- Phase 3: Baseline CPU evaluation -- 2-arg (explicit shard count)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_baseline_evaluation_with_shards';

-- Phase 3: Baseline CPU evaluation -- 3-arg (shards + concurrency)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_baseline_evaluation_with_shards_and_concurrency';

-- Phase 4: AutoGluon SPCS evaluation -- 4-arg (CLUSTER_SHARDS=0 -> single-node, no Ray)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_evaluation(
  AG_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_evaluation_default';

-- Phase 4: AutoGluon SPCS evaluation -- 9-arg (all parameters explicit)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_evaluation(
  AG_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING,
  AUTOGLUON_TASK_CPUS INTEGER,
  RAY_READY_TIMEOUT_SECONDS INTEGER,
  WORKER_SUBMIT_STAGGER_SECONDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_evaluation_full';

-- Phase 5: Aggregation -- 1-arg
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_aggregation_default';

-- Phase 5: Aggregation -- 4-arg (explicit shard counts)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  EXPECTED_DEEPSET_SHARDS INTEGER,
  EXPECTED_BASELINE_SHARDS INTEGER,
  EXPECTED_AG_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_aggregation_full';

-- ============================================================
-- SPCS PREFLIGHT PROBES (nonlinear AutoGluon uses SPCS path)
-- Recommended preflight order:
--   1. CALL run_synthetic_nonlinear_autogluon_spcs_import_probe('<AG_IMAGE>', 1);
--   2. CALL run_synthetic_nonlinear_autogluon_spcs_session_probe('<AG_IMAGE>', 1);
--   3. CALL run_synthetic_nonlinear_autogluon_spcs_capacity_probe('<AG_IMAGE>', 6, 4, 6);
--   4. CALL run_synthetic_nonlinear_autogluon_spcs_worker_access_probe('<AG_IMAGE>', 6, 4, 6);
-- Only proceed to full evaluation after all four probes succeed.
-- ============================================================

-- Import probe -- measures container startup latency with preinstalled deps
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_import_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_import_probe';

-- Session probe -- validates OAuth token injection and Snowpark session creation
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_session_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_session_probe';

-- Capacity probe -- 4-arg: verifies containers start on AUTOGLUON_CPU_POOL
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_capacity_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_capacity_probe';

-- Capacity probe -- 7-arg: extended with Ray timeout, stagger, keep-on-failure
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_capacity_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  RAY_READY_TIMEOUT_SECONDS INTEGER,
  WORKER_SUBMIT_STAGGER_SECONDS INTEGER,
  KEEP_SUPPORT_JOBS_ON_FAILURE BOOLEAN
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_capacity_probe';

-- Worker-access probe -- 4-arg: validates dataset access via presigned URLs
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_worker_access_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_worker_access_probe';

-- Worker-access probe -- 6-arg: extended with stagger and keep-on-failure
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_worker_access_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  WORKER_SUBMIT_STAGGER_SECONDS INTEGER,
  KEEP_SUPPORT_JOBS_ON_FAILURE BOOLEAN
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_spcs_worker_access_probe';

-- ============================================================
-- MIXED-CATEGORICAL NONLINEAR REGRESSION EVALUATION PROCEDURES
-- Suite: suite_id='nonlinear_mixed_regression'
-- Eval table: NONLINEAR_MIXED_REGRESSION_DATASET_INDEX
-- Env: SYNREG_IS_MIXED_CATEGORICAL=true
--      SYNREG_INDEX_TABLE=NONLINEAR_MIXED_REGRESSION_DATASET_INDEX
-- ============================================================

-- Phase 1: Prep
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_prep';

-- Phase 2: DeepSet GPU evaluation
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_deepset_evaluation';

-- Phase 3: Baseline CPU evaluation -- 1-arg
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_baseline_evaluation';

-- Phase 3: Baseline CPU evaluation -- 2-arg (explicit shard count)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS           INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_baseline_evaluation_with_shards';

-- Phase 3: Baseline CPU evaluation -- 3-arg (shards + concurrency)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT  STRING,
  BASELINE_SHARDS            INTEGER,
  BASELINE_CONCURRENT_NODES  INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_baseline_evaluation_with_shards_and_concurrency';

-- Phase 4: AutoGluon SPCS evaluation -- 4-arg
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_autogluon_spcs_evaluation(
  AG_IMAGE                      STRING,
  AUTOGLUON_CLUSTER_SHARDS      INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD   INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_autogluon_spcs_evaluation';

-- Phase 4: AutoGluon SPCS evaluation -- 9-arg (all parameters explicit)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_autogluon_spcs_evaluation(
  AG_IMAGE                      STRING,
  AUTOGLUON_CLUSTER_SHARDS      INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD   INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS  INTEGER,
  AUTOGLUON_PRESETS             STRING,
  AUTOGLUON_TASK_CPUS           INTEGER,
  RAY_READY_TIMEOUT_SECONDS     INTEGER,
  WORKER_SUBMIT_STAGGER_SECONDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_autogluon_spcs_evaluation_full';

-- Phase 5: Aggregation -- 1-arg
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_aggregation';

-- Phase 5: Aggregation -- 4-arg (explicit expected shard counts)
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_mixed_regression_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  EXPECTED_DEEPSET_SHARDS   INTEGER,
  EXPECTED_BASELINE_SHARDS  INTEGER,
  EXPECTED_AG_SHARDS        INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_mixed_regression_aggregation_full';

-- ============================================================
-- RUNBOOK
-- Run these statements in order for each training + eval cycle.
-- Replace <ENV> with the Container Runtime image tag, e.g. '2.5.0-py311'.
-- Replace <AG_IMAGE> with the SPCS AutoGluon image reference.
-- ============================================================

-- -- STEP 1: Upload scripts to MODEL_STAGE
-- PUT file://src/*.py     @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- -- STEP 2: Build training index (numeric regression)
-- CALL build_meta_nonlinear_dataset_index(FALSE, 1000);
-- -- formula: train=800, val=100, test=100 for total=1000

-- -- STEP 3: (Optional) Build mixed / classification indices
-- CALL build_meta_nonlinear_dataset_index(TRUE, 1000);                       -- mixed regression
-- CALL build_meta_nonlinear_classification_dataset_index(FALSE, 1000);        -- numeric classification
-- CALL build_meta_nonlinear_classification_dataset_index(TRUE, 1000);         -- mixed classification

-- -- STEP 4: Pretrain backbone
-- CALL run_pretrain_pipeline_nonlinear(
--   'market_exchangeable_icl',
--   'synthetic_regression_nonlinear',
--   'inductive_forecasting'
-- );

-- -- STEP 5: HPO sweep
-- CALL run_hpo_pipeline(
--   'market_exchangeable_icl',
--   'synthetic_regression_nonlinear',
--   'inductive_forecasting',
--   'nonlinear_model',
--   '',
--   '@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt'
-- );

-- -- STEP 6: Final training
-- CALL run_model_training(
--   'market_exchangeable_icl',
--   'synthetic_regression_nonlinear',
--   'inductive_forecasting'
-- );

-- -- STEP 7: Stage eval datasets
-- PUT file://data/nonlinear_v2/poly_quad/*.parquet        @EVALUATION_DATASET_STAGE/nonlinear_v2/poly_quad/       AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://data/nonlinear_v2/sin_low/*.parquet          @EVALUATION_DATASET_STAGE/nonlinear_v2/sin_low/         AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://data/nonlinear_v2/hinge/*.parquet            @EVALUATION_DATASET_STAGE/nonlinear_v2/hinge/           AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://data/nonlinear_v2/sparse_interact/*.parquet  @EVALUATION_DATASET_STAGE/nonlinear_v2/sparse_interact/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://data/nonlinear_v2/mixed_linear/*.parquet     @EVALUATION_DATASET_STAGE/nonlinear_v2/mixed_linear/    AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://data/nonlinear_v2/demand_mono/*.parquet      @EVALUATION_DATASET_STAGE/nonlinear_v2/demand_mono/     AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://data/nonlinear_v2/nonlinear_v2_manifest.json @EVALUATION_DATASET_STAGE/nonlinear_v2/                 AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- -- STEP 8: SPCS preflight probes
-- CALL run_synthetic_nonlinear_autogluon_spcs_import_probe('<AG_IMAGE>', 1);
-- CALL run_synthetic_nonlinear_autogluon_spcs_session_probe('<AG_IMAGE>', 1);
-- CALL run_synthetic_nonlinear_autogluon_spcs_capacity_probe('<AG_IMAGE>', 6, 4, 6);
-- CALL run_synthetic_nonlinear_autogluon_spcs_worker_access_probe('<AG_IMAGE>', 6, 4, 6);

-- -- STEP 9: Regression evaluation (numeric)
-- CALL run_synthetic_nonlinear_prep('<ENV>');
-- CALL run_synthetic_nonlinear_deepset_evaluation('<ENV>');
-- CALL run_synthetic_nonlinear_baseline_evaluation('<ENV>');
-- CALL run_synthetic_nonlinear_autogluon_spcs_evaluation('<AG_IMAGE>', 6, 4, 1, 6, 300, 'best_quality', 1, 600, 1);
-- CALL run_synthetic_nonlinear_aggregation('<ENV>');

-- -- STEP 9b: Regression evaluation (mixed)
-- CALL run_synthetic_nonlinear_mixed_regression_prep('<ENV>');
-- CALL run_synthetic_nonlinear_mixed_regression_deepset_evaluation('<ENV>');
-- CALL run_synthetic_nonlinear_mixed_regression_baseline_evaluation('<ENV>');
-- CALL run_synthetic_nonlinear_mixed_regression_autogluon_spcs_evaluation('<AG_IMAGE>', 0, 1, 0);
-- CALL run_synthetic_nonlinear_mixed_regression_aggregation('<ENV>');

-- -- Verification
-- SELECT suite_id, COUNT(*) n FROM NONLINEAR_REGRESSION_DATASET_INDEX GROUP BY suite_id;
-- -- Expect: nonlinear_v1 -> 400, nonlinear -> 420

-- -- Utility: suspend all compute pools
-- ALTER COMPUTE POOL DEEPSET_GPU_POOL    SUSPEND;
-- ALTER COMPUTE POOL DEEPSET_CPU_POOL    SUSPEND;
-- ALTER COMPUTE POOL AUTOGLUON_CPU_POOL  SUSPEND;
-- ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
