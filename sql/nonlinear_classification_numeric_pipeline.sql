-- Sampling policy: n ~ Poisson(200), p ~ Poisson(10), rejection-sampled n >= 5p.
-- 5% wide-p tail: p drawn from truncated Poisson(25) in [20,30] (~5% of tasks).
-- p_signal + p_noise = p_total = p; active_s = number of non-zero beta coefficients.
-- See: docs/cursor_dataset_generation.md and src/data_generation/dgp_helpers.py:641-661
SELECT CURRENT_USER();
SELECT
    CURRENT_ORGANIZATION_NAME() AS orgname,
    CURRENT_ACCOUNT_NAME() AS acctname;

CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
CREATE SCHEMA  IF NOT EXISTS TABPFN_DB.TABPFN_SCHEMA;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- Consolidated training stage for all families (Parquet files)
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for model checkpoints (best_nonlinear.pt and best_nonlinear_cls.pt)
CREATE STAGE IF NOT EXISTS MODEL_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for benchmark suites
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for deploying Snowflake ML jobs
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for synthetic eval data (Parquet files)
CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

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
  MAX_NODES = 27  -- 9 baseline methods × target_instances=3 each; all run in parallel
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

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TABPFN_PYPI_EAI
  ALLOWED_NETWORK_RULES = (SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE)
  ENABLED = TRUE
  COMMENT = 'Allows TabPFN ML Jobs to install approved PyPI dependencies in Snowflake-managed Container Runtime.';
GRANT USAGE ON INTEGRATION TABPFN_PYPI_EAI TO ROLE ACCOUNTADMIN;
GRANT DATABASE ROLE SNOWFLAKE.PYPI_REPOSITORY_USER TO ROLE ACCOUNTADMIN;

-- ============================================================
-- NONLINEAR CLASSIFICATION TRAINING DATA STAGE
-- Consolidated into META_DATASET_STAGE (created above).
-- Layout:
--   @META_DATASET_STAGE/nonlinear/classification/numeric/{train,val,test}/
--   @META_DATASET_STAGE/nonlinear/classification/mixed/{train,val,test}/
-- ============================================================
-- (No separate stage CREATE needed — META_DATASET_STAGE covers all families.)

-- ============================================================
-- TRAINING INDEX TABLE
-- Numeric-only nonlinear classification training tasks.
-- CREATE TRANSIENT TABLE IF NOT EXISTS preserves existing data
-- across pipeline re-runs.
-- ============================================================
CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_CLASSIFICATION_DATASET_INDEX (
  split                  STRING NOT NULL,
  task_id                STRING NOT NULL,
  stage_path             STRING NOT NULL,  -- nonlinear/classification/numeric/{split}/{task_id}.parquet
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

-- ============================================================
-- INDEX BUILDER PROCEDURE
-- IS_MIXED_CATEGORICAL=FALSE → numeric/ subdir + numeric index table
-- IS_MIXED_CATEGORICAL=TRUE  → mixed/   subdir + mixed   index table
-- ============================================================

CREATE OR REPLACE PROCEDURE build_meta_nonlinear_classification_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_training_job.py',
    '@MODEL_STAGE/scripts/task_routing.py',
    '@MODEL_STAGE/scripts/constants.py'
  )
  HANDLER = 'run_training_job.build_meta_nonlinear_classification_dataset_index_with_flag_and_total';

-- ============================================================
-- PRETRAIN PROCEDURE (nonlinear backbone, 3-arg form)
-- Produces: @MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt
-- Shared across all nonlinear suites (regression + classification).
-- ============================================================

CREATE OR REPLACE PROCEDURE run_pretrain_pipeline_nonlinear(
  MODEL_FAMILY         STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_pretrain_job.py',
    '@MODEL_STAGE/scripts/task_routing.py',
    '@MODEL_STAGE/scripts/constants.py'
  )
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline_nonlinear_model';

-- ============================================================
-- HPO PIPELINE PROCEDURE (shared across suites)
-- ============================================================

CREATE OR REPLACE PROCEDURE run_hpo_pipeline(
  MODEL_FAMILY         STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  HPO_SWEEP_MODE       STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline_model_sweep';

-- ============================================================
-- MODEL TRAINING PROCEDURE (shared across suites)
-- ============================================================

CREATE OR REPLACE PROCEDURE run_model_training(
  MODEL_FAMILY         STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_model_training_job.py',
    '@MODEL_STAGE/scripts/task_routing.py',
    '@MODEL_STAGE/scripts/constants.py'
  )
  HANDLER = 'run_model_training_job.run_model_training_model';

GRANT DATABASE ROLE SNOWFLAKE.PYPI_REPOSITORY_USER TO ROLE ACCOUNTADMIN;

-- MODEL_ARCH_VERSION for this pipeline: 'model5_lbacnp'
-- run_hpo_job.py propagates MODEL_ARCH_VERSION='model5_lbacnp' in env_vars when HPO_SWEEP_MODE='lbacnp_model'.
-- train.py reads model_arch_version exclusively from best_config.json (written by HPO).
-- Snowflake SET session variables are NOT visible to stored-proc Python handlers; no SET needed.

-- ============================================================
-- TRAINING CALL CHAIN (nonlinear classification, numeric)
-- TRAINING_DATA_FAMILY = 'synthetic_nonlinear_classification'
-- MODEL_FAMILY         = 'market_exchangeable_icl'
-- MODEL_DESIGN_PATTERN = 'inductive_forecasting'
-- HPO_SWEEP_MODE       = 'lbacnp_model'
-- IS_MIXED_CATEGORICAL = FALSE
-- ============================================================

CALL build_meta_nonlinear_classification_dataset_index(FALSE, 1000);

-- Single pretrain run producing pretrain_nonlinear_meta.pt
-- (no gate sweep; HPO pins fusion_gate_hidden_dim and does not tune it)
CALL run_pretrain_pipeline_nonlinear(
  'market_exchangeable_icl', 'synthetic_nonlinear_classification',
  'inductive_forecasting'
);
CALL run_hpo_pipeline(
  'market_exchangeable_icl', 'synthetic_nonlinear_classification',
  'inductive_forecasting', 'lbacnp_model'
);
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_nonlinear_meta[.]pt';
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json;

CALL run_model_training(
  'market_exchangeable_icl',
  'synthetic_nonlinear_classification',
  'inductive_forecasting'
);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;


-- ============================================================
-- EVALUATION INDEX TABLE (nonlinear classification, numeric)
-- 38 columns: 23 base + 15 classification-specific.
-- CREATE OR REPLACE — this table is recreated each eval cycle;
-- ALTER TABLE migrations below are idempotent guard-rails.
-- ============================================================

CREATE OR REPLACE TRANSIENT TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX (
  suite_id                   STRING,
  suite_family               STRING,
  dataset_id                 NUMBER,
  dataset_seed               NUMBER,
  stage_path                 STRING,
  prior_name                 STRING,
  prior_version              STRING,
  prior_regime               STRING,
  split_seeds                ARRAY,
  n_total                    NUMBER,
  n_train_default            NUMBER,
  n_holdout_default          NUMBER,
  p_signal                   NUMBER,
  p_noise                    NUMBER,
  p_total                    NUMBER,
  target_noise_scale         FLOAT,
  training_size_anchor       BOOLEAN,
  feature_noise_level        FLOAT,
  eval_weight                FLOAT,
  payload_bytes              NUMBER,
  created_at                 TIMESTAMP_NTZ,
  logical_dataset_key        STRING,
  source_suite_id            STRING,
  feature_regime             STRING,
  covariance_type            STRING,
  rho                        FLOAT,
  suite_component            STRING,
  condition_id               STRING,
  teacher_seed               NUMBER,
  sample_seed                NUMBER,
  num_classes                NUMBER,
  label_noise_rate           FLOAT,
  class_imbalance_type       STRING,
  margin_level               STRING,
  temperature                FLOAT,
  realized_num_classes       NUMBER,
  realized_label_noise_rate  FLOAT,
  realized_margin_level      STRING
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- Schema migration — idempotent, safe to re-run
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS feature_regime             STRING;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS covariance_type            STRING;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS rho                        FLOAT;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS suite_component            STRING;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS condition_id               STRING;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS teacher_seed               NUMBER;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS sample_seed                NUMBER;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS num_classes                NUMBER;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS label_noise_rate           FLOAT;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS class_imbalance_type       STRING;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS margin_level               STRING;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS temperature                FLOAT;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS realized_num_classes       NUMBER;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS realized_label_noise_rate  FLOAT;
ALTER TABLE NONLINEAR_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS realized_margin_level      STRING;

-- ============================================================
-- UNIVERSAL NONLINEAR CLASSIFICATION EVALUATION PROCEDURES
-- One proc name per phase; IS_MIXED_CATEGORICAL BOOLEAN (first param, required)
-- routes between numeric (FALSE) and mixed-categorical (TRUE) suites.
-- Handler module:    run_nonlinear_classification_evaluation
-- Checkpoint:        @MODEL_STAGE/checkpoints/best_nonlinear_cls.pt
-- Numeric results:   @EVALUATION_RESULTS_STAGE/nonlinear/classification/numeric/{suite_id}/
-- Mixed results:     @EVALUATION_RESULTS_STAGE/nonlinear/classification/mixed/{suite_id}/
-- Feature selector:  train_f_classif
--
-- Stage dependencies (PUT before running procedures):
--   PUT file://scripts/evaluation/run_nonlinear_classification_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/evaluation/run_linear_classification_evaluation.py    @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/preparation/prepare_nonlinear_classification.py       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/evaluation/evaluate_nonlinear_classification.py       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://src/evaluation/baseline_models.py                             @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://src/evaluation/autogluon_models.py                            @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- ============================================================

-- =============================================================================
-- Phase 1: Prep — index nonlinear classification datasets
-- =============================================================================

CREATE OR REPLACE PROCEDURE run_nonlinear_classification_prep(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_prep';


-- =============================================================================
-- Phase 2: DeepSet GPU evaluation
-- =============================================================================

CREATE OR REPLACE PROCEDURE run_nonlinear_classification_deepset_evaluation(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_deepset_evaluation';


-- =============================================================================
-- Phase 3: Baseline CPU evaluation
-- Overload 1: bench_rt + boolean (default shards)
-- Overload 2: bench_rt + shards + boolean
-- =============================================================================

-- Overload 1: bench_rt + boolean (default shard count)
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_baseline_evaluation(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_baseline_evaluation';

-- Overload 2: is_mixed + bench_rt + shards
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_baseline_evaluation(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING, BASELINE_SHARDS INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_baseline_evaluation';


-- =============================================================================
-- Phase 4: AutoGluon evaluation
-- Overload 1: bench_rt + ag_rt + boolean (default shard/concurrency params)
-- Overload 2: full 6-arg + boolean
-- Default: 60 concurrent nodes (AUTOGLUON_CPU_POOL MAX_NODES = 60).
-- =============================================================================

-- Overload 1: is_mixed + bench_rt + ag_rt (defaults)
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_autogluon_evaluation(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING, AG_RUNTIME_ENVIRONMENT STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_autogluon_evaluation';

-- Overload 2: is_mixed + explicit shard count + tuning params
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_autogluon_evaluation(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING, AG_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_SHARDS INTEGER, AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_TIME_LIMIT INTEGER, AUTOGLUON_PRESETS STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_autogluon_evaluation';


-- =============================================================================
-- Phase 5: Aggregation
-- Overload 1: bench_rt + boolean (default expected shard counts)
-- Overload 2: explicit shards + boolean
-- =============================================================================

-- Overload 1: is_mixed + bench_rt (default expected shard counts)
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_aggregation(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_aggregation';

-- Overload 2: is_mixed + explicit expected shard counts
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_aggregation(
  IS_MIXED_CATEGORICAL BOOLEAN, BENCH_RUNTIME_ENVIRONMENT STRING,
  EXPECTED_DEEPSET_SHARDS INTEGER, EXPECTED_BASELINE_SHARDS INTEGER, EXPECTED_AG_SHARDS INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_aggregation';


-- Setup SPCS image for AutoGluon
SET AUTOGLUON_IMAGE_REF = 'dvxcfsm-hs34800.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/model_image_repository/tabpfn-autogluon-ray:1.0.0';
SET SPCS_RAY_HEAD_DNS_SUFFIX = 'tabpfn_schema.tabpfn_db.snowflakecomputing.internal';

-- ============================================================
-- EVALUATION CALL CHAIN (nonlinear classification, numeric)
-- MODEL5/LBACNP: model_arch_version resolved to 'model5_lbacnp' by run_hpo_job.py
-- and written into best_config.json; run_model_training_job.py reads it from there.
-- ============================================================

-- Prepare evaluation data (indexes NONLINEAR_CLASSIFICATION_DATASET_INDEX)
CALL run_nonlinear_classification_prep(FALSE, '2.5.0-py311');

-- DeepSet MC evaluation (GPU pool, checkpoint: best_nonlinear_cls.pt)
CALL run_nonlinear_classification_deepset_evaluation(FALSE, '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;

-- Baseline CPU evaluation (9 methods, plain baseline, all in parallel)
CALL run_nonlinear_classification_baseline_evaluation(FALSE, '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;

-- AutoGluon evaluation (60 concurrent nodes on AUTOGLUON_CPU_POOL)
CALL run_nonlinear_classification_autogluon_evaluation(FALSE, '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

-- Aggregation (I/O only; needs CPU pool)
ALTER COMPUTE POOL DEEPSET_CPU_POOL RESUME;
CALL run_nonlinear_classification_aggregation(FALSE, '2.5.0-py311');

ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL DEEPSET_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SET AUTO_SUSPEND_SECS = 300;


