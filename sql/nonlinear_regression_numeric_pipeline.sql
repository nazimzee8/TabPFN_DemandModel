-- Nonlinear regression (numeric-only) training + evaluation pipeline.
-- TRAINING_DATA_FAMILY = 'synthetic_regression_nonlinear'
-- Feature selector: train_f_regression
-- Checkpoint: @MODEL_STAGE/checkpoints/best_regression.pt
-- Results: @EVALUATION_RESULTS_STAGE/nonlinear/regression/numeric/{suite_id}/
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

-- Stage for model checkpoints (best_regression.pt)
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

-- Numeric-only nonlinear regression training tasks
-- 23-col schema matches build_meta_nonlinear_regression_dataset_index.py _write_index schema.
-- feature_regime is also required by the nonlinear HPO SELECT in snowflake_io.py.
CREATE OR REPLACE TRANSIENT TABLE META_NONLINEAR_REGRESSION_DATASET_INDEX (
  split                    STRING NOT NULL,
  task_id                  STRING NOT NULL,
  stage_path               STRING NOT NULL,   -- nonlinear/regression/numeric/{split}/{task_id}.parquet
  n                        NUMBER,
  p                        NUMBER,
  n_train                  NUMBER,
  n_test                   NUMBER,
  prior_regime             STRING,
  hpo_bucket               NUMBER,
  profile                  STRING,
  feature_regime           STRING,
  p_signal                 NUMBER,
  p_noise                  NUMBER,
  p_total                  NUMBER,
  active_s                 NUMBER,
  sparsity_ratio           FLOAT,
  covariance_type          STRING,
  rho                      FLOAT,
  target_noise_scale       FLOAT,
  feature_noise_level      FLOAT,
  sample_complexity_bucket STRING,
  has_noise_features       BOOLEAN,
  has_feature_noise        BOOLEAN
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p, n_train);

CREATE OR REPLACE PROCEDURE build_meta_nonlinear_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_training_job.py',
    '@MODEL_STAGE/scripts/task_routing.py',
    '@MODEL_STAGE/scripts/constants.py'
  )
  HANDLER = 'run_training_job.build_meta_nonlinear_dataset_index_with_flag_and_total';

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

-- Execute ML Training Pipeline (Nonlinear Regression Numeric)
CALL build_meta_nonlinear_dataset_index(FALSE, 1000);
-- Nonlinear pretrain: no gate-dim argument; produces pretrain_nonlinear_meta.pt.
CALL run_pretrain_pipeline_nonlinear(
  'market_exchangeable_icl',
  'synthetic_regression_nonlinear',
  'inductive_forecasting'
);
CALL run_hpo_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_nonlinear',
  'inductive_forecasting', 'lbacnp_model'
);
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_nonlinear_meta[.]pt';
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json;

CALL run_model_training(
  'market_exchangeable_icl',
  'synthetic_regression_nonlinear',
  'inductive_forecasting'
);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;


-- ============================================================
-- NONLINEAR REGRESSION NUMERIC EVALUATION PROCEDURES
-- Results: @EVALUATION_RESULTS_STAGE/nonlinear/regression/numeric/{suite_id}/
-- Checkpoint: @MODEL_STAGE/checkpoints/best_regression.pt
-- Feature selector: train_f_regression
-- ============================================================

CREATE OR REPLACE TRANSIENT TABLE NONLINEAR_REGRESSION_DATASET_INDEX (
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
  feature_noise_level    FLOAT,
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

-- Idempotent migration: add the 4 mixed-categorical columns that the prep INSERT always emits
-- (p_num and p_cat are NULL for pure-numeric rows; columns match the mixed-pipeline DDL).
-- Matches nonlinear_regression_mixed_pipeline.sql and nonlinear_classification_numeric_pipeline.sql.
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS
  p_num                     NUMBER;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS
  p_cat                     NUMBER;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS
  categorical_cardinalities ARRAY;
ALTER TABLE NONLINEAR_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS
  max_cardinality           NUMBER;

CREATE OR REPLACE PROCEDURE run_nonlinear_regression_prep(
  IS_MIXED_CATEGORICAL      BOOLEAN,
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_prep';

CREATE OR REPLACE PROCEDURE run_nonlinear_regression_deepset_evaluation(
  IS_MIXED_CATEGORICAL      BOOLEAN,
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_deepset_evaluation';

CREATE OR REPLACE PROCEDURE run_nonlinear_regression_baseline_evaluation(
  IS_MIXED_CATEGORICAL       BOOLEAN,
  BENCH_RUNTIME_ENVIRONMENT  STRING,
  BASELINE_SHARDS            INTEGER DEFAULT 6,
  BASELINE_CONCURRENT_NODES  INTEGER DEFAULT 6
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_nonlinear_regression_autogluon_evaluation(
  IS_MIXED_CATEGORICAL           BOOLEAN,
  AG_RUNTIME_ENVIRONMENT         STRING,
  AUTOGLUON_CLUSTER_SHARDS       INTEGER DEFAULT 0,
  AUTOGLUON_WORKERS_PER_SHARD    INTEGER DEFAULT 1,
  AUTOGLUON_CONCURRENT_CLUSTERS  INTEGER DEFAULT 0
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_nonlinear_regression_aggregation(
  IS_MIXED_CATEGORICAL      BOOLEAN,
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/run_linear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/evaluate_nonlinear_regression.py',
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_regression_evaluation.run_nonlinear_regression_aggregation';

-- Setup SPCS image for AutoGluon
SET AUTOGLUON_IMAGE_REF = 'dvxcfsm-hs34800.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/model_image_repository/tabpfn-autogluon-ray:1.0.0';
SET SPCS_RAY_HEAD_DNS_SUFFIX = 'tabpfn_schema.tabpfn_db.snowflakecomputing.internal';

-- Prepare evaluation data
CALL run_nonlinear_regression_prep(FALSE, '2.5.0-py311');

-- Evaluation (MODEL5/LBACNP: model_arch_version resolved to 'model5_lbacnp' by run_hpo_job.py
-- and written into best_config.json; run_model_training_job.py reads it from there)
CALL run_nonlinear_regression_deepset_evaluation(FALSE, '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_nonlinear_regression_baseline_evaluation(FALSE, '2.5.0-py311', 6, 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_nonlinear_regression_autogluon_evaluation(FALSE, $AUTOGLUON_IMAGE_REF, 0, 1, 0);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
ALTER COMPUTE POOL DEEPSET_CPU_POOL RESUME;
CALL run_nonlinear_regression_aggregation(FALSE, '2.5.0-py311');
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL DEEPSET_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
