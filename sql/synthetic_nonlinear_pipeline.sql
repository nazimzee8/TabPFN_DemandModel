SELECT CURRENT_USER();
SELECT
    CURRENT_ORGANIZATION_NAME() AS orgname,
    CURRENT_ACCOUNT_NAME() AS acctname;

CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
CREATE SCHEMA  IF NOT EXISTS TABPFN_DB.TABPFN_SCHEMA;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- Stage for training data (Parquet files)
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- META_NONLINEAR_DATASET_STAGE: nonlinear training parquet files.
CREATE STAGE IF NOT EXISTS META_NONLINEAR_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for model checkpoints (best.pt)
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

CREATE COMPUTE POOL IF NOT EXISTS DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = GPU_NV_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

CREATE COMPUTE POOL IF NOT EXISTS DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

CREATE COMPUTE POOL IF NOT EXISTS AUTOGLUON_CPU_POOL
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

CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_DATASET_INDEX (
  split        STRING NOT NULL,
  task_id      STRING NOT NULL,
  stage_path   STRING NOT NULL,
  n            NUMBER,
  p            NUMBER,
  n_train      NUMBER,
  n_test       NUMBER,
  prior_regime STRING,
  hpo_bucket   NUMBER
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p, n_train);

-- Training procedures are registered by sql/run_training_job.sql.
-- Do not redefine them here to avoid stale overloads.

-- 0. Build nonlinear training index (run once after staging parquets)
CALL build_meta_nonlinear_dataset_index();

-- 1. Nonlinear pretrain — writes pretrain_nonlinear_meta.pt
CALL run_pretrain_pipeline_nonlinear(
  'market_exchangeable_icl',
  'synthetic_regression_nonlinear',
  'inductive_forecasting'
);

-- 2. HPO (single nonlinear_meta sweep with pretrain warm-start)
CALL run_hpo_pipeline(
  'market_exchangeable_icl',
  'synthetic_regression_nonlinear',
  'inductive_forecasting',
  'nonlinear_meta',
  '',
  '@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt'
);

-- 3. Final training
CALL run_model_training(
  'market_exchangeable_icl',
  'synthetic_regression_nonlinear',
  'inductive_forecasting'
);

-- Evaluation index and procedures: see sql/05_synthetic_nonlinear_evaluation_pipeline.sql
