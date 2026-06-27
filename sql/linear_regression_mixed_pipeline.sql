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

-- Stage for model checkpoints (best_regression.pt and best_linear_cls.pt)
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

-- Mixed-categorical linear regression training tasks
-- (stored under @META_DATASET_STAGE/linear/regression/mixed/)
CREATE TRANSIENT TABLE IF NOT EXISTS META_MIXED_REGRESSION_DATASET_INDEX (
  split                    STRING  NOT NULL,
  task_id                  STRING  NOT NULL,
  stage_path               STRING  NOT NULL,  -- linear/regression/mixed/{split}/{task_id}.parquet
  n                        NUMBER,
  p                        NUMBER,            -- p_num + p_cat (required by RUNTIME_INDEX_COLUMNS)
  p_num                    NUMBER,
  p_cat                    NUMBER,
  n_train                  NUMBER,
  n_test                   NUMBER,
  prior_regime             STRING,
  hpo_bucket               NUMBER,
  schema_version           STRING,
  task_family              STRING,            -- e.g. 'linear_regression'
  training_data_family     STRING,            -- e.g. 'synthetic_linear_regression_mixed_categorical'
  task_objective           STRING,            -- e.g. 'inductive_regression'
  categorical_cardinalities ARRAY            -- JSON array of per-feature cardinalities
) DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p_num, p_cat, n_train);

CREATE OR REPLACE PROCEDURE build_meta_regression_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_training_job.py',
    '@MODEL_STAGE/scripts/task_routing.py',
    '@MODEL_STAGE/scripts/constants.py'
  )
  HANDLER = 'run_training_job.build_meta_regression_dataset_index_with_flag_and_total';

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
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_pretrain_job.py',
    '@MODEL_STAGE/scripts/task_routing.py',
    '@MODEL_STAGE/scripts/constants.py'
  )
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline_model_gate';

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

CREATE OR REPLACE PROCEDURE run_model_training(
  MODEL_FAMILY STRING,
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

-- Execute ML Training Pipeline (Mixed-Categorical Regression)
-- MODEL_FAMILY            = market_exchangeable_icl
-- TRAINING_DATA_FAMILY    = synthetic_linear_regression_mixed_categorical
-- MODEL_DESIGN_PATTERN    = inductive_forecasting
-- HPO_SWEEP_MODE          = lbacnp_model
-- IS_MIXED_CATEGORICAL    = TRUE  → mixed/ subdir + META_MIXED_REGRESSION_DATASET_INDEX
CALL build_meta_regression_dataset_index(TRUE, 1000);
-- Single pretrain run at fusion_gate_hidden_dim=64 (gate sweep removed;
-- HPO pins fusion_gate_hidden_dim=64 and no longer tunes this dimension).
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_linear_regression_mixed_categorical',
  'inductive_forecasting', 64
);
CALL run_hpo_pipeline(
  'market_exchangeable_icl', 'synthetic_linear_regression_mixed_categorical',
  'inductive_forecasting', 'lbacnp_model'
);
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_gate.*[.]pt';
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json;

CALL run_model_training(
  'market_exchangeable_icl',
  'synthetic_linear_regression_mixed_categorical',
  'inductive_forecasting'
);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;


-- ============================================================
-- REGRESSION LINEAR MIXED-CATEGORICAL EVAL INDEX TABLE
-- Results: @EVALUATION_RESULTS_STAGE/linear/regression/mixed/<suite_id>/
-- Checkpoint: @MODEL_STAGE/checkpoints/best_regression.pt
-- Feature selector: train_f_regression
-- Routing env var: SYNREG_IS_MIXED_CATEGORICAL (set TRUE by mixed handlers)
-- ============================================================

CREATE OR REPLACE TRANSIENT TABLE LINEAR_MIXED_REGRESSION_DATASET_INDEX (
  suite_id             STRING,
  suite_family         STRING,
  dataset_id           NUMBER,
  dataset_seed         NUMBER,
  global_idx           NUMBER,
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
  -- Mixed-categorical columns
  p_num                NUMBER,
  p_cat                NUMBER,
  categorical_cardinalities ARRAY,      -- JSON array [3, 5, 10, ...]
  max_cardinality      NUMBER,
  cat_effect_scale     FLOAT,
  missing_rate         FLOAT,
  -- Standard regression columns
  target_noise_scale   FLOAT,
  training_size_anchor BOOLEAN,
  feature_noise_level  FLOAT,
  eval_weight          FLOAT,
  payload_bytes        NUMBER,
  created_at           TIMESTAMP_NTZ,
  logical_dataset_key  STRING,
  source_suite_id      STRING,
  base_task_id         STRING,          -- NULL for standalone; set for paired perturbation tasks
  perturbation_condition STRING,        -- e.g. 'feature_noise_0.10', 'target_noise_2.0'
  task_seed            NUMBER,          -- seed for reproducibility
  schema_version       STRING,
  task_family          STRING,
  distribution_family  STRING,
  covariance_type      STRING,
  rho                  FLOAT,
  target_noise_type    STRING,
  is_training_allowed  BOOLEAN,
  is_eval_only         BOOLEAN,
  is_ood               BOOLEAN,
  is_hidden_holdout    BOOLEAN,
  difficulty_score     FLOAT,
  difficulty_tier      STRING,
  difficulty_reasons   STRING,
  estimated_memory_bytes NUMBER,
  estimated_memory_gib FLOAT,
  memory_class         STRING,
  hidden_holdout_suite_id STRING,
  task_fingerprint     STRING,
  training_data_family STRING,
  task_objective       STRING
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- ============================================================
-- REGRESSION LINEAR PROCEDURES (universal — numeric + mixed)
-- Results: @EVALUATION_RESULTS_STAGE/linear/regression/<numeric|mixed>/<suite_id>/
-- Checkpoint: @MODEL_STAGE/checkpoints/best_regression.pt
-- Feature selector: train_f_regression
-- IS_MIXED_CATEGORICAL BOOLEAN routes each handler to the
-- numeric or mixed output path (FALSE = numeric, TRUE = mixed).
-- These proc definitions are IDENTICAL to linear_regression_numeric_pipeline.sql;
-- only the CALL statements below differ (TRUE instead of FALSE).
-- ============================================================

CREATE OR REPLACE PROCEDURE run_linear_regression_prep(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_linear_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_linear_regression.py'
  )
  HANDLER = 'run_linear_regression_evaluation.run_linear_regression_prep';

CREATE OR REPLACE PROCEDURE run_linear_regression_deepset_evaluation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_regression_evaluation.py')
  HANDLER = 'run_linear_regression_evaluation.run_linear_regression_deepset_evaluation';

CREATE OR REPLACE PROCEDURE run_linear_regression_baseline_evaluation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS               INTEGER DEFAULT 6,
  BASELINE_CONCURRENT_NODES     INTEGER DEFAULT 6
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_regression_evaluation.py')
  HANDLER = 'run_linear_regression_evaluation.run_linear_regression_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_linear_regression_autogluon_evaluation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CONCURRENT_NODES    INTEGER DEFAULT 60
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_regression_evaluation.py')
  HANDLER = 'run_linear_regression_evaluation.run_linear_regression_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_linear_regression_aggregation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_regression_evaluation.py')
  HANDLER = 'run_linear_regression_evaluation.run_linear_regression_aggregation';

-- Setup SPCS image for AutoGluon
SET AUTOGLUON_IMAGE_REF = 'dvxcfsm-hs34800.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/model_image_repository/tabpfn-autogluon-ray:1.0.0';
SET SPCS_RAY_HEAD_DNS_SUFFIX = 'tabpfn_schema.tabpfn_db.snowflakecomputing.internal';

-- Prepare evaluation data (mixed-categorical)
CALL run_linear_regression_prep(TRUE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF);

-- Evaluation (MODEL5/LBACNP: model_arch_version='model5_lbacnp' is written into best_config.json
-- by HPO; train.py reads it from hyper_params["model_arch_version"] at runtime)
-- Checkpoint: @MODEL_STAGE/checkpoints/best_regression.pt
CALL run_linear_regression_deepset_evaluation(TRUE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_linear_regression_baseline_evaluation(TRUE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF, 6, 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_linear_regression_autogluon_evaluation(TRUE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF, 60);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
ALTER COMPUTE POOL DEEPSET_CPU_POOL RESUME;
CALL run_linear_regression_aggregation(TRUE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL DEEPSET_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
