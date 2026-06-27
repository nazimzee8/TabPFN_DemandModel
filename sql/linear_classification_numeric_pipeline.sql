-- Sampling policy: n ~ Poisson(200), p ~ Poisson(10), rejection-sampled n >= 5p.
-- 5% wide-p tail: p drawn from truncated Poisson(25) in [20,30] (~5% of tasks).
-- p_signal + p_noise = p_total = p; active_s = number of non-zero classification weight vectors.
-- See: docs/cursor_dataset_generation.md and src/data_generation/dgp_helpers.py
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

-- Stage for model checkpoints (best_classification.pt)
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
-- TRAINING INDEX TABLE
-- Numeric-only linear classification training tasks
-- ============================================================

-- 18-col schema matches build_meta_classification_dataset_index.py _write schema exactly.
-- (task_seed / schema_version removed: not emitted by the index builder, not consumed by
--  snowflake_io.py selects.  Switch to CREATE OR REPLACE so re-deployments enforce the schema.)
CREATE OR REPLACE TRANSIENT TABLE META_CLASSIFICATION_DATASET_INDEX (
  split                 STRING NOT NULL,
  task_id               STRING NOT NULL,
  stage_path            STRING NOT NULL,   -- linear/classification/numeric/{split}/{task_id}.parquet
  n                     NUMBER,
  p                     NUMBER,
  n_train               NUMBER,
  n_test                NUMBER,
  prior_regime          STRING,
  hpo_bucket            NUMBER,
  num_classes           NUMBER,
  classification_regime STRING,
  task_objective        STRING,
  p_signal              NUMBER,
  p_noise               NUMBER,
  p_total               NUMBER,
  class_imbalance_type  STRING,
  label_noise_rate      FLOAT,
  feature_noise_level   FLOAT
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, num_classes, classification_regime, p, n_train);

-- ============================================================
-- TRAINING INDEX BUILDER PROCEDURE
-- IS_MIXED_CATEGORICAL = FALSE → linear/classification/numeric/ subdir + META_CLASSIFICATION_DATASET_INDEX
-- ============================================================

CREATE OR REPLACE PROCEDURE build_meta_classification_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_training_job.py',
    '@MODEL_STAGE/scripts/task_routing.py',
    '@MODEL_STAGE/scripts/constants.py'
  )
  HANDLER = 'run_training_job.build_meta_classification_dataset_index_with_flag_and_total';

-- ============================================================
-- TRAINING PROCEDURES
-- ============================================================

CREATE OR REPLACE PROCEDURE run_pretrain_pipeline(
  MODEL_FAMILY         STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  GATE_HIDDEN_DIM      INTEGER
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

-- ============================================================
-- Execute ML Training Pipeline (Classification Numeric)
-- MODEL_FAMILY          = market_exchangeable_icl
-- TRAINING_DATA_FAMILY  = synthetic_linear_classification
-- MODEL_DESIGN_PATTERN  = inductive_forecasting
-- HPO_SWEEP_MODE        = lbacnp_model
-- MODEL_ARCH_VERSION    = model5_lbacnp
-- ============================================================

CALL build_meta_classification_dataset_index(FALSE, 1000);
-- Single pretrain run at fusion_gate_hidden_dim=64 (gate sweep removed;
-- HPO pins fusion_gate_hidden_dim=64 and no longer tunes this dimension).
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_linear_classification',
  'inductive_forecasting', 64
);
CALL run_hpo_pipeline(
  'market_exchangeable_icl', 'synthetic_linear_classification',
  'inductive_forecasting', 'lbacnp_model'
);
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_gate.*[.]pt';
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json;

CALL run_model_training(
  'market_exchangeable_icl',
  'synthetic_linear_classification',
  'inductive_forecasting'
);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;


-- ============================================================
-- EVALUATION INDEX TABLE
-- ============================================================

CREATE OR REPLACE TRANSIENT TABLE LINEAR_CLASSIFICATION_DATASET_INDEX (
  suite_id                 STRING,
  suite_family             STRING,
  dataset_id               NUMBER,
  dataset_seed             NUMBER,
  global_idx               NUMBER,
  stage_path               STRING,
  prior_regime             STRING,
  classification_regime    STRING,
  split_seeds              ARRAY,
  n_total                  NUMBER,
  n_train_default          NUMBER,
  n_holdout_default        NUMBER,
  p_signal                 NUMBER,
  p_noise                  NUMBER,
  p_total                  NUMBER,
  num_classes              NUMBER,
  feature_noise_level      FLOAT,
  training_size_anchor     BOOLEAN,
  eval_weight              FLOAT,
  payload_bytes            NUMBER,
  created_at               TIMESTAMP_NTZ,
  logical_dataset_key      STRING,
  source_suite_id          STRING,
  task_family              STRING,          -- "linear_classification"
  task_objective           STRING,          -- "inductive_classification"
  label_noise_rate         FLOAT,
  class_imbalance_type     STRING,
  temperature              FLOAT,
  margin_level             STRING,
  sample_complexity_bucket STRING,
  task_seed                NUMBER,          -- SeedSequence-derived seed for this eval task
  schema_version           STRING,          -- v1 and v2 readers are supported
  coeff_schema_version     STRING,          -- 'canonical_v1' = W_true always (p,K); NULL = legacy
  distribution_family      STRING,
  covariance_type          STRING,
  rho                      FLOAT,
  is_training_allowed      BOOLEAN,
  is_eval_only             BOOLEAN,
  is_ood                   BOOLEAN,
  is_hidden_holdout        BOOLEAN,
  difficulty_score         FLOAT,
  difficulty_tier          STRING,
  difficulty_reasons       STRING,
  estimated_memory_bytes   NUMBER,
  estimated_memory_gib     FLOAT,
  memory_class             STRING,
  hidden_holdout_suite_id  STRING,
  task_fingerprint         STRING
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- ============================================================
-- CLASSIFICATION LINEAR PROCEDURES
-- Results:    @EVALUATION_RESULTS_STAGE/linear/classification/numeric/<suite_id>/
-- Checkpoint: @MODEL_STAGE/checkpoints/best_classification.pt
-- Feature selector: train_f_classif
-- IS_MIXED_CATEGORICAL: FALSE (numeric only)
--
-- Runtime env vars consumed by run_linear_classification_evaluation.py:
--   SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH  → @MODEL_STAGE/checkpoints/best_classification.pt
--   SYNCLS_RESULTS_STAGE                  → @EVALUATION_RESULTS_STAGE/linear/classification/numeric/<suite_id>
--   TRAINING_DATA_FAMILY                  → synthetic_linear_classification
--   SYNTHETIC_CLASSIFICATION_SUITE_ID     → suite identifier
--   SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR → train_f_classif
--   SYNCLS_INDEX_TABLE                    → LINEAR_CLASSIFICATION_DATASET_INDEX
-- ============================================================

CREATE OR REPLACE PROCEDURE run_linear_classification_prep(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_linear_classification.py'
  )
  HANDLER = 'run_linear_classification_evaluation.run_linear_classification_prep';

CREATE OR REPLACE PROCEDURE run_linear_classification_deepset_evaluation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_classification_evaluation.py')
  HANDLER = 'run_linear_classification_evaluation.run_linear_classification_deepset_evaluation';

-- NOTE (metric_schema_version):
--   '1.0' = stored DGP teacher probabilities (NOT model inference; deprecated)
--   '2.0' = real MODEL5/LBACNP forward pass using actual checkpoint logits
-- Results with different metric_schema_version are NOT comparable.
-- Always filter by metric_schema_version in aggregation queries.

CREATE OR REPLACE PROCEDURE run_linear_classification_baseline_evaluation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS               INTEGER DEFAULT 10,
  BASELINE_CONCURRENT_NODES     INTEGER DEFAULT 10
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_classification_evaluation.py')
  HANDLER = 'run_linear_classification_evaluation.run_linear_classification_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_linear_classification_autogluon_evaluation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CONCURRENT_NODES    INTEGER DEFAULT 60
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_classification_evaluation.py')
  HANDLER = 'run_linear_classification_evaluation.run_linear_classification_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_linear_classification_aggregation(
  IS_MIXED_CATEGORICAL          BOOLEAN,
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_linear_classification_evaluation.py')
  HANDLER = 'run_linear_classification_evaluation.run_linear_classification_aggregation';

-- ============================================================
-- Setup SPCS image refs (set once; referenced by all eval CALLs via $var)
-- ============================================================
SET AUTOGLUON_IMAGE_REF = 'dvxcfsm-hs34800.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/model_image_repository/tabpfn-autogluon-ray:1.0.0';
SET SPCS_RAY_HEAD_DNS_SUFFIX = 'tabpfn_schema.tabpfn_db.snowflakecomputing.internal';

-- ============================================================
-- Execute Evaluation Pipeline (Classification Numeric)
-- (MODEL5/LBACNP: model_arch_version resolved to 'model5_lbacnp' by run_hpo_job.py
-- and written into best_config.json; run_model_training_job.py reads it from there)
-- ============================================================

-- Prepare evaluation data (generates LINEAR_CLASSIFICATION_DATASET_INDEX rows
-- and stages Parquet eval tasks under @EVALUATION_DATASET_STAGE)
CALL run_linear_classification_prep(FALSE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF);

-- DeepSet-MC inference (GPU, 10 nodes, Gloo direct)
CALL run_linear_classification_deepset_evaluation(FALSE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;

-- Baseline methods (CPU, 9 methods × target_instances=3, all in parallel → 27 nodes)
CALL run_linear_classification_baseline_evaluation(FALSE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;

-- AutoGluon (CPU, 60 concurrent nodes, AUTOGLUON_CPU_POOL)
CALL run_linear_classification_autogluon_evaluation(FALSE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF, 60);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

-- Aggregation (I/O only, target_instances=1, world_size=1; resume CPU pool for warehouse compute)
ALTER COMPUTE POOL DEEPSET_CPU_POOL RESUME;
CALL run_linear_classification_aggregation(FALSE, '2.5.0-py311', '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL DEEPSET_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
