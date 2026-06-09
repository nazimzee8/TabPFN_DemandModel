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
--   @META_REGRESSION_DATASET_STAGE/numeric/{train,val,test}/
--   @META_REGRESSION_DATASET_STAGE/mixed/{train,val,test}/
--   @META_CLASSIFICATION_DATASET_STAGE/numeric/{train,val,test}/
--   @META_CLASSIFICATION_DATASET_STAGE/mixed/{train,val,test}/
-- ============================================================
CREATE STAGE IF NOT EXISTS META_REGRESSION_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

CREATE STAGE IF NOT EXISTS META_CLASSIFICATION_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for model checkpoints (best.pt)
CREATE STAGE IF NOT EXISTS MODEL_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for benchmark suite result CSVs
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for ML-job payload packages
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for synthetic eval data (Parquet files)
CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ============================================================
-- STAGE PATH CONTRACT
-- Training data (one-row-per-task index):
--   @META_REGRESSION_DATASET_STAGE/numeric/{split}/{task_id}.parquet
--   @META_REGRESSION_DATASET_STAGE/mixed/{split}/{task_id}.parquet
--   @META_CLASSIFICATION_DATASET_STAGE/numeric/{split}/{task_id}.parquet
--   @META_CLASSIFICATION_DATASET_STAGE/mixed/{split}/{task_id}.parquet
-- Eval data (one-row-per-sample, separate schema):
--   @EVALUATION_DATASET_STAGE/linear/{filename}.parquet
--   @EVALUATION_DATASET_STAGE/synthetic_classification_prepared/{suite_id}/{family}/
-- Results:
--   @EVALUATION_RESULTS_STAGE/linear/regression/numeric/<suite_id>/
--   @EVALUATION_RESULTS_STAGE/linear/regression/mixed/<suite_id>/
--   @EVALUATION_RESULTS_STAGE/linear/classification/numeric/<suite_id>/
--   @EVALUATION_RESULTS_STAGE/linear/classification/mixed/<suite_id>/
-- ============================================================

-- ============================================================
-- COMPUTE POOLS
-- ============================================================
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
  MAX_NODES = 10
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
-- TRAINING INDEX TABLES
-- ============================================================

-- Numeric-only linear regression training tasks
CREATE TRANSIENT TABLE IF NOT EXISTS META_REGRESSION_DATASET_INDEX (
  split              STRING NOT NULL,
  task_id            STRING NOT NULL,
  stage_path         STRING NOT NULL,   -- numeric/{split}/{task_id}.parquet
  n                  NUMBER,
  p                  NUMBER,
  n_train            NUMBER,
  n_test             NUMBER,
  prior_regime       STRING,
  hpo_bucket         NUMBER,
  task_seed          NUMBER,            -- SeedSequence-derived task seed
  schema_version     STRING             -- schema version of parquet source
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p, n_train);

-- Mixed-categorical linear regression training tasks
-- (stored under @META_REGRESSION_DATASET_STAGE/mixed/)
CREATE TRANSIENT TABLE IF NOT EXISTS META_MIXED_REGRESSION_DATASET_INDEX (
  split              STRING  NOT NULL,
  task_id            STRING  NOT NULL,
  stage_path         STRING  NOT NULL,  -- mixed/{split}/{task_id}.parquet
  n                  NUMBER,
  p                  NUMBER,            -- p_num + p_cat (required by RUNTIME_INDEX_COLUMNS)
  p_num              NUMBER,
  p_cat              NUMBER,
  n_train            NUMBER,
  n_test             NUMBER,
  prior_regime       STRING,
  hpo_bucket         NUMBER,
  task_seed          NUMBER,
  schema_version     STRING,
  task_family        STRING,            -- e.g. 'linear_regression'
  training_data_family STRING,          -- e.g. 'synthetic_linear_regression_mixed_categorical'
  task_objective     STRING,            -- e.g. 'inductive_regression'
  categorical_cardinalities VARIANT     -- JSON array of per-feature cardinalities
) DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p_num, p_cat, n_train);

-- Numeric-only linear classification training tasks
CREATE TRANSIENT TABLE IF NOT EXISTS META_CLASSIFICATION_DATASET_INDEX (
  split              STRING NOT NULL,
  task_id            STRING NOT NULL,
  stage_path         STRING NOT NULL,   -- numeric/{split}/{task_id}.parquet
  n                  NUMBER,
  p                  NUMBER,
  n_train            NUMBER,
  n_test             NUMBER,
  prior_regime       STRING,
  hpo_bucket         NUMBER,
  num_classes        NUMBER,
  classification_regime STRING,
  task_objective     STRING,
  p_signal           NUMBER,
  p_noise            NUMBER,
  p_total            NUMBER,
  class_imbalance_type STRING,
  label_noise_rate   FLOAT,
  feature_noise_level FLOAT,
  task_seed          NUMBER,            -- SeedSequence-derived seed for training task
  schema_version     STRING             -- 'canonical_v1' after Issue 1 fix; NULL = legacy
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, num_classes, classification_regime, p, n_train);

-- Mixed-categorical linear classification training tasks
-- (stored under @META_CLASSIFICATION_DATASET_STAGE/mixed/)
CREATE TRANSIENT TABLE IF NOT EXISTS META_MIXED_CATEGORICAL_DATASET_INDEX (
  split              STRING  NOT NULL,
  task_id            STRING  NOT NULL,
  stage_path         STRING  NOT NULL,  -- mixed/{split}/{task_id}.parquet
  n                  NUMBER,
  p                  NUMBER,            -- p_num + p_cat (required by RUNTIME_INDEX_COLUMNS)
  p_num              NUMBER,
  p_cat              NUMBER,
  n_train            NUMBER,
  n_test             NUMBER,
  prior_regime       STRING,
  hpo_bucket         NUMBER,
  num_classes        NUMBER,
  classification_regime STRING,
  task_objective     STRING,
  class_imbalance_type STRING,
  label_noise_rate   FLOAT,
  feature_noise_level FLOAT,
  temperature        FLOAT,
  task_seed          NUMBER,
  schema_version     STRING,
  task_family        STRING,            -- e.g. 'linear_classification'
  training_data_family STRING,          -- e.g. 'synthetic_linear_classification_mixed_categorical'
  categorical_cardinalities VARIANT     -- JSON array of per-feature cardinalities
) DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, num_classes, classification_regime, p_num, p_cat, n_train);

-- ============================================================
-- TRAINING INDEX BUILDER PROCEDURES
-- IS_MIXED_CATEGORICAL = FALSE → numeric/ subdir + numeric index table
-- IS_MIXED_CATEGORICAL = TRUE  → mixed/   subdir + mixed   index table
-- ============================================================

-- Regression index builders
CREATE OR REPLACE PROCEDURE build_meta_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_dataset_index_with_flag';

CREATE OR REPLACE PROCEDURE build_meta_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_dataset_index_with_flag_and_total';

-- Classification index builders
CREATE OR REPLACE PROCEDURE build_meta_classification_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_classification_dataset_index_with_flag';

CREATE OR REPLACE PROCEDURE build_meta_classification_dataset_index(IS_MIXED_CATEGORICAL BOOLEAN, EXPECTED_TOTAL INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_classification_dataset_index_with_flag_and_total';

-- ============================================================
-- EVALUATION INDEX TABLES
-- Linear regression: LINEAR_REGRESSION_DATASET_INDEX (numeric)
--                    LINEAR_REGRESSION_MIXED_DATASET_INDEX (mixed-categorical)
-- Linear classification: LINEAR_CLASSIFICATION_DATASET_INDEX (numeric)
--                        LINEAR_CLASSIFICATION_MIXED_DATASET_INDEX (mixed-categorical)
-- ============================================================

CREATE OR REPLACE TRANSIENT TABLE LINEAR_REGRESSION_DATASET_INDEX (
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

CREATE OR REPLACE TRANSIENT TABLE LINEAR_REGRESSION_MIXED_DATASET_INDEX (
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
  categorical_cardinalities VARIANT,    -- JSON array [3, 5, 10, ...]
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
  base_task_id         STRING,
  perturbation_condition STRING,
  task_seed            NUMBER,
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

CREATE OR REPLACE TRANSIENT TABLE LINEAR_CLASSIFICATION_DATASET_INDEX (
  suite_id             STRING,
  suite_family         STRING,
  dataset_id           NUMBER,
  dataset_seed         NUMBER,
  global_idx           NUMBER,
  stage_path           STRING,
  prior_regime         STRING,
  classification_regime STRING,
  split_seeds          ARRAY,
  n_total              NUMBER,
  n_train_default      NUMBER,
  n_holdout_default    NUMBER,
  p_signal             NUMBER,
  p_noise              NUMBER,
  p_total              NUMBER,
  num_classes          NUMBER,
  feature_noise_level  FLOAT,
  training_size_anchor BOOLEAN,
  eval_weight          FLOAT,
  payload_bytes        NUMBER,
  created_at           TIMESTAMP_NTZ,
  logical_dataset_key  STRING,
  source_suite_id      STRING,
  task_family          STRING,          -- "linear_classification"
  task_objective       STRING,          -- "inductive_classification"
  label_noise_rate     FLOAT,
  class_imbalance_type STRING,
  temperature          FLOAT,
  margin_level         STRING,
  sample_complexity_bucket STRING,
  task_seed            NUMBER,          -- SeedSequence-derived seed for this eval task
  schema_version       STRING,          -- v1 and v2 readers are supported
  coeff_schema_version STRING,          -- 'canonical_v1' = W_true always (p,K); NULL = legacy
  distribution_family  STRING,
  covariance_type      STRING,
  rho                  FLOAT,
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
  task_fingerprint     STRING
) DATA_RETENTION_TIME_IN_DAYS = 0;

CREATE OR REPLACE TRANSIENT TABLE LINEAR_CLASSIFICATION_MIXED_DATASET_INDEX (
  suite_id             STRING,
  suite_family         STRING,
  dataset_id           NUMBER,
  dataset_seed         NUMBER,
  global_idx           NUMBER,
  stage_path           STRING,
  prior_regime         STRING,
  classification_regime STRING,
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
  categorical_cardinalities VARIANT,
  max_cardinality      NUMBER,
  cat_effect_scale     FLOAT,
  missing_rate         FLOAT,
  -- Classification columns
  num_classes          NUMBER,
  feature_noise_level  FLOAT,
  training_size_anchor BOOLEAN,
  eval_weight          FLOAT,
  payload_bytes        NUMBER,
  created_at           TIMESTAMP_NTZ,
  logical_dataset_key  STRING,
  source_suite_id      STRING,
  task_family          STRING,
  task_objective       STRING,
  label_noise_rate     FLOAT,
  class_imbalance_type STRING,
  temperature          FLOAT,
  margin_level         STRING,
  sample_complexity_bucket STRING,
  task_seed            NUMBER,
  schema_version       STRING,
  coeff_schema_version STRING,
  distribution_family  STRING,
  covariance_type      STRING,
  rho                  FLOAT,
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
  training_data_family STRING
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- ============================================================
-- TRAINING PROCEDURES
-- ============================================================

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

-- ============================================================
-- RUNTIME VARIABLES — set before CALL or injected by procedure
-- ============================================================
-- REGRESSION LINEAR
--   SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH  → @MODEL_STAGE/checkpoints/best_regression.pt
--   SYNREG_RESULTS_STAGE                  → @EVALUATION_RESULTS_STAGE/linear/regression/numeric/<suite_id>
--   TRAINING_DATA_FAMILY                  → synthetic_linear_regression
--   SYNTHETIC_REGRESSION_SUITE_ID         → suite identifier
--   SYNTHETIC_REGRESSION_FEATURE_SELECTOR → train_f_regression
--   SYNREG_INDEX_TABLE                    → LINEAR_REGRESSION_DATASET_INDEX (numeric)
--                                           or LINEAR_REGRESSION_MIXED_DATASET_INDEX (mixed)
--
-- CLASSIFICATION LINEAR
--   SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH  → @MODEL_STAGE/checkpoints/best_classification.pt
--   SYNCLS_RESULTS_STAGE                  → @EVALUATION_RESULTS_STAGE/linear/classification/numeric/<suite_id>
--   TRAINING_DATA_FAMILY                  → synthetic_linear_classification
--   SYNTHETIC_CLASSIFICATION_SUITE_ID     → suite identifier
--   SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR → train_f_classif
--   SYNCLS_INDEX_TABLE                    → LINEAR_CLASSIFICATION_DATASET_INDEX (numeric)
--                                           or LINEAR_CLASSIFICATION_MIXED_DATASET_INDEX (mixed)
-- ============================================================

-- ============================================================
-- REGRESSION LINEAR PROCEDURES
-- Results: @EVALUATION_RESULTS_STAGE/linear/regression/numeric/<suite_id>/
--          @EVALUATION_RESULTS_STAGE/linear/regression/mixed/<suite_id>/
-- Checkpoint: @MODEL_STAGE/checkpoints/best_regression.pt
-- Feature selector: train_f_regression
-- ============================================================

CREATE OR REPLACE PROCEDURE run_synthetic_regression_linear_prep(
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_all_v1',
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_linear_prep';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_linear_deepset_evaluation(
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_all_v1',
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_linear_deepset_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_linear_baseline_evaluation(
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_all_v1',
  BASELINE_SHARDS               INTEGER DEFAULT 10,
  BASELINE_CONCURRENT_NODES     INTEGER DEFAULT 10,
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_linear_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_linear_autogluon_evaluation(
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                        STRING  DEFAULT 'linear_all_v1',
  AUTOGLUON_CLUSTER_SHARDS        INTEGER DEFAULT 6,
  AUTOGLUON_WORKERS_PER_SHARD     INTEGER DEFAULT 4,
  AUTOGLUON_TASK_CPUS             INTEGER DEFAULT 1,
  AUTOGLUON_CONCURRENT_CLUSTERS   INTEGER DEFAULT 6,
  AUTOGLUON_TIME_LIMIT_SECONDS    INTEGER DEFAULT 300,
  AUTOGLUON_PRESETS               STRING  DEFAULT 'high_quality',
  IS_MIXED_CATEGORICAL            BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_linear_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_linear_aggregation(
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_all_v1',
  EXPECTED_AUTOGLUON_SHARDS     INTEGER DEFAULT 6,
  EXPECTED_BASELINE_SHARDS      INTEGER DEFAULT 10,
  EXPECTED_DEEPSET_SHARDS       INTEGER DEFAULT 10,
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_linear_aggregation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_linear_pipeline(
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_all_v1',
  BASELINE_SHARDS               INTEGER DEFAULT 10,
  BASELINE_CONCURRENT_NODES     INTEGER DEFAULT 10,
  AUTOGLUON_CLUSTER_SHARDS      INTEGER DEFAULT 6,
  AUTOGLUON_WORKERS_PER_SHARD   INTEGER DEFAULT 4,
  AUTOGLUON_TASK_CPUS           INTEGER DEFAULT 1,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER DEFAULT 6,
  AUTOGLUON_TIME_LIMIT_SECONDS  INTEGER DEFAULT 300,
  AUTOGLUON_PRESETS             STRING  DEFAULT 'high_quality',
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_linear_pipeline';

-- ============================================================
-- CLASSIFICATION LINEAR PROCEDURES
-- Results: @EVALUATION_RESULTS_STAGE/linear/classification/numeric/<suite_id>/
--          @EVALUATION_RESULTS_STAGE/linear/classification/mixed/<suite_id>/
-- Checkpoint: @MODEL_STAGE/checkpoints/best_classification.pt
-- Feature selector: train_f_classif
-- ============================================================

CREATE OR REPLACE PROCEDURE run_synthetic_classification_linear_prep(
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_classification_stat_aware',
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py'
  )
  HANDLER = 'run_synthetic_classification_evaluation.run_synthetic_classification_linear_prep';

CREATE OR REPLACE PROCEDURE run_synthetic_classification_linear_deepset_evaluation(
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_classification_stat_aware',
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py'
  )
  HANDLER = 'run_synthetic_classification_evaluation.run_synthetic_classification_linear_deepset_evaluation';

-- NOTE (metric_schema_version):
--   '1.0' = stored DGP teacher probabilities (NOT model inference; deprecated)
--   '2.0' = real MODEL4 forward pass using actual checkpoint logits
-- Results with different metric_schema_version are NOT comparable.
-- Always filter by metric_schema_version in aggregation queries.

CREATE OR REPLACE PROCEDURE run_synthetic_classification_linear_baseline_evaluation(
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_classification_stat_aware',
  BASELINE_SHARDS               INTEGER DEFAULT 10,
  BASELINE_CONCURRENT_NODES     INTEGER DEFAULT 10,
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py'
  )
  HANDLER = 'run_synthetic_classification_evaluation.run_synthetic_classification_linear_baseline_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_classification_linear_autogluon_evaluation(
  BENCHMARK_RUNTIME_ENVIRONMENT   STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT   STRING,
  SUITE_ID                        STRING  DEFAULT 'linear_classification_stat_aware',
  AUTOGLUON_CLUSTER_SHARDS        INTEGER DEFAULT 0,
  AUTOGLUON_WORKERS_PER_SHARD     INTEGER DEFAULT 1,
  AUTOGLUON_TASK_CPUS             INTEGER DEFAULT 1,
  AUTOGLUON_CONCURRENT_CLUSTERS   INTEGER DEFAULT 10,
  AUTOGLUON_TIME_LIMIT_SECONDS    INTEGER DEFAULT 300,
  AUTOGLUON_PRESETS               STRING  DEFAULT 'high_quality',
  IS_MIXED_CATEGORICAL            BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py'
  )
  HANDLER = 'run_synthetic_classification_evaluation.run_synthetic_classification_linear_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_classification_linear_aggregation(
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_classification_stat_aware',
  EXPECTED_AUTOGLUON_SHARDS     INTEGER DEFAULT 0,
  EXPECTED_BASELINE_SHARDS      INTEGER DEFAULT 10,
  EXPECTED_DEEPSET_SHARDS       INTEGER DEFAULT 10,
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py'
  )
  HANDLER = 'run_synthetic_classification_evaluation.run_synthetic_classification_linear_aggregation';

CREATE OR REPLACE PROCEDURE run_synthetic_classification_linear_pipeline(
  PREP_RUNTIME_ENVIRONMENT      STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  SUITE_ID                      STRING  DEFAULT 'linear_classification_stat_aware',
  BASELINE_SHARDS               INTEGER DEFAULT 10,
  BASELINE_CONCURRENT_NODES     INTEGER DEFAULT 10,
  AUTOGLUON_CLUSTER_SHARDS      INTEGER DEFAULT 0,
  AUTOGLUON_WORKERS_PER_SHARD   INTEGER DEFAULT 1,
  AUTOGLUON_TASK_CPUS           INTEGER DEFAULT 1,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER DEFAULT 10,
  AUTOGLUON_TIME_LIMIT_SECONDS  INTEGER DEFAULT 300,
  AUTOGLUON_PRESETS             STRING  DEFAULT 'high_quality',
  IS_MIXED_CATEGORICAL          BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py'
  )
  HANDLER = 'run_synthetic_classification_evaluation.run_synthetic_classification_linear_pipeline';

CREATE OR REPLACE PROCEDURE index_synthetic_classification_eval_suite(
  SUITE_ID      STRING  DEFAULT 'linear_classification_stat_aware',
  FORCE_REBUILD BOOLEAN DEFAULT FALSE
)
  RETURNS STRING LANGUAGE PYTHON RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/prepare_synthetic_classification.py')
  HANDLER = 'prepare_synthetic_classification.index_classification_eval_suite';

-- ============================================================
-- RUNBOOK
-- Run these statements in order for each training + eval cycle.
-- Replace <ENV> with the Container Runtime image tag, e.g. '2.5.0-py311'.
-- ============================================================

-- ── STEP 1: Upload scripts to MODEL_STAGE ────────────────────
-- PUT file://src/*.py     @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- ── STEP 2: Build training index (numeric) ───────────────────
-- CALL build_meta_dataset_index(FALSE, 1000);          -- regression, numeric
-- CALL build_meta_classification_dataset_index(FALSE, 1000);  -- classification, numeric

-- ── STEP 3: (Optional) Build mixed-categorical index ─────────
-- CALL build_meta_dataset_index(TRUE, 1000);           -- regression, mixed
-- CALL build_meta_classification_dataset_index(TRUE, 1000);   -- classification, mixed

-- ── STEP 4: Pretrain backbone ────────────────────────────────
-- CALL run_pretrain_pipeline(
--   'market_exchangeable_icl', 'synthetic_linear_regression',
--   'inductive_forecasting', 64
-- );

-- ── STEP 5: HPO sweep ────────────────────────────────────────
-- CALL run_hpo_pipeline(
--   'market_exchangeable_icl', 'synthetic_linear_regression',
--   'inductive_forecasting', 'linear_model'
-- );

-- ── STEP 6: Final training ───────────────────────────────────
-- CALL run_model_training(
--   'market_exchangeable_icl', 'synthetic_linear_regression',
--   'inductive_forecasting'
-- );

-- ── STEP 7: Regression linear evaluation (numeric) ───────────
-- CALL run_synthetic_regression_linear_pipeline(
--   '<ENV>', '<ENV>', '<ENV>',
--   'linear_all_v1',          -- SUITE_ID
--   10, 10,                   -- BASELINE_SHARDS, BASELINE_CONCURRENT_NODES
--   6,  4,  1,  6,  300,      -- AutoGluon: shards, workers, cpus, clusters, time_limit
--   'high_quality',           -- AUTOGLUON_PRESETS
--   FALSE                     -- IS_MIXED_CATEGORICAL
-- );

-- ── STEP 7b: Regression linear evaluation (mixed) ────────────
-- CALL run_synthetic_regression_linear_pipeline(
--   '<ENV>', '<ENV>', '<ENV>',
--   'linear_mixed_v1',
--   10, 10, 6, 4, 1, 6, 300, 'high_quality',
--   TRUE                      -- IS_MIXED_CATEGORICAL
-- );

-- ── STEP 8: Classification linear evaluation (numeric) ───────
-- CALL run_synthetic_classification_linear_pipeline(
--   '<ENV>', '<ENV>', '<ENV>',
--   'linear_classification_stat_aware',
--   10, 10, 0, 1, 1, 10, 300, 'high_quality',
--   FALSE
-- );

-- ── STEP 8b: Classification linear evaluation (mixed) ────────
-- CALL run_synthetic_classification_linear_pipeline(
--   '<ENV>', '<ENV>', '<ENV>',
--   'linear_classification_mixed_v1',
--   10, 10, 0, 1, 1, 10, 300, 'high_quality',
--   TRUE
-- );

-- ── Utility: suspend all compute pools ───────────────────────
-- ALTER COMPUTE POOL DEEPSET_GPU_POOL    SUSPEND;
-- ALTER COMPUTE POOL DEEPSET_CPU_POOL    SUSPEND;
-- ALTER COMPUTE POOL AUTOGLUON_CPU_POOL  SUSPEND;
-- ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
