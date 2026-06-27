-- =============================================================================
-- nonlinear_classification_mixed_pipeline.sql
-- =============================================================================
-- Fully self-contained Snowflake pipeline for the nonlinear classification
-- mixed-categorical training + evaluation suite.
--
-- TRAINING_DATA_FAMILY : synthetic_nonlinear_classification_mixed_categorical
-- Training index table : META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX
-- Eval index table     : NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX
-- Checkpoint           : @MODEL_STAGE/checkpoints/best_nonlinear_cls.pt
-- Result path          : @EVALUATION_RESULTS_STAGE/nonlinear/classification/mixed/{suite_id}/
-- Feature selector     : train_f_classif
-- Model arch           : model5_lbacnp  (HPO_SWEEP_MODE=lbacnp_model)
-- Handler module       : run_nonlinear_classification_evaluation
--
-- Stage dependencies (PUT before running):
--   PUT file://src/*.py     @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- =============================================================================

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

-- Stage for model checkpoints (best_nonlinear_cls.pt etc.)
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

-- =============================================================================
-- TRAINING INDEX TABLE
-- Mixed-categorical nonlinear classification training tasks
-- Stored under @META_DATASET_STAGE/nonlinear/classification/mixed/
-- =============================================================================

CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX (
  split                STRING  NOT NULL,
  task_id              STRING  NOT NULL,
  stage_path           STRING  NOT NULL,  -- nonlinear/classification/mixed/{split}/{task_id}.parquet
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
  categorical_cardinalities ARRAY
) DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, num_classes, classification_regime, p_num, p_cat, n_train);

-- =============================================================================
-- TRAINING INDEX BUILDER PROCEDURE
-- IS_MIXED_CATEGORICAL = TRUE -> mixed/ subdir + META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX
-- =============================================================================

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

-- =============================================================================
-- NONLINEAR PRETRAIN PROCEDURE
-- 3-arg variant (no fusion gate sweep); produces pretrain_nonlinear_meta.pt
-- =============================================================================

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

-- =============================================================================
-- HPO PROCEDURE
-- =============================================================================

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

-- =============================================================================
-- MODEL TRAINING PROCEDURE
-- =============================================================================

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

-- =============================================================================
-- EXECUTE TRAINING PIPELINE (nonlinear classification mixed-categorical)
-- =============================================================================

CALL build_meta_nonlinear_classification_dataset_index(TRUE, 1000);

-- Nonlinear pretrain (no gate sweep; pins fusion_gate_hidden_dim internally).
-- Produces @MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt
CALL run_pretrain_pipeline_nonlinear(
  'market_exchangeable_icl',
  'synthetic_nonlinear_classification_mixed_categorical',
  'inductive_forecasting'
);

CALL run_hpo_pipeline(
  'market_exchangeable_icl',
  'synthetic_nonlinear_classification_mixed_categorical',
  'inductive_forecasting',
  'lbacnp_model'
);
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_nonlinear_meta[.]pt';
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json;

CALL run_model_training(
  'market_exchangeable_icl',
  'synthetic_nonlinear_classification_mixed_categorical',
  'inductive_forecasting'
);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;

-- =============================================================================
-- EVAL INDEX TABLE
-- NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX
-- NEVER DROP this table. The prep handler uses DELETE WHERE suite_id only.
-- OR REPLACE is intentional here (fresh schema on each pipeline deployment).
-- =============================================================================

CREATE OR REPLACE TRANSIENT TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX (
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
  realized_margin_level      STRING,
  p_num                      NUMBER,
  p_cat                      NUMBER,
  categorical_cardinalities  VARIANT,
  cat_effect_scale           FLOAT,
  max_cardinality            NUMBER,
  missing_rate               FLOAT
) DATA_RETENTION_TIME_IN_DAYS = 0;

-- Schema migration -- idempotent, safe to re-run
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS feature_regime             STRING;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS covariance_type            STRING;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS rho                        FLOAT;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS suite_component            STRING;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS condition_id               STRING;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS teacher_seed               NUMBER;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS sample_seed                NUMBER;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS num_classes                NUMBER;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS label_noise_rate           FLOAT;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS class_imbalance_type       STRING;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS margin_level               STRING;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS temperature                FLOAT;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS realized_num_classes       NUMBER;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS realized_label_noise_rate  FLOAT;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS realized_margin_level      STRING;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS p_num                      NUMBER;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS p_cat                      NUMBER;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS categorical_cardinalities  VARIANT;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS cat_effect_scale           FLOAT;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS max_cardinality            NUMBER;
ALTER TABLE NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX ADD COLUMN IF NOT EXISTS missing_rate               FLOAT;

-- =============================================================================
-- Stored procedure IMPORTS block (shared across all 5 evaluation procedures)
-- =============================================================================
-- IMPORTS required for every procedure in this file:
--   '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py'
--   '@MODEL_STAGE/scripts/run_linear_classification_evaluation.py'
--   '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py'
--   '@MODEL_STAGE/scripts/evaluate_nonlinear_classification.py'
--   '@MODEL_STAGE/scripts/baseline_models.py'
--   '@MODEL_STAGE/scripts/autogluon_models.py'

-- =============================================================================
-- UNIVERSAL NONLINEAR CLASSIFICATION EVALUATION PROCEDURES
-- One proc name per phase; IS_MIXED_CATEGORICAL BOOLEAN (first param, required)
-- routes between numeric (FALSE) and mixed-categorical (TRUE) suites.
-- Handler module:    run_nonlinear_classification_evaluation
-- Checkpoint:        @MODEL_STAGE/checkpoints/best_nonlinear_cls.pt
-- Numeric results:   @EVALUATION_RESULTS_STAGE/nonlinear/classification/numeric/{suite_id}/
-- Mixed results:     @EVALUATION_RESULTS_STAGE/nonlinear/classification/mixed/{suite_id}/
-- Feature selector:  train_f_classif
-- =============================================================================

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

-- =============================================================================
-- Setup SPCS image for AutoGluon
-- =============================================================================
SET AUTOGLUON_IMAGE_REF = 'dvxcfsm-hs34800.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/model_image_repository/tabpfn-autogluon-ray:1.0.0';
SET SPCS_RAY_HEAD_DNS_SUFFIX = 'tabpfn_schema.tabpfn_db.snowflakecomputing.internal';

-- =============================================================================
-- EXECUTE EVALUATION PIPELINE (nonlinear classification mixed-categorical)
-- Choreography: prep -> deepset -> GPU SUSPEND -> baseline -> CPU SUSPEND
--               -> autogluon -> AG_CPU SUSPEND -> CPU RESUME -> aggregation
-- =============================================================================

-- Phase 1: Prepare evaluation data (index NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX)
CALL run_nonlinear_classification_prep(TRUE, '2.5.0-py311');

-- Phase 2: DeepSet GPU evaluation
-- (MODEL5/LBACNP: model_arch_version resolved to 'model5_lbacnp' by run_hpo_job.py
-- and written into best_config.json; run_model_training_job.py reads it from there)
CALL run_nonlinear_classification_deepset_evaluation(TRUE, '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;

-- Phase 3: Baseline CPU evaluation
CALL run_nonlinear_classification_baseline_evaluation(TRUE, '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;

-- Phase 4: AutoGluon evaluation (60 concurrent nodes via AUTOGLUON_CPU_POOL MAX_NODES=60)
CALL run_nonlinear_classification_autogluon_evaluation(TRUE, '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

-- Phase 5: Aggregation (CPU only; resume CPU pool for aggregation I/O)
ALTER COMPUTE POOL DEEPSET_CPU_POOL RESUME;
CALL run_nonlinear_classification_aggregation(TRUE, '2.5.0-py311');

ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL DEEPSET_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SET AUTO_SUSPEND_SECS = 300;

-- =============================================================================
-- END-TO-END PIPELINE INVOCATION EXAMPLES (mixed-categorical suite)
-- Pass IS_MIXED_CATEGORICAL=TRUE to route all universal procs to the mixed suite.
-- =============================================================================
--
-- Standard (default shards):
--   CALL run_nonlinear_classification_prep(TRUE, '2.5.0-py311');
--   CALL run_nonlinear_classification_deepset_evaluation(TRUE, '2.5.0-py311');
--   ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
--   CALL run_nonlinear_classification_baseline_evaluation(TRUE, '2.5.0-py311');
--   ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
--   CALL run_nonlinear_classification_autogluon_evaluation(TRUE, '2.5.0-py311', $AUTOGLUON_IMAGE_REF);
--   ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
--   ALTER COMPUTE POOL DEEPSET_CPU_POOL RESUME;
--   CALL run_nonlinear_classification_aggregation(TRUE, '2.5.0-py311');
--
-- With explicit shards:
--   CALL run_nonlinear_classification_baseline_evaluation(TRUE, '2.5.0-py311', 10);
--   CALL run_nonlinear_classification_autogluon_evaluation(TRUE, '2.5.0-py311', $AUTOGLUON_IMAGE_REF, 10, 1, 300, 'high_quality');
--   CALL run_nonlinear_classification_aggregation(TRUE, '2.5.0-py311', 10, 10, 10);
--
-- Verification:
--   SELECT suite_id, COUNT(*) n FROM NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX GROUP BY suite_id;
--
-- Utility: suspend all compute pools
--   ALTER COMPUTE POOL DEEPSET_GPU_POOL   SUSPEND;
--   ALTER COMPUTE POOL DEEPSET_CPU_POOL   SUSPEND;
--   ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
--   ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
