-- =============================================================================
-- synthetic_nonlinear_classification_pipeline.sql
-- =============================================================================
-- Snowflake DDL + stored procedures for the nonlinear CLASSIFICATION
-- evaluation suite (suite_id='nonlinear_classification', 11 families).
--
-- Shared infrastructure (stages, compute pools, EAI) is defined in
-- synthetic_nonlinear_pipeline.sql — this file only adds the classification
-- index table and its five evaluation procedures.
--
-- Handler module: run_nonlinear_classification_evaluation
--
-- Stage dependencies (PUT before running procedures):
--   PUT file://scripts/run_nonlinear_classification_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/run_synthetic_classification_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/prepare_nonlinear_classification.py        @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/evaluate_linear_classification.py          @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/prepare_synthetic_classification.py        @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- =============================================================================

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- =============================================================================
-- Index table DDL (38 columns: 23 base + 15 classification-specific)
-- =============================================================================

CREATE TRANSIENT TABLE IF NOT EXISTS NONLINEAR_CLASSIFICATION_DATASET_INDEX (
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
  feature_noise_level        NUMBER,
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

-- =============================================================================
-- Schema migration — idempotent, safe to re-run
-- =============================================================================

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

-- =============================================================================
-- NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX — eval-data index (mixed-cat cls)
-- =============================================================================
-- NEVER DROP this table. The prep handler uses DELETE WHERE suite_id only.

CREATE TRANSIENT TABLE IF NOT EXISTS NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX (
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
  feature_noise_level        NUMBER,
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

-- Schema migration — idempotent, safe to re-run
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
-- Stored procedure IMPORTS block (shared across all 5 procedures)
-- =============================================================================
-- IMPORTS required for every procedure in this file:
--   '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py'
--   '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py'
--   '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py'
--   '@MODEL_STAGE/scripts/evaluate_linear_classification.py'
--   '@MODEL_STAGE/scripts/prepare_synthetic_classification.py'
--   '@MODEL_STAGE/scripts/baseline_models.py'
--   '@MODEL_STAGE/scripts/autogluon_models.py'

-- =============================================================================
-- Phase 1: Prep — index nonlinear classification datasets
-- =============================================================================

CREATE OR REPLACE PROCEDURE run_nonlinear_classification_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_prep';


-- =============================================================================
-- Phase 2: DeepSet GPU evaluation
-- =============================================================================

CREATE OR REPLACE PROCEDURE run_nonlinear_classification_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_deepset_evaluation';


-- =============================================================================
-- Phase 3: Baseline CPU evaluation — three overloads
-- =============================================================================

-- Overload 1: bench_rt only (default shard count)
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_baseline_evaluation';

-- Overload 2: bench_rt + BASELINE_SHARDS
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_baseline_evaluation';

-- Overload 3: bench_rt + BASELINE_SHARDS + BASELINE_CONCURRENT_NODES
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_baseline_evaluation';


-- =============================================================================
-- Phase 4: AutoGluon evaluation — two overloads
-- =============================================================================

-- Overload 1: bench_rt + ag_rt (defaults)
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AG_RUNTIME_ENVIRONMENT    STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_autogluon_evaluation';

-- Overload 2: explicit shard count + tuning params
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AG_RUNTIME_ENVIRONMENT    STRING,
  AUTOGLUON_SHARDS          INTEGER,
  AUTOGLUON_TASK_CPUS       INTEGER,
  AUTOGLUON_TIME_LIMIT      INTEGER,
  AUTOGLUON_PRESETS         STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_autogluon_evaluation';


-- =============================================================================
-- Phase 5: Aggregation — two overloads
-- =============================================================================

-- Overload 1: bench_rt only (default expected shard counts)
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_aggregation';

-- Overload 2: explicit expected shard counts
CREATE OR REPLACE PROCEDURE run_nonlinear_classification_aggregation(
  BENCH_RUNTIME_ENVIRONMENT    STRING,
  EXPECTED_DEEPSET_SHARDS      INTEGER,
  EXPECTED_BASELINE_SHARDS     INTEGER,
  EXPECTED_AG_SHARDS           INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_classification_aggregation';


-- =============================================================================
-- End-to-end pipeline invocation example
-- =============================================================================
--
-- CALL run_nonlinear_classification_prep('2.5.0-py311');
-- CALL run_nonlinear_classification_deepset_evaluation('2.5.0-py311');
-- CALL run_nonlinear_classification_baseline_evaluation('2.5.0-py311');
-- CALL run_nonlinear_classification_autogluon_evaluation('2.5.0-py311', '2.5.0-py311');
-- CALL run_nonlinear_classification_aggregation('2.5.0-py311');
--
-- Or with explicit shards:
-- CALL run_nonlinear_classification_baseline_evaluation('2.5.0-py311', 10, 10);
-- CALL run_nonlinear_classification_autogluon_evaluation('2.5.0-py311', '2.5.0-py311', 10, 1, 300, 'high_quality');
-- CALL run_nonlinear_classification_aggregation('2.5.0-py311', 10, 10, 10);


-- =============================================================================
-- Mixed-categorical nonlinear classification evaluation procedures
-- Suite: suite_id='nonlinear_mixed_classification'
-- Eval table: NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX
-- These procedures inject SYNCLS_IS_MIXED_CATEGORICAL=true and
-- SYNCLS_INDEX_TABLE=NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX.
-- Run the 5 phases in order just like the standard nonlinear classification suite.
-- =============================================================================

-- Phase 1: Prep
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_prep';

-- Phase 2: DeepSet GPU evaluation
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_deepset_evaluation';

-- Phase 3: Baseline CPU evaluation — 1-arg (defaults)
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_baseline_evaluation';

-- Phase 3: Baseline CPU evaluation — 2-arg (explicit shard count)
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS           INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_baseline_evaluation_with_shards';

-- Phase 4: AutoGluon evaluation — 2-arg (bench_rt + ag_rt, default shards)
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AG_RUNTIME_ENVIRONMENT    STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_autogluon_evaluation';

-- Phase 4: AutoGluon evaluation — 6-arg full
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AG_RUNTIME_ENVIRONMENT    STRING,
  AUTOGLUON_SHARDS          INTEGER,
  AUTOGLUON_TASK_CPUS       INTEGER,
  AUTOGLUON_TIME_LIMIT      INTEGER,
  AUTOGLUON_PRESETS         STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_autogluon_evaluation_full';

-- Phase 5: Aggregation — 1-arg (defaults)
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_aggregation';

-- Phase 5: Aggregation — 4-arg (explicit expected shard counts)
CREATE OR REPLACE PROCEDURE run_nonlinear_mixed_classification_aggregation(
  BENCH_RUNTIME_ENVIRONMENT    STRING,
  EXPECTED_DEEPSET_SHARDS      INTEGER,
  EXPECTED_BASELINE_SHARDS     INTEGER,
  EXPECTED_AG_SHARDS           INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_nonlinear_classification_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_classification_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_nonlinear_classification.py',
    '@MODEL_STAGE/scripts/evaluate_linear_classification.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_classification.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_nonlinear_classification_evaluation.run_nonlinear_mixed_classification_aggregation_full';

-- =============================================================================
-- End-to-end mixed-categorical nonlinear classification pipeline
-- =============================================================================
--
-- CALL run_nonlinear_mixed_classification_prep('2.5.0-py311');
-- CALL run_nonlinear_mixed_classification_deepset_evaluation('2.5.0-py311');
-- CALL run_nonlinear_mixed_classification_baseline_evaluation('2.5.0-py311');
-- CALL run_nonlinear_mixed_classification_autogluon_evaluation('2.5.0-py311', '2.5.0-py311');
-- CALL run_nonlinear_mixed_classification_aggregation('2.5.0-py311');
--
-- Or with explicit shards:
-- CALL run_nonlinear_mixed_classification_baseline_evaluation('2.5.0-py311', 10);
-- CALL run_nonlinear_mixed_classification_autogluon_evaluation('2.5.0-py311', '2.5.0-py311', 10, 1, 300, 'high_quality');
-- CALL run_nonlinear_mixed_classification_aggregation('2.5.0-py311', 10, 10, 10);
