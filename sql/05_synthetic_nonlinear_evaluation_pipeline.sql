-- =============================================================================
-- 05_synthetic_nonlinear_evaluation_pipeline.sql
-- Nonlinear regression evaluation suite (nonlinear_v1, regimes I–L)
-- =============================================================================
--
-- Prerequisites:
--   1. Stage updated Python scripts:
--      PUT file://scripts/run_synthetic_nonlinear_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/evaluate_synthetic_nonlinear.py       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://src/evaluate_synthetic_regression.py          @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   2. Stage evaluation datasets (generated locally by scripts/generate_nonlinear.py):
--      PUT file://data/nonlinear_regression/I/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/I/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_regression/J/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/J/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_regression/K/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/K/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_regression/L/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/L/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_regression/nonlinear_manifest.json @EVALUATION_DATASET_STAGE/nonlinear/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   3. Execute this file to create the index table and register stored procedures.
--   4. Run evaluation:
--      CALL run_synthetic_nonlinear_prep('2.5.0-py311');
--      CALL run_synthetic_nonlinear_deepset_evaluation('2.5.0-py311');
--      CALL run_synthetic_nonlinear_baseline_evaluation('2.5.0-py311');
--      CALL run_synthetic_nonlinear_autogluon_spcs_evaluation('<AG_IMAGE>', 6, 4, 1, 6, 300, 'best_quality', 600, 1);
--      CALL run_synthetic_nonlinear_aggregation('2.5.0-py311');
--   5. Verify index (expect I/J/K/L each with 100 rows):
--      SELECT prior_regime, COUNT(*) AS n
--      FROM SYNTHETIC_NONLINEAR_DATASET_INDEX
--      WHERE suite_id = 'nonlinear_v1'
--      GROUP BY prior_regime ORDER BY prior_regime;
--   6. Verify outputs:
--      LIST @EVALUATION_RESULTS_STAGE/nonlinear/;

-- ---------------------------------------------------------------------------
-- Index table
-- Schema is identical to SYNTHETIC_REGRESSION_DATASET_INDEX.
-- ---------------------------------------------------------------------------

CREATE TRANSIENT TABLE IF NOT EXISTS SYNTHETIC_NONLINEAR_DATASET_INDEX (
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

-- ---------------------------------------------------------------------------
-- Shared IMPORTS list (all scripts needed by orchestration + evaluation)
-- ---------------------------------------------------------------------------
-- All procedures below use:
--   HANDLER = 'run_synthetic_nonlinear_evaluation.<function_name>'
-- and the same IMPORTS list.

-- ---------------------------------------------------------------------------
-- Phase 1: Prep — index 400 nonlinear datasets
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_prep';

-- ---------------------------------------------------------------------------
-- Phase 2: DeepSet GPU evaluation — 10 shards on DEEPSET_GPU_POOL
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_deepset_evaluation';

-- ---------------------------------------------------------------------------
-- Phase 3: Baseline CPU evaluation — three overloads
-- Overload 1: bench_rt only (uses default shard count from env / SYNREG_CPU_SHARDS)
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_baseline_evaluation_default';

-- Overload 2: bench_rt + explicit BASELINE_SHARDS
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_baseline_evaluation_with_shards';

-- Overload 3: bench_rt + BASELINE_SHARDS + BASELINE_CONCURRENT_NODES
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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_baseline_evaluation_with_shards_and_concurrency';

-- ---------------------------------------------------------------------------
-- Phase 4: AutoGluon SPCS evaluation — two overloads
--
-- 4-arg form: (AG_IMAGE, CLUSTER_SHARDS, WORKERS_PER_SHARD, CONCURRENT_CLUSTERS)
--   CLUSTER_SHARDS=0 → single-node (no Ray)
--   CLUSTER_SHARDS>0 → Ray distributed
-- ---------------------------------------------------------------------------

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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_evaluation_default';

-- 9-arg full form: all parameters explicit
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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_evaluation_full';

-- ---------------------------------------------------------------------------
-- Phase 5: Aggregation — two overloads
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_aggregation_default';

-- Full aggregation with explicit expected shard counts:
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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_nonlinear.py',
    '@MODEL_STAGE/scripts/evaluate_synthetic_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_aggregation_full';
