-- =============================================================================
-- 05_synthetic_nonlinear_evaluation_pipeline.sql
-- nonlinear evaluation suite (suite_id='nonlinear', 6 families, 420 datasets)
-- =============================================================================
--
-- Prerequisites:
--   1. Stage updated Python scripts (PUT before executing this file):
--      PUT file://src/generate_nonlinear_dgp.py             @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/evaluate_synthetic_nonlinear.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/run_synthetic_nonlinear_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/run_synthetic_regression_evaluation.py   @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://src/evaluate_synthetic_regression.py             @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/prepare_synthetic_regression.py          @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://src/baseline_models.py                           @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://src/autogluon_models.py                          @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
--   2. Stage evaluation datasets (generated locally by scripts/generate_nonlinear_v2.py):
--      PUT file://data/nonlinear_v2/poly_quad/*.parquet     @EVALUATION_DATASET_STAGE/nonlinear_v2/poly_quad/     AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_v2/sin_low/*.parquet       @EVALUATION_DATASET_STAGE/nonlinear_v2/sin_low/       AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_v2/hinge/*.parquet         @EVALUATION_DATASET_STAGE/nonlinear_v2/hinge/         AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_v2/sparse_interact/*.parquet @EVALUATION_DATASET_STAGE/nonlinear_v2/sparse_interact/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_v2/mixed_linear/*.parquet  @EVALUATION_DATASET_STAGE/nonlinear_v2/mixed_linear/  AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_v2/demand_mono/*.parquet   @EVALUATION_DATASET_STAGE/nonlinear_v2/demand_mono/   AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://data/nonlinear_v2/nonlinear_v2_manifest.json @EVALUATION_DATASET_STAGE/nonlinear_v2/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
--   3. Execute Section A (schema migration) — read-only discovery first:
--      SELECT column_name, data_type FROM information_schema.columns
--      WHERE table_name = 'NONLINEAR_REGRESSION_DATASET_INDEX' ORDER BY ordinal_position;
--
--   4. Execute Section B (stored procedures).
--
--   5. Run SPCS preflight probes:
--      CALL run_synthetic_nonlinear_autogluon_spcs_import_probe('<AG_IMAGE>', 1);
--      CALL run_synthetic_nonlinear_autogluon_spcs_session_probe('<AG_IMAGE>', 1);
--      CALL run_synthetic_nonlinear_autogluon_spcs_capacity_probe('<AG_IMAGE>', 6, 4, 6);
--      CALL run_synthetic_nonlinear_autogluon_spcs_worker_access_probe('<AG_IMAGE>', 6, 4, 6);
--
--   6. Run evaluation phases:
--      CALL run_synthetic_nonlinear_prep('2.5.0-py311');
--      CALL run_synthetic_nonlinear_deepset_evaluation('2.5.0-py311');
--      CALL run_synthetic_nonlinear_baseline_evaluation('2.5.0-py311');
--      CALL run_synthetic_nonlinear_autogluon_spcs_evaluation('<AG_IMAGE>', 6, 4, 6, 300, 'best_quality', 1, 600, 1);
--      CALL run_synthetic_nonlinear_aggregation('2.5.0-py311');
--
--   7. Verify index (expect ~70 rows per family):
--      SELECT suite_id, COUNT(*) n FROM NONLINEAR_REGRESSION_DATASET_INDEX GROUP BY suite_id;
--      -- Expect: nonlinear_v1 → 400, nonlinear → 420
--
--      SELECT prior_regime, COUNT(*) n FROM NONLINEAR_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'nonlinear' GROUP BY prior_regime ORDER BY prior_regime;
--      -- Expect: 6 rows, ~70 each
--
--      SELECT feature_regime, COUNT(*) n FROM NONLINEAR_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'nonlinear' GROUP BY feature_regime ORDER BY feature_regime;
--      -- Expect: 7 rows, ~60 each
--
--      SELECT logical_dataset_key, COUNT(*) n FROM NONLINEAR_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'nonlinear' GROUP BY logical_dataset_key HAVING n > 1;
--      -- Expect: 0 rows (no duplicates)
--
--   8. Verify outputs:
--      LIST @EVALUATION_RESULTS_STAGE/nonlinear/regression/numeric/nonlinear/;
--
-- =============================================================================


-- =============================================================================
-- Section A — Schema Migration
-- =============================================================================
-- Run ONCE before Phase 1.
-- All statements use ADD COLUMN IF NOT EXISTS — safe to re-run multiple times.
-- New columns are nullable — v1 rows (suite_id='nonlinear_v1') will have NULL
-- in all 13 columns. Existing queries that SELECT specific columns are unaffected.
--
-- NOTE: feature_noise_level column type remains NUMBER for v1 compatibility.
-- v2 rows will insert float literals (e.g., 0.25) instead of integer literals.
-- To fix the column type (destructive, manual, requires table recreation):
--   -- CREATE TRANSIENT TABLE NONLINEAR_REGRESSION_DATASET_INDEX_NEW AS
--   --   SELECT *, CAST(feature_noise_level AS FLOAT) AS feature_noise_level_float
--   --   FROM NONLINEAR_REGRESSION_DATASET_INDEX;
--   -- (Then swap tables manually after verifying)

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


-- =============================================================================
-- Section B — Shared IMPORTS
-- =============================================================================
-- All procedures below use the same IMPORTS list.
-- Handler module: run_synthetic_nonlinear_evaluation

-- =============================================================================
-- Phase 1: Prep — index 420 nonlinear_v2 datasets
-- =============================================================================

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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_prep';


-- =============================================================================
-- Phase 2: DeepSet GPU evaluation — 10 shards on DEEPSET_GPU_POOL
-- =============================================================================

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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_deepset_evaluation';


-- =============================================================================
-- Phase 3: Baseline CPU evaluation — three overloads
-- =============================================================================

-- Overload 1: bench_rt only (uses default shard count from env / SYNREG_CPU_SHARDS)
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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_baseline_evaluation_with_shards_and_concurrency';


-- =============================================================================
-- Phase 4: AutoGluon SPCS evaluation — two overloads
--
-- 4-arg form: (AG_IMAGE, CLUSTER_SHARDS, WORKERS_PER_SHARD, CONCURRENT_CLUSTERS)
--   CLUSTER_SHARDS=0 → single-node (no Ray)
--   CLUSTER_SHARDS>0 → Ray distributed
--
-- 9-arg form: all parameters explicit
-- =============================================================================

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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_evaluation_default';

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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_evaluation_full';


-- =============================================================================
-- Phase 5: Aggregation — two overloads
-- =============================================================================

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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_aggregation_default';

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
    '@MODEL_STAGE/scripts/generate_nonlinear_dgp.py',
    '@MODEL_STAGE/scripts/baseline_models.py',
    '@MODEL_STAGE/scripts/autogluon_models.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_aggregation_full';


-- =============================================================================
-- SPCS Preflight Probes
-- =============================================================================
-- Recommended preflight order:
--   1. CALL run_synthetic_nonlinear_autogluon_spcs_import_probe('<AG_IMAGE>', 1);
--   2. CALL run_synthetic_nonlinear_autogluon_spcs_session_probe('<AG_IMAGE>', 1);
--   3. CALL run_synthetic_nonlinear_autogluon_spcs_capacity_probe('<AG_IMAGE>', 6, 4, 6);
--   4. CALL run_synthetic_nonlinear_autogluon_spcs_worker_access_probe('<AG_IMAGE>', 6, 4, 6);
-- Only proceed to the full evaluation after all four probes succeed.

-- Import probe — measures container startup latency with preinstalled deps.
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_import_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_import_probe';

-- Session probe — validates OAuth token injection and Snowpark session creation.
CREATE OR REPLACE PROCEDURE run_synthetic_nonlinear_autogluon_spcs_session_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_session_probe';

-- Capacity probe — verifies containers start on AUTOGLUON_CPU_POOL.
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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_capacity_probe';

-- Capacity probe — extended overload with Ray timeout, stagger, and keep-on-failure.
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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_capacity_probe';

-- Worker-access probe — validates dataset access via presigned URLs. No AG training.
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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_worker_access_probe';

-- Worker-access probe — extended overload with stagger and keep-on-failure.
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
    '@MODEL_STAGE/scripts/run_synthetic_nonlinear_evaluation.py',
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py'
  )
  HANDLER = 'run_synthetic_nonlinear_evaluation.run_synthetic_nonlinear_autogluon_spcs_worker_access_probe';
