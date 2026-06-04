-- ============================================================
-- Combined Suite Evaluation (linear_all_v1, 400 datasets)
-- ============================================================
-- Runbook:
--   Prerequisites (both source suites must already be indexed):
--     CALL run_synthetic_regression_prep('<bench_rt>', '<bench_rt>', '<ag_rt>');
--     CALL run_synthetic_regression_ood_full_evaluation('<bench_rt>', '<ag_rt>');
--   1. Stage updated Python scripts:
--      PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--      PUT file://scripts/prepare_synthetic_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   2. Call procedure:
--      CALL run_synthetic_regression_combined_evaluation('2.5.0-py311', '2.5.0-py311');
--   3. Verify index (expect A/B/C/D/E/F/G/H each with 50 rows):
--      SELECT prior_regime, COUNT(*) AS n
--      FROM SYNTHETIC_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'linear_all_v1'
--      GROUP BY prior_regime ORDER BY prior_regime;
--   4. Verify source lineage:
--      SELECT source_suite_id, COUNT(*) AS n
--      FROM SYNTHETIC_REGRESSION_DATASET_INDEX
--      WHERE suite_id = 'linear_all_v1'
--      GROUP BY source_suite_id ORDER BY source_suite_id;
--      -- Expected: linear_poisson_v1_recommended=200, ood_linear_full_v1=200
--   5. Verify outputs:
--      LIST @EVALUATION_RESULTS_STAGE/combined/;
--   6. Migration note — if SYNTHETIC_REGRESSION_DATASET_INDEX exists without source_suite_id:
--      ALTER TABLE SYNTHETIC_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS source_suite_id STRING;
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation_legacy_concurrency';

-- Combined Suite — Split-Phase Procedures
-- Drop older non-AutoGluon phase overloads that carried AUTOGLUON_RUNTIME_ENVIRONMENT
-- only for signature symmetry. DeepSet, baseline, and aggregation use BENCH_RUNTIME_ENVIRONMENT.
DROP PROCEDURE IF EXISTS run_synthetic_regression_combined_deepset_evaluation(STRING, STRING);
DROP PROCEDURE IF EXISTS run_synthetic_regression_combined_baseline_evaluation(STRING, STRING);
DROP PROCEDURE IF EXISTS run_synthetic_regression_combined_baseline_evaluation(STRING, STRING, INTEGER);
DROP PROCEDURE IF EXISTS run_synthetic_regression_combined_baseline_evaluation(STRING, STRING, INTEGER, INTEGER);
DROP PROCEDURE IF EXISTS run_synthetic_regression_combined_aggregation(STRING, STRING);
DROP PROCEDURE IF EXISTS run_synthetic_regression_combined_aggregation(STRING, STRING, INTEGER);
DROP PROCEDURE IF EXISTS run_synthetic_regression_combined_aggregation(STRING, STRING, INTEGER, INTEGER, INTEGER);

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_prep';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_deepset_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_evaluation_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_evaluation';

-- Combined baseline evaluation with explicit shard count: BASELINE_SHARDS controls the
-- number of shard files written; BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS.
-- 1 baseline shard = 1 single-node MLJob = 1 output shard file.
-- Increasing BASELINE_SHARDS increases required concurrent CPU nodes and output file count.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_evaluation_with_shards';

-- Combined AutoGluon evaluation — two-argument form uses all defaults from env:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_evaluation_default';

-- Combined AutoGluon evaluation — full dynamic form.
-- Ray distributed cluster-shard mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--   AUTOGLUON_CLUSTER_SHARDS:      number of Ray cluster shards / output shard files (e.g. 6)
--   AUTOGLUON_WORKERS_PER_SHARD:   target_instances per MLJob cluster (e.g. 4)
--   AUTOGLUON_TASK_CPUS:           CPUs per individual AutoGluon fit  (default 1)
--   AUTOGLUON_CONCURRENT_CLUSTERS: must equal AUTOGLUON_CLUSTER_SHARDS; lower values fail fast
--   AUTOGLUON_TIME_LIMIT_SECONDS:  per-fit time limit in seconds      (default 300)
--   AUTOGLUON_PRESETS:             AutoGluon presets string           (default best_quality)
--   RAY_READY_TIMEOUT_SECONDS:     optional Ray readiness timeout     (default 600)
--   RAY_READY_POLL_SECONDS:        optional Ray readiness poll period  (default 10)
--   These map to MLJob env vars SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS and
--   SYNREG_RAY_CLUSTER_READY_POLL_SECONDS.
--   entrypoint:                    always autogluon_ray.py (derived internally)
--   total containers = AUTOGLUON_CLUSTER_SHARDS x AUTOGLUON_WORKERS_PER_SHARD
--   aggregation expects N = AUTOGLUON_CLUSTER_SHARDS output shard files
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_evaluation';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING,
  RAY_READY_TIMEOUT_SECONDS INTEGER,
  RAY_READY_POLL_SECONDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_evaluation';

-- Combined aggregation — two-argument form defaults to SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT=6:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation_default';

-- Combined aggregation — three-argument form with explicit expected AutoGluon shard count:
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  EXPECTED_AUTOGLUON_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation_ag';

-- Combined aggregation - full explicit form matching the Python implementation
-- signature after session:
--   (bench_rt, expected_ag_shards, expected_baseline_shards, expected_deepset_shards)
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  EXPECTED_AUTOGLUON_SHARDS INTEGER,
  EXPECTED_BASELINE_SHARDS INTEGER,
  EXPECTED_DEEPSET_SHARDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_aggregation';

-- Combined all-in-one evaluation with explicit baseline shard count.
-- BASELINE_SHARDS: number of baseline shard files to write (default 6); must equal BASELINE_CONCURRENT_NODES.
-- BASELINE_CONCURRENT_NODES: required single-wave CPU nodes; must equal BASELINE_SHARDS.
-- Aggregation automatically uses the resolved AutoGluon execution plan output_shards.
-- 1 baseline shard = 1 single-node MLJob = 1 output shard file.
-- AutoGluon mode is selected by AUTOGLUON_CLUSTER_SHARDS (see evaluation procedure above).
-- Entrypoints are derived internally from the mode and are not accepted as arguments.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT_SECONDS INTEGER,
  AUTOGLUON_PRESETS STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_evaluation_with_baseline_shards';

-- ===========================================================================
-- AutoGluon SPCS Custom Image Backend (one-time setup)
-- ===========================================================================
-- Run this section once before using the SPCS AutoGluon procedures.
-- Skip if using the default MLJob backend (SYNREG_AUTOGLUON_EXECUTION_BACKEND=mljob).
--
-- Step A: Create the Snowflake image repository.
CREATE IMAGE REPOSITORY IF NOT EXISTS AUTOGLUON_IMAGE_REPOSITORY;

-- Verify creation and retrieve the repository URL (needed for docker tag/push below):
SHOW IMAGE REPOSITORIES;
-- Copy the repository_url value from the output. It will look like:
--   <account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY

-- Step B: Build and push the custom AutoGluon image (run these in your terminal,
--         substituting <repository_url> with the value from SHOW IMAGE REPOSITORIES above):
--
--   docker build --platform linux/amd64 \
--     -f docker/autogluon/Dockerfile \
--     -t tabpfn-autogluon-ray:1.0.0 .
--
--   docker login <account>.registry.snowflakecomputing.com
--
--   docker tag tabpfn-autogluon-ray:1.0.0 \
--     <repository_url>/tabpfn-autogluon-ray:1.0.0
--
--   docker push <repository_url>/tabpfn-autogluon-ray:1.0.0

-- Verify the image is visible in Snowflake before proceeding:
SHOW IMAGES IN IMAGE REPOSITORY AUTOGLUON_IMAGE_REPOSITORY;

-- Step C: Stage the SPCS-specific scripts alongside the existing scripts.
--   PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/autogluon_ray.py                       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/spcs_ray_head.py                       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/spcs_ray_worker.py                     @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/spcs_snowpark_session_probe.py         @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--   PUT file://scripts/autogluon_import_timing_probe.py       @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- Step D: Create the SPCS stored procedures.
--
-- SPCS import probe — measures container startup latency with preinstalled deps (no pip).
-- AUTOGLUON_SPCS_IMAGE is the full OCI image reference in the Snowflake image repository.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_spcs_import_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_spcs_import_probe';

-- SPCS Snowpark session probe: validates OAuth token injection (snowflakeService.enabled=true)
-- and Snowpark session creation inside SPCS containers.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_spcs_session_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_spcs_session_probe';

-- SPCS capacity probe — verifies custom-image containers start on AUTOGLUON_CPU_POOL
-- and that Ray is importable. No AutoGluon training.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_capacity_probe';

-- 7-argument overload — adds explicit Ray readiness timeout, worker stagger, and keep-on-failure flag.
-- RAY_READY_TIMEOUT_SECONDS: how long each coordinator waits for all Ray workers to join (default 900 s).
-- WORKER_SUBMIT_STAGGER_SECONDS: sleep between worker job submissions (1 s recommended to reduce burst pressure).
-- KEEP_SUPPORT_JOBS_ON_FAILURE: when TRUE, worker support jobs are not cancelled on failure; inspect logs then cancel manually.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
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
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_capacity_probe';

-- SPCS worker-access probe — driver queries index, builds compact item dicts,
-- workers validate dataset access via presigned URLs. No AutoGluon training.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_worker_access_probe(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_worker_access_probe';

-- SPCS AutoGluon evaluation — no runtime_environment, no pip_requirements.
-- All dependencies are preinstalled in the custom OCI image.
-- Single-node mode: AUTOGLUON_CLUSTER_SHARDS = 0, one SPCS job service per shard.
-- Ray distributed mode: AUTOGLUON_CLUSTER_SHARDS > 0, self-managed Ray head+workers+driver.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_evaluation(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT INTEGER,
  AUTOGLUON_PRESETS STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_evaluation';

-- 9-argument overload — adds explicit Ray readiness timeout and worker submission stagger.
-- RAY_READY_TIMEOUT_SECONDS: how long each coordinator waits for all Ray workers to join.
-- WORKER_SUBMIT_STAGGER_SECONDS: sleep between worker job submissions (1s resolved the startup issue).
-- KEEP_SUPPORT_JOBS_ON_FAILURE is intentionally not exposed here; use SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE env var for diagnostics.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_evaluation(
  AUTOGLUON_SPCS_IMAGE STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_TASK_CPUS INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  AUTOGLUON_TIME_LIMIT INTEGER,
  AUTOGLUON_PRESETS STRING,
  RAY_READY_TIMEOUT_SECONDS INTEGER,
  WORKER_SUBMIT_STAGGER_SECONDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_evaluation';

-- ===========================================================================
-- SPCS runbook — hardened 7-step preflight + evaluation sequence.
--
-- Prerequisites:
--   1. Build and push the custom image:
--      docker build --platform linux/amd64 -f docker/autogluon/Dockerfile -t tabpfn-autogluon-ray:1.0.0 .
--      docker tag tabpfn-autogluon-ray:1.0.0 <account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY/tabpfn-autogluon-ray:1.0.0
--      docker push <account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY/tabpfn-autogluon-ray:1.0.0
--   2. Verify the push in Snowflake:
--      SHOW IMAGES IN IMAGE REPOSITORY AUTOGLUON_IMAGE_REPOSITORY;
--   3. Use the full pushed image reference as the first argument to every SPCS call:
--      <image_ref> = '<account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY/tabpfn-autogluon-ray:1.0.0'
--      Optional legacy fallback for old calls that still pass 'spcs_job':
--      ALTER SESSION SET SYNREG_AUTOGLUON_SPCS_IMAGE = '<image_ref>';
--      For multi-shard Ray mode, set per-shard DNS suffix (schema/db in lowercase):
ALTER SESSION SET SPCS_RAY_HEAD_DNS_SUFFIX = 'tabpfn_schema.tabpfn_db.snowflakecomputing.internal';
--
-- Step 1: Import probe — verify image starts and AutoGluon/Ray/Snowpark imports succeed.
CALL run_synthetic_regression_autogluon_spcs_import_probe('<image_ref>', 1);
--
-- Step 2: Session probe — verify Snowflake OAuth token injection (snowflakeService.enabled=true)
--         and that Snowpark session creation works inside SPCS containers.
--         Required for drivers to query SYNTHETIC_REGRESSION_DATASET_INDEX and GET_PRESIGNED_URL().
CALL run_synthetic_regression_autogluon_spcs_session_probe('<image_ref>', 1);
--
-- Step 3: Capacity probe — single-node mode (CLUSTER_SHARDS=0), 6 concurrent probes.
--         Then Ray mode (CLUSTER_SHARDS=1, 2 workers) to validate self-managed Ray topology.
CALL run_synthetic_regression_combined_autogluon_spcs_capacity_probe('<image_ref>', 0, 1, 6);
CALL run_synthetic_regression_combined_autogluon_spcs_capacity_probe('<image_ref>', 1, 2, 1);
--
-- Step 4: Worker-access probe — validate presigned-URL dataset loading via Ray workers.
CALL run_synthetic_regression_combined_autogluon_spcs_worker_access_probe('<image_ref>', 0, 1, 6);
--
-- Step 5: One-shard mini evaluation — verify end-to-end with 1 shard in single-node mode.
CALL run_synthetic_regression_combined_autogluon_spcs_evaluation(
  '<image_ref>', 0, 1, 1, 1, 300, 'best_quality'
);
--
-- Step 6a: Full evaluation — single-node mode (no Ray, one job per shard, 6 shards):
CALL run_synthetic_regression_combined_autogluon_spcs_evaluation(
  '<image_ref>', 0, 1, 1, 6, 300, 'best_quality'
);
--
-- Step 6b: Full evaluation — Ray distributed mode (6 clusters x 4 workers, self-managed Ray).
--   Each shard's Ray head gets a unique address derived from SPCS_RAY_HEAD_DNS_SUFFIX (set above).
--   Head starts with --num-cpus=0; driver expects workers_per_shard+1 live nodes.
--   Each driver verifies cluster identity via custom Ray resource before submitting work.
--   Coordinator topology: 6 coordinators + 24 workers = 30 containers.
--   Each coordinator merges Ray head (--num-cpus=0) + AutoGluon driver in one container.
--   Only the 24 worker containers are schedulable AutoGluon workers; coordinators are downsized.
--   Default resources: coordinator 1/2 CPU, 4Gi/8Gi memory; worker 1 CPU, 8Gi/16Gi memory.
--   Override with SYNREG_SPCS_RAY_COORDINATOR_*, SYNREG_SPCS_RAY_WORKER_*,
--   or SYNREG_SPCS_SINGLE_NODE_*.
CALL run_synthetic_regression_combined_autogluon_spcs_evaluation(
  '<image_ref>', 6, 4, 1, 6, 300, 'best_quality'
);
--
-- Step 6c: Full evaluation with explicit Ray readiness timeout and worker stagger (recommended).
--   RAY_READY_TIMEOUT_SECONDS=600: coordinators wait up to 10 minutes for all workers to join.
--   WORKER_SUBMIT_STAGGER_SECONDS=1: 1s delay between worker submissions avoids bursty SPCS scheduling.
--   KEEP_SUPPORT_JOBS_ON_FAILURE is intentionally not exposed; set env var for diagnostics only.
CALL run_synthetic_regression_combined_autogluon_spcs_evaluation(
  '<image_ref>', 6, 4, 1, 6, 300, 'best_quality', 600, 1
);
--
-- Step 7: Aggregation (N=6 must match cluster_shards used in Step 6a or 6b):
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', 6);

-- ===========================================================================
-- Combined split-phase execution — Ray distributed AutoGluon (default):
-- ===========================================================================
-- Step 1: Run combined split phases.
CALL run_synthetic_regression_combined_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_combined_deepset_evaluation('2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_baseline_evaluation('2.5.0-py311', 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- Ray mode: 6 clusters x 4 workers = 24 concurrent CPU_X64_M nodes, 1 CPU per fit task
CALL run_synthetic_regression_combined_autogluon_evaluation(
  '2.5.0-py311', '2.5.0-py311', 6, 4, 1, 6, 300, 'best_quality'
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
-- Aggregation expects 6 AutoGluon shard files (N=6 matches cluster_shards above)
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', 6);
