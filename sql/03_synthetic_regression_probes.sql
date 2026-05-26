-- ============================================================
-- Main Synthetic Regression Pipeline — Split-Phase Stored Procedures
-- (linear_poisson_v1_recommended, 200 datasets, all methods)
-- ============================================================

CREATE OR REPLACE PROCEDURE run_synthetic_regression_runtime_probes(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_runtime_probes';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_capacity_probe_default';

-- Single-wave capacity probes: BASELINE_CONCURRENT_NODES must equal 6
-- (SYNREG_CPU_SHARDS) and AUTOGLUON_CONCURRENT_NODES must equal 60
-- (SYNREG_AUTOGLUON_SHARDS). Lower values fail fast instead of batching.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_capacity_probe';

-- Baseline capacity probe: BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS
-- (default SYNREG_CPU_SHARDS=6). Lower values are rejected; request quota or
-- increase BASELINE_SHARDS through SYNREG_BASELINE_SHARDS or the BASELINE_SHARDS arg.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_baseline_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_baseline_capacity_probe';

-- Legacy AutoGluon capacity probe: AUTOGLUON_CONCURRENT_NODES must equal
-- SYNREG_AUTOGLUON_SHARDS (60 by default). Lower values are rejected.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_capacity_probe';

-- Combined baseline capacity probe: verify DEEPSET_CPU_POOL can scale to all
-- BASELINE_SHARDS (default SYNREG_CPU_SHARDS=6) in one wave. Single-wave execution
-- is enforced. Lower BASELINE_CONCURRENT_NODES values fail fast. Increase BASELINE_SHARDS
-- through SYNREG_BASELINE_SHARDS env var or the explicit BASELINE_SHARDS SQL arg.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_capacity_probe_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_capacity_probe';

-- Combined baseline capacity probe with explicit shard count: BASELINE_SHARDS sets the
-- number of probes submitted; BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS.
-- Use when intentionally running with a non-default baseline shard count.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_baseline_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  BASELINE_SHARDS INTEGER,
  BASELINE_CONCURRENT_NODES INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_baseline_capacity_probe_with_shards';

-- Combined AutoGluon capacity probe — verify AUTOGLUON_CPU_POOL node availability
-- before evaluation. Mirrors the two execution modes of the evaluation procedure:
--
-- Ray distributed mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--   Submits one ray_capacity_probe.py job per cluster shard, each with
--   target_instances=AUTOGLUON_WORKERS_PER_SHARD. Verifies the pool can satisfy
--   AUTOGLUON_CLUSTER_SHARDS * AUTOGLUON_WORKERS_PER_SHARD nodes simultaneously.
--   AUTOGLUON_CONCURRENT_CLUSTERS must equal AUTOGLUON_CLUSTER_SHARDS.
--   RAY_READY_TIMEOUT_SECONDS defaults to 300 seconds; RAY_READY_POLL_SECONDS
--   defaults to 10 seconds. Use the extended overload when diagnosing cold-start
--   or startup-convergence behavior. These map to MLJob env vars
--   SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS and
--   SYNREG_RAY_CLUSTER_READY_POLL_SECONDS.
--
-- Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS = 0):
--   Submits AUTOGLUON_CONCURRENT_CLUSTERS single-node capacity_probe.py jobs
--   (target_instances=1). No Ray. AUTOGLUON_WORKERS_PER_SHARD must be 1.
-- This probe verifies infrastructure startup/resource visibility only; it does not
-- validate the worker dataset-access descriptor. Run the worker-access probe below
-- after this succeeds and before distributed AutoGluon evaluation.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_capacity_probe_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_capacity_probe';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_capacity_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER,
  RAY_READY_TIMEOUT_SECONDS INTEGER,
  RAY_READY_POLL_SECONDS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_capacity_probe';

-- Combined AutoGluon worker-access probe — verify the worker data-access path
-- after the capacity probe and before full AutoGluon evaluation. Uses the same
-- runtime topology parameters as the capacity probe.
--
-- Ray distributed mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--   Submits one autogluon_worker_access_probe.py job per cluster shard, each with
--   target_instances=AUTOGLUON_WORKERS_PER_SHARD. The Ray driver loads metadata
--   from SYNTHETIC_REGRESSION_DATASET_INDEX, builds compact item dicts with
--   dataset_access.mode='scoped_file_url', and sends only those dicts to workers
--   as Ray task arguments. The driver derives dataset_access.scoped_url with
--   BUILD_SCOPED_FILE_URL. Workers use SnowflakeFile.open(scoped_url) and do not
--   create Snowpark sessions or query SYNTHETIC_REGRESSION_DATASET_INDEX.
--   No AutoGluon training and no full-suite dataset fan-out.
--
-- Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS = 0):
--   Submits AUTOGLUON_CONCURRENT_CLUSTERS single-node access probes
--   (target_instances=1). No Ray. AUTOGLUON_WORKERS_PER_SHARD must be 1.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_worker_access_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_worker_access_probe_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_worker_access_probe(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_CLUSTER_SHARDS INTEGER,
  AUTOGLUON_WORKERS_PER_SHARD INTEGER,
  AUTOGLUON_CONCURRENT_CLUSTERS INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_worker_access_probe';

-- AutoGluon import timing probe — measures dependency bootstrap latency.
--
-- Stage the probe script before calling this procedure:
--   PUT file:///path/to/scripts/autogluon_import_timing_probe.py
--     @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
-- With-pip mode (WITH_PIP=TRUE, default):
--   pip_requirements=autogluon.tabular==1.3.0 is installed at container startup.
--   Time from MLJob submission to the python_entrypoint_started JSON log line
--   approximates Snowflake scheduling + image startup + AutoGluon pip bootstrap.
--   autogluon_import_complete.import_seconds measures pure import overhead after
--   the environment is ready.
--
-- No-pip baseline mode (WITH_PIP=FALSE):
--   No pip install at startup and no AutoGluon/Ray import is attempted. The probe
--   should succeed even when AutoGluon is not installed, giving a clean scheduling
--   + image startup baseline.
--   Compare pip vs no-pip probe waves to estimate bootstrap overhead under concurrency.
--
-- PROBE_COUNT: number of independent single-instance MLJobs to submit concurrently.
--   Use PROBE_COUNT=8 to simulate the concurrency of a full AutoGluon evaluation wave.
CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_import_timing_probe(
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_import_timing_probe_default';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_import_timing_probe(
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  WITH_PIP BOOLEAN,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_import_timing_probe';

-- ===========================================================================
-- AutoGluon execution mode examples
-- ===========================================================================
-- Required operational sequence:
--   1. run_synthetic_regression_runtime_probes
--   2. run_synthetic_regression_combined_autogluon_capacity_probe
--   3. run_synthetic_regression_combined_autogluon_worker_access_probe
--   4. run_synthetic_regression_combined_autogluon_evaluation
--   5. run_synthetic_regression_combined_aggregation

-- A. Ray distributed cluster-shard mode (AUTOGLUON_CLUSTER_SHARDS > 0):
--    6 clusters x 4 workers = 24 containers; 6 AutoGluon shard files.
--    Use when Ray memory is sufficient and dynamic work-item scheduling is desired.
--
-- Capacity probe (verifies 24 CPU_X64_M nodes across 6 Ray clusters):
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   6, 4, 6,
--   300, 10  -- RAY_READY_TIMEOUT_SECONDS, RAY_READY_POLL_SECONDS
-- );
-- Worker-access probe (verifies compact item dict transfer and scoped_file_url access):
-- CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   6, 4, 6
-- );
-- Evaluation (6 cluster shards x 4 workers each; aggregation expects N=6):
-- CALL run_synthetic_regression_combined_autogluon_evaluation(
--   '2.5.0-py311', '2.5.0-py311',
--   6, 4, 1, 6, 300, 'best_quality',
--   600, 10  -- RAY_READY_TIMEOUT_SECONDS, RAY_READY_POLL_SECONDS
-- );

-- B. Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS = 0):
--    30 containers; 30 AutoGluon shard files. No Ray.
--    Use when memory is constrained or Ray/object-store issues are unreliable.
--    AUTOGLUON_WORKERS_PER_SHARD must be 1.
--    AUTOGLUON_CONCURRENT_CLUSTERS is the concurrent single-node shard count.
--
-- Capacity probe (verifies 30 single-node CPU_X64_M containers):
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   0, 1, 30
-- );
-- Worker-access probe (verifies single-node metadata/dataset access path):
-- CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
--   '2.5.0-py311', '2.5.0-py311',
--   0, 1, 30
-- );
-- Evaluation (30 single-node shards; aggregation expects N=30):
-- CALL run_synthetic_regression_combined_autogluon_evaluation(
--   '2.5.0-py311', '2.5.0-py311',
--   0, 1, 1, 30, 300, 'best_quality'
-- );

-- ===========================================================================
-- Combined split-phase execution — Ray distributed AutoGluon (default):
-- ===========================================================================
-- Step 0: Verify node quota before committing to the evaluation runs.
-- CALL run_synthetic_regression_combined_baseline_capacity_probe('2.5.0-py311', '2.5.0-py311', 6);
-- ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- CALL run_synthetic_regression_combined_autogluon_capacity_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
-- CALL run_synthetic_regression_combined_autogluon_worker_access_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
-- ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
