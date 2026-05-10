-- Diagnose a failed CALL run_training_pipeline().
--
-- Use this worksheet after a Snowflake incident or Ray/Prometheus warning. The
-- Ray scrape warning below is usually observability noise unless the MLJob logs
-- also show worker, GCS, object store, Tuner, or PyTorchDistributor failures:
--
--   Error on ingesting samples with different value but same timestamp
--
-- Record the failed CALL query ID, role, database/schema, and timestamp from
-- Snowsight Query History before changing training code.

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- 1. Find recent stored procedure calls. Open the failed CALL in Snowsight Query
-- History and inspect child queries/logs from there.
SELECT
  query_id,
  start_time,
  end_time,
  execution_status,
  error_code,
  error_message,
  role_name,
  database_name,
  schema_name,
  query_text
FROM TABLE(
  INFORMATION_SCHEMA.QUERY_HISTORY(
    END_TIME_RANGE_START => DATEADD('hour', -24, CURRENT_TIMESTAMP()),
    RESULT_LIMIT => 100
  )
)
WHERE query_text ILIKE '%run_training_pipeline%'
ORDER BY start_time DESC;

-- 2. Classify the failed phase from handoff artifacts.
-- If @MODEL_STAGE/hpo/ and @MODEL_STAGE/checkpoints/ are both empty and the HPO
-- job was canceled before best_config.json, training never started. Inspect SPCS
-- job service state and container logs first.
-- If best_config.json exists but best.pt is missing, HPO finished and full
-- training failed.
LIST @MODEL_STAGE/hpo/;
LIST @MODEL_STAGE/checkpoints/;

-- 3. Confirm the exact staged code that the stored procedure and MLJobs run.
-- Before rerun, confirm hpo.py, train.py, model.py, snowflake_io.py, and
-- run_training_job.py are all present and are not only .gz duplicates.
-- hpo.py should be uploaded with max_concurrent_trials=4, and run_training_job.py
-- should submit HPO with target_instances=10.
LIST @MODEL_STAGE/scripts/;

-- 4. Find recent MLJob-backed SPCS job services around the CALL failure timestamp.
-- Inspect failed HPO or Training job logs in Snowsight. Escalate Ray warnings only
-- when paired with actual Ray worker crashes, GCS/object-store errors, trial
-- failures, or PyTorchDistributor failures.
--
SHOW JOB SERVICES IN ACCOUNT;
SHOW JOB SERVICES IN COMPUTE POOL DEEPSET_GPU_POOL;

-- Replace with the exact HPO service name from SHOW JOB SERVICES.
DESC SERVICE TABPFN_DB.TABPFN_SCHEMA.<HPO_SERVICE_NAME>;

SELECT SYSTEM$GET_SERVICE_LOGS(
  'TABPFN_DB.TABPFN_SCHEMA.<HPO_SERVICE_NAME>',
  0,
  'main',
  500
);

-- If service logs are unavailable, record these SHOW JOB SERVICES row fields:
-- name, status, created_on, updated_on, current_instances, target_instances,
-- compute_pool, query_warehouse, managing_object_domain, managing_object_name.

-- 5. Verify GPU pool capacity/readiness. Full training needs the pool to scale to
-- five active GPU_NV_M nodes because train.py uses PyTorchDistributor with
-- TRAIN_NUM_NODES=5.
SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';

-- Branch guide:
-- - HPO canceled before best_config.json: inspect SPCS job service state and
--   container logs first. Training did not start.
-- - Python import/module errors: fix stage upload or procedure imports.
-- - Tuner/Ray trial failure: focus on src/hpo.py, dataset materialization, and
--   GPU resource settings.
-- - Logs empty or current_instances never exceeded 0: focus on compute pool
--   capacity, role privileges, service scheduling, or Snowflake incident/support.
-- - Parent CALL run_training_pipeline() canceled: treat HPO cancellation as
--   propagated from the parent query.
-- - best_config.json exists and best.pt missing: investigate full training,
--   especially 5-node pool readiness and PyTorchDistributor startup logs.
-- - no MLJob was created: investigate stored procedure import/package execution
--   or the Snowflake incident before job submission.
-- - Snowflake canceled internally with no useful logs: open Snowflake Support
--   with the failed CALL query ID, HPO service name, incident number, and exact
--   timestamps.
