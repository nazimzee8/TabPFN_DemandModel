-- run_training_job.sql
-- Run in Snowsight or SnowSQL after the Docker image has been pushed to TABPFN_REPO.

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- ── Step 0: Find your registry host ───────────────────────────────────────────
-- Run this query first, then copy the value from the repository_url column.
-- The host is the domain portion, e.g.: abc123.registry.snowflakecomputing.com
-- SHOW IMAGE REPOSITORIES IN SCHEMA TABPFN_SCHEMA;

-- ── Step 1: Ensure output stage exists ────────────────────────────────────────
-- train.py uploads best.pt here via session.file.put("best.pt", "@MODEL_STAGE/checkpoints/")
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ── Step 2: Create compute pool ────────────────────────────────────────────────
-- GPU_NV_M = 1× A10G GPU with 4× host RAM vs GPU_NV_S.
-- Required for 4 DataLoader worker processes; GPU_NV_S causes host-RAM OOM before GPU is used.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY — must drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_M;

-- Verify the pool is ACTIVE before continuing (re-run until Status = ACTIVE):
-- SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';

-- ── Step 3: Launch training job ───────────────────────────────────────────────
-- EXECUTE JOB SERVICE runs the container to completion then terminates.
-- The container CMD runs:  python train.py && python evaluate.py ...
EXECUTE JOB SERVICE
  IN COMPUTE POOL DEEPSET_GPU_POOL
  NAME = TABPFN_TRAINING_JOB
  FROM SPECIFICATION $$
  spec:
    containers:
      - name: trainer
        -- !! REPLACE YOUR_REGISTRY_HOST below with the repository_url host from Step 0 !!
        -- Run: SHOW IMAGE REPOSITORIES IN SCHEMA TABPFN_SCHEMA;
        -- Copy the repository_url value (e.g. abc123.registry.snowflakecomputing.com/...)
        image: YOUR_REGISTRY_HOST/tabpfn_db/tabpfn_schema/tabpfn_repo/deepset-trainer:latest
        volumeMounts:
          - name: data
            mountPath: /data
    volumes:
      - name: data
        source: "@META_DATASET_STAGE"
  $$;

-- ── Step 4: Monitor ───────────────────────────────────────────────────────────
-- Check job status:
-- SHOW JOB SERVICES LIKE 'TABPFN_TRAINING_JOB';

-- Stream container logs (last 100 lines):
-- CALL SYSTEM$GET_SERVICE_LOGS('TABPFN_TRAINING_JOB', '0', 'trainer', 100);

-- ── Step 5: Verify output ─────────────────────────────────────────────────────
-- Confirms best.pt was uploaded on training completion:
-- LIST @MODEL_STAGE/checkpoints/;
-- LIST @MODEL_STAGE/results/;
