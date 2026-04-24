-- run_training_job.sql
-- Run in Snowsight or SnowSQL after the Docker image has been pushed to TABPFN_REPO.
-- Replace <REGISTRY_HOST> with the value from:  SHOW IMAGE REPOSITORIES IN SCHEMA TABPFN_SCHEMA;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- ── Step 1: Ensure output stage exists ────────────────────────────────────────
-- train.py uploads best.pt here via session.file.put("best.pt", "@MODEL_STAGE/checkpoints/")
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ── Step 2: Create compute pool (skip if already exists) ──────────────────────
-- GPU_NV_S = 1× A10G GPU, 24 GB VRAM. Takes 1–3 min to reach ACTIVE on first creation.
CREATE COMPUTE POOL IF NOT EXISTS DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S;

-- Verify the pool is ACTIVE before continuing (re-run until Status = ACTIVE):
-- SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';

-- ── Step 3: Launch training job ───────────────────────────────────────────────
-- EXECUTE JOB SERVICE runs the container to completion then terminates.
-- The container CMD runs:  python train.py && python evaluate.py ...
-- Replace <REGISTRY_HOST> before running.
EXECUTE JOB SERVICE
  IN COMPUTE POOL DEEPSET_GPU_POOL
  NAME = TABPFN_TRAINING_JOB
  FROM SPECIFICATION $$
  spec:
    containers:
      - name: trainer
        image: <REGISTRY_HOST>/tabpfn_db/tabpfn_schema/tabpfn_repo/deepset-trainer:latest
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
