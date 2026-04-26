-- run_training_job.sql
-- One-time environment setup + compute pool creation for the DeepSet training pipeline.
-- Run steps 0–2 and step 4 in Snowsight or SnowSQL.
-- Steps 3 and 3b (PUT) must be run in SnowSQL — PUT is not supported in the Snowsight web UI.
-- The Docker image is NOT required — training uses the Snowflake Container Runtime for ML.

-- ── Step 0: Create database and schema ────────────────────────────────────────
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
USE DATABASE TABPFN_DB;
CREATE SCHEMA IF NOT EXISTS TABPFN_SCHEMA;
USE SCHEMA TABPFN_SCHEMA;

-- ── Step 1: Create stages ──────────────────────────────────────────────────────
-- META_DATASET_STAGE: training data uploaded via PUT (step 3 below).
-- The Container Runtime mounts this stage at /data/ inside the training container.
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- MODEL_STAGE: train.py writes best.pt here; evaluate.py writes test_report.csv here.
--              Python scripts are staged here for the orchestrator container.
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ── Step 2: Create compute pool ────────────────────────────────────────────────
-- GPU_NV_S: 1× A10G per node, 0.57 cr/hr → ~$1.14–1.71/node/hr (Standard/Enterprise).
-- MAX_NODES=4: up to 4 nodes for DDP training; 2 nodes for parallel HPO trials.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY — must drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 4          -- was 2; training now uses 4 nodes
  INSTANCE_FAMILY = GPU_NV_S;

-- Verify the pool reaches ACTIVE state before submitting the job (re-run until Status = ACTIVE):
SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';

-- ── Step 3: Upload training data (SnowSQL only) ─────────────────────────────────
-- PUT file://C:/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE;
--
-- Verify:
-- LIST @META_DATASET_STAGE/train/;
-- LIST @META_DATASET_STAGE/val/;
-- LIST @META_DATASET_STAGE/test/;

-- ── Step 3b: Upload Python scripts (SnowSQL only) ───────────────────────────────
-- Re-run whenever any script changes:
-- PUT file://C:/Documents/TabPFN_DemandModel/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
-- Verify:
-- LIST @MODEL_STAGE/scripts/;

-- ── Step 4: Create and call the orchestrator stored procedure ──────────────────
-- run_pipeline() downloads scripts from @MODEL_STAGE/scripts/ to /tmp/scripts/,
-- then submits HPO, training, and evaluation as sequential MLJob phases.
-- All execution happens within Snowflake — no local Python environment needed.
-- Re-run CREATE OR REPLACE after uploading a new version of run_training_job.py.
CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';

CALL run_training_pipeline();

-- ── Step 5: Verify output ──────────────────────────────────────────────────────
-- Confirms artifacts were written on completion:
LIST @MODEL_STAGE/hpo/;
LIST @MODEL_STAGE/checkpoints/;
LIST @MODEL_STAGE/results/;

-- ── Step 6: Download outputs (SnowSQL only) ────────────────────────────────────
-- GET @MODEL_STAGE/checkpoints/best.pt    file://C:/Users/nazer/Downloads/;
-- GET @MODEL_STAGE/results/test_report.csv file://C:/Users/nazer/Downloads/;
