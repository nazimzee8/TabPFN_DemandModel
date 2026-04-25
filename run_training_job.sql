-- run_training_job.sql
-- One-time environment setup + compute pool creation for the DeepSet training pipeline.
-- Run in Snowsight or SnowSQL before executing run_training_job.py.
-- The Docker image is NOT required — training uses the Snowflake Container Runtime for ML.

-- ── Step 0: Create database and schema ────────────────────────────────────────
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
USE DATABASE TABPFN_DB;
CREATE SCHEMA IF NOT EXISTS TABPFN_SCHEMA;
USE SCHEMA TABPFN_SCHEMA;

-- ── Step 1: Create stages ──────────────────────────────────────────────────────
-- META_DATASET_STAGE: training data uploaded via PUT (see Snowflake_Training.md).
-- The Container Runtime mounts this stage at /data/ inside the training container.
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- MODEL_STAGE: train.py writes best.pt here; evaluate.py writes test_report.csv here.
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ── Step 2: Create compute pool ────────────────────────────────────────────────
-- GPU_NV_S: 1× A10G per node, 0.57 cr/hr → ~$1.14–1.71/node/hr (Standard/Enterprise).
-- 2 nodes for training (DDP) or 2 parallel HPO trials = ~$2.28–3.42/hr total.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY — must drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 2
  INSTANCE_FAMILY = GPU_NV_S;

-- Verify the pool reaches ACTIVE state before running the Python job (re-run until Status = ACTIVE):
-- SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';

-- ── Step 3: Upload training data ───────────────────────────────────────────────
-- Run these PUT commands in SnowSQL (not supported in Snowsight web UI):
--
--   PUT file:///c/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE;
--   PUT file:///c/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE;
--   PUT file:///c/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE;
--
-- Verify:
--   LIST @META_DATASET_STAGE/train/;
--   LIST @META_DATASET_STAGE/val/;
--   LIST @META_DATASET_STAGE/test/;

-- ── Step 4: Submit jobs ────────────────────────────────────────────────────────
-- Training is submitted via run_training_job.py (Container Runtime for ML — no Docker build needed):
--
--   set SNOWFLAKE_ACCOUNT=<account-identifier>
--   set SNOWFLAKE_USER=<user>
--   set SNOWFLAKE_PASSWORD=<password>
--   python run_training_job.py
--
-- The script runs three phases in sequence:
--   1. HPO  — 20 BayesOpt trials × 30 epochs on 2 GPU_NV_S nodes
--   2. Train — full DDP training on 2 GPU_NV_S nodes using the best HPO config
--   3. Eval  — evaluation on 1 GPU_NV_S node; writes test_report.csv

-- ── Step 5: Verify output ─────────────────────────────────────────────────────
-- Confirms artifacts were uploaded on completion:
-- LIST @MODEL_STAGE/hpo/;
-- LIST @MODEL_STAGE/checkpoints/;
-- LIST @MODEL_STAGE/results/;

-- ── Step 6: Download outputs (SnowSQL only) ────────────────────────────────────
-- Or use download_results.py for a Python-based alternative.
LIST @MODEL_STAGE/checkpoints/;
LIST @MODEL_STAGE/results/;

-- GET to local path (adjust path for your OS):
--   Windows:  file://C:/Users/<you>/Downloads/
--   macOS:    file:///Users/<you>/Downloads/
GET @MODEL_STAGE/checkpoints/best.pt    file://C:/Users/nazer/Downloads/;
GET @MODEL_STAGE/results/test_report.csv file://C:/Users/nazer/Downloads/;
