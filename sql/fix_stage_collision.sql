-- fix_stage_collision.sql
-- Fixes the AUTO_COMPRESS collision where both .py and .py.gz exist on @MODEL_STAGE/scripts/.
-- The uncompressed .py takes name precedence, so the container was running stale code.
--
-- Run ALL steps in SnowSQL (PUT/REMOVE commands are not supported in Snowsight).
--
--   snowsql -a <account> -u <username>
--   USE DATABASE TABPFN_DB;
--   USE SCHEMA TABPFN_SCHEMA;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- ── Step 1: Remove stale uncompressed files ──────────────────────────────────
REMOVE @MODEL_STAGE/scripts/hpo.py;
REMOVE @MODEL_STAGE/scripts/run_training_job.py;

-- ── Step 2: Re-upload both files with AUTO_COMPRESS=FALSE ───────────────────
-- This lands the .py file itself on the stage — no ambiguity with a .gz sibling.
PUT file://C:\Documents\TabPFN_DemandModel\hpo.py
    @MODEL_STAGE/scripts/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE;

PUT file://C:\Documents\TabPFN_DemandModel\run_training_job.py
    @MODEL_STAGE/scripts/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE;

-- ── Step 3: Remove stale .gz files ──────────────────────────────────────────
REMOVE @MODEL_STAGE/scripts/hpo.py.gz;
REMOVE @MODEL_STAGE/scripts/run_training_job.py.gz;

-- ── Step 4: Verify — expect exactly two .py files, no .gz, fresh timestamps ─
LIST @MODEL_STAGE/scripts/;

-- ── Step 5: Recreate stored procedure to bust the IMPORTS cache ─────────────
-- The procedure caches @MODEL_STAGE/scripts/run_training_job.py at creation time.
-- CREATE OR REPLACE forces a fresh read of the newly uploaded file.
CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';

-- ── Step 6: Run the pipeline ─────────────────────────────────────────────────
CALL run_training_pipeline();
