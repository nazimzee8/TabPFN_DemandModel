-- run_training_job.sql
-- One-time environment setup + compute pool creation for the DeepSet training and evaluation pipelines.
-- Run steps 0-2, 2b, and 4 in Snowsight or SnowSQL.
-- Steps 3 and 3b (PUT) must be run in SnowSQL; PUT is not supported in Snowsight.
-- Training uses the Snowflake Container Runtime for ML.
-- For SnowSQL/connectors, YOUR_ACCOUNT_NAME means the Snowflake account identifier
-- from <account_identifier>.snowflakecomputing.com, preferably organization-account_name.

-- Step 0: Create database and schema
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
USE DATABASE TABPFN_DB;
CREATE SCHEMA IF NOT EXISTS TABPFN_SCHEMA;
USE SCHEMA TABPFN_SCHEMA;

-- Step 1: Create stages
-- Dedicated database and schema own project stages, compute pools, and model registry entries.
-- META_DATASET_STAGE: train/val/test synthetic parquet and staged benchmark datasets.
-- MLJobs materialize this stage into ephemeral container-local /tmp/data.
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- MODEL_STAGE: scripts, HPO config, and model checkpoints only.
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- EVALUATION_RESULTS_STAGE: all synthetic reports, benchmark part CSVs, and
-- the canonical final comparison file model_comparison.csv.
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- MLJOB_PAYLOAD_STAGE: bare stage name passed to submit_from_stage(stage_name=...).
-- It is separate from @MODEL_STAGE/scripts/, which is passed as source.
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Step 2: Create compute pools
-- GPU_NV_S: 1 A10G per node. MAX_NODES=4 supports 4-node DDP training.
-- This can exceed the earlier $5/hr budget cap when all GPU nodes are active.
-- CPU_X64_XS handles baseline benchmark jobs with bounded concurrency.
-- AUTOGLUON_CPU_POOL uses CPU_X64_M for the separate stacked-ensemble benchmark.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY; drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 4
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_SUSPEND_SECS = 300;

DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;
CREATE COMPUTE POOL DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 3
  INSTANCE_FAMILY = CPU_X64_XS
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

-- Verify pools reach ACTIVE before submitting the job:
SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';
SHOW COMPUTE POOLS LIKE 'DEEPSET_CPU_POOL';
SHOW COMPUTE POOLS LIKE 'AUTOGLUON_CPU_POOL';

-- Step 2b: Allow benchmark jobs to fetch datasets from inside Snowflake
-- Run with a role that can create network rules in TABPFN_SCHEMA.
-- OpenML hosts are required for evaluation. Kaggle API/www and Google storage hosts are
-- required for the one-off Kaggle download job. PyPI hosts are needed when MLJobs
-- install packages at runtime through pip_requirements.
USE ROLE SYSADMIN;
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

CREATE OR REPLACE NETWORK RULE openml_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'www.openml.org',
    'openml.org',
    'api.openml.org',
    'pypi.org',
    'files.pythonhosted.org'
  );

-- If CALL download_kaggle_to_stage() fails with host='api.kaggle.com', recreate
-- this network rule and the benchmark_external_access integration below.

CREATE OR REPLACE NETWORK RULE kaggle_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'api.kaggle.com',
    'www.kaggle.com',
    'kaggle.com',
    'storage.googleapis.com',
    'pypi.org',
    'files.pythonhosted.org'
  );

-- Kaggle API authentication requires the Kaggle account username/handle,
-- not the email address used to log in. A 401 Unauthorized from
-- DownloadDataFiles means Kaggle rejected these secret values.
-- If that happens, generate a fresh Kaggle API token and recreate this secret.
-- Keep this statement in Snowsight/SnowSQL history only; do not commit real keys.
CREATE OR REPLACE SECRET KAGGLE_API_SECRET
  TYPE = PASSWORD
  USERNAME = '<kaggle_account_username_not_email>'
  PASSWORD = '<kaggle_api_key>';

-- Run with ACCOUNTADMIN or another role granted CREATE EXTERNAL ACCESS INTEGRATION.
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION benchmark_external_access
  ALLOWED_NETWORK_RULES = (openml_network_rule, kaggle_network_rule)
  ALLOWED_AUTHENTICATION_SECRETS = (KAGGLE_API_SECRET)
  ENABLED = TRUE;

-- If a non-admin role submits the procedure or MLJobs, grant it least-privilege access:
-- GRANT READ ON SECRET KAGGLE_API_SECRET TO ROLE <job_submitter_role>;
-- GRANT USAGE ON INTEGRATION benchmark_external_access TO ROLE <job_submitter_role>;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- Step 3: Upload training and benchmark data (SnowSQL only)
-- PUT file://C:/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE;
-- PUT file://C:/Documents/TabPFN_DemandModel/data/kaggle/*.npz    @META_DATASET_STAGE/kaggle/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
-- Verify:
-- LIST @META_DATASET_STAGE/train/;
-- LIST @META_DATASET_STAGE/val/;
-- LIST @META_DATASET_STAGE/test/;
-- LIST @META_DATASET_STAGE/kaggle/;

-- Step 3b: Upload Python scripts (SnowSQL only)
-- Re-run whenever any script changes:
-- PUT file://C:/Documents/TabPFN_DemandModel/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--
-- Verify:
-- LIST @MODEL_STAGE/scripts/;

-- Step 4: Create and call the orchestrator stored procedure
-- download_kaggle_to_stage() is a separate setup job. Run it after scripts are
-- uploaded and before the first benchmark/training run that needs Kaggle data.
CREATE OR REPLACE PROCEDURE download_kaggle_to_stage()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_kaggle_download';

-- run_pipeline() launches only HPO and training. It produces
-- @MODEL_STAGE/hpo/best_config.json and @MODEL_STAGE/checkpoints/best.pt.
CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';

-- run_evaluation_pipeline() requires @MODEL_STAGE/checkpoints/best.pt, then runs
-- synthetic evaluation, DeepSetModel-MC, baseline benchmarks, AutoGluon, and the
-- aggregate comparison job.
CREATE OR REPLACE PROCEDURE run_evaluation_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_evaluation_pipeline';

CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;

-- Repair for Kaggle 401 Unauthorized:
-- USE ROLE SYSADMIN;
-- USE DATABASE TABPFN_DB;
-- USE SCHEMA TABPFN_SCHEMA;
-- CREATE OR REPLACE SECRET KAGGLE_API_SECRET
--   TYPE = PASSWORD
--   USERNAME = '<kaggle_account_username_not_email>'
--   PASSWORD = '<new_kaggle_api_key>';
-- Then re-run:
-- CALL download_kaggle_to_stage();
-- LIST @META_DATASET_STAGE/kaggle/;

-- Training only: HPO writes best_config.json; training consumes it and writes best.pt.
CALL run_training_pipeline();
LIST @MODEL_STAGE/hpo/;
LIST @MODEL_STAGE/checkpoints/;

-- Evaluation only: requires @MODEL_STAGE/checkpoints/best.pt and does not read best_config.json.
CALL run_evaluation_pipeline();

-- Step 5: Verify output
LIST @MODEL_STAGE/hpo/;
LIST @MODEL_STAGE/checkpoints/;
LIST @EVALUATION_RESULTS_STAGE/;

-- Step 6: Download outputs (SnowSQL only)
-- GET @MODEL_STAGE/checkpoints/best.pt 'file://C:/Documents/TabPFN_DemandModel/results/';
-- GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
