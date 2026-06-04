SELECT CURRENT_USER();
SELECT
    CURRENT_ORGANIZATION_NAME() AS orgname,
    CURRENT_ACCOUNT_NAME() AS acctname;

CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH;
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
CREATE SCHEMA  IF NOT EXISTS TABPFN_DB.TABPFN_SCHEMA;

USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

-- Stage for training data (Parquet files)
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for model checkpoints (best.pt)
CREATE STAGE IF NOT EXISTS MODEL_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for benchmark suites
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE 
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for deploying Snowflake ML jobs
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Stage for synthetic eval data (Parquet files)
CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE 
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = GPU_NV_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;
CREATE COMPUTE POOL DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 60
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TABPFN_PYPI_EAI
  ALLOWED_NETWORK_RULES = (SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE)
  ENABLED = TRUE
  COMMENT = 'Allows TabPFN ML Jobs to install approved PyPI dependencies in Snowflake-managed Container Runtime.';
GRANT USAGE ON INTEGRATION TABPFN_PYPI_EAI TO ROLE ACCOUNTADMIN;
GRANT DATABASE ROLE SNOWFLAKE.PYPI_REPOSITORY_USER TO ROLE ACCOUNTADMIN;

-- Model training procedures
CREATE TRANSIENT TABLE IF NOT EXISTS META_DATASET_INDEX (
  split STRING NOT NULL,
  task_id STRING NOT NULL,
  stage_path STRING NOT NULL,
  n NUMBER,
  p NUMBER,
  n_train NUMBER,
  n_test NUMBER,
  prior_regime STRING,
  hpo_bucket NUMBER
)
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY (split, hpo_bucket, prior_regime, p, n_train);

CREATE OR REPLACE PROCEDURE build_meta_dataset_index()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_dataset_index';

CREATE OR REPLACE PROCEDURE run_pretrain_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  GATE_HIDDEN_DIM INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_pretrain_job.py')
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline_model_gate';

CREATE OR REPLACE PROCEDURE run_hpo_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  HPO_SWEEP_MODE STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline_model_sweep';

CREATE OR REPLACE PROCEDURE run_model_ddp_memory_probe(
  MODEL_DESIGN_PATTERN STRING,
  MODEL_FAMILY STRING,
  N_CONTEXT INTEGER,
  P_FEATURES INTEGER,
  M_QUERY INTEGER,
  D_PHI INTEGER,
  N_BLOCKS INTEGER,
  RUN_BACKWARD BOOLEAN
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_ddp_memory_probe';

CREATE OR REPLACE PROCEDURE run_model_training(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_training_model';

GRANT DATABASE ROLE SNOWFLAKE.PYPI_REPOSITORY_USER TO ROLE ACCOUNTADMIN;

-- Execute ML Training Pipeline
CALL build_meta_dataset_index();
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_combined',
  'inductive_forecasting', 32
);
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_combined',
  'inductive_forecasting', 64
);
CALL run_pretrain_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_combined',
  'inductive_forecasting', 128
);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SET AUTO_SUSPEND_SECS = 300;
CALL run_hpo_pipeline(
  'market_exchangeable_icl', 'synthetic_regression_combined',     
  'inductive_forecasting', 'ridge_residual'
);
CALL run_model_training(
  'market_exchangeable_icl',           
  'synthetic_regression_combined',     
  'inductive_forecasting'
);

DROP TABLE IF EXISTS SYNTHETIC_REGRESSION_DATASET_INDEX;
CREATE TRANSIENT TABLE IF NOT EXISTS SYNTHETIC_REGRESSION_DATASET_INDEX (
  suite_id             STRING,
  suite_family         STRING,
  dataset_id           NUMBER,
  dataset_seed         NUMBER,
  stage_path           STRING,
  prior_name           STRING,
  prior_version        STRING,
  prior_regime         STRING,
  split_seeds          ARRAY,
  n_total              NUMBER,
  n_train_default      NUMBER,
  n_holdout_default    NUMBER,
  p_signal             NUMBER,
  p_noise              NUMBER,
  p_total              NUMBER,
  target_noise_scale   FLOAT,
  training_size_anchor BOOLEAN,
  feature_noise_level  NUMBER,
  eval_weight          FLOAT,
  payload_bytes        NUMBER,
  created_at           TIMESTAMP_NTZ,
  logical_dataset_key  STRING,
  source_suite_id      STRING
) DATA_RETENTION_TIME_IN_DAYS = 0;

CREATE IMAGE REPOSITORY IF NOT EXISTS MODEL_IMAGE_REPOSITORY;
SHOW IMAGE REPOSITORIES;

CREATE ROLE IF NOT EXISTS TABPFN_IMAGE_PUSHER;
CREATE USER IF NOT EXISTS GITHUB_ACTIONS_IMAGE_PUSHER
  TYPE = SERVICE
  LOGIN_NAME = 'GITHUB_ACTIONS_IMAGE_PUSHER'
  COMMENT = 'GitHub Actions service user for pushing SPCS AutoGluon images';

GRANT ROLE TABPFN_IMAGE_PUSHER TO USER GITHUB_ACTIONS_IMAGE_PUSHER;

ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
  SET DEFAULT_ROLE = 'TABPFN_IMAGE_PUSHER';

GRANT USAGE ON DATABASE TABPFN_DB TO ROLE TABPFN_IMAGE_PUSHER;
GRANT USAGE ON SCHEMA TABPFN_DB.TABPFN_SCHEMA TO ROLE TABPFN_IMAGE_PUSHER;
GRANT READ, WRITE ON IMAGE REPOSITORY TABPFN_DB.TABPFN_SCHEMA.MODEL_IMAGE_REPOSITORY
  TO ROLE TABPFN_IMAGE_PUSHER;

CREATE AUTHENTICATION POLICY IF NOT EXISTS TABPFN_GITHUB_ACTIONS_PAT_POLICY
  PAT_POLICY = (
    NETWORK_POLICY_EVALUATION = ENFORCED_NOT_REQUIRED
    REQUIRE_ROLE_RESTRICTION_FOR_SERVICE_USERS = TRUE
  )
  COMMENT = 'Allow role-restricted PATs for GitHub Actions image push service user';

ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
  SET AUTHENTICATION POLICY TABPFN_GITHUB_ACTIONS_PAT_POLICY;

ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
    REMOVE PROGRAMMATIC ACCESS TOKEN GITHUB_ACTIONS_DOCKER_PUSH;
    
ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
  ADD PROGRAMMATIC ACCESS TOKEN GITHUB_ACTIONS_DOCKER_PUSH
  ROLE_RESTRICTION = 'TABPFN_IMAGE_PUSHER'
  DAYS_TO_EXPIRY = 90
  COMMENT = 'GitHub Actions Docker login for Snowflake image registry';

SHOW USER PROGRAMMATIC ACCESS TOKENS FOR USER
  GITHUB_ACTIONS_IMAGE_PUSHER;

  -- Prepare synthetic regression evaluation data
CREATE OR REPLACE PROCEDURE run_synthetic_regression_prep(
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
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_prep';

-- OOD Full Suite — Split-Phase Procedures
CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_full_prep(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_ood_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_prep';

-- Combined Suite — Split-Phase Procedures
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

CREATE OR REPLACE PROCEDURE run_synthetic_regression_ood_deepset_pilot(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = (
    '@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py',
    '@MODEL_STAGE/scripts/prepare_ood_regression.py',
    '@MODEL_STAGE/scripts/prepare_synthetic_regression.py'
  )
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_ood_deepset_pilot';

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_deepset_evaluation(
  BENCH_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_deepset_evaluation';

-- Combined baseline capacity probe — verify DEEPSET_CPU_POOL can scale before evaluation:
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
  
-- Combined AutoGluon capacity probe — verify AUTOGLUON_CPU_POOL can satisfy the distributed
-- envelope (concurrent_clusters * workers_per_shard nodes) before evaluation:
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

-- SPCS custom image variation of AutoGluon
CREATE OR REPLACE PROCEDURE run_synthetic_regression_autogluon_spcs_import_probe(
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
  PROBE_COUNT INTEGER
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_spcs_import_probe';

-- Verification of OAuth for Snowpark session injection
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

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
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

CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_worker_access_probe(
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
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
  IMPORTS = ('@MODEL_STAGE/scripts/run_synthetic_regression_evaluation.py')
  HANDLER = 'run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_worker_access_probe';

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

-- Aggregate evaluation data
CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_aggregation(
  BENCH_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING,
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

-- Prepare evaluation data
CALL run_synthetic_regression_prep('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_ood_full_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_combined_prep('2.5.0-py311', '2.5.0-py311');

-- Setup SPCS image for AutoGluon
SET AUTOGLUON_IMAGE_REF = 'dvxcfsm-hs34800.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/model_image_repository/tabpfn-autogluon-ray:1.0.0';
SET SPCS_RAY_HEAD_DNS_SUFFIX = 'tabpfn_schema.tabpfn_db.snowflakecomputing.internal';
CALL run_synthetic_regression_autogluon_spcs_import_probe($AUTOGLUON_IMAGE_REF, 1);
CALL run_synthetic_regression_autogluon_spcs_session_probe($AUTOGLUON_IMAGE_REF, 1);
CALL run_synthetic_regression_combined_autogluon_spcs_worker_access_probe($AUTOGLUON_IMAGE_REF, 1, 8, 1, 1, FALSE);
CALL run_synthetic_regression_combined_autogluon_spcs_capacity_probe($AUTOGLUON_IMAGE_REF,  6, 4, 6, 300, 1, TRUE);

-- Recommended: 6 clusters x 4 workers = 24 concurrent CPU_X64_M nodes, 1 CPU per fit task
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*best[.]pt';
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL RESUME;
CALL run_synthetic_regression_combined_autogluon_spcs_evaluation($AUTOGLUON_IMAGE_REF, 6, 4, 1, 6, 300, 'high_quality', 600, 1);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
ALTER COMPUTE POOL DEEPSET_GPU_POOL RESUME;
CALL run_synthetic_regression_combined_deepset_evaluation('2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_baseline_evaluation('2.5.0-py311', 10, 10);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', $AUTOGLUON_IMAGE_REF, 6, 10, 10);
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60;
ALTER COMPUTE POOL DEEPSET_GPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL DEEPSET_CPU_POOL SET AUTO_SUSPEND_SECS = 300;
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SET AUTO_SUSPEND_SECS = 300;

-- Debug AutoGluon
/*SHOW JOB SERVICES IN COMPUTE POOL DEEPSET_GPU_POOL;
SELECT
  "name" AS service_name,
  "database_name" AS database_name,
  "schema_name" AS schema_name,
  "status" AS status,
  "target_instances" AS target_instances,
  "current_instances" AS current_instances,
  "created_on" AS created_on,
  "updated_on" AS updated_on
FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
ORDER BY "created_on" DESC
LIMIT 10;
SHOW SERVICE CONTAINERS IN SERVICE SPCS_CAP_RAY_COORD_2A41EC8E_0;
CALL SYSTEM$GET_SERVICE_STATUS('EVALUATE_SYNTHETIC_REGRESSION_40F4B086_1UZT721JVBOXT');
CALL SYSTEM$GET_SERVICE_LOGS('EVALUATE_SYNTHETIC_REGRESSION_40F4B086_1UZT721JVBOXT', '0', 'main', 1000);*/



