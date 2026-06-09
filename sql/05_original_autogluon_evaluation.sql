-- prepare_benchmark_datasets() fetches OpenML/Kaggle benchmark datasets once,
-- stages them to @META_REGRESSION_DATASET_STAGE/benchmark_prepared/, and writes
-- benchmark_manifest.json. Run before run_evaluation_pipeline(), or let
-- run_evaluation_pipeline() call it automatically.
-- @META_REGRESSION_DATASET_STAGE/benchmark_prepared/ is created by this procedure.
-- It contains benchmark_manifest.json and prepared .npz files for all benchmark datasets.
-- Benchmark shard jobs read exact prepared files from this prefix; they do not
-- call OpenML directly and they load only one owned dataset at a time.
-- The procedure also refreshes BENCHMARK_DATASET_INDEX from manifest metadata.
-- To rebuild: set BENCHMARK_FORCE_REBUILD=true env var before calling this procedure.
CREATE OR REPLACE PROCEDURE prepare_benchmark_datasets()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python', 'openml==0.15.1')
  EXTERNAL_ACCESS_INTEGRATIONS = (BENCHMARK_EXTERNAL_ACCESS)
  IMPORTS = ('@MODEL_STAGE/scripts/prepare_benchmark_datasets.py')
  HANDLER = 'prepare_benchmark_datasets.prepare_datasets';

-- run_evaluation_pipeline() requires @MODEL_STAGE/checkpoints/best.pt and
-- @MODEL_STAGE/scripts/runtime_probe.py. It preflights compute pools and
-- configured evaluation runtime images with runtime-specific REQUIRED_IMPORTS,
-- then runs synthetic evaluation and benchmark dataset preparation. The prep
-- job always runs as a lightweight manifest/BENCHMARK_DATASET_INDEX validation
-- step before shard submission, even when benchmark_manifest.json already exists.
CREATE OR REPLACE PROCEDURE run_evaluation_pipeline(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_pipeline';

-- Drop old zero-argument overload if it exists:
DROP PROCEDURE IF EXISTS run_evaluation_pipeline();

-- Call:
-- CALL run_evaluation_pipeline('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');

-- run_evaluation_runtime_probes() runs all preflight checks without submitting
-- evaluation jobs. Use this to validate runtime environments before a full run.
CREATE OR REPLACE PROCEDURE run_evaluation_runtime_probes(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_runtime_probes';

-- Call:
-- CALL run_evaluation_runtime_probes('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');

-- run_evaluation_capacity_probe() is a lightweight quota/capacity check. It submits
-- capacity_probe.py in 3 non-overlapping phases matching the fixed evaluation pipeline
-- envelope (GPU=10, CPU=3, AutoGluon=30). Run between runtime probes and the full pipeline.
CREATE OR REPLACE PROCEDURE run_evaluation_capacity_probe(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_capacity_probe';

-- Call:
-- CALL run_evaluation_capacity_probe('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');

-- Split-phase evaluation: run each phase independently to release quota between pools.
-- Recommended run sequence under tight node quota:
--   CALL run_evaluation_runtime_probes('<prep>', '<bench>', '<ag>');
--   CALL run_evaluation_prep('<prep>', '<bench>', '<ag>');
--   CALL run_deepset_evaluation('<prep>', '<bench>', '<ag>');
--   ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
--   CALL run_baseline_evaluation('<prep>', '<bench>', '<ag>');
--   ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
--   CALL run_autogluon_evaluation('<prep>', '<bench>', '<ag>');
--   ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
--   CALL run_evaluation_aggregation('<prep>', '<bench>', '<ag>');
--
-- run_evaluation_prep() fetches/validates benchmark manifest and index on DEEPSET_CPU_POOL.
CREATE OR REPLACE PROCEDURE run_evaluation_prep(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_prep';

-- run_deepset_evaluation() runs synthetic eval and 10 DeepSet GPU shards on DEEPSET_GPU_POOL.
-- Requires @MODEL_STAGE/checkpoints/best.pt.
CREATE OR REPLACE PROCEDURE run_deepset_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_deepset_evaluation';

-- run_baseline_evaluation() runs 3 CPU baseline benchmark shards on DEEPSET_CPU_POOL.
CREATE OR REPLACE PROCEDURE run_baseline_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_baseline_evaluation';

-- run_autogluon_evaluation() runs 30 AutoGluon shards (max 30 concurrent) on AUTOGLUON_CPU_POOL.
CREATE OR REPLACE PROCEDURE run_autogluon_evaluation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_autogluon_evaluation';

-- run_evaluation_aggregation() runs the benchmark aggregation job on DEEPSET_CPU_POOL
-- and returns a listing of @EVALUATION_RESULTS_STAGE. Can be re-run without re-running
-- prior phases if benchmark_parts/ files already exist on stage.
CREATE OR REPLACE PROCEDURE run_evaluation_aggregation(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_aggregation';

-- ============================================================
-- Calls
-- ============================================================

-- Evaluation only: requires @MODEL_STAGE/checkpoints/best.pt and does not read best_config.json.
-- Step 1: Validate runtime images (serialized probes, no model/data loaded).
CALL run_evaluation_runtime_probes(
  '<prep_runtime_image_name>',
  '<benchmark_runtime_image_name>',
  '<autogluon_runtime_image_name>'
);

-- Step 2: Validate node quota (capacity_probe.py, 3 phases: GPU=10, CPU=3, AutoGluon=30).
-- If this fails with a node limit error: SHOW COMPUTE POOLS; suspend idle pools;
-- wait for active jobs to finish; or request higher Snowflake account node quota.
CALL run_evaluation_capacity_probe(
  '<prep_runtime_image_name>',
  '<benchmark_runtime_image_name>',
  '<autogluon_runtime_image_name>'
);

-- Step 3 (recommended under tight quota): Split-phase evaluation.
-- Run each phase independently and suspend its pool to release quota before the next.
CALL run_evaluation_prep('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');

CALL run_deepset_evaluation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;

CALL run_baseline_evaluation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;

CALL run_autogluon_evaluation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

CALL run_evaluation_aggregation('<prep_runtime_image_name>', '<benchmark_runtime_image_name>', '<autogluon_runtime_image_name>');

-- Step 3 (legacy convenience, holds all 3 pools simultaneously):
-- CALL run_evaluation_pipeline(
--   '<prep_runtime_image_name>',
--   '<benchmark_runtime_image_name>',
--   '<autogluon_runtime_image_name>'
-- );

-- Step 5: Verify output
LIST @MODEL_STAGE/hpo/;
LIST @MODEL_STAGE/checkpoints/;
LIST @EVALUATION_RESULTS_STAGE/;

-- Step 6: Download outputs (SnowSQL only)
-- GET @MODEL_STAGE/checkpoints/best.pt 'file://C:/Documents/TabPFN_DemandModel/results/';
-- GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
