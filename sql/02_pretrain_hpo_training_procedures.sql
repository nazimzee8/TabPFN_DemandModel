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

-- build_meta_dataset_index() launches build_meta_dataset_index.py on the CPU
-- pool. It lists @META_DATASET_STAGE/{train,val,test}/, reads scalar parquet
-- metadata, truncates/rebuilds META_DATASET_INDEX, and validates 800/100/100.
CREATE OR REPLACE PROCEDURE build_meta_dataset_index()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_dataset_index';

-- run_pretrain_pipeline() — pretrain entrypoints.
--
-- Production path: use the 4-arg gate-specific overload for all three gate candidates.
-- HPO requires pretrain_gate32.pt, pretrain_gate64.pt, and pretrain_gate128.pt to exist
-- before starting. HPO fails hard if any are missing.
--
-- 4-arg gate-specific form (production):
--   CALL run_pretrain_pipeline(MODEL_FAMILY, TRAINING_DATA_FAMILY, MODEL_DESIGN_PATTERN, GATE_DIM);
--   Example: CALL run_pretrain_pipeline('market_exchangeable_icl',
--                'synthetic_regression_combined', 'inductive_forecasting', 64);
--
-- 0-arg form (env-var defaults; writes pretrain.pt — legacy, not used by HPO flow):
--   CALL run_pretrain_pipeline();
-- 3-arg form (explicit selectors; writes pretrain.pt — legacy, not used by HPO flow):
--   CALL run_pretrain_pipeline(
--       'market_exchangeable_icl',
--       'synthetic_regression_combined',
--       'inductive_forecasting'
--   );
CREATE OR REPLACE PROCEDURE run_pretrain_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_pretrain_job.py')
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline';

CREATE OR REPLACE PROCEDURE run_pretrain_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_pretrain_job.py')
  HANDLER = 'run_pretrain_job.run_pretrain_pipeline_model';

-- Gate-specific pretrain: writes pretrain_gate<N>.pt for one HPO gate candidate.
-- Must be called for all three candidates (32, 64, 128) before run_hpo_pipeline().
-- The ridge_residual HPO tunes gate_hidden_dim; each trial requires a matching
-- pretrain checkpoint to warm-start from.
--   CALL run_pretrain_pipeline(
--       'market_exchangeable_icl',       -- MODEL_FAMILY
--       'synthetic_regression_combined', -- TRAINING_DATA_FAMILY
--       'inductive_forecasting',         -- MODEL_DESIGN_PATTERN
--       64                               -- GATE_HIDDEN_DIM (32, 64, or 128)
--   );
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

-- run_hpo_pipeline() launches hpo.py on the GPU pool using Ray Tune for distributed HPO.
-- Sweep-specific outputs: best_config_ridge_residual.json, best_config_architecture.json.
-- Merged output (written by architecture sweep): best_config.json.
-- Keys: lr, weight_decay, dropout, d_phi, d_rho, pool, n_sab_feat,
--   use_ridge_expert, ridge_lambda, gate_hidden_dim, use_huber, huber_delta, lambda_l1,
--   model_family, model_arch_version, model_design_pattern, hpo_sweep_mode, _meta.
-- Note: HPO only supports inductive_forecasting; transductive_completion raises in hpo.py.
--
-- Two-sweep strategy (recommended):
--   Sweep 1 (ridge_residual): Fixed architecture (d_phi=128, n_sab_feat=1). Tunes
--     optimizer, Ridge Expert (ridge_lambda, gate_hidden_dim ∈ {32,64,128}), and
--     robust loss params. Requires gate-specific pretrain checkpoints to exist.
--     Writes best_config_ridge_residual.json and best_config.json.
--   Run model_ddp_memory_probe for worst-case d_phi=256, n_blocks=2 before sweep 2.
--   Sweep 2 (architecture): Tunes d_phi and n_sab_feat. Freezes optimizer params from
--     sweep 1 via HPO_BASELINE_CONFIG_STAGE_PATH. Allows cold-start on arch mismatch.
--     Writes best_config_architecture.json and merged best_config.json.
--
-- Zero-arg (uses env-var defaults; HPO_SWEEP_MODE defaults to ridge_residual):
--   CALL run_hpo_pipeline();
-- Three-arg (explicit model selectors; HPO_SWEEP_MODE from env-var, defaults ridge_residual):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting'
--   );
-- Four-arg (production-recommended explicit form):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual'
--   );
CREATE OR REPLACE PROCEDURE run_hpo_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline';

CREATE OR REPLACE PROCEDURE run_hpo_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline_model';

-- Four-arg overload: explicit HPO_SWEEP_MODE selector.
--
-- Sweep 1 (ridge_residual — tunes optimizer/regularization):
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual'
--   );
-- Sweep 2 (architecture — tunes d_phi/n_sab_feat; run after ridge_residual + memory probe):
-- Use five-arg overload below to pass HPO_BASELINE_CONFIG_STAGE_PATH for architecture sweep.
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

-- Five-arg overload: explicit HPO_SWEEP_MODE + baseline config stage path.
-- HPO_BASELINE_CONFIG_STAGE_PATH is required when HPO_SWEEP_MODE='architecture';
-- pass '' for ridge_residual.
--
-- Two-sweep HPO (recommended):
--   -- Sweep 1: ridge_residual
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'ridge_residual', ''
--   );
--   -- Run memory probe before architecture sweep:
--   CALL run_model_ddp_memory_probe(
--       'model3', 'inductive_forecasting', 'market_exchangeable_icl',
--       200, 128, 128, 256, 2, TRUE
--   );
--   -- Sweep 2: architecture (requires best_config_ridge_residual.json from sweep 1)
--   CALL run_hpo_pipeline(
--       'market_exchangeable_icl', 'synthetic_regression_combined',
--       'inductive_forecasting', 'architecture',
--       '@MODEL_STAGE/hpo/best_config_ridge_residual.json'
--   );
CREATE OR REPLACE PROCEDURE run_hpo_pipeline(
  MODEL_FAMILY STRING,
  TRAINING_DATA_FAMILY STRING,
  MODEL_DESIGN_PATTERN STRING,
  HPO_SWEEP_MODE STRING,
  HPO_BASELINE_CONFIG_STAGE_PATH STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline_model_sweep_with_baseline';

-- run_model_training() reads @MODEL_STAGE/hpo/best_config.json, passes it to
-- train.py as BEST_CONFIG, and produces @MODEL_STAGE/checkpoints/best.pt.
-- Checkpoint metadata includes: model_family, task_type, training_data_family,
--   best_val_mse, train_mse_at_best, best_epoch, pytorch_version.
-- Zero-arg form uses env-var defaults (MODEL_FAMILY, TRAINING_DATA_FAMILY, MODEL_DESIGN_PATTERN):
CREATE OR REPLACE PROCEDURE run_model_training()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_training';

-- Parameterized form — same explicit runtime lineage variables as pretrain and HPO:
--   CALL run_model_training(
--       'market_exchangeable_icl',       -- MODEL_FAMILY
--       'synthetic_regression_combined', -- TRAINING_DATA_FAMILY
--       'inductive_forecasting'          -- MODEL_DESIGN_PATTERN
--   );
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

CREATE OR REPLACE PROCEDURE run_training_runtime_probe(target_instances INTEGER)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_training_runtime_probe';

-- Usage:
--   CALL run_training_runtime_probe(1);   -- single-node probe
--   CALL run_training_runtime_probe(2);   -- 2-node probe (optional)
--   CALL run_training_runtime_probe(5);   -- 5-node probe (optional)
--   CALL run_training_runtime_probe(10);  -- full-topology probe
--
-- Expected success markers in job logs:
--   [runtime_probe] entered Python
--   [runtime_probe] completed
--
-- If logs show Prometheus mmap panic before '[runtime_probe] entered Python',
-- escalate to Snowflake Support as a managed MLJob/Ray/Prometheus runtime issue.

-- run_training_pipeline() runs the full 7-step two-sweep pipeline in sequence:
--   Step 1  (Validate)            META_DATASET_INDEX counts, columns, stage access
--   Step 2  (Pretrain)          → @MODEL_STAGE/checkpoints/pretrain.pt
--   Step 3  (HPO ridge_residual)→ @MODEL_STAGE/hpo/best_config_ridge_residual.json
--   Step 4  (Memory probe)        Worst-case d_phi=256, n_blocks=2; guards against OOM
--   Step 5  (HPO architecture)  → @MODEL_STAGE/hpo/best_config.json (merged)
--   Step 6  (Load config)         Reads merged best_config.json
--   Step 7  (Final training)    → @MODEL_STAGE/checkpoints/best.pt
--                                  (with _meta.pretrain_checkpoint_stage_path)
--   Step 6  (Load config)        Downloads and parses best_config.json
--   Step 7  (Final training)  → @MODEL_STAGE/checkpoints/best.pt
-- Architecture HPO trials freeze optimizer params from best_config_ridge_residual.json
-- via HPO_BASELINE_CONFIG_STAGE_PATH. Cold-start on arch mismatch is allowed.
-- Final training uses PRETRAIN_LOAD_POLICY=allow_cold_start_on_arch_mismatch with
-- PRETRAIN_CHECKPOINT_PATH=@MODEL_STAGE/checkpoints/pretrain.pt.
CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';

-- run_hpo_epoch_test() runs baseline and marginal HPO epoch timing sweeps.
-- Result at @EPOCH_STAGE/hpo_timing.json; read with SELECT below.
CREATE OR REPLACE PROCEDURE run_hpo_epoch_test()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_epoch_tests.py')
  HANDLER = 'run_epoch_tests.run_hpo_epoch_test';

-- run_train_epoch_test() runs one DDP training epoch with the production
-- GPU_NV_M topology: 10 nodes x 4 workers/node = world_size 40.
-- Result at @EPOCH_STAGE/train_timing.json; read with SELECT below.
CREATE OR REPLACE PROCEDURE run_train_epoch_test()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_epoch_tests.py')
  HANDLER = 'run_epoch_tests.run_train_epoch_test';

-- run_model_ddp_memory_probe() measures peak CUDA memory per DDP worker for a
-- representative MODEL3 ICL shape before pretrain / HPO / final training.
--
-- Run RUN_BACKWARD=TRUE (always) because MODEL3 meta-training uses back-propagation;
-- the probe with backward gives a faithful peak-memory measurement for training-regime
-- validation that covers gradient tensors and activation storage during backprop.
--
-- Launches model_ddp_memory_probe.py on DEEPSET_GPU_POOL with the same topology as
-- training (TRAIN_NUM_NODES=10 nodes x 4 workers = world_size 40).
--
-- Diagnostic JSON uploaded to @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json.
--
-- Usage:
--   CALL run_training_runtime_probe(1);
--   CALL run_training_runtime_probe(10);
--
--   CALL run_model_ddp_memory_probe(
--       'inductive_forecasting',
--       'market_exchangeable_icl',
--       200,    -- N_CONTEXT
--       128,    -- P_FEATURES
--       128,    -- M_QUERY
--       128,    -- D_PHI
--       1,      -- N_BLOCKS
--       TRUE    -- RUN_BACKWARD (always TRUE for training-regime validation)
--   );
--
--   LIST @MODEL_STAGE/diagnostics/;
--   SELECT $1
--   FROM @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json
--     (FILE_FORMAT => (TYPE = JSON));
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

-- ============================================================
-- Calls
-- ============================================================

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

-- Step 4b: Epoch calibration (run before upgrading compute pool)
-- Requires @MODEL_STAGE/scripts/ populated with src/*.py and scripts/*.py (Step 3b above).
LIST @MODEL_STAGE/scripts/ PATTERN='.*(hpo_epoch_test|train_epoch_test|train|model|snowflake_io)[.]py';
CALL run_hpo_epoch_test();
SELECT $1 FROM @EPOCH_STAGE/hpo_timing.json (FILE_FORMAT => (TYPE = JSON));
-- Inspect HPO timing by phase, not only epoch_time_s. Query wall time includes
-- MLJob startup, Ray Tune scheduling, metadata selection, stage materialization,
-- and trial epoch compute.
SELECT
  $1:metadata:metadata_selection_time_s::FLOAT AS metadata_selection_time_s,
  $1:metadata:materialization_time_s::FLOAT AS materialization_time_s,
  $1:summary:mean_epoch_time_s::FLOAT AS mean_epoch_time_s,
  $1:summary:max_epoch_time_s::FLOAT AS max_epoch_time_s,
  $1:summary:parallel_trials::NUMBER AS parallel_trials,
  $1:summary:hpo_rounds::NUMBER AS hpo_rounds,
  $1:summary:estimated_hpo_wall_time_s_mean::FLOAT AS estimated_hpo_wall_time_s_mean,
  $1:summary:estimated_hpo_wall_time_s_conservative::FLOAT AS estimated_hpo_wall_time_s_conservative
FROM @EPOCH_STAGE/hpo_timing.json (FILE_FORMAT => (TYPE = JSON));

CALL run_train_epoch_test();
SELECT $1 FROM @EPOCH_STAGE/train_timing.json (FILE_FORMAT => (TYPE = JSON));
-- Decision gate:
--   summary.parallel_trials = 20 and summary.hpo_rounds = 1
--       -> 5 nodes GPU_NV_M (20 concurrent, ~30-60 min HPO including overhead)
--   high materialization time
--       -> inspect metadata selection/materialization before changing topology
--   summary.max_epoch_time_s > 30 s    -> re-evaluate; consider GPU_NV_L or reducing num_trials

-- Preferred staged training: HPO writes best_config.json; training consumes it and writes best.pt.
-- Rebuild META_DATASET_INDEX before HPO/training whenever staged synthetic parquet changes.
-- Pre-warm: issue RESUME so the GPU pool transitions SUSPENDED→ACTIVE while the
-- CPU index job runs. With AUTO_RESUME=TRUE the pool also starts on job submission,
-- but this moves the ~3-5 min startup wait off the critical path of run_hpo_pipeline().
ALTER COMPUTE POOL DEEPSET_GPU_POOL RESUME;

-- This runs on DEEPSET_CPU_POOL (GPU pool starts warming in background):
CALL build_meta_dataset_index();
-- Verify full split counts are 800/100/100 and deterministic HPO subset query
-- above returns 200 train rows and 40 val rows.

-- GPU pool should be ACTIVE by now; no startup wait inside the jobs:
-- Runtime probe (run before final training if MLJob startup failures are suspected):
CALL run_training_runtime_probe(1);   -- single-node probe
CALL run_training_runtime_probe(2);   -- 2-node probe (optional)
CALL run_training_runtime_probe(5);   -- 5-node probe (optional)
CALL run_training_runtime_probe(10);  -- full-topology probe
-- Two-sweep HPO strategy (recommended):
--
-- Option A: Gate-specific pretrain + ridge_residual sweep only (single-sweep):
--   Gate-specific pretrains are required for ridge_residual HPO.
--   HPO tunes gate_hidden_dim across all three candidates.
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
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_gate.*[.]pt';

-- HPO sweep 1: ridge_residual — fixed d_phi/n_sab_feat, tune optimizer/Ridge Expert/gate.
-- Writes best_config_ridge_residual.json and best_config.json.
CALL run_hpo_pipeline(
  'market_exchangeable_icl',
  'synthetic_regression_combined',
  'inductive_forecasting',
  'ridge_residual'
);
LIST @MODEL_STAGE/hpo/;
-- Inspect best_config_ridge_residual.json:
SELECT $1 FROM @MODEL_STAGE/hpo/best_config_ridge_residual.json (FILE_FORMAT => (TYPE = JSON));

-- Option B (recommended): Continue with architecture sweep after ridge_residual:
-- Mandatory memory probe before architecture HPO (worst-case: d_phi=256, n_blocks=2).
CALL run_model_ddp_memory_probe(
  'model3', 'inductive_forecasting', 'market_exchangeable_icl',
  200, 128, 128, 256, 2, TRUE
);

-- HPO sweep 2: architecture — tunes d_phi and n_sab_feat; freezes optimizer from sweep 1.
-- HPO_BASELINE_CONFIG_STAGE_PATH must point to best_config_ridge_residual.json from sweep 1.
-- Writes best_config_architecture.json and merged best_config.json.
CALL run_hpo_pipeline(
  'market_exchangeable_icl',
  'synthetic_regression_combined',
  'inductive_forecasting',
  'architecture',
  '@MODEL_STAGE/hpo/best_config_ridge_residual.json'
);
LIST @MODEL_STAGE/hpo/;
-- Inspect merged best_config.json; expected: _meta.sweeps.ridge_residual + .architecture:
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));

CALL run_model_training();
LIST @MODEL_STAGE/checkpoints/;

-- Optional one-call training convenience wrapper:
-- CALL run_training_pipeline();
