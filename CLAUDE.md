# TabPFN DemandModel — Claude Persistent Instruction Manual

This file is loaded automatically by Claude Code at every session start. It describes the full
architecture, coding standards, pipeline design, and operational guardrails for this project.
Do not delete or rename this file.

---

## 1. Project Overview

TabPFN DemandModel is a MODEL3-ICL tabular regression model trained and evaluated entirely within
Snowflake Snowpark Container Services (SPCS). The model is trained on synthetic linear-regression
priors and benchmarked against ten classical baselines plus AutoGluon on both an in-distribution
synthetic suite and an OOD parity pilot suite. The primary deliverables are model comparison
summaries (MODEL3-ICL vs. baselines on MSE, rank, win rate) and stability charts demonstrating
MODEL3-ICL's robustness to feature noise and consistent performance across training set sizes.

---

## 2. Repository Layout

```
src/                    Core Python modules (model, train, evaluate, snowflake_io, etc.)
scripts/                MLJob submission scripts and orchestration procedures
scripts/ood_regression/ OOD data generation and indexing scripts (staged flat to @MODEL_STAGE/scripts/)
tests/                  pytest test suite (mocks Snowflake; no live connections)
docs/                   Reference documentation (Snowflake_Training.md, regression_evaluation.md)
sql/                    SQL DDL, training job SQL, result retrieval helpers
data/                   Local output of generate_synthetic_regression.py and generate_ood_eval_data.py
results/                Local downloaded evaluation artifacts
```

---

## 3. Tech Stack & Runtime

- **Python 3.11** — Snowflake managed runtime image `2.5.0-py311`
- **Snowflake Snowpark + SPCS MLJob API** — `submit_from_stage`, `PyTorchDistributor`
- **PyTorch** — MODEL3-ICL model; checkpoint v4 format
- **Ray Tune** — HPO: 20 trials, 5 nodes, 4 concurrent/node (= 20 one-GPU trial slots)
- **scikit-learn, XGBoost, LightGBM** — preinstalled in `2.5.0-py311`
- **CatBoost `1.2.10`** — NOT preinstalled; pip-installed per baseline shard job via `TABPFN_PYPI_EAI`
- **AutoGluon `1.3.0`** — NOT preinstalled; pip-installed per AutoGluon shard job via `TABPFN_PYPI_EAI`
- **openml `0.15.1`** — NOT preinstalled; pip-installed for the prep job only via `TABPFN_PYPI_EAI`

**Compute pools:**

| Pool | Nodes | GPU | Purpose |
|------|-------|-----|---------|
| `DEEPSET_GPU_POOL` | MAX=10, GPU_NV_M | 4×A10G/node | Training, HPO, MODEL3-ICL eval shards |
| `DEEPSET_CPU_POOL` | MAX=6, CPU_X64_M | — | Prep, baseline shards (6), aggregation |
| `AUTOGLUON_CPU_POOL` | MAX=6, CPU_X64_M | — | AutoGluon cluster shards (6×4 workers, `ray_work_items` mode) |

**External access:**
- `TABPFN_PYPI_EAI` — single EAI for all pip installs (CatBoost, AutoGluon, openml)
- `BENCHMARK_EXTERNAL_ACCESS` — OpenML/Kaggle API network egress (prep job only)

---

## 4. Snowflake Stage Ownership

| Stage | Contents |
|-------|----------|
| `@META_DATASET_STAGE` | Benchmark metadata index + prepared `.npz` splits (`benchmark_prepared/`) |
| `@MODEL_STAGE/scripts/` | All runnable MLJob code from `src/*.py` and `scripts/*.py` (flat, no subdirectories) |
| `@MODEL_STAGE/hpo/` | `best_config_ridge_residual.json` (sweep 1), `best_config_architecture.json` (sweep 2), `best_config.json` (merged final); `hpo_failure.json` on failure |
| `@MODEL_STAGE/checkpoints/` | `pretrain_gate32.pt`, `pretrain_gate64.pt`, `pretrain_gate128.pt` (one per gate candidate), `best.pt` (v4 format) |
| `@EVALUATION_RESULTS_STAGE` | All evaluation output CSVs, charts, manifests |
| `@EVALUATION_RESULTS_STAGE/regression/{suite_id}/` | Synthetic regression shard part CSVs (path scoped by `SYNREG_RESULTS_STAGE` env var) |
| `@EVALUATION_RESULTS_STAGE/ood_parity/` | OOD pilot MODEL3-ICL-only shard part CSVs |
| `@EVALUATION_RESULTS_STAGE/ood_full/` | OOD full suite (ood_linear_full_v1) aggregation outputs |
| `@MLJOB_PAYLOAD_STAGE` | Ephemeral MLJob payloads (managed by `submit_from_stage`) |
| `@EVALUATION_DATASET_STAGE/primary/` | In-distribution synthetic parquet files (200 datasets) |
| `@EVALUATION_DATASET_STAGE/ood_parity/{E,F,G,H}/` | OOD parity source pool — 200 parquet files (50 per regime); serves both pilot (80-row subset) and full suite (all 200 rows) |
| `@EPOCH_STAGE` | Epoch calibration artifacts (`hpo_timing.json`, `train_timing.json`) |

**Invariants:**
- Never embed suite IDs, version strings, or method names in stage subdirectory paths (exception:
  `{suite_id}` in `SYNREG_RESULTS_STAGE`, which is an env var value, not a hardcoded string).
- Never use `AUTO_COMPRESS=TRUE` for PUT commands — causes silent read failures.
- `@META_DATASET_STAGE` must never be read or written by any OOD script.
- `@EVALUATION_DATASET_STAGE` must never hold production training parquet.

---

## 5. Data Engineering Pipeline

### Step 1 — Metadata index (`build_meta_dataset_index.py`)

- Reads Kaggle/OpenML parquet files from `@META_DATASET_STAGE`
- Validates split counts: `train=800`, `val=100`, `test=100`
- Writes `META_DATASET_INDEX` Snowflake table
- Rebuild with `CALL build_meta_dataset_index();` whenever training parquet is restaged

### Step 2 — Benchmark preparation (`prepare_benchmark_datasets.py`)

- Reads `META_DATASET_INDEX`; fetches raw OpenML/Kaggle datasets
- Normalises features, writes `.npz` splits to `@META_DATASET_STAGE/benchmark_prepared/`
- Writes `benchmark_manifest.json`; idempotent (skips if valid manifest present)
- Use `BENCHMARK_FORCE_REBUILD=true` to force a full reprepare

### Step 3 — Local synthetic data generation (run locally before staging)

Both scripts run **locally** and produce parquet files that are PUT to Snowflake stages before
the Snowflake prep jobs run. Their rows are indexed into the **same**
`SYNTHETIC_REGRESSION_DATASET_INDEX` table, differentiated by `suite_id`.

**`scripts/generate_synthetic_regression.py`** — in-distribution primary suite
- Generates 200 parquet files across 4 regimes (A/B/C/D, 50 per regime)
- Default `suite_id=linear_poisson_v1_recommended`
- Writes `generated_locally=true, format="parquet"` manifest
- Output: `data/primary/`; stage: `@EVALUATION_DATASET_STAGE/primary/`
- Extended NPZ suites (`feature_noise`, `training_size`, `target_noise`) are generated
  inside `prepare_synthetic_regression.py` in Snowflake, not here

**`scripts/ood_regression/generate_ood_eval_data.py`** — OOD parity suite
- Generates **200** parquet files (50/regime); source pool for both pilot (80 indexed) and full
  suite (200 indexed); invoke with `--n_datasets 200`
- Output: `data/ood_regression/{E,F,G,H}/`; stage: `@EVALUATION_DATASET_STAGE/ood_parity/{E,F,G,H}/`
- Also prints required SnowSQL PUT commands
- `generate_ood_eval_data.py` is a local-only CLI and must **never** be staged to Snowflake

### Step 4 — Synthetic regression indexing (`prepare_synthetic_regression.py` — Snowflake job)

- Detects `generated_locally=true, format="parquet"` manifest; validates via `_validate_parquet_index()`
- Inserts in-distribution rows into `SYNTHETIC_REGRESSION_DATASET_INDEX` with
  `suite_id=linear_poisson_v1_recommended`
- Also generates and indexes optional extended NPZ suites if configured
- Idempotent: skips rebuild if valid; force rebuild with `SYNTHETIC_REGRESSION_FORCE_REBUILD=true`
- `DELETE WHERE suite_id=...` by default (preserves OOD rows); `DROP TABLE` only when
  `SYNTHETIC_REGRESSION_DROP_INDEX_TABLE=true` is explicitly set
- `_assert_index_populated()` is called after every `return` — zero rows is always fatal

### Step 5a — OOD pilot indexing (`prepare_ood_regression.py` — Snowflake job, pilot)

- Reads OOD manifest from `@EVALUATION_DATASET_STAGE/ood_parity/ood_manifest.json`
- Invoked with `OOD_REGRESSION_N_DATASETS=80` (preferred; `OOD_REGRESSION_N_PILOT=80` as legacy
  fallback): inserts 80 OOD rows (20 per regime E/F/G/H) under `suite_id=ood_linear_pilot_v1`
- Uses `DELETE WHERE suite_id=ood_linear_pilot_v1` for its own truncation; never touches
  in-distribution rows
- `_submit_ood_prep` always passes both `OOD_REGRESSION_N_DATASETS` and `OOD_REGRESSION_N_PILOT`

### Step 5b — OOD full suite indexing (`prepare_ood_regression.py` — Snowflake job, full suite)

- Same script as Step 5a, invoked with `OOD_REGRESSION_N_DATASETS=200`
- Inserts all 200 OOD rows (50 per regime E/F/G/H) under `suite_id=ood_linear_full_v1`
- Uses `DELETE WHERE suite_id=ood_linear_full_v1`; never touches pilot or in-distribution rows

### Step 5c — Combined suite indexing (`prepare_synthetic_regression.py` — Snowflake job, combined)

- Triggered by setting `SYNTHETIC_REGRESSION_SUITE_ID=linear_all_v1`; script detects this and
  calls `prepare_combined_suite()` instead of the normal generation flow
- **Prerequisites**: both `linear_poisson_v1_recommended` (Step 3) and `ood_linear_full_v1` (Step 5b)
  must already be indexed
- Loads `suite_family=primary` rows from `linear_poisson_v1_recommended` (200 rows, regimes A/B/C/D)
  and all rows from `ood_linear_full_v1` (200 rows, regimes E/F/G/H)
- Remaps `suite_id → linear_all_v1`, records `source_suite_id` for lineage,
  normalizes `split_seeds → [0, 1, 2]`, rebuilds `logical_dataset_key`
- Validates exactly 400 rows and 50 per regime A–H before inserting
- No parquet files are modified; stage paths are preserved from source rows
- Orchestrated by `run_synthetic_regression_combined_evaluation()`; aggregation outputs to
  `@EVALUATION_RESULTS_STAGE/combined/`

**Shared index architecture:** `SYNTHETIC_REGRESSION_DATASET_INDEX` is the single source of
truth for all synthetic evaluation suites. The evaluation procedure `load_synthetic_regression_index()`
always filters `WHERE suite_id = SYNREG_SUITE_ID` — controlling which suite runs by setting the
`SYNTHETIC_REGRESSION_SUITE_ID` env var before launching evaluation shards.

---

## 6. Model Architecture (MODEL3 — DeepSetICLModel)

- **Bounded-context ensemble**: 5 random context windows × 200 training rows per window
- **MC dropout**: K=8 forward passes per (sample, window)
- **Feature selection**: train-only `f_regression`, hard cap at 128 features
  (`SYNTHETIC_REGRESSION_DEEPSET_FEATURE_CAP`; `BENCHMARK_DEEPSET_FEATURE_CAP`)
- **Test batch size**: 128 rows (`SYNTHETIC_REGRESSION_TEST_BATCH_SIZE`)
- **Output**: mean prediction across 5×8=40 forward passes
- Selection is train-only: fit on `(X_train, y_train)`, applied to both train and holdout
- Baselines and AutoGluon receive full un-capped feature matrices
- Reference: `src/model.py`, `src/evaluate_synthetic_regression.py`

**Gated Ridge Expert (enabled by default):**
- `use_ridge_expert=True` by default in `ModelConfig`; prediction form: `ridge + gate × neural`
- `ridge_lambda=1.0` (default); tuned via HPO as `tune.loguniform(1e-3, 1e1)`
- `gate_hidden_dim ∈ {32, 64, 128}` — tuned by HPO (`tune.choice`); each candidate requires its own `pretrain_gate<N>.pt`
- `RidgeExpert` is stateless — zero trainable parameters; no `nn.Parameter` or `nn.Module`
- Gate sigmoid output in `(0, 1)`; gate MLP trains via gradient descent; ridge path is analytic
- `best_config.json` always includes `use_ridge_expert`, `ridge_lambda`, `gate_hidden_dim`
- Architecture mismatch check (in both `hpo.py` and `train.py`) includes `use_ridge_expert` and
  `gate_hidden_dim`; a pretrain checkpoint trained with ridge expert disabled will not warm-start a
  ridge-expert-enabled model (mismatch detected, training starts from scratch)

**Permutation test gate:** `run_permutation_tests(model)` is called immediately after
checkpoint load. If any of the 7 permutation-invariance tests fail, evaluation aborts with
a `RuntimeError`. This gate ensures the checkpoint actually implements MODEL3-ICL invariance.

---

## 7. Snowflake Training & Fine-Tuning

Three-phase pipeline, each submitted as an MLJob:

### Phase A — Pretraining (`run_pretrain_job.py` → `src/train.py`)

- Topology: 10 nodes × 4 workers = world size 40
- **Gate-specific pretrains required**: run one job per `gate_hidden_dim` candidate
  (32, 64, 128). Each writes `@MODEL_STAGE/checkpoints/pretrain_gate<N>.pt`.
- The 0-arg `run_pretrain_pipeline()` and 3-arg form are kept for backward compat
  (write `pretrain.pt`) but are not used by the HPO-based production flow.
- All three gate checkpoints must exist before HPO starts; HPO fails hard if any are missing.

### Phase B — HPO (`run_hpo_job.py` → `src/hpo.py`)

- Ray Tune on `DEEPSET_GPU_POOL`, 5 nodes, 20 trials, 4 concurrent/node
- All Snowflake stage access and data materialisation happen in the **driver only** before
  `tune.run()`. Ray workers must consume payloads via Ray object store — never open a
  Snowpark session inside a trial worker.
- Report metrics as `tune.report({"val_mse": value})` (dict-style); never keyword-style
- `tune.run(metric="val_mse", mode="min")`; `best_config.json` keys: `lr, weight_decay,
  d_phi, d_rho, dropout, pool, n_sab_feat, use_ridge_expert, ridge_lambda, gate_hidden_dim,
  use_huber, huber_delta, lambda_l1, hpo_sweep_mode`
- `HPO_SWEEP_MODE` controls which search space is used (see HPO sweep strategy section below)
- Writes `@MODEL_STAGE/hpo/best_config.json`; inspect `hpo_failure.json` first on failure
- Do not fetch Ray checkpoints directly; read only via Snowflake stage path

### Phase C — Final training (`run_model_training_job.py` → `src/train.py`)

- Topology: 10 nodes × 4 workers = world size 40
- `EXPECTED_TRAIN_WORLD_SIZE=40`, `STRICT_WORLD_SIZE_CHECK=true`
- SQL-sharded by DDP rank: `MOD(ROW_NUMBER() OVER (PARTITION BY split ORDER BY task_id) - 1, world_size) = rank`
- Warm-starts from the HPO-selected gate checkpoint; writes `@MODEL_STAGE/checkpoints/best.pt` (v4 format)
- Pretrain checkpoint is resolved strictly (no cold-start, no `pretrain.pt` fallback):
  1. `best_config._meta.pretrain_checkpoint_stage_path` (written by HPO)
  2. Fallback: `@MODEL_STAGE/checkpoints/pretrain_gate<gate_hidden_dim>.pt`
  3. `FileNotFoundError` before `submit_from_stage()` if neither exists
- Checkpoint loading: always use `load_checkpoint_compat()` with three fallback paths:
  1. `weights_only=True` (preferred, v4 checkpoints)
  2. `safe_globals([ModelConfig]) + weights_only=True` (legacy pickled cfg)
  3. `weights_only=False` (only if `ALLOW_UNSAFE_TORCH_LOAD=true`)
- `ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS` is currently `"true"` as a temporary
  escape hatch. Revert to `"false"` only after running `scripts/migrate_checkpoint.py`
  and verifying no `[SECURITY WARNING]` log lines appear.

**Checkpoint v4 format (canonical):**
```python
{
    "checkpoint_format_version": 4,
    "cfg": dataclasses.asdict(model.cfg),   # plain dict, NOT ModelConfig instance
    "state_dict": model.state_dict(),
    "metadata": {"source": "train.py", "pytorch_version": torch.__version__, ...},
}
```

**Failure diagnosis sequence (do not skip):**
1. No `[train.py main] entered main` → failure before train.py main
2. No `[train.py main] starting PyTorchDistributor.run` → failure inside main setup
3. No `[train_fn] entered train_fn` → failure in PyTorchDistributor/Ray worker launch
4. No `[train_fn] topology:` → failure around `get_context()`
5. Topology present → diagnose from actual train_fn error

If `Failed to mmap`, `activeQueryTracker`, or `Unable to create mmap-ed active query log`
appear in logs **and** Python boundary markers are absent, treat as Snowflake MLJob/Ray/
Prometheus runtime startup failure — do not rewrite model code.

### HPO sweep strategy for MODEL3-ICL with Ridge Expert

**Recommended: two-sweep strategy (`ridge_residual` → `architecture`).**

Sweep-specific outputs: `best_config_ridge_residual.json`, `best_config_architecture.json`.
Merged final: `best_config.json` (written by architecture sweep).
`_meta.sweeps.ridge_residual` and `_meta.sweeps.architecture` record both sweep results.
`HPO_BASELINE_CONFIG_STAGE_PATH` must be set to `@MODEL_STAGE/hpo/best_config_ridge_residual.json` when calling architecture sweep.
Run `model_ddp_memory_probe` (worst-case `d_phi=256, n_blocks=2`) as a **mandatory gate** before architecture HPO.

| Mode | Architecture | Tuned hyperparameters | Requires |
|------|-------------|----------------------|----------|
| `ridge_residual` (default) | Fixed: `d_phi=128`, `n_sab_feat=1` | `lr`, `weight_decay`, `dropout`, `ridge_lambda`, `gate_hidden_dim ∈ {32,64,128}`, `use_huber`, `huber_delta`, `lambda_l1` | Gate-specific pretrain checkpoints |
| `architecture` | `d_phi ∈ {64,128,192,256}`, `n_sab_feat ∈ {1,2}` | Frozen from ridge_residual baseline via `HPO_BASELINE_CONFIG_STAGE_PATH` | Ridge_residual sweep + memory probe |

**Gate-specific pretrain requirement (ridge_residual):**
HPO samples `gate_hidden_dim ∈ {32, 64, 128}`. Each trial must warm-start from its matching
`pretrain_gate<N>.pt`. All three checkpoints must exist before HPO starts. Missing any one
checkpoint causes HPO to fail with `FileNotFoundError` (driver-side, before Ray init).

**Architecture sweep cold-start policy:**
Gates checkpoints are optional for architecture sweep. If missing, all trials cold-start
(`PRETRAIN_LOAD_POLICY=allow_cold_start_on_arch_mismatch`).

**Pretrain checkpoint mismatch policy** (`PRETRAIN_LOAD_POLICY`):
- `ridge_residual` result → `require_match` (gate checkpoints must exactly match trial architecture)
- `architecture` result → `allow_cold_start_on_arch_mismatch` (set automatically by `run_model_training_job.py` based on `best_config.hpo_sweep_mode`)

**Fixed across all sweep spaces:**
- `d_rho=256` — not wired into the active forward pass. Change only after wiring `d_rho`.
- `pool="pna"` — `SetPool` not called in the current ICL forward path.
- `use_ridge_expert=True` — fixed `True`.

**Operational order (two-sweep, recommended):**
1. Run gate pretrains: `CALL run_pretrain_pipeline(..., 32/64/128)`.
2. Run `CALL run_hpo_pipeline(..., 'ridge_residual', '')`. Writes `best_config_ridge_residual.json`.
3. Run memory probe: `CALL run_model_ddp_memory_probe('model3', ..., 256, 2, TRUE)`.
4. Run `CALL run_hpo_pipeline(..., 'architecture', '@MODEL_STAGE/hpo/best_config_ridge_residual.json')`. Writes merged `best_config.json`.
5. Run `CALL run_model_training()`. Reads `best_config.json`; `PRETRAIN_LOAD_POLICY` set from `hpo_sweep_mode`.

**SQL overloads:**
```sql
-- Sweep 1: ridge_residual
CALL run_hpo_pipeline(
    'market_exchangeable_icl', 'synthetic_regression_combined',
    'inductive_forecasting', 'ridge_residual', ''
);

-- Memory probe (mandatory before architecture sweep):
CALL run_model_ddp_memory_probe(
    'model3', 'inductive_forecasting', 'market_exchangeable_icl',
    200, 128, 128, 256, 2, TRUE
);

-- Sweep 2: architecture (requires best_config_ridge_residual.json)
CALL run_hpo_pipeline(
    'market_exchangeable_icl', 'synthetic_regression_combined',
    'inductive_forecasting', 'architecture',
    '@MODEL_STAGE/hpo/best_config_ridge_residual.json'
);
```

---

## 8. Benchmark Evaluation Architecture

Orchestrated by `scripts/run_evaluation_test.py`. Use skill `evaluation-pipeline` when modifying or validating this pipeline.

### Phase sequence (phase-gated):

| Phase | Description |
|-------|-------------|
| 0 | Runtime probes — serialised (one at a time); validate each environment |
| 1+2 | Capacity probes — GPU → CPU → AutoGluon (non-overlapping) |
| 3 | Prep — 1 CPU job (`prepare_benchmark_datasets.py`) |
| 4 | MODEL3-ICL GPU shards — 10 shards, `DEEPSET_GPU_POOL` |
| 5 | Baseline CPU shards — 6 shards, `DEEPSET_CPU_POOL`; `catboost==1.2.10` + EAI |
| 6 | AutoGluon shards — 6 cluster shards × 4 workers (`ray_work_items` mode), `AUTOGLUON_CPU_POOL`; `autogluon.tabular==1.3.0` + EAI |
| 7 | Aggregation — 1 CPU job |

**Shard assignment:** `row_index % num_shards` — order must be deterministic.

**Part file pattern:** `{method}_shard{i}_of_{n}_detailed.csv` in
`@EVALUATION_RESULTS_STAGE/benchmark/`.

**pip / EAI allowlist:**

| Job | pip_requirements | external_access_integrations |
|-----|-----------------|------------------------------|
| GPU benchmark probe (0), CPU benchmark probe (1) | none | none |
| CPU baseline probe (2) | `catboost==1.2.10` | `TABPFN_PYPI_EAI` |
| Prep CPU probe (3) | `openml==0.15.1` | `TABPFN_PYPI_EAI` |
| AutoGluon CPU probe (4) | `autogluon.tabular==1.3.0` | `TABPFN_PYPI_EAI` |
| MODEL3-ICL GPU shards | none | none |
| Baseline shard jobs (×3) | `catboost==1.2.10` | `TABPFN_PYPI_EAI` |
| AutoGluon shard jobs (×30) | `autogluon.tabular==1.3.0` | `TABPFN_PYPI_EAI` |
| Aggregate job | none | none |
| Prep ML Job | `openml==0.15.1` | `[BENCHMARK_EXTERNAL_ACCESS, TABPFN_PYPI_EAI]` |

**Split-phase evaluation design:** 5 independent stored procedures expose each benchmark
phase: `run_evaluation_prep`, `run_deepset_evaluation`, `run_baseline_evaluation`,
`run_autogluon_evaluation`, `run_evaluation_aggregation`. After completing a phase, issue
`ALTER COMPUTE POOL <pool> SUSPEND` before calling the next phase (tight quota).

**Canonical benchmark outputs:**
- `@EVALUATION_RESULTS_STAGE/model_comparison.csv`
- `@EVALUATION_RESULTS_STAGE/model_comparison_summary.csv`

---

## 9. Synthetic Regression Evaluation Architecture

Orchestrated by `scripts/run_synthetic_regression_evaluation.py`. Use skill `evaluation-pipeline` when modifying or validating this pipeline.

### Suite definitions:

| Suite | `suite_id` | Datasets | Seeds | Regimes | Methods |
|-------|-----------|----------|-------|---------|---------|
| In-distribution primary | `linear_poisson_v1_recommended` | 200 | [0–4] | A/B/C/D | MODEL3-ICL + 10 baselines + AutoGluon |
| OOD parity pilot | `ood_linear_pilot_v1` | 80 | [0–2] | E/F/G/H | MODEL3-ICL only |
| OOD full suite | `ood_linear_full_v1` | 200 | [0–2] | E/F/G/H | MODEL3-ICL + 10 baselines + AutoGluon |
| Combined (primary + OOD) | `linear_all_v1` | 400 | [0–2] | A/B/C/D/E/F/G/H | MODEL3-ICL + 10 baselines + AutoGluon |
| Nonlinear | `nonlinear_v1` | 400 | [0–2] | I/J/K/L | MODEL3-ICL + 10 baselines + AutoGluon |
| Feature noise (NPZ) | same as primary | 80×6 | [0–2] | A/B/C/D | All |
| Training size (NPZ) | same as primary | 40×8 | [0–2] | A/B/C/D | All |
| Target noise (NPZ) | same as primary | 40×5 | [0–2] | A/B/C/D | All |

### Phase sequence (same pattern as benchmark):

| Phase | Description |
|-------|-------------|
| 1 | Runtime probes (serialised) |
| 2 | Capacity probes (GPU → CPU → AG, non-overlapping) |
| 3 | Prep — 1 CPU job (`prepare_synthetic_regression.py`) |
| 4 | MODEL3-ICL GPU shards (10, `DEEPSET_GPU_POOL`) — `SYNREG_RESULTS_STAGE=@EVALUATION_RESULTS_STAGE/regression/{suite_id}` |
| 5 | Baseline CPU shards (6, `DEEPSET_CPU_POOL`) — same `SYNREG_RESULTS_STAGE` |
| 6 | AutoGluon cluster shards (6×4 workers, `ray_work_items` mode, `AUTOGLUON_CPU_POOL`) — same `SYNREG_RESULTS_STAGE` |
| 7 | Aggregation (1 CPU job) — reads from suite-specific prefix; validates `suite_id` in all rows |
| 8 | OOD pilot runs as separate procedure: `run_synthetic_regression_ood_deepset_pilot` (only MODEL3-ICL, 5 GPU shards, results → `@EVALUATION_RESULTS_STAGE/ood_parity/`) |
| 9 | OOD full suite runs as separate procedure: `run_synthetic_regression_ood_full_evaluation` (prep + MODEL3-ICL + baselines + AutoGluon + aggregation for 200-dataset OOD suite; aggregation outputs → `SYNREG_OUTPUT_STAGE=@EVALUATION_RESULTS_STAGE/ood_full`) |
| 10 | Combined suite runs as separate procedure: `run_synthetic_regression_combined_evaluation` (combined prep + MODEL3-ICL + baselines + AutoGluon + aggregation for 400-dataset combined suite; aggregation outputs → `@EVALUATION_RESULTS_STAGE/combined`; requires both `linear_poisson_v1_recommended` and `ood_linear_full_v1` to be indexed first) |

### Split-phase stored procedures

All three suites expose individual phase procedures. Use these instead of the all-in-one wrappers when operating under tight node quota.

| Suite | Phase | Stored Procedure | Pool(s) Used |
|-------|-------|-----------------|-------------|
| Main (`linear_poisson_v1_recommended`) | Runtime probes | `run_synthetic_regression_runtime_probes` | GPU + CPU + AG |
| Main | Capacity probe | `run_synthetic_regression_capacity_probe` | GPU + CPU + AG |
| Main | Prep | `run_synthetic_regression_prep` | `DEEPSET_CPU_POOL` |
| Main | DeepSet eval | `run_synthetic_regression_deepset_evaluation` | `DEEPSET_GPU_POOL` |
| Main | Baseline eval | `run_synthetic_regression_baseline_evaluation` | `DEEPSET_CPU_POOL` |
| Main | AutoGluon eval | `run_synthetic_regression_autogluon_evaluation` | `AUTOGLUON_CPU_POOL` |
| Main | Aggregation | `run_synthetic_regression_aggregation` | `DEEPSET_CPU_POOL` |
| Main | All-in-one | `run_synthetic_regression_pipeline` | all three pools |
| OOD full (`ood_linear_full_v1`) | Prep | `run_synthetic_regression_ood_full_prep` | `DEEPSET_CPU_POOL` |
| OOD full | DeepSet eval | `run_synthetic_regression_ood_full_deepset_evaluation` | `DEEPSET_GPU_POOL` |
| OOD full | Baseline eval | `run_synthetic_regression_ood_full_baseline_evaluation` | `DEEPSET_CPU_POOL` |
| OOD full | AutoGluon eval | `run_synthetic_regression_ood_full_autogluon_evaluation` | `AUTOGLUON_CPU_POOL` |
| OOD full | Aggregation | `run_synthetic_regression_ood_full_aggregation` | `DEEPSET_CPU_POOL` |
| OOD full | All-in-one | `run_synthetic_regression_ood_full_evaluation` | all three pools |
| Combined (`linear_all_v1`) | Prep | `run_synthetic_regression_combined_prep` | `DEEPSET_CPU_POOL` |
| Combined | DeepSet eval | `run_synthetic_regression_combined_deepset_evaluation` | `DEEPSET_GPU_POOL` |
| Combined | Baseline eval | `run_synthetic_regression_combined_baseline_evaluation` | `DEEPSET_CPU_POOL` |
| Combined | AutoGluon eval | `run_synthetic_regression_combined_autogluon_evaluation` | `AUTOGLUON_CPU_POOL` |
| Combined | Aggregation | `run_synthetic_regression_combined_aggregation` | `DEEPSET_CPU_POOL` |
| Combined | All-in-one | `run_synthetic_regression_combined_evaluation` | all three pools |
| Nonlinear (`nonlinear_v1`) | Prep | `run_synthetic_nonlinear_prep` | `DEEPSET_CPU_POOL` |
| Nonlinear | DeepSet eval | `run_synthetic_nonlinear_deepset_evaluation` | `DEEPSET_GPU_POOL` |
| Nonlinear | Baseline eval | `run_synthetic_nonlinear_baseline_evaluation` | `DEEPSET_CPU_POOL` |
| Nonlinear | AutoGluon eval (SPCS) | `run_synthetic_nonlinear_autogluon_spcs_evaluation` | `AUTOGLUON_CPU_POOL` |
| Nonlinear | Aggregation | `run_synthetic_nonlinear_aggregation` | `DEEPSET_CPU_POOL` |

### Split-phase invocation pattern

Issue `ALTER COMPUTE POOL <pool> SUSPEND` after each compute-heavy phase to release quota before the next phase starts.

**Main pipeline (`linear_poisson_v1_recommended`):**
```sql
CALL run_synthetic_regression_prep('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_deepset_evaluation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_baseline_evaluation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_synthetic_regression_autogluon_evaluation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
CALL run_synthetic_regression_aggregation('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');
```

**OOD full suite (`ood_linear_full_v1`):**
```sql
CALL run_synthetic_regression_ood_full_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_ood_full_deepset_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_ood_full_baseline_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_synthetic_regression_ood_full_autogluon_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
CALL run_synthetic_regression_ood_full_aggregation('2.5.0-py311', '2.5.0-py311');
```

**Combined suite (`linear_all_v1`) — distributed AutoGluon default:**
```sql
-- Step 0: capacity probes (recommended before first run or after pool changes)
CALL run_synthetic_regression_combined_baseline_capacity_probe('2.5.0-py311', '2.5.0-py311', 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_autogluon_capacity_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

-- Step 1: combined split phases
CALL run_synthetic_regression_combined_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_combined_deepset_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_baseline_evaluation('2.5.0-py311', '2.5.0-py311', 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- 6 clusters x 4 workers = 24 concurrent CPU_X64_M nodes; 1 CPU per AutoGluon fit task
CALL run_synthetic_regression_combined_autogluon_capacity_probe(
  '2.5.0-py311', '2.5.0-py311', 6, 4, 6
);
CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
  '2.5.0-py311', '2.5.0-py311', 6, 4, 6
);
CALL run_synthetic_regression_combined_autogluon_evaluation(
  '2.5.0-py311', '2.5.0-py311', 6, 4, 1, 6, 300, 'best_quality'
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
-- Aggregation expects N=6 AutoGluon shard files (must match cluster_shards above)
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', '2.5.0-py311', 6);
```

**Nonlinear suite (`nonlinear_v1`) — SPCS AutoGluon backend:**
```sql
-- Prerequisites: generate + stage datasets first
-- python scripts/generate_nonlinear.py --n-datasets 400 --out-dir data/nonlinear_regression/
-- PUT data to @EVALUATION_DATASET_STAGE/nonlinear/{I,J,K,L}/ + manifest

CALL run_synthetic_nonlinear_prep('2.5.0-py311');
CALL run_synthetic_nonlinear_deepset_evaluation('2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_nonlinear_baseline_evaluation('2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
-- 6 clusters x 4 workers x 1 CPU/task; SYNREG_INDEX_TABLE injected automatically
CALL run_synthetic_nonlinear_autogluon_spcs_evaluation(
  '<AG_IMAGE>', 6, 4, 1, 6, 300, 'best_quality', 600, 1
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
CALL run_synthetic_nonlinear_aggregation('2.5.0-py311');
```

### Nonlinear suite data

- **Nonlinear training data:** 1000 parquet files (800/100/100 train/val/test split), regimes I–L,
  generated by `src/generate_nonlinear_dgp.py` → `data/nonlinear/{train,val,test}/`. Same
  single-row parquet schema as linear training data (X_train, y_train, X_test, betaX_test).
- **Nonlinear evaluation suite (`nonlinear_v1`, regimes I–L):** 400 datasets (100/regime):
  Quadratic (I), Sinusoidal (J), Pairwise Interactions (K), ReLU/Threshold (L). Same `(n,p)`
  Poisson sampling as primary suite. Separate index table: `SYNTHETIC_NONLINEAR_DATASET_INDEX`.
  Evaluation reads from this table when `SYNREG_INDEX_TABLE=SYNTHETIC_NONLINEAR_DATASET_INDEX`
  is set (configured automatically by `run_synthetic_nonlinear_evaluation.py` handlers).
  Generation: `python scripts/generate_nonlinear.py` → PUT → `CALL run_synthetic_nonlinear_prep(...)`.
  Results: `@EVALUATION_RESULTS_STAGE/nonlinear/`.
  SQL procedures: `sql/05_synthetic_nonlinear_evaluation_pipeline.sql`.

### Runtime-configurable baseline shard count:

Baseline shards are now runtime-configurable. Default remains 6.

| Variable / arg | Default | Description |
|----------------|---------|-------------|
| `SYNREG_BASELINE_SHARDS` (env) / `BASELINE_SHARDS` (SQL arg) | `SYNREG_CPU_SHARDS` = 6 | Number of baseline shard files written; must equal `BASELINE_CONCURRENT_NODES` |
| `SYNREG_BASELINE_CONCURRENT_NODES` / `BASELINE_CONCURRENT_NODES` | 6 | Required single-wave CPU nodes; must equal `BASELINE_SHARDS` |

**Baseline shard guardrails:**

- **1 baseline shard = 1 single-node MLJob = 1 output shard file.** Increasing `BASELINE_SHARDS` proportionally increases the output shard file count and required concurrent CPU nodes on `DEEPSET_CPU_POOL`.
- `BASELINE_CONCURRENT_NODES` must **equal** `BASELINE_SHARDS`. Lower values are rejected (no silent multi-wave batching). Higher values are also rejected unless `BASELINE_SHARDS` is increased to match.
- Aggregation must expect the same resolved baseline shard count (`SYNREG_EXPECTED_BASELINE_SHARDS`). The all-in-one `run_synthetic_regression_combined_evaluation` wires this automatically.
- Run capacity probe first: `run_synthetic_regression_combined_baseline_capacity_probe('2.5.0-py311', '2.5.0-py311', <BASELINE_SHARDS>, <BASELINE_CONCURRENT_NODES>)`.

### Combined AutoGluon execution modes (linear_all_v1):

Two supported modes are selected by `AUTOGLUON_CLUSTER_SHARDS` (SQL arg) /
`SYNREG_AUTOGLUON_CLUSTER_SHARDS` (env):

#### A. Ray distributed cluster-shard mode (`AUTOGLUON_CLUSTER_SHARDS > 0`)

**Default configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNREG_AUTOGLUON_CLUSTER_SHARDS` | 6 | Number of logical Ray cluster shards (> 0 selects this mode) |
| `SYNREG_AUTOGLUON_WORKERS_PER_SHARD` | 4 | `target_instances` per MLJob cluster |
| `AUTOGLUON_TASK_CPUS` | 1 | CPUs per individual AutoGluon fit |
| `SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS` | 6 | Max simultaneous MLJob clusters; must equal `CLUSTER_SHARDS` |
| `SYNREG_AUTOGLUON_DISTRIBUTED_MODE` | `ray_work_items` | Distribution strategy |
| `SYNREG_WORKER_DATA_ACCESS_MODE` | `driver_presigned_url` | Driver-derived presigned HTTPS URL; workers download via urllib without a Snowpark session |
| `SYNREG_MAX_WORK_ITEM_BYTES` | 8192 | Compact Ray item metadata size guard |

**Architecture guardrails:**

- AutoGluon is distributed across **independent work items**, not a single multi-node AutoGluon
  fit. Each MLJob cluster runs one logical shard; Ray distributes the shard's work items across
  the cluster's `target_instances` nodes.
- **SPCS custom-image backend (coordinator topology):** When `SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job`,
  each shard submits one coordinator SPCS service (`spcs_ray_coordinator.py`) and N worker SPCS
  services (`spcs_ray_worker.py`). For 6×4 this is **30 containers** (6 coordinators + 24 workers),
  not 36. The coordinator starts `ray start --head --num-cpus=0` locally, then runs `autogluon_ray.py`
  with `RAY_HEAD_ADDRESS=localhost:<port>`. Workers connect via the coordinator's external DNS address.
  **SPCS DNS rule:** underscores in the service name are replaced by dashes in DNS
  (e.g. `spcs_ray_coord_r0_0` → `spcs-ray-coord-r0-0.<suffix>`). The coordinator's SPCS spec
  exposes TCP endpoints for all required Ray ports: `ray-head` (6379), `ray-node-manager` (6380),
  `ray-object-manager` (6381), `ray-runtime-env-agent` (6382), and `ray-worker-ports` (portRange
  10002–10010). Worker specs expose all except `ray-head`. All ports must be deterministic (set via
  `--node-manager-port` etc.) and declared as SPCS TCP endpoints — SPCS only allows
  service-to-service traffic on declared endpoints. Do not add a separate head or driver service —
  the coordinator replaces both. Resource profile: `SYNREG_SPCS_RAY_COORDINATOR_*` (default 1/2 CPU, 4Gi/8Gi memory).
- **Session-free worker dataset loading:** The Ray driver opens Snowpark only to query
  `SYNTHETIC_REGRESSION_DATASET_INDEX` and derive `dataset_access.scoped_url` with
  `BUILD_SCOPED_FILE_URL`. Each Ray worker receives only a compact item dict and opens
  the scoped URL with `SnowflakeFile.open(scoped_url)`. Workers must not call
  `Session.builder.getOrCreate()`, must not query the index, and the driver must not
  call `ray.put(dataset)`. `MAX_IN_FLIGHT` bounds active worker-loaded AutoGluon fits.
- The driver process inside each MLJob writes exactly **one** output file:
  `AutoGluon_shard{i}_of_{N}_detailed.csv`. Worker Ray tasks never write stage files.
- Aggregation must expect `SYNREG_EXPECTED_AG_SHARDS=N` matching `SYNREG_AUTOGLUON_CLUSTER_SHARDS`.
  With the recommended default: `N=6`.
- **Always run capacity probes** (`run_synthetic_regression_combined_autogluon_capacity_probe`)
  before increasing `SYNREG_AUTOGLUON_WORKERS_PER_SHARD` or `SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS`.
  Default requests: 6 concurrent clusters × 4 workers = **24 CPU_X64_M nodes**.
- `autogluon_ray.py` **fails fast** if Ray cannot initialise across
  `target_instances > 1`. It does not silently fall back to single-node mode (which would write
  duplicate shard files from each instance).
- `CONCURRENT_CLUSTERS` must **equal** `CLUSTER_SHARDS` (single-wave enforcement). Lower values
  are rejected fast. Higher values are also rejected unless `CLUSTER_SHARDS` is increased to match.
- Aggregation does not launch AutoGluon workers. It only consumes completed shard CSVs.
  One Ray cluster maps to exactly one logical AutoGluon shard.
- The entrypoint (`autogluon_ray.py`) is **derived internally** from the mode. It is not
  accepted as a runtime argument or SQL procedure parameter.

#### B. Single-node shard mode (`AUTOGLUON_CLUSTER_SHARDS = 0`)

**Configuration:**

| Variable | Value | Description |
|----------|-------|-------------|
| `SYNREG_AUTOGLUON_CLUSTER_SHARDS` | **0** | Selects single-node mode |
| `SYNREG_AUTOGLUON_WORKERS_PER_SHARD` | 1 | Must be 1; each shard is one container |
| `SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS` | N | Number of shards submitted (= `output_shards`) |

- Each shard is a **single-instance** MLJob (`target_instances=1`) running
  `evaluate_synthetic_regression.py` with `mode=autogluon`.
- Output files follow the same naming convention:
  `AutoGluon_shard{i}_of_{N}_detailed.csv` where `N = CONCURRENT_CLUSTERS`.
- Aggregation must expect `SYNREG_EXPECTED_AG_SHARDS=N` matching `CONCURRENT_CLUSTERS`.
  `run_synthetic_regression_combined_evaluation` wires this automatically via
  `ag_plan.output_shards`.
- Capacity probes use `capacity_probe.py` with `target_instances=1` (not `ray_capacity_probe.py`).
- `WORKERS_PER_SHARD` must be **1** in this mode. Values > 1 are rejected by
  `_resolve_combined_autogluon_execution_plan` before any Snowflake call.
- The entrypoint (`evaluate_synthetic_regression.py`) is **derived internally** from the mode.
  It is not accepted as a runtime argument or SQL procedure parameter.

**Multi-instance entrypoint allowlist (both modes):**

`_submit_synreg` enforces an explicit allowlist for `target_instances > 1`. Only three entrypoints
are permitted: `autogluon_ray.py` (production distributed AutoGluon),
`ray_capacity_probe.py` (capacity probing), and `autogluon_worker_access_probe.py`
(worker-access probing). All other entrypoints — including
`evaluate_synthetic_regression.py` and `capacity_probe.py` — are rejected before the
Snowflake submit call. Do not weaken this guard.
- `MODEL_ARCH_VERSION` is **not** a Snowflake SQL/runtime selector. The model default is
  MODEL3/DeepSet ICL. Internal checkpoint metadata may contain `model_arch_version='model3'`
  for compatibility — do not remove that without updating all validation and tests.
- `run_model_training` accepts the same explicit runtime lineage variables as pretrain and HPO:
  `MODEL_FAMILY`, `TRAINING_DATA_FAMILY`, `MODEL_DESIGN_PATTERN`.

### Critical invariants:

- All three suites support split-phase execution. After each compute-heavy phase, issue
  `ALTER COMPUTE POOL <pool> SUSPEND` before the next phase to release quota.
- All-in-one wrappers (`run_synthetic_regression_pipeline`,
  `run_synthetic_regression_ood_full_evaluation`,
  `run_synthetic_regression_combined_evaluation`) call the individual phase functions
  internally — they hold all pools simultaneously. Use individual phases under tight quota.
- `SYNREG_RESULTS_STAGE` must always include `{suite_id}` in the path
- Index query must use `ORDER BY suite_id, suite_family, prior_regime, dataset_id,
  feature_noise_level, target_noise_scale, stage_path` for deterministic shard assignment
- Aggregation must validate all ingested rows have `suite_id == SYNREG_SUITE_ID`
- OOD shards must always set `SYNTHETIC_REGRESSION_MODE=deepset` explicitly
- `write_part_csv_to_stage()` must never upload a blank CSV — an empty `output_rows` is fatal
- `_wait_done` must check `job.status == "DONE"` after `job.wait()` returns
- Aggregation must NOT write to `model_comparison.csv` or `benchmark_parts/` — only to
  files with the `synthetic_regression_` prefix
- OOD prep always creates the index table before deleting: `create_synreg_index_table()` is
  called before `_truncate_ood_index()` to ensure the table exists before any DELETE
- Cache collision prevention: `load_prepared_synthetic_dataset` uses full stage path (not
  basename) to derive local cache filename — regime is embedded in the filename via
  `_stage_path_to_local_name()`; e.g. `ood_parity__E__dataset_0000.parquet`
- `logical_dataset_key` is persisted in `SYNTHETIC_REGRESSION_DATASET_INDEX`; zero-padded
  format `{suite_id}:{regime}:{dataset_id:04d}` everywhere; evaluation falls back to
  zero-padded format when the index row has no value
- All-method suites (`linear_poisson_v1_recommended`, `ood_linear_full_v1`, `linear_all_v1`)
  enforce shard completeness when `SYNREG_EXPECTED_DEEPSET_SHARDS` > 0 in the aggregation job;
  a missing MODEL3-ICL/baseline/AutoGluon shard raises RuntimeError before aggregation writes any output
- Combined suite (`linear_all_v1`) is an index-level composition only — no parquet files are merged
  or rewritten; `prepare_combined_suite()` copies rows from the primary in-distribution suite
  (family=`primary`) and the full OOD suite, remapping `suite_id`, `source_suite_id`, `split_seeds`,
  and `logical_dataset_key`; both source suites must be indexed before combined prep runs
- `source_suite_id` column in `SYNTHETIC_REGRESSION_DATASET_INDEX` records which source suite
  contributed each row in a combined suite (NULL for primary/OOD suites); migration:
  `ALTER TABLE SYNTHETIC_REGRESSION_DATASET_INDEX ADD COLUMN IF NOT EXISTS source_suite_id STRING;`
- `OOD_REGRESSION_N_DATASETS` is the canonical env var for OOD dataset count;
  `OOD_REGRESSION_N_PILOT` is kept as a backward-compatible fallback; `_submit_ood_prep`
  passes both env vars so old and new versions of the script both work

---

## 10. Data Generation Process (DGP)

Root RNG: `np.random.default_rng(seed=20260512)` — **never change this seed**.

All metrics are computed against `betaX` (noiseless signal), not noisy `y`.

### In-distribution regimes:

| Regime | Features | Coefficients | Noise | Notes |
|--------|----------|-------------|-------|-------|
| A | N(0,1) iid | N(0,1) | N(0,1) | Standard |
| B | N(0,1) iid | 70% sparse, N(0,2) | N(0,1) | Sparse signal |
| C | N(0,1) iid | N(0,1) | t(3) | Heavy-tailed noise |
| D | AR(1) ρ=0.6 | N(0,1) | N(0,1) | Correlated features |

### OOD regimes (E/F/G/H):

| Regime | Description |
|--------|-------------|
| E–H | Unseen at training time — parity pilot to test OOD generalisation |

---

## 11. Expected Pipeline Outputs

All final files written to `@EVALUATION_RESULTS_STAGE/`.

### Primary comparison outputs (always required):

| File | Description |
|------|-------------|
| `synthetic_regression_model_comparison.csv` | Full results: one row per (method, dataset, split_seed, condition) with rank and ratio columns |
| `synthetic_regression_model_comparison_summary.csv` | One row per method: mean/median MSE, win rates, ratio medians — **the top-level summary** |

### Breakdown outputs (required for analysis):

| File | Description |
|------|-------------|
| `synthetic_regression_summary_by_regime.csv` | MSE breakdown by (suite_family, prior_regime, method) |
| `synthetic_regression_summary_by_feature_noise.csv` | Noise robustness: `mse_degradation_vs_noise0`, `relative_mse_degradation_pct` per noise level |
| `synthetic_regression_summary_by_training_size.csv` | Sample efficiency: `mse_improvement_vs_25`, `relative_mse_improvement_pct` per n_train |

### Chart-ready data (required for visualisation):

| File | Description |
|------|-------------|
| `synthetic_regression_chart_data_noise_features.csv` | x=noise_level, y=relative_degradation — MODEL3-ICL stability vs. baselines |
| `synthetic_regression_chart_data_training_size.csv` | x=n_train, y=relative_improvement — MODEL3-ICL sample efficiency |
| `synthetic_regression_chart_data_model_rank.csv` | x=method, y=mean_rank — overall ranking |

The two stability charts (`noise_features` and `training_size`) are the primary visual
evidence for MODEL3-ICL's resistance to noise and consistent performance across training set sizes.

### OOD full suite outputs (written to `@EVALUATION_RESULTS_STAGE/ood_full/`):

**Required (always written):**

| File | Description |
|------|-------------|
| `synthetic_regression_model_comparison.csv` | Full results with rank and ratio columns |
| `synthetic_regression_model_comparison_summary.csv` | Per-method win rates, ratio medians |
| `synthetic_regression_summary_by_regime.csv` | MSE breakdown by (suite_family, prior_regime, method) |
| `synthetic_regression_chart_data_model_rank.csv` | Method ranking chart data |

**Conditional (written only when NPZ extended suite data is present):**

| File | Description |
|------|-------------|
| `synthetic_regression_summary_by_feature_noise.csv` | Noise robustness (absent for OOD suite) |
| `synthetic_regression_summary_by_training_size.csv` | Sample efficiency (absent for OOD suite) |
| `synthetic_regression_chart_data_noise_features.csv` | Noise chart data (absent for OOD suite) |
| `synthetic_regression_chart_data_training_size.csv` | Training size chart data (absent for OOD suite) |

### Manifests (always produced):

- `synthetic_regression_aggregation_manifest.json` — success: input shards, output files, validation flags
- `synthetic_regression_aggregation_failure.json` — failure: error_type, message, traceback, remediation

Always inspect `synthetic_regression_aggregation_failure.json` first when final CSVs are missing.
Use `sql/verify_synthetic_regression_outputs.sql` before `sql/get_synthetic_regression_outputs.sql`.
SnowSQL GET commands must use `PARALLEL = 4` (never `PARALLEL = 0`).

---

## 12. Coding Standards

- **New pipeline phases** follow prep → shards → aggregation. Always add `SYNREG_RESULTS_STAGE`
  to every shard job's `env_vars`; never hardcode stage paths in evaluation code.
- **Shard functions** are dataset-first: own a subset of index rows, loop seeds/methods,
  write part CSVs.
- **Index queries** always include a stable compound `ORDER BY`; never query without ordering.
- **pip requirements:** only baseline and AutoGluon jobs carry `pip_requirements` and
  `external_access_integrations`. MODEL3-ICL GPU jobs and the aggregation job carry neither.
- **All pip jobs** use the single `TABPFN_PYPI_EAI` integration.
- **Checkpoints:** always write v4 format (plain dict `cfg`); always load via
  `load_checkpoint_compat()`.
- **Tests:** mock Snowflake sessions with `MagicMock` + `side_effect`; use `patch` for env flags;
  never import Snowflake at module level in tests. Row mocks must support both `as_dict()` and
  string-key `r["name"]` access.
- **`_patch_submit` test helper** must patch `_submit_synreg`, `_ensure_compute_pool_usable`,
  `_wait_job_group`, `_wait_done`, and `_list_stage`.
- **Stage paths:** never embed suite IDs, versions, or method names in Snowflake stage
  subdirectory paths (exception: `{suite_id}` in `SYNREG_RESULTS_STAGE`, which is an env
  var value, not a hardcoded string).
- **`_truncate_synreg_index`:** default is `DELETE WHERE suite_id=...`; DROP TABLE only when
  `SYNTHETIC_REGRESSION_DROP_INDEX_TABLE=true` is explicitly set.
- **Lazy imports:** prefer method-specific lazy imports in `evaluate.py` so an unrelated
  model package failure does not break a shard for another method.
- **Entrypoint boundaries:** `evaluate_synthetic_regression.py` must never import from
  `evaluate.py`. Entrypoint scripts are runnable orchestration surfaces, not shared
  libraries; reusable MODEL3-ICL, baseline, AutoGluon, and ranking code belongs in stable
  `src/*.py` helper modules.
- **Script staging:** all scripts staged flat to `@MODEL_STAGE/scripts/` — including OOD
  scripts from `scripts/ood_regression/` (no subdirectories on stage).
- **PUT commands:** always use `AUTO_COMPRESS=FALSE OVERWRITE=TRUE`.
- **Snowflake row key normalisation:** Any `Row.as_dict()` result crossing from Snowflake
  into Python evaluator or indexing code must be normalised to lowercase keys immediately:
  `{str(k).lower(): v for k, v in row.as_dict().items()}`.
  Do not assume Snowflake preserves lowercase column names. This is mandatory for
  `SYNTHETIC_REGRESSION_DATASET_INDEX` rows because downstream evaluation code expects
  lowercase keys such as `stage_path`, `split_seeds`, `dataset_id`, `suite_id`,
  `suite_family`, and `prior_regime`.
- **Model family routing:** Never hardcode `DeepSetICLModel(cfg=cfg)` or
  `DeepSetCompletionModel(cfg=cfg)` directly. Always call `_instantiate_model(cfg)`
  from `model.py`. Routing is determined by `cfg.model_family`
  ("market_exchangeable_icl" | "market_exchangeable_completion").
- **`_checkpoint_architecture_mismatches` fields:** Must include `"model_family"` alongside
  the other 8 structural fields. A mismatch on `model_family` aborts warm-start.
- **Checkpoint version:** MODEL3 checkpoints always save `checkpoint_format_version=4`.
  Never save with version 2 or 3 (legacy MODEL2 values).

---

## 13. Operational Guardrails (Critical Rules)

### Data safety

- Never materialise full benchmark datasets locally inside an MLJob; use `@META_DATASET_STAGE` paths.
- Never use `AUTO_COMPRESS=TRUE` for PUT commands (causes silent read failures).
- OOD scripts never touch `@META_DATASET_STAGE`; only `@EVALUATION_DATASET_STAGE`.
- `data/ood_regression/` is the local directory; `@EVALUATION_DATASET_STAGE/ood_parity/` is the stage prefix.
- `@EVALUATION_DATASET_STAGE` must never hold production training parquet.

### Index safety

- `SYNTHETIC_REGRESSION_DATASET_INDEX` stores both in-distribution and OOD rows, differentiated by `suite_id`.
- Force rebuild must only delete `WHERE suite_id = '{active}'`; never DROP the whole table
  unless `SYNTHETIC_REGRESSION_DROP_INDEX_TABLE=true`.
- `_assert_index_populated()` is called after every insert; zero rows is always fatal.
- OOD truncation: `DELETE WHERE suite_id = 'ood_linear_pilot_v1'` only; never `TRUNCATE TABLE` or `DROP TABLE`.
- `create_or_validate_manifest()` must check that `SYNTHETIC_REGRESSION_DATASET_INDEX` has rows
  for the `suite_id` — not just that manifest JSON files exist on stage.

### Training topology

- World size = 10 nodes × 4 workers = 40. `STRICT_WORLD_SIZE_CHECK=true` enforces this.
- HPO: 5 nodes, 20 trials, 4 concurrent/node (`MAX_NODES=10`, never exceed).
- `run_model_training_job.py` must keep `target_instances` equal to `TRAIN_NUM_NODES`.

### Evaluation phase gating

- MODEL3-ICL shards must complete before baselines submit; baselines before AutoGluon;
  all three before aggregation.
- Runtime probes are serialised (one at a time; wait for each before next).
- Capacity probes are phase-gated: GPU → CPU → AG (non-overlapping).
- `target_instances=1` for every MLJob except Ray-coordinated multi-instance entrypoints.
  Only `autogluon_ray.py`, `ray_capacity_probe.py`, and
  `autogluon_worker_access_probe.py` are permitted with `target_instances > 1`.
  Any other entrypoint with `target_instances > 1` is rejected by `_submit_synreg`
  before the Snowflake call.
- `run_evaluation_pipeline()` is phase-gated; never collapse back into overlapping fan-out
  unless node quota has been raised and verified.

### Compute pool preflight

- Any job targeting `DEEPSET_CPU_POOL` or `DEEPSET_GPU_POOL` must call
  `_ensure_compute_pool_usable()` first (resumes SUSPENDED pools).
- For cost control: keep pools suspended between runs with `AUTO_RESUME=TRUE` and short
  `AUTO_SUSPEND_SECS` (60–300 s). Do not keep `DEEPSET_GPU_POOL` warm between sessions.

### Dependency pinning

| Package | Pinned version | Constant |
|---------|---------------|----------|
| catboost | `1.2.10` | `CATBOOST_VERSION` / `BASELINE_EXTRA_PIP_REQUIREMENTS` |
| autogluon.tabular | `1.3.0` | `AUTOGLUON_VERSION` / `AUTOGLUON_EXTRA_PIP_REQUIREMENTS` |
| ray | (unpinned) | `SYNREG_RAY_PIP`; Ray-mode AG jobs use `SYNREG_AG_RAY_PIP = SYNREG_AG_PIP + SYNREG_RAY_PIP` |
| openml | `0.15.1` | `OPENML_VERSION` / `PREP_EXTRA_PIP_REQUIREMENTS` |

- Always pin exact versions (`==`). Never use `>=` or unpinned requirements.
- MODEL3-ICL GPU jobs: no pip, no EAI.
- Aggregation job: no pip, no EAI.
- `autogluon.tabular` stays lazily imported inside `predict_autogluon()`.
- Ray-mode capacity probe, worker-access probe, and distributed AutoGluon evaluation jobs all
  carry `SYNREG_AG_RAY_PIP` + `SYNREG_PYPI_EAI`; single-node AutoGluon jobs carry
  `SYNREG_AG_PIP` only and no Ray env vars.
- Ray readiness is a runtime knob, not a hardcoded startup assumption. Capacity probe
  overloads can pass `RAY_READY_TIMEOUT_SECONDS` and `RAY_READY_POLL_SECONDS`
  after the topology arguments; default is 300s / 10s. Distributed AutoGluon
  evaluation can pass the same two values after `AUTOGLUON_PRESETS`; default is
  600s / 10s. The MLJob entrypoints receive them as
  `SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS` and
  `SYNREG_RAY_CLUSTER_READY_POLL_SECONDS`.
- **SPCS probe diagnostic env vars:**
  - `SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS` (default 900) — how long the capacity probe polls
    Ray for full cluster readiness before timing out.
  - `SYNREG_SPCS_WORKER_SUBMIT_STAGGER_SECONDS` (default 0) — sleep between SPCS worker job
    submissions; use 10s to test whether bursty scheduling causes partial worker join.
  - `SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE` (default false) — when true, worker support jobs
    are not cancelled after a coordinator failure, leaving them accessible for log inspection.
    Must be cancelled manually after diagnostics.
- **Driver failure artifact (`@EVALUATION_RESULTS_STAGE/debug`):** If the distributed
  AutoGluon evaluation fails after Ray readiness (task submission, result collection, or CSV
  write), the coordinator driver writes a JSON artifact to `@EVALUATION_RESULTS_STAGE/debug`.
  Fields include: traceback, run_id, shard_index, submitted/completed counts, cluster_resources,
  available_resources, and sanitized item metadata. Presigned/scoped URLs are always redacted.
  Retrieve with `LIST @EVALUATION_RESULTS_STAGE/debug;` then `GET`.
- **SPCS Ray object-store defaults (256 MiB):** Coordinator and worker default to `268435456`
  bytes Ray object-store memory. Datasets are fetched via presigned URLs — not through Ray
  object store. Override via `SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES` /
  `SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES` only if diagnostics prove it necessary.
  Do not increase to old 500 MB / 2 GB defaults without evidence of object-store pressure.
- **SPCS evaluation 9-arg overload (topology: 30 containers):** Use the new 9-argument overload
  to set readiness timeout and worker stagger at call time (no env setup required):
  ```sql
  CALL run_synthetic_regression_combined_autogluon_spcs_evaluation(
    $AUTOGLUON_IMAGE_REF,
    6, 4, 1, 6, 300, 'best_quality',
    600,  -- RAY_READY_TIMEOUT_SECONDS
    1     -- WORKER_SUBMIT_STAGGER_SECONDS
  );
  ```
  For 6×4 this submits **30 containers** (6 coordinators + 24 workers). `KEEP_SUPPORT_JOBS_ON_FAILURE`
  is intentionally not a SQL parameter — use `SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE` env var for
  diagnostics only.
- **SPCS Ray cancellation diagnostics:** Worker and coordinator wrapper scripts register
  SIGTERM/SIGINT handlers that emit structured JSON events (`spcs_ray_worker_signal_received`,
  `spcs_ray_coordinator_signal_received`) before exiting. Events include uptime, Ray PID,
  configured ports, run_id, shard_index. Ray log tails are dumped if present. Exit code is
  `128 + signal_number`. `ray_capacity_probe.py` emits JSON readiness and timeout events
  (`ray_capacity_probe_readiness`, `ray_capacity_probe_timeout`) via its `_log()` helper.
  Cancelled workers with no logs at all were terminated before Python started; those with
  signal logs were alive and cancelled externally.
- **Lazy Torch import in `evaluate_synthetic_regression.py`:** `torch`, `deepset_inference`,
  and `model` are imported lazily inside the DeepSet/checkpoint functions that need them
  (`safe_torch_load_with_legacy_escape_hatch`, `normalize_checkpoint_cfg`,
  `load_best_deepset_checkpoint`, `apply_deepset_feature_selection`,
  `run_deepset_synthetic_regression`). AutoGluon, worker-access, sharding, and aggregation
  paths are fully torch-free. Running a DeepSet path in a runtime without torch raises a
  clear `RuntimeError` via `_import_torch()`. Do not re-add `import torch` or
  `from deepset_inference import` / `from model import` at module level.

### HPO guardrails

- All Snowflake I/O happens in the HPO driver only; Ray workers use Ray object store.
- Never use keyword-style `tune.report(val_mse=value)` — use dict-style only.
- `TunerConfig` documented parameters only: `metric`, `mode`, `search_alg`, `num_trials`,
  `max_concurrent_trials`, `resource_per_trial`. Do not add undocumented kwargs.
- `resource_per_trial={"GPU": 1}` is mandatory — Snowflake does not allocate GPUs automatically.
- Do not call `scale_cluster()` from `hpo.py`; scaling is at submission time only.

### Checkpoint safety

- Never save `ModelConfig` objects directly in PyTorch checkpoints (breaks `weights_only=True`).
- Always serialize `cfg` as `dataclasses.asdict(model.cfg)`. Checkpoint format version
  is always `4` for MODEL3. Use `checkpoint_format_version = 4`.
- Consumers must normalise `ckpt["cfg"]` dict back to `ModelConfig` before comparing fields.
- `weights_only=False` can execute arbitrary code — only for internally trusted checkpoints;
  never for third-party checkpoints.
- Synthetic regression evaluation shards must set `ALLOW_UNSAFE_TORCH_LOAD=true`
  directly for `SYNTHETIC_REGRESSION_MODE` of `deepset` and `baselines` only.
  AutoGluon shards never call `torch.load` and must NOT receive this env var.
- `TORCH_UNLOAD=true` is only a compatibility alias inside the evaluator;
  orchestration must set `ALLOW_UNSAFE_TORCH_LOAD=true` directly. This exception
  is not a policy for third-party checkpoints.
- Migration: `python scripts/migrate_checkpoint.py --stage-name MODEL_STAGE --name best.pt`

### Script staging path invariant

- `prepare_ood_regression.py` must be staged **flat** to `@MODEL_STAGE/scripts/`.
- `generate_ood_eval_data.py` is local-only and must never be staged to Snowflake.
- Stage all scripts with:
  ```sql
  PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  ```
  New helper modules under `src/` must be covered by the `src/*.py` PUT so Snowflake
  MLJobs can import them from `@MODEL_STAGE/scripts/`.
- Current synthetic-evaluation upload reference after evaluator/helper changes:
  ```sql
  PUT file://C:/Documents/TabPFN_DemandModel/src/evaluate_synthetic_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/src/deepset_inference.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/src/baseline_models.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/src/autogluon_models.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/src/evaluation_metrics.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/src/evaluate.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/scripts/autogluon_ray.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/scripts/ray_capacity_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/scripts/autogluon_worker_access_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  PUT file://C:/Documents/TabPFN_DemandModel/scripts/autogluon_import_timing_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
  LIST @MODEL_STAGE/scripts/ PATTERN='.*(evaluate_synthetic_regression|deepset_inference|baseline_models|autogluon_models|evaluation_metrics|evaluate|run_synthetic_regression_evaluation|autogluon_ray|ray_capacity_probe|autogluon_worker_access_probe|autogluon_import_timing_probe)[.]py';
  ```
  Use the targeted block above when only the evaluator/helper refactor changed. Use
  the broad `src/*.py` plus `scripts/*.py` PUT block when procedure dependencies or
  multiple shared modules may have changed. `scripts/ood_regression/generate_ood_eval_data.py`
  remains local-only and must not be staged.
- `autogluon_ray.py`, `ray_capacity_probe.py`, and `autogluon_worker_access_probe.py`
  are the allowlisted multi-instance entrypoints; all must always be staged before running
  the combined AutoGluon evaluation, capacity probe, or worker-access probe procedures.

### OOD path invariants

- The pilot (`ood_linear_pilot_v1`, 80 datasets) is indexed with `OOD_REGRESSION_N_PILOT=80`
  via `_submit_ood_prep(session, OOD_PILOT_SUITE_ID, 80, bench_rt)`.
- The full suite (`ood_linear_full_v1`, 200 datasets) is indexed with `OOD_REGRESSION_N_PILOT=200`
  via `_submit_ood_prep(session, OOD_FULL_SUITE_ID, 200, bench_rt)`.
- Both use `prepare_ood_regression.py` via the parameterized `_submit_ood_prep(suite_id, n_datasets)`.
- `run_synthetic_regression_prep` never touches OOD indexing.
- `SYNREG_OUTPUT_STAGE` must always be set to `OOD_FULL_OUTPUT_STAGE` in the full suite
  aggregation job; omitting it would overwrite in-distribution results at `@EVALUATION_RESULTS_STAGE`.
- `SYNREG_CONDITIONAL_OUTPUTS` (4 noise/training-size CSVs) are written only when NPZ extended
  suite data is present — never raise on absent OOD conditional outputs.

### Kaggle download

- `401 Unauthorized` → recreate `KAGGLE_API_SECRET` with correct username/token.
- `403 Forbidden` → accept competition rules at kaggle.com while logged in as the API token's account.
- Always run `CALL download_kaggle_to_stage();` then `LIST @META_DATASET_STAGE/kaggle/;` to verify.

## Guardrail: Query Collapse and Early Feature Compression

The MODEL3-ICL model must not be evaluated solely through full benchmark aggregation before
passing architecture sanity checks.

Required pre-evaluation checks:
1. Query sensitivity ratio.
2. Sample permutation invariance.
3. Feature permutation consistency.
4. Duplicate-context stability.
5. Simple linear recovery smoke test.
6. Ridge diagnostic comparison.

The architecture must preserve feature identity through sample-level evidence aggregation
before pooling features.

Explicit linear/ridge inductive bias is allowed only as a modular prior expert or residual
baseline. It must not replace the neural market model. Future extensions must support sparse
cross-price effects, low-rank market structures, treatment effects, seasonality, and
market-demand priors.

## Guardrail: Defer HPO Expansion Until Architecture Sanity Passes

Do not broaden HPO over model width, pooling, SAB depth, or optimizer settings until the
corrected architecture passes query sanity checks. The first priority is to eliminate query
collapse and early feature compression. HPO expansion comes after the model demonstrates
query-sensitive behavior on controlled synthetic contexts.

## Guardrail: DeepSetICLModel (MODEL3) is the Only Supported Model

`DeepSetICLModel` is the sole production model for all synthetic regression
training, OOD synthetic regression, and future market mental model prior training.
Retired model classes have been removed from the codebase and must not be
reintroduced. Do not set `MODEL_FAMILY` to any retired value
("deepset", "market_aware"). `normalize_checkpoint_cfg()` raises `RuntimeError` for
checkpoints carrying retired family values.

## Guardrail: Synthetic Regression Evaluation Must Instantiate from Checkpoint cfg

`load_best_deepset_checkpoint()` in `evaluate_synthetic_regression.py` must always call
`_instantiate_model(cfg)` — never hardcode `DeepSetICLModel(cfg=cfg)` or
`DeepSetCompletionModel(cfg=cfg)`. The model class is determined by
`cfg.model_family` from the checkpoint.

## Guardrail: Device-Aware Sanity Checks

Sanity checks in `src/sanity_checks.py` must be called with the same CUDA device as training
or evaluation (`run_all_checks(model=model, device=torch.device("cuda:0"))`). Checks that
create tensors on CPU while the model is on CUDA will produce false positives for
device-placement bugs.

## Guardrail: Trained-Checkpoint Gates Must Pass Before DeepSet Synthetic Regression Evaluation

`SYNREG_RUN_CHECKPOINT_GATES=true` (default) and `SYNREG_CHECKPOINT_GATE_STRICT=true`
(default) are required for all DeepSet synthetic regression shards. Do not set
`SYNREG_CHECKPOINT_GATE_STRICT=false` in production without explicit justification. A
failing gate indicates model collapse (NaN/Inf, constant output, or severe Ridge
underperformance) that would invalidate evaluation results.

## Guardrail: Train-Time Sanity Gate Before DDP/compile

`TRAIN_RUN_SANITY_CHECKS=true` (default) runs structural checks on the model before
`torch.compile()` and `DistributedDataParallel` wrapping. `TRAIN_SANITY_CHECK_STRICT=true`
(default) raises `RuntimeError` on failure. Do not disable without explicit justification.

## Guardrail: run_permutation_tests() Is Architecture-Aware

`run_permutation_tests()` in `src/deepset_inference.py` dispatches to a
`market_exchangeable_icl` or `market_exchangeable_completion` branch based on
`cfg.model_family`. Never call it with a model whose `model_family` is absent or
unknown — it will raise `ValueError`. Tests for ICL cover row/column permutation
invariance, finite output, and batch-query shape. Completion tests cover row/column
equivariance and output shape (n, p). The function always restores the model's
original training mode.

## Guardrail: HPO Must Propagate model_family Into best_config.json

`src/hpo.py` defaults to `MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "market_exchangeable_icl")`.
The `model_family` key is written into every `best_config.json` uploaded to `@MODEL_STAGE/hpo/`.
`train.py` reads it via `hyper_params.get("model_family", MODEL_FAMILY)`. Do not allow
`best_config.json` to be uploaded without a `model_family` key.

## Guardrail: Checkpoint Metadata Includes Training Metrics

Checkpoints saved by `train.py` must include `best_val_mse`, `train_mse_at_best`, and
`best_epoch` in the `metadata` dict. These fields enable `check_train_val_gap()` in
`sanity_checks.py` to detect severe overfitting before evaluation. Legacy checkpoints without
these keys pass the gap gate silently (backward compatible).

## Guardrail: TRAINING_DATA_FAMILY Must Be Explicit in Production Submissions

All Snowflake production training submissions (pretrain, HPO, final training) must explicitly
set `TRAINING_DATA_FAMILY` in `env_vars`. Do not rely on `train.py` defaulting to `"unknown"`.

**Allowed values:**
- `synthetic_regression_combined` — combined suite `linear_all_v1` (primary + OOD). **Use for all checkpoints evaluated by synthetic regression evaluation procedures.**
- `synthetic_regression_primary` — primary in-distribution synthetic regression data only.
- `synthetic_regression_ood` — OOD synthetic regression data only.
- `market_mental_model` — market mental model prior training.
- `unknown` — local/dev/ad hoc only; never acceptable in production Snowflake submissions.

**Key rules:**
- `run_training_job.py`, `run_model_training_job.py`, and `run_hpo_job.py` each expose
  `DEFAULT_TRAINING_DATA_FAMILY = os.getenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")`.
- Pretrain and final training in `run_training_job.py` must use the **same** value.
- `train.py` validates the value at module load time and raises `ValueError` for unknown values.
- Checkpoint `metadata["training_data_family"]` is populated from `TRAINING_DATA_FAMILY`; it enables end-to-end lineage tracing from staged data → checkpoint → evaluation.
- Combined suite `linear_all_v1` = `synthetic_regression_combined`. Never default production
  synthetic regression evaluation checkpoints to `synthetic_regression_primary`.

## Guardrail: MODEL3 Architecture Selectors Must Be Propagated Explicitly

`MODEL_DESIGN_PATTERN` must be set explicitly in all production Snowflake training
submissions. `MODEL_FAMILY` defaults to `"market_exchangeable_icl"` (MODEL3 ICL).

**Key rules:**
- `MODEL_DESIGN_PATTERN="inductive_forecasting"` (default) → `model_family="market_exchangeable_icl"` → `DeepSetICLModel`
- `MODEL_DESIGN_PATTERN="transductive_completion"` requires `MODEL_FAMILY="market_exchangeable_completion"` → `DeepSetCompletionModel`
- MODEL3 checkpoints use `checkpoint_format_version=4` and must include `model_arch_version`,
  `model_design_pattern`, and `task_objective` in checkpoint metadata.
- `run_model_training_job.py`, `run_pretrain_job.py`, and `run_hpo_job.py` each expose
  `DEFAULT_MODEL_FAMILY` and `DEFAULT_MODEL_DESIGN_PATTERN` constants that propagate
  through `env_vars` to all MLJob children.

## Guardrail: Run MODEL3 DDP Memory Probe Before Training

Run `CALL run_model_ddp_memory_probe(...)` with `RUN_BACKWARD=TRUE` before every MODEL3
pretrain / HPO / final-training submission when shape parameters change.

**Key rules:**
- Always use `RUN_BACKWARD=TRUE`. MODEL3 meta-training uses back-propagation; a forward-only
  probe understates peak memory by the 20/8 = 2.5× activation-factor ratio.
- The static memory estimate `H_bytes = m * n * p * d_phi * 4` (float32) is computed before
  any tensor allocation. If `H_bytes * 20 * 1.5 > CUDA_total * 0.9`, the probe raises before
  OOM instead of crashing a running training job.
- Probe results are uploaded to `@MODEL_STAGE/diagnostics/model_ddp_memory_probe.json`.
  Read them with `SELECT $1 FROM @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json
  (FILE_FORMAT => (TYPE = JSON));`
- Probe handler is `run_model_training_job.run_model_ddp_memory_probe`. The procedure
  signature and handler are defined in `sql/run_training_job.sql`; the probe script is
  `scripts/model_ddp_memory_probe.py`.

**Example probe call:**
```sql
CALL run_model_ddp_memory_probe(
    'inductive_forecasting', 'market_exchangeable_icl',
    200, 128, 128, 128, 1, TRUE
);
```
