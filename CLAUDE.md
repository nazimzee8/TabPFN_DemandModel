# TabPFN DemandModel — Claude Persistent Instruction Manual

This file is loaded automatically by Claude Code at every session start. It describes the full
architecture, coding standards, pipeline design, and operational guardrails for this project.
Do not delete or rename this file.

---

## 1. Project Overview

TabPFN DemandModel is a DeepSet tabular regression model trained and evaluated entirely within
Snowflake Snowpark Container Services (SPCS). The model is trained on synthetic linear-regression
priors and benchmarked against ten classical baselines plus AutoGluon on both an in-distribution
synthetic suite and an OOD parity pilot suite. The primary deliverables are model comparison
summaries (DeepSet vs. baselines on MSE, rank, win rate) and stability charts demonstrating
DeepSet's robustness to feature noise and consistent performance across training set sizes.

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
- **PyTorch** — DeepSet model; checkpoint v2 format
- **Ray Tune** — HPO: 20 trials, 5 nodes, 4 concurrent/node (= 20 one-GPU trial slots)
- **scikit-learn, XGBoost, LightGBM** — preinstalled in `2.5.0-py311`
- **CatBoost `1.2.10`** — NOT preinstalled; pip-installed per baseline shard job via `TABPFN_PYPI_EAI`
- **AutoGluon `1.3.0`** — NOT preinstalled; pip-installed per AutoGluon shard job via `TABPFN_PYPI_EAI`
- **openml `0.15.1`** — NOT preinstalled; pip-installed for the prep job only via `TABPFN_PYPI_EAI`

**Compute pools:**

| Pool | Nodes | GPU | Purpose |
|------|-------|-----|---------|
| `DEEPSET_GPU_POOL` | MAX=10, GPU_NV_M | 4×A10G/node | Training, HPO, DeepSet eval shards |
| `DEEPSET_CPU_POOL` | MAX=3, CPU_X64_M | — | Prep, baseline shards, aggregation |
| `AUTOGLUON_CPU_POOL` | MAX=30, CPU_X64_M | — | AutoGluon shards |

**External access:**
- `TABPFN_PYPI_EAI` — single EAI for all pip installs (CatBoost, AutoGluon, openml)
- `BENCHMARK_EXTERNAL_ACCESS` — OpenML/Kaggle API network egress (prep job only)

---

## 4. Snowflake Stage Ownership

| Stage | Contents |
|-------|----------|
| `@META_DATASET_STAGE` | Benchmark metadata index + prepared `.npz` splits (`benchmark_prepared/`) |
| `@MODEL_STAGE/scripts/` | All runnable MLJob code from `src/*.py` and `scripts/*.py` (flat, no subdirectories) |
| `@MODEL_STAGE/hpo/` | `best_config.json` on HPO success; `hpo_failure.json` on failure |
| `@MODEL_STAGE/checkpoints/` | `pretrain.pt`, `best.pt` (v2 format) |
| `@EVALUATION_RESULTS_STAGE` | All evaluation output CSVs, charts, manifests |
| `@EVALUATION_RESULTS_STAGE/regression/{suite_id}/` | Synthetic regression shard part CSVs (path scoped by `SYNREG_RESULTS_STAGE` env var) |
| `@EVALUATION_RESULTS_STAGE/ood_parity/` | OOD pilot DeepSet-only shard part CSVs |
| `@EVALUATION_RESULTS_STAGE/ood_full/` | OOD full suite (ood_linear_full_v1) aggregation outputs |
| `@MLJOB_PAYLOAD_STAGE` | Ephemeral MLJob payloads (managed by `submit_from_stage`) |
| `@EVAL_DATASET_STAGE/primary/` | In-distribution synthetic parquet files (200 datasets) |
| `@EVAL_DATASET_STAGE/ood_parity/{E,F,G,H}/` | OOD parity source pool — 200 parquet files (50 per regime); serves both pilot (80-row subset) and full suite (all 200 rows) |
| `@EPOCH_STAGE` | Epoch calibration artifacts (`hpo_timing.json`, `train_timing.json`) |

**Invariants:**
- Never embed suite IDs, version strings, or method names in stage subdirectory paths (exception:
  `{suite_id}` in `SYNREG_RESULTS_STAGE`, which is an env var value, not a hardcoded string).
- Never use `AUTO_COMPRESS=TRUE` for PUT commands — causes silent read failures.
- `@META_DATASET_STAGE` must never be read or written by any OOD script.
- `@EVAL_DATASET_STAGE` must never hold production training parquet.

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
- Output: `data/primary/`; stage: `@EVAL_DATASET_STAGE/primary/`
- Extended NPZ suites (`feature_noise`, `training_size`, `target_noise`) are generated
  inside `prepare_synthetic_regression.py` in Snowflake, not here

**`scripts/ood_regression/generate_ood_eval_data.py`** — OOD parity suite
- Generates **200** parquet files (50/regime); source pool for both pilot (80 indexed) and full
  suite (200 indexed); invoke with `--n_datasets 200`
- Output: `data/ood_regression/{E,F,G,H}/`; stage: `@EVAL_DATASET_STAGE/ood_parity/{E,F,G,H}/`
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

- Reads OOD manifest from `@EVAL_DATASET_STAGE/ood_parity/ood_manifest.json`
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

## 6. Model Architecture (DeepSet)

- **Bounded-context ensemble**: 5 random context windows × 200 training rows per window
- **MC dropout**: K=8 forward passes per (sample, window)
- **Feature selection**: train-only `f_regression`, hard cap at 128 features
  (`SYNTHETIC_REGRESSION_DEEPSET_FEATURE_CAP`; `BENCHMARK_DEEPSET_FEATURE_CAP`)
- **Test batch size**: 128 rows (`SYNTHETIC_REGRESSION_TEST_BATCH_SIZE`)
- **Output**: mean prediction across 5×8=40 forward passes
- Selection is train-only: fit on `(X_train, y_train)`, applied to both train and holdout
- Baselines and AutoGluon receive full un-capped feature matrices
- Reference: `src/model.py`, `src/evaluate_synthetic_regression.py`

**Permutation test gate:** `run_permutation_tests(model)` is called immediately after
checkpoint load. If any of the 7 permutation-invariance tests fail, evaluation aborts with
a `RuntimeError`. This gate ensures the checkpoint actually implements DeepSet invariance.

---

## 7. Snowflake Training & Fine-Tuning

Three-phase pipeline, each submitted as an MLJob:

### Phase A — Pretraining (`run_pretrain_job.py` → `src/train.py`)

- Topology: 10 nodes × 4 workers = world size 40
- Warm-starts are supported; writes `@MODEL_STAGE/checkpoints/pretrain.pt`

### Phase B — HPO (`run_hpo_job.py` → `src/hpo.py`)

- Ray Tune on `DEEPSET_GPU_POOL`, 5 nodes, 20 trials, 4 concurrent/node
- All Snowflake stage access and data materialisation happen in the **driver only** before
  `tune.run()`. Ray workers must consume payloads via Ray object store — never open a
  Snowpark session inside a trial worker.
- Report metrics as `tune.report({"val_mse": value})` (dict-style); never keyword-style
- `tune.run(metric="val_mse", mode="min")`; `best_config.json` keys: `lr, weight_decay,
  d_phi, d_rho, dropout, pool`
- Writes `@MODEL_STAGE/hpo/best_config.json`; inspect `hpo_failure.json` first on failure
- Do not fetch Ray checkpoints directly; read only via Snowflake stage path

### Phase C — Final training (`run_model_training_job.py` → `src/train.py`)

- Topology: 10 nodes × 4 workers = world size 40
- `EXPECTED_TRAIN_WORLD_SIZE=40`, `STRICT_WORLD_SIZE_CHECK=true`
- SQL-sharded by DDP rank: `MOD(ROW_NUMBER() OVER (PARTITION BY split ORDER BY task_id) - 1, world_size) = rank`
- Warm-starts from `pretrain.pt`; writes `@MODEL_STAGE/checkpoints/best.pt` (v2 format)
- Checkpoint loading: always use `load_checkpoint_compat()` with three fallback paths:
  1. `weights_only=True` (preferred, v2 checkpoints)
  2. `safe_globals([ModelConfig]) + weights_only=True` (legacy pickled cfg)
  3. `weights_only=False` (only if `ALLOW_UNSAFE_TORCH_LOAD=true`)
- `ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS` is currently `"true"` as a temporary
  escape hatch. Revert to `"false"` only after running `scripts/migrate_checkpoint.py`
  and verifying no `[SECURITY WARNING]` log lines appear.

**Checkpoint v2 format (canonical):**
```python
{
    "checkpoint_format_version": 2,
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

---

## 8. Benchmark Evaluation Architecture

Orchestrated by `scripts/run_evaluation_test.py`.

### Phase sequence (phase-gated):

| Phase | Description |
|-------|-------------|
| 0 | Runtime probes — serialised (one at a time); validate each environment |
| 1+2 | Capacity probes — GPU → CPU → AutoGluon (non-overlapping) |
| 3 | Prep — 1 CPU job (`prepare_benchmark_datasets.py`) |
| 4 | DeepSet GPU shards — 10 shards, `DEEPSET_GPU_POOL` |
| 5 | Baseline CPU shards — 3 shards, `DEEPSET_CPU_POOL`; `catboost==1.2.10` + EAI |
| 6 | AutoGluon shards — 30 shards, `AUTOGLUON_CPU_POOL`; `autogluon.tabular==1.3.0` + EAI |
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
| DeepSet GPU shards | none | none |
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

Orchestrated by `scripts/run_synthetic_regression_evaluation.py`.

### Suite definitions:

| Suite | `suite_id` | Datasets | Seeds | Regimes | Methods |
|-------|-----------|----------|-------|---------|---------|
| In-distribution primary | `linear_poisson_v1_recommended` | 200 | [0–4] | A/B/C/D | DeepSet + 10 baselines + AutoGluon |
| OOD parity pilot | `ood_linear_pilot_v1` | 80 | [0–2] | E/F/G/H | DeepSet only |
| OOD full suite | `ood_linear_full_v1` | 200 | [0–2] | E/F/G/H | DeepSet + 10 baselines + AutoGluon |
| Combined (primary + OOD) | `linear_all_v1` | 400 | [0–2] | A/B/C/D/E/F/G/H | DeepSet + 10 baselines + AutoGluon |
| Feature noise (NPZ) | same as primary | 80×6 | [0–2] | A/B/C/D | All |
| Training size (NPZ) | same as primary | 40×8 | [0–2] | A/B/C/D | All |
| Target noise (NPZ) | same as primary | 40×5 | [0–2] | A/B/C/D | All |

### Phase sequence (same pattern as benchmark):

| Phase | Description |
|-------|-------------|
| 1 | Runtime probes (serialised) |
| 2 | Capacity probes (GPU → CPU → AG, non-overlapping) |
| 3 | Prep — 1 CPU job (`prepare_synthetic_regression.py`) |
| 4 | DeepSet GPU shards (10, `DEEPSET_GPU_POOL`) — `SYNREG_RESULTS_STAGE=@EVALUATION_RESULTS_STAGE/regression/{suite_id}` |
| 5 | Baseline CPU shards (3, `DEEPSET_CPU_POOL`) — same `SYNREG_RESULTS_STAGE` |
| 6 | AutoGluon shards (30, `AUTOGLUON_CPU_POOL`) — same `SYNREG_RESULTS_STAGE` |
| 7 | Aggregation (1 CPU job) — reads from suite-specific prefix; validates `suite_id` in all rows |
| 8 | OOD pilot runs as separate procedure: `run_synthetic_regression_ood_deepset_pilot` (only DeepSet, 5 GPU shards, results → `@EVALUATION_RESULTS_STAGE/ood_parity/`) |
| 9 | OOD full suite runs as separate procedure: `run_synthetic_regression_ood_full_evaluation` (prep + DeepSet + baselines + AutoGluon + aggregation for 200-dataset OOD suite; aggregation outputs → `SYNREG_OUTPUT_STAGE=@EVALUATION_RESULTS_STAGE/ood_full`) |
| 10 | Combined suite runs as separate procedure: `run_synthetic_regression_combined_evaluation` (combined prep + DeepSet + baselines + AutoGluon + aggregation for 400-dataset combined suite; aggregation outputs → `@EVALUATION_RESULTS_STAGE/combined`; requires both `linear_poisson_v1_recommended` and `ood_linear_full_v1` to be indexed first) |

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

**Combined suite (`linear_all_v1`):**
```sql
CALL run_synthetic_regression_combined_prep('2.5.0-py311', '2.5.0-py311');
CALL run_synthetic_regression_combined_deepset_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_baseline_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_autogluon_evaluation('2.5.0-py311', '2.5.0-py311');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', '2.5.0-py311');
```

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
  a missing DeepSet/baseline/AutoGluon shard raises RuntimeError before aggregation writes any output
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
| `synthetic_regression_chart_data_noise_features.csv` | x=noise_level, y=relative_degradation — DeepSet stability vs. baselines |
| `synthetic_regression_chart_data_training_size.csv` | x=n_train, y=relative_improvement — DeepSet sample efficiency |
| `synthetic_regression_chart_data_model_rank.csv` | x=method, y=mean_rank — overall ranking |

The two stability charts (`noise_features` and `training_size`) are the primary visual
evidence for DeepSet's resistance to noise and consistent performance across training set sizes.

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
  `external_access_integrations`. DeepSet GPU jobs and the aggregation job carry neither.
- **All pip jobs** use the single `TABPFN_PYPI_EAI` integration.
- **Checkpoints:** always write v2 format (plain dict `cfg`); always load via
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
  libraries; reusable DeepSet, baseline, AutoGluon, and ranking code belongs in stable
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
- **Model family routing:** Never hardcode `DeepSetModel(cfg=cfg)` in `evaluate.py`,
  `train.py`, or `hpo.py`. Always call `_instantiate_model(cfg)` from `model.py`.
  Routing is determined by `cfg.model_family` ("deepset" | "market_aware").
- **`_checkpoint_architecture_mismatches` fields:** Must include `"model_family"` alongside
  the other 8 structural fields. A mismatch on `model_family` aborts warm-start.
- **Checkpoint version:** `MarketAwareDeepSetModel` checkpoints save
  `checkpoint_format_version=3`. `DeepSetModel` checkpoints keep version 2.
  Never save a `market_aware` model with version 2.
- **`n_sab_sample_per_feature` guard:** Default is 0. Do not set > 0 in production
  evaluation without first verifying `p <= 10` or adding chunking over the `m*p` batch dim.
  At `m=128, p=128, n=200`, enabling SAB over n produces a 2.6 GB fp32 attention matrix.

---

## 13. Operational Guardrails (Critical Rules)

### Data safety

- Never materialise full benchmark datasets locally inside an MLJob; use `@META_DATASET_STAGE` paths.
- Never use `AUTO_COMPRESS=TRUE` for PUT commands (causes silent read failures).
- OOD scripts never touch `@META_DATASET_STAGE`; only `@EVAL_DATASET_STAGE`.
- `data/ood_regression/` is the local directory; `@EVAL_DATASET_STAGE/ood_parity/` is the stage prefix.
- `@EVAL_DATASET_STAGE` must never hold production training parquet.

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

- DeepSet shards must complete before baselines submit; baselines before AutoGluon;
  all three before aggregation.
- Runtime probes are serialised (one at a time; wait for each before next).
- Capacity probes are phase-gated: GPU → CPU → AG (non-overlapping).
- `target_instances=1` for every MLJob (shard jobs are independent single-node jobs).
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
| openml | `0.15.1` | `OPENML_VERSION` / `PREP_EXTRA_PIP_REQUIREMENTS` |

- Always pin exact versions (`==`). Never use `>=` or unpinned requirements.
- DeepSet GPU jobs: no pip, no EAI.
- Aggregation job: no pip, no EAI.
- `autogluon.tabular` stays lazily imported inside `predict_autogluon()`.

### HPO guardrails

- All Snowflake I/O happens in the HPO driver only; Ray workers use Ray object store.
- Never use keyword-style `tune.report(val_mse=value)` — use dict-style only.
- `TunerConfig` documented parameters only: `metric`, `mode`, `search_alg`, `num_trials`,
  `max_concurrent_trials`, `resource_per_trial`. Do not add undocumented kwargs.
- `resource_per_trial={"GPU": 1}` is mandatory — Snowflake does not allocate GPUs automatically.
- Do not call `scale_cluster()` from `hpo.py`; scaling is at submission time only.

### Checkpoint safety

- Never save `ModelConfig` objects directly in PyTorch checkpoints (breaks `weights_only=True`).
- Always serialize `cfg` as `dataclasses.asdict(model.cfg)` (plain dict). Checkpoint format version: v3 for `market_aware`, v2 for `deepset`. Use `checkpoint_format_version = 3 if cfg.model_family == "market_aware" else 2`.
- Consumers must normalise `ckpt["cfg"]` dict back to `ModelConfig` before comparing fields.
- `weights_only=False` can execute arbitrary code — only for internally trusted checkpoints;
  never for third-party checkpoints.
- Synthetic regression evaluation shards must set `ALLOW_UNSAFE_TORCH_LOAD=true`
  directly for every `SYNTHETIC_REGRESSION_MODE` of `deepset`, `baselines`, or
  `autogluon`. Baseline and AutoGluon shards may still touch the TabPFN
  checkpoint or shared evaluation utilities, so they carry the same temporary
  trusted-checkpoint exception for internal Snowflake staged checkpoints.
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
  LIST @MODEL_STAGE/scripts/ PATTERN='.*(evaluate_synthetic_regression|deepset_inference|baseline_models|autogluon_models|evaluation_metrics|evaluate|run_synthetic_regression_evaluation)[.]py';
  ```
  Use the targeted block above when only the evaluator/helper refactor changed. Use
  the broad `src/*.py` plus `scripts/*.py` PUT block when procedure dependencies or
  multiple shared modules may have changed. `scripts/ood_regression/generate_ood_eval_data.py`
  remains local-only and must not be staged.

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

The DeepSet model must not be evaluated solely through full benchmark aggregation before
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

## Guardrail: MarketAwareDeepSetModel is the Production Default

`MarketAwareDeepSetModel` is the default production model for all synthetic regression training,
OOD synthetic regression, and future market mental model prior training.
`DeepSetModel` is available only for compatibility and ablation. Do not revert
`DEEPSET_MODEL_FAMILY` to `"deepset"` without an explicit ablation reason.

## Guardrail: Synthetic Regression Evaluation Must Instantiate from Checkpoint cfg

`load_best_deepset_checkpoint()` in `evaluate_synthetic_regression.py` must always call
`_instantiate_model(cfg)` — never hardcode `DeepSetModel(cfg=cfg)` or
`MarketAwareDeepSetModel(cfg=cfg)`. The model class is determined by `cfg.model_family`
from the checkpoint.

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

`run_permutation_tests()` in `src/deepset_inference.py` dispatches to a `market_aware` or
`deepset` branch based on `cfg.model_family`. Never call it with a model whose `model_family`
is absent or unknown — it will raise `ValueError`. Tests 3-5 for `market_aware` cover feature
equivariance, finite output, and batch-query shape; Tests 3-7 for `deepset` cover sample and
feature equivariance. The function always restores the model's original training mode.

## Guardrail: HPO Must Propagate model_family Into best_config.json

`src/hpo.py` defaults to `HPO_MODEL_FAMILY = os.environ.get("DEEPSET_MODEL_FAMILY", "market_aware")`.
The `model_family` key is written into every `best_config.json` uploaded to `@MODEL_STAGE/hpo/`.
`train.py` reads it via `hyper_params.get("model_family", DEEPSET_MODEL_FAMILY)`. Do not allow
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

`MODEL_ARCH_VERSION` and `MODEL3_DESIGN_PATTERN` must be set explicitly in all production
Snowflake training submissions when running MODEL3. Default values (`model2` / `inductive_forecasting`)
preserve MODEL2 behavior.

**Key rules:**
- `MODEL_ARCH_VERSION="model2"` (default) routes to `MarketAwareDeepSetModel`. MODEL3 code paths
  do NOT activate unless `MODEL_ARCH_VERSION="model3"`.
- `MODEL3_DESIGN_PATTERN="inductive_forecasting"` (default) does NOT trigger MODEL3 unless
  `MODEL_ARCH_VERSION="model3"` is also set.
- Required MODEL3 combinations:
  - `model3` + `inductive_forecasting` → `model_family="market_exchangeable_icl"` → `MarketExchangeableICLModel`
  - `model3` + `transductive_completion` → `model_family="market_exchangeable_completion"` → `MarketExchangeableCompletionModel`
- MODEL3 checkpoints use `checkpoint_format_version=4` and must include `model_arch_version`,
  `model3_design_pattern`, and `task_objective` in checkpoint metadata.
- Do not mix MODEL2 and MODEL3 selectors (e.g., `model_arch_version="model2"` with a MODEL3 family).
  `ModelConfig.__post_init__` raises `ValueError` for invalid combinations.
- `run_training_job.py`, `run_model_training_job.py`, and `run_hpo_job.py` each expose
  `DEFAULT_MODEL_ARCH_VERSION` and `DEFAULT_MODEL3_DESIGN_PATTERN` constants that propagate
  through `env_vars` to all MLJob children.

## Guardrail: MODEL3 Must Not Mutate MODEL2 Classes

`MarketAwareDeepSetModel` is the current production model and must not be modified in place
to add MODEL3 behavior. MODEL3 is implemented as separate classes:
- `MarketExchangeableICLModel` — inductive forecasting
- `MarketExchangeableCompletionModel` — transductive completion

Shared primitives (`ExchangeableMatrixBlock`, `ColumnEncoder`, `CellEncoder`, `_masked_mean`)
are module-level additions to `model.py` and do not alter existing class behavior.
