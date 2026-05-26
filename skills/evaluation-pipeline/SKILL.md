---
name: evaluation-pipeline
description: >
  Reference for the synthetic-regression evaluation pipeline. Use when
  modifying any of: run_synthetic_regression_evaluation.py,
  evaluate_synthetic_regression.py, autogluon_ray.py,
  prepare_synthetic_regression.py, or when adding a new evaluation suite or model.
  Covers sharding, env vars, compute topology, checkpoint contract, and the
  reproducibility checklist for new suites.
---

## 1. Compute Topology

| Pool | MAX_NODES | GPU | Purpose |
|------|-----------|-----|---------|
| `DEEPSET_GPU_POOL` | 10 | 4×A10G/node | DeepSet shards |
| `DEEPSET_CPU_POOL` | 6 | — | Prep, baselines, aggregation |
| `AUTOGLUON_CPU_POOL` | 24 | — | AutoGluon cluster shards |

---

## 2. Shard Topology (per mode)

| Mode | N shards | Pool | pip | EAI | Entrypoint |
|------|----------|------|-----|-----|------------|
| `deepset` | 10 | GPU | none | none | `evaluate_synthetic_regression.py` |
| `baselines` | 6 | CPU | `catboost==1.2.10` | `TABPFN_PYPI_EAI` | `evaluate_synthetic_regression.py` |
| `autogluon` (single-node) | 60 | `AUTOGLUON_CPU_POOL` | `autogluon.tabular==1.3.0` | `TABPFN_PYPI_EAI` | `evaluate_synthetic_regression.py` |
| `autogluon` (ray_work_items) | 6 clusters × 4 workers | `AUTOGLUON_CPU_POOL` | `autogluon.tabular==1.3.0` + `ray` | `TABPFN_PYPI_EAI` | `autogluon_ray.py` |
| `aggregate` | 1 | CPU | none | none | `evaluate_synthetic_regression.py` |

**Shard assignment:** `enumerate_idx % SYNREG_NUM_SHARDS == SYNREG_SHARD_INDEX` (deterministic, always `ORDER BY suite_id, suite_family, prior_regime, dataset_id, ...`).

`_synreg_shard_env(mode, suite_id, num_shards, shard_index, results_stage, extra_env)` builds env dict for each shard job.

---

## 3. Suite Catalog

| suite_id | Datasets | Seeds | Regimes | Methods |
|----------|----------|-------|---------|---------|
| `linear_poisson_v1_recommended` | 200 | [0–4] | A/B/C/D | DeepSet + 10 baselines + AutoGluon |
| `ood_linear_pilot_v1` | 80 | [0–2] | E/F/G/H | DeepSet only |
| `ood_linear_full_v1` | 200 | [0–2] | E/F/G/H | DeepSet + 10 baselines + AutoGluon |
| `linear_all_v1` | 400 (200+200) | [0–2] | A–H | DeepSet + 10 baselines + AutoGluon |

**Sub-families within primary suite** (all share `linear_poisson_v1_recommended`):
- `primary` — 200 datasets × 5 seeds
- `feature_noise` — 80 base datasets × 3 seeds × 6 noise levels (0/10/25/50/75/100)
- `training_size` — 40 large datasets × 3 seeds × 8 n_train values (25–4832)
- `target_noise` — 40 datasets × 3 seeds × 5 noise scales (disabled by default)

---

## 4. Phase Sequence

Same pattern for all three full suites (main / OOD full / combined):

| Phase | Description | Compute pool |
|-------|-------------|-------------|
| 1 | Runtime probes (serialised) | all 3 |
| 2 | Capacity probes (GPU→CPU→AG, non-overlapping) | all 3 |
| 3 | Prep — `prepare_synthetic_regression.py` | `DEEPSET_CPU_POOL` |
| 4 | DeepSet — 10 GPU shards | `DEEPSET_GPU_POOL` |
| 5 | Baselines — `BASELINE_SHARDS` CPU shards (default 6); 1 shard = 1 MLJob = 1 shard file | `DEEPSET_CPU_POOL` |
| 6 | AutoGluon — 6 cluster shards × 4 workers | `AUTOGLUON_CPU_POOL` |
| 7 | Aggregation — 1 CPU job | `DEEPSET_CPU_POOL` |

OOD pilot (`ood_linear_pilot_v1`) = prep + 5 DeepSet shards only; no baselines/AG/aggregation.

Issue `ALTER COMPUTE POOL <pool> SUSPEND` between phases when operating under quota.

---

## 5. Stage Paths

- `@EVALUATION_DATASET_STAGE` — input dataset payloads (NPZ/Parquet per dataset)
- `@EVALUATION_DATASET_STAGE/{suite_family}/` — dataset files uploaded by prep job
- `@EVALUATION_RESULTS_STAGE/regression/{suite_id}/` — shard part CSVs (`SYNREG_RESULTS_STAGE`)
- `@EVALUATION_RESULTS_STAGE/ood_full/` / `/combined/` — aggregation output for OOD/combined
- `@EVALUATION_RESULTS_STAGE/synthetic_regression_charts/` — optional PNG charts
- `@MODEL_STAGE/checkpoints/best.pt` — default DeepSet checkpoint (`SYNREG_DEEPSET_CKPT_STAGE`)
- `SYNTHETIC_REGRESSION_DATASET_INDEX` — Snowflake table; source of truth for all suites

---

## 6. DeepSet Evaluation

**Checkpoint preflight** (in orchestrator, before GPU submission):
`_stage_file_exists(session, ckpt_dir, ckpt_filename)` → `RuntimeError` if missing.

**Checkpoint loading** (`load_best_deepset_checkpoint` in `evaluate_synthetic_regression.py`):
1. `session.file.get(SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH, ckpt_dir)`
2. Resolve local path: exact filename from stage path → glob variants → any `.pt` → `RuntimeError`
3. `safe_torch_load_with_legacy_escape_hatch(local_path, device)` (weights_only=True preferred)
4. `validate_checkpoint_payload` → `normalize_checkpoint_cfg` → `_instantiate_model(cfg)`
5. `run_permutation_tests(model)` — fail-fast
6. Optional: `run_checkpoint_gates(model, ...)` (controlled by `SYNREG_RUN_CHECKPOINT_GATES`)

**Key DeepSet inference env vars:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH` | `@MODEL_STAGE/checkpoints/best.pt` | Checkpoint stage path |
| `MC_K` | 8 | Monte Carlo samples per ensemble member |
| `SYNTHETIC_REGRESSION_CONTEXT_SIZE` | 200 | Rows per context window |
| `SYNTHETIC_REGRESSION_CONTEXT_ENSEMBLES` | 5 | Context windows per prediction |
| `SYNTHETIC_REGRESSION_TEST_BATCH_SIZE` | 128 | Holdout batch size |
| `SYNTHETIC_REGRESSION_DEEPSET_FEATURE_CAP` | 128 | Max selected features |
| `SYNTHETIC_REGRESSION_FEATURE_SELECTOR` | `train_f_regression` | Feature selection method |
| `BENCHMARK_REQUIRE_CUDA` | (unset) | Fail if CUDA unavailable |
| `SYNREG_RUN_CHECKPOINT_GATES` | `true` | Run quality gates |
| `SYNREG_CHECKPOINT_GATE_STRICT` | `true` | Raise on gate failure |
| `SYNREG_GATE_MAX_RIDGE_RATIO` | 10.0 | Ridge/query std ratio threshold |
| `SYNREG_GATE_MIN_QUERY_STD` | 1e-6 | Min query std on shuffled inputs |
| `ALLOW_UNSAFE_TORCH_LOAD` | `true` (set by orch) | Permit weights_only=False fallback |

---

## 7. Checkpoint Contract

Requirements for a valid checkpoint in this pipeline:

- `checkpoint_format_version == 4`
- Keys: `cfg` (dataclass dict), `state_dict`, `metadata`
- `cfg.model_family` must be `market_exchangeable_icl` (only supported family)
- `cfg.model_arch_version == "model3"`
- `metadata.task_objective == "inductive_regression"`
- Retired families (`deepset`, `market_aware`, `market_exchangeable_completion`) → rejected

---

## 8. Baseline Evaluation

**10 models** (run sequentially per shard, same split):
`FixedRidgeLambda1`, `LinearRegression`, `Ridge`, `RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`, `KNN`, `SVR`, `MLP`

**Runtime-configurable shard count:**

| Variable / SQL arg | Default | Description |
|--------------------|---------|-------------|
| `SYNREG_BASELINE_SHARDS` / `BASELINE_SHARDS` | `SYNREG_CPU_SHARDS` = 6 | Number of baseline shard files written; 1 shard = 1 MLJob = 1 output file |
| `SYNREG_BASELINE_CONCURRENT_NODES` / `BASELINE_CONCURRENT_NODES` | 6 | Required single-wave CPU nodes; must equal `BASELINE_SHARDS` |

`BASELINE_CONCURRENT_NODES` must **equal** `BASELINE_SHARDS`. Lower values are rejected (no multi-wave batching). Aggregation must expect the same resolved shard count.

**Memory guards:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `BENCHMARK_CPU_MAX_PROCESSED_FEATURES` | 2000 | Skip if features > threshold |
| `BENCHMARK_CPU_MAX_MATRIX_BYTES` | 536870912 (~512 MB) | Skip if matrix bytes > threshold |

**Pip requirement:** `catboost==1.2.10` via `TABPFN_PYPI_EAI` (not preinstalled).

For `training_size` suite family: each index row expands across `SYNTHETIC_REGRESSION_TRAIN_SIZE_GRID` (default `25,50,100,200,500,1000,2000,4832`).

---

## 9. AutoGluon Execution Modes

Two execution modes are supported for the combined suite (`linear_all_v1`). The mode is selected
by the `AUTOGLUON_CLUSTER_SHARDS` SQL argument / `SYNREG_AUTOGLUON_CLUSTER_SHARDS` env var.

### 9A. Ray distributed cluster-shard mode (`CLUSTER_SHARDS > 0`)

**Topology:** N cluster shards submitted as MLJobs with `target_instances=WORKERS_PER_SHARD`.

Each MLJob:
1. Starts Ray cluster (`ray.init()`; abort if fails)
2. Driver opens Snowpark session and loads metadata only from `SYNTHETIC_REGRESSION_DATASET_INDEX`
3. Driver expands work items, assigns shard via modulo, and builds small item dicts
4. Driver enriches each item dict with `dataset_access.mode='scoped_file_url'` and
   `dataset_access.scoped_url` derived from the stage path with `BUILD_SCOPED_FILE_URL`
5. Driver submits `_autogluon_work_item.remote(item)` and passes **only the small item dict**
6. Each worker receives the small item dict as a Ray task argument and loads its own dataset
   with `SnowflakeFile.open(scoped_url)` without creating a Snowpark session
7. Workers apply dataset-size, feature, and matrix guards, then run AutoGluon fit/predict
8. Bounded in-flight pool (size = `MAX_IN_FLIGHT`) bounds active worker-loaded fits
9. Driver collects small result rows, writes **one CSV per cluster shard** (workers never write)

**Worker-local dataset loading:** The driver does **not** download full datasets and does
**not** call `ray.put(dataset)`. Full dataset payloads must not flow through the Ray
driver or Ray object store. The Ray task argument is a small serializable item dict,
not the dataset bytes. Session-free worker loading is required: the driver generates
the scoped file URL while it has a Snowpark session, and the worker opens that scoped
URL with `snowflake.snowpark.files.SnowflakeFile`. Workers must not call
`Session.builder.getOrCreate()` and must never query `SYNTHETIC_REGRESSION_DATASET_INDEX`.
If scoped URLs fail in the Snowflake MLJob environment, fail clearly; do not silently
return to driver-side dataset downloads or worker-created Snowpark sessions.

**Key env vars:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `SYNREG_AUTOGLUON_CLUSTER_SHARDS` | 6 | Logical cluster shards (> 0 → Ray mode) |
| `SYNREG_AUTOGLUON_WORKERS_PER_SHARD` | 4 | `target_instances` per MLJob |
| `AUTOGLUON_TASK_CPUS` | 1 | CPUs per Ray task |
| `AUTOGLUON_TIME_LIMIT` | 300 | Per-fit seconds |
| `AUTOGLUON_PRESETS` | `best_quality` | AutoGluon preset |
| `SYNREG_AUTOGLUON_MAX_IN_FLIGHT` | `WORKERS_PER_SHARD` | Pending Ray task pool size |
| `BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES` | 5368709120 (~5 GB) | /tmp guard |
| `SYNREG_AUTOGLUON_DISTRIBUTED_MODE` | `ray_work_items` | Distribution mode |
| `SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS` | 6 | Must equal `CLUSTER_SHARDS` (single-wave) |
| `SYNREG_WORKER_DATA_ACCESS_MODE` | `scoped_file_url` | Driver-derived scoped URL; workers do not create Snowpark sessions |
| `SYNREG_MAX_WORK_ITEM_BYTES` | 8192 | Compact Ray item metadata size guard |
| `BENCHMARK_CPU_MAX_PROCESSED_FEATURES` | 512 | Skip work item if number of features exceeds threshold (note: baseline mode default is 2000) |
| `BENCHMARK_CPU_MAX_MATRIX_BYTES` | 2,147,483,648 | Skip work item if feature matrix bytes exceed threshold (note: baseline mode default is 536,870,912) |

**Derived entrypoint:** Ray mode derives `autogluon_ray.py` internally. Do not expose
or accept an arbitrary runtime entrypoint for normal distributed AutoGluon execution.

**Startup validation sequence (fail-before-CSV guarantee):**
- `DISTRIBUTED_MODE != "ray_work_items"` → `RuntimeError` immediately
- `NUM_SHARDS != CLUSTER_SHARDS` → `RuntimeError` (enforces single-wave: concurrent clusters == total shards)
- `ray.init()` fails → `RuntimeError` (abort before any work item is processed)
- CPU count check (`psutil.cpu_count() < MIN_CPUS`) → `RuntimeError`

**SQL example (Ray mode):**
```sql
CALL run_synthetic_regression_combined_autogluon_capacity_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
CALL run_synthetic_regression_combined_autogluon_worker_access_probe('2.5.0-py311', '2.5.0-py311', 6, 4, 6);
CALL run_synthetic_regression_combined_autogluon_evaluation('2.5.0-py311', '2.5.0-py311', 6, 4, 1, 6, 300, 'best_quality');
```

**AutoGluon import timing probe** — measures scheduling + image startup + pip bootstrap latency:
```sql
-- Single pip-mode probe (default):
CALL run_synthetic_regression_autogluon_import_timing_probe('2.5.0-py311');

-- 8 concurrent pip-mode probes (simulates full evaluation wave concurrency):
CALL run_synthetic_regression_autogluon_import_timing_probe('2.5.0-py311', TRUE, 8);

-- 8 concurrent no-pip probes (scheduling + image startup baseline; skips AutoGluon/Ray imports):
CALL run_synthetic_regression_autogluon_import_timing_probe('2.5.0-py311', FALSE, 8);
```
Interpretation: time from MLJob submission to `python_entrypoint_started` ≈ scheduling + image pull + pip install in pip mode. In no-pip baseline mode, the probe emits `autogluon_import_skipped` / `ray_import_skipped` and should succeed even when AutoGluon is absent. `autogluon_import_complete.import_seconds` is pure import overhead in pip/preinstalled validation modes. Compare pip vs no-pip waves to isolate bootstrap cost. Stage with:
```sql
PUT file://C:/Documents/TabPFN_DemandModel/scripts/autogluon_import_timing_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

### 9B. Single-node shard mode (`CLUSTER_SHARDS = 0`)

**Topology:** N single-instance MLJobs (`target_instances=1`), each running
`evaluate_synthetic_regression.py` with `mode=autogluon`. No Ray cluster is formed.

| Variable | Value | Purpose |
|----------|-------|---------|
| `SYNREG_AUTOGLUON_CLUSTER_SHARDS` | **0** | Selects single-node mode |
| `SYNREG_AUTOGLUON_WORKERS_PER_SHARD` | 1 | Must be 1; each shard is one container |
| `SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS` | N | Number of shards = `output_shards` |

**Derived entrypoint:** Single-node mode derives `evaluate_synthetic_regression.py`
internally. Do not expose or accept an arbitrary runtime entrypoint for normal
single-node AutoGluon execution.

- Output files: `AutoGluon_shard{i}_of_{N}_detailed.csv` where `N = CONCURRENT_CLUSTERS`.
- Aggregation `SYNREG_EXPECTED_AG_SHARDS` = `CONCURRENT_CLUSTERS` (not `CLUSTER_SHARDS`).
  `run_synthetic_regression_combined_evaluation` wires this automatically via `ag_plan.output_shards`.
- Capacity probes use `capacity_probe.py` with `target_instances=1`.
- `WORKERS_PER_SHARD > 1` is rejected by `_resolve_combined_autogluon_execution_plan`.
- The `autogluon_ray.py` entrypoint is forbidden in this mode (rejected at plan resolution).

**SQL example (single-node mode):**
```sql
CALL run_synthetic_regression_combined_autogluon_capacity_probe('2.5.0-py311', '2.5.0-py311', 0, 1, 30);
CALL run_synthetic_regression_combined_autogluon_worker_access_probe('2.5.0-py311', '2.5.0-py311', 0, 1, 30);
CALL run_synthetic_regression_combined_autogluon_evaluation('2.5.0-py311', '2.5.0-py311', 0, 1, 1, 30, 300, 'best_quality');
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', '2.5.0-py311', 30);
```

### 9C. Capacity and worker-access probes

**Capacity probe:** `run_synthetic_regression_combined_autogluon_capacity_probe`
validates the execution envelope without running AutoGluon training.

Signature:
```sql
CALL run_synthetic_regression_combined_autogluon_capacity_probe(
  '<image_repo>', '<image_tag>',
  <AUTOGLUON_CLUSTER_SHARDS>, <AUTOGLUON_WORKERS_PER_SHARD>, <AUTOGLUON_CONCURRENT_CLUSTERS>
);
```

- Ray mode (`CLUSTER_SHARDS > 0`): submits one MLJob per cluster shard, each with
  `target_instances=WORKERS_PER_SHARD`, and runs `ray_capacity_probe.py`.
  Uses the same Ray dependency contract as distributed evaluation
  (`SYNREG_AG_RAY_PIP`).
- Ray capacity readiness is configurable per call through the extended overload:
  `(..., CLUSTER_SHARDS, WORKERS_PER_SHARD, CONCURRENT_CLUSTERS,
  RAY_READY_TIMEOUT_SECONDS, RAY_READY_POLL_SECONDS)`. Default is 300s timeout,
  10s poll.
- Single-node mode (`CLUSTER_SHARDS = 0`): submits `CONCURRENT_CLUSTERS`
  single-instance probes and runs `capacity_probe.py`.
- The probe verifies container/Ray startup and resource visibility only. It does not
  download full datasets and does not run AutoGluon.

**AutoGluon Ray evaluation readiness:** the distributed evaluation extended overload
accepts `RAY_READY_TIMEOUT_SECONDS` and `RAY_READY_POLL_SECONDS` after
`AUTOGLUON_PRESETS`. Default is 600s timeout, 10s poll. The entrypoint receives these
as `SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS` and
`SYNREG_RAY_CLUSTER_READY_POLL_SECONDS`.

**Worker-access probe:** `run_synthetic_regression_combined_autogluon_worker_access_probe`
validates that the deployed worker data-access path works with the same runtime
parameters as the capacity probe.

Signature:
```sql
CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
  '<image_repo>', '<image_tag>',
  <AUTOGLUON_CLUSTER_SHARDS>, <AUTOGLUON_WORKERS_PER_SHARD>, <AUTOGLUON_CONCURRENT_CLUSTERS>
);
```

- Ray mode uses the same one-cluster-per-shard and `target_instances=WORKERS_PER_SHARD`
  topology as the capacity probe and distributed evaluation.
- The Ray driver loads one or more metadata rows from `SYNTHETIC_REGRESSION_DATASET_INDEX`,
  builds small item dicts, and passes those dicts to Ray workers as task arguments.
- Workers verify they can receive the dict and resolve the same `scoped_file_url`
  access mechanism used by production evaluation. The probe must remain lightweight: no AutoGluon training, no full-suite
  dataset fan-out, and no driver `ray.put(dataset)`.
- Single-node mode uses the same single-instance topology as the single-node capacity
  probe. It verifies the single-node metadata/dataset access path without Ray.
- Failure messages should distinguish scheduling/startup failures from metadata,
  URL/access-descriptor, or local download failures.

### 9D. Common guardrails (both modes)

**Ephemeral AutoGluon artifacts:**
- `TabularPredictor.fit(..., path=tmp_dir)` uses `cleanup=True`; all `.pkl` and model artifacts are deleted after inference. No AutoGluon model artifacts are uploaded to Snowflake stages. Only the results CSV is uploaded.

**Multi-instance entrypoint allowlist (`_submit_synreg` guard):**
- `target_instances > 1` is only permitted for Ray-coordinated entrypoints. The allowlist (`_MULTI_INSTANCE_SYNREG_ENTRYPOINTS`) contains exactly these entries:
  - `autogluon_ray.py` — production distributed AutoGluon Ray entrypoint; calls `ray.init(address="auto")`; driver writes exactly one shard CSV; workers execute work items only.
  - `ray_capacity_probe.py` — lightweight Ray capacity probe; calls `ray.init(address="auto")`; verifies `EXPECTED_RAY_NODES` and `EXPECTED_RAY_CPUS_MIN`; writes no shard outputs.
  - `autogluon_worker_access_probe.py` — lightweight worker-access probe; calls `ray.init(address="auto")`; verifies that small item dicts can move from driver to worker and that the configured worker dataset access descriptor can be resolved; writes no shard outputs.
- All other entrypoints (`evaluate_synthetic_regression.py`, `capacity_probe.py`, etc.) must remain single-instance. Submitting them with `target_instances > 1` raises `RuntimeError` before the Snowflake call.
- Aggregation does not launch AutoGluon workers. It only consumes completed shard CSVs. One Ray cluster maps to exactly one logical AutoGluon shard.
- All allowlisted entrypoints must be staged to `@MODEL_STAGE/scripts/` before running combined AutoGluon evaluation, capacity probe, or worker-access probe procedures.

---

## 10. Aggregation Mode

**Inputs required:**
- All `*_detailed.csv` part files in `SYNREG_RESULTS_STAGE`
- Each file ≥ 64 bytes (completeness check)
- `SYNREG_EXPECTED_DEEPSET_SHARDS`, `SYNREG_EXPECTED_BASELINE_SHARDS`, `SYNREG_EXPECTED_AG_SHARDS` (0 = skip check)

**Canonical outputs** (all written to `SYNREG_OUTPUT_STAGE`):
1. `synthetic_regression_model_comparison.csv` — full ranked rows
2. `synthetic_regression_model_comparison_summary.csv` — per-method aggregates
3. `synthetic_regression_summary_by_regime.csv`
4. `synthetic_regression_summary_by_feature_noise.csv`
5. `synthetic_regression_summary_by_training_size.csv`
6. `synthetic_regression_summary_by_regime_p_quartile_n_quartile.csv`
7. `synthetic_regression_summary_by_target_noise.csv` (if target_noise suite)
8. `synthetic_regression_chart_data_noise_features.csv`
9. `synthetic_regression_chart_data_training_size.csv`
10. `synthetic_regression_chart_data_model_rank.csv`
11. `synthetic_regression_aggregation_manifest.json`
12. Optional: PNG charts → `@EVALUATION_RESULTS_STAGE/synthetic_regression_charts/`

**Ratio columns computed:** `ratio_mse_to_fixed_ridge`, `ratio_mse_to_autogluon`, `ratio_mse_to_best_tree`, `beats_fixed_ridge`, `beats_autogluon`, `beats_best_tree`, `is_best_mse`, `is_top3_mse`.

---

## 11. All Env Vars (Compact Reference)

**Sharding & orchestration** (in `evaluate_synthetic_regression.py`):

| Variable | Default |
|----------|---------|
| `SYNTHETIC_REGRESSION_SUITE_ID` | `linear_poisson_v1_recommended` |
| `SYNTHETIC_REGRESSION_NUM_SHARDS` | 1 |
| `SYNTHETIC_REGRESSION_SHARD_INDEX` | 0 |
| `SYNTHETIC_REGRESSION_MODE` | `deepset` |
| `SYNREG_RESULTS_STAGE` | `@EVALUATION_RESULTS_STAGE/regression` |
| `SYNREG_OUTPUT_STAGE` | `@EVALUATION_RESULTS_STAGE` |
| `SYNTHETIC_REGRESSION_LOCAL_CACHE` | `/tmp/synreg_eval_data` |
| `SYNTHETIC_REGRESSION_CKPT_LOCAL` | `/tmp/synreg_ckpt/best.pt` |

**Prep phase** (in `prepare_synthetic_regression.py`):

| Variable | Default |
|----------|---------|
| `SYNTHETIC_REGRESSION_BASE_SEED` | 20260512 |
| `SYNTHETIC_REGRESSION_FORCE_REBUILD` | false |
| `SYNTHETIC_REGRESSION_PRIMARY_DATASETS` | 200 |
| `SYNTHETIC_REGRESSION_PRIMARY_SPLIT_SEEDS` | `0,1,2,3,4` |
| `SYNTHETIC_REGRESSION_FEATURE_NOISE_DATASETS` | 80 |
| `SYNTHETIC_REGRESSION_FEATURE_NOISE_LEVELS` | `0,10,25,50,75,100` |
| `SYNTHETIC_REGRESSION_TRAIN_SIZE_DATASETS` | 40 |
| `SYNTHETIC_REGRESSION_TRAIN_SIZE_GRID` | `25,50,100,200,500,1000,2000,4832` |
| `SYNTHETIC_REGRESSION_HOLDOUT_SIZE` | 1371 |

---

## 12. Adding a New Evaluation Suite

1. **Design index schema**: define `suite_id`, `suite_family`, `prior_regime`, `split_seeds`, `n_total`, `p_signal`, `p_noise`, `n_train_default`, `n_holdout_default`, `logical_dataset_key`.
2. **Generate datasets**: write or extend `prepare_synthetic_regression.py` with a `prepare_{family}_suite()` function. Each dataset = NPZ/Parquet with `X`, `y`, `betaX` (noiseless target), metadata dict.
3. **Upload datasets**: `session.file.put(local_path, "@EVALUATION_DATASET_STAGE/{family}/", auto_compress=False, overwrite=True)`.
4. **Insert index rows**: call `_insert_synreg_index_rows(session, rows)` (existing helper). Set `SYNREG_RESULTS_STAGE` env var to a suite-specific subdirectory.
5. **Register constants** in `run_synthetic_regression_evaluation.py`: new `SUITE_ID`, `GPU_SHARDS`, `CPU_SHARDS`, results stage path.
6. **Add handler functions** following existing pattern: `_prep`, `_deepset_evaluation`, `_baseline_evaluation`, `_autogluon_evaluation`, `_aggregation`, `_evaluation` (all-in-one). Add checkpoint preflight to the deepset handler.
7. **Add SQL stored procedures** in `sql/run_training_job.sql` (copy nearest existing overload family, adjust HANDLER).
8. **Add tests**: `_patch_submit` + `_stage_file_exists` patch; assert correct shard counts, env vars, pool assignments.

**Invariants that must hold for new suites:**
- Never hardcode `suite_id` in stage paths; always use `SYNREG_RESULTS_STAGE` env var.
- Shard assignment must be deterministic `ORDER BY` on canonical columns.
- Combined suites are index-level only; never rewrite or merge parquet files.
- Aggregation must cross-validate `suite_id` in all ingested rows (contamination guard).
- OOD prep always calls `create_synreg_index_table()` before any `_truncate_*_index()`.
- All shard jobs: `target_instances=1`, no pip unless catboost/AG, no EAI unless catboost/AG.
