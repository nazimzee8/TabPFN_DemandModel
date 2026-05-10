You are helping the user operate the TabPFN DeepSet training pipeline on Snowflake SPCS.
Provide a concise runbook covering the sections below. Answer follow-up questions using
the details in each section.

---

## 1. Pipeline Phases

| Phase | Entrypoint | Stored Procedure | Output |
|---|---|---|---|
| Pretrain | `train.py` | `CALL run_pretrain_pipeline()` | `@MODEL_STAGE/checkpoints/pretrain.pt` |
| HPO | `hpo.py` | `CALL run_hpo_pipeline()` | `@MODEL_STAGE/hpo/best_config.json` |
| Final training | `train.py` | `CALL run_model_training()` | `@MODEL_STAGE/checkpoints/best.pt` |
| Benchmark prep | `prepare_benchmark_datasets.py` | `CALL prepare_benchmark_datasets()` | `@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json` |
| Evaluation | `evaluate.py` | `CALL run_evaluation_pipeline()` | `@EVALUATION_RESULTS_STAGE/model_comparison.csv` |
| Full pipeline | Pretrain → HPO → Final | `CALL run_training_pipeline()` | same as above |

Durations (GPU_NV_M): Pretrain ~45–90 min, HPO ~40–50 min, Final training ~45–90 min, Evaluation ~5–10 min.

---

## 2. Epoch Calibration (Run Before Pool Upgrade)

Upload `src/*.py` to `@EPOCH_STAGE` first (SnowSQL):
```sql
PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @EPOCH_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

Run calibration jobs:
```sql
CALL run_hpo_epoch_test();
SELECT $1 FROM @EPOCH_STAGE/hpo_timing.json (FILE_FORMAT => (TYPE = JSON));

CALL run_train_epoch_test();
SELECT $1 FROM @EPOCH_STAGE/train_timing.json (FILE_FORMAT => (TYPE = JSON));
```

Decision gate on `hpo_timing.json → epoch_time_s`:
- `<= 20 s` → 4 nodes GPU_NV_M (8 concurrent trials, ~50 min HPO)
- `20–30 s` → 6 nodes GPU_NV_M (12 concurrent trials, ~45–60 min HPO)
- `> 30 s`  → re-evaluate; consider GPU_NV_L or reducing `num_trials`

---

## 3. Submission

**Preferred (split):**
```sql
CALL run_pretrain_pipeline();
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain[.]pt';
CALL run_hpo_pipeline();
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));
CALL run_model_training();
LIST @MODEL_STAGE/checkpoints/;
CALL run_evaluation_pipeline();   -- runs prep + shards + aggregate internally
```

**Run benchmark prep manually (optional — run_evaluation_pipeline calls it automatically):**
```sql
CALL prepare_benchmark_datasets();
LIST @META_DATASET_STAGE/benchmark_prepared/;
SELECT $1 FROM @META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json (FILE_FORMAT => (TYPE=JSON));
CALL run_evaluation_pipeline();
```

**Convenience wrapper (single call):**
```sql
CALL run_training_pipeline();
```

**Kaggle data download (one-off setup):**
```sql
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
```

---

## 4. Key Configuration

| Parameter | Value | Location |
|---|---|---|
| GPU pool | GPU_NV_M | `sql/run_training_job.sql` |
| Pool MAX_NODES | 10 | `sql/run_training_job.sql` |
| HPO target_instances | 5 | `run_training_job.py`, `run_hpo_job.py` |
| num_samples | 20 | `src/hpo.py` (tune.run) |
| resources_per_trial | `{"gpu": 1}` | `src/hpo.py` |
| Training target_instances | 10 (TRAIN_NUM_NODES) | `run_model_training_job.py`, `run_pretrain_job.py` |
| num_workers_per_node | 4 | `src/train.py` (PyTorchScalingConfig) |
| Total DDP world_size | 40 (10 × 4) | derived |
| num_trials | 40 | `src/hpo.py` |
| epochs per trial | 30 | `src/hpo.py` |
| `CHECKPOINT_OUTPUT_NAME` | `pretrain.pt` (Phase 1) or `best.pt` (Phase 3) | env_vars in `run_pretrain_job.py`, `run_training_job.py` |
| `PRETRAIN_CHECKPOINT_PATH` | `@MODEL_STAGE/checkpoints/pretrain.pt` | env_vars in `run_model_training_job.py`, `run_training_job.py` |
| `EXPECTED_TRAIN_WORLD_SIZE` | `40` (TRAIN_NUM_NODES × 4) | `run_pretrain_job.py`, `run_model_training_job.py` |
| `STRICT_WORLD_SIZE_CHECK` | `true` | `run_pretrain_job.py`, `run_model_training_job.py` |

---

## 5. Stage Layout

| Stage | Contents |
|---|---|
| `@META_DATASET_STAGE/train/` | Training parquet files |
| `@META_DATASET_STAGE/val/` | Validation parquet files |
| `@META_DATASET_STAGE/test/` | Test parquet files |
| `@META_DATASET_STAGE/kaggle/` | Kaggle .npz benchmark datasets |
| `@META_DATASET_STAGE/benchmark_prepared/` | `benchmark_manifest.json` + prepared `.npz` files for all benchmark datasets |
| `@MODEL_STAGE/scripts/` | All `src/*.py` + `scripts/*.py` (job entrypoints) |
| `@MODEL_STAGE/hpo/` | `best_config.json` (or `hpo_failure.json`) |
| `@MODEL_STAGE/checkpoints/` | `best.pt` |
| `@EVALUATION_RESULTS_STAGE/synthetic/` | `test_report.csv`, `mc_report.csv` |
| `@EVALUATION_RESULTS_STAGE/benchmark_parts/` | Per-method detailed CSVs |
| `@EVALUATION_RESULTS_STAGE/` | `model_comparison.csv` (canonical output) |
| `@EPOCH_STAGE/` | All `src/*.py` + `hpo_timing.json` + `train_timing.json` |
| `@MLJOB_PAYLOAD_STAGE` | MLJob payload (managed by `submit_from_stage`) |

Upload scripts (SnowSQL, re-run on any change):
```sql
PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @EPOCH_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

---

## 5b. Data Loading Architecture (Train Jobs)

Training shards `META_DATASET_INDEX` directly by DDP `rank` and `world_size` in SQL.
`ShardedDataConnector` is NOT the production training path; it is only used in
`train_epoch_test.py` for epoch calibration.

**In `train_fn()` (every DDP worker):**
```python
train_rows = select_rank_sharded_index_rows("train", rank=rank, world_size=world_size)
val_rows   = select_rank_sharded_index_rows("val",   rank=rank, world_size=world_size)
files_by_split = materialize_indexed_meta_dataset(DATA_DIR, rows=train_rows + val_rows)
```

**Sharding SQL:** `WHERE MOD(ROW_NUMBER() OVER (PARTITION BY split ORDER BY task_id) - 1, world_size) = rank`

| Split | Total | Workers | Per worker |
|-------|-------|---------|------------|
| train | 800   | 40      | 20 (exact) |
| val   | 100   | 40      | 2–3        |

HPO uses `select_meta_dataset_index_rows()` + `materialize_indexed_meta_dataset()` in
the driver only (before `tune.run()`).

## 5c. Evaluation Parallelism

| Workload | Pool | Mode | Jobs |
|---|---|---|---|
| Synthetic eval | GPU | Single-process | 1 node |
| Benchmark dataset prep | CPU | Single-process | 1 node |
| DeepSet benchmark | GPU | Independent shards | 10 × 1 node |
| Each CPU baseline | CPU | Independent shards | 3 × 1 node |
| AutoGluon | AUTOGLUON_CPU | Independent shards | 30 × 1 node |
| Aggregate | CPU | Single-process | 1 node |

`external_access_integrations` for OpenML now only applies to the prep job (`prepare_benchmark_datasets.py`). Benchmark shard jobs use `BENCHMARK_EXTERNAL_ACCESS` only for pip install of scikit-learn/xgboost/etc.

`submit_from_stage(target_instances=N)` does not set PyTorch distributed env vars.
CPU benchmarks use `BENCHMARK_NUM_SHARDS` / `BENCHMARK_SHARD_INDEX` instead of
`dist.init_process_group()`.

---

## 6. Monitoring

```sql
-- Check compute pool status
SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';

-- List running/completed job services
SHOW JOB SERVICES IN COMPUTE POOL DEEPSET_GPU_POOL;

-- Get logs for a specific job service (replace <job_service_name>)
CALL SYSTEM$GET_SERVICE_LOGS('<job_service_name>', 0, 'main', 1000);

-- Or from Python (in stored procedure context):
-- job.get_logs()
```

---

## 7. Checking Outputs

```sql
-- HPO output
LIST @MODEL_STAGE/hpo/;
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));

-- Training checkpoint
LIST @MODEL_STAGE/checkpoints/;

-- Evaluation results
LIST @EVALUATION_RESULTS_STAGE/;

-- Epoch calibration results
LIST @EPOCH_STAGE/;
SELECT $1 FROM @EPOCH_STAGE/hpo_timing.json (FILE_FORMAT => (TYPE = JSON));
SELECT $1 FROM @EPOCH_STAGE/train_timing.json (FILE_FORMAT => (TYPE = JSON));
```

Download outputs (SnowSQL):
```sql
GET @MODEL_STAGE/checkpoints/best.pt 'file://C:/Documents/TabPFN_DemandModel/results/';
GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: 517003` | `scale_cluster()` called inside MLJob | Removed from `hpo.py`; never add it back. Set `target_instances` at submission time. |
| `CUDA out of memory` | Batch too large or model too wide | Reduce `d_phi`/`d_rho`, lower `max_concurrent_trials` to 1, or switch to GPU_NV_L. |
| `FileNotFoundError: DATA_DIR` | Stage materialization failed | Check for rank-sharded row errors. `LIST @META_DATASET_STAGE/train/`. Verify `META_DATASET_INDEX` is populated. |
| `RuntimeError: World-size mismatch: expected 40 but ... reports N` | `STRICT_WORLD_SIZE_CHECK=true` and actual world_size ≠ 40 | Verify `target_instances` in `submit_from_stage()` equals `num_nodes` in `PyTorchScalingConfig` (both must be 10). Check pool has all nodes healthy. |
| `ValueError: RANK expected, but not set` | `submit_from_stage(target_instances=N)` without `PyTorchDistributor` does not set `RANK` | Use `BENCHMARK_NUM_SHARDS` + `BENCHMARK_SHARD_INDEX` for CPU benchmarks. |
| `NameError: train_job` | `submit_from_stage(...)` result not assigned | Assign: `train_job = submit_from_stage(...)`. |
| HPO `float() argument must be a string...` | Raw tuple/list in search space | Use `loguniform()`, `uniform()`, `choice()` from `snowflake.ml.modeling.tune`. |
| `CALL run_hpo_pipeline()` uses wrong node count | `run_hpo_job.py` stale `target_instances` | Must be `target_instances=6`. |
| `~/.config.toml permission denied` in Tuner | HOME not set | Pass `session=session` explicitly to `Tuner`; set `HOME=/tmp` in env_vars. |
| Kaggle 401 Unauthorized | Wrong Kaggle username (use handle, not email) | Recreate `KAGGLE_API_SECRET` with correct username/key; re-run `CALL download_kaggle_to_stage()`. |
| OpenML fetching fails in shard jobs | `ALLOW_BENCHMARK_RUNTIME_FETCH` not set | Run `CALL prepare_benchmark_datasets()` first; check `BENCHMARK_MANIFEST_PATH` env var is passed to shards. |

**`run_hpo_job.py` vs `run_training_job.py`**: `run_hpo_job.py` handles only
`CALL run_hpo_pipeline()` (HPO alone). `run_training_job.py` handles
`CALL run_training_pipeline()` (HPO + training), `CALL run_hpo_epoch_test()`,
`CALL run_train_epoch_test()`, and `CALL download_kaggle_to_stage()`.
`run_evaluation_test.py` handles `CALL run_evaluation_pipeline()` (was in `run_training_job.py`).

---

## 9. Cost Reference (GPU_NV_M)

| Phase | Nodes | Credits/node/hr | Duration | Approx. cost |
|---|---|---|---|---|
| HPO (20 trials × 1 round) | 5 | 1.42 | ~40–50 min | ~$4.73–5.92 |
| Full training (DDP, world_size=40) | 10 | 1.42 | ~25–50 min | ~$5.92–11.83 |
| Evaluation | 1 | 1.42 | ~5–10 min | ~$0.12–0.24 |
| **Total** | | | **~90–150 min** | **~$10–16** |

- Pool auto-suspends after 300 s idle (no charge in `SUSPENDED` state).
- Calibrate with `CALL run_hpo_epoch_test()` before committing to full HPO run.
- HPO cost dominates; reducing `num_trials` or shortening epoch count cuts cost proportionally.
