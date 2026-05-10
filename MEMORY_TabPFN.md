# Project Memory

## Snowflake Stage Ownership

- `@META_DATASET_STAGE`: train/val/test synthetic parquet datasets and staged benchmark datasets.
- `@MODEL_STAGE/scripts/`: all runnable MLJob code from `src/*.py` and `scripts/*.py`.
- `@MODEL_STAGE/hpo/`: `best_config.json` on HPO success and `hpo_failure.json` on Python-started HPO failure.
- `@MODEL_STAGE/checkpoints/`: model checkpoints, especially `best.pt`.
- `@EVALUATION_RESULTS_STAGE`: all evaluation reports, per-method benchmark parts, and comparison CSVs.
- `@EPOCH_STAGE`: output-only epoch calibration artifacts (`hpo_timing.json`, `train_timing.json`, and error JSONs).
- `@MLJOB_PAYLOAD_STAGE`: MLJob payload stage managed by `submit_from_stage`.

Canonical benchmark outputs: `@EVALUATION_RESULTS_STAGE/model_comparison.csv` and
`@EVALUATION_RESULTS_STAGE/model_comparison_summary.csv`.

## Snowflake-Only Training Guardrails

- Never download or materialize `@META_DATASET_STAGE` to the local workstation.
- MLJobs may materialize staged parquet only inside Snowflake container-local `/tmp/data`.
- Use `auto_compress=False` for JSON, checkpoint, CSV, and NPZ stage uploads.
- Pass Snowflake secrets into MLJob containers through `spec_overrides`; do not fetch secret values inside scripts.
- Never use `MIN_NODES = 0` for Snowflake compute pools; use `MIN_NODES = 1` with suspend settings for cost control.
- `submit_from_stage.stage_name` is a bare payload stage name such as `MLJOB_PAYLOAD_STAGE`, not an `@STAGE` path.
- Snowflake MLJob secrets use `spec.containers[].secrets[]`, not Kubernetes-style `env.valueFrom`.
- Benchmark jobs must install their dependencies and fail loudly if any dependency is unavailable.
- Benchmark ranking broke once when `predict_autogluon()` was inserted inside `add_rank_columns()` before the rank loop; keep aggregation smoke tests.
- DeepSet benchmark uses `DeepSetModel-MC bounded-context ensemble`, not exact
  full-context DeepSet inference. Snowflake OOM occurred when MC dropout forwarded
  the full 90% processed train split with the full test split. The fixed evaluation
  path still evaluates every assigned dataset, seed, and test row: 90/10 split
  first, train-only preprocessing, five deterministic 200-row contexts sampled
  only from processed train, same processed full test split per context, 128-row
  test chunks for memory only, average prediction vectors, then compute metrics
  once. Use `MC_K=8` for the first stable full run; upgrade to `MC_K=16` only after
  memory/runtime stability is proven.
- Upload scripts with `PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;` and `PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;`.
- `@EPOCH_STAGE` is not a code source. Epoch calibration MLJobs also load code from `@MODEL_STAGE/scripts/`.

## Metadata Index Guardrails

- `META_DATASET_INDEX` is required before HPO and training. It is a metadata pruning layer over staged parquet payloads, not a copy of those payloads.
- Rebuild it with `CALL build_meta_dataset_index();` whenever synthetic parquet is regenerated or restaged.
- The index build validates full split counts: `train=800`, `val=100`, `test=100`.
- HPO uses a deterministic balanced subset from the index: `train=200`, `val=40`, selected by `(split, hpo_bucket)` ranking.
- Snowflake jobs should fail fast if `META_DATASET_INDEX` is missing, empty, lacks required columns, returns invalid rows, or cannot provide the required HPO subset.

## Snowflake HPO Guardrails

- HPO driver-side selection can successfully produce `hpo_rows`, while Ray trial workers may still lack an active Snowflake session. `materialize_indexed_meta_dataset(rows=...)` must never silently fall back to local files inside Snowflake HPO; explicit indexed rows require Snowflake stage access in the worker.
- Canonical signature for the old misleading fallback: `FileNotFoundError: Local fallback needs at least 200 train parquet files under /tmp/data/train; found 0`.
- Grafana plugin key DNS errors in service logs are incidental noise unless they appear with actual HPO trial failures or Python tracebacks.
- Snowflake ML `TunerConfig.max_concurrent_trials` is per node, not total cluster-wide concurrency.
- Current GPU pool is `DEEPSET_GPU_POOL` on `GPU_NV_M`: 4 A10G GPUs per node, `MAX_NODES=10`. GPU_NV_L is not provisioned on this account.
- Canonical HPO submission: `target_instances=5` in both `scripts/run_training_job.py` and `scripts/run_hpo_job.py`.
- Canonical HPO tuner config: `num_trials=20`, `max_concurrent_trials=4`, and `resource_per_trial={"GPU": 1}`. 5 nodes Ã— 4 concurrent/node = 20 parallel one-GPU trials = 1 round of 20 trials.
- Ray Tune metric reporting must be metrics-dict style for every metric, e.g. `tune.report({"val_mse": value})` or the local compatibility helper. Do not use keyword-style `tune.report(val_mse=value)` in Snowflake HPO; Snowflake's Ray runtime can raise `TypeError: report() got an unexpected keyword argument 'val_mse'`.
- Preserve metric keys exactly, especially `val_mse`: `tune.run(metric="val_mse", mode="min")` selects the best trial by that key, and downstream `best_config.json` must remain `{lr, weight_decay, d_phi, d_rho, dropout, pool}`.
- For HPO launched via `submit_from_stage`, parallel GPU capacity comes from the MLJob `target_instances` setting. Do not call `scale_cluster()` from `hpo.py` in this path.
- `scale_cluster()` failed inside an MLJob/JOB service with `Multi node head service does not have a valid service type` / error 517003; keep HPO scaling at submission time.
- Omitting `resource_per_trial={"GPU": 1}` is not a workaround. Snowflake does not allocate GPU resources to Tuner trials automatically, so GPU HPO trials must request GPUs explicitly.
- If HPO trials are stuck or pending, first suspect over-requested GPU resources or insufficient compute-pool capacity. If trials run on CPU, first suspect a missing explicit GPU resource request.
- Inspect `@MODEL_STAGE/hpo/hpo_failure.json` first for HPO failures. Use service logs only if that JSON artifact is missing or incomplete, which suggests failure before or outside Python artifact upload.
- `debug/hpo_failure.json` is a **locally-downloaded snapshot** â€” it is not auto-synced from the stage. Always compare `LIST @MODEL_STAGE/hpo/` timestamps before concluding `main()` was not reached.
- Do **not** add `uses_snowflake_trainer` (or other undocumented kwargs) to `TunerConfig`. The documented parameters are: `metric`, `mode`, `search_alg`, `num_trials`, `max_concurrent_trials`, `resource_per_trial`. Undocumented kwargs raise `TypeError` on library version bumps inside `main()`.
- Legacy Snowflake ML Tuner path only: `ctx.report()` inside `train_for_hpo()` must pass `model=model.to("cpu")`. Without it, `tuner.run()` raises `TypeError: Path must be a string` when loading `TunerResults.best_model` after all trials finish.
- Legacy Snowflake ML Tuner path only: `train_for_hpo()` must NOT wrap the model with `torch.compile()`. Snowflake pickle-serialises the model passed to `ctx.report(model=)`; compiled `OptimizedModule` objects are not picklable â€” all trials fail silently to store an artifact â†’ `best_model_path = None` â†’ same TypeError. `hpo_epoch_test.py` uses a plain uncompiled model and is the reference baseline.
- `TunerResults.best_result` is a DataFrame; read the first row and prefer `config/<param>` columns, with raw parameter names only as compatibility fallback.
- Before rerunning canceled HPO, confirm `hpo.py`, `train.py`, `model.py`, `snowflake_io.py`, `run_hpo_job.py`, `run_model_training_job.py`, and `run_training_job.py` are present under `@MODEL_STAGE/scripts/` and are not only `.gz` duplicates.
- `SnowparkSQLException: 000603 (XX000) / Processing aborted due to error 300002` at `job.wait()` means the SPCS service terminated abnormally â€” the container crashed before reaching a clean terminal state, so `DESCRIBE SERVICE` returns 300002 instead of a status row. Diagnostic: (1) check `@MODEL_STAGE/hpo/hpo_failure.json` â€” if present it contains the Python traceback; (2) if absent, the container was killed before hpo.py's exception handler ran â€” inspect Snowsight container logs for OOM or pre-Python crash details. `_wait_done()` in `run_hpo_job.py` and `run_training_job.py` now catches 300002/000603 and re-raises as `RuntimeError` with this diagnostic guidance.

## Split Training Guardrails

- Prefer split procedures: `CALL run_pretrain_pipeline();` â†’ `CALL run_hpo_pipeline();` â†’
  `CALL run_model_training();`.
- `run_model_training()` reads `@MODEL_STAGE/hpo/best_config.json`, passes it as `BEST_CONFIG`,
  and writes `@MODEL_STAGE/checkpoints/best.pt`.
- Canonical training topology: `TRAIN_NUM_NODES=10`, `num_workers_per_node=4`,
  `world_size=40` (10 Ã— 4). Both pretrain and final training use this topology.
- Full training uses the full indexed train/val splits; HPO uses the 200/40 subset.
- Canonical production training shards `META_DATASET_INDEX` directly by DDP `rank` and
  `world_size` with SQL `ROW_NUMBER() OVER (PARTITION BY split ORDER BY task_id) - 1`
  and `MOD(rn, world_size) = rank`. Do not use worker-side
  `ShardedDataConnector.get_shard().to_pandas()` in pretrain or final training.
- The `SF_PYTORCH` / `Multi-node training requires a stage...` message is informational
  when `artifact_stage_location` was omitted. Production training must call
  `distributor.run(artifact_stage_location="TABPFN_DB.TABPFN_SCHEMA.MODEL_STAGE")`.
- The split-training root cause was connector shard conversion/materialization around
  `shard.to_pandas()` after the log line `Loading data into a pandas dataframe`, not
  the standard 800-task divisibility contract.
- After this fix, restage at least `src/train.py` and `src/snowflake_io.py`:
  `PUT file://C:/Documents/TabPFN_DemandModel/src/train.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;`
  and
  `PUT file://C:/Documents/TabPFN_DemandModel/src/snowflake_io.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;`.
- **`EXPECTED_TRAIN_WORLD_SIZE`** (env var, set to `40`): expected DDP world size. Logged on
  rank 0 at startup in `train_fn`. Set explicitly in both submission scripts.
- **`STRICT_WORLD_SIZE_CHECK`** (env var, `true`): if `true` and actual world_size differs from
  `EXPECTED_TRAIN_WORLD_SIZE`, `train_fn` raises `RuntimeError` before data materialization.
  Set to `true` in both `run_pretrain_job.py` and `run_model_training_job.py`.

## Epoch Calibration Guardrails

- `run_hpo_epoch_test()` writes `@EPOCH_STAGE/hpo_timing.json`; `run_train_epoch_test()` writes `@EPOCH_STAGE/train_timing.json`.
- Read `hpo_timing.json` by phase. HPO wall time includes MLJob startup, Ray/Tuner scheduling, metadata selection, stage materialization, and epoch compute; the epoch timing alone is not total HPO runtime.
- For the current design, the intended decision gate is 5 x `GPU_NV_M` for HPO: 20 trials / 20 concurrent = 1 round Ã— 30 epochs â‰ˆ 31.6 min.
- Before epoch calibration, verify `hpo_epoch_test.py`, `train_epoch_test.py`, `train.py`, `model.py`, and `snowflake_io.py` exist under `@MODEL_STAGE/scripts/`.

## Kaggle Snowflake Download Troubleshooting

- Log line `Loaded Kaggle credentials from MLJob secret environment.` means Snowflake injected non-empty secret values at runtime.
- `401 Unauthorized` from `DownloadDataFiles` means Kaggle rejected the username/token; recreate `KAGGLE_API_SECRET` using the exact Kaggle username and API token.
- `403 Forbidden` from `DownloadDataFiles` means Kaggle authenticated the account but blocked competition file download; accept the competition rules while logged into the same Kaggle account used by the token.
- Accept rules for these Kaggle Playground Series competitions before downloading:
  - https://www.kaggle.com/competitions/playground-series-s3e3/rules
  - https://www.kaggle.com/competitions/playground-series-s3e5/rules
  - https://www.kaggle.com/competitions/playground-series-s3e9/rules
  - https://www.kaggle.com/competitions/playground-series-s3e22/rules
  - https://www.kaggle.com/competitions/playground-series-s3e26/rules
- After accepting rules, rerun:

```sql
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
```

## HPO / Ray Tune Guardrails

- HPO runs through Ray-based Snowflake ML Jobs, not Snowflake Tuner.
- HPO uses 20 trials total.
- HPO target topology is 5 nodes × 4 GPUs/workers per node = 20 concurrent one-GPU trial slots.
- Pretraining and final training target topology is 10 nodes × 4 workers per node.
- HPO Ray workers must never perform Snowflake I/O.
- All Snowflake stage access, Snowpark session creation, metadata selection, checkpoint download, and data materialization must happen only in the HPO driver before tune.run().
- Ray workers must consume training/validation records and pretrain checkpoint payloads through Ray object store.
- The Snowpark-loaded-in-worker warning is a guardrail warning, not automatically a fatal error.
- For Ray Tune metric reporting, never use bare keyword-style tune.report(val_mse=...).
- Always report HPO metrics using a Ray-version-compatible dictionary-based helper, preferring ray.train.report({"val_mse": value}) and falling back to tune.report({"val_mse": value}) or legacy tune.report(**metrics).
- HPO success requires every successful trial to report the metric key "val_mse", because tune.run() selects best_config using metric="val_mse", mode="min".
- If tune.run() completes but analysis.best_config is None, first suspect that all trials failed before reporting "val_mse".
- Do not treat GPU capacity as the root cause when logs show 20.0/20 GPUs allocated and all trials RUNNING.
- Do not treat missing HPO data as the root cause when worker logs show train=200 and val=40 records.
- Do not treat missing pretrain checkpoint as the root cause when worker logs show "[HPO trial] Loaded pretrain checkpoint from Ray object store."
- Keep HPO architecture fixed to the pretrain checkpoint unless intentionally regenerating pretrain artifacts: d_phi=128, d_rho=256, pool="pna".
- Any future Claude change to hpo.py must preserve driver-only Snowflake I/O and object-store worker payloads.

## Final Training Runtime / Prometheus Guardrails

- HPO is currently working. Do not rewrite HPO when debugging final model training unless final training explicitly fails while reading @MODEL_STAGE/hpo/best_config.json.
- Final model training is launched by run_model_training_job.py and uses train.py as the MLJob entrypoint.
- Final training target topology is 10 nodes × 4 workers per node = 40 PyTorch workers.
- run_model_training_job.py must keep target_instances equal to TRAIN_NUM_NODES.
- train.py must keep PyTorchScalingConfig num_nodes equal to TRAIN_NUM_NODES and num_workers_per_node=4.
- EXPECTED_TRAIN_WORLD_SIZE must equal TRAIN_NUM_NODES × 4.
- STRICT_WORLD_SIZE_CHECK must remain true for final model training.
- Absence of train_failure.json means train.py's Python-side failure handler likely did not complete. Treat that absence as a diagnostic signal, not an indication that training succeeded.
- If logs show "Unable to create mmap-ed active query log", "Failed to mmap", "activeQueryTracker", or "data/queries.active", classify the failure boundary as Snowflake MLJob/Ray runtime startup before train_fn unless train_fn logs are also present.
- Do not diagnose failures as DataLoader, DDP, NCCL, model architecture, pretrain checkpoint, or world-size issues unless logs show train.py/train_fn was entered.
- The required boundary log sequence is: "[train.py main] entered main" → "[train.py main] starting PyTorchDistributor.run" → "[train_fn] entered train_fn" → "[train_fn] topology:".
- If "[train_fn] topology:" appears, then world-size and worker topology can be diagnosed from actual logs.
- If "[train_fn] topology:" does not appear, do not claim a world-size mismatch.
- Prometheus mmap panic is an infrastructure/runtime startup symptom. No verified env var exists to disable Prometheus or Ray dashboard through submit_from_stage. Do not add speculative RAY_DISABLE or include_dashboard env vars.
- If the Prometheus mmap panic persists after diagnostics confirm the failure boundary, escalate as a Snowflake MLJob/Ray runtime infrastructure issue rather than rewriting model code.
- Always print diagnostic JSON payloads to stdout before attempting Snowflake stage upload so diagnostics survive even if stage upload fails.
- Search container logs for "[TRAINING FAILURE JSON]" — full failure payload is printed before upload attempt.
- Search container logs for "[train.py main] entered main" — confirms train.py was executed in the container.
- Query LIST @MODEL_STAGE/checkpoints/ PATTERN='.*training_submission_started[.]json' — its presence confirms the stored procedure reached submit_from_stage; the train.py version additionally confirms train.py main() was reached.
- train_failure.json is uploaded when distributor.run(...) raises inside train.py's Python exception handler; logs must still contain the full failure JSON even if upload fails.
- Final training should warm-start from @MODEL_STAGE/checkpoints/pretrain.pt when present.
- Final training writes @MODEL_STAGE/checkpoints/best.pt on success.
- Do not reduce final training from 10 nodes to work around startup issues unless explicitly instructed.
- Failure boundary diagnosis sequence: (1) no "[train.py main] entered main" → failure before train.py main; (2) no "[train.py main] starting PyTorchDistributor.run" → failure inside train.py main setup; (3) no "[train_fn] entered train_fn" → failure inside PyTorchDistributor/Ray worker launch; (4) no "[train_fn] topology:" → failure around get_context(); (5) topology present then failure → diagnose from actual train_fn error.
- The canonical final-training topology is different from HPO: final training uses 10 nodes, HPO uses 5 nodes for 20 one-GPU trials.

## Snowflake MLJob Runtime Startup Guardrails

### Prometheus mmap startup failure markers

If Snowflake MLJob logs contain any of:
- `Failed to mmap`
- `data/queries.active`
- `activeQueryTracker`
- `Unable to create mmap-ed active query log`

treat the failure as a Snowflake MLJob/Ray/Prometheus runtime startup failure unless Python
training boundary markers prove otherwise.

### Boundary markers

- `[train.py main] entered main` — train.py executed in the container.
- `[train.py main] starting PyTorchDistributor.run` — distributor about to launch.
- `[train_fn] entered train_fn` — distributed training function reached; valid to debug DDP/model.
- `[train_fn] topology` — worker topology visible; valid to debug world-size issues.
- `[TRAINING FAILURE JSON]` — Python-side exception handler executed.
- `[runtime_probe] entered Python` — runtime probe reached user Python code.
- `[runtime_probe] completed` — runtime probe finished successfully.

### Guardrail

Do not debug model architecture, DDP, NCCL, dataset sharding, parquet materialization, or HPO
configuration until Python boundary markers prove the failure reached that layer.

### Runtime pinning

Training/HPO/runtime-probe submissions use the account default unless explicitly
changed. Evaluation submissions are intentionally pinned through required env
vars: `PREP_RUNTIME_ENVIRONMENT`, `BENCHMARK_RUNTIME_ENVIRONMENT`, and
`AUTOGLUON_RUNTIME_ENVIRONMENT`. `run_evaluation_test.py` must pass those values
as `runtime_environment` and must not use per-job `pip_requirements`.

### Runtime probe workflow

Before debugging training code when infrastructure startup failures are suspected:
1. `CALL run_training_runtime_probe(1);` — single-node probe.
2. Confirm `[runtime_probe] entered Python` appears in job logs.
3. If single-node passes, `CALL run_training_runtime_probe(10);` — full-topology probe.
4. If full-topology passes, `CALL run_model_training();`.
5. If either probe fails before `[runtime_probe] entered Python` and Prometheus mmap markers appear,
   escalate to Snowflake Support as a managed MLJob/Ray/Prometheus runtime issue.

### train_failure.json interpretation

Absence of `train_failure.json` when Prometheus mmap panic appears is expected — the failure
occurred before Python-side exception handling could run. Do not interpret absence as proof that
diagnostics are broken.

### Topology invariants (do not change without explicit instruction)

- Final training: 10 nodes × 4 workers = 40 world size, `target_instances=10`.
- Pretraining: 10 nodes × 4 workers = 40 world size.
- HPO: 5 nodes × 4 concurrent trials = 20 total trial slots.

## Evaluation Parallelism Guardrails

- `submit_from_stage(target_instances=N)` does NOT inject `RANK`, `WORLD_SIZE`,
  `MASTER_ADDR`, or `MASTER_PORT`. Those are only set by `PyTorchDistributor`.
- Do not call `dist.init_process_group()` for CPU baselines or AutoGluon benchmark jobs.
- `target_instances > 1` is not equivalent to a valid PyTorch process group.
- CPU baseline benchmark topology is exactly 3 combined single-node shard jobs.
  Each baseline shard receives `BENCHMARK_METHODS=<all 9 baseline methods>` for
  XGBoost, LightGBM, CatBoost, RandomForest, KNN, LinearRegression, Ridge, SVR,
  and MLP. Per-method baseline shard submission is disallowed.
  `DEEPSET_CPU_POOL MAX_NODES=3` is the intended capacity boundary for these
  three combined baseline shard jobs.
- AutoGluon runs as independent single-node shard jobs. Each shard receives:
  - `BENCHMARK_NUM_SHARDS`: total shard count
  - `BENCHMARK_SHARD_INDEX`: this shard's 0-based index
  - `target_instances=1`
- `BENCHMARK_NUM_CPUS` controls AutoGluon `predictor.fit(num_cpus=...)` and
  defaults to `1`; sklearn baseline `n_jobs=1` settings stay unchanged.
- Each shard writes a unique part CSV:
  `benchmark_parts/<method>_shard{i}_of_{n}_detailed.csv`
- The aggregate job globs `benchmark_parts/*_detailed.csv*` and combines all shards.
- DeepSet GPU evaluation also uses independent shard jobs (`GPU_BENCHMARK_SHARDS=10`).
- Synthetic eval runs as a single-process job (1 GPU node, 100 test files).
- PyTorch distributed evaluation (`dist.init_process_group`) is only entered when all
  of `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` are set in the environment.
  If missing and `world_size > 1` is requested without shard mode, evaluate.py raises
  a clear RuntimeError.

## Benchmark Dataset and Dependency Boundary Guardrails

- Benchmark datasets (OpenML + Kaggle) must be fetched and staged exactly once before
  any model shard job runs. Run `CALL prepare_benchmark_datasets()` or let
  `run_evaluation_pipeline()` handle it automatically.
- `run_evaluation_pipeline()` must preflight
  `@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json` before
  submitting prep and must skip the prep MLJob when that manifest already exists.
- `prepare_benchmark_datasets.py` is the only production code path that may fetch
  OpenML datasets or normalize raw staged Kaggle data. It writes
  `@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json` and prepared
  `.npz` files under `@META_DATASET_STAGE/benchmark_prepared/{openml,kaggle}/`.
- Benchmark shard jobs consume prepared staged datasets only. They read
  `benchmark_manifest.json`, partition manifest entries by
  `BENCHMARK_NUM_SHARDS` and `BENCHMARK_SHARD_INDEX`, and only then download the
  prepared files assigned to that shard. Sharding happens before prepared dataset
  download.
- Shard-consumed prepared files must load with `np.load(..., allow_pickle=False)`.
  No Python objects are allowed in prepared files or the manifest.
- Production benchmark mode calls `run_prepared_benchmark()` only. `evaluate.py`
  must not import, require, or call OpenML APIs; OpenML is a preparation
  dependency only. Benchmark shard jobs must never use any OpenML/Kaggle fetch
  path directly.
- Benchmark shard jobs must not receive `BENCHMARK_EXTERNAL_ACCESS`; only
  dataset preparation and optional Kaggle setup jobs may use that integration.
- Runtime image boundary: `run_evaluation_test.py` must submit prep/evaluation
  MLJobs with `runtime_environment`, never with per-job `pip_requirements`.
  Required controls are `PREP_RUNTIME_ENVIRONMENT` for
  `prepare_benchmark_datasets.py`, `BENCHMARK_RUNTIME_ENVIRONMENT` for
  non-AutoGluon `evaluate.py` jobs (synthetic, DeepSet benchmark, baselines,
  aggregate), and `AUTOGLUON_RUNTIME_ENVIRONMENT` for AutoGluon shards.
  Dependency boundary still applies inside those runtime images: `openml`
  belongs only to prep; benchmark shard runtimes must not need OpenML/Kaggle
  fetch APIs.
- Prefer method-specific lazy imports in `evaluate.py` so an unrelated model
  package failure does not break a shard for another method.
- `BENCHMARK_EXTERNAL_ACCESS` is required for dataset preparation and optional
  Kaggle download/setup. It is not a signal that benchmark shards may fetch OpenML
  or Kaggle data at runtime.
- `BENCHMARK_FORCE_REBUILD=true` triggers a full reprepare of the benchmark stage.
  Use when OpenML study membership changes or Kaggle datasets are updated.
- The preparation job is idempotent by default: if `benchmark_manifest.json` already
  exists on stage and `BENCHMARK_FORCE_REBUILD=false`, the job validates and exits early.

## Checkpoint Serialization Guardrails

- Do not save `ModelConfig` or other custom Python objects directly in PyTorch checkpoints.
  In PyTorch 2.6+, `torch.load()` defaults to `weights_only=True` and refuses to unpickle
  custom classes.
- Canonical safe checkpoint format (`checkpoint_format_version=2`):
  ```python
  {
      "checkpoint_format_version": 2,
      "cfg": dataclasses.asdict(model.cfg),   # plain dict, not ModelConfig instance
      "state_dict": model.state_dict(),
      "metadata": {"source": "train.py", "pytorch_version": torch.__version__, ...},
  }
  ```
- `evaluate.py` loads checkpoints via `load_checkpoint_compat()`:
  1. `torch.load(..., weights_only=True)` — preferred; works for v2 checkpoints.
  2. `safe_globals([ModelConfig])` + `weights_only=True` — for legacy checkpoints.
  3. `weights_only=False` — only if `ALLOW_UNSAFE_TORCH_LOAD=true`; prints security warning.
- Do not default to `weights_only=False`. It can execute arbitrary code from a checkpoint.
- Future checkpoint metadata must contain only primitive-safe values (str, int, float, bool,
  None, or nested lists/dicts of those types). No custom objects in metadata.
- `train.py` uses `dataclasses.asdict(ckpt.cfg)` when saving so new checkpoints are v2-safe.
- After staging updated `train.py` and `evaluate.py`, newly trained checkpoints will load
  without any fallback. Existing legacy `best.pt` / `pretrain.pt` will be handled by the
  `safe_globals` fallback transparently.
