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
  first, train-only preprocessing, train-only feature selection capped by
  `BENCHMARK_DEEPSET_FEATURE_CAP` (default `model.cfg.d_phi`), five deterministic
  non-overlapping train-only context windows capped at 200 rows, same capped
  processed full test split per context, 128-row test chunks for memory only,
  average prediction vectors, then compute metrics once. Use `MC_K=8` for the
  first stable full run; upgrade to `MC_K=16` only after memory/runtime stability
  is proven.
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
changed. Evaluation submissions are intentionally pinned by passing three runtime
image names as explicit stored procedure arguments to `run_evaluation_pipeline()`.
The 3-runtime architecture (`PREP_RUNTIME_ENVIRONMENT`, `BENCHMARK_RUNTIME_ENVIRONMENT`,
`AUTOGLUON_RUNTIME_ENVIRONMENT`) is preserved. `BENCHMARK_RUNTIME_ENVIRONMENT` is
used for synthetic eval, DeepSet benchmark shards, CPU baseline shards, and the
aggregate job. `2.5.0-py311` is the known-good Snowflake-managed benchmark runtime;
it includes `torch`, `pyarrow`, `pandas`, `scipy`, `sklearn`, `xgboost`, `lightgbm`,
and CUDA, but does NOT include `catboost`.

CPU baseline shard jobs (carrying `BENCHMARK_METHODS` that include `CatBoost`) install
`catboost` at job submission time via `pip_requirements=["catboost"]`. All other
eval/prep/aggregate/AutoGluon jobs pass no `pip_requirements`.

`run_evaluation_test.py` must not rely on OS environment variables for runtime
image names in production — local shell variables, SnowSQL session variables, and
Snowsight worksheet variables do not become OS environment variables inside the
Python stored procedure runtime. `run_evaluation_test.py` exposes the selected
value in container env vars as `EVAL_RUNTIME_ENVIRONMENT`. Before
expensive evaluation/prep work, it verifies `@MODEL_STAGE/checkpoints/best.pt`,
requires `@MODEL_STAGE/scripts/runtime_probe.py`, preflights compute pools
(`DEEPSET_GPU_POOL`, `DEEPSET_CPU_POOL`, `AUTOGLUON_CPU_POOL`), and submits
`runtime_probe.py` on 5 probes: benchmark GPU, benchmark aggregate CPU, CPU baseline
(with `pip_requirements=["catboost"]`), prep CPU, and AutoGluon CPU, each with
runtime-specific `REQUIRED_IMPORTS`.
Evaluation runtime probes must remain serialized under the current Snowflake
node quota: submit one probe, wait for it to finish, then submit the next. Even
`target_instances=1` consumes account node quota, so do not fan out probes
concurrently across `DEEPSET_GPU_POOL`, `DEEPSET_CPU_POOL`, and
`AUTOGLUON_CPU_POOL` unless a higher node quota has been confirmed.

Compute pool preflight handles SUSPENDED pools:
- `SUSPENDED + AUTO_RESUME=TRUE` — returns immediately and lets `submit_from_stage()` trigger
  the resume on first job submission. No ALTER is issued.
- `SUSPENDED + AUTO_RESUME=FALSE` (or unknown) — issues `ALTER COMPUTE POOL X RESUME` and
  sleeps 10 s before returning. This is the explicit-resume path.
For development cost control: keep pools suspended between runs, keep `AUTO_RESUME=TRUE`,
and use a short `AUTO_SUSPEND_SECS` (60–300 s). Do not keep `DEEPSET_GPU_POOL` warm
between debugging sessions.

Evaluation dependency guardrail: never add a global benchmark dependency gate
that requires every baseline package for every shard. Dependency validation must
follow the selected `BENCHMARK_METHOD`/`BENCHMARK_METHODS`: scikit-learn is the
shared prepared-benchmark dependency for DeepSet, CPU baselines, and AutoGluon;
XGBoost, LightGBM, and CatBoost are required only for those exact methods.
`autogluon.tabular` stays lazily imported inside `predict_autogluon()`.

AutoGluon runtime probe required imports: `autogluon.tabular`, `numpy`, `pandas`,
`sklearn`, `scipy`, `pyarrow`, `torch`, and `snowflake.snowpark`. `xgboost`,
`lightgbm`, and `catboost` are intentionally excluded — AutoGluon shards run
`BENCHMARK_METHOD=AutoGluon`, which has no code path through those packages.
The probe must include `scipy`, `pyarrow`, and `torch` because `evaluate.py`
hard-imports them at module startup (not inside benchmark method branches), and
AutoGluon shards execute `evaluate.py`. If `evaluate.py` is later refactored to
lazy-import those modules, the AutoGluon probe list should be revisited.
Method-aware dependency validation in `evaluate.py` must remain intact regardless.

If `CALL run_evaluation_pipeline(...)` fails in under a few seconds with
`PREP_RUNTIME_ENVIRONMENT is required`, the procedure did not receive runtime
image names. Pass the three runtime image names as procedure arguments and
recreate the procedure if the old zero-argument signature is still installed:

  DROP PROCEDURE IF EXISTS run_evaluation_pipeline();
  -- Then recreate with three STRING arguments.

New canonical call:
  CALL run_evaluation_pipeline(
    '<prep_runtime_image_name>',
    '<benchmark_runtime_image_name>',
    '<autogluon_runtime_image_name>'
  );

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
- Runtime preflight probes are intentionally serial. A successful 10-node
  training run does not imply enough spare quota for concurrent evaluation
  probes across `DEEPSET_GPU_POOL`, `DEEPSET_CPU_POOL`, and
  `AUTOGLUON_CPU_POOL`; keep the submit-then-wait ordering unless Snowflake node
  quota is raised and verified.
- `run_evaluation_pipeline()` is phase-gated: DeepSet GPU shards (phase 3) must all
  finish before CPU baseline shards (phase 4) are submitted; CPU baseline shards must
  all finish before AutoGluon shards (phase 5) are submitted; all AutoGluon shards must
  finish before the aggregate job (phase 6). Phases 0, 1+2 (synthetic + prep), and 6
  are unchanged.
- `AUTOGLUON_BENCHMARK_SHARDS=30` total AutoGluon shard jobs; `AUTOGLUON_MAX_CONCURRENT_SHARDS=30`
  maximum in-flight at once. AutoGluon is submitted and waited in one full-concurrency batch when
  `AUTOGLUON_CPU_POOL MAX_NODES` and account quota allow it.
- `run_evaluation_capacity_probe()` is a lightweight pre-check that allocates nodes in
  the planned phase sizes (GPU=10, CPU=3, AutoGluon=30) using `capacity_probe.py` (no
  model, no benchmark data, no heavy imports — just a 30-second sleep). Phases are
  non-overlapping: GPU phase must finish before CPU phase starts; CPU phase must finish
  before AutoGluon phase starts. Recommended run order:
  1. `CALL run_evaluation_runtime_probes(...)` — validate runtime images
  2. `CALL run_evaluation_capacity_probe(...)` — validate node quota
  3. `CALL run_evaluation_pipeline(...)` — full evaluation
  If the capacity probe fails with a node limit error: `SHOW COMPUTE POOLS`; suspend
  idle pools; wait for active jobs to finish; or request higher Snowflake account node
  quota before retrying.
- PyTorch distributed evaluation (`dist.init_process_group`) is only entered when all
  of `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` are set in the environment.
  If missing and `world_size > 1` is requested without shard mode, evaluate.py raises
  a clear RuntimeError.
- **Split-phase evaluation design**: 5 independent stored procedures expose each benchmark
  phase — `run_evaluation_prep`, `run_deepset_evaluation`, `run_baseline_evaluation`,
  `run_autogluon_evaluation`, `run_evaluation_aggregation`. Each targets only its own
  compute pool and can be called and retried independently.
- **Manual pool suspend pattern**: completing a phase does NOT release its compute pool
  quota. The operator must issue `ALTER COMPUTE POOL <pool> SUSPEND` after each phase
  before calling the next phase. Do not skip this step under tight quota.
- **Do not collapse back into overlapping fan-out**: do not revert to a single procedure
  that holds all three pools simultaneously unless node quota has been raised and verified.
- **No runtime probes in split procedures**: `run_evaluation_runtime_probes()` must be
  called once by the operator before the first split-phase procedure. The split procedures
  themselves do not re-run probes.
- **`run_evaluation_pipeline()` sequencing change**: the monolithic pipeline was refactored
  to use the same phase helpers; synthetic eval and prep are now sequential (prep → deepset
  phase) rather than concurrent. Split procedures are the recommended path for tight quota.

## Benchmark Dataset and Dependency Boundary Guardrails

- Benchmark datasets (OpenML + Kaggle) must be fetched and staged exactly once before
  any model shard job runs. Run `CALL prepare_benchmark_datasets()` or let
  `run_evaluation_pipeline()` handle it automatically.
- `run_evaluation_pipeline()` must always submit the prep MLJob before benchmark
  shards as a lightweight manifest and `BENCHMARK_DATASET_INDEX` validation
  step. Existing valid manifests exit early inside `prepare_benchmark_datasets.py`;
  the orchestrator must not skip prep solely because `benchmark_manifest.json`
  exists.
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
- `_submit_eval()` in `run_evaluation_test.py` uses a module-level `_UNSET`
  sentinel as the default for its `pip_requirements` parameter. Do not remove
  this parameter or the sentinel. The Phase 4 baseline shard loop must always
  pass `pip_requirements=list(BASELINE_EXTRA_PIP_REQUIREMENTS)` explicitly at
  the call site — do not revert to implicit inference from `BENCHMARK_METHODS`.
  The explicit passing is the guarantee that `catboost` is installed before
  `evaluate.py` starts; removing it silently breaks baseline evaluation on any
  managed runtime that does not include `catboost`.
- Orchestration test invariants for `pip_requirements` (do not weaken):
  - 5 runtime probes total; probe at index 2 is the CPU baseline probe and must
    carry `pip_requirements == list(BASELINE_EXTRA_PIP_REQUIREMENTS)`.
  - Probes at indices 0, 1, 3, 4 must carry no `pip_requirements`.
  - All 3 baseline shard eval jobs (`BENCHMARK_METHODS == BASELINE_METHODS`) must
    carry `pip_requirements == list(BASELINE_EXTRA_PIP_REQUIREMENTS)`.
  - All other eval jobs (synthetic, DeepSet, AutoGluon, aggregate) must carry no
    `pip_requirements`.
- DeepSet benchmark detail rows must include `raw_features`,
  `processed_features`, `selected_features`, `feature_selector`, and
  `feature_cap`. The selector is deterministic train-only
  `train_f_regression`; CPU baselines and AutoGluon continue receiving the full
  processed matrices.
- CPU baselines, AutoGluon, and DeepSet feature selection must skip oversized
  processed rows instead of capping CPU comparison inputs or crashing. Defaults:
  `BENCHMARK_CPU_MAX_PROCESSED_FEATURES=2000`,
  `BENCHMARK_CPU_MAX_MATRIX_BYTES=536870912`, and
  `BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES=5368709120`. Skips must emit NaN
  benchmark rows with a machine-readable `skip_reason`.
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
- Canonical v2 checkpoints save `cfg` as a plain dict via `dataclasses.asdict(model.cfg)`.
  Consumers must normalize `ckpt["cfg"]` back to `ModelConfig` before comparing architecture
  fields. Do not compare dict-form checkpoint configs with `getattr(saved_cfg, field, None)`,
  because it returns `None` for dict keys and causes false architecture mismatches in HPO.
  Known failure signature: HPO reports `saved=None` for every architecture field while
  printing `saved={'d_phi': 128, ...}` in the same error payload. Fix consumer-side
  normalization first; do not regenerate `pretrain.pt` unless the checkpoint is missing `cfg`
  or has a truly incompatible architecture.
- `evaluate.py` loads checkpoints via `load_checkpoint_compat()`:
  1. `torch.load(..., weights_only=True)` — preferred; works for v2 checkpoints.
  2. `safe_globals([ModelConfig])` + `weights_only=True` — for legacy checkpoints.
  3. `weights_only=False` — only if `ALLOW_UNSAFE_TORCH_LOAD=true`; prints security warning.
- `ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS` in `run_evaluation_test.py` is **currently
  `"true"`** as a temporary escape hatch for the legacy `best.pt` checkpoint (pre-v2 pickle
  format that fails both `weights_only=True` paths in `load_checkpoint_compat()`). All eval
  jobs receive `ALLOW_UNSAFE_TORCH_LOAD=true` via `_submit_eval()`.
- **Guardrail**: whenever `best.pt` is a legacy pickle checkpoint (i.e., `load_checkpoint_compat()`
  requires `weights_only=False`), `ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS` MUST be
  `"true"`. Do not set it back to `"false"` until `best.pt` has been migrated and confirmed
  loadable with `weights_only=True` alone.
- Revert to `"false"` only after running:
  ```bash
  python scripts/migrate_checkpoint.py --stage-name MODEL_STAGE --name best.pt
  ```
  and verifying evaluation completes without the `[SECURITY WARNING] ALLOW_UNSAFE_TORCH_LOAD`
  log line. Also update the test assertion message in
  `tests/test_run_evaluation_test_orchestration.py`.
- `weights_only=False` can execute arbitrary code from a checkpoint. Only trusted internally
  generated checkpoints should be loaded this way. Never set this for third-party checkpoints.
- Migration command for Snowflake stage checkpoints:
  ```bash
  python scripts/migrate_checkpoint.py --stage-name MODEL_STAGE --name best.pt
  python scripts/migrate_checkpoint.py --stage-name MODEL_STAGE --name pretrain.pt
  # Backup written automatically to @MODEL_STAGE/checkpoints/best.pt.bak
  ```
- Migration command for local checkpoints:
  ```bash
  python scripts/migrate_checkpoint.py --path /tmp/best.pt
  # Backup written to /tmp/best.pt.bak
  ```
- Do not default to `weights_only=False`. It can execute arbitrary code from a checkpoint.
- Future checkpoint metadata must contain only primitive-safe values (str, int, float, bool,
  None, or nested lists/dicts of those types). No custom objects in metadata.
- `train.py` uses `dataclasses.asdict(ckpt.cfg)` when saving so new checkpoints are v2-safe.
- After staging updated `train.py` and `evaluate.py`, newly trained checkpoints will load
  without any fallback. Existing legacy `best.pt` / `pretrain.pt` must be migrated using
  `scripts/migrate_checkpoint.py` to eliminate the `[SECURITY WARNING] ALLOW_UNSAFE_TORCH_LOAD`
  log lines.

## Snowflake CatBoost Dependency Guardrails

- CatBoost is not preinstalled in the Snowflake-managed Container Runtime (`2.5.0-py311`).
  Always treat it as a custom PyPI dependency for ML Jobs.
- Every CatBoost MLJob submission must pass BOTH `pip_requirements` AND
  `external_access_integrations`. Passing only `pip_requirements` without the EAI will
  fail with pip network errors or `ModuleNotFoundError` even though `pip_requirements`
  was received by the container.
- The project EAI for PyPI access is `TABPFN_CATBOOST_PYPI_EAI`, created using
  `SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE`. Do not use `BENCHMARK_EXTERNAL_ACCESS` for
  PyPI — that integration is scoped to OpenML and Kaggle hosts only.
- Always pin the CatBoost version (`CATBOOST_VERSION = "1.2.10"`). Do not float to
  unpinned `"catboost"`.
- Notebook-level EAI toggles do not propagate into submitted ML Jobs. Never use
  notebook import success as proof that a submitted probe can import CatBoost.
- Do not use inline `pip install catboost` inside probe scripts or benchmark jobs.
  CatBoost must be installed via the submission-time `pip_requirements` mechanism.
- `CATBOOST_PYPI_EAI = "TABPFN_CATBOOST_PYPI_EAI"` is the canonical constant in
  `run_evaluation_test.py`. Both the CPU baseline probe (probe index 2) and all 3
  baseline shard eval jobs must carry this EAI.
- Test invariants: baseline probe (index 2) and all baseline shard jobs assert
  `external_access_integrations == [CATBOOST_PYPI_EAI]`; deepset and autogluon jobs
  assert no `external_access_integrations`.

---

## Snowflake pip Dependency Guardrails (2.5.0-py311 Runtime)

### Missing packages

The `2.5.0-py311` Snowflake-managed runtime does **not** include these packages;
they must be installed per-job:

| Package | Pinned version | Constant |
|---------|---------------|----------|
| `catboost` | `1.2.10` | `CATBOOST_VERSION` / `BASELINE_EXTRA_PIP_REQUIREMENTS` |
| `openml` | `0.15.1` | `OPENML_VERSION` / `PREP_EXTRA_PIP_REQUIREMENTS` |
| `autogluon.tabular` | `1.3.0` | `AUTOGLUON_VERSION` / `AUTOGLUON_EXTRA_PIP_REQUIREMENTS` |

> `openml` is resolved via **two paths**: the stored procedure `PACKAGES` clause (for
> `CALL prepare_benchmark_datasets()`) AND `pip_requirements` in the ML Job (for
> `run_evaluation_pipeline()` → `_submit_dataset_prep()`). Both paths are required.

### One EAI rule

`TABPFN_PYPI_EAI = "TABPFN_PYPI_EAI"` — used for **all** pip installs.
Replaces the former `TABPFN_CATBOOST_PYPI_EAI`. Created in `sql/run_training_job.sql` Step 2c.

### BENCHMARK_EXTERNAL_ACCESS vs PYPI_EAI

- `BENCHMARK_EXTERNAL_ACCESS` — runtime API calls (OpenML downloads, Kaggle API). Network egress to external services.
- `TABPFN_PYPI_EAI` — pip package install from PyPI. Uses `SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE`.
- The dataset prep job (`prepare_benchmark_datasets.py`) needs **both**.

### Canonical constants (scripts/run_evaluation_test.py)

```python
PYPI_EAI = "TABPFN_PYPI_EAI"
BENCHMARK_EXTERNAL_ACCESS_EAI = "BENCHMARK_EXTERNAL_ACCESS"

CATBOOST_VERSION = "1.2.10"
BASELINE_EXTRA_PIP_REQUIREMENTS = [f"catboost=={CATBOOST_VERSION}"]

OPENML_VERSION = "0.15.1"
PREP_EXTRA_PIP_REQUIREMENTS = [f"openml=={OPENML_VERSION}"]

AUTOGLUON_VERSION = "1.3.0"
AUTOGLUON_EXTRA_PIP_REQUIREMENTS = [f"autogluon.tabular=={AUTOGLUON_VERSION}"]
```

### Pinning rule

Always pin exact versions (`==`). Never use `>=` or unpinned requirements.
Update only when the managed runtime image changes or a security fix is required.

### Job allowlist / denylist

Jobs that **must** carry `pip_requirements` + `TABPFN_PYPI_EAI`:
- CPU baseline probe (probe index 2): `BASELINE_EXTRA_PIP_REQUIREMENTS`
- Prep CPU runtime probe (probe index 3): `PREP_EXTRA_PIP_REQUIREMENTS`
- AutoGluon CPU runtime probe (probe index 4): `AUTOGLUON_EXTRA_PIP_REQUIREMENTS`
- CPU baseline shard jobs (×3): `BASELINE_EXTRA_PIP_REQUIREMENTS`
- AutoGluon shard jobs (×30): `AUTOGLUON_EXTRA_PIP_REQUIREMENTS`
- `prepare_benchmark_datasets.py` ML Job: `PREP_EXTRA_PIP_REQUIREMENTS` +
  `["BENCHMARK_EXTERNAL_ACCESS", PYPI_EAI]`

Jobs that **must not** carry `pip_requirements`:
- GPU benchmark probe (0), CPU benchmark probe (1)
- DeepSet GPU shard jobs, synthetic eval job, aggregate job

### Test invariants

- `probe_jobs[2].pip_requirements == list(BASELINE_EXTRA_PIP_REQUIREMENTS)`
- `probe_jobs[3].pip_requirements == list(PREP_EXTRA_PIP_REQUIREMENTS)`
- `probe_jobs[4].pip_requirements == list(AUTOGLUON_EXTRA_PIP_REQUIREMENTS)`
- `probe_jobs[:2]` — no `pip_requirements`, no `external_access_integrations`
- `probe_jobs[2/3/4].external_access_integrations == [PYPI_EAI]`
- All baseline shard jobs: `external_access_integrations == [PYPI_EAI]`
- All AutoGluon shard jobs: `pip_requirements == list(AUTOGLUON_EXTRA_PIP_REQUIREMENTS)` and `external_access_integrations == [PYPI_EAI]`
- DeepSet, synthetic, aggregate jobs: no `pip_requirements`, no `external_access_integrations`
- prep ML Job: `external_access_integrations == [BENCHMARK_EXTERNAL_ACCESS_EAI, PYPI_EAI]`
  and `pip_requirements == list(PREP_EXTRA_PIP_REQUIREMENTS)`

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

## MarketAwareDeepSetModel Production Default Guardrails

- `MarketAwareDeepSetModel` is the default production model (`DEEPSET_MODEL_FAMILY=market_aware`). Do not revert to `"deepset"` without an explicit ablation reason.
- `load_best_deepset_checkpoint()` must always call `_instantiate_model(cfg)` — never hardcode the model class directly.
- Sanity checks must be called with the same device as training: `run_all_checks(model=model, device=torch.device("cuda:0"))`.
- `SYNREG_RUN_CHECKPOINT_GATES=true` and `SYNREG_CHECKPOINT_GATE_STRICT=true` are required defaults for all DeepSet synthetic regression shards. A failing gate (NaN/Inf, constant output, Ridge underperformance) invalidates evaluation results.
- `TRAIN_RUN_SANITY_CHECKS=true` and `TRAIN_SANITY_CHECK_STRICT=true` run structural checks before `torch.compile()` and DDP wrapping. Do not disable without explicit justification.
- `run_permutation_tests()` in `deepset_inference.py` is architecture-aware; dispatches to `market_aware` (Tests 1-5) or `deepset` (Tests 1-7) branches. Do not call with unknown `model_family`.
- `best_config.json` from HPO must include `model_family` (written by `hpo.py`). `train.py` reads it from `BEST_CONFIG` env var.
- Checkpoint `metadata` must include `best_val_mse`, `train_mse_at_best`, `best_epoch` for the train/val gap gate.
- **TRAINING_DATA_FAMILY must be explicit in all Snowflake production submissions.** `run_training_job.py`, `run_model_training_job.py`, and `run_hpo_job.py` all default to `synthetic_regression_combined` (combined suite `linear_all_v1`). Use `unknown` only for local/dev runs. Production synthetic regression evaluation checkpoints must be tagged `synthetic_regression_combined`. `train.py` raises `ValueError` at module load time for invalid values.
- **MODEL3 architecture selectors.** `MODEL_ARCH_VERSION="model2"` (default) preserves MODEL2 behavior. `MODEL3_DESIGN_PATTERN="inductive_forecasting"` (default) does NOT activate MODEL3 code unless `MODEL_ARCH_VERSION="model3"` is also set. All three submission scripts propagate `DEFAULT_MODEL_ARCH_VERSION` and `DEFAULT_MODEL3_DESIGN_PATTERN` through `env_vars`. MODEL3 combinations: `model3`+`inductive_forecasting`→`market_exchangeable_icl`; `model3`+`transductive_completion`→`market_exchangeable_completion`. MODEL3 checkpoints use format version 4.
- **MODEL3 must not mutate MODEL2 classes.** `MarketAwareDeepSetModel` is unchanged. MODEL3 is implemented as `MarketExchangeableICLModel` and `MarketExchangeableCompletionModel` with shared primitives (`ExchangeableMatrixBlock`, `ColumnEncoder`, `CellEncoder`, `_masked_mean`). Factory `_instantiate_model(cfg)` routes all four families.
