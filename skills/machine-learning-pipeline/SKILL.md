---
name: machine-learning-pipeline
description: Document and explain the DeepSet training and evaluation pipeline for this repo. Use when Codex needs to describe or update how synthetic meta-datasets are generated, how DeepSet is trained across many parquet datasets in Snowpark Container Services, how evaluate.py runs in the same SPCS container on unseen test data, or how the model is compared against the fixed ridge-regression baseline.
---

# Machine Learning Pipeline

## Overview

Use this skill to explain the research workflow implemented in this repo for training and evaluating the DeepSet demand model inside Snowpark Container Services (SPCS).

Ground every explanation in the current code:
- `generate_dgp.py` generates many single-task parquet files under `data/train`, `data/val`, and `data/test`.
- `train.py` trains a `DeepSetModel` over the train split, uses the validation split for early stopping, writes `best.pt`, and uploads that checkpoint to `@MODEL_STAGE/checkpoints/` when running inside SPCS.
- `hpo.py` selects a deterministic 200 train / 40 validation subset from
  `META_DATASET_INDEX`, runs a Ray worker Snowpark/session/stage preflight,
  materializes only those staged parquet payloads, and tunes only `lr`,
  `weight_decay`, and `dropout` with fixed architecture
  `d_phi=128`, `d_rho=256`, `pool="pna"`.
- `evaluate.py` loads only `best.pt` from `@MODEL_STAGE/checkpoints/`, runs permutation checks, writes synthetic reports under `results/synthetic/`, writes per-method benchmark files under `results/benchmark_parts/`, aggregates `results/model_comparison.csv`, and uploads evaluation CSVs to `@EVALUATION_RESULTS_STAGE/` when running inside SPCS.

## Explain The Data Layout

Describe the generated data as a collection of many meta-datasets rather than one flat training table.

State the generation contract from `generate_dgp.py`:
- Each parquet file contains one synthetic regression task.
- Each task stores `X_train`, `y_train`, `X_test`, `betaX_test`, `n`, `p`, `n_train`, `n_test`, and `prior_regime`.
- `betaX_test` is the noiseless target used for evaluation.
- With `--n_datasets 1000`, the script writes 800 training tasks, 100 validation tasks, and 100 test tasks.

When describing Snowflake execution, note that these parquet files are uploaded to
`@META_DATASET_STAGE`, indexed in `META_DATASET_INDEX`, and explicitly materialized
by MLJobs into ephemeral container-local `DATA_DIR` (default `/tmp/data`). HPO,
production training, and epoch calibration all choose files through
`META_DATASET_INDEX`; staged parquet remains the payload storage. Do not describe
local workstation downloads of `@META_DATASET_STAGE`.

## Explain The SPCS Execution Flow

Use this flow when summarizing the end-to-end pipeline:

1. Generate synthetic datasets locally with `python generate_dgp.py --n_datasets 1000 --out_dir data/`.
2. Upload `data/train/*.parquet`, `data/val/*.parquet`, and `data/test/*.parquet` to `@META_DATASET_STAGE`.
3. Upload Python scripts (`*.py`) to `@MODEL_STAGE/scripts/` via SnowSQL `PUT`.
4. Create and call the `run_training_pipeline()` Snowpark stored procedure (step 4 in
   `run_training_job.sql`). The procedure imports `run_training_job.py` from the stage
   and submits **three** sequential MLJobs:
   - **Phase 1 (Pretrain)**: `train.py` with `CHECKPOINT_OUTPUT_NAME=pretrain.pt` and no `BEST_CONFIG`. Trains with default hyperparameters; writes `@MODEL_STAGE/checkpoints/pretrain.pt`.
   - **Phase 2 (HPO)**: `hpo.py` with no explicit checkpoint env var. HPO requires `@MODEL_STAGE/checkpoints/pretrain.pt`, runs a Ray worker Snowpark/session/stage preflight before trials, and warm-starts every trial from the checkpoint. Writes `@MODEL_STAGE/hpo/best_config.json`.
   - **Phase 3 (Final training)**: `train.py` with `BEST_CONFIG` from Phase 2 and `PRETRAIN_CHECKPOINT_PATH=@MODEL_STAGE/checkpoints/pretrain.pt`. Fine-tunes the pre-trained model with the best hyperparameters; writes `@MODEL_STAGE/checkpoints/best.pt`.

5. Verify `@MODEL_STAGE/checkpoints/best.pt`, then call `run_evaluation_pipeline()`.
   Evaluation must not depend on `best_config.json`; `best.pt` is the handoff contract.
6. During training, save the best checkpoint to `best.pt` using `model._orig_mod.state_dict()` (unwrapped from `torch.compile`) and upload it to `@MODEL_STAGE/checkpoints/`.
7. During evaluation, load `best.pt`, run permutation-invariance checks, materialize
   the held-out test split inside the Snowflake container, evaluate `/tmp/data/test`,
   write `results/synthetic/test_report.csv`, write `results/synthetic/mc_report.csv`,
   write per-method benchmark part files, aggregate `results/model_comparison.csv`,
   and upload evaluation CSVs to `@EVALUATION_RESULTS_STAGE/`.

When discussing outputs, be explicit:
- Model artifact: `best.pt`
- In-container synthetic reports: `results/synthetic/test_report.csv`, `results/synthetic/mc_report.csv`
- Snowflake checkpoint stage: `@MODEL_STAGE/checkpoints/`
- Snowflake evaluation stage: `@EVALUATION_RESULTS_STAGE/` for synthetic reports, `benchmark_parts/<method>_detailed.csv`, `model_comparison.csv`, and `model_comparison_summary.csv`
- HPO artifact: `@MODEL_STAGE/hpo/best_config.json`

### Snowflake Runtime Requirements

When documenting or patching Snowflake execution, include these runtime constraints:
- Stage uploads for JSON and checkpoint artifacts must use deterministic filenames and `auto_compress=False`; expected targets are `@MODEL_STAGE/hpo/best_config.json` and `@MODEL_STAGE/checkpoints/best.pt`.
- `@MODEL_STAGE` owns scripts, HPO config, and checkpoints only. Evaluation CSVs, including the canonical `model_comparison.csv`, belong under `@EVALUATION_RESULTS_STAGE/`.
- PyTorchDistributor context access must use getter methods such as `get_rank()`, `get_local_rank()`, and `get_world_size()`, not direct context attributes.
- Training must wrap the model in `DistributedDataParallel` for real multi-worker gradient synchronization; samplers and collectives alone are not enough.
- DDP training requires the train split count to be divisible by `world_size`; do not rely on padded duplicate train tasks to equalize backward steps.
- Validation must not use padded `DistributedSampler` rows. Shard validation with a no-padding rank slice, reduce global `(sum_loss, total_count)`, and compute weighted validation MSE from those reduced totals.
- Snowflake MLJobs explicitly materialize staged parquet into container-local
  `/tmp/data`; `stage_name` is the payload stage, not a dataset mount.
- Snowflake HPO, production training, and epoch calibration must materialize
  through `META_DATASET_INDEX`. If an active Snowflake session exists, missing,
  empty, incomplete, or insufficient index rows are fatal startup errors.
- `submit_from_stage(source=...)` points at `@MODEL_STAGE/scripts/`, but `stage_name` is the bare MLJob payload stage name `MLJOB_PAYLOAD_STAGE`, not `@MODEL_STAGE`.
- Snowflake compute pools cannot use `MIN_NODES = 0`; set CPU pools to `MIN_NODES = 1` and use `AUTO_SUSPEND_SECS` and/or `INITIALLY_SUSPENDED` for cost control.
- Kaggle benchmark `.npz` files persist in `@META_DATASET_STAGE/kaggle/`.
- Kaggle MLJob secrets must use Snowflake service spec syntax under `spec.containers[].secrets[]` with `snowflakeSecret`, `secretKeyRef`, and `envVarName`; do not use Kubernetes-style `env.valueFrom`.
- OpenML benchmark datasets are fetched at benchmark runtime inside Snowflake.
- OpenML/Kaggle benchmark rows are OOD smoke/generalization evidence, not strict
  TabPFN paper replication.
- OpenML benchmark jobs require benchmark Python dependencies and external network
  access through `external_access_integrations`; jobs should fail loudly when
  dependencies are unavailable.
- `fetch_tabpfn_datasets()` filters OpenML datasets with `max_cat_fraction=0.3` — datasets where more than 30% of features are categorical are skipped (after OneHot expansion they become high-d sparse matrices outside the model's numeric training distribution).
- `fetch_staged_kaggle_datasets()` skips any `.npz` file where `categorical_indicator.any()` is True (`require_numeric_only=True`). s3e5 (Wine Quality, 11 numeric) and s3e9 (Concrete Strength, 8 numeric) are confirmed all-numeric and always pass this guard.
- AutoGluon is a separate benchmark method named exactly `AutoGluon`. Its MLJob uses
  `AUTOGLUON_CPU_POOL` (`CPU_X64_M`, MIN=1, MAX=1), installs
  the shared benchmark dependencies plus `autogluon.tabular[all]==1.0.0`, sets
  `AUTOGLUON_TIME_LIMIT=300`, fits
  `TabularPredictor` with `presets="best_quality"`, `num_cpus=1`, `num_gpus=0`,
  writes `@EVALUATION_RESULTS_STAGE/benchmark_parts/AutoGluon_detailed.csv`, and
  cleans temporary model artifacts under `/tmp` after each fit.
- Keep a tiny benchmark aggregation smoke test covering two methods, two datasets,
  and two reps so `normalize_benchmark_columns()` and rank columns cannot silently
  regress.
- Evaluation MLJobs require `PREP_RUNTIME_ENVIRONMENT`,
  `BENCHMARK_RUNTIME_ENVIRONMENT`, and `AUTOGLUON_RUNTIME_ENVIRONMENT`.
  `run_evaluation_test.py` passes them as `runtime_environment`, exposes the
  selected value as `EVAL_RUNTIME_ENVIRONMENT`, uses only narrow per-job
  `pip_requirements` for dependencies missing from the managed runtime, and
  preflights configured runtime/compute-pool pairs with `runtime_probe.py`
  before expensive work.
- Evaluation preflight probes must stay serialized under the current Snowflake
  node quota. `target_instances=1` still consumes account node quota, so
  `run_evaluation_pipeline()` and `run_evaluation_runtime_probes()` submit one
  runtime probe, wait for it to finish, and only then submit the next probe.
  Do not reintroduce concurrent runtime probe submission across
  `DEEPSET_GPU_POOL`, `DEEPSET_CPU_POOL`, and `AUTOGLUON_CPU_POOL` unless the
  account node quota has been raised and verified.
- `run_evaluation_pipeline()` is phase-gated to avoid Snowflake node quota bursts:
  DeepSet GPU shards (phase 3, 10 nodes) all finish before CPU baseline shards
  (phase 4, 3 nodes) start; CPU baseline shards finish before AutoGluon shards
  (phase 5) start. AutoGluon shards run in batches of `AUTOGLUON_MAX_CONCURRENT_SHARDS=30`
  (30 total, one full-concurrency batch when `AUTOGLUON_CPU_POOL MAX_NODES` and account quota allow it).
- `run_evaluation_capacity_probe()` validates current account capacity before expensive
  evaluation work. It submits `capacity_probe.py` (no model, no data — 30-second sleep)
  in 3 non-overlapping phases matching the evaluation envelope: GPU=10, CPU=3,
  AutoGluon=30. Phases are fully serialized. Recommended run order:
  1. `CALL run_evaluation_runtime_probes(...)` — validate runtime images
  2. `CALL run_evaluation_capacity_probe(...)` — validate node quota
  3. Split-phase evaluation (recommended) or `run_evaluation_pipeline()` (legacy)
  If the capacity probe fails with a node limit error: `SHOW COMPUTE POOLS`; suspend
  idle pools; wait for active jobs to finish; or request higher node quota.
- **Split-phase evaluation** — use when account node quota cannot hold all three pools
  simultaneously for a full ~2-hour run. Five independent stored procedures expose each
  phase; the operator suspends each pool after its phase to release quota:
  ```sql
  CALL run_evaluation_runtime_probes('<prep>', '<bench>', '<ag>');
  CALL run_evaluation_prep('<prep>', '<bench>', '<ag>');
  CALL run_deepset_evaluation('<prep>', '<bench>', '<ag>');
  ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;
  CALL run_baseline_evaluation('<prep>', '<bench>', '<ag>');
  ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;
  CALL run_autogluon_evaluation('<prep>', '<bench>', '<ag>');
  ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;
  CALL run_evaluation_aggregation('<prep>', '<bench>', '<ag>');
  ```
  - Each pool is held only for its own phase; quota is released before the next phase begins.
  - `run_autogluon_evaluation()` submits 30 shards in batches of `AUTOGLUON_MAX_CONCURRENT_SHARDS=30`.
  - `ALTER COMPUTE POOL ... SUSPEND` must be issued manually after each phase; completing a
    phase does not automatically release quota.
  - `run_evaluation_aggregation()` can be re-run without re-running prior phases if
    `benchmark_parts/` files already exist on `@EVALUATION_RESULTS_STAGE`.
  - Do not collapse the phases back into a single overlapping fan-out under tight quota.

## Explain How DeepSet Is Trained

Describe `train.py` as training across many tasks, not rows from one dataset.

Call out these implementation details:
- `DATA_DIR` defaults to `/tmp/data`.
- Training selects every `train` and `val` row from `META_DATASET_INDEX`, ordered
  by `split, task_id`, materializes those `stage_path` payloads into `/tmp/data`,
  then reads every selected parquet file in `/tmp/data/train`; validation reads
  every selected parquet file in `/tmp/data/val`.
- Each parquet file is treated as one meta-dataset.
- For each task, the model consumes `(X_train, y_train, X_test)` — all m test rows are passed in a single batched forward call. The model returns a vector of m scalar predictions.
- The per-task loss is the mean squared error between the batched prediction vector and `betaX_test` across that task's m test rows.
- Epoch-level train and validation MSE are aggregated over all task-level predictions.
- The optimizer is Adam with learning rate `1e-3` and weight decay `1e-4`.
- Early stopping uses patience `10` and maximum epochs `200`.
- Training uses BF16 autocast, GradScaler, and `torch.compile(mode="reduce-overhead")` to maximize GPU utilization.
- A DataLoader with 4 worker processes and `prefetch_factor=2` overlaps Parquet I/O with GPU computation.
- The best checkpoint is saved via `model._orig_mod.state_dict()` (the unwrapped module inside `torch.compile`) and uploaded to `@MODEL_STAGE/checkpoints/`.

When explaining the learned artifact, describe `best.pt` as the serialized state dict of the DeepSet architecture (saved via `model._orig_mod` to unwrap `torch.compile`) that is later reused for held-out evaluation.

### HPO Design

Describe `hpo.py` as mandatory warm-start RandomSearch over optimizer/dropout
parameters with a fixed architecture, not as a six-parameter architecture sweep.

Call out these implementation details:
- HPO uses a table-backed pruning layer over staged parquet payloads. It should
  select train/validation candidates from `META_DATASET_INDEX`, then materialize
  only the selected parquet files into container-local `DATA_DIR`.
- The default HPO search subset is deterministic and balanced: 200 train tasks
  and 40 validation tasks. Full production training still uses the full train
  and validation splits.
- The deterministic HPO subset ranks rows within `(split, hpo_bucket)` by
  `prior_regime, p, n_train, task_id`, then selects each split by
  `bucket_rank, hpo_bucket, prior_regime, p, n_train, task_id`.
- Before launching trials, HPO must run a Ray worker Snowpark/session preflight:
  each Ray node calls `Session.builder.getOrCreate()`, verifies `SELECT 1`,
  verifies `LIST @META_DATASET_STAGE`, and downloads one selected HPO parquet to
  `/tmp`. Failures here are startup failures in Snowpark/session/stage access,
  not GPU capacity failures.
- Avoid per-trial full downloads or scans of `@META_DATASET_STAGE`; each trial
  materializes only the selected HPO subset from `META_DATASET_INDEX`.
- Before constructing the Ray Tune search space, HPO reads the scalar metadata
  needed for cardinality bounds, including `p`, `n_train`, `prior_regime`,
  `split`, and `hpo_bucket`.
- It computes observed `max(p)` and `max(n_train)` across train/val tasks.
- HPO fixes `d_phi=128`, `d_rho=256`, and `pool="pna"` for every trial.
- HPO tunes only `lr`, `weight_decay`, and `dropout`.
- HPO fails before launching trials if staged train/val metadata is missing,
  non-positive, if `@MODEL_STAGE/checkpoints/pretrain.pt` is absent, or if
  selected rows exceed `max(p) > 128` or `max(n_train) > 256`.
- Snowflake ML `TunerConfig.max_concurrent_trials` is per node. The current HPO
  topology uses `GPU_NV_M` nodes, which have two A10G GPUs per node.
- GPU trials must set `resource_per_trial={"GPU": 1}`. Do not omit this to
  increase apparent concurrency; Snowflake Tuner does not allocate GPUs to trials
  automatically.
- Do not recommend `scale_cluster()` for MLJob HPO. SPCS MLJob services cannot be
  dynamically re-scaled after submission; set HPO cluster size when submitting
  the MLJob.
- Default HPO parallelism is 20 concurrent trials: submit the HPO MLJob with
  `target_instances=5` and Ray Tune `resources_per_trial={"gpu": 1}`.
- `best_config.json` records tuned `lr`, `weight_decay`, and `dropout` plus the
  fixed `d_phi=128`, `d_rho=256`, and `pool="pna"` and is uploaded to
  `@MODEL_STAGE/hpo/` with `auto_compress=False`.
- HPO warm-starts every trial from `@MODEL_STAGE/checkpoints/pretrain.pt`.
  Missing checkpoint, failed download, or checkpoint architecture mismatch is a
  hard failure; do not fall back to random init.
- Each Ray worker downloads `pretrain.pt` independently to a rank-local `/tmp` directory (no shared filesystem across nodes).
- The pretrain checkpoint is not used as an optimizer warm-start; only `state_dict` is loaded. Each trial's optimizer initializes from scratch.

### Performance Design

HPO and full training use different parallelism semantics:
- **HPO (`tune.run()`)** warm-starts every trial from `pretrain.pt`; 30-epoch fine-tuning evaluates the marginal improvement of each optimizer/dropout config on top of the base model.
- **Full training (`PyTorchDistributor`)** uses distributed data parallel training
  on the full train/validation splits after HPO writes `best_config.json`.

The intended training configuration is distributed GPU utilization across 10 nodes:

- **`GPU_NV_M` / `DEEPSET_GPU_POOL` (10 training nodes, 5 HPO nodes)** - 4x A10G
  per node and may exceed the earlier $5/hr target.
- **`PyTorchDistributor`** with `num_nodes=10, num_workers_per_node=4` — handles Ray
  cluster setup, DDP process group initialization, and result collection automatically.
- **`DistributedDataParallel` wrapping** is required so gradients synchronize across workers.
- **DDP split accounting** requires 800 training tasks / 40 ranks = 20 backward
  steps per rank. Future train split sizes must be audited for divisibility by
  `world_size`; do not let train samplers pad duplicate tasks.
- **No-padding validation sharding** partitions the current 100 validation tasks across
  40 ranks (most ranks get 2–3 tasks), then reduces `(sum_loss, total_count)` so
  the early-stop metric is an exact global weighted MSE.
- **`DistributedSampler`** partitions 800 training tasks across 40 GPU processes
  (~20 tasks/GPU/epoch); `set_epoch()` called each epoch for correct shuffling.
- **Validation reduction** uses a no-padding rank slice and all-reduces
  `(sum_loss, total_count)` before computing exact weighted global MSE;
  `dist.broadcast(stop, src=0)` propagates the stop signal.
- **`num_workers=4`** per process (GPU_NV_M has enough vCPU headroom for
  the main training process).
- **Batched forward pass** — all m test rows are passed to the model in a single call; do
  not describe or implement row-by-row forward iteration.
- **BF16 autocast + GradScaler** — halves tensor bandwidth and activates Tensor Cores.
- **`torch.compile(mode="reduce-overhead")`** — fuses GPU kernels, reducing launch overhead.

### Architecture Design

When discussing the model architecture, note:
- `d_phi` (default 128) must be >= p (feature count) for the feature-level aggregation
  to be expressive enough to distinguish all multisets of feature vectors.
- `d_rho` (default 256) must be >= n_train (training-set size) for the sample-level
  aggregation to be expressive enough to distinguish all training contexts.
- HPO enforces these expressivity constraints by scanning staged train/val
  metadata and failing before trials unless observed `max(p) <= 128` and
  `max(n_train) <= 256`.
- **SAB (Self-Attention Block)** from Set Transformer replaces the simple linear
  equivariance layer. At the feature level, features within each sample attend to
  each other before feature pooling. At the sample level, training samples attend
  to each other before sample pooling. This is strictly more expressive than the
  original λI + γ/n·11ᵀ equivariance and is the mechanism by which "each sample
  learns from the others before being aggregated."
- **Pooling** uses one of seven modes configured via `ModelConfig.pool`: `sum`,
  `mean`, `max`, `pna` (sum+mean+max+std), `learned` (softmax-weighted sum),
  `attn` (single-seed cross-attention / PMA), or `multipool` (pna + attn, for
  ablation). Do not describe the pooling as "mean pooling" — the default is PNA.
- **Normalization**: `norm_feat=True` standardizes X_train columns per-context and
  applies the same statistics to x_test. `norm_target=True` standardizes y_train and
  denormalizes the output. Both are per-context (no global running statistics).
- **ModelConfig** bundles all hyperparameters. Always instantiate via
  `DeepSetModel(cfg=ModelConfig(...))`. The checkpoint file `best.pt` stores both
  `state_dict` and `cfg` so the exact architecture is reproducible from the file alone.
- Do not describe the pooling as "mean pooling" — the model uses PNA or another
  configured mode.
- Do not claim d_phi=64 or d_rho=64 — defaults are 128 and 256.
- Do not describe equivariance as a "scalar linear layer" — it is SAB by default.

## Explain How Generalization Is Evaluated

Describe `evaluate.py` as the unseen-data evaluation step for the trained DeepSet checkpoint.

State the evaluation contract clearly:
- It loads `best.pt`.
- It runs permutation-invariance checks as architecture sanity tests.
- It materializes the held-out test split inside the Snowflake container and evaluates
  only `/tmp/data/test`.
- It produces per-task records and then aggregates them by `prior_regime` and across all test tasks.
- Prepared benchmark DeepSet rows use `DeepSetModel-MC bounded-context ensemble`,
  not exact full-context inference: 90/10 split first, train-only preprocessing,
  DeepSet-only train-only `train_f_regression` feature selection capped by
  `BENCHMARK_DEEPSET_FEATURE_CAP` (default `model.cfg.d_phi`), five deterministic
  non-overlapping train-only context windows capped at 200 rows, prediction-level
  averaging over the full capped test split, then one metric computation.
- DeepSet benchmark detail rows include `raw_features`, `processed_features`,
  `selected_features`, `feature_selector`, and `feature_cap`; CPU baselines and
  AutoGluon still receive the full processed matrices.

Use the repo's current metric names, but explain their meaning precisely:
- `model_mse`: mean squared error between DeepSet predictions and `betaX_test` on unseen test tasks.
- `mean_model_mse`: average of `model_mse` across tasks within each regime and across the full test set.
- `ols_mse`: current code label for the baseline error, but this is not true ordinary least squares.
- `mean_ols_mse`: average baseline MSE across tasks within each regime and across the full test set.
- `ratio_model_ols`: `mean_model_mse / mean_ols_mse`; values below `1.0` mean DeepSet outperforms the baseline on average.
- `count`: number of evaluated tasks in each aggregate row.

When interpreting generalization, emphasize that the main research question is whether DeepSet achieves lower MSE than the fixed linear baseline on unseen tasks sampled from the same synthetic task family.

## Describe The Baseline Correctly

Do not describe the current baseline as plain OLS.

Explain it as:
- Ridge regression with fixed L2 penalty `lambda = 1`
- Closed-form estimator `(X^T X + I)^(-1) X^T y`
- No hyperparameter tuning
- Evaluated on the same unseen test tasks as DeepSet
- Compared using MSE against the noiseless target `betaX_test`

If you mention the code variable names, clarify that `ols_mse` and `mean_ols_mse` are legacy labels for this fixed ridge baseline.

## Research Caveats

Include these caveats when discussing the current evaluation:
- The evaluation target is `betaX_test`, the noiseless linear signal, not the noisy observed response.
- The baseline is a fixed ridge model, not tuned ridge and not exact OLS.
- The current report provides MSE summaries and the DeepSet-to-baseline ratio, but not confidence intervals, hypothesis tests, RMSE, MAE, R-squared, or calibration metrics.
- The permutation checks validate symmetry and equivariance properties of the architecture; they do not themselves measure predictive generalization.

## Preferred Phrasing

Prefer wording like:
- "The pipeline has three phases: pre-training with default hyperparameters writes `pretrain.pt`; HPO fine-tunes from `pretrain.pt` to find the best config; final training fine-tunes from `pretrain.pt` with `best_config.json` to produce `best.pt`."
- "`CHECKPOINT_OUTPUT_NAME=pretrain.pt` in Phase 1 and `CHECKPOINT_OUTPUT_NAME=best.pt` (default) in Phase 3 distinguish the two `train.py` invocations."
- "HPO warm-start is mandatory: every trial loads `@MODEL_STAGE/checkpoints/pretrain.pt`, and missing, inaccessible, or architecture-mismatched checkpoints fail the run."
- "DeepSet is trained over many synthetic regression tasks stored as parquet meta-datasets."
- "`run_training_pipeline()` submits only HPO and training; `run_evaluation_pipeline()` separately consumes `@MODEL_STAGE/checkpoints/best.pt` for synthetic evaluation and benchmarks."
- "`PyTorchDistributor` manages Ray, DDP, and result collection; `train_fn` receives hyperparameters and a distributed context via `get_context()`."
- "HPO runs RandomSearch over `lr`, `weight_decay`, and `dropout`: it selects a deterministic balanced subset from `META_DATASET_INDEX`, enforces `max(p) <= 128` and `max(n_train) <= 256`, then runs 20 trials (20 concurrent, 1 round on 5 GPU_NV_M nodes) using fixed `d_phi=128`, `d_rho=256`, and `pool='pna'`."
- "The compute pool uses `DEEPSET_GPU_POOL` with `MAX_NODES = 10` for 5-node HPO and 10-node DDP training; this can exceed the earlier $5/hr budget cap."
- "Generalization is assessed by comparing DeepSet test MSE against a fixed ridge-regression baseline on unseen datasets."
- "A ratio below 1.0 in `ratio_model_ols` indicates lower average error for DeepSet than for the baseline."
- "All m test rows are passed to the model in a single batched forward call; the model returns a vector of m scalar predictions."
- "The DataLoader prefetches Parquet files across 4 worker processes so the A10G GPU is never waiting for data."
- "phi maps each (y_i, x_ij, x_test_j) triple into a d_phi-dimensional embedding; d_phi must be at least as large as the number of features to preserve set information."
- "PNA pooling (sum + mean + max + std) prevents multiset collisions that would cause distinct training contexts to share the same latent representation."
- "SAB (Self-Attention Block) replaces the simple linear equivariance layer; features attend to each other before feature pooling, and samples attend to each other before sample pooling."
- "All hyperparameters are bundled in ModelConfig and stored alongside the checkpoint, making the architecture fully reproducible from best.pt alone."
- "Feature and target normalization are applied per-context inside forward(), using statistics computed from X_train and y_train of the current task."
- "Use pool='multipool' to ablate all aggregation statistics simultaneously."

Avoid wording like:
- "The model trains on one large dataset."
- "The baseline is ordinary least squares" unless the code is changed.
- "Permutation tests prove the model generalizes".
- Per-row iteration language such as "for each test row k, the model predicts X_test[k]" — the model uses batched forward.
- "mean pooling" as the sole descriptor — the model uses PNA (four aggregation statistics).
- Claiming d_phi=64 or d_rho=64 are the defaults — they were raised to 128 and 256.
- Describing the equivariance as a "scalar linear layer" or "λ/γ scaling" — the model uses SAB by default (n_sab_feat=1, n_sab_samp=1).
- Instantiating the model with flat kwargs `DeepSetModel(d_phi=128, ...)` in new code — always use `DeepSetModel(cfg=ModelConfig(...))`.
- Describing "best.pt" as a plain state dict — it now stores {"state_dict": ..., "cfg": ...}.
- Describing training as single-GPU after this change.
- Running `run_training_job.py` from the local machine or describing it as a locally-executed script — it runs as a Snowpark stored procedure handler inside Snowflake.
- Citing `GPU_NV_L` as required for the default runbook; current guidance uses
  `GPU_NV_M` unless measured phase timings justify a topology change.
- Referring to Docker or container image build/push commands — the pipeline uses the Snowflake Container Runtime; no custom image is built or maintained.
- Describing HPO as always starting from random initialization after this change.
- Describing `pretrain.pt` as an evaluation artifact — it is an intermediate training checkpoint consumed by HPO and final training, not by `evaluate.py`.
