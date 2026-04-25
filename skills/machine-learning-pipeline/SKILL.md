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
- `evaluate.py` loads `best.pt`, runs permutation checks, evaluates only the held-out test split, writes `results/test_report.csv`, and uploads the report to `@MODEL_STAGE/results/` when running inside SPCS.

## Explain The Data Layout

Describe the generated data as a collection of many meta-datasets rather than one flat training table.

State the generation contract from `generate_dgp.py`:
- Each parquet file contains one synthetic regression task.
- Each task stores `X_train`, `y_train`, `X_test`, `betaX_test`, `n`, `p`, `n_train`, `n_test`, and `prior_regime`.
- `betaX_test` is the noiseless target used for evaluation.
- With `--n_datasets 1000`, the script writes 800 training tasks, 100 validation tasks, and 100 test tasks.

When describing Snowflake execution, note that these parquet files are uploaded to `@META_DATASET_STAGE` and mounted into the container at `/data`.

## Explain The SPCS Execution Flow

Use this flow when summarizing the end-to-end pipeline:

1. Generate synthetic datasets locally with `python generate_dgp.py --n_datasets 1000 --out_dir data/`.
2. Upload `data/train/*.parquet`, `data/val/*.parquet`, and `data/test/*.parquet` to `@META_DATASET_STAGE`.
3. Run the container in SPCS with `@META_DATASET_STAGE` mounted to `/data`.
4. Training and evaluation are submitted as Snowflake ML Jobs via `run_training_job.py`
   (not via EXECUTE JOB SERVICE). The Container Runtime provides a managed GPU image;
   no custom Docker image is required.
   `PyTorchDistributor` splits the 800 training tasks across 2 GPU_NV_S nodes using DDP.

5. During training, save the best checkpoint to `best.pt` using `model._orig_mod.state_dict()` (unwrapped from `torch.compile`) and upload it to `@MODEL_STAGE/checkpoints/`.
6. During evaluation, load `best.pt`, run permutation-invariance checks, evaluate on `/data/test`, write `results/test_report.csv`, and upload it to `@MODEL_STAGE/results/`.

When discussing outputs, be explicit:
- Model artifact: `best.pt`
- In-container report: `results/test_report.csv`
- Snowflake checkpoint stage: `@MODEL_STAGE/checkpoints/`
- Snowflake evaluation stage: `@MODEL_STAGE/results/`

## Explain How DeepSet Is Trained

Describe `train.py` as training across many tasks, not rows from one dataset.

Call out these implementation details:
- `DATA_DIR` defaults to `/data`.
- Training reads every parquet file in `/data/train`; validation reads every parquet file in `/data/val`.
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

### Performance Design

The training pipeline is configured for distributed GPU utilization across 2 nodes:

- **`GPU_NV_S` (2 nodes)** — 0.57 cr/node/hr ≈ $2.28–3.42/hr total; 1× A10G per node.
- **`PyTorchDistributor`** with `num_nodes=2, num_workers_per_node=1` — handles Ray
  cluster setup, DDP process group initialization, and result collection automatically.
- **`DistributedSampler`** partitions 800 training tasks across 2 GPU processes
  (~400 tasks/GPU/epoch); `set_epoch()` called each epoch for correct shuffling.
- **`dist.all_reduce(val_tensor, AVG)`** — aggregates validation MSE across ranks before
  the early-stop check; `dist.broadcast(stop, src=0)` propagates the stop signal.
- **`num_workers=4`** per process (GPU_NV_S has ~12 vCPUs; 4 workers leaves headroom for
  the main training process).
- **Batched forward pass** — all m test rows are passed to the model in a single call; do
  not describe or implement row-by-row forward iteration.
- **BF16 autocast + GradScaler** — halves tensor bandwidth and activates Tensor Cores.
- **`torch.compile(mode="reduce-overhead")`** — fuses GPU kernels, reducing launch overhead.

### Architecture Design

When discussing the model architecture, note:
- `d_phi` (default 128) must be >= p (feature count) for the feature-level aggregation
  to be expressive enough to distinguish all multisets of feature vectors.
- `d_rho` (default 256) must be >= n (training-set size) for the sample-level
  aggregation to be expressive enough to distinguish all training contexts.
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
- It evaluates only the held-out test tasks in `/data/test`.
- It produces per-task records and then aggregates them by `prior_regime` and across all test tasks.

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
- "DeepSet is trained over many synthetic regression tasks stored as parquet meta-datasets."
- "Training is submitted as an ML Job via `run_training_job.py` using the Snowflake Container Runtime for ML."
- "`PyTorchDistributor` manages Ray, DDP, and result collection; `train_fn` receives hyperparameters and a distributed context via `get_context()`."
- "HPO runs 20 Bayesian Optimization trials (30 epochs each) in parallel before full training."
- "The compute pool uses `GPU_NV_S` (2 nodes) at ~$2.28–3.42/hr — within the $1–5/hr budget."
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
- Referring to EXECUTE JOB SERVICE as the deployment mechanism.
- Citing `GPU_NV_M` or `GPU_NV_L` as the required pool.
- Referring to Docker or container image build/push commands — the pipeline uses the Snowflake Container Runtime; no custom image is built or maintained.
