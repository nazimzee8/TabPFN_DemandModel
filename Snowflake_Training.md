# Snowflake Training — DeepSet TabPFN

Describes how to run the DeepSet training pipeline inside a Snowflake environment
using Snowpark Container Services (SPCS) and Snowflake Model Registry.

---

## Snowflake Environment Construction

Ordered setup: from local machine to a running SPCS training job with model checkpoint
written back to Snowflake.

**Steps:**

1. **Write `model.py`, `train.py`, and `evaluate.py`** on the local machine (DeepSet architecture, training loop, and evaluation script).
2. **Generate datasets locally** via `generate_dgp.py` (outputs `data/train/` 800 files, `data/val/` 100 files, `data/test/` 100 files).
3. **Run `run_training_job.sql`** in Snowsight or SnowSQL — creates the database, schema, stages (`@META_DATASET_STAGE`, `@MODEL_STAGE`), and compute pool (`DEEPSET_GPU_POOL`, `GPU_NV_S`, MIN=1, MAX=4). Verify the pool reaches `ACTIVE` state before continuing.
4. **Upload Parquet files** to `@META_DATASET_STAGE` via SnowSQL `PUT` (step 3 in the SQL file).
5. **Upload Python scripts** to `@MODEL_STAGE/scripts/` via SnowSQL `PUT` (step 3b in the SQL file).
6. **Run `CALL run_training_pipeline()`** (step 4 in `run_training_job.sql`) from Snowsight or SnowSQL —
   executes the Snowpark stored procedure, which submits HPO, training, and evaluation as
   sequential MLJob phases. No local Python environment or credentials are required beyond
   the initial SQL session.

**Data and artifact flow:**

```
Local machine
  └── *.parquet ──PUT──→ @META_DATASET_STAGE ──vol mount──→ /data/
  └── *.py ──PUT──→ @MODEL_STAGE/scripts/ ──vol mount──→ /opt/app/

Snowsight / SnowSQL
  └── CALL run_training_pipeline() ──→ Snowpark stored procedure
        └── run_training_job.run_pipeline(session) submits:

Container Runtime — Phase 1: HPO (2 nodes)
  └── hpo.py → @MODEL_STAGE/hpo/best_config.json

Container Runtime — Phase 2: Training (4 nodes, DDP)
  ├── DataLoader (4 workers, prefetch_factor=2) reads /data/train/*.parquet
  ├── trains DeepSet (phi, rho, psi + 4 equivariant scalars)
  │     BF16 autocast + GradScaler, batched forward over all m test rows
  │     torch.compile(mode="reduce-overhead") fuses GPU kernels
  └── writes best.pt → @MODEL_STAGE/checkpoints/best.pt

Container Runtime — Phase 3: Evaluation (1 node)
  └── evaluate.py → @MODEL_STAGE/results/test_report.csv

Model Registry
  └── DEEPSET_TABPFN_V1!PREDICT() ← loads from @MODEL_STAGE/checkpoints/best.pt
```

---

## Data Storage

Variable-shape datasets (pickle) cannot be stored in flat Snowflake tables efficiently.
Use an **internal named stage** with Parquet files:

```
@META_DATASET_STAGE/
  train/   ← 800 parquet files (one per meta-task)
  val/     ← 100 parquet files
  test/    ← 100 parquet files
```

Each Parquet file contains: `X_train`, `y_train`, `X_test`, `betaX_test`,
`n`, `p`, `prior_regime`, stored as nested arrays via VARIANT or PyArrow list types.

### Generating datasets locally

Prerequisites: `pip install numpy pyarrow`

```bash
cd /c/Documents/TabPFN_DemandModel
python generate_dgp.py --n_datasets 1000 --out_dir data/
# Writes: data/train/ (800 files), data/val/ (100), data/test/ (100)
```

### Uploading to Snowflake via SnowSQL

> **Important**: `PUT` is a client-side command that streams files from your local disk
> directly to Snowflake's internal stage. It is **not supported** in the Snowsight web UI
> ("Unsupported feature 'unsupported_requested_format:snowflake'"). Use SnowSQL instead.

Install SnowSQL: https://docs.snowflake.com/en/user-guide/snowsql-install-config

Connect:

```bash
snowsql -a <your_account_identifier> -u <your_username>
```

Then run the three `PUT` commands inside SnowSQL:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

PUT file:///c/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE;
PUT file:///c/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE;
PUT file:///c/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE;
```

Verify the upload:

```sql
LIST @META_DATASET_STAGE/train/;
LIST @META_DATASET_STAGE/val/;
LIST @META_DATASET_STAGE/test/;
```

### Uploading Python scripts via SnowSQL

Run once, and re-run whenever any script changes:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

PUT file://C:/Documents/TabPFN_DemandModel/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

Verify:

```sql
LIST @MODEL_STAGE/scripts/;
```

---

## Prerequisite SQL Objects

Run these once in Snowsight (or SnowSQL) before any other step.

```sql
-- Dedicated database and schema that own all project objects
-- (stages, image repository, compute pool, service, model registry entries).
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
USE DATABASE TABPFN_DB;
CREATE SCHEMA IF NOT EXISTS TABPFN_SCHEMA;
USE SCHEMA TABPFN_SCHEMA;

-- Internal stage for raw Parquet meta-datasets (train / val / test splits).
-- The Container Runtime mounts this stage as a read-only volume at /data/.
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Internal stage for model artifacts: best.pt checkpoint and evaluation results.
-- train.py and evaluate.py write here via session.file.put() from inside the container.
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

---

## Compute: Container Runtime for ML

### 1. Compute Pool

```sql
-- GPU_NV_S: 1× A10G per node, 0.57 cr/hr. MAX_NODES=4: up to 4 nodes for DDP training,
-- 2 nodes for parallel HPO trials.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY — must drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 4
  INSTANCE_FAMILY = GPU_NV_S;
```

### 2. Job Submission (MLJob)

`run_training_job.py` is deployed as a Snowpark Python stored procedure. The procedure
submits HPO, training, and evaluation as sequential MLJob phases using scripts already
on `@MODEL_STAGE/scripts/` — all within Snowflake. No local Python environment is needed.

#### What is an MLJob container?

An MLJob container is a short-lived compute environment that Snowflake starts on
one or more nodes in your GPU compute pool to run a single Python script. When
`submit_from_stage()` is called, Snowflake pulls the managed ML runtime image onto
the requested nodes, runs your entrypoint (e.g. `train.py`), writes outputs to the
stage, then shuts the container down. PyTorch, Ray, and `snowflake-ml-python` are
pre-installed — no Docker build or image management is required.

Create the procedure (re-run after uploading an updated `run_training_job.py`):

```sql
CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';
```

Then call it:

```sql
CALL run_training_pipeline();
```

Each phase uses `compute_pool="DEEPSET_GPU_POOL"` with the Snowflake-managed GPU runtime image:

| Phase | Entrypoint | Instances | Output |
|---|---|---|---|
| HPO | `hpo.py` | 2 | `@MODEL_STAGE/hpo/best_config.json` |
| Training | `train.py` | 4 (DDP) | `@MODEL_STAGE/checkpoints/best.pt` |
| Evaluation | `evaluate.py` | 1 | `@MODEL_STAGE/results/test_report.csv` |

### 3. Checkpoint Output

Write best checkpoint back to the model stage on improvement:

```python
session.file.put(
    local_file_name="best.pt",
    stage_location="@MODEL_STAGE/checkpoints/",
    overwrite=True,
)
```

---

## Distributed Training & Hyperparameter Optimization

### Container Runtime for ML

Snowflake Container Runtime for ML provides a managed, GPU-enabled image with PyTorch,
Ray, and `snowflake-ml-python` pre-installed.

- No container image build or push needed — scripts are read directly from
  `@MODEL_STAGE/scripts/` via `submit_from_stage()`.
- Runtime image: `snowflake/ml-runtime-gpu:latest` (Snowflake-managed).
- Jobs submitted from the stored procedure via `run_training_job.py`.

> Scripts are referenced directly from the stage via `source=` in
> `submit_from_stage()`. No Docker image is required or maintained.

### Distributed Training — PyTorchDistributor

- Class: `snowflake.ml.modeling.distributors.pytorch.PyTorchDistributor`
- Manages Ray cluster setup, DDP process group initialization, and result collection
  internally — no manual `torchrun` or rank-environment setup required.
- `PyTorchScalingConfig(num_nodes=2, num_workers_per_node=1, ...)` maps to
  2× GPU_NV_S nodes (one A10G each).
- `get_context()` inside `train_fn` provides `local_rank`, `rank`, and `world_size`.
- `DistributedSampler` with `ctx.rank` / `ctx.world_size` splits the 800 training
  tasks across 2 GPU processes (~400 tasks/GPU/epoch).
- `dist.all_reduce(val_tensor, AVG)` aggregates validation MSE across ranks before
  the early-stop check; `dist.broadcast(stop, src=0)` propagates the stop signal.

### Hyperparameter Optimization — Tuner + BayesOpt

- Class: `snowflake.ml.modeling.tune.Tuner`
- Algorithm: `BayesOpt` (Gaussian-process surrogate; minimizes trials needed vs.
  grid or random search).
- Search space: `lr`, `weight_decay`, `d_phi`, `d_rho`, `dropout`, `pool`.
- 20 trials, 30-epoch runs each; best config written to
  `@MODEL_STAGE/hpo/best_config.json` on completion.
- To use a simpler baseline, swap `BayesOpt()` → `RandomSearch()` in `hpo.py`.

### Compute Pool & Cost

| Configuration | Credits/node/hr | Nodes | Total cost/hr |
|---|---|---|---|
| GPU_NV_S (this design) | 0.57 | 2 | ~$2.28–3.42 |
| previous single-node | 2.68 | 1 | ~$5.36–8.04 |

- 2-node GPU_NV_S pool: 1.14 cr/hr ≈ **$2.28–3.42/hr** (Standard/Enterprise) — ~80%
  cheaper than the previous single-node configuration.
- Pool suspends when idle; no charges in `SUSPENDED` state.

### Estimated End-to-End Cost

| Phase | Nodes | Cost/hr | Duration | Total |
|---|---|---|---|---|
| HPO (20 trials × 30 epochs) | 2 × GPU_NV_S | ~$2.28–3.42 | ~40–60 min | ~$1.52–3.42 |
| Full training (DDP) | 4 × GPU_NV_S | ~$4.56–6.84 | ~8–13 min | ~$0.61–1.48 |
| Evaluation | 1 × GPU_NV_S | ~$1.14–1.71 | ~5–10 min | ~$0.10–0.29 |
| **Total** | | | **~53–83 min** | **~$2.23–5.19** |

---

## Training Loop Adaptation

Inside the container, `train.py` follows this loop:

```python
import os, glob
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DeepSetModel

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = "/data"
PATIENCE   = 10
MAX_EPOCHS = 200
USE_AMP    = DEVICE == "cuda"

# --- Dataset and DataLoader ---
class ParquetMetaDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path  = self.files[idx]
        table = pq.read_table(path)
        d     = table.to_pydict()
        return (
            torch.tensor(np.array(d["X_train"][0]),    dtype=torch.float32),
            torch.tensor(np.array(d["y_train"][0]),    dtype=torch.float32),
            torch.tensor(np.array(d["X_test"][0]),     dtype=torch.float32),
            torch.tensor(np.array(d["betaX_test"][0]), dtype=torch.float32),
        )

def identity_collate(batch):
    return batch[0]   # batch_size=1; skip default list-wrapping

train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.parquet")))
val_files   = sorted(glob.glob(os.path.join(DATA_DIR, "val",   "*.parquet")))

train_loader = DataLoader(
    ParquetMetaDataset(train_files), batch_size=1, shuffle=True,
    num_workers=4, prefetch_factor=2, pin_memory=USE_AMP, collate_fn=identity_collate,
)
val_loader = DataLoader(
    ParquetMetaDataset(val_files), batch_size=1, shuffle=False,
    num_workers=4, prefetch_factor=2, pin_memory=USE_AMP, collate_fn=identity_collate,
)

# --- Model, compiler, optimizer, scaler ---
from model import DeepSetModel, ModelConfig

cfg = ModelConfig(
    d_phi=128, d_rho=256, pool="pna", n_heads=4,
    n_sab_feat=1, n_sab_samp=1,
    norm_feat=True, norm_target=True, dropout=0.1,
)
model     = DeepSetModel(cfg=cfg).to(DEVICE)
model     = torch.compile(model, mode="reduce-overhead")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# --- Training loop ---
best_val_mse   = float("inf")
patience_count = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    for X_train, y_train, X_test, betaX_test in train_loader:
        X_train    = X_train.to(DEVICE)
        y_train    = y_train.to(DEVICE)
        X_test     = X_test.to(DEVICE)
        betaX_test = betaX_test.to(DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
            y_hat = model(X_train, y_train, X_test)          # batched: (m,)
            loss  = nn.functional.mse_loss(y_hat, betaX_test)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # --- Validation ---
    model.eval()
    val_mses = []
    with torch.no_grad():
        for X_train, y_train, X_test, betaX_test in val_loader:
            X_train    = X_train.to(DEVICE)
            y_train    = y_train.to(DEVICE)
            X_test     = X_test.to(DEVICE)
            betaX_test = betaX_test.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
                y_hat = model(X_train, y_train, X_test)
            val_mses.append(nn.functional.mse_loss(y_hat.float(), betaX_test.float()).item())
    val_mse = float(np.mean(val_mses))
    print(f"Epoch {epoch:3d}  val_mse={val_mse:.4f}")

    if val_mse < best_val_mse:
        best_val_mse   = val_mse
        patience_count = 0
        # torch.compile wraps the module; _orig_mod holds the original state dict
        ckpt_module = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({"state_dict": ckpt_module.state_dict(), "cfg": ckpt_module.cfg}, "best.pt")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

# Upload best checkpoint to Snowflake stage
# (run via Snowpark session inside container)
```

---

## Performance Optimizations

The original training loop ran row-by-row: for each meta-dataset, `forward()` was called
once per test row (`X_test[k]`), producing ~3.6 M serial GPU kernel launches across 800
training files × ~4,500 test rows. This serialized inference saturated the CPU and left
GPU Tensor Cores idle.

### 1. Batched Forward Pass

`forward()` accepts `x_test` of shape `(m, p)` (all test rows at once) and returns
predictions of shape `(m,)`. A single GPU kernel dispatch replaces the per-row loop.

### 2. BF16 Autocast

`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward and
backward passes. BF16 halves tensor bandwidth and fully activates Tensor Cores on A10G,
roughly doubling throughput. `GradScaler` is included for safety with larger gradient
magnitudes.

### 3. DataLoader with Background Workers

```python
DataLoader(..., num_workers=4, prefetch_factor=2, pin_memory=True)
```

Four OS processes read and decode Parquet files while the GPU trains on the previous
batch, removing I/O from the critical path. `pin_memory=True` enables DMA transfers
between page-locked host RAM and GPU memory.

### 4. `torch.compile`

```python
model = torch.compile(model, mode="reduce-overhead")
```

`torch.compile` traces the graph and fuses repeated small kernels (linear + ReLU,
residual add) into single fused kernels, reducing Python interpreter overhead and GPU
kernel launch latency. The compiled model is saved via `model._orig_mod` to avoid
`torch.compile` wrapper artefacts in the checkpoint.

### 5. GPU_NV_S Compute Pool

`GPU_NV_S` provides 1× A10G GPU per node and ~12 vCPUs. 4 DataLoader worker processes
fit comfortably within available host RAM, leaving headroom for the main training
process. 4 nodes are used for DDP training; 2 nodes for parallel HPO trials.

### Cost Comparison

| Configuration | Estimated wall-clock | Notes |
|---|---|---|
| GPU_NV_S, row-by-row, FP32 | ~4 hours | Original |
| GPU_NV_S × 2, batched, BF16, DDP, compile | ~15–25 minutes | Previous optimized |
| GPU_NV_S × 4, batched, BF16, DDP, compile | ~8–13 minutes | Current (4-node DDP) |

Estimates assume 800 training files × 200 epochs with early stopping at epoch ~100.

---

## Architecture Design

### Latent Space Dimensions

DeepSet universality (Zaheer et al. 2017) requires:

- **`d_phi >= p`** (number of features): `phi` maps each `(y_i, x_ij, x_test_j)` triple
  into R^{d_phi}. For the feature-level aggregation to represent all possible set
  functions over p feature vectors, d_phi must span at least p dimensions.
- **`d_rho >= n`** (number of training samples): `rho` maps each aggregated sample
  embedding into R^{d_rho}. For the sample-level aggregation to represent all possible
  sample-set functions, d_rho must span at least n dimensions.

Current defaults: `d_phi=128`, `d_rho=256`. If your tasks have more than 128 features
or more than 256 training samples, increase these accordingly.

### Phi Injectivity

`phi: R^3 → R^{d_phi}` must be injective — different input triples must produce
different embeddings — so that no two training examples collapse to the same vector
before aggregation. With `d_phi=128` (far larger than the 3-dimensional input), a
trained ReLU MLP is injective on the training manifold by standard covering arguments.

### Continuity

`phi` and `rho` must be continuous: a small perturbation in the input must produce
a small change in the output, so that the aggregated representation varies smoothly.
ReLU networks are piecewise linear and therefore Lipschitz continuous — this
requirement is satisfied by the architecture as-is.

### PNA Pooling and the Sum/Mean Collision Problem

**The problem:** even with an injective `phi`, two *different* multisets can satisfy

```
mean(phi(x) for x in S1)  ==  mean(phi(x) for x in S2),   S1 ≠ S2
```

This "multiset collision" causes the model to map distinct training contexts to the
same latent representation, losing information that is relevant for the prediction.

**The fix — Principal Neighbourhood Aggregation (PNA):** instead of aggregating with
mean alone, concatenate four statistics over the set dimension:

```
pool(S) = cat[ sum_phi, mean_phi, max_phi, std_phi ]   ∈ R^{4·d_phi}
```

Two sets that share the same mean will generally differ in at least one of sum, max,
or std, yielding a distinct joint embedding. PNA is applied at *both* pooling stages
(feature-level and sample-level), so collisions are suppressed throughout the network.
The learnable equivariance layers (λ, γ) continue to operate *before* pooling and are
unaffected by this change.

PNA increases the rho input from `d_phi → 4·d_phi` and the psi input from
`d_rho → 4·d_rho`. The extra parameters are absorbed by rho and psi without changing
the output interface.

### Self-Attention Blocks (SAB)

The simple linear equivariance layer (λI + γ/n·11ᵀ) is replaced by one or more
**Self-Attention Blocks** from the Set Transformer (Lee et al. 2019), applied at both
the feature level (features attend to each other per sample) and the sample level
(samples attend to each other before final pooling):

```
X → φ → SAB_feat → pool_feat → ρ → SAB_samp → pool_samp → ψ
```

`SAB(X) = MAB(X, X)` where `MAB(Q, K) = LayerNorm(H + FFN(H))` and
`H = LayerNorm(Q + Dropout(MHA(Q, K, K)))`. SAB is permutation equivariant:
`SAB(X[π]) = SAB(X)[π]` for any permutation π — a strictly more expressive
generalisation of the original λ/γ equivariance. The number of SAB layers is
controlled by `n_sab_feat` and `n_sab_samp` in `ModelConfig`.

Setting `n_sab_feat=0, n_sab_samp=0` recovers the original linear equivariance
layers exactly (backward-compatible with old checkpoints).

### Pooling Modes

`SetPool` provides a unified interface for seven permutation-invariant pooling modes:

| Mode | Output dim | Description |
|---|---|---|
| `sum` | d | Element sum |
| `mean` | d | Element mean |
| `max` | d | Element max |
| `pna` | 4d | Concat[sum, mean, max, std] |
| `learned` | d | Softmax-weighted sum (learned scores) |
| `attn` | d | PMA: single-seed cross-attention |
| `multipool` | 5d | Concat[pna, attn] — for ablation |

PNA and multipool are the most expressive. Use `pool="multipool"` to run ablations
comparing all statistics simultaneously. Use `pool="attn"` for the Set Transformer
canonical pooling.

### Normalization Strategy

Two per-context normalizations are applied inside `forward()`:
- **Feature normalization** (`norm_feat=True`): each column of X_train is
  standardised to zero mean and unit variance; the same statistics are applied to
  x_test. This makes the model scale-invariant to feature magnitudes.
- **Target normalization** (`norm_target=True`): y_train is standardised before
  being fed to φ; the final prediction is denormalized back to the original scale.
  This removes sensitivity to the absolute scale of the regression target.

Both normalizations use per-context statistics (computed from X_train / y_train
of the current task), not global running statistics — the model requires no warm-up
and works immediately on any new task.

Batch normalization is not used: SPCS runs each meta-dataset as a batch of 1, so
BN statistics would be degenerate, and BN over the set dimension would break
permutation invariance with small sets.

### ModelConfig Hyperparameterization

All hyperparameters are bundled in `ModelConfig` (a `dataclasses.dataclass`):

| Field | Default | Description |
|---|---|---|
| `d_phi` | 128 | phi output dim (≥ p for universality) |
| `d_rho` | 256 | rho output dim (≥ n for universality) |
| `pool` | `"pna"` | Pooling mode (see table above) |
| `n_heads` | 4 | Attention heads for SAB / AttentionPool |
| `n_sab_feat` | 1 | SAB layers at feature level |
| `n_sab_samp` | 1 | SAB layers at sample level |
| `norm_feat` | `True` | Feature standardization |
| `norm_target` | `True` | Target standardization |
| `dropout` | 0.1 | Dropout in MLPs and SAB |

---

## Model Registry

After training, register the model via Snowflake Model Registry for versioned
deployment and SQL-callable inference:

```python
from snowflake.ml.registry import Registry

reg = Registry(session=session, database="TABPFN_DB", schema="TABPFN_SCHEMA")
reg.log_model(
    model=deepset_model,
    model_name="DEEPSET_TABPFN_V1",
    version_name="v1",
    sample_input_data=sample_batch,
)
```

SQL-callable inference after registration:

```sql
SELECT DEEPSET_TABPFN_V1!PREDICT(X_TRAIN, Y_TRAIN, X_TEST)
FROM INFERENCE_TABLE;
```

---

## Model Output

`best.pt` is the pretrained model artifact. It encodes the learned PPD approximation
procedure — the full state dict of the DeepSet (phi, rho, psi MLPs and the four
equivariant scalars λ_1, γ_1, λ_2, γ_2).

**Key properties:**
- Stored at `@MODEL_STAGE/checkpoints/best.pt`.
- Created by `torch.save({"state_dict": ..., "cfg": ...}, "best.pt")` whenever val MSE improves.
- Uploaded from the training container via `session.file.put("best.pt", "@MODEL_STAGE/checkpoints/", overwrite=True)`.

**Used for inference on any new synthetic dataset without retraining:**

```python
import torch
from model import DeepSetModel, ModelConfig

ckpt = torch.load("best.pt", map_location="cpu")
if isinstance(ckpt, dict) and "cfg" in ckpt:
    cfg, state_dict = ckpt["cfg"], ckpt["state_dict"]
else:                                       # legacy bare state_dict
    cfg = ModelConfig(d_phi=128, d_rho=256, pool="pna",
                      n_sab_feat=0, n_sab_samp=0,
                      norm_feat=False, norm_target=False)
    state_dict = ckpt
model = DeepSetModel(cfg=cfg)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    y_hat = model(X_train_new, y_train_new, x_test_new)
```

**SQL-callable via Model Registry:**

```sql
SELECT DEEPSET_TABPFN_V1!PREDICT(X_TRAIN, Y_TRAIN, X_TEST)
FROM INFERENCE_TABLE;
```

The registered model loads from `@MODEL_STAGE/checkpoints/best.pt` at inference time,
so no Docker image rebuild is required to update predictions after retraining.

---

## OOD Evaluation Stored Procedure

Run after training as a Snowpark Python stored procedure. Returns a result table
stratified by regime, feature count quartile, and set size quartile.

```python
from snowflake.snowpark import functions as F

def evaluate_ood(session):
    test_files = session.file.list("@META_DATASET_STAGE/test/")
    results = []

    for f in test_files:
        dataset = load_parquet(f)
        y_hat   = model.predict(dataset)
        mse     = float(((y_hat - dataset["betaX_test"]) ** 2).mean())
        results.append({
            "prior_regime": dataset["prior_regime"],
            "p_quartile":   quartile(int(dataset["p"])),
            "n_quartile":   quartile(int(dataset["n"])),
            "mse":          mse,
        })

    return (
        session.create_dataframe(results)
        .group_by("prior_regime", "p_quartile", "n_quartile")
        .agg(F.mean("mse").alias("mean_mse"), F.stddev("mse").alias("std_mse"))
        .sort("prior_regime", "p_quartile", "n_quartile")
    )
```

Expected output shape: one row per `(regime, p_quartile, n_quartile)` combination,
with `mean_mse` and `std_mse` columns. Regime A should have the lowest `mean_mse`;
Regimes C (heavy-tail noise) and D (correlated X) higher but not catastrophic.

---

## Downloading Results Locally

`MODEL_STAGE` holds two artifacts written by the training and evaluation jobs:

| Stage path | File | Written by |
|---|---|---|
| `@MODEL_STAGE/checkpoints/best.pt` | Model checkpoint | `train.py` (on each val MSE improvement) |
| `@MODEL_STAGE/results/test_report.csv` | OOD evaluation results | `evaluate.py` |

Use `download_results.py` to pull both files to your local machine.

### Prerequisites

```bash
pip install snowflake-snowpark-python   # already in requirements.txt
```

### Set credentials

Set the following environment variables before running `download_results.py`:

| Variable | Required | Default |
|---|---|---|
| `SNOWFLAKE_ACCOUNT` | yes | — |
| `SNOWFLAKE_USER` | yes | — |
| `SNOWFLAKE_PASSWORD` | yes | — |
| `SNOWFLAKE_WAREHOUSE` | no | `COMPUTE_WH` |

### Run

```bash
cd C:/Documents/TabPFN_DemandModel
python download_results.py
```

The script:
1. Connects using the env-var credentials (same `TABPFN_DB` / `TABPFN_SCHEMA` / `COMPUTE_WH` defaults as the training job).
2. Lists `@MODEL_STAGE` and prints every file found.
3. Downloads `@MODEL_STAGE/checkpoints/` and `@MODEL_STAGE/results/` into the local `models/` directory (created automatically).

### Expected output

```
Connected to Snowflake.
Stage contents:
  @model_stage/checkpoints/best.pt.gz
  @model_stage/results/test_report.csv.gz

Downloading @MODEL_STAGE/checkpoints/ ...
Downloading @MODEL_STAGE/results/ ...

Done. Files saved to ./models/
  models/best.pt
  models/test_report.csv
```

> **Note:** Snowflake automatically decompresses `.gz` files on `GET`/`session.file.get()`,
> so the files land as `best.pt` and `test_report.csv` (not `.gz`).

### Load the checkpoint

```python
import torch
from model import DeepSetModel

ckpt  = torch.load("models/best.pt", map_location="cpu")
model = DeepSetModel(cfg=ckpt["cfg"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

### Inspect evaluation results

```python
import pandas as pd
df = pd.read_csv("models/test_report.csv")
print(df)
```
