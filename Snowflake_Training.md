# Snowflake Training — DeepSet TabPFN

Describes how to run the DeepSet training pipeline inside a Snowflake environment
using Snowpark Container Services (SPCS) and Snowflake Model Registry.

---

## Snowflake Environment Construction

Ordered setup: from local machine to a running SPCS training job with model checkpoint
written back to Snowflake.

**Steps:**

1. **Write `model.py` and `train.py`** on the local machine (DeepSet architecture + training loop).
2. **Generate Parquet files** (`meta_train.parquet`, `meta_val.parquet`) via the local DGP script.
3. **Upload Parquet files** to `@META_DATASET_STAGE` via `PUT`.
4. **Build Docker image** (`docker build`) using the Dockerfile below; copies both `model.py` and `train.py`.
5. **Push image** to the Snowflake Image Registry.
6. **Create Compute Pool** (`GPU_NV_S`, 1 A10G node) if it does not already exist.
7. **Deploy SPCS Service** with the stage volume mount so `/data/` maps to `@META_DATASET_STAGE`.
8. **Container runs**: reads `/data/train/*.parquet` and `/data/val/*.parquet`, trains the DeepSet,
   writes `best.pt` to `@MODEL_STAGE/checkpoints/best.pt` on each validation improvement,
   and stops on early stopping (patience=10, monitored metric: val MSE).

**Data and artifact flow:**

```
Local machine
  ├── model.py ────────────────────────────→ Docker image → SPCS container
  ├── train.py ────────────────────────────→ Docker image → SPCS container
  └── *.parquet ──PUT──→ @META_DATASET_STAGE ──vol mount──→ /data/

SPCS container (A10G GPU)
  ├── reads  /data/train/*.parquet + /data/val/*.parquet
  ├── trains DeepSet (phi, rho, psi + 4 equivariant scalars)
  ├── writes best.pt ──PUT──→ @MODEL_STAGE/checkpoints/best.pt
  └── stops on early stopping (patience=10, val MSE)

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

Upload command (run from local machine or Snowpark session):

```sql
PUT file:///local/path/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE;
```

---

## Prerequisite SQL Objects

```sql
CREATE DATABASE TABPFN_DB;
CREATE SCHEMA TABPFN_SCHEMA;
CREATE STAGE META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE MODEL_STAGE        ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE IMAGE REPOSITORY TABPFN_REPO;
```

---

## Compute: Snowpark Container Services (SPCS)

PyTorch training requires a custom container.

### 1. Docker Image

Base: `python:3.11-slim`. Add: `torch`, `numpy`, `pyarrow`, `snowflake-snowpark-python`.

Example `Dockerfile`:

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir torch numpy pyarrow snowflake-snowpark-python
WORKDIR /app
COPY model.py .
COPY train.py .
CMD ["python", "train.py"]
```

### 2. Push to Snowflake Image Registry

```bash
docker tag deepset-trainer <account>.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/tabpfn_repo/deepset-trainer:v1
docker push <account>.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/tabpfn_repo/deepset-trainer:v1
```

### 3. Compute Pool

```sql
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1 MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S;   -- single A10G
```

### 4. Service Spec

Mount the stage as a volume so the container can read Parquet files directly from disk:

```yaml
spec:
  containers:
    - name: trainer
      image: <account>.registry.snowflakecomputing.com/tabpfn_db/tabpfn_schema/tabpfn_repo/deepset-trainer:v1
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      source: "@META_DATASET_STAGE"
```

Deploy the service:

```sql
CREATE SERVICE DEEPSET_TRAINER_SVC
  IN COMPUTE POOL DEEPSET_GPU_POOL
  FROM SPECIFICATION $$
  <paste yaml above>
  $$;
```

### 5. Checkpoint Output

Write best checkpoint back to the model stage on improvement:

```python
session.file.put(
    local_file_name="best.pt",
    stage_location="@MODEL_STAGE/checkpoints/",
    overwrite=True,
)
```

---

## Training Loop Adaptation

Inside the container, `train.py` follows this loop:

```python
import os, glob, random
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

DATA_DIR   = "/data"
PATIENCE   = 10
MAX_EPOCHS = 200

# --- Load file lists ---
train_files = glob.glob(os.path.join(DATA_DIR, "train", "*.parquet"))
val_files   = glob.glob(os.path.join(DATA_DIR, "val",   "*.parquet"))

def load_parquet(path):
    table = pq.read_table(path)
    d = table.to_pydict()
    return {k: torch.tensor(np.array(v[0]), dtype=torch.float32) for k, v in d.items()}

# --- Training loop ---
best_val_mse   = float("inf")
patience_count = 0

for epoch in range(MAX_EPOCHS):
    random.shuffle(train_files)

    model.train()
    for fpath in train_files:
        batch = load_parquet(fpath)
        optimizer.zero_grad()
        y_hat = model(batch["X_train"], batch["y_train"], batch["X_test"])
        loss  = nn.functional.mse_loss(y_hat, batch["betaX_test"])
        loss.backward()
        optimizer.step()

    # --- Validation ---
    model.eval()
    val_mses = []
    with torch.no_grad():
        for fpath in val_files:
            batch   = load_parquet(fpath)
            y_hat   = model(batch["X_train"], batch["y_train"], batch["X_test"])
            val_mses.append(nn.functional.mse_loss(y_hat, batch["betaX_test"]).item())
    val_mse = float(np.mean(val_mses))
    print(f"Epoch {epoch:3d}  val_mse={val_mse:.4f}")

    if val_mse < best_val_mse:
        best_val_mse   = val_mse
        patience_count = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

# Upload best checkpoint to Snowflake stage
# (run via Snowpark session inside container)
```

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
- Created by `torch.save(model.state_dict(), "best.pt")` whenever val MSE improves.
- Uploaded from the SPCS container via `session.file.put("best.pt", "@MODEL_STAGE/checkpoints/", overwrite=True)`.

**Used for inference on any new synthetic dataset without retraining:**

```python
import torch
from model import DeepSetModel   # same architecture class

model = DeepSetModel(d_phi=64, d_rho=64)
model.load_state_dict(torch.load("best.pt", map_location="cpu"))
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
