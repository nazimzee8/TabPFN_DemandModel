"""
train.py

Training loop for the DeepSetModel inside a Snowpark Container Services (SPCS)
environment.  Reads meta-datasets from Parquet files, trains with early stopping,
and uploads the best checkpoint to a Snowflake model stage.

Usage (inside container):
    python train.py
"""

import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from snowflake.ml.modeling.distributors.pytorch import (
    PyTorchDistributor,
    PyTorchScalingConfig,
    WorkerResourceConfig,
    get_context,
)

from model import DeepSetModel, ModelConfig

# ---------------------------------------------------------------------------
# Key constants
# ---------------------------------------------------------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR     = "/data"
PATIENCE     = 10
MAX_EPOCHS   = 200
D_PHI        = 128
D_RHO        = 256
POOL         = "pna"      # "sum"|"mean"|"max"|"pna"|"learned"|"attn"|"multipool"
N_HEADS      = 4
N_SAB_FEAT   = 1
N_SAB_SAMP   = 1
NORM_FEAT    = True
NORM_TARGET  = True
LR           = 1e-3
WEIGHT_DECAY = 1e-4
USE_AMP      = DEVICE == "cuda"


# ---------------------------------------------------------------------------
# Parquet loader
# ---------------------------------------------------------------------------

def load_parquet(path):
    """
    Load a single-row meta-dataset Parquet file.

    Returns:
        X_train    : torch.FloatTensor  (n_train, p)
        y_train    : torch.FloatTensor  (n_train,)
        X_test     : torch.FloatTensor  (n_test,  p)
        betaX_test : torch.FloatTensor  (n_test,)
    """
    table = pq.read_table(path)
    d = table.to_pydict()
    X_train    = torch.tensor(np.array(d["X_train"][0]),    dtype=torch.float32)
    y_train    = torch.tensor(np.array(d["y_train"][0]),    dtype=torch.float32)
    X_test     = torch.tensor(np.array(d["X_test"][0]),     dtype=torch.float32)
    betaX_test = torch.tensor(np.array(d["betaX_test"][0]), dtype=torch.float32)
    return X_train, y_train, X_test, betaX_test


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

class ParquetMetaDataset(Dataset):
    """Each item is one meta-dataset (X_train, y_train, X_test, betaX_test)."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_parquet(self.files[idx])


def identity_collate(batch):
    """batch_size=1; return the item directly without default list-wrapping."""
    return batch[0]


def make_loader(files, shuffle):
    return DataLoader(
        ParquetMetaDataset(files),
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=USE_AMP,
        collate_fn=identity_collate,
    )


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, scaler, training: bool, device, use_amp):
    """
    Iterate over all meta-datasets in `loader`.  If training=True, backprop per dataset.
    Returns mean MSE across all test-row predictions in the epoch.
    """
    model.train(training)
    total_loss  = 0.0
    total_count = 0

    for X_train, y_train, X_test, betaX_test in loader:
        X_train    = X_train.to(device)
        y_train    = y_train.to(device)
        X_test     = X_test.to(device)
        betaX_test = betaX_test.to(device)

        if training:
            optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            y_hat = model(X_train, y_train, X_test)          # batched: (m,)
            loss  = F.mse_loss(y_hat, betaX_test)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss  += loss.item() * X_test.shape[0]
        total_count += X_test.shape[0]

    return total_loss / max(total_count, 1)


# ---------------------------------------------------------------------------
# Snowpark upload
# ---------------------------------------------------------------------------

def upload_to_snowflake(local_path: str, stage_path: str):
    """
    Upload a local file to a Snowflake internal stage via Snowpark.
    Wrapped in try/except so it degrades gracefully when running locally.
    """
    try:
        from snowflake.snowpark import Session

        connection_params = {
            "host":          os.environ["SNOWFLAKE_HOST"],
            "account":       os.environ.get("SNOWFLAKE_ACCOUNT_OVERRIDE", ""),
            "authenticator": "oauth",
            "token":         open("/snowflake/session/token").read(),
            "warehouse":     os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
            "database":      "TABPFN_DB",
            "schema":        "TABPFN_SCHEMA",
        }
        session = Session.builder.configs(connection_params).create()
        session.file.put(local_path, stage_path, overwrite=True)
        print(f"Uploaded {local_path} to {stage_path}")
    except Exception as exc:
        print(f"[WARNING] Snowpark upload failed (skipping): {exc}")


# ---------------------------------------------------------------------------
# Distributed training function (invoked by PyTorchDistributor on each worker)
# ---------------------------------------------------------------------------

def train_fn(dataset_map, hyper_params):
    """
    Invoked by PyTorchDistributor on each worker.
    DDP process group is initialized automatically by the distributor.
    """
    import torch.distributed as dist

    ctx       = get_context()
    device    = f"cuda:{ctx.local_rank}"
    is_main   = (ctx.rank == 0)
    use_amp   = True

    lr           = float(hyper_params.get("lr",           LR))
    weight_decay = float(hyper_params.get("weight_decay", WEIGHT_DECAY))
    d_phi        = int(hyper_params.get("d_phi",          D_PHI))
    d_rho        = int(hyper_params.get("d_rho",          D_RHO))
    dropout      = float(hyper_params.get("dropout",      0.1))
    pool         = hyper_params.get("pool",               POOL)
    max_epochs   = int(hyper_params.get("max_epochs",     MAX_EPOCHS))

    # --- DataLoader with DistributedSampler ---
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.parquet")))
    val_files   = sorted(glob.glob(os.path.join(DATA_DIR, "val",   "*.parquet")))
    train_dataset = ParquetMetaDataset(train_files)
    val_dataset   = ParquetMetaDataset(val_files)

    train_sampler = DistributedSampler(train_dataset, num_replicas=ctx.world_size,
                                       rank=ctx.rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=ctx.world_size,
                                       rank=ctx.rank, shuffle=False)
    train_loader  = DataLoader(train_dataset, batch_size=1, sampler=train_sampler,
                               num_workers=4, prefetch_factor=2, pin_memory=True,
                               collate_fn=identity_collate)
    val_loader    = DataLoader(val_dataset,   batch_size=1, sampler=val_sampler,
                               num_workers=4, prefetch_factor=2, pin_memory=True,
                               collate_fn=identity_collate)

    # --- Model (DDP wrapping handled by PyTorchDistributor) ---
    cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool,
                        n_heads=N_HEADS, n_sab_feat=N_SAB_FEAT, n_sab_samp=N_SAB_SAMP,
                        norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=dropout)
    model = DeepSetModel(cfg=cfg).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_mse   = float("inf")
    patience_count = 0

    dist.init_process_group(backend="nccl")  # no-op if already initialized by distributor

    for epoch in range(1, max_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_mse = run_epoch(model, train_loader, optimizer, scaler, True,  device, use_amp)
        with torch.no_grad():
            val_mse = run_epoch(model, val_loader,   None,      scaler, False, device, use_amp)

        val_t = torch.tensor(val_mse, device=device)
        dist.all_reduce(val_t, op=dist.ReduceOp.AVG)
        val_mse = val_t.item()

        if is_main:
            print(f"Epoch {epoch:3d}  val_mse={val_mse:.4f}")
            if val_mse < best_val_mse:
                best_val_mse   = val_mse
                patience_count = 0
                ckpt = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({"state_dict": ckpt.state_dict(), "cfg": ckpt.cfg}, "best.pt")
            else:
                patience_count += 1

        stop = torch.tensor(int(patience_count >= PATIENCE), device=device)
        dist.broadcast(stop, src=0)
        if stop.item():
            if is_main:
                print("Early stopping.")
            break

    if is_main:
        upload_to_snowflake("best.pt", "@MODEL_STAGE/checkpoints/")

    return {"val_mse": best_val_mse}


# ---------------------------------------------------------------------------
# Entry point — submits train_fn via PyTorchDistributor
# ---------------------------------------------------------------------------

def main():
    import json
    hyper_params = json.loads(os.environ.get("BEST_CONFIG", "{}"))

    distributor = PyTorchDistributor(
        train_func=train_fn,
        scaling_config=PyTorchScalingConfig(
            num_nodes=2,
            num_workers_per_node=1,   # 1 A10G per GPU_NV_S node
            resource_requirements_per_worker=WorkerResourceConfig(
                num_cpus=4,
                num_gpus=1,
            ),
        ),
    )
    result = distributor.run(hyper_params=hyper_params)
    print("Training result:", result)


if __name__ == "__main__":
    main()
