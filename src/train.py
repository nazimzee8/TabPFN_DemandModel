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
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
try:
    from snowflake.ml.modeling.distributors.pytorch import (
        PyTorchDistributor,
        PyTorchScalingConfig,
        WorkerResourceConfig,
        get_context,
    )
except ImportError:
    PyTorchDistributor = None
    PyTorchScalingConfig = None
    WorkerResourceConfig = None
    get_context = None

from model import DeepSetModel, ModelConfig
from snowflake_io import materialize_meta_dataset_stage

# ---------------------------------------------------------------------------
# Key constants
# ---------------------------------------------------------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR     = "/tmp/data"
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

USE_HUBER    = False   # off by default; toggle via hyper_params["use_huber"]
LAMBDA_L1    = 0.0     # L1 penalty coefficient; helps Regime B sparse β
HUBER_DELTA  = 1.0     # δ for Huber loss; robustness to Regime C heavy-tailed ε


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

def run_epoch(model, loader, optimizer, scaler, training: bool, device, use_amp,
              loss_fn=None, l1_lambda=0.0, return_sum_count=False):
    """
    Iterate over all meta-datasets in `loader`.  If training=True, backprop per dataset.
    Returns mean loss across all test-row predictions in the epoch.

    Args:
        loss_fn:   callable(y_hat, y) → scalar loss, or None for MSE.
        l1_lambda: L1 penalty coefficient on model parameters (training only).
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
            loss  = F.mse_loss(y_hat, betaX_test) if loss_fn is None else loss_fn(y_hat, betaX_test)

        if l1_lambda > 0.0 and training:
            loss = loss + l1_lambda * sum(p.abs().sum() for p in model.parameters())

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss  += loss.item() * X_test.shape[0]
        total_count += X_test.shape[0]

    if return_sum_count:
        return total_loss, total_count
    return total_loss / max(total_count, 1)


def make_no_padding_rank_indices(n_items, rank, world_size):
    """Shard validation items without sampler padding or duplicate examples."""
    return list(range(rank, n_items, world_size))


def loss_sum_count_to_mse(loss_sum, total_count):
    """Compute a weighted MSE from summed squared loss and prediction count."""
    return float(loss_sum) / max(int(total_count), 1)


def reduce_loss_sum_count(loss_sum, total_count, device, dist_module):
    """All-reduce validation loss sums and counts, then return exact global MSE."""
    stats = torch.tensor([float(loss_sum), float(total_count)], device=device)
    dist_module.all_reduce(stats, op=dist_module.ReduceOp.SUM)
    return loss_sum_count_to_mse(stats[0].item(), stats[1].item())


# ---------------------------------------------------------------------------
# Snowpark upload
# ---------------------------------------------------------------------------

def upload_to_snowflake(local_path: str, stage_path: str):
    """
    Upload a local file to a Snowflake internal stage via Snowpark.
    Wrapped in try/except so it degrades gracefully when running locally.
    """
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        session.file.put(local_path, stage_path, overwrite=True, auto_compress=False)
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
    local_rank = ctx.get_local_rank() if hasattr(ctx, "get_local_rank") else ctx.local_rank
    rank = ctx.get_rank() if hasattr(ctx, "get_rank") else ctx.rank
    world_size = ctx.get_world_size() if hasattr(ctx, "get_world_size") else ctx.world_size
    device    = f"cuda:{local_rank}"
    is_main   = (rank == 0)
    use_amp   = True

    lr           = float(hyper_params.get("lr",           LR))
    weight_decay = float(hyper_params.get("weight_decay", WEIGHT_DECAY))
    d_phi        = int(hyper_params.get("d_phi",          D_PHI))
    d_rho        = int(hyper_params.get("d_rho",          D_RHO))
    dropout      = float(hyper_params.get("dropout",      0.1))
    pool         = hyper_params.get("pool",               POOL)
    max_epochs   = int(hyper_params.get("max_epochs",     MAX_EPOCHS))

    use_huber   = bool(hyper_params.get("use_huber",   USE_HUBER))
    lambda_l1   = float(hyper_params.get("lambda_l1",  LAMBDA_L1))
    huber_delta = float(hyper_params.get("huber_delta", HUBER_DELTA))

    _huber_loss = nn.HuberLoss(delta=huber_delta) if use_huber else None
    loss_fn     = (lambda y_hat, y: _huber_loss(y_hat, y)) if use_huber else None

    # --- DataLoader with DistributedSampler ---
    materialize_meta_dataset_stage(DATA_DIR, splits=("train", "val"))
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.parquet")))
    val_files   = sorted(glob.glob(os.path.join(DATA_DIR, "val",   "*.parquet")))
    if not train_files or not val_files:
        raise FileNotFoundError(
            f"Training requires non-empty train and val parquet splits under {DATA_DIR}; "
            f"found train={len(train_files)}, val={len(val_files)}"
        )
    if len(train_files) % world_size != 0:
        raise ValueError(
            f"Training split has {len(train_files)} tasks, which is not divisible by "
            f"world_size={world_size}. Regenerate or restage train data so every DDP "
            "rank has the same number of backward steps."
        )

    train_dataset = ParquetMetaDataset(train_files)
    val_dataset_all = ParquetMetaDataset(val_files)
    val_indices = make_no_padding_rank_indices(len(val_files), rank, world_size)
    val_dataset = Subset(val_dataset_all, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    train_loader  = DataLoader(train_dataset, batch_size=1, sampler=train_sampler,
                               num_workers=4, prefetch_factor=2, pin_memory=True,
                               collate_fn=identity_collate)
    val_loader    = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                               num_workers=4, prefetch_factor=2, pin_memory=True,
                               collate_fn=identity_collate)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # --- Model ---
    cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool,
                        n_heads=N_HEADS, n_sab_feat=N_SAB_FEAT, n_sab_samp=N_SAB_SAMP,
                        norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=dropout)
    model = DeepSetModel(cfg=cfg).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_mse   = float("inf")
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_mse = run_epoch(model, train_loader, optimizer, scaler, True,  device, use_amp,
                              loss_fn=loss_fn, l1_lambda=lambda_l1)
        with torch.no_grad():
            val_loss_sum, val_count = run_epoch(
                model, val_loader, None, scaler, False, device, use_amp,
                loss_fn=loss_fn, l1_lambda=0.0, return_sum_count=True,
            )

        val_mse = reduce_loss_sum_count(val_loss_sum, val_count, device, dist)

        if is_main:
            print(f"Epoch {epoch:3d}  val_mse={val_mse:.4f}")
            if val_mse < best_val_mse:
                best_val_mse   = val_mse
                patience_count = 0
                ckpt = model.module if isinstance(model, DistributedDataParallel) else model
                ckpt = ckpt._orig_mod if hasattr(ckpt, "_orig_mod") else ckpt
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
    if PyTorchDistributor is None:
        raise RuntimeError(
            "snowflake.ml is required to submit distributed training. "
            "Run train.py inside the Snowflake ML runtime or install snowflake-ml-python."
        )
    hyper_params = json.loads(os.environ.get("BEST_CONFIG", "{}"))

    distributor = PyTorchDistributor(
        train_func=train_fn,
        scaling_config=PyTorchScalingConfig(
            num_nodes=int(os.environ.get("TRAIN_NUM_NODES", "4")),
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
