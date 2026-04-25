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

def run_epoch(model, loader, optimizer, scaler, training: bool):
    """
    Iterate over all meta-datasets in `loader`.  If training=True, backprop per dataset.
    Returns mean MSE across all test-row predictions in the epoch.
    """
    model.train(training)
    total_loss  = 0.0
    total_count = 0

    for X_train, y_train, X_test, betaX_test in loader:
        X_train    = X_train.to(DEVICE)
        y_train    = y_train.to(DEVICE)
        X_test     = X_test.to(DEVICE)
        betaX_test = betaX_test.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
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
# Main training loop
# ---------------------------------------------------------------------------

def main():
    # Discover split files
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.parquet")))
    val_files   = sorted(glob.glob(os.path.join(DATA_DIR, "val",   "*.parquet")))

    print(f"Found {len(train_files)} train files, {len(val_files)} val files.")

    if not train_files:
        raise RuntimeError(
            f"No training files found under {DATA_DIR}/train/. "
            "Check that @META_DATASET_STAGE/train/ is populated and the stage volume is mounted correctly."
        )
    if not val_files:
        raise RuntimeError(
            f"No validation files found under {DATA_DIR}/val/. "
            "Check that @META_DATASET_STAGE/val/ is populated."
        )

    # Build model, compile, optimizer, scaler
    cfg = ModelConfig(
        d_phi=D_PHI, d_rho=D_RHO, pool=POOL, n_heads=N_HEADS,
        n_sab_feat=N_SAB_FEAT, n_sab_samp=N_SAB_SAMP,
        norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=0.1,
    )
    model     = DeepSetModel(cfg=cfg).to(DEVICE)
    torch._dynamo.config.suppress_errors = True
    model     = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    train_loader = make_loader(train_files, shuffle=True)
    val_loader   = make_loader(val_files,   shuffle=False)

    best_val_mse     = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_mse = run_epoch(model, train_loader, optimizer, scaler, training=True)

        with torch.no_grad():
            val_mse = run_epoch(model, val_loader, optimizer=None, scaler=scaler, training=False)

        print(f"Epoch {epoch:3d}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}")

        # Early stopping / checkpoint
        if val_mse < best_val_mse:
            best_val_mse     = val_mse
            patience_counter = 0
            # torch.compile wraps the module; _orig_mod holds the original state dict
            ckpt_module = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"state_dict": ckpt_module.state_dict(), "cfg": ckpt_module.cfg}, "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {PATIENCE} consecutive epochs).")
                break

    print(f"Training complete. Best val MSE: {best_val_mse:.4f}")

    # Upload checkpoint to Snowflake (no-op if not in SPCS)
    upload_to_snowflake("best.pt", "@MODEL_STAGE/checkpoints/")


if __name__ == "__main__":
    main()
