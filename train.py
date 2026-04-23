"""
train.py

Training loop for the DeepSetModel inside a Snowpark Container Services (SPCS)
environment.  Reads meta-datasets from Parquet files, trains with early stopping,
and uploads the best checkpoint to a Snowflake model stage.

Usage (inside container):
    python train.py
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

from model import DeepSetModel

# ---------------------------------------------------------------------------
# Key constants
# ---------------------------------------------------------------------------
DATA_DIR     = "/data"
PATIENCE     = 10
MAX_EPOCHS   = 200
D_PHI        = 64
D_RHO        = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4


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
# Dataset file discovery
# ---------------------------------------------------------------------------

def get_parquet_files(split_dir):
    """Return sorted list of .parquet file paths in split_dir."""
    files = [
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".parquet")
    ]
    files.sort()
    return files


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(model, files, optimizer, training: bool):
    """
    Iterate over all files in `files`.  If training=True, backprop per dataset.
    Returns mean MSE across all test-row predictions in the epoch.
    """
    model.train(training)
    total_loss  = 0.0
    total_count = 0

    for path in files:
        X_train, y_train, X_test, betaX_test = load_parquet(path)

        if training:
            optimizer.zero_grad()

        losses = []
        for k in range(X_test.shape[0]):
            y_hat  = model(X_train, y_train, X_test[k])          # scalar
            loss_k = F.mse_loss(y_hat.unsqueeze(0), betaX_test[k].unsqueeze(0))
            losses.append(loss_k)

        loss = torch.stack(losses).mean()

        if training:
            loss.backward()
            optimizer.step()

        total_loss  += loss.item() * len(losses)
        total_count += len(losses)

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
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    train_files = get_parquet_files(train_dir)
    val_files   = get_parquet_files(val_dir)

    print(f"Found {len(train_files)} train files, {len(val_files)} val files.")

    # Build model and optimizer
    model     = DeepSetModel(d_phi=D_PHI, d_rho=D_RHO, pool="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_mse     = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle train files each epoch
        random.shuffle(train_files)

        train_mse = run_epoch(model, train_files, optimizer, training=True)

        with torch.no_grad():
            val_mse = run_epoch(model, val_files, optimizer=None, training=False)

        print(f"Epoch {epoch:3d}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}")

        # Early stopping / checkpoint
        if val_mse < best_val_mse:
            best_val_mse     = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {PATIENCE} consecutive epochs).")
                break

    print(f"Training complete. Best val MSE: {best_val_mse:.4f}")

    # Load best weights
    model.load_state_dict(torch.load("best.pt", map_location="cpu"))

    # Upload checkpoint to Snowflake (no-op if not in SPCS)
    upload_to_snowflake("best.pt", "@MODEL_STAGE/checkpoints/")


if __name__ == "__main__":
    main()
