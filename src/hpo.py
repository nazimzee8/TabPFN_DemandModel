"""
Hyperparameter optimization using Snowflake ML Tuner + RandomSearch.

Run before train.py to find the best config. Writes best_config.json to
@MODEL_STAGE/hpo/ on completion.

Usage (via run_training_job.py or directly):
    python hpo.py
"""
import os
os.environ["HOME"] = "/tmp"   # SPCS: redirect ~ to writable path before connector import

import glob
import json
import tempfile
import torch
import pyarrow.parquet as pq
from snowflake.ml.modeling.tune import Tuner, TunerConfig, get_tuner_context, uniform, loguniform, choice
from snowflake.ml.modeling.tune.search import RandomSearch
from snowflake.snowpark.context import get_active_session

from train import (run_epoch, make_loader, DATA_DIR, PATIENCE,
                   N_HEADS, N_SAB_FEAT, N_SAB_SAMP, NORM_FEAT, NORM_TARGET)
from model import DeepSetModel, ModelConfig
from snowflake_io import materialize_meta_dataset_stage


BASE_D_PHI_CANDIDATES = [64, 128, 256, 512]
BASE_D_RHO_CANDIDATES = [128, 256, 512, 1024]


def materialize_hpo_splits():
    materialize_meta_dataset_stage(DATA_DIR, splits=("train", "val"))
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.parquet")))
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, "val", "*.parquet")))
    if not train_files or not val_files:
        raise FileNotFoundError(
            f"HPO requires non-empty train and val parquet splits under {DATA_DIR}; "
            f"found train={len(train_files)}, val={len(val_files)}"
        )
    return train_files, val_files


def _read_positive_cardinality(values_by_column, column, path):
    values = values_by_column.get(column)
    if not values or values[0] is None:
        raise ValueError(f"HPO cardinality scan found missing {column!r} in {path}")

    value = int(values[0])
    if value <= 0:
        raise ValueError(
            f"HPO cardinality scan found non-positive {column!r}={value} in {path}"
        )
    return value


def scan_hpo_cardinalities(files):
    max_p = 0
    max_n_train = 0
    for path in files:
        try:
            table = pq.read_table(path, columns=["p", "n_train"])
        except Exception as exc:
            raise ValueError(
                f"HPO cardinality scan could not read required columns "
                f"'p' and 'n_train' from {path}"
            ) from exc

        values_by_column = table.to_pydict()
        max_p = max(max_p, _read_positive_cardinality(values_by_column, "p", path))
        max_n_train = max(
            max_n_train,
            _read_positive_cardinality(values_by_column, "n_train", path),
        )

    if max_p <= 0 or max_n_train <= 0:
        raise ValueError(
            f"HPO cardinality scan failed: max_p={max_p}, max_n_train={max_n_train}"
        )
    return max_p, max_n_train


def cardinality_aware_candidates(name, base_candidates, observed_name, observed_value):
    candidates = [
        d for d in base_candidates
        if d >= observed_value and d % N_HEADS == 0
    ]
    if not candidates:
        raise ValueError(
            f"No HPO candidates for {name} satisfy {observed_name}={observed_value} "
            f"with N_HEADS={N_HEADS}. Expand the base {name} candidate list beyond "
            f"{base_candidates} before launching HPO."
        )
    return candidates


def train_for_hpo():
    ctx = get_tuner_context()
    hp  = ctx.get_hyper_params()

    lr           = float(hp.get("lr",           1e-3))
    weight_decay = float(hp.get("weight_decay", 1e-4))
    d_phi        = int(hp.get("d_phi",          128))
    d_rho        = int(hp.get("d_rho",          256))
    dropout      = float(hp.get("dropout",      0.1))
    pool         = hp.get("pool",               "pna")

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    train_files, val_files = materialize_hpo_splits()
    train_loader = make_loader(train_files, shuffle=True)
    val_loader   = make_loader(val_files,   shuffle=False)

    cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool,
                        n_heads=N_HEADS, n_sab_feat=N_SAB_FEAT, n_sab_samp=N_SAB_SAMP,
                        norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=dropout)
    model = DeepSetModel(cfg=cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_mse   = float("inf")
    patience_count = 0

    for epoch in range(1, 31):   # 30 epochs max for HPO signal
        run_epoch(model, train_loader, optimizer, scaler, True,  device, use_amp)
        with torch.no_grad():
            val_mse = run_epoch(model, val_loader, None, scaler, False, device, use_amp)
        if val_mse < best_val_mse:
            best_val_mse   = val_mse
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break

    ctx.report(metrics={"val_mse": best_val_mse})


train_files, val_files = materialize_hpo_splits()
max_p, max_n_train = scan_hpo_cardinalities(train_files + val_files)
d_phi_candidates = cardinality_aware_candidates(
    "d_phi", BASE_D_PHI_CANDIDATES, "max(p)", max_p
)
d_rho_candidates = cardinality_aware_candidates(
    "d_rho", BASE_D_RHO_CANDIDATES, "max(n_train)", max_n_train
)
print(
    "Cardinality-aware HPO dimensions:",
    {
        "max_p": max_p,
        "max_n_train": max_n_train,
        "d_phi_candidates": d_phi_candidates,
        "d_rho_candidates": d_rho_candidates,
    },
)


tuner = Tuner(
    train_for_hpo,
    search_space={
        "lr":           loguniform(1e-4, 1e-2),
        "weight_decay": loguniform(1e-5, 1e-3),
        "d_phi":        choice(d_phi_candidates),
        "d_rho":        choice(d_rho_candidates),
        "dropout":      uniform(0.0, 0.3),
        "pool":         choice(["pna", "attn", "multipool"]),
    },
    tuner_config=TunerConfig(
        num_trials=40,
        metric="val_mse",
        mode="min",
        search_alg=RandomSearch(random_state=42),
        max_concurrent_trials=4,
        resource_per_trial={"GPU": 1},
        uses_snowflake_trainer=False,
    ),
)

results  = tuner.run()
best_row = results.best_result
best_config = {
    "lr":           float(best_row["lr"]),
    "weight_decay": float(best_row["weight_decay"]),
    "d_phi":        int(best_row["d_phi"]),
    "d_rho":        int(best_row["d_rho"]),
    "dropout":      float(best_row["dropout"]),
    "pool":         str(best_row["pool"]),
}
print("Best hyperparameters:", best_config)

# Reuse the active session for upload
session = get_active_session()
with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
    tmp = os.path.join(tmp_dir, "best_config.json")
    with open(tmp, "w") as f:
        json.dump(best_config, f)
    session.file.put(tmp, "@MODEL_STAGE/hpo/", overwrite=True, auto_compress=False)
print("Uploaded best_config.json to @MODEL_STAGE/hpo/")
