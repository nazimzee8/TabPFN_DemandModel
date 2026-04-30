"""
generate_dgp.py

Generate meta-datasets from a synthetic demand data-generating process (DGP).

Usage:
    python generate_dgp.py --n_datasets 1000 --out_dir data/
"""

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------

def sample_params(rng):
    """Rejection-sample (n, p) until constraints are satisfied."""
    while True:
        p = rng.poisson(10)
        n = rng.poisson(200)
        if p >= 1 and n >= 5 and n >= 5 * p:
            return n, p


def generate_X_regime_A(rng, n, p):
    return rng.standard_normal((n, p))


def generate_X_regime_D(rng, n, p):
    """AR(1) design matrix with rho=0.6."""
    X = np.empty((n, p))
    X[:, 0] = rng.standard_normal(n)
    for k in range(1, p):
        X[:, k] = 0.6 * X[:, k - 1] + np.sqrt(0.64) * rng.standard_normal(n)
    return X


def generate_dataset(rng, regime):
    """Generate a single dataset for the given regime."""
    n, p = sample_params(rng)

    # --- Features ---
    if regime in ("A", "B", "C"):
        X = generate_X_regime_A(rng, n, p)
    else:  # D
        X = generate_X_regime_D(rng, n, p)

    # --- Coefficients ---
    if regime == "B":
        beta = rng.normal(0, 2, size=p)
        mask = rng.random(p) < 0.70
        beta[mask] = 0.0
    else:
        beta = rng.standard_normal(p)

    # --- Noise ---
    if regime == "C":
        eps = rng.standard_t(df=3, size=n)
    else:
        eps = rng.standard_normal(n)

    # --- Outcome ---
    y = X @ beta + eps          # (n,)
    betaX = X @ beta            # (n,)  — noiseless linear part

    # --- Train / test split ---
    n_train = int(0.8 * n)
    n_test = n - n_train
    # Guarantee at least one test sample
    if n_test < 1:
        n_train -= 1
        n_test = 1

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    betaX_test = betaX[n_train:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "betaX_test": betaX_test,
        "n": n,
        "p": p,
        "n_train": n_train,
        "n_test": n_test,
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def write_parquet(ds, filepath):
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_test = ds["X_test"]
    betaX_test = ds["betaX_test"]
    n = ds["n"]
    p = ds["p"]
    n_train = ds["n_train"]
    n_test = ds["n_test"]
    regime = ds["regime"]

    table = pa.table({
        "X_train":      pa.array([X_train.tolist()],    type=pa.list_(pa.list_(pa.float64()))),
        "y_train":      pa.array([y_train.tolist()],    type=pa.list_(pa.float64())),
        "X_test":       pa.array([X_test.tolist()],     type=pa.list_(pa.list_(pa.float64()))),
        "betaX_test":   pa.array([betaX_test.tolist()], type=pa.list_(pa.float64())),
        "n":            pa.array([int(n)],              type=pa.int64()),
        "p":            pa.array([int(p)],              type=pa.int64()),
        "n_train":      pa.array([int(n_train)],        type=pa.int64()),
        "n_test":       pa.array([int(n_test)],         type=pa.int64()),
        "prior_regime": pa.array([regime],              type=pa.utf8()),
    })
    pq.write_table(table, filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DGP meta-datasets.")
    parser.add_argument("--n_datasets", type=int, default=1000,
                        help="Total number of datasets to generate (default: 1000).")
    parser.add_argument("--out_dir", type=str, default="data/",
                        help="Root output directory (default: data/).")
    args = parser.parse_args()

    n_datasets = args.n_datasets
    out_dir = args.out_dir

    # Split sizes
    n_train_split = int(0.8 * n_datasets)
    n_val_split   = int(0.1 * n_datasets)
    n_test_split  = n_datasets - n_train_split - n_val_split

    # Create output directories
    train_dir = os.path.join(out_dir, "train")
    val_dir   = os.path.join(out_dir, "val")
    test_dir  = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    regimes = ["A", "B", "C", "D"]
    rng = np.random.default_rng(seed=42)

    print(f"Generating {n_datasets} datasets → {train_dir}, {val_dir}, {test_dir}")
    print(f"  train: {n_train_split}  val: {n_val_split}  test: {n_test_split}")

    for idx in range(n_datasets):
        regime = regimes[rng.integers(0, 4)]
        ds = generate_dataset(rng, regime)

        # Determine split and filename
        if idx < n_train_split:
            split_dir = train_dir
            filename = f"dataset_{idx:04d}.parquet"
        elif idx < n_train_split + n_val_split:
            split_dir = val_dir
            local_idx = idx - n_train_split
            filename = f"dataset_{local_idx:04d}.parquet"
        else:
            split_dir = test_dir
            local_idx = idx - n_train_split - n_val_split
            filename = f"dataset_{local_idx:04d}.parquet"

        filepath = os.path.join(split_dir, filename)
        write_parquet(ds, filepath)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1:4d}/{n_datasets}] written {filename} to {split_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
