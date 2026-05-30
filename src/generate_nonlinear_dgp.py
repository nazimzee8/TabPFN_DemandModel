"""
generate_nonlinear_dgp.py

Generate meta-datasets from nonlinear synthetic data-generating processes (DGPs).
Mirrors src/generate_dgp.py exactly in structure, using nonlinear regimes I–L.

Usage:
    python src/generate_nonlinear_dgp.py --n_datasets 1000 --out_dir data/nonlinear/
"""

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NONLINEAR_REGIMES = ["I", "J", "K", "L"]
BASE_SEED         = 20260601


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------

def _sample_n_p(rng):
    """Rejection-sample (n, p) until constraints are satisfied.

    p ~ Poisson(10), reject until p >= 1.
    n ~ Poisson(200), reject until n >= 5 and n >= 5 * p.
    """
    while True:
        p = int(rng.poisson(10))
        n = int(rng.poisson(200))
        if p >= 1 and n >= 5 and n >= 5 * p:
            return n, p


def _generate_dataset(rng, regime):
    """Generate a single nonlinear dataset for the given regime.

    Regimes
    -------
    I — Quadratic:      betaX = (X**2) @ beta_sq
    J — Sinusoidal:     betaX = sin(2π * (X @ beta)) * sqrt(p)
    K — Interactions:   betaX = sum of pairwise multiplicative terms
    L — ReLU:           betaX = sum(relu(x_i * beta_i)) - mean
    """
    n, p = _sample_n_p(rng)
    X = rng.standard_normal((n, p))

    if regime == "I":
        # Quadratic: each feature contributes x_i^2 * beta_i
        beta_sq = rng.standard_normal(p)
        betaX = (X ** 2) @ beta_sq

    elif regime == "J":
        # Sinusoidal: nonlinear projection through sin
        beta = rng.standard_normal(p)
        beta /= (np.linalg.norm(beta) / np.sqrt(p))   # normalise for stable z scale
        betaX = np.sin(2 * np.pi * (X @ beta)) * np.sqrt(p)

    elif regime == "K":
        # Pairwise multiplicative interactions
        n_pairs = max(1, p // 2)
        pairs = rng.choice(p, size=(n_pairs, 2), replace=False)
        c = rng.standard_normal(n_pairs)
        betaX = np.zeros(n)
        for k, (i, j) in enumerate(pairs):
            betaX += c[k] * X[:, i] * X[:, j]

    elif regime == "L":
        # ReLU / threshold activation, centred
        beta = rng.standard_normal(p)
        betaX = np.sum(np.maximum(0.0, X * beta[np.newaxis, :]), axis=1)
        betaX -= betaX.mean()

    else:
        raise ValueError(f"Unknown nonlinear regime: {regime!r}")

    # Gaussian observation noise
    y = betaX + rng.standard_normal(n)

    # Train / test split (80 / 20 within dataset)
    n_train = int(0.8 * n)
    n_test = n - n_train
    if n_test < 1:
        n_train -= 1
        n_test = 1

    X_train      = X[:n_train]
    y_train      = y[:n_train]
    X_test       = X[n_train:]
    betaX_test   = betaX[n_train:]

    return {
        "X_train":    X_train,
        "y_train":    y_train,
        "X_test":     X_test,
        "betaX_test": betaX_test,
        "n":          n,
        "p":          p,
        "n_train":    n_train,
        "n_test":     n_test,
        "regime":     regime,
    }


# ---------------------------------------------------------------------------
# Parquet writer — identical schema to generate_dgp.py
# ---------------------------------------------------------------------------

def write_parquet(ds, filepath):
    """Write a dataset dict to a single-row parquet file.

    Schema matches generate_dgp.py (X_train, y_train, X_test, betaX_test,
    n, p, n_train, n_test, prior_regime).
    """
    table = pa.table({
        "X_train":    pa.array([ds["X_train"].tolist()],    type=pa.list_(pa.list_(pa.float64()))),
        "y_train":    pa.array([ds["y_train"].tolist()],    type=pa.list_(pa.float64())),
        "X_test":     pa.array([ds["X_test"].tolist()],     type=pa.list_(pa.list_(pa.float64()))),
        "betaX_test": pa.array([ds["betaX_test"].tolist()], type=pa.list_(pa.float64())),
        "n":          pa.array([int(ds["n"])],              type=pa.int64()),
        "p":          pa.array([int(ds["p"])],              type=pa.int64()),
        "n_train":    pa.array([int(ds["n_train"])],        type=pa.int64()),
        "n_test":     pa.array([int(ds["n_test"])],         type=pa.int64()),
        "prior_regime": pa.array([ds["regime"]],            type=pa.utf8()),
    })
    pq.write_table(table, filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate nonlinear DGP meta-datasets (regimes I–L)."
    )
    parser.add_argument(
        "--n_datasets", type=int, default=1000,
        help="Total number of datasets to generate (default: 1000).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/nonlinear/",
        help="Root output directory (default: data/nonlinear/).",
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help=f"RNG seed (default: {BASE_SEED}).",
    )
    args = parser.parse_args()

    n_datasets = args.n_datasets
    out_dir    = args.out_dir

    # Split sizes: 80 % train, 10 % val, 10 % test
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

    rng = np.random.default_rng(seed=args.seed)

    print(f"Generating {n_datasets} nonlinear datasets -> {train_dir}, {val_dir}, {test_dir}")
    print(f"  train: {n_train_split}  val: {n_val_split}  test: {n_test_split}")

    for idx in range(n_datasets):
        regime = NONLINEAR_REGIMES[rng.integers(0, 4)]
        ds = _generate_dataset(rng, regime)

        # Determine split and local filename
        if idx < n_train_split:
            split_dir  = train_dir
            filename   = f"dataset_{idx:04d}.parquet"
        elif idx < n_train_split + n_val_split:
            split_dir  = val_dir
            local_idx  = idx - n_train_split
            filename   = f"dataset_{local_idx:04d}.parquet"
        else:
            split_dir  = test_dir
            local_idx  = idx - n_train_split - n_val_split
            filename   = f"dataset_{local_idx:04d}.parquet"

        filepath = os.path.join(split_dir, filename)
        write_parquet(ds, filepath)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1:4d}/{n_datasets}] written {filename} to {split_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
