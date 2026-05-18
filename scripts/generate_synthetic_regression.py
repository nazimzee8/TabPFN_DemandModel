"""
generate_synthetic_regression.py
=================================
Generate synthetic regression evaluation data (primary suite) as parquet files locally.

Mirrors generate_dgp.py exactly — produces parquet files that can be PUT to
@EVAL_DATASET_STAGE/synthetic_regression_prepared/{suite_id}/primary/ and indexed
by prepare_synthetic_regression.py (parquet workflow branch).

Usage:
    python scripts/generate_synthetic_regression.py --n_datasets 200 --out_dir data/
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# DGP helpers (replicated from prepare_synthetic_regression.py)
# ---------------------------------------------------------------------------

def _sample_params(rng: np.random.Generator) -> tuple[int, int]:
    """Rejection-sample (n, p) with p>=1, n>=5, n>=5p, Poisson(200)/Poisson(10)."""
    while True:
        n = int(rng.poisson(200))
        p = int(rng.poisson(10))
        if p >= 1 and n >= 5 and n >= 5 * p:
            return n, p


def _generate_X_regime_A(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Standard normal features."""
    return rng.standard_normal((n, p))


def _generate_X_regime_D(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """AR(1) features with rho=0.6."""
    X = np.zeros((n, p))
    X[0] = rng.standard_normal(p)
    for t in range(1, n):
        X[t] = 0.6 * X[t - 1] + math.sqrt(0.64) * rng.standard_normal(p)
    return X


def _generate_dataset(rng: np.random.Generator, regime: str) -> dict:
    """Generate a single primary-suite dataset for the given regime.

    Regimes A/B/C/D — same DGP as _generate_core_dataset in
    prepare_synthetic_regression.py.
    """
    n, p = _sample_params(rng)

    # Features
    if regime == "D":
        X = _generate_X_regime_D(rng, n, p)
    else:
        X = _generate_X_regime_A(rng, n, p)

    # Coefficients
    if regime == "B":
        beta = rng.normal(0, 2, p)
        mask = rng.random(p) < 0.70
        beta[mask] = 0.0
    else:
        beta = rng.standard_normal(p)

    # Noise
    if regime == "C":
        eps = rng.standard_t(df=3, size=n)
    else:
        eps = rng.standard_normal(n)

    betaX = X @ beta
    y = betaX + eps  # target_noise_scale = 1.0

    return {
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "prior_regime": regime,
        "n_total": n,
        "p_signal": p,
    }


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def _write_parquet(ds: dict, filepath: str) -> None:
    """Write one dataset row to a parquet file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "X":                    pa.array([ds["X"].tolist()],     type=pa.list_(pa.list_(pa.float64()))),
        "y":                    pa.array([ds["y"].tolist()],     type=pa.list_(pa.float64())),
        "betaX":                pa.array([ds["betaX"].tolist()], type=pa.list_(pa.float64())),
        "suite_family":         pa.array(["primary"],            type=pa.utf8()),
        "prior_regime":         pa.array([ds["prior_regime"]],   type=pa.utf8()),
        "n_total":              pa.array([ds["n_total"]],        type=pa.int64()),
        "p_signal":             pa.array([ds["p_signal"]],       type=pa.int64()),
        "p_noise":              pa.array([0],                    type=pa.int64()),
        "p_total":              pa.array([ds["p_signal"]],       type=pa.int64()),
        "target_noise_scale":   pa.array([1.0],                  type=pa.float64()),
        "training_size_anchor": pa.array([False],                type=pa.bool_()),
        "feature_noise_level":  pa.array([0],                    type=pa.int64()),
    })
    pq.write_table(table, filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic regression evaluation data (primary suite) as parquet files."
    )
    parser.add_argument(
        "--n_datasets", type=int, default=200,
        help="Number of primary-suite datasets to generate (default: 200).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/",
        help="Root output directory (default: data/).",
    )
    parser.add_argument(
        "--suite_id", type=str, default="linear_poisson_v1_recommended",
        help="Suite identifier used in directory path and manifest (default: linear_poisson_v1_recommended).",
    )
    parser.add_argument(
        "--base_seed", type=int, default=20260512,
        help="Base RNG seed (default: 20260512).",
    )
    args = parser.parse_args()

    n_datasets = args.n_datasets
    out_dir = args.out_dir
    suite_id = args.suite_id
    base_seed = args.base_seed

    primary_split_seeds = [0, 1, 2, 3, 4]
    regimes = ["A", "B", "C", "D"]
    per_regime = n_datasets // len(regimes)

    stage_prefix = "@EVAL_DATASET_STAGE"

    primary_dir = os.path.join(out_dir, "synthetic_regression_prepared", suite_id, "primary")
    os.makedirs(primary_dir, exist_ok=True)

    rng = np.random.default_rng(seed=base_seed)

    print(f"Generating {n_datasets} datasets ({per_regime} per regime) -> {primary_dir}")
    print(f"  suite_id={suite_id}  base_seed={base_seed}  split_seeds={primary_split_seeds}")

    dataset_records = []
    global_idx = 0

    for regime in regimes:
        for _ in range(per_regime):
            ds = _generate_dataset(rng, regime)
            dataset_seed = int(rng.integers(0, 2**31))

            filename = f"dataset_{global_idx:04d}.parquet"
            filepath = os.path.join(primary_dir, filename)
            _write_parquet(ds, filepath)
            payload_bytes = os.path.getsize(filepath)

            n_total = ds["n_total"]
            n_holdout = n_total // 5
            n_train = n_total - n_holdout

            stage_path = f"{stage_prefix}/primary/{filename}"

            dataset_records.append({
                "dataset_id":          global_idx,
                "dataset_seed":        dataset_seed,
                "suite_family":        "primary",
                "stage_path":          stage_path,
                "prior_regime":        regime,
                "split_seeds":         primary_split_seeds,
                "n_total":             n_total,
                "n_train_default":     n_train,
                "n_holdout_default":   n_holdout,
                "p_signal":            ds["p_signal"],
                "p_noise":             0,
                "p_total":             ds["p_signal"],
                "target_noise_scale":  1.0,
                "training_size_anchor": False,
                "feature_noise_level": 0,
                "payload_bytes":       payload_bytes,
            })

            global_idx += 1
            if global_idx % 50 == 0:
                print(f"  [{global_idx:4d}/{n_datasets}] generated dataset_{global_idx - 1:04d}.parquet")

    stage_paths = [r["stage_path"] for r in dataset_records]

    manifest = {
        "suite_id":                     suite_id,
        "base_seed":                    base_seed,
        "generated_locally":            True,
        "format":                       "parquet",
        "n_datasets_primary":           n_datasets,
        "n_datasets_feature_noise":     0,
        "n_datasets_training_size":     0,
        "n_datasets_target_noise":      0,
        "primary_split_seeds":          primary_split_seeds,
        "stage_paths":                  stage_paths,
        "datasets":                     dataset_records,
    }

    manifest_dir = os.path.join(out_dir, "synthetic_regression_prepared", suite_id)
    manifest_path = os.path.join(manifest_dir, "synthetic_regression_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {global_idx} parquet files written to:")
    print(f"  {primary_dir}")
    print(f"Manifest written to:")
    print(f"  {manifest_path}")

    abs_primary = Path(primary_dir).resolve().as_posix()
    abs_manifest = Path(manifest_path).resolve().as_posix()

    print(
        "\n--- SnowSQL PUT commands ---\n"
        "REMOVE @EVAL_DATASET_STAGE/primary/;\n"
        f"PUT file://{abs_primary}/*.parquet\n"
        "    @EVAL_DATASET_STAGE/primary/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_manifest}\n"
        "    @EVAL_DATASET_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    )


if __name__ == "__main__":
    main()
