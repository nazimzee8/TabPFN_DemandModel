"""
generate_nonlinear.py
=====================
Local CLI script to generate nonlinear synthetic regression datasets for
DeepSet evaluation.

Mirrors scripts/ood_regression/generate_ood_eval_data.py in structure.

Usage
-----
    python scripts/generate_nonlinear.py \\
        --n-datasets 400 \\
        --out-dir data/nonlinear_regression/ \\
        --seed 20260601

Output
------
    data/nonlinear_regression/{regime}/dataset_{idx:04d}.parquet   (400 files total)
    data/nonlinear_regression/nonlinear_manifest.json

After running locally, stage to Snowflake (PUT commands are printed at the end).

Nonlinear Regimes
-----------------
I — Quadratic:            betaX = (X**2) @ beta_sq
J — Sinusoidal:           betaX = sin(2π * (X @ beta_norm)) * sqrt(p)
K — Pairwise Interactions: betaX = sum of c_k * x_i * x_j
L — ReLU / Threshold:     betaX = sum(relu(x_i * beta_i)) - mean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NONLINEAR_REGIMES     = ["I", "J", "K", "L"]
N_DATASETS_DEFAULT    = 400          # 100 per regime
BASE_SEED_DEFAULT     = 20260601
OUT_DIR_DEFAULT       = "data/nonlinear_regression"
SPLIT_SEEDS           = [0, 1, 2]
POISSON_N_MU          = 200
POISSON_P_MU          = 10


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_n_p(rng: np.random.Generator):
    """Rejection-sample (n_total, p_signal) with same constraints as training DGP.

    p ~ Poisson(10), reject until p >= 1.
    n ~ Poisson(200), reject until n >= 5 and n >= 5 * p.
    """
    while True:
        n = int(rng.poisson(POISSON_N_MU))
        p = int(rng.poisson(POISSON_P_MU))
        if p >= 1 and n >= 5 and n >= 5 * p:
            return n, p


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _generate_dataset(regime: str, dataset_seed: int) -> dict:
    """Generate one nonlinear evaluation dataset.

    Uses a deterministic per-dataset RNG seeded from dataset_seed.
    Returns a dict matching the evaluation parquet schema.
    """
    rng = np.random.default_rng(dataset_seed)
    n_total, p_signal = _sample_n_p(rng)
    X = rng.standard_normal((n_total, p_signal))

    if regime == "I":
        # Quadratic: each feature contributes x_i^2 * beta_i
        beta_sq = rng.standard_normal(p_signal)
        betaX = (X ** 2) @ beta_sq

    elif regime == "J":
        # Sinusoidal: nonlinear projection through sin
        beta = rng.standard_normal(p_signal)
        beta /= (np.linalg.norm(beta) / np.sqrt(p_signal))
        betaX = np.sin(2 * np.pi * (X @ beta)) * np.sqrt(p_signal)

    elif regime == "K":
        # Pairwise multiplicative interactions
        n_pairs = max(1, p_signal // 2)
        pairs = rng.choice(p_signal, size=(n_pairs, 2), replace=False)
        c = rng.standard_normal(n_pairs)
        betaX = np.zeros(n_total)
        for k, (i, j) in enumerate(pairs):
            betaX += c[k] * X[:, i] * X[:, j]

    elif regime == "L":
        # ReLU / threshold activation, centred
        beta = rng.standard_normal(p_signal)
        betaX = np.sum(np.maximum(0.0, X * beta[np.newaxis, :]), axis=1)
        betaX -= betaX.mean()

    else:
        raise ValueError(f"Unknown nonlinear regime: {regime!r}")

    y = betaX + rng.standard_normal(n_total)

    return {
        "X":                    X.astype(np.float64),
        "y":                    y.astype(np.float64),
        "betaX":                betaX.astype(np.float64),
        "suite_family":         "primary",
        "prior_regime":         regime,
        "n_total":              n_total,
        "p_signal":             p_signal,
        "p_noise":              0,
        "p_total":              p_signal,
        "target_noise_scale":   1.0,
        "training_size_anchor": False,
        "feature_noise_level":  0,
    }


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def _write_parquet(ds: dict, filepath: Path) -> None:
    """Write dataset dict to a parquet file with the evaluation schema."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "X":                    pa.array([ds["X"].tolist()],     type=pa.list_(pa.list_(pa.float64()))),
        "y":                    pa.array([ds["y"].tolist()],     type=pa.list_(pa.float64())),
        "betaX":                pa.array([ds["betaX"].tolist()], type=pa.list_(pa.float64())),
        "suite_family":         pa.array([ds["suite_family"]],   type=pa.utf8()),
        "prior_regime":         pa.array([ds["prior_regime"]],   type=pa.utf8()),
        "n_total":              pa.array([ds["n_total"]],        type=pa.int64()),
        "p_signal":             pa.array([ds["p_signal"]],       type=pa.int64()),
        "p_noise":              pa.array([ds["p_noise"]],        type=pa.int64()),
        "p_total":              pa.array([ds["p_total"]],        type=pa.int64()),
        "target_noise_scale":   pa.array([ds["target_noise_scale"]], type=pa.float64()),
        "training_size_anchor": pa.array([ds["training_size_anchor"]], type=pa.bool_()),
        "feature_noise_level":  pa.array([ds["feature_noise_level"]], type=pa.int64()),
    })
    pq.write_table(table, str(filepath))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nonlinear synthetic regression datasets (regimes I/J/K/L)."
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=N_DATASETS_DEFAULT,
        help=f"Total number of datasets to generate (default: {N_DATASETS_DEFAULT}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=OUT_DIR_DEFAULT,
        help=f"Output directory (default: {OUT_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED_DEFAULT,
        help=f"Global base RNG seed (default: {BASE_SEED_DEFAULT}).",
    )
    args = parser.parse_args()

    n_datasets: int = args.n_datasets
    out_dir = Path(args.out_dir)
    base_seed: int = args.seed

    if n_datasets % len(NONLINEAR_REGIMES) != 0:
        print(
            f"[WARN] --n-datasets={n_datasets} is not divisible by {len(NONLINEAR_REGIMES)}. "
            "Some regimes will have one extra dataset.",
            file=sys.stderr,
        )

    per_regime = n_datasets // len(NONLINEAR_REGIMES)
    remainder  = n_datasets % len(NONLINEAR_REGIMES)

    manifest_datasets: list[dict] = []
    global_idx = 0

    for r_idx, regime in enumerate(NONLINEAR_REGIMES):
        regime_count = per_regime + (1 if r_idx < remainder else 0)
        regime_dir = out_dir / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        for ds_idx in range(regime_count):
            # Derive deterministic per-dataset seed from base seed + regime index + dataset index
            dataset_seed = int(
                np.random.default_rng([base_seed, r_idx, ds_idx]).integers(2**63)
            )
            ds = _generate_dataset(regime, dataset_seed)
            filename = f"{regime}/dataset_{ds_idx:04d}.parquet"
            outpath = out_dir / filename
            _write_parquet(ds, outpath)

            manifest_datasets.append({
                "regime":             regime,
                "dataset_id":         ds_idx,
                "filename":           filename,
                "n_total":            int(ds["n_total"]),
                "p_signal":           int(ds["p_signal"]),
                "p_noise":            int(ds["p_noise"]),
                "p_total":            int(ds["p_total"]),
                "target_noise_scale": float(ds["target_noise_scale"]),
                "dataset_seed":       dataset_seed,
            })
            global_idx += 1

        print(
            f"[INFO] Regime {regime}: {regime_count} datasets written to {regime_dir}/",
            flush=True,
        )

    # Write manifest
    manifest = {
        "suite_id":   "nonlinear_v1",
        "base_seed":  base_seed,
        "n_datasets": global_idx,
        "regimes":    NONLINEAR_REGIMES,
        "split_seeds": SPLIT_SEEDS,
        "datasets":   manifest_datasets,
    }
    manifest_path = out_dir / "nonlinear_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[INFO] Manifest written: {manifest_path} ({global_idx} datasets)", flush=True)

    # Print PUT commands for Snowflake staging
    abs_I = (out_dir / "I").resolve().as_posix()
    abs_J = (out_dir / "J").resolve().as_posix()
    abs_K = (out_dir / "K").resolve().as_posix()
    abs_L = (out_dir / "L").resolve().as_posix()
    abs_manifest = manifest_path.resolve().as_posix()
    print(
        "\n--- SnowSQL PUT commands ---\n"
        f"PUT file://{abs_I}/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/I/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_J}/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/J/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_K}/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/K/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_L}/*.parquet @EVALUATION_DATASET_STAGE/nonlinear/L/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_manifest}    @EVALUATION_DATASET_STAGE/nonlinear/   AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    )


if __name__ == "__main__":
    main()
