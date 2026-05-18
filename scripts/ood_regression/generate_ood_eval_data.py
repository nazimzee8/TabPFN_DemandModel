"""
generate_ood_eval_data.py
=========================
Local CLI script to generate out-of-distribution (OOD) synthetic regression
datasets for DeepSet evaluation.

Follows generate_dgp.py pattern exactly: standalone, no project imports.

Usage
-----
    python scripts/ood_regression/generate_ood_eval_data.py \\
        --n_datasets 200 \\
        --out_dir data/ood_regression/ \\
        --seed 20260513

Output
------
    data/ood_regression/{regime}/dataset_{idx:04d}.parquet   (200 files total)
    data/ood_regression/ood_manifest.json

After running locally, stage to Snowflake (PUT commands are printed at the end of the script).

OOD Regimes
-----------
E — Laplace(0, 1/√2) features (unit var, heavy tails), dense N(0,1) β, N(0,1) noise
F — Uniform(−√3, √3) features (unit var, bounded), N(0,4) β 95 % sparse, N(0,1) noise
G — N(0,I) features (same as training), dense N(0,1) β, Cauchy(0,1) noise
H — Block-diagonal Σ (within-block ρ=0.7, block_size=3), dense N(0,1) β, N(0,1) noise
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

OOD_REGIMES = ["E", "F", "G", "H"]


def sample_params_ood(rng: np.random.Generator):
    """Rejection-sample (n, p) with same constraints as training DGP.

    n ~ Poisson(200), p ~ Poisson(10), p >= 1, n >= 5, n >= 5 * p.
    """
    while True:
        n = int(rng.poisson(200))
        p = int(rng.poisson(10))
        if p >= 1 and n >= 5 and n >= 5 * p:
            return n, p


# ---------------------------------------------------------------------------
# Feature generators
# ---------------------------------------------------------------------------

def _generate_X_E(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Laplace(0, 1/√2) — unit variance, heavy tails."""
    return rng.laplace(0.0, 1.0 / math.sqrt(2), (n, p))


def _generate_X_F(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Uniform(−√3, √3) — unit variance, bounded support."""
    return rng.uniform(-math.sqrt(3), math.sqrt(3), (n, p))


def _generate_X_G(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """N(0, I) — same as training regimes A/C."""
    return rng.standard_normal((n, p))


def _generate_X_H(
    rng: np.random.Generator, n: int, p: int, block_size: int = 3, rho: float = 0.7
) -> np.ndarray:
    """Block-diagonal Σ: within-block ρ=0.7, across-block independent.

    Each block is drawn via Cholesky decomposition of the compound-symmetry
    covariance matrix Σ_k = ρ·1·1ᵀ + (1−ρ)·I.
    """
    X = np.zeros((n, p))
    for start in range(0, p, block_size):
        end = min(start + block_size, p)
        k = end - start
        Sigma = np.full((k, k), rho) + np.eye(k) * (1.0 - rho)
        L = np.linalg.cholesky(Sigma)
        X[:, start:end] = rng.standard_normal((n, k)) @ L.T
    return X


def _generate_features(rng: np.random.Generator, n: int, p: int, regime: str) -> np.ndarray:
    if regime == "E":
        return _generate_X_E(rng, n, p)
    if regime == "F":
        return _generate_X_F(rng, n, p)
    if regime == "G":
        return _generate_X_G(rng, n, p)
    if regime == "H":
        return _generate_X_H(rng, n, p)
    raise ValueError(f"Unknown OOD regime: {regime!r}")


# ---------------------------------------------------------------------------
# Coefficient and noise generators
# ---------------------------------------------------------------------------

def _generate_beta(rng: np.random.Generator, p: int, regime: str) -> np.ndarray:
    if regime == "F":
        # N(0, 4) coefficients with 95 % sparsity
        beta = rng.normal(0.0, 2.0, p)   # std=2 → var=4
        beta[rng.random(p) < 0.95] = 0.0
    else:
        beta = rng.standard_normal(p)
    return beta


def _generate_noise(rng: np.random.Generator, n: int, regime: str) -> np.ndarray:
    if regime == "G":
        return rng.standard_cauchy(n)
    return rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_ood_dataset(
    rng: np.random.Generator, regime: str
) -> dict:
    """Generate one OOD dataset.

    Returns a dict with arrays and scalar metadata matching the
    serialize_synthetic_npz() format used by the eval pipeline.
    """
    n, p = sample_params_ood(rng)
    X = _generate_features(rng, n, p, regime)
    beta = _generate_beta(rng, p, regime)
    betaX = X @ beta
    eps = _generate_noise(rng, n, regime)
    y = betaX + eps
    return {
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "suite_family": "ood_primary",
        "prior_regime": regime,
        "n_total": n,
        "p_signal": p,
        "p_noise": 0,
        "p_total": p,
        "target_noise_scale": 1.0,
    }


def _write_parquet(ds: dict, filepath: Path) -> None:
    """Write dataset dict to a parquet file using the same schema as generate_synthetic_regression.py."""
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
        "training_size_anchor": pa.array([False],                type=pa.bool_()),
        "feature_noise_level":  pa.array([0],                   type=pa.int64()),
    })
    pq.write_table(table, str(filepath))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OOD synthetic regression datasets (regimes E/F/G/H)."
    )
    parser.add_argument(
        "--n_datasets",
        type=int,
        default=200,
        help="Total number of datasets to generate (evenly split across regimes).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/ood_regression/",
        help="Output directory. Parquet files go to <out_dir>/{regime}/dataset_XXXX.parquet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260513,
        help="Global base RNG seed for reproducibility.",
    )
    args = parser.parse_args()

    n_datasets: int = args.n_datasets
    out_dir = Path(args.out_dir)
    base_seed: int = args.seed

    if n_datasets % len(OOD_REGIMES) != 0:
        print(
            f"[WARN] --n_datasets={n_datasets} is not divisible by {len(OOD_REGIMES)}. "
            "Some regimes will have one extra dataset.",
            file=sys.stderr,
        )

    per_regime = n_datasets // len(OOD_REGIMES)
    remainder = n_datasets % len(OOD_REGIMES)

    manifest_datasets: list[dict] = []
    global_idx = 0

    for r_idx, regime in enumerate(OOD_REGIMES):
        regime_count = per_regime + (1 if r_idx < remainder else 0)
        regime_dir = out_dir / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        for ds_idx in range(regime_count):
            # Derive a deterministic per-dataset seed from the global seed.
            dataset_seed = int(
                np.random.default_rng(base_seed + global_idx).integers(0, 2**31)
            )
            rng = np.random.default_rng(dataset_seed)

            ds = generate_ood_dataset(rng, regime)
            filename = f"{regime}/dataset_{ds_idx:04d}.parquet"
            outpath = out_dir / filename
            _write_parquet(ds, outpath)

            manifest_datasets.append(
                {
                    "regime": regime,
                    "dataset_id": ds_idx,
                    "filename": filename,
                    "n_total": int(ds["n_total"]),
                    "p_signal": int(ds["p_signal"]),
                    "p_noise": int(ds["p_noise"]),
                    "p_total": int(ds["p_total"]),
                    "target_noise_scale": float(ds["target_noise_scale"]),
                    "dataset_seed": dataset_seed,
                }
            )
            global_idx += 1

        print(
            f"[INFO] Regime {regime}: {regime_count} datasets written to {regime_dir}/",
            flush=True,
        )

    manifest = {
        "ood_version": "v1",
        "base_seed": base_seed,
        "n_datasets": global_idx,
        "regimes": OOD_REGIMES,
        "datasets": manifest_datasets,
    }
    manifest_path = out_dir / "ood_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[INFO] Manifest written: {manifest_path} ({global_idx} datasets)", flush=True)

    abs_E = (out_dir / "E").resolve().as_posix()
    abs_F = (out_dir / "F").resolve().as_posix()
    abs_G = (out_dir / "G").resolve().as_posix()
    abs_H = (out_dir / "H").resolve().as_posix()
    abs_manifest = manifest_path.resolve().as_posix()
    print(
        "\n--- SnowSQL PUT commands ---\n"
        f"PUT file://{abs_E}/*.parquet @EVAL_DATASET_STAGE/ood_parity/E/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_F}/*.parquet @EVAL_DATASET_STAGE/ood_parity/F/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_G}/*.parquet @EVAL_DATASET_STAGE/ood_parity/G/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_H}/*.parquet @EVAL_DATASET_STAGE/ood_parity/H/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT file://{abs_manifest}   @EVAL_DATASET_STAGE/ood_parity/   AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    )


if __name__ == "__main__":
    main()
