#!/usr/bin/env python
"""Regenerate mixed-categorical nonlinear training data for both task families.

Drives ``src/data_generation/generate_nonlinear_dgp.py`` with the canonical flags for:
  - synthetic_nonlinear_regression_mixed_categorical     → data/nonlinear_regression/mixed/{train,val,test}
  - synthetic_nonlinear_classification_mixed_categorical → data/nonlinear_classification/mixed/{train,val,test}

Usage
-----
# Both families (default):
python scripts/generation/run_generate_nonlinear_mixed.py

# One family only:
python scripts/generation/run_generate_nonlinear_mixed.py --only regression
python scripts/generation/run_generate_nonlinear_mixed.py --only classification

Notes
-----
- Each out_dir is cleaned before regeneration so stale parquet from a prior run
  cannot linger.
- n_datasets=1000 → 800/100/100 train/val/test split (80/10/10, hard-coded in
  generate_nonlinear_dgp.py).
- The nonlinear generator uses --seed (not --base_seed) and has no --profile flag.

See CLAUDE.md section 10 for DGP documentation.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "src" / "data_generation" / "generate_nonlinear_dgp.py"


def _run(args: list[str]) -> None:
    cmd = [sys.executable, str(GEN), *args]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def regression() -> None:
    """Regenerate mixed-categorical nonlinear regression training data (clean out_dir first)."""
    out = REPO / "data" / "nonlinear_regression" / "mixed"
    if out.exists():
        print(f"Removing existing {out} before regeneration...")
        shutil.rmtree(out)
    _run([
        "--task_family", "synthetic_nonlinear_regression_mixed_categorical",
        "--n_datasets", "1000",
        "--out_dir", "data/nonlinear_regression/mixed",
        "--seed", "42",
    ])


def classification() -> None:
    """Regenerate mixed-categorical nonlinear classification training data (clean out_dir first)."""
    out = REPO / "data" / "nonlinear_classification" / "mixed"
    if out.exists():
        print(f"Removing existing {out} before regeneration...")
        shutil.rmtree(out)
    _run([
        "--task_family", "synthetic_nonlinear_classification_mixed_categorical",
        "--n_datasets", "1000",
        "--out_dir", "data/nonlinear_classification/mixed",
        "--seed", "42",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate mixed-categorical nonlinear regression + classification training data.",
    )
    ap.add_argument(
        "--only",
        choices=["regression", "classification"],
        default=None,
        help="Run only one task family. Omit to run both.",
    )
    args = ap.parse_args()

    if args.only in (None, "regression"):
        regression()
    if args.only in (None, "classification"):
        classification()


if __name__ == "__main__":
    main()
