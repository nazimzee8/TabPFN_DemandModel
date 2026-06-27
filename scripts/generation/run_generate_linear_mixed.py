#!/usr/bin/env python
"""Regenerate mixed-categorical linear training data for both task families.

Drives ``src/data_generation/generate_dgp.py`` with the canonical flags for:
  - linear_regression_mixed_categorical     → data/linear_regression/mixed/{train,val,test}
  - linear_classification_mixed_categorical → data/linear_classification/mixed/{train,val,test}

Usage
-----
# Both families (default):
python scripts/generation/run_generate_linear_mixed.py

# One family only:
python scripts/generation/run_generate_linear_mixed.py --only regression
python scripts/generation/run_generate_linear_mixed.py --only classification

Notes
-----
- Only the mixed/ subdir is cleaned before each run; the coexisting numeric/ subdir
  (produced by run_generate_linear_numeric.py) is left intact.
- n_datasets=1000 → 800/100/100 train/val/test split (80/10/10, hard-coded in
  generate_dgp.py).

See CLAUDE.md section 10 for DGP documentation.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "src" / "data_generation" / "generate_dgp.py"


def _run(args: list[str]) -> None:
    cmd = [sys.executable, str(GEN), *args]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def regression() -> None:
    """Regenerate mixed-categorical linear regression training data (clean mixed/ first)."""
    mixed = REPO / "data" / "linear_regression" / "mixed"
    if mixed.exists():
        print(f"Removing existing {mixed} before regeneration...")
        shutil.rmtree(mixed)
    _run([
        "--task_family", "linear_regression_mixed_categorical",
        "--n_datasets", "1000",
        "--out_dir", "data/linear_regression",
        "--profile", "linear_regression_mixed_categorical_stat_aware",
        "--base_seed", "42",
    ])


def classification() -> None:
    """Regenerate mixed-categorical linear classification training data (clean mixed/ first)."""
    mixed = REPO / "data" / "linear_classification" / "mixed"
    if mixed.exists():
        print(f"Removing existing {mixed} before regeneration...")
        shutil.rmtree(mixed)
    _run([
        "--task_family", "linear_classification_mixed_categorical",
        "--n_datasets", "1000",
        "--out_dir", "data/linear_classification",
        "--profile", "linear_classification_mixed_categorical_stat_aware",
        "--base_seed", "42",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate mixed-categorical linear regression + classification training data.",
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
