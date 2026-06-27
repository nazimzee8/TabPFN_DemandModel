#!/usr/bin/env python
"""Regenerate numeric-only linear training data for both task families.

Drives ``src/data_generation/generate_dgp.py`` with the canonical flags for:
  - linear_regression     → data/linear_regression/numeric/{train,val,test}
  - linear_classification → data/linear_classification/numeric/{train,val,test}

Usage
-----
# Both families (default):
python scripts/run_generate_linear_numeric.py

# One family only:
python scripts/run_generate_linear_numeric.py --only regression
python scripts/run_generate_linear_numeric.py --only classification

Notes
-----
- Regression overwrites in place; any pre-existing eval/ subdir is left untouched.
- Classification requires an empty --out_dir, so data/linear_classification is
  deleted entirely before generation when it already exists.
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
    """Regenerate numeric linear regression training data (purge numeric/ first)."""
    # Purge the numeric split dirs so no stale grid-era parquet files from a
    # prior generation round can mix with the new cursor-era output.
    # eval/ is left untouched (pre-existing eval datasets are kept by design).
    numeric_dir = REPO / "data" / "linear_regression" / "numeric"
    if numeric_dir.exists():
        print(f"Removing existing {numeric_dir} before regeneration...")
        shutil.rmtree(numeric_dir)
    _run([
        "--task_family", "linear_regression",
        "--n_datasets", "1000",
        "--out_dir", "data/linear_regression",
        "--profile", "linear_stat_aware",
        "--base_seed", "42",
        "--store_beta",
        "--store_teacher_preds",
    ])


def classification() -> None:
    """Regenerate numeric linear classification training data (purge numeric/ first).

    Only the numeric/ subdir is removed so the mixed/ subdir (produced by
    run_generate_linear_mixed.py) is left intact and both runners can safely
    execute in parallel without clobbering each other.
    """
    numeric_dir = REPO / "data" / "linear_classification" / "numeric"
    if numeric_dir.exists():
        print(f"Removing existing {numeric_dir} before regeneration...")
        shutil.rmtree(numeric_dir)
    _run([
        "--task_family", "linear_classification",
        "--n_datasets", "1000",
        "--out_dir", "data/linear_classification",
        "--profile", "linear_classification_stat_aware",
        "--base_seed", "42",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate numeric-only linear regression + classification training data.",
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
