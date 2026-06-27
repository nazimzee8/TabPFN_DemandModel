#!/usr/bin/env python
"""Regenerate numeric-only nonlinear training data for both task families.

Drives ``src/data_generation/generate_nonlinear_dgp.py`` with the canonical flags for:
  - synthetic_nonlinear_regression     → data/nonlinear_regression/numeric/{train,val,test}
  - synthetic_nonlinear_classification → data/nonlinear_classification/numeric/{train,val,test}

Usage
-----
# Both families (default):
python scripts/generation/run_generate_nonlinear_numeric.py

# One family only:
python scripts/generation/run_generate_nonlinear_numeric.py --only regression
python scripts/generation/run_generate_nonlinear_numeric.py --only classification

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
    """Regenerate numeric nonlinear regression training data (clean out_dir first)."""
    out = REPO / "data" / "nonlinear_regression" / "numeric"
    if out.exists():
        print(f"Removing existing {out} before regeneration...")
        shutil.rmtree(out)
    _run([
        "--n_datasets", "1000",
        "--out_dir", "data/nonlinear_regression/numeric",
        "--seed", "42",
    ])


def classification() -> None:
    """Regenerate numeric nonlinear classification training data (clean out_dir first)."""
    out = REPO / "data" / "nonlinear_classification" / "numeric"
    if out.exists():
        print(f"Removing existing {out} before regeneration...")
        shutil.rmtree(out)
    _run([
        "--n_datasets", "1000",
        "--out_dir", "data/nonlinear_classification/numeric",
        "--seed", "42",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate numeric-only nonlinear regression + classification training data.",
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
