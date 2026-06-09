"""
generate_nonlinear.py
=====================
Local CLI script to generate nonlinear synthetic regression evaluation datasets.

Supports both v2 (backward-compat) and v3 DGP suites, and both pure numeric
and mixed-categorical variants (controlled by ``--mixed-categorical``).

Usage
-----
    # v2 backward-compat (unchanged):
    python scripts/generate_nonlinear.py --n-datasets 420 --out-dir data/nonlinear_v2/

    # v3 suite (17 families, ~630 datasets):
    python scripts/generate_nonlinear.py --v3 --n-datasets 630 --out-dir data/nonlinear_v3/

    # v3 mixed-categorical regression:
    python scripts/generate_nonlinear.py --v3 --mixed-categorical --n-datasets 100

    # Smoke test:
    python scripts/generate_nonlinear.py --v3 --n-datasets 5 --dry-run

    # Generate core component only:
    python scripts/generate_nonlinear.py --v3 --suite-component core

Output
------
    v2:        data/nonlinear_v2/{family}/dataset_{idx:05d}.parquet
    v3:        data/nonlinear_v3/{family}/dataset_{idx:05d}.parquet
    v3 mixed:  data/nonlinear_mixed_regression_v1/{family}/dataset_{idx:05d}.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from generate_nonlinear_dgp import (  # noqa: E402
    # v2 backward-compat
    enumerate_suite_cells,
    generate_v2_dataset,
    write_parquet_eval,
    LearnabilityGateError,
    V2_TARGET_FAMILIES,
    V2_SPLIT_SEEDS,
    BASE_SEED,
    # v3 extensions
    V3_TARGET_FAMILIES,
    enumerate_regression_suite_cells_v3,
    generate_v3_regression_dataset,
    generate_v3_mixed_regression_dataset,
    write_parquet_nonlinear_regression_eval,
)


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------

def _preflight_check(cells: list[dict]) -> None:
    """Check for duplicate logical_dataset_key and condition_id. Exit 1 on error."""
    keys = [c["logical_dataset_key"] for c in cells]
    cond_ids = [c["condition_id"] for c in cells]

    dup_keys = [k for k in set(keys) if keys.count(k) > 1]
    dup_conds = [k for k in set(cond_ids) if cond_ids.count(k) > 1]

    errors = []
    if dup_keys:
        errors.append(f"Duplicate logical_dataset_key: {dup_keys[:5]}")
    if dup_conds:
        errors.append(f"Duplicate condition_id: {dup_conds[:5]}")

    if errors:
        for msg in errors:
            print(f"[ERROR] Preflight failed: {msg}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Preflight OK: {len(cells)} cells, no duplicates.", flush=True)


# ---------------------------------------------------------------------------
# v2 dataset generation (backward compat)
# ---------------------------------------------------------------------------

def _generate_and_write_v2(
    cell: dict,
    idx: int,
    out_dir: Path,
) -> dict | None:
    """Generate one v2 dataset and write it. Returns manifest entry or None."""
    target_family = cell["target_family"]
    family_dir = out_dir / target_family
    family_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{target_family}/dataset_{idx:05d}.parquet"
    outpath = out_dir / filename

    extra_kw = {k: v for k, v in cell.items() if k not in (
        "target_family", "feature_regime", "n", "p_signal",
        "target_noise_scale", "target_noise_type", "suite_component",
        "teacher_seed", "sample_seed", "logical_dataset_key", "condition_id",
    )}

    try:
        ds = generate_v2_dataset(
            target_family=cell["target_family"],
            feature_regime=cell["feature_regime"],
            n=cell["n"],
            p_signal=cell["p_signal"],
            target_noise_scale=cell["target_noise_scale"],
            target_noise_type=cell["target_noise_type"],
            suite_component=cell["suite_component"],
            teacher_seed=cell["teacher_seed"],
            sample_seed=cell["sample_seed"],
            **extra_kw,
        )
    except LearnabilityGateError as exc:
        print(
            f"[WARN] LearnabilityGateError for {cell['logical_dataset_key']!r}: {exc}",
            file=sys.stderr,
        )
        return None

    write_parquet_eval(ds, str(outpath))

    return {
        "logical_dataset_key": cell["logical_dataset_key"],
        "condition_id":        cell["condition_id"],
        "target_family":       cell["target_family"],
        "feature_regime":      cell["feature_regime"],
        "suite_component":     cell["suite_component"],
        "dataset_idx":         idx,
        "filename":            filename,
        "n_total":             int(ds["n_total"]),
        "p_signal":            int(ds["p_signal"]),
        "p_noise":             int(ds["p_noise"]),
        "p_total":             int(ds["p_total"]),
        "target_noise_scale":  float(ds["target_noise_scale"]),
        "target_noise_type":   ds["target_noise_type"],
        "feature_noise_sigma": float(ds["feature_noise_sigma"]),
        "normalization_constant": float(ds["normalization_constant"]),
        "teacher_seed":        int(cell["teacher_seed"]),
        "sample_seed":         int(cell["sample_seed"]),
        "snr_target":          float(ds["snr_target"]),
        "rho":                 float(ds["rho"]),
        "active_fraction":     float(ds["active_fraction"]),
        "covariance_type":     ds["covariance_type"],
        "noise_feature_fraction": float(ds.get("noise_feature_fraction", 0.0)),
    }


# ---------------------------------------------------------------------------
# v3 dataset generation
# ---------------------------------------------------------------------------

def _generate_and_write_v3(
    cell: dict,
    idx: int,
    out_dir: Path,
    *,
    is_mixed_categorical: bool = False,
) -> dict | None:
    """Generate one v3 regression dataset and write it. Returns manifest entry or None."""
    target_family = cell["target_family"]
    family_dir = out_dir / target_family
    family_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{target_family}/dataset_{idx:05d}.parquet"
    outpath = out_dir / filename

    extra_kw = {k: v for k, v in cell.items() if k not in (
        "target_family", "feature_regime", "n", "p_signal",
        "target_noise_scale", "target_noise_type", "suite_component",
        "teacher_seed", "sample_seed", "logical_dataset_key", "condition_id",
        "is_mixed_categorical",
    )}

    try:
        if is_mixed_categorical:
            ds = generate_v3_mixed_regression_dataset(
                target_family=cell["target_family"],
                feature_regime=cell["feature_regime"],
                n=cell["n"],
                p_signal=cell["p_signal"],
                target_noise_scale=cell["target_noise_scale"],
                target_noise_type=cell["target_noise_type"],
                suite_component=cell["suite_component"],
                teacher_seed=cell["teacher_seed"],
                sample_seed=cell["sample_seed"],
                **extra_kw,
            )
        else:
            ds = generate_v3_regression_dataset(
                target_family=cell["target_family"],
                feature_regime=cell["feature_regime"],
                n=cell["n"],
                p_signal=cell["p_signal"],
                target_noise_scale=cell["target_noise_scale"],
                target_noise_type=cell["target_noise_type"],
                suite_component=cell["suite_component"],
                teacher_seed=cell["teacher_seed"],
                sample_seed=cell["sample_seed"],
                **extra_kw,
            )
    except LearnabilityGateError as exc:
        print(
            f"[WARN] LearnabilityGateError for {cell['logical_dataset_key']!r}: {exc}",
            file=sys.stderr,
        )
        return None

    write_parquet_nonlinear_regression_eval(
        ds, str(outpath), is_mixed_categorical=is_mixed_categorical
    )

    entry: dict[str, Any] = {
        "logical_dataset_key": cell["logical_dataset_key"],
        "condition_id":        cell["condition_id"],
        "target_family":       cell["target_family"],
        "feature_regime":      cell["feature_regime"],
        "suite_component":     cell["suite_component"],
        "dataset_idx":         idx,
        "filename":            filename,
        "n_total":             int(ds["n_total"]),
        "p_signal":            int(ds["p_signal"]),
        "p_noise":             int(ds["p_noise"]),
        "p_total":             int(ds["p_total"]),
        "target_noise_scale":  float(ds["target_noise_scale"]),
        "target_noise_type":   ds["target_noise_type"],
        "feature_noise_sigma": float(ds["feature_noise_sigma"]),
        "normalization_constant": float(ds["normalization_constant"]),
        "teacher_seed":        int(cell["teacher_seed"]),
        "sample_seed":         int(cell["sample_seed"]),
        "snr_target":          float(ds["snr_target"]),
        "rho":                 float(ds["rho"]),
        "active_fraction":     float(ds["active_fraction"]),
        "covariance_type":     ds["covariance_type"],
        "nonlinear_family":    ds.get("nonlinear_family", ""),
        "parametric_model_class": ds.get("parametric_model_class", ""),
        "task_family":         ds.get("task_family", ""),
        "schema_version":      ds.get("schema_version", ""),
    }
    if is_mixed_categorical:
        entry["p_cat"] = int(ds.get("p_cat", 0))
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nonlinear synthetic regression evaluation datasets."
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=None,
        help="Total datasets to generate. Default: 420 (v2) or full suite (v3).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults depend on --v3 and --mixed-categorical.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help=f"Base RNG seed (default: {BASE_SEED}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print cell specs and exit without writing any files.",
    )
    parser.add_argument(
        "--suite-component",
        choices=["core", "robustness", "ood", "all"],
        default="all",
        help="Generate only a subset of suite components (default: all).",
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        default=False,
        help="Use v3 DGP (17 families, extended noise/features). Default: v2.",
    )
    parser.add_argument(
        "--mixed-categorical",
        action="store_true",
        default=False,
        help="Generate mixed-categorical variant (implies --v3).",
    )
    args = parser.parse_args()

    # --mixed-categorical implies --v3
    if args.mixed_categorical:
        args.v3 = True

    # Resolve defaults
    if args.v3:
        if args.mixed_categorical:
            default_out = "data/nonlinear_mixed_regression_v1"
            default_n = 100
            manifest_filename = "nonlinear_mixed_regression_v1_manifest.json"
            suite_id = "nonlinear_mixed_regression"
            stage_prefix = "@EVALUATION_DATASET_STAGE/nonlinear_mixed_regression_v1"
            prior_version = "mixed_v1"
            target_families_list = V3_TARGET_FAMILIES
        else:
            default_out = "data/nonlinear_v3"
            default_n = 630
            manifest_filename = "nonlinear_v3_manifest.json"
            suite_id = "nonlinear_v3"
            stage_prefix = "@EVALUATION_DATASET_STAGE/nonlinear_v3"
            prior_version = "v3"
            target_families_list = V3_TARGET_FAMILIES
    else:
        default_out = "data/nonlinear_v2"
        default_n = 420
        manifest_filename = "nonlinear_v2_manifest.json"
        suite_id = "nonlinear"
        stage_prefix = "@EVALUATION_DATASET_STAGE/nonlinear_v2"
        prior_version = "v2"
        target_families_list = list(V2_TARGET_FAMILIES)

    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out)
    n_datasets_limit = args.n_datasets if args.n_datasets is not None else default_n

    # Enumerate cells
    if args.v3:
        all_cells = enumerate_regression_suite_cells_v3(
            base_seed=args.seed, is_mixed_categorical=args.mixed_categorical
        )
    else:
        all_cells = enumerate_suite_cells(base_seed=args.seed)

    if args.suite_component != "all":
        all_cells = [c for c in all_cells if c["suite_component"] == args.suite_component]

    cells = all_cells[:n_datasets_limit]

    if args.dry_run:
        mode_label = "v3 mixed-cat" if args.mixed_categorical else ("v3" if args.v3 else "v2")
        print(f"[DRY-RUN] Mode: {mode_label} | Would generate {len(cells)} datasets:")
        for c in cells[:20]:
            family_key = c.get("target_family", c.get("nonlinear_family", "?"))
            print(
                f"  {c['logical_dataset_key']} | "
                f"{family_key}/{c['feature_regime']} | "
                f"n={c['n']} p={c['p_signal']} | "
                f"noise={c['target_noise_type']} scale={c['target_noise_scale']} | "
                f"{c['suite_component']}"
            )
        if len(cells) > 20:
            print(f"  ... ({len(cells) - 20} more)")
        return

    _preflight_check(cells)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_datasets: list[dict] = []
    n_ok = 0
    n_err = 0

    for idx, cell in enumerate(cells):
        if args.v3:
            entry = _generate_and_write_v3(
                cell, idx, out_dir, is_mixed_categorical=args.mixed_categorical
            )
        else:
            entry = _generate_and_write_v2(cell, idx, out_dir)

        if entry is None:
            n_err += 1
        else:
            manifest_datasets.append(entry)
            n_ok += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(cells):
            print(
                f"[INFO] [{idx + 1:4d}/{len(cells)}] "
                f"ok={n_ok} err={n_err}",
                flush=True,
            )

    manifest = {
        "suite_id":      suite_id,
        "prior_name":    "nonlinear",
        "prior_version": prior_version,
        "base_seed":     args.seed,
        "n_datasets":    n_ok,
        "target_families": target_families_list,
        "split_seeds":   list(V2_SPLIT_SEEDS),
        "is_mixed_categorical": args.mixed_categorical if args.v3 else False,
        "datasets":      manifest_datasets,
    }
    manifest_path = out_dir / manifest_filename
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(
        f"[INFO] Manifest written: {manifest_path} ({n_ok} datasets, {n_err} errors).",
        flush=True,
    )

    # SnowSQL PUT commands
    abs_manifest = manifest_path.resolve().as_posix()
    print("\n--- SnowSQL PUT commands ---")
    written_families = sorted({e["target_family"] for e in manifest_datasets})
    for family in written_families:
        family_dir_abs = (out_dir / family).resolve().as_posix()
        print(
            f"PUT file://{family_dir_abs}/*.parquet "
            f"{stage_prefix}/{family}/ "
            f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
        )
    print(
        f"PUT file://{abs_manifest} "
        f"{stage_prefix}/ "
        f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    )


if __name__ == "__main__":
    main()
