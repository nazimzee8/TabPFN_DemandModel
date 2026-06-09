"""
generate_nonlinear_classification.py
====================================
Local CLI script to generate nonlinear synthetic classification evaluation datasets.

Supports both pure numeric and mixed-categorical variants, controlled by the
``--mixed-categorical`` flag (per architecture SKILL.md: one script per concern).

Usage
-----
    # Pure numeric classification:
    python scripts/generate_nonlinear_classification.py \\
        --n-datasets 400 --out-dir data/nonlinear_classification_v1/

    # Mixed-categorical classification:
    python scripts/generate_nonlinear_classification.py \\
        --mixed-categorical --n-datasets 100

    # Smoke test:
    python scripts/generate_nonlinear_classification.py --n-datasets 5 --dry-run

    # Core component only:
    python scripts/generate_nonlinear_classification.py --suite-component core

Output
------
    Numeric:   data/nonlinear_classification_v1/{family}/dataset_{idx:05d}.parquet
    Mixed-cat: data/nonlinear_mixed_classification_v1/{family}/dataset_{idx:05d}.parquet
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
    LearnabilityGateError,
    V3_CLASSIFICATION_FAMILIES,
    BASE_SEED,
    enumerate_classification_suite_cells,
    generate_nonlinear_classification_dataset,
    generate_nonlinear_mixed_classification_dataset,
    write_parquet_nonlinear_classification_eval,
)


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------

def _preflight_check(cells: list[dict]) -> None:
    """Check for duplicate logical_dataset_key and condition_id."""
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
# Dataset generation
# ---------------------------------------------------------------------------

def _generate_and_write(
    cell: dict,
    idx: int,
    out_dir: Path,
    *,
    is_mixed_categorical: bool = False,
) -> dict | None:
    """Generate one classification dataset and write it."""
    family = cell["nonlinear_family"]
    family_dir = out_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{family}/dataset_{idx:05d}.parquet"
    outpath = out_dir / filename

    extra_kw = {k: v for k, v in cell.items() if k not in (
        "nonlinear_family", "feature_regime", "n", "p_signal",
        "num_classes", "temperature", "label_noise_rate",
        "class_imbalance_type", "margin_level", "suite_component",
        "teacher_seed", "sample_seed", "logical_dataset_key", "condition_id",
        "is_mixed_categorical",
    )}

    try:
        if is_mixed_categorical:
            ds = generate_nonlinear_mixed_classification_dataset(
                nonlinear_family=family,
                feature_regime=cell["feature_regime"],
                n=cell["n"],
                p_signal=cell["p_signal"],
                num_classes=cell["num_classes"],
                temperature=cell["temperature"],
                label_noise_rate=cell["label_noise_rate"],
                class_imbalance_type=cell["class_imbalance_type"],
                margin_level=cell["margin_level"],
                suite_component=cell["suite_component"],
                teacher_seed=cell["teacher_seed"],
                sample_seed=cell["sample_seed"],
                **extra_kw,
            )
        else:
            ds = generate_nonlinear_classification_dataset(
                nonlinear_family=family,
                feature_regime=cell["feature_regime"],
                n=cell["n"],
                p_signal=cell["p_signal"],
                num_classes=cell["num_classes"],
                temperature=cell["temperature"],
                label_noise_rate=cell["label_noise_rate"],
                class_imbalance_type=cell["class_imbalance_type"],
                margin_level=cell["margin_level"],
                suite_component=cell["suite_component"],
                teacher_seed=cell["teacher_seed"],
                sample_seed=cell["sample_seed"],
                **extra_kw,
            )
    except (LearnabilityGateError, Exception) as exc:
        print(
            f"[WARN] Error for {cell['logical_dataset_key']!r}: {exc}",
            file=sys.stderr,
        )
        return None

    write_parquet_nonlinear_classification_eval(
        ds, str(outpath), is_mixed_categorical=is_mixed_categorical
    )

    entry: dict[str, Any] = {
        "logical_dataset_key":  cell["logical_dataset_key"],
        "condition_id":         cell["condition_id"],
        "nonlinear_family":     family,
        "feature_regime":       cell["feature_regime"],
        "suite_component":      cell["suite_component"],
        "dataset_idx":          idx,
        "filename":             filename,
        "n_total":              int(ds["n_total"]),
        "p_signal":             int(ds["p_signal"]),
        "p_noise":              int(ds["p_noise"]),
        "p_total":              int(ds["p_total"]),
        "num_classes":          int(ds["num_classes"]),
        "realized_num_classes": int(ds["realized_num_classes"]),
        "temperature":          float(ds["temperature"]),
        "label_noise_rate":     float(ds["label_noise_rate"]),
        "realized_label_noise_rate": float(ds["realized_label_noise_rate"]),
        "class_imbalance_type": ds["class_imbalance_type"],
        "margin_level":         ds["margin_level"],
        "realized_margin_level": ds["realized_margin_level"],
        "teacher_seed":         int(cell["teacher_seed"]),
        "sample_seed":          int(cell["sample_seed"]),
        "task_family":          ds.get("task_family", ""),
        "schema_version":       ds.get("schema_version", ""),
    }
    if is_mixed_categorical:
        entry["p_cat"] = int(ds.get("p_cat", 0))
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nonlinear synthetic classification evaluation datasets."
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=None,
        help="Total datasets. Default: 400 (numeric) or 100 (mixed-cat).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default depends on --mixed-categorical.",
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
        "--mixed-categorical",
        action="store_true",
        default=False,
        help="Generate mixed-categorical classification variant.",
    )
    args = parser.parse_args()

    if args.mixed_categorical:
        default_out = "data/nonlinear_mixed_classification_v1"
        default_n = 100
        manifest_filename = "nonlinear_mixed_classification_v1_manifest.json"
        suite_id = "nonlinear_mixed_classification"
        stage_prefix = "@EVALUATION_DATASET_STAGE/nonlinear_mixed_classification_v1"
    else:
        default_out = "data/nonlinear_classification_v1"
        default_n = 400
        manifest_filename = "nonlinear_classification_v1_manifest.json"
        suite_id = "nonlinear_classification"
        stage_prefix = "@EVALUATION_DATASET_STAGE/nonlinear_classification_v1"

    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out)
    n_datasets_limit = args.n_datasets if args.n_datasets is not None else default_n

    all_cells = enumerate_classification_suite_cells(
        base_seed=args.seed, is_mixed_categorical=args.mixed_categorical
    )

    if args.suite_component != "all":
        all_cells = [c for c in all_cells if c["suite_component"] == args.suite_component]

    cells = all_cells[:n_datasets_limit]

    if args.dry_run:
        mode_label = "mixed-categorical" if args.mixed_categorical else "numeric"
        print(f"[DRY-RUN] Mode: {mode_label} | Would generate {len(cells)} datasets:")
        for c in cells[:20]:
            print(
                f"  {c['logical_dataset_key']} | "
                f"{c['nonlinear_family']}/{c['feature_regime']} | "
                f"n={c['n']} p={c['p_signal']} K={c['num_classes']} | "
                f"T={c['temperature']} ln={c['label_noise_rate']} | "
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
        entry = _generate_and_write(
            cell, idx, out_dir, is_mixed_categorical=args.mixed_categorical
        )
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
        "suite_id":              suite_id,
        "prior_name":            "nonlinear_classification",
        "prior_version":         "v1",
        "base_seed":             args.seed,
        "n_datasets":            n_ok,
        "classification_families": list(V3_CLASSIFICATION_FAMILIES),
        "is_mixed_categorical":  args.mixed_categorical,
        "datasets":              manifest_datasets,
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
    written_families = sorted({e["nonlinear_family"] for e in manifest_datasets})
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
