"""
generate_synthetic_classification.py
=====================================
Generate synthetic classification evaluation data as Parquet files locally.

Writes to data/synthetic_classification_prepared/{suite_id}/ with schema_version
"linear_classification_eval_v1". Supports 8 suite families:
primary, feature_noise, label_noise, training_size, class_imbalance, margin,
num_classes, ood.

Usage (smoke test):
    python scripts/generate_synthetic_classification.py \\
        --n_datasets 20 --n_datasets_per_sweep 3 \\
        --suite_id linear_classification_smoke_v1 \\
        --base_seed 20260512 --store_class_params

Canonical suite:
    python scripts/generate_synthetic_classification.py \\
        --n_datasets 1000 --n_datasets_per_sweep 50 \\
        --suite_id linear_classification_v1 \\
        --base_seed 20260512 --store_class_params
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import (  # noqa: E402
    CURRICULUM_POLICIES,
    DEFAULT_GPU_GUARD_BYTES,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_FEATURE_CAP,
    DEFAULT_TEST_BATCH_SIZE,
    MAX_MEMORY_RISK_CHOICES,
    CLASSIFICATION_COVERAGE_AXES,
)
from dgp_helpers import (
    ALLOCATION_MODES,
    CLASSIFICATION_EVAL_SCHEMA_VERSION,
    CLASSIFICATION_EVAL_ONLY_REGIMES,
    CLASSIFICATION_EVAL_SUITE_FAMILIES,
    CLASSIFICATION_PROFILES,
    CLASSIFICATION_REGIME_REGISTRY,
    CLASS_COUNT_WEIGHTS,
    IMBALANCE_WEIGHTS,
    REGIME_WEIGHTS,
    allocate_classification_tasks,
    allocate_mixed_classification_tasks,
    allocate_regimes,
    apply_symmetric_label_noise,
    build_coverage_audit,
    build_difficulty_audit,
    build_memory_audit,
    build_mixed_classification_dataset,
    build_task_fingerprint,
    build_train_eval_alignment_report,
    compute_difficulty_metadata,
    compute_difficulty_score,
    compute_classification_diagnostics,
    compute_classification_teacher,
    estimate_generation_memory,
    estimate_memory_risk,
    generate_classification_dataset,
    mark_unseen_query_categories,
    parse_difficulty_mix,
    split_classification_dataset,
    validate_classification_dataset,
    validate_controlled_manifest,
    validate_mixed_classification_dataset,
    validate_per_split_k_coverage,
    write_parquet_classification_eval,
    write_parquet_mixed_classification_dgp,
)

# Env-var mixed-categorical detection (for orchestrator-driven generation)
SYNCLS_IS_MIXED_CATEGORICAL = os.getenv(
    "SYNCLS_IS_MIXED_CATEGORICAL", "false"
).lower() in ("1", "true", "yes")

STAGE_PREFIX = "@EVALUATION_DATASET_STAGE"
OUTPUT_SUBDIR = "synthetic_classification_prepared"

# Magic constant for SeedSequence derivation (eval tasks only; training uses generate_dgp.py pattern)
_EVAL_SEED_MAGIC = 0xECA1C1A5
_HIDDEN_EVAL_SEED_MAGIC = 0xC1A55EED

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic classification evaluation data."
    )
    p.add_argument("--suite_id", type=str, default=None,
                   help="Suite identifier (auto-generated if not provided).")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--n_datasets", type=int, default=500,
                   help="Primary suite size.")
    p.add_argument("--n_datasets_per_sweep", type=int, default=50,
                   help="Datasets per config in non-primary families.")
    p.add_argument("--eval_only_n_datasets", type=int, default=None)
    p.add_argument("--hidden_holdout_n_datasets", type=int, default=None)
    p.add_argument(
        "--hidden_holdout_suite_id",
        type=str,
        default="linear_classification_hidden_holdout_v1",
    )
    p.add_argument("--hidden_holdout_base_seed", type=int, default=20260607)
    p.add_argument(
        "--allocation_mode",
        choices=sorted(ALLOCATION_MODES),
        default="balanced",
    )
    p.add_argument("--strict_coverage", action="store_true", default=False)
    p.add_argument(
        "--no-strict_coverage", dest="strict_coverage", action="store_false"
    )
    p.add_argument("--coverage_config", type=str, default=None)
    p.add_argument("--profile", type=str,
                   default="linear_classification_stat_aware",
                   choices=CLASSIFICATION_PROFILES)
    p.add_argument("--output_root", type=str,
                   default="data/synthetic_classification_prepared")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--dry_run", action="store_true", default=False)

    # Standard grids
    p.add_argument("--n_grid", type=int, nargs="+",
                   default=[100, 200, 500, 1000, 2000, 5000])
    p.add_argument("--p_signal_grid", type=int, nargs="+",
                   default=[2, 5, 10, 20, 50, 100])
    p.add_argument("--p_noise_grid", type=int, nargs="+",
                   default=[0, 5, 10, 25, 50],
                   help="Number of irrelevant features (integer count) per sweep point")
    p.add_argument("--feature_noise_grid", type=float, nargs="+",
                   default=[0.0, 0.05, 0.10, 0.25],
                   help="Additive feature-noise amplitude per sweep point")
    p.add_argument("--feature_noise_amplitude_grid", type=float, nargs="+",
                   default=None,
                   help="Explicit additive feature-noise amplitude grid (overrides "
                        "--feature_noise_grid when set)")

    # Classification-specific grids
    p.add_argument("--num_classes_grid", type=int, nargs="+",
                   default=[2, 3, 5, 10])
    p.add_argument("--temperature_grid", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--label_noise_grid", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.10, 0.20])
    p.add_argument("--class_imbalance_grid", type=str, nargs="+",
                   default=["balanced", "mild", "moderate", "severe"],
                   choices=["balanced", "mild", "moderate", "severe"])
    p.add_argument("--margin_grid", type=str, nargs="+",
                   default=["low", "medium", "high"],
                   choices=["low", "medium", "high"])
    p.add_argument("--coefficient_scale_grid", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--intercept_scale_grid", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0])

    # Classification flags
    p.add_argument("--store_class_params", action="store_true", default=True)
    p.add_argument("--no-store_class_params", dest="store_class_params",
                   action="store_false")
    p.add_argument("--store_class_teacher_preds", action="store_true", default=False)
    p.add_argument("--require_class_teachers", action="store_true", default=False)
    p.add_argument("--allow_underdetermined", action="store_true", default=False)

    # Suite family enable/disable
    for fam in sorted(CLASSIFICATION_EVAL_SUITE_FAMILIES):
        safe = fam.replace("_", "_")
        p.add_argument(f"--include_{fam}", action="store_true",
                       dest=f"include_{fam}", default=None)
        p.add_argument(f"--no-include_{fam}", action="store_false",
                       dest=f"include_{fam}")

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--min_regime_count", type=int, default=10)
    p.add_argument("--min_suite_family_count", type=int, default=5)
    p.add_argument(
        "--curriculum_policy",
        choices=sorted(CURRICULUM_POLICIES),
        default="core_first",
    )
    p.add_argument(
        "--difficulty_mix", type=str, default="core=0.65,robust=0.30,stress=0.05",
    )
    p.add_argument(
        "--max_memory_risk",
        choices=list(MAX_MEMORY_RISK_CHOICES),
        default="high",
    )
    p.add_argument("--memory_guard_bytes", type=int, default=DEFAULT_GPU_GUARD_BYTES)
    p.add_argument("--allow_memory_stress", action="store_true", default=False)
    p.add_argument("--emit_memory_stress_suite", action="store_true", default=False)
    p.add_argument("--emit_alignment_report", action="store_true", default=False)
    p.add_argument("--train_manifest_path", type=str, default=None)
    p.add_argument("--eval_manifest_path", type=str, default=None)
    p.add_argument("--alignment_report_out", type=str, default=None)
    p.add_argument("--strict_manifest_validation", action="store_true", default=False)
    p.add_argument("--dry_run_allocation", action="store_true", default=False)
    p.add_argument("--write_allocation_audit_only", action="store_true", default=False)
    p.add_argument(
        "--mixed_categorical",
        action="store_true",
        default=False,
        help=(
            "Generate mixed-categorical classification datasets (numeric + categorical features). "
            "Output goes to mixed_classification_prepared/ instead of synthetic_classification_prepared/."
        ),
    )

    return p.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.n_datasets < 1:
        raise ValueError("--n_datasets must be >= 1")
    if args.n_datasets_per_sweep < 1:
        raise ValueError("--n_datasets_per_sweep must be >= 1")
    enabled = _get_enabled_families(args)
    if not enabled:
        raise ValueError("At least one suite family must be enabled.")
    if args.require_class_teachers and not args.store_class_teacher_preds:
        raise ValueError(
            "--require_class_teachers requires --store_class_teacher_preds"
        )
    if args.hidden_holdout_suite_id == args.suite_id:
        raise ValueError(
            "--hidden_holdout_suite_id must be a separate logical seed namespace"
        )


def _get_enabled_families(args: argparse.Namespace) -> list[str]:
    order = ["primary", "feature_noise", "label_noise", "training_size",
             "class_imbalance", "margin", "num_classes", "ood",
             "eval_only_unseen", "hidden_holdout", "stress"]
    legacy_families = order[:8]
    if all(getattr(args, f"include_{f}", None) is False for f in legacy_families):
        return [
            f for f in order[8:]
            if getattr(args, f"include_{f}", None) is True
        ]
    if args.profile == "classification_legacy_debug":
        return [
            f for f in order
            if getattr(args, f"include_{f}", None) is True or (
                f == "primary" and getattr(args, f"include_{f}", None) is not False
            )
        ]
    return [
        f for f in order
        if getattr(args, f"include_{f}", None) is not False
    ]


def _make_suite_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"cls_eval_{ts}"


def _check_output_dir(suite_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if suite_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Output directory already exists: {suite_dir}\n"
            "Use --overwrite to regenerate."
        )


# ---------------------------------------------------------------------------
# Global Index Counter
# ---------------------------------------------------------------------------

class _GlobalIndexCounter:
    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        v = self._value
        self._value += 1
        return v

    @property
    def value(self) -> int:
        return self._value


def _task_seed(
    args: argparse.Namespace,
    global_idx: int,
    suite_family: str,
) -> int:
    hidden = suite_family == "hidden_holdout"
    base_seed = args.hidden_holdout_base_seed if hidden else args.base_seed
    magic = _HIDDEN_EVAL_SEED_MAGIC if hidden else _EVAL_SEED_MAGIC
    return int(
        np.random.SeedSequence([base_seed, global_idx, magic]).generate_state(1)[0]
    )


# ---------------------------------------------------------------------------
# CLS-S1: Read-after-write parquet field validation
# ---------------------------------------------------------------------------

# Minimum columns that must be present in every written classification eval parquet
# (write_parquet_classification_eval per-row format).  CLS-S1 read-after-write check.
_REQUIRED_EVAL_FIELDS = frozenset({
    "feature_vector", "label", "split",
    "task_family", "task_objective", "num_classes",
    "suite_id", "suite_family", "global_idx",
    "n_train", "n_test",
})


def _assert_required_eval_fields(path: Path) -> None:
    """Read the just-written parquet and verify all required columns are present.

    Raises ValueError immediately if the file is missing columns, so the error
    is caught at generation time rather than hours later at index time (CLS-S1).
    """
    import pyarrow.parquet as pq
    table = pq.read_table(str(path))
    names = set(table.schema.names)
    missing = _REQUIRED_EVAL_FIELDS - names
    if missing:
        raise ValueError(
            f"Read-after-write validation failed for {path}: "
            f"missing required columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Dataset record builder
# ---------------------------------------------------------------------------

def _compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_dataset_record(
    filepath: Path,
    suite_dir: Path,
    dataset_idx: int,
    global_idx: int,
    suite_family: str,
    regime: str,
    ds: dict,
    n_rows: int,
    is_tabpfn_anchor: bool = False,
    extra: dict | None = None,
    task_seed: int | None = None,
) -> dict:
    rel = filepath.relative_to(suite_dir).as_posix()
    checksum = _compute_sha256(filepath)
    n_train = int(ds["n_train"])
    n_test = int(ds["n_test"])
    difficulty = compute_difficulty_metadata(
        n_train=n_train,
        p_total=int(ds["p_total"]),
        noise_scale=float(ds.get("label_noise_rate", 0.0)) * 10.0,
        feature_noise_level=float(ds.get("feature_noise_level", 0.0)),
        sparsity_ratio=float(ds.get("sparsity_ratio", 0.0)),
        rho=float(ds.get("rho", 0.0)),
        num_classes=int(ds["num_classes"]),
        imbalance_strength={
            "balanced": 0.0, "mild": 0.15, "moderate": 0.35, "severe": 0.60
        }.get(str(ds.get("class_imbalance_type", "balanced")), 0.0),
    )
    memory = estimate_generation_memory(n_train, n_test, int(ds["p_total"]))
    fingerprint = build_task_fingerprint(
        task_family="linear_classification",
        regime=regime,
        suite_family=suite_family,
        n_total=n_train + n_test,
        p_total=int(ds["p_total"]),
        covariance_type=str(ds.get("covariance_type", "iid")),
        noise_type="symmetric_label_noise",
        sparsity_ratio=float(ds.get("sparsity_ratio", 0.0)),
        classification_geometry=(
            f"K={int(ds['num_classes'])};"
            f"imbalance={ds.get('class_imbalance_type', '')};"
            f"margin={ds.get('margin_level', '')}"
        ),
    )
    spec = CLASSIFICATION_REGIME_REGISTRY.get(regime, {})
    return {
        "filepath": rel,
        "dataset_idx": dataset_idx,
        "global_idx": global_idx,
        "suite_family": suite_family,
        "regime": regime,
        "task_seed": task_seed,
        "K": int(ds["num_classes"]),
        "imbalance_type": ds.get("class_imbalance_type", ""),
        "margin_level": ds.get("margin_level", ""),
        "label_noise_rate": float(ds.get("label_noise_rate", 0.0)),
        "feature_noise_level": float(ds.get("feature_noise_level", 0.0)),
        "n_train": n_train,
        "n_test": n_test,
        "n_features": int(ds["p_total"]),
        "n_rows": n_rows,
        "is_tabpfn_anchor": is_tabpfn_anchor,
        "file_checksum_sha256": checksum,
        "distribution_family": ds.get(
            "distribution_family", spec.get("distribution_family", "gaussian_linear_logits")
        ),
        "covariance_type": ds.get("covariance_type", "iid"),
        "rho": float(ds.get("rho", 0.0)),
        "temperature": float(ds.get("temperature", 1.0)),
        "is_training_allowed": bool(spec.get("is_training_allowed", True)),
        "is_eval_only": bool(spec.get("is_eval_only", False)),
        "is_ood": suite_family in {
            "ood", "eval_only_unseen", "hidden_holdout", "stress"
        },
        "is_hidden_holdout": suite_family == "hidden_holdout",
        "task_fingerprint": fingerprint,
        **difficulty,
        **memory,
        "extra": extra or {},
    }


# ---------------------------------------------------------------------------
# Grids helper
# ---------------------------------------------------------------------------

def _build_grids(args: argparse.Namespace) -> dict:
    # feature_noise_amplitude_grid (explicit flag) overrides feature_noise_grid
    # p_noise_grid: number of irrelevant features (integer count) — separate from amplitude
    amplitude_grid = (
        [float(x) for x in args.feature_noise_amplitude_grid]
        if getattr(args, "feature_noise_amplitude_grid", None) is not None
        else [float(x) for x in args.feature_noise_grid]
    )
    return {
        "n_grid": args.n_grid,
        "p_signal_grid": args.p_signal_grid,
        "p_noise_grid": args.p_noise_grid,
        "active_s_grid": [2, 4, 8, 16, 32],
        "rho_grid": [0.0, 0.3, 0.6, 0.9],
        "feature_noise_grid": amplitude_grid,
        "feature_noise_amplitude_grid": amplitude_grid,
        "num_classes_grid": args.num_classes_grid,
        "temperature_grid": args.temperature_grid,
        "label_noise_grid": args.label_noise_grid,
        "class_imbalance_grid": args.class_imbalance_grid,
        "margin_grid": args.margin_grid,
        "coefficient_scale_grid": args.coefficient_scale_grid,
        "intercept_scale_grid": args.intercept_scale_grid,
    }


# ---------------------------------------------------------------------------
# Single task generator
# ---------------------------------------------------------------------------

def _generate_one_task(
    args: argparse.Namespace,
    grids: dict,
    assignment: dict,
    global_idx: int,
    suite_id: str,
    suite_family: str,
    dataset_idx: int,
    out_path: Path,
    is_tabpfn_anchor: bool = False,
) -> tuple[dict, int]:
    """Generate one task, write parquet, return (record_dict, n_rows)."""
    task_seed = _task_seed(args, global_idx, suite_family)
    rng = np.random.default_rng(task_seed)
    regime = assignment["classification_regime"]

    ds_raw = generate_classification_dataset(
        rng, regime, assignment, grids, task_seed=task_seed
    )
    if suite_family == "feature_noise":
        level = float(grids["feature_noise_grid"][0])
        ds_raw["X"] = (
            ds_raw["X_clean"] + level * rng.standard_normal(ds_raw["X_clean"].shape)
        )
        ds_raw["feature_noise_level"] = level
    validate_classification_dataset(ds_raw)

    ds_split = split_classification_dataset(ds_raw)
    diagnostics = compute_classification_diagnostics(ds_raw)

    teacher = compute_classification_teacher(
        ds_split["X_train"], ds_split["y_train"],
        ds_split["X_test"], ds_raw["num_classes"],
        requested=args.store_class_teacher_preds or args.require_class_teachers,
    )
    if args.require_class_teachers and not teacher["teacher_available"]:
        raise RuntimeError(
            f"Required teacher failed: regime={regime}, seed={task_seed}, "
            f"reason={teacher['teacher_failure_reason']}"
        )

    n_rows = write_parquet_classification_eval(
        ds_split,
        str(out_path),
        suite_id=suite_id,
        suite_family=suite_family,
        dataset_idx=dataset_idx,
        global_idx=global_idx,
        profile=args.profile,
        regime=regime,
        task_seed=task_seed,
        store_class_params=args.store_class_params,
        store_teacher_preds=args.store_class_teacher_preds,
        teacher_results=teacher,
        diagnostics=diagnostics,
        is_tabpfn_anchor=is_tabpfn_anchor,
        extra_metadata={
            **compute_difficulty_metadata(
                n_train=int(ds_split["n_train"]),
                p_total=int(ds_split["p_total"]),
                noise_scale=float(ds_split.get("label_noise_rate", 0.0)) * 10.0,
                feature_noise_level=float(ds_split.get("feature_noise_level", 0.0)),
                sparsity_ratio=float(ds_split.get("sparsity_ratio", 0.0)),
                rho=float(ds_split.get("rho", 0.0)),
                num_classes=int(ds_split["num_classes"]),
            ),
            **estimate_generation_memory(
                int(ds_split["n_train"]),
                int(ds_split["n_test"]),
                int(ds_split["p_total"]),
            ),
            "is_eval_only": bool(
                CLASSIFICATION_REGIME_REGISTRY.get(regime, {}).get(
                    "is_eval_only", False
                )
            ),
            "is_training_allowed": bool(
                CLASSIFICATION_REGIME_REGISTRY.get(regime, {}).get(
                    "is_training_allowed", True
                )
            ),
            "is_ood": suite_family in {
                "ood", "eval_only_unseen", "hidden_holdout", "stress"
            },
            "is_hidden_holdout": suite_family == "hidden_holdout",
            "hidden_holdout_suite_id": (
                args.hidden_holdout_suite_id
                if suite_family == "hidden_holdout"
                else ""
            ),
            "task_fingerprint": build_task_fingerprint(
                task_family="linear_classification",
                regime=regime,
                suite_family=suite_family,
                n_total=int(ds_split["n_train"] + ds_split["n_test"]),
                p_total=int(ds_split["p_total"]),
                covariance_type=str(ds_split.get("covariance_type", "iid")),
                noise_type="symmetric_label_noise",
                sparsity_ratio=float(ds_split.get("sparsity_ratio", 0.0)),
                classification_geometry=(
                    f"K={int(ds_split['num_classes'])};"
                    f"imbalance={ds_split.get('class_imbalance_type', '')};"
                    f"margin={ds_split.get('margin_level', '')}"
                ),
            ),
        },
    )

    # CLS-S1: read-after-write validation — catch missing/truncated parquets immediately
    # rather than hours later at index time.
    _assert_required_eval_fields(out_path)

    return ds_split, n_rows


def _default_assignment(
    args: argparse.Namespace,
    rng: np.random.Generator,
    regime: str | None = None,
    K: int | None = None,
    imbalance: str | None = None,
    margin: str | None = None,
    label_noise: float | None = None,
) -> dict:
    """Build a minimal assignment dict for families that don't use allocate_classification_tasks."""
    regime = regime or "A_iid_dense_logistic"
    K = K or 2
    imbalance = imbalance or "balanced"
    margin = margin or "medium"
    label_noise = label_noise if label_noise is not None else 0.0
    return {
        "classification_regime": regime,
        "num_classes": K,
        "class_imbalance_type": imbalance,
        "margin_level": margin,
        "label_noise_rate": label_noise,
        "coefficient_regime": None,
    }


# ---------------------------------------------------------------------------
# Suite family generators
# ---------------------------------------------------------------------------

def _generate_primary(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    family_dir = suite_dir / "primary"
    family_dir.mkdir(parents=True, exist_ok=True)
    n = args.n_datasets

    assignments, _ = allocate_classification_tasks(
        n, args.profile, args.base_seed,
        allow_underdetermined=args.allow_underdetermined,
        num_classes=grids["num_classes_grid"],
        imbalance_levels=grids["class_imbalance_grid"],
        margin_levels=grids["margin_grid"],
        label_noise_values=grids["label_noise_grid"],
        allocation_mode=args.allocation_mode,
    )

    records = []
    for dataset_idx, assignment in enumerate(assignments):
        global_idx = counter.next()
        out_path = family_dir / f"dataset_{dataset_idx:04d}.parquet"
        ds_split, n_rows = _generate_one_task(
            args, grids, assignment, global_idx, suite_id, "primary",
            dataset_idx, out_path
        )
        _task_seed = int(
            np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
            .generate_state(1)[0]
        )
        rec = _make_dataset_record(
            out_path, suite_dir, dataset_idx, global_idx, "primary",
            assignment["classification_regime"], ds_split, n_rows,
            task_seed=_task_seed,
        )
        records.append(rec)
        if args.verbose or (dataset_idx + 1) % args.log_every == 0:
            print(f"  [primary {dataset_idx + 1:4d}/{n}] {out_path.name}")
    return records


def _generate_feature_noise(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    # Use grids["feature_noise_grid"] which respects --feature_noise_amplitude_grid override
    noise_levels = [float(value) for value in grids.get("feature_noise_grid", args.feature_noise_grid)]
    records = []
    rng_alloc = np.random.default_rng(args.base_seed ^ 0xFEA7)
    regimes = list(REGIME_WEIGHTS[args.profile].keys())

    for level in noise_levels:
        level_name = str(level).replace(".", "p")
        subdir = suite_dir / "feature_noise" / f"level_{level_name}"
        subdir.mkdir(parents=True, exist_ok=True)
        for dataset_idx in range(args.n_datasets_per_sweep):
            global_idx = counter.next()
            regime = str(rng_alloc.choice(regimes))
            if "J_low_n_high_p" in regime and not args.allow_underdetermined:
                regime = "A_iid_dense_logistic"
            assignment = _default_assignment(args, rng_alloc, regime=regime)
            # Override feature_noise_level in grids for this task
            task_grids = {**grids, "feature_noise_grid": [float(level)]}
            out_path = subdir / f"dataset_{dataset_idx:04d}.parquet"
            ds_split, n_rows = _generate_one_task(
                args, task_grids, assignment, global_idx, suite_id,
                "feature_noise", dataset_idx, out_path
            )
            _task_seed_fn = int(
                np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
                .generate_state(1)[0]
            )
            rec = _make_dataset_record(
                out_path, suite_dir, dataset_idx, global_idx,
                "feature_noise", regime, ds_split, n_rows,
                extra={"noise_level": level},
                task_seed=_task_seed_fn,
            )
            records.append(rec)
    return records


def _generate_label_noise(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.20]
    records = []
    rng_alloc = np.random.default_rng(args.base_seed ^ 0x1ABE1)
    regimes = list(REGIME_WEIGHTS[args.profile].keys())

    for rate in noise_rates:
        rate_str = f"{rate:.2f}".replace(".", "p")
        subdir = suite_dir / "label_noise" / f"rate_{rate_str}"
        subdir.mkdir(parents=True, exist_ok=True)

        for dataset_idx in range(args.n_datasets_per_sweep):
            global_idx = counter.next()
            regime = str(rng_alloc.choice(regimes))
            if "J_low_n_high_p" in regime and not args.allow_underdetermined:
                regime = "A_iid_dense_logistic"
            # Override label_noise_rate
            assignment = _default_assignment(args, rng_alloc, regime=regime, label_noise=rate)
            out_path = subdir / f"dataset_{dataset_idx:04d}.parquet"
            ds_split, n_rows = _generate_one_task(
                args, grids, assignment, global_idx, suite_id,
                "label_noise", dataset_idx, out_path
            )
            _task_seed_ln = int(
                np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
                .generate_state(1)[0]
            )
            rec = _make_dataset_record(
                out_path, suite_dir, dataset_idx, global_idx,
                "label_noise", regime, ds_split, n_rows,
                extra={"noise_rate": rate},
                task_seed=_task_seed_ln,
            )
            records.append(rec)
    return records


def _manual_split(
    ds_raw: dict,
    n_train: int,
    n_test: int,
) -> dict:
    """Override the 80/20 split with fixed n_train / n_test counts."""
    import numpy as np
    n_needed = n_train + n_test
    n_total = int(ds_raw["n_total"])
    # If the dataset is smaller than needed, repeat rows (unlikely but safe)
    if n_total < n_needed:
        factor = int(np.ceil(n_needed / n_total))
        X = np.tile(ds_raw["X"], (factor, 1))[:n_needed]
        y = np.tile(ds_raw["y"], factor)[:n_needed]
        y_clean = np.tile(ds_raw["y_clean"], factor)[:n_needed]
        lnm = np.tile(ds_raw["label_noise_mask"], factor)[:n_needed]
        logits = np.tile(ds_raw["logits"], (factor, 1))[:n_needed]
        probs = np.tile(ds_raw["probs"], (factor, 1))[:n_needed]
    else:
        X = ds_raw["X"][:n_needed]
        y = ds_raw["y"][:n_needed]
        y_clean = ds_raw["y_clean"][:n_needed]
        lnm = ds_raw["label_noise_mask"][:n_needed]
        logits = ds_raw["logits"][:n_needed]
        probs = ds_raw["probs"][:n_needed]

    split = dict(ds_raw)
    split.update({
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "y_clean_train": y_clean[:n_train],
        "y_clean_test": y_clean[n_train:],
        "label_noise_mask_train": lnm[:n_train],
        "label_noise_mask_test": lnm[n_train:],
        "logits_train": logits[:n_train],
        "logits_test": logits[n_train:],
        "probs_train": probs[:n_train],
        "probs_test": probs[n_train:],
        "n": n_train + n_test,
        "p": int(ds_raw["p_total"]),
        "n_train": n_train,
        "n_test": n_test,
        "train_class_counts": np.bincount(
            y[:n_train], minlength=ds_raw["num_classes"]
        ).astype(np.int64),
        "test_class_counts": np.bincount(
            y[n_train:], minlength=ds_raw["num_classes"]
        ).astype(np.int64),
    })
    return split


def _generate_training_size(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    n_holdout = 1371
    n_train_values = [25, 50, 100, 200, 500, 1000, 2000, 4832]
    anchor_n_train = 4832
    records = []
    rng_alloc = np.random.default_rng(args.base_seed ^ 0x7EA1)
    regimes = list(REGIME_WEIGHTS[args.profile].keys())

    for n_train in n_train_values:
        is_anchor = (n_train == anchor_n_train)
        subdir = suite_dir / "training_size" / f"n{n_train}"
        subdir.mkdir(parents=True, exist_ok=True)

        for dataset_idx in range(args.n_datasets_per_sweep):
            global_idx = counter.next()
            task_seed = int(
                np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
                .generate_state(1)[0]
            )
            rng = np.random.default_rng(task_seed)
            regime = str(rng_alloc.choice(regimes))
            if "J_low_n_high_p" in regime and not args.allow_underdetermined:
                regime = "A_iid_dense_logistic"
            assignment = _default_assignment(args, rng_alloc, regime=regime)

            # Need enough rows total; use allow_underdetermined for small n_train
            task_grids = {**grids}
            n_needed = n_train + n_holdout
            # Force n to be large enough in grids
            big_n = max(n_needed, max(grids["n_grid"]))
            task_grids["n_grid"] = [big_n]

            ds_raw = generate_classification_dataset(
                rng, regime, assignment, task_grids, task_seed=task_seed,
                max_attempts=64,
            )
            validate_classification_dataset(ds_raw)
            ds_split = _manual_split(ds_raw, n_train, n_holdout)
            diagnostics = compute_classification_diagnostics(ds_raw)
            teacher = compute_classification_teacher(
                ds_split["X_train"], ds_split["y_train"],
                ds_split["X_test"], ds_raw["num_classes"],
                requested=args.store_class_teacher_preds or args.require_class_teachers,
            )
            if args.require_class_teachers and not teacher["teacher_available"]:
                raise RuntimeError(
                    f"Required teacher failed: regime={regime}, seed={task_seed}"
                )

            out_path = subdir / f"dataset_{dataset_idx:04d}.parquet"
            n_rows = write_parquet_classification_eval(
                ds_split,
                str(out_path),
                suite_id=suite_id,
                suite_family="training_size",
                dataset_idx=dataset_idx,
                global_idx=global_idx,
                profile=args.profile,
                regime=regime,
                task_seed=task_seed,
                store_class_params=args.store_class_params,
                store_teacher_preds=args.store_class_teacher_preds,
                teacher_results=teacher,
                diagnostics=diagnostics,
                is_tabpfn_anchor=is_anchor,
            )
            rec = _make_dataset_record(
                out_path, suite_dir, dataset_idx, global_idx,
                "training_size", regime, ds_split, n_rows,
                is_tabpfn_anchor=is_anchor,
                extra={"n_train_sweep": n_train},
                task_seed=task_seed,
            )
            records.append(rec)
    return records


def _generate_class_imbalance(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    k_values = [2, 3, 5, 10]
    imbalance_levels = ["balanced", "mild", "moderate", "severe"]
    records = []
    rng_alloc = np.random.default_rng(args.base_seed ^ 0xC1A55)
    regimes = list(REGIME_WEIGHTS[args.profile].keys())

    for K in k_values:
        for imbalance in imbalance_levels:
            cell_name = f"K{K}_{imbalance}"
            subdir = suite_dir / "class_imbalance" / cell_name
            subdir.mkdir(parents=True, exist_ok=True)

            for dataset_idx in range(args.n_datasets_per_sweep):
                global_idx = counter.next()
                regime = str(rng_alloc.choice(regimes))
                if "J_low_n_high_p" in regime and not args.allow_underdetermined:
                    regime = "A_iid_dense_logistic"
                # Force binary regimes to K=2
                if K > 2 and regime in {
                    "A_iid_dense_logistic", "B_iid_sparse_logistic",
                    "D_correlated_ar_logistic"
                }:
                    regime = "C_label_noise_margin"
                assignment = _default_assignment(
                    args, rng_alloc, regime=regime, K=K, imbalance=imbalance
                )
                task_grids = {**grids, "num_classes_grid": [K],
                              "class_imbalance_grid": [imbalance]}
                out_path = subdir / f"dataset_{dataset_idx:04d}.parquet"
                ds_split, n_rows = _generate_one_task(
                    args, task_grids, assignment, global_idx, suite_id,
                    "class_imbalance", dataset_idx, out_path
                )
                _task_seed_ci = int(
                    np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
                    .generate_state(1)[0]
                )
                rec = _make_dataset_record(
                    out_path, suite_dir, dataset_idx, global_idx,
                    "class_imbalance", regime, ds_split, n_rows,
                    extra={"K": K, "imbalance": imbalance},
                    task_seed=_task_seed_ci,
                )
                records.append(rec)
    return records


def _generate_margin(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    margin_levels = ["low", "medium", "high"]
    records = []
    rng_alloc = np.random.default_rng(args.base_seed ^ 0xA791)
    regimes = list(REGIME_WEIGHTS[args.profile].keys())

    for margin in margin_levels:
        subdir = suite_dir / "margin" / margin
        subdir.mkdir(parents=True, exist_ok=True)
        for dataset_idx in range(args.n_datasets_per_sweep):
            global_idx = counter.next()
            regime = str(rng_alloc.choice(regimes))
            if "J_low_n_high_p" in regime and not args.allow_underdetermined:
                regime = "A_iid_dense_logistic"
            assignment = _default_assignment(
                args, rng_alloc, regime=regime, margin=margin
            )
            task_grids = {**grids, "margin_grid": [margin]}
            out_path = subdir / f"dataset_{dataset_idx:04d}.parquet"
            ds_split, n_rows = _generate_one_task(
                args, task_grids, assignment, global_idx, suite_id,
                "margin", dataset_idx, out_path
            )
            _task_seed_mg = int(
                np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
                .generate_state(1)[0]
            )
            rec = _make_dataset_record(
                out_path, suite_dir, dataset_idx, global_idx,
                "margin", regime, ds_split, n_rows,
                extra={"margin": margin},
                task_seed=_task_seed_mg,
            )
            records.append(rec)
    return records


def _generate_num_classes(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    k_values = [2, 3, 5, 10]
    records = []
    rng_alloc = np.random.default_rng(args.base_seed ^ 0xC1A55E5)
    regimes = list(REGIME_WEIGHTS[args.profile].keys())

    for K in k_values:
        subdir = suite_dir / "num_classes" / f"K{K}"
        subdir.mkdir(parents=True, exist_ok=True)
        for dataset_idx in range(args.n_datasets_per_sweep):
            global_idx = counter.next()
            regime = str(rng_alloc.choice(regimes))
            if "J_low_n_high_p" in regime and not args.allow_underdetermined:
                regime = "A_iid_dense_logistic"
            if K > 2 and regime in {
                "A_iid_dense_logistic", "B_iid_sparse_logistic",
                "D_correlated_ar_logistic"
            }:
                regime = "C_label_noise_margin"
            assignment = _default_assignment(args, rng_alloc, regime=regime, K=K)
            task_grids = {**grids, "num_classes_grid": [K]}
            out_path = subdir / f"dataset_{dataset_idx:04d}.parquet"
            ds_split, n_rows = _generate_one_task(
                args, task_grids, assignment, global_idx, suite_id,
                "num_classes", dataset_idx, out_path
            )
            _task_seed_nc = int(
                np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
                .generate_state(1)[0]
            )
            rec = _make_dataset_record(
                out_path, suite_dir, dataset_idx, global_idx,
                "num_classes", regime, ds_split, n_rows,
                extra={"K": K},
                task_seed=_task_seed_nc,
            )
            records.append(rec)
    return records


# OOD scenario specs
_OOD_SCENARIOS = [
    {
        "name": "heavy_tailed",
        "regime": "C_label_noise_margin",
        "K": 2,
        "imbalance": "balanced",
        "margin": "low",
        "label_noise": 0.10,
        "grids_override": {"temperature_grid": [4.0, 5.0]},
    },
    {
        "name": "bounded",
        "regime": "A_iid_dense_logistic",
        "K": 2,
        "imbalance": "balanced",
        "margin": "medium",
        "label_noise": 0.0,
        "grids_override": {},
    },
    {
        "name": "equicorrelated",
        "regime": "I_equicorrelated_classification",
        "K": 3,
        "imbalance": "mild",
        "margin": "medium",
        "label_noise": 0.0,
        "grids_override": {},
    },
    {
        "name": "high_dim_sparse",
        "regime": "F_high_dim_sparse_softmax",
        "K": 5,
        "imbalance": "balanced",
        "margin": "high",
        "label_noise": 0.0,
        "grids_override": {},
    },
    {
        "name": "severe_imbalance",
        "regime": "G_noise_features_classification",
        "K": 2,
        "imbalance": "severe",
        "margin": "medium",
        "label_noise": 0.0,
        "grids_override": {},
    },
    {
        "name": "high_noise",
        "regime": "K_feature_noise_classification",
        "K": 2,
        "imbalance": "balanced",
        "margin": "medium",
        "label_noise": 0.20,
        "grids_override": {"feature_noise_grid": [50.0, 75.0, 100.0]},
    },
]


def _generate_ood(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
) -> list[dict]:
    records = []

    for scenario in _OOD_SCENARIOS:
        name = scenario["name"]
        regime = scenario["regime"]
        # Skip J regime if underdetermined not allowed
        if "J_low_n_high_p" in regime and not args.allow_underdetermined:
            regime = "A_iid_dense_logistic"
        subdir = suite_dir / "ood" / name
        subdir.mkdir(parents=True, exist_ok=True)
        assignment = _default_assignment(
            args,
            np.random.default_rng(args.base_seed),
            regime=regime,
            K=scenario["K"],
            imbalance=scenario["imbalance"],
            margin=scenario["margin"],
            label_noise=scenario["label_noise"],
        )
        task_grids = {**grids, **scenario.get("grids_override", {})}
        # Ensure class_imbalance_grid and margin_grid include the scenario values
        task_grids["class_imbalance_grid"] = [scenario["imbalance"]]
        task_grids["margin_grid"] = [scenario["margin"]]
        task_grids["num_classes_grid"] = [scenario["K"]]

        for dataset_idx in range(args.n_datasets_per_sweep):
            global_idx = counter.next()
            out_path = subdir / f"dataset_{dataset_idx:04d}.parquet"
            ds_split, n_rows = _generate_one_task(
                args, task_grids, assignment, global_idx, suite_id,
                "ood", dataset_idx, out_path
            )
            _task_seed_ood = int(
                np.random.SeedSequence([args.base_seed, global_idx, _EVAL_SEED_MAGIC])
                .generate_state(1)[0]
            )
            rec = _make_dataset_record(
                out_path, suite_dir, dataset_idx, global_idx,
                "ood", regime, ds_split, n_rows,
                extra={"scenario": name},
                task_seed=_task_seed_ood,
            )
            records.append(rec)
    return records


def _generate_distribution_family(
    suite_dir: Path,
    args: argparse.Namespace,
    suite_id: str,
    counter: _GlobalIndexCounter,
    grids: dict,
    *,
    suite_family: str,
    regimes: list[str],
    count: int,
) -> list[dict]:
    family_dir = suite_dir / suite_family
    family_dir.mkdir(parents=True, exist_ok=True)
    assignments = allocate_regimes(
        count,
        regimes,
        mode=(
            "balanced"
            if args.allocation_mode == "weighted"
            else args.allocation_mode
        ),
        axis_values={
            "num_classes": grids["num_classes_grid"],
            "margin_level": grids["margin_grid"],
        },
    )
    records: list[dict] = []
    for dataset_idx, allocated in enumerate(assignments):
        global_idx = counter.next()
        regime = str(allocated["regime"])
        assignment = _default_assignment(
            args,
            np.random.default_rng(_task_seed(args, global_idx, suite_family)),
            regime=regime,
            K=int(allocated.get("num_classes", 2)),
            margin=str(allocated.get("margin_level", "medium")),
        )
        out_path = family_dir / f"dataset_{dataset_idx:04d}.parquet"
        ds_split, n_rows = _generate_one_task(
            args,
            grids,
            assignment,
            global_idx,
            suite_id,
            suite_family,
            dataset_idx,
            out_path,
        )
        records.append(
            _make_dataset_record(
                out_path,
                suite_dir,
                dataset_idx,
                global_idx,
                suite_family,
                regime,
                ds_split,
                n_rows,
                extra={
                    "hidden_holdout_suite_id": (
                        args.hidden_holdout_suite_id
                        if suite_family == "hidden_holdout"
                        else None
                    )
                },
                task_seed=_task_seed(args, global_idx, suite_family),
            )
        )
    return records


def _generate_eval_only_unseen(
    suite_dir, args, suite_id, counter, grids
) -> list[dict]:
    return _generate_distribution_family(
        suite_dir,
        args,
        suite_id,
        counter,
        grids,
        suite_family="eval_only_unseen",
        regimes=list(CLASSIFICATION_EVAL_ONLY_REGIMES),
        count=int(args.eval_only_n_datasets or args.n_datasets),
    )


def _generate_hidden_holdout(
    suite_dir, args, suite_id, counter, grids
) -> list[dict]:
    return _generate_distribution_family(
        suite_dir,
        args,
        suite_id,
        counter,
        grids,
        suite_family="hidden_holdout",
        regimes=list(CLASSIFICATION_EVAL_ONLY_REGIMES),
        count=int(args.hidden_holdout_n_datasets or args.n_datasets),
    )


def _generate_stress(
    suite_dir, args, suite_id, counter, grids
) -> list[dict]:
    regimes = [
        regime
        for regime in REGIME_WEIGHTS["linear_classification_stress"]
        if args.allow_underdetermined
        or regime != "J_low_n_high_p_classification"
    ]
    return _generate_distribution_family(
        suite_dir,
        args,
        suite_id,
        counter,
        grids,
        suite_family="stress",
        regimes=regimes,
        count=args.n_datasets,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_FAMILY_GENERATORS = {
    "primary": _generate_primary,
    "feature_noise": _generate_feature_noise,
    "label_noise": _generate_label_noise,
    "training_size": _generate_training_size,
    "class_imbalance": _generate_class_imbalance,
    "margin": _generate_margin,
    "num_classes": _generate_num_classes,
    "ood": _generate_ood,
    "eval_only_unseen": _generate_eval_only_unseen,
    "hidden_holdout": _generate_hidden_holdout,
    "stress": _generate_stress,
}


# ---------------------------------------------------------------------------
# Count accumulators
# ---------------------------------------------------------------------------

def _accumulate_counts(realized: dict, records: list[dict]) -> None:
    for rec in records:
        for key, field in [
            ("regime", "regime"),
            ("K", "K"),
            ("imbalance", "imbalance_type"),
            ("margin", "margin_level"),
            ("label_noise", "label_noise_rate"),
            ("feature_noise", "feature_noise_level"),
        ]:
            val = str(rec.get(field, ""))
            realized[key][val] = realized[key].get(val, 0) + 1


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def _get_git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _build_manifest(
    args: argparse.Namespace,
    suite_id: str,
    suite_dir: Path,
    enabled: list[str],
    all_records: list[dict],
    realized_counts: dict,
) -> dict:
    profile_weights = dict(REGIME_WEIGHTS[args.profile])
    # Effective weights from primary records
    total = len(all_records)
    regime_totals: dict[str, int] = {}
    for r in all_records:
        rr = r["regime"]
        regime_totals[rr] = regime_totals.get(rr, 0) + 1
    effective_weights = {k: v / total if total > 0 else 0.0
                         for k, v in regime_totals.items()}

    # Checksums
    file_checksums = {r["filepath"]: r["file_checksum_sha256"] for r in all_records}
    # Output checksum: hash all file checksums in global_idx order
    sorted_records = sorted(all_records, key=lambda r: r["global_idx"])
    combined = "".join(r["file_checksum_sha256"] for r in sorted_records)
    output_checksum = hashlib.sha256(combined.encode()).hexdigest()

    stage_paths: dict[str, str] = {}
    for fam in enabled:
        stage_paths[fam] = (
            f"{STAGE_PREFIX}/{OUTPUT_SUBDIR}/{suite_id}/{fam}/"
        )

    grids = _build_grids(args)

    # Realized suite family counts
    family_counts = {}
    for fam in enabled:
        family_counts[fam] = sum(1 for r in all_records if r["suite_family"] == fam)

    normal_seeds = {
        int(r["task_seed"]) for r in all_records
        if r["suite_family"] != "hidden_holdout"
    }
    hidden_seeds = {
        int(r["task_seed"]) for r in all_records
        if r["suite_family"] == "hidden_holdout"
    }
    coverage_minimums: dict[str, int] = {}
    if args.coverage_config:
        config_path = Path(args.coverage_config)
        config = json.loads(
            config_path.read_text() if config_path.exists() else args.coverage_config
        )
        coverage_minimums = {str(k): int(v) for k, v in config.items()}
    coverage_counts = {
        family: sum(r["suite_family"] == family for r in all_records)
        for family in enabled
    }
    coverage_missing = {
        family: count
        for family, count in coverage_counts.items()
        if count < coverage_minimums.get("suite_family", 1)
    }
    seed_audit = {
        "all_dataset_seeds_unique": len(all_records)
        == len({int(r["task_seed"]) for r in all_records}),
        "hidden_normal_seed_overlap": sorted(hidden_seeds & normal_seeds),
        "hidden_holdout_suite_id": args.hidden_holdout_suite_id,
    }
    coverage_audit = {
        "ok": not coverage_missing,
        "counts": coverage_counts,
        "missing": coverage_missing,
    }
    # Build richer audit sections
    difficulty_mix_dict = {"core": 0.65, "robust": 0.30, "stress": 0.05}
    if hasattr(args, "difficulty_mix") and args.difficulty_mix:
        try:
            difficulty_mix_dict = parse_difficulty_mix(args.difficulty_mix)
        except Exception:
            pass
    rich_memory_audit = build_memory_audit(
        all_records,
        memory_guard_bytes=getattr(args, "memory_guard_bytes", DEFAULT_GPU_GUARD_BYTES),
        default_context_size=DEFAULT_CONTEXT_SIZE,
        default_test_batch_size=DEFAULT_TEST_BATCH_SIZE,
        default_feature_cap=DEFAULT_FEATURE_CAP,
    )
    rich_difficulty_audit = build_difficulty_audit(
        all_records,
        curriculum_policy=getattr(args, "curriculum_policy", "core_first"),
        configured_mix=difficulty_mix_dict,
    )
    return {
        "suite_id": suite_id,
        "task_family": "linear_classification",
        "task_objective": "inductive_classification",
        "schema_version": CLASSIFICATION_EVAL_SCHEMA_VERSION,
        "profile": args.profile,
        "base_seed": args.base_seed,
        "hidden_holdout_base_seed": args.hidden_holdout_base_seed,
        "hidden_holdout_suite_id": args.hidden_holdout_suite_id,
        "allocation_mode": args.allocation_mode,
        "strict_coverage": args.strict_coverage,
        "n_datasets": total,
        "n_datasets_primary": args.n_datasets,
        "n_datasets_per_sweep": args.n_datasets_per_sweep,
        "enabled_suite_families": enabled,
        "configured_profile_weights": profile_weights,
        "effective_profile_weights": effective_weights,
        "realized_regime_counts": realized_counts.get("regime", {}),
        "realized_suite_family_counts": family_counts,
        "realized_K_counts": realized_counts.get("K", {}),
        "realized_imbalance_counts": realized_counts.get("imbalance", {}),
        "realized_margin_counts": realized_counts.get("margin", {}),
        "realized_label_noise_counts": realized_counts.get("label_noise", {}),
        "realized_feature_noise_counts": realized_counts.get("feature_noise", {}),
        "datasets": all_records,
        "file_checksums": file_checksums,
        "output_checksum": output_checksum,
        "stage_paths": stage_paths,
        "grid_metadata": {
            "n_grid": grids["n_grid"],
            "p_signal_grid": grids["p_signal_grid"],
            "p_noise_grid": grids["p_noise_grid"],
            "feature_noise_grid": grids["feature_noise_grid"],
            "feature_noise_amplitude_grid": grids.get("feature_noise_amplitude_grid", grids["feature_noise_grid"]),
            "num_classes_grid": grids["num_classes_grid"],
            "temperature_grid": grids["temperature_grid"],
            "label_noise_grid": grids["label_noise_grid"],
            "class_imbalance_grid": grids["class_imbalance_grid"],
            "margin_grid": grids["margin_grid"],
            "coefficient_scale_grid": grids["coefficient_scale_grid"],
            "intercept_scale_grid": grids["intercept_scale_grid"],
        },
        "generation_flags": {
            "store_class_params": args.store_class_params,
            "store_class_teacher_preds": args.store_class_teacher_preds,
            "require_class_teachers": args.require_class_teachers,
            "allow_underdetermined": args.allow_underdetermined,
        },
        "seed_audit": seed_audit,
        "train_eval_alignment_audit": {
            "eval_only_regimes_in_training": [],
            "dataset_seed_overlap": [],
            "fingerprint_overlap": [],
        },
        "coverage_audit": coverage_audit,
        "memory_audit": rich_memory_audit,
        "difficulty_audit": rich_difficulty_audit,
        "alignment_audit": {"emitted": False},
        "generation_controls": {
            "task_family": "linear_classification",
            "allocation_mode": getattr(args, "allocation_mode", "weighted_quota"),
            "curriculum_policy": getattr(args, "curriculum_policy", "core_first"),
            "difficulty_mix": getattr(args, "difficulty_mix", "core=0.65,robust=0.30,stress=0.05"),
            "memory_guard_bytes": getattr(args, "memory_guard_bytes", DEFAULT_GPU_GUARD_BYTES),
            "min_regime_count": getattr(args, "min_regime_count", 10),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_elapsed_seconds": 0.0,  # filled by caller
        "git_revision": _get_git_revision(),
        "source_git_revision": _get_git_revision(),
        "python_version": sys.version,
        # F1: per-split K coverage audit
        "k_coverage_audit": validate_per_split_k_coverage(
            all_records,
            required_k=sorted(set(grids.get("num_classes_grid", [2, 3, 5, 10]))),
            strict_coverage=getattr(args, "strict_coverage", False),
        ),
        # F10: completion gates
        "completion_gates": {
            "binary_datasets_present": realized_counts.get("K", {}).get(2, 0) > 0,
            "multiclass_datasets_present": any(
                realized_counts.get("K", {}).get(k, 0) > 0 for k in [3, 5, 10]
            ),
            "all_suite_families_present": set(enabled) <= set(family_counts),
        },
    }


# ---------------------------------------------------------------------------
# SnowSQL output
# ---------------------------------------------------------------------------

def _print_snowsql_commands(
    suite_id: str,
    suite_dir: Path,
    enabled: list[str],
    manifest_path: Path,
) -> None:
    flat_families = {"primary"}
    print("=" * 80)
    print("SnowSQL commands to upload this evaluation suite:")
    print("=" * 80)

    for fam in enabled:
        fam_dir = suite_dir / fam
        stage_base = f"{STAGE_PREFIX}/{OUTPUT_SUBDIR}/{suite_id}/{fam}"
        print(f"\n-- Suite family: {fam}")
        if fam in flat_families:
            abs_fam = fam_dir.resolve().as_posix()
            print(f"REMOVE {stage_base}/;")
            print(f"PUT file://{abs_fam}/*.parquet")
            print(f"    {stage_base}/")
            print(f"    AUTO_COMPRESS=FALSE OVERWRITE=TRUE;")
        else:
            # One PUT per subdirectory
            subdirs = sorted(fam_dir.iterdir()) if fam_dir.exists() else []
            for sub in subdirs:
                if sub.is_dir():
                    abs_sub = sub.resolve().as_posix()
                    stage_sub = f"{stage_base}/{sub.name}"
                    print(f"REMOVE {stage_sub}/;")
                    print(f"PUT file://{abs_sub}/*.parquet")
                    print(f"    {stage_sub}/")
                    print(f"    AUTO_COMPRESS=FALSE OVERWRITE=TRUE;")

    # Manifest
    abs_manifest = manifest_path.resolve().as_posix()
    stage_root = f"{STAGE_PREFIX}/{OUTPUT_SUBDIR}/{suite_id}"
    print(f"\n-- Manifest")
    print(f"PUT file://{abs_manifest}")
    print(f"    {stage_root}/")
    print(f"    AUTO_COMPRESS=FALSE OVERWRITE=TRUE;")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Dry-run summary
# ---------------------------------------------------------------------------

def _print_dry_run_summary(
    args: argparse.Namespace,
    suite_id: str,
    enabled: list[str],
) -> None:
    print("DRY RUN — no files will be written.")
    print(f"  suite_id    : {suite_id}")
    print(f"  output_root : {args.output_root}")
    print(f"  profile     : {args.profile}")
    print(f"  base_seed   : {args.base_seed}")
    print(f"  enabled     : {enabled}")
    print(f"  n_datasets  : {args.n_datasets} (primary)")
    print(f"  n_per_sweep : {args.n_datasets_per_sweep}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

_MIXED_CLS_EVAL_SEED_MAGIC = 0x4D49584C  # "MIXL"


def _run_mixed_classification(args: argparse.Namespace) -> None:
    """Generate mixed-categorical classification evaluation datasets.

    Uses allocate_mixed_classification_tasks + build_mixed_classification_dataset.
    Output goes to mixed_classification_prepared/{suite_id}/{regime}/.
    Stage path: @EVALUATION_DATASET_STAGE/mixed_classification_prepared/...
    """
    suite_id = args.suite_id or _make_suite_id()
    root = Path(args.output_root).parent / "mixed_classification_prepared"
    suite_dir = root / suite_id

    if args.dry_run:
        _print_dry_run_summary(args, suite_id, ["mixed_categorical_classification"])
        return

    suite_dir.mkdir(parents=True, exist_ok=True)

    assignments, alloc_audit = allocate_mixed_classification_tasks(
        args.n_datasets,
        args.profile,
        args.base_seed,
        num_classes_grid=args.num_classes_grid,
        label_noise_grid=list(args.label_noise_grid),
        allow_underdetermined=args.allow_underdetermined,
    )

    records: list[dict] = []
    for idx, assignment in enumerate(assignments):
        task_seed = int(
            np.random.SeedSequence(
                [args.base_seed, idx, _MIXED_CLS_EVAL_SEED_MAGIC]
            ).generate_state(1)[0]
        )
        rng = np.random.default_rng(task_seed)
        regime = str(assignment["prior_regime"])
        ds = build_mixed_classification_dataset(rng, regime, assignment)
        validate_mixed_classification_dataset(ds)

        n = int(ds["n"])
        n_train = int(0.8 * n)
        n_test = max(1, n - n_train)
        X_cat_train = ds["X_cat"][:n_train]
        X_cat_test = ds["X_cat"][n_train:]
        unknown_mask_test = mark_unseen_query_categories(X_cat_train, X_cat_test)

        ds_split = {
            "X_num_train":           ds["X_num"][:n_train],
            "X_num_test":            ds["X_num"][n_train:],
            "y_train":               ds["y"][:n_train],
            "y_test":                ds["y"][n_train:],
            "X_cat_train":           X_cat_train,
            "X_cat_test":            X_cat_test,
            "cat_missing_mask_train":ds["missing_mask"][:n_train],
            "cat_missing_mask_test": ds["missing_mask"][n_train:],
            "cat_unknown_mask_test": unknown_mask_test,
            "categorical_cardinalities": ds["cardinalities"],
            "n": n,
            "p_num": ds["p_num"],
            "p_cat": ds["p_cat"],
            "num_classes": int(ds["num_classes"]),
            "n_train": n_train,
            "n_test": n_test,
            "prior_regime": ds["prior_regime"],
            "schema_version": ds["schema_version"],
            "training_data_family": ds["training_data_family"],
            "task_family": ds["task_family"],
            "task_objective": ds["task_objective"],
            "imbalance_type":        ds.get("imbalance_type", "balanced"),
            "temperature":           ds.get("temperature", 1.0),
            "label_noise_rate":      ds.get("label_noise_rate", 0.0),
            "W_num":                 ds["W_num"],
            "b":                     ds["b"],
            "cat_class_effects":     ds["cat_class_effects"],
            "numeric_support_mask":  ds["numeric_support_mask"],
            "cat_support_mask":      ds["cat_support_mask"],
        }

        regime_dir = suite_dir / regime
        regime_dir.mkdir(parents=True, exist_ok=True)
        filename = f"dataset_{idx:05d}.parquet"
        filepath = regime_dir / filename
        write_parquet_mixed_classification_dgp(ds_split, str(filepath))

        stage_path = (
            f"@EVALUATION_DATASET_STAGE/mixed_classification_prepared/"
            f"{suite_id}/{regime}/{filename}"
        )
        records.append({
            "dataset_id":           idx,
            "suite_id":             suite_id,
            "suite_family":         "primary",
            "dataset_seed":         task_seed,
            "stage_path":           stage_path,
            "prior_regime":         regime,
            "n_total":              n,
            "n_train_default":      n_train,
            "n_holdout_default":    n_test,
            "p_num":                int(ds["p_num"]),
            "p_cat":                int(ds["p_cat"]),
            "p_num_signal":         int(ds["p_num_signal"]),
            "p_cat_signal":         int(ds["p_cat_signal"]),
            "p_total":              int(ds["p_num"]) + int(ds["p_cat"]),
            "num_classes":          int(ds["num_classes"]),
            "cardinalities":        ds["cardinalities"],
            "missing_rate":         float(assignment.get("missing_rate", 0.0)),
            "task_family":          ds["task_family"],
            "training_data_family": ds["training_data_family"],
            "schema_version":       ds["schema_version"],
        })

        if (idx + 1) % args.log_every == 0 or (idx + 1) == len(assignments):
            print(f"  [{idx + 1:4d}/{len(assignments)}] {filepath.name}")

    manifest = {
        "suite_id":          suite_id,
        "task_family":       "mixed_categorical_classification",
        "profile":           args.profile,
        "base_seed":         args.base_seed,
        "n_datasets":        len(records),
        "mixed_categorical": True,
        "allocation_audit":  alloc_audit,
        "datasets":          records,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = suite_dir / "mixed_classification_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"\nDone. {len(records)} mixed-categorical classification datasets written to:\n"
        f"  {suite_dir}\nManifest: {manifest_path}"
    )

    # SnowSQL PUT commands
    abs_manifest = manifest_path.resolve().as_posix()
    stage_root = f"@EVALUATION_DATASET_STAGE/mixed_classification_prepared/{suite_id}"
    print("\n--- SnowSQL PUT commands ---")
    written_regimes = sorted({r["prior_regime"] for r in records})
    for regime in written_regimes:
        regime_dir_abs = (suite_dir / regime).resolve().as_posix()
        print(
            f"PUT file://{regime_dir_abs}/*.parquet "
            f"{stage_root}/{regime}/ "
            f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
        )
    print(
        f"PUT file://{abs_manifest} "
        f"{stage_root}/ "
        f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    )


def run_linear_classification_generation(args: argparse.Namespace) -> None:
    if getattr(args, "mixed_categorical", False) or SYNCLS_IS_MIXED_CATEGORICAL:
        _run_mixed_classification(args)
        return
    suite_id = args.suite_id or _make_suite_id()
    suite_dir = Path(args.output_root).resolve() / suite_id
    enabled = _get_enabled_families(args)

    _check_output_dir(suite_dir, args.overwrite, args.dry_run)

    if args.dry_run:
        _print_dry_run_summary(args, suite_id, enabled)
        return

    suite_dir.mkdir(parents=True, exist_ok=True)
    for fam in enabled:
        (suite_dir / fam).mkdir(parents=True, exist_ok=True)

    grids = _build_grids(args)
    counter = _GlobalIndexCounter()
    all_records: list[dict] = []
    realized_counts: dict[str, dict] = {
        k: {} for k in ["regime", "K", "imbalance", "margin", "label_noise", "feature_noise"]
    }

    t0 = time.monotonic()
    for fam in enabled:
        print(f"\n--- Generating suite family: {fam} ---")
        gen_fn = _FAMILY_GENERATORS[fam]
        records = gen_fn(suite_dir, args, suite_id, counter, grids)
        all_records.extend(records)
        _accumulate_counts(realized_counts, records)
        print(f"  {fam}: {len(records)} datasets")

    elapsed = time.monotonic() - t0

    manifest = _build_manifest(
        args, suite_id, suite_dir, enabled, all_records, realized_counts
    )
    manifest["generation_elapsed_seconds"] = round(elapsed, 2)
    if args.strict_coverage:
        failures = []
        if not manifest["coverage_audit"]["ok"]:
            failures.append(
                f"coverage={manifest['coverage_audit']['missing']}"
            )
        if not manifest["seed_audit"]["all_dataset_seeds_unique"]:
            failures.append("dataset seeds are not unique")
        if manifest["seed_audit"]["hidden_normal_seed_overlap"]:
            failures.append("hidden and normal seed namespaces overlap")
        if failures:
            raise ValueError("strict suite validation failed: " + "; ".join(failures))

    manifest_path = suite_dir / "synthetic_classification_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. {len(all_records)} datasets written to:")
    print(f"  {suite_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"output_checksum: {manifest['output_checksum']}")

    _print_snowsql_commands(suite_id, suite_dir, enabled, manifest_path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_linear_classification_generation(args)


if __name__ == "__main__":
    main()
