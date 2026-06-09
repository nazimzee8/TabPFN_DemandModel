"""
generate_dgp.py

Generate meta-datasets from a synthetic demand data-generating process (DGP).

Usage:
    # Legacy (original behaviour, bit-identical):
    python src/generate_dgp.py --n_datasets 1000 --profile legacy --base_seed 42

    # Extended linear regimes (recommended):
    python src/generate_dgp.py \
        --n_datasets 1000 \
        --out_dir data/linear_stat_aware_train \
        --profile linear_stat_aware \
        --base_seed 42 \
        --store_teacher_preds \
        --store_beta
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dgp_helpers import (
    ALL_PROFILES,
    CLASSIFICATION_PROFILES,
    REGRESSION_PROFILES,
    REGIME_WEIGHTS,
    allocate_classification_tasks,
    allocate_regression_tasks,
    allocate_mixed_regression_tasks,
    allocate_mixed_classification_tasks,
    build_dataset_from_regime,
    build_mixed_regression_dataset,
    build_mixed_classification_dataset,
    compute_classification_diagnostics,
    compute_classification_teacher,
    compute_linear_diagnostics,
    compute_linear_teacher_targets,
    generate_classification_dataset,
    mark_unseen_query_categories,
    parse_float_grid,
    parse_int_grid,
    parse_string_grid,
    sample_n_p,
    split_classification_dataset,
    validate_classification_dataset,
    validate_dataset,
    validate_mixed_regression_dataset,
    validate_mixed_classification_dataset,
    write_classification_parquet,
    write_parquet_dgp,
    write_parquet_mixed_regression_dgp,
    write_parquet_mixed_classification_dgp,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task_family",
        choices=(
            "linear_regression", "linear_classification",
            "linear_regression_mixed_categorical",
            "linear_classification_mixed_categorical",
        ),
        default="linear_regression",
        help="Task family (default: linear_regression).",
    )
    parser.add_argument(
        "--n_datasets", type=int, default=1000,
        help="Total number of datasets to generate (default: 1000).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Root output directory (regression default: data/).",
    )
    parser.add_argument(
        "--profile", type=str, default=None, choices=ALL_PROFILES,
        help="DGP profile. The default is resolved from --task_family.",
    )
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Base RNG seed (default: 42).",
    )
    parser.add_argument(
        "--n_grid", type=str, default="32,64,128,256,512,1024",
        help="Comma-separated grid of n values for non-legacy profiles.",
    )
    parser.add_argument(
        "--p_signal_grid", type=str, default="4,8,16,32,64",
        help="Comma-separated grid of p_signal values.",
    )
    parser.add_argument(
        "--p_noise_grid", type=str, default="0,8,24,56,120",
        help="Comma-separated grid of p_noise values.",
    )
    parser.add_argument(
        "--active_s_grid", type=str, default="2,4,8,16,32",
        help="Comma-separated grid of active_s values.",
    )
    parser.add_argument(
        "--rho_grid", type=str, default="0.0,0.3,0.6,0.9",
        help="Comma-separated grid of correlation rho values.",
    )
    parser.add_argument(
        "--feature_noise_grid", type=str, default="0.0,0.05,0.10,0.25",
        help="Comma-separated grid of feature noise values.",
    )
    parser.add_argument(
        "--allow_underdetermined", action="store_true", default=False,
        help="Allow the J low-n/high-p regime.",
    )
    parser.add_argument(
        "--allocation_mode",
        type=str,
        default="weighted_quota",
        choices=["weighted", "weighted_quota", "balanced", "balanced_cartesian", "curriculum"],
        help="Regime allocation mode (default: weighted_quota).",
    )
    parser.add_argument(
        "--min_regime_count",
        type=int,
        default=10,
        help="Minimum datasets per regime in weighted_quota/balanced modes.",
    )
    parser.add_argument(
        "--strict_coverage",
        action="store_true",
        default=False,
        help="Raise if any regime falls below min_regime_count.",
    )
    parser.add_argument(
        "--curriculum_policy",
        type=str,
        default="core_first",
        choices=["none", "core_first", "balanced_tiers", "stress_eval_only"],
        help="Curriculum ordering policy (default: core_first).",
    )
    parser.add_argument(
        "--difficulty_mix",
        type=str,
        default="core=0.65,robust=0.30,stress=0.05",
        help="Difficulty tier mix (default: core=0.65,robust=0.30,stress=0.05).",
    )
    parser.add_argument(
        "--max_memory_risk",
        type=str,
        default="high",
        choices=["low", "medium", "high", "exceeds_default_guard"],
        help="Maximum allowed memory risk bucket (default: high).",
    )
    parser.add_argument(
        "--memory_guard_bytes",
        type=int,
        default=268_435_456,
        help="GPU memory guard in bytes (default: 256 MB).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic DGP meta-datasets.")
    _add_common_arguments(parser)
    parser.add_argument(
        "--target_noise_grid", type=str, default="0.25,0.5,1.0,2.0",
        help="Comma-separated grid of target noise scale values.",
    )
    parser.add_argument(
        "--num_classes_grid", type=str, default="2,3,5,10",
        help="Classification class-count grid.",
    )
    parser.add_argument(
        "--temperature_grid", type=str, default="0.5,1.0,2.0,4.0",
        help="Classification temperature grid.",
    )
    parser.add_argument(
        "--label_noise_grid", type=str, default="0.0,0.02,0.05,0.10",
        help="Classification symmetric label-noise grid.",
    )
    parser.add_argument(
        "--class_imbalance_grid",
        type=str,
        default="balanced,mild,moderate,severe",
        help="Classification imbalance-level grid.",
    )
    parser.add_argument(
        "--margin_grid", type=str, default="low,medium,high",
        help="Classification normalized-margin buckets.",
    )
    parser.add_argument(
        "--coefficient_scale_grid", type=str, default="0.5,1.0,2.0",
        help="Classification coefficient scale grid.",
    )
    parser.add_argument(
        "--intercept_scale_grid", type=str, default="0.0,0.5,1.0,2.0",
        help="Classification intercept initialization scale grid.",
    )
    parser.add_argument(
        "--store_teacher_preds", action="store_true", default=False,
        help="Store ridge predictions for each lambda in parquet.",
    )
    parser.add_argument(
        "--store_beta", action="store_true", default=False,
        help="Store beta and support_mask in parquet.",
    )
    parser.add_argument(
        "--store_linear_moments", action="store_true", default=False,
        help="(Unused in DGP writer; accepted for CLI parity with eval script.)",
    )
    parser.add_argument(
        "--store_class_params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store true classification parameters (default: enabled).",
    )
    parser.add_argument(
        "--store_class_teacher_preds", action="store_true", default=False,
        help="Fit and store L2 logistic teacher outputs.",
    )
    parser.add_argument(
        "--require_class_teachers", action="store_true", default=False,
        help="Fail generation when a requested classification teacher is unavailable.",
    )
    # Mixed-categorical arguments
    mixed_group = parser.add_argument_group("mixed-categorical")
    mixed_group.add_argument(
        "--p_num_signal_grid", type=str, default="2,4,8,16,32",
        help="Grid of numeric signal feature counts.",
    )
    mixed_group.add_argument(
        "--p_num_noise_grid", type=str, default="0,8,24,56",
        help="Grid of numeric noise feature counts.",
    )
    mixed_group.add_argument(
        "--p_cat_signal_grid", type=str, default="1,2,4,8",
        help="Grid of categorical signal feature counts.",
    )
    mixed_group.add_argument(
        "--p_cat_noise_grid", type=str, default="0,2,4,8",
        help="Grid of categorical noise feature counts.",
    )
    mixed_group.add_argument(
        "--cat_cardinality_grid", type=str, default="2,3,5,10,20,50",
        help="Grid of cardinality values per categorical feature.",
    )
    mixed_group.add_argument(
        "--cat_effect_scale_grid", type=str, default="0.25,0.5,1.0,2.0",
        help="Grid of categorical effect scale values.",
    )
    mixed_group.add_argument(
        "--cat_missing_rate_grid", type=str, default="0.0,0.01,0.05,0.10",
        help="Grid of categorical missing rates.",
    )
    mixed_group.add_argument(
        "--cat_imbalance_grid", type=str, default="balanced,mild,moderate,severe",
        help="Grid of categorical imbalance types.",
    )
    mixed_group.add_argument(
        "--store_cat_params", action="store_true", default=True,
        help="Store categorical parameters in parquet (default: True).",
    )
    return parser


def _split_directories(out_dir: str, subdir: str = "") -> tuple[str, str, str]:
    """Return (train_dir, val_dir, test_dir), creating them under out_dir[/subdir]/.

    When *subdir* is non-empty the layout is:
        out_dir/<subdir>/train/
        out_dir/<subdir>/val/
        out_dir/<subdir>/test/
    This matches the Snowflake stage convention:
        @META_REGRESSION_DATASET_STAGE/numeric/{split}/  (numeric families)
        @META_REGRESSION_DATASET_STAGE/mixed/{split}/   (mixed-categorical families)
    """
    base = os.path.join(out_dir, subdir) if subdir else out_dir
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
    test_dir  = os.path.join(base, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)
    return train_dir, val_dir, test_dir


def _output_path(
    idx: int,
    n_train_split: int,
    n_val_split: int,
    directories: tuple[str, str, str],
) -> tuple[str, str]:
    train_dir, val_dir, test_dir = directories
    if idx < n_train_split:
        return train_dir, f"dataset_{idx:04d}.parquet"
    if idx < n_train_split + n_val_split:
        local_idx = idx - n_train_split
        return val_dir, f"dataset_{local_idx:04d}.parquet"
    local_idx = idx - n_train_split - n_val_split
    return test_dir, f"dataset_{local_idx:04d}.parquet"


def _run_regression(args: argparse.Namespace) -> None:
    n_datasets = args.n_datasets
    out_dir = args.out_dir or "data/"
    profile = args.profile
    base_seed = args.base_seed

    n_grid = parse_int_grid(args.n_grid)
    p_signal_grid = parse_int_grid(args.p_signal_grid)
    p_noise_grid = parse_int_grid(args.p_noise_grid)
    active_s_grid = parse_int_grid(args.active_s_grid)
    rho_grid = parse_float_grid(args.rho_grid)
    target_noise_grid = parse_float_grid(args.target_noise_grid)
    feature_noise_grid = parse_float_grid(args.feature_noise_grid)

    n_train_split = int(0.8 * n_datasets)
    n_val_split   = int(0.1 * n_datasets)
    n_test_split  = n_datasets - n_train_split - n_val_split

    directories = _split_directories(out_dir, subdir="numeric")
    train_dir, val_dir, test_dir = directories

    assignments, allocation_audit = allocate_regression_tasks(
        n_datasets,
        profile,
        base_seed,
        allow_underdetermined=args.allow_underdetermined,
        n_grid=n_grid,
        p_signal_grid=p_signal_grid,
        p_noise_grid=p_noise_grid,
        active_s_grid=active_s_grid,
        rho_grid=rho_grid,
        target_noise_grid=target_noise_grid,
        feature_noise_grid=feature_noise_grid,
        allocation_mode=getattr(args, "allocation_mode", "weighted_quota"),
        min_regime_count=getattr(args, "min_regime_count", 10),
        strict_coverage=getattr(args, "strict_coverage", False),
    )
    rng = np.random.default_rng(seed=base_seed)

    regimes = list(dict(REGIME_WEIGHTS[profile]).keys())
    print(f"Profile: {profile}")
    print(f"Generating {n_datasets} datasets -> {train_dir}, {val_dir}, {test_dir}")
    print(f"  train: {n_train_split}  val: {n_val_split}  test: {n_test_split}")
    print(f"  base_seed={base_seed}  regimes={regimes}")

    for idx, assignment in enumerate(assignments):
        regime = assignment["prior_regime"]
        params = {
            k: assignment[k]
            for k in ["n", "p_signal", "p_noise", "active_s", "rho",
                      "target_noise_scale", "feature_noise_level", "covariance_type"]
            if k in assignment
        }
        ds = build_dataset_from_regime(rng, regime, profile, params)
        validate_dataset(ds)

        n = ds["n_total"]
        n_train = int(0.8 * n)
        n_test = max(1, n - n_train)
        if n_test < 1:
            n_train -= 1
            n_test = 1

        ds_split = {
            "X_train":      ds["X"][:n_train],
            "y_train":      ds["y"][:n_train],
            "X_test":       ds["X"][n_train:],
            "y_test":       ds["y"][n_train:],          # NEW
            "betaX_test":   ds["betaX"][n_train:],
            "betaX_train":  ds["betaX"][:n_train],      # NEW
            "n":            n,
            "p":            ds["p_total"],
            "n_train":      n_train,
            "n_test":       n_test,
            "prior_regime": ds["prior_regime"],
            # Propagate new metadata fields
            "p_signal":             ds["p_signal"],
            "p_noise":              ds["p_noise"],
            "p_total":              ds["p_total"],
            "active_s":             ds["active_s"],
            "sparsity_ratio":       ds["sparsity_ratio"],
            "covariance_type":      ds["covariance_type"],
            "rho":                  ds["rho"],
            "target_noise_scale":   ds["target_noise_scale"],
            "feature_noise_level":  ds["feature_noise_level"],
            "beta":                 ds["beta"],
            "support_mask":         ds["support_mask"],
        }

        teacher_results = compute_linear_teacher_targets(
            ds_split["X_train"], ds_split["y_train"],
            ds_split["X_test"],  ds_split["y_test"],
        )
        diagnostics = compute_linear_diagnostics(ds["X"], ds["beta"], ds["p_signal"])

        split_dir, filename = _output_path(
            idx, n_train_split, n_val_split, directories
        )

        filepath = os.path.join(split_dir, filename)
        write_parquet_dgp(
            ds_split, filepath,
            store_beta=args.store_beta,
            store_teacher_preds=args.store_teacher_preds,
            teacher_results=teacher_results,
            diagnostics=diagnostics,
        )

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1:4d}/{n_datasets}] written {filename} to {split_dir}")

    print("Done.")


def _validate_classification_root(out_dir: str) -> Path:
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "linear_classification requires an explicit empty --out_dir; "
            f"{root} is not empty"
        )
    return root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_classification_manifest(
    root: Path,
    args: argparse.Namespace,
    generation_command: list[str],
    grids: dict[str, list],
    allocation_audit: dict,
    records: list[dict],
    split_counts: dict[str, int],
) -> None:
    files = sorted(root.rglob("*.parquet"))
    checksums = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in files
    }
    aggregate = hashlib.sha256()
    for relative_path, checksum in checksums.items():
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(checksum.encode("ascii"))
        aggregate.update(b"\n")
    output_checksum = aggregate.hexdigest()

    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in records:
            key = str(record[field])
            result[key] = result.get(key, 0) + 1
        return result

    suffix_input = (
        f"linear_classification|{args.profile}|linear_classification_v1|"
        f"{args.base_seed}|{output_checksum}"
    )
    suffix = hashlib.sha256(suffix_input.encode("utf-8")).hexdigest()[:12]
    manifest = {
        "suite_id": (
            f"linear_classification-{args.profile}-"
            f"linear_classification_v1-{args.base_seed}-{suffix}"
        ),
        "task_family": "linear_classification",
        "task_objective": "inductive_classification",
        "profile": args.profile,
        "base_seed": args.base_seed,
        "n_datasets": args.n_datasets,
        "outer_split_counts": split_counts,
        "generation_command": generation_command,
        "git_revision": _git_revision(),
        "schema_version": "linear_classification_v1",
        "grid_values": grids,
        "profile_weights": {
            regime: count / args.n_datasets
            for regime, count in allocation_audit["regime_counts"].items()
        },
        "configured_profile_weights": REGIME_WEIGHTS[args.profile],
        "allow_underdetermined": args.allow_underdetermined,
        "store_class_params": args.store_class_params,
        "store_class_teacher_preds": args.store_class_teacher_preds,
        "require_class_teachers": args.require_class_teachers,
        "allocation": allocation_audit,
        "realized_regime_counts": counts("classification_regime"),
        "realized_K_counts": counts("num_classes"),
        "realized_imbalance_counts": counts("class_imbalance_type"),
        "realized_temperature_counts": counts("temperature"),
        "realized_margin_counts": counts("margin_level"),
        "realized_label_noise_counts": counts("label_noise_rate"),
        "file_checksums": checksums,
        "output_checksum": output_checksum,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_classification(
    args: argparse.Namespace,
    generation_command: list[str],
) -> None:
    if args.out_dir is None:
        raise ValueError(
            "linear_classification requires an explicit classification-specific --out_dir"
        )
    if args.n_datasets < 1:
        raise ValueError("--n_datasets must be at least 1")
    root = _validate_classification_root(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    directories = _split_directories(str(root), subdir="numeric")

    grids = {
        "n_grid": parse_int_grid(args.n_grid),
        "p_signal_grid": parse_int_grid(args.p_signal_grid),
        "p_noise_grid": parse_int_grid(args.p_noise_grid),
        "active_s_grid": parse_int_grid(args.active_s_grid),
        "rho_grid": parse_float_grid(args.rho_grid),
        "feature_noise_grid": parse_float_grid(args.feature_noise_grid),
        "num_classes_grid": parse_int_grid(args.num_classes_grid),
        "temperature_grid": parse_float_grid(args.temperature_grid),
        "label_noise_grid": parse_float_grid(args.label_noise_grid),
        "class_imbalance_grid": parse_string_grid(args.class_imbalance_grid),
        "margin_grid": parse_string_grid(args.margin_grid),
        "coefficient_scale_grid": parse_float_grid(args.coefficient_scale_grid),
        "intercept_scale_grid": parse_float_grid(args.intercept_scale_grid),
    }
    if not grids["num_classes_grid"] or not set(
        grids["num_classes_grid"]
    ).issubset({2, 3, 5, 10}):
        raise ValueError("--num_classes_grid supports only 2,3,5,10")
    if not set(grids["class_imbalance_grid"]).issubset(
        {"balanced", "mild", "moderate", "severe"}
    ):
        raise ValueError("unknown class imbalance level")
    if not {"balanced", "mild"}.intersection(grids["class_imbalance_grid"]):
        raise ValueError(
            "--class_imbalance_grid must include balanced or mild for A/C regimes"
        )
    if not set(grids["margin_grid"]).issubset({"low", "medium", "high"}):
        raise ValueError("unknown margin level")

    assignments, allocation_audit = allocate_classification_tasks(
        args.n_datasets,
        args.profile,
        args.base_seed,
        allow_underdetermined=args.allow_underdetermined,
        num_classes=grids["num_classes_grid"],
        imbalance_levels=grids["class_imbalance_grid"],
        margin_levels=grids["margin_grid"],
        label_noise_values=grids["label_noise_grid"],
    )
    n_train_split = int(0.8 * args.n_datasets)
    n_val_split = int(0.1 * args.n_datasets)
    split_counts = {
        "train": n_train_split,
        "val": n_val_split,
        "test": args.n_datasets - n_train_split - n_val_split,
    }
    records: list[dict] = []
    print(f"Profile: {args.profile}")
    print(f"Generating {args.n_datasets} classification datasets -> {root}")
    for idx, assignment in enumerate(assignments):
        task_seed = int(
            np.random.SeedSequence([args.base_seed, idx, 0xC1A55]).generate_state(1)[0]
        )
        rng = np.random.default_rng(task_seed)
        regime = assignment["classification_regime"]
        ds = generate_classification_dataset(
            rng,
            regime,
            assignment,
            grids,
            task_seed=task_seed,
        )
        validate_classification_dataset(ds)
        ds_split = split_classification_dataset(ds)
        diagnostics = compute_classification_diagnostics(ds)
        teacher = compute_classification_teacher(
            ds_split["X_train"],
            ds_split["y_train"],
            ds_split["X_test"],
            ds["num_classes"],
            requested=args.store_class_teacher_preds or args.require_class_teachers,
        )
        if args.require_class_teachers and not teacher["teacher_available"]:
            raise RuntimeError(
                "required classification teacher failed: "
                f"regime={regime}, seed={task_seed}, "
                f"reason={teacher['teacher_failure_reason']}"
            )
        split_dir, filename = _output_path(
            idx, n_train_split, n_val_split, directories
        )
        filepath = os.path.join(split_dir, filename)
        write_classification_parquet(
            ds_split,
            filepath,
            store_class_params=args.store_class_params,
            store_teacher_preds=args.store_class_teacher_preds,
            teacher_results=teacher,
            diagnostics=diagnostics,
        )
        records.append({
            "classification_regime": regime,
            "num_classes": ds["num_classes"],
            "class_imbalance_type": ds["class_imbalance_type"],
            "temperature": ds["temperature"],
            "margin_level": ds["margin_level"],
            "label_noise_rate": ds["label_noise_rate"],
        })
        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1:4d}/{args.n_datasets}] written {filename}")
    _write_classification_manifest(
        root,
        args,
        generation_command,
        grids,
        allocation_audit,
        records,
        split_counts,
    )
    print("Done.")


def _run_mixed_regression(args: argparse.Namespace) -> None:
    """Generate mixed-categorical regression meta-datasets."""
    from constants import MIXED_REG_DGP_SCHEMA_VERSION
    n_datasets = args.n_datasets
    out_dir = args.out_dir
    if out_dir is None:
        raise ValueError(
            "linear_regression_mixed_categorical requires an explicit --out_dir"
        )
    profile = args.profile
    base_seed = args.base_seed

    n_train_split = int(0.8 * n_datasets)
    n_val_split = int(0.1 * n_datasets)
    directories = _split_directories(out_dir, subdir="mixed")

    assignments, allocation_audit = allocate_mixed_regression_tasks(
        n_datasets,
        profile,
        base_seed,
        n_grid=parse_int_grid(args.n_grid),
        p_num_signal_grid=parse_int_grid(args.p_num_signal_grid),
        p_num_noise_grid=parse_int_grid(args.p_num_noise_grid),
        p_cat_signal_grid=parse_int_grid(args.p_cat_signal_grid),
        p_cat_noise_grid=parse_int_grid(args.p_cat_noise_grid),
        cat_cardinality_grid=parse_int_grid(args.cat_cardinality_grid),
        cat_effect_scale_grid=parse_float_grid(args.cat_effect_scale_grid),
        cat_missing_rate_grid=parse_float_grid(args.cat_missing_rate_grid),
        cat_imbalance_grid=parse_string_grid(args.cat_imbalance_grid),
        target_noise_grid=parse_float_grid(args.target_noise_grid),
        rho_grid=parse_float_grid(args.rho_grid),
        allow_underdetermined=args.allow_underdetermined,
    )
    print(f"Profile: {profile}")
    print(f"Generating {n_datasets} mixed-categorical regression datasets -> {out_dir}")

    for idx, assignment in enumerate(assignments):
        task_seed = int(
            np.random.SeedSequence([base_seed, idx, 0xCA7E60]).generate_state(1)[0]
        )
        rng = np.random.default_rng(task_seed)
        regime = assignment["prior_regime"]
        ds = build_mixed_regression_dataset(rng, regime, assignment)
        validate_mixed_regression_dataset(ds)

        n = int(ds["n"])
        n_train = int(0.8 * n)
        n_test = max(1, n - n_train)

        X_cat_train = ds["X_cat"][:n_train]
        X_cat_test = ds["X_cat"][n_train:]
        unknown_mask_test = mark_unseen_query_categories(X_cat_train, X_cat_test)

        ds_split = {
            "X_num_train": ds["X_num"][:n_train],
            "X_num_test":  ds["X_num"][n_train:],
            "y_train":     ds["y"][:n_train],
            "y_test":      ds["y"][n_train:],
            "X_cat_train": X_cat_train,
            "X_cat_test":  X_cat_test,
            "cat_missing_mask_train": ds["missing_mask"][:n_train],
            "cat_missing_mask_test":  ds["missing_mask"][n_train:],
            "cat_unknown_mask_test":  unknown_mask_test,
            "categorical_cardinalities": ds["cardinalities"],
            "n": n,
            "p_num": ds["p_num"],
            "p_cat": ds["p_cat"],
            "n_train": n_train,
            "n_test": n_test,
            "prior_regime": ds["prior_regime"],
            "schema_version": ds["schema_version"],
            "training_data_family": ds["training_data_family"],
            "task_family": ds["task_family"],
            "task_objective": ds["task_objective"],
            "beta_num": ds["beta_num"],
            "cat_effects": ds["cat_effects"],
            "numeric_support_mask": ds["numeric_support_mask"],
            "cat_support_mask": ds["cat_support_mask"],
        }

        split_dir, filename = _output_path(idx, n_train_split, n_val_split, directories)
        filepath = os.path.join(split_dir, filename)
        write_parquet_mixed_regression_dgp(
            ds_split, filepath, store_cat_params=args.store_cat_params,
        )

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1:4d}/{n_datasets}] written {filename} to {split_dir}")

    print("Done.")


def _run_mixed_classification(args: argparse.Namespace) -> None:
    """Generate mixed-categorical classification meta-datasets."""
    from constants import MIXED_CLS_DGP_SCHEMA_VERSION
    n_datasets = args.n_datasets
    out_dir = args.out_dir
    if out_dir is None:
        raise ValueError(
            "linear_classification_mixed_categorical requires an explicit --out_dir"
        )
    profile = args.profile
    base_seed = args.base_seed

    n_train_split = int(0.8 * n_datasets)
    n_val_split = int(0.1 * n_datasets)
    directories = _split_directories(out_dir, subdir="mixed")

    assignments, allocation_audit = allocate_mixed_classification_tasks(
        n_datasets,
        profile,
        base_seed,
        n_grid=parse_int_grid(args.n_grid),
        p_num_signal_grid=parse_int_grid(args.p_num_signal_grid),
        p_num_noise_grid=parse_int_grid(args.p_num_noise_grid),
        p_cat_signal_grid=parse_int_grid(args.p_cat_signal_grid),
        p_cat_noise_grid=parse_int_grid(args.p_cat_noise_grid),
        cat_cardinality_grid=parse_int_grid(args.cat_cardinality_grid),
        cat_effect_scale_grid=parse_float_grid(args.cat_effect_scale_grid),
        cat_missing_rate_grid=parse_float_grid(args.cat_missing_rate_grid),
        cat_imbalance_grid=parse_string_grid(args.cat_imbalance_grid),
        num_classes_grid=parse_int_grid(args.num_classes_grid),
        temperature_grid=parse_float_grid(args.temperature_grid),
        label_noise_grid=parse_float_grid(args.label_noise_grid),
        rho_grid=parse_float_grid(args.rho_grid),
        allow_underdetermined=args.allow_underdetermined,
    )
    print(f"Profile: {profile}")
    print(f"Generating {n_datasets} mixed-categorical classification datasets -> {out_dir}")

    for idx, assignment in enumerate(assignments):
        task_seed = int(
            np.random.SeedSequence([base_seed, idx, 0xCA7C15]).generate_state(1)[0]
        )
        rng = np.random.default_rng(task_seed)
        regime = assignment["prior_regime"]
        ds = build_mixed_classification_dataset(rng, regime, assignment)
        validate_mixed_classification_dataset(ds)

        n = int(ds["n"])
        n_train = int(0.8 * n)
        n_test = max(1, n - n_train)

        X_cat_train = ds["X_cat"][:n_train]
        X_cat_test = ds["X_cat"][n_train:]
        unknown_mask_test = mark_unseen_query_categories(X_cat_train, X_cat_test)

        ds_split = {
            "X_num_train": ds["X_num"][:n_train],
            "X_num_test":  ds["X_num"][n_train:],
            "y_train":     ds["y"][:n_train],
            "y_test":      ds["y"][n_train:],
            "X_cat_train": X_cat_train,
            "X_cat_test":  X_cat_test,
            "cat_missing_mask_train": ds["missing_mask"][:n_train],
            "cat_missing_mask_test":  ds["missing_mask"][n_train:],
            "cat_unknown_mask_test":  unknown_mask_test,
            "categorical_cardinalities": ds["cardinalities"],
            "n": n,
            "p_num": ds["p_num"],
            "p_cat": ds["p_cat"],
            "n_train": n_train,
            "n_test": n_test,
            "num_classes": ds["num_classes"],
            "prior_regime": ds["prior_regime"],
            "schema_version": ds["schema_version"],
            "training_data_family": ds["training_data_family"],
            "task_family": ds["task_family"],
            "task_objective": ds["task_objective"],
            "imbalance_type": ds.get("imbalance_type", "balanced"),
            "temperature": ds.get("temperature", 1.0),
            "label_noise_rate": ds.get("label_noise_rate", 0.0),
            "W_num": ds["W_num"],
            "b": ds["b"],
            "cat_class_effects": ds["cat_class_effects"],
            "numeric_support_mask": ds["numeric_support_mask"],
            "cat_support_mask": ds["cat_support_mask"],
        }

        split_dir, filename = _output_path(idx, n_train_split, n_val_split, directories)
        filepath = os.path.join(split_dir, filename)
        write_parquet_mixed_classification_dgp(
            ds_split, filepath, store_class_cat_params=args.store_cat_params,
        )

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1:4d}/{n_datasets}] written {filename} to {split_dir}")

    print("Done.")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _PROFILE_DEFAULTS = {
        "linear_regression": "linear_stat_aware",
        "linear_classification": "linear_classification_stat_aware",
        "linear_regression_mixed_categorical": "linear_regression_mixed_categorical_stat_aware",
        "linear_classification_mixed_categorical": "linear_classification_mixed_categorical_stat_aware",
    }
    _PROFILE_GROUPS = {
        "linear_regression": REGRESSION_PROFILES,
        "linear_classification": CLASSIFICATION_PROFILES,
        "linear_regression_mixed_categorical": REGRESSION_PROFILES,
        "linear_classification_mixed_categorical": CLASSIFICATION_PROFILES,
    }
    if args.profile is None:
        args.profile = _PROFILE_DEFAULTS.get(args.task_family, "linear_stat_aware")
    expected_profiles = _PROFILE_GROUPS.get(args.task_family, REGRESSION_PROFILES)
    if args.profile not in expected_profiles:
        parser.error(
            f"profile {args.profile!r} does not belong to task family "
            f"{args.task_family!r}"
        )
    command = [str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
    if args.task_family == "linear_regression":
        _run_regression(args)
    elif args.task_family == "linear_classification":
        _run_classification(args, command)
    elif args.task_family == "linear_regression_mixed_categorical":
        _run_mixed_regression(args)
    elif args.task_family == "linear_classification_mixed_categorical":
        _run_mixed_classification(args)


if __name__ == "__main__":
    main()
