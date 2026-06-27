"""Generate bias-resistant linear regression evaluation suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Bootstrap: add all src/ + scripts/ subdirs to sys.path for flat-import resolution.
_p = Path(__file__).resolve()
for _anc in _p.parents:
    if (_anc / "_bootstrap.py").exists():
        sys.path.insert(0, str(_anc))
        break
import _bootstrap  # noqa: E402,F401
del _p, _anc

from constants import (  # noqa: E402
    CURRICULUM_POLICIES,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_FEATURE_CAP,
    DEFAULT_GPU_GUARD_BYTES,
    DEFAULT_TEST_BATCH_SIZE,
    MAX_MEMORY_RISK_CHOICES,
    REGRESSION_COVERAGE_AXES,
)
from dgp_helpers import (  # noqa: E402
    ALLOCATION_MODES,
    LEGACY_REGRESSION_EVAL_SCHEMA_VERSION,
    REGIME_WEIGHTS,
    REGRESSION_EVAL_ONLY_REGIMES,
    REGRESSION_EVAL_SCHEMA_VERSION,
    REGRESSION_EVAL_SUITE_FAMILIES,
    REGRESSION_PROFILES,
    REGRESSION_REGIME_REGISTRY,
    _compute_sample_complexity_bucket,
    allocate_regimes,
    allocate_regression_tasks,
    allocate_mixed_regression_tasks,
    build_coverage_audit,
    build_dataset_from_regime,
    build_difficulty_audit,
    build_memory_audit,
    build_mixed_regression_dataset,
    build_task_fingerprint,
    build_train_eval_alignment_report,
    compute_difficulty_metadata,
    compute_difficulty_score,
    compute_linear_diagnostics,
    compute_linear_teacher_targets,
    estimate_generation_memory,
    estimate_memory_risk,
    mark_unseen_query_categories,
    parse_difficulty_mix,
    parse_float_grid,
    parse_int_grid,
    sample_n_p,
    validate_controlled_manifest,
    validate_coverage,
    validate_dataset,
    validate_mixed_regression_dataset,
    write_parquet_eval,
    write_parquet_mixed_regression_dgp,
)

# Env-var mixed-categorical detection (for orchestrator-driven generation)
SYNREG_IS_MIXED_CATEGORICAL = os.getenv(
    "SYNREG_IS_MIXED_CATEGORICAL", "false"
).lower() in ("1", "true", "yes")

_REG_SEED_MAGIC = 0xDE5EED42
_REG_HIDDEN_SEED_MAGIC = 0x48DDE771
_FAMILY_MAGIC = {
    family: int.from_bytes(hashlib.sha256(family.encode()).digest()[:4], "little")
    for family in REGRESSION_EVAL_SUITE_FAMILIES
}
_FAMILY_ORDER = [
    "primary",
    "feature_noise",
    "target_noise",
    "training_size",
    "sparsity",
    "correlation",
    "dimensionality",
    "ood",
    "eval_only_unseen",
    "hidden_holdout",
    "stress",
]


def _add_family_flags(parser: argparse.ArgumentParser) -> None:
    for family in _FAMILY_ORDER:
        default = True if family == "primary" else None
        parser.add_argument(
            f"--include_{family}",
            dest=f"include_{family}",
            action="store_true",
            default=default,
        )
        parser.add_argument(
            f"--no-include_{family}",
            dest=f"include_{family}",
            action="store_false",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate schema-v2 synthetic linear regression evaluation data."
    )
    parser.add_argument("--n_datasets", type=int, default=200)
    parser.add_argument("--eval_only_n_datasets", type=int, default=None)
    parser.add_argument("--hidden_holdout_n_datasets", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--suite_id", type=str, default="linear_regression_eval_v2")
    parser.add_argument(
        "--hidden_holdout_suite_id",
        type=str,
        default="linear_regression_hidden_holdout_v1",
    )
    parser.add_argument("--base_seed", type=int, default=20260512)
    parser.add_argument("--hidden_holdout_base_seed", type=int, default=20260607)
    parser.add_argument(
        "--profile",
        type=str,
        default="linear_stat_aware",
        choices=REGRESSION_PROFILES,
    )
    parser.add_argument(
        "--allocation_mode",
        choices=sorted(ALLOCATION_MODES),
        default="balanced",
    )
    parser.add_argument("--n_grid", type=str, default="32,64,128,256,512,1024")
    parser.add_argument("--p_signal_grid", type=str, default="4,8,16,32,64")
    parser.add_argument("--p_noise_grid", type=str, default="0,8,24,56,120")
    parser.add_argument("--active_s_grid", type=str, default="2,4,8,16,32")
    parser.add_argument("--rho_grid", type=str, default="-0.6,0.0,0.3,0.6,0.9")
    parser.add_argument("--target_noise_grid", type=str, default="0.25,0.5,1.0,2.0")
    parser.add_argument("--feature_noise_grid", type=str, default="0.0,0.05,0.10,0.25")
    parser.add_argument("--allow_underdetermined", action="store_true")
    parser.add_argument("--store_teacher_preds", action="store_true")
    parser.add_argument("--store_beta", action="store_true")
    parser.add_argument("--store_linear_moments", action="store_true")
    parser.add_argument("--strict_coverage", action="store_true", default=False)
    parser.add_argument(
        "--no-strict_coverage", dest="strict_coverage", action="store_false"
    )
    parser.add_argument(
        "--coverage_config",
        type=str,
        default=None,
        help="JSON object or JSON file containing per-axis minimum counts.",
    )
    parser.add_argument(
        "--training_manifest",
        action="append",
        default=[],
        help="Manifest used for strict train/eval seed and fingerprint leakage checks.",
    )
    parser.add_argument("--legacy_schema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--mixed_categorical",
        action="store_true",
        default=False,
        help=(
            "Generate mixed-categorical regression datasets (numeric + categorical features). "
            "Output goes to linear_regression/mixed/ instead of linear_regression/numeric/."
        ),
    )
    _add_family_flags(parser)
    parser.add_argument("--min_regime_count", type=int, default=10)
    parser.add_argument("--min_suite_family_count", type=int, default=5)
    parser.add_argument(
        "--curriculum_policy",
        choices=sorted(CURRICULUM_POLICIES),
        default="core_first",
    )
    parser.add_argument(
        "--difficulty_mix", type=str, default="core=0.65,robust=0.30,stress=0.05",
    )
    parser.add_argument(
        "--max_memory_risk",
        choices=list(MAX_MEMORY_RISK_CHOICES),
        default="high",
    )
    parser.add_argument("--memory_guard_bytes", type=int, default=DEFAULT_GPU_GUARD_BYTES)
    parser.add_argument("--allow_memory_stress", action="store_true", default=False)
    parser.add_argument("--emit_memory_stress_suite", action="store_true", default=False)
    parser.add_argument("--emit_alignment_report", action="store_true", default=False)
    parser.add_argument("--train_manifest_path", type=str, default=None)
    parser.add_argument("--eval_manifest_path", type=str, default=None)
    parser.add_argument("--alignment_report_out", type=str, default=None)
    parser.add_argument("--strict_manifest_validation", action="store_true", default=False)
    parser.add_argument("--dry_run_allocation", action="store_true", default=False)
    parser.add_argument("--write_allocation_audit_only", action="store_true", default=False)
    return parser.parse_args(argv)


def _enabled_families(args: argparse.Namespace) -> list[str]:
    if args.profile == "legacy":
        return [
            family
            for family in _FAMILY_ORDER
            if getattr(args, f"include_{family}") is True
        ]
    defaults = {
        "primary",
        "feature_noise",
        "target_noise",
        "training_size",
        "sparsity",
        "correlation",
        "dimensionality",
        "ood",
        "eval_only_unseen",
        "hidden_holdout",
        "stress",
    }
    return [
        family
        for family in _FAMILY_ORDER
        if (
            getattr(args, f"include_{family}") is True
            or (
                getattr(args, f"include_{family}") is None
                and family in defaults
            )
        )
    ]


def _load_coverage_config(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    path = Path(value)
    payload = json.loads(path.read_text()) if path.exists() else json.loads(value)
    return {str(key): int(minimum) for key, minimum in payload.items()}


def _derive_seed(
    args: argparse.Namespace,
    family: str,
    family_idx: int,
) -> int:
    hidden = family == "hidden_holdout"
    base = args.hidden_holdout_base_seed if hidden else args.base_seed
    magic = _REG_HIDDEN_SEED_MAGIC if hidden else _REG_SEED_MAGIC
    return int(
        np.random.SeedSequence(
            [base, _FAMILY_MAGIC[family], family_idx, magic]
        ).generate_state(1)[0]
    )


def _family_count(args: argparse.Namespace, family: str) -> int:
    if family == "eval_only_unseen":
        return int(args.eval_only_n_datasets or args.n_datasets)
    if family == "hidden_holdout":
        return int(args.hidden_holdout_n_datasets or args.n_datasets)
    return int(args.n_datasets)


def _regular_regimes(args: argparse.Namespace) -> tuple[list[str], dict[str, float]]:
    weights = dict(REGIME_WEIGHTS[args.profile])
    if not args.allow_underdetermined:
        weights.pop("J_low_n_high_p", None)
    return list(weights), weights


def _axis_for_family(args: argparse.Namespace, family: str) -> dict[str, list[Any]]:
    mapping = {
        "feature_noise": {
            "feature_noise_level": parse_float_grid(args.feature_noise_grid)
        },
        "target_noise": {
            "target_noise_scale": parse_float_grid(args.target_noise_grid)
        },
        "training_size": {"n": parse_int_grid(args.n_grid)},
        "sparsity": {"active_s": parse_int_grid(args.active_s_grid)},
        "correlation": {"rho": parse_float_grid(args.rho_grid)},
        "dimensionality": {"p_signal": parse_int_grid(args.p_signal_grid)},
    }
    return mapping.get(family, {})


def _family_assignments(
    args: argparse.Namespace,
    family: str,
    count: int,
) -> list[dict[str, Any]]:
    regular, weights = _regular_regimes(args)
    if family in {"eval_only_unseen", "hidden_holdout", "ood"}:
        regimes = list(REGRESSION_EVAL_ONLY_REGIMES)
        mode = "balanced" if args.allocation_mode == "weighted" else args.allocation_mode
        return allocate_regimes(
            count,
            regimes,
            mode=mode,
            axis_values=_axis_for_family(args, family),
        )
    if family == "stress":
        regimes = [
            regime
            for regime in REGIME_WEIGHTS["linear_stress"]
            if args.allow_underdetermined or regime != "J_low_n_high_p"
        ]
        return allocate_regimes(count, regimes, mode="balanced")
    return allocate_regimes(
        count,
        regular,
        mode=args.allocation_mode,
        weights=weights,
        axis_values=_axis_for_family(args, family),
    )


def _sample_params(
    rng: np.random.Generator,
    args: argparse.Namespace,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    params = sample_n_p(
        rng,
        args.profile,
        parse_int_grid(args.n_grid),
        parse_int_grid(args.p_signal_grid),
        parse_int_grid(args.p_noise_grid),
        parse_int_grid(args.active_s_grid),
        parse_float_grid(args.rho_grid),
        parse_float_grid(args.target_noise_grid),
        parse_float_grid(args.feature_noise_grid),
        allow_underdetermined=args.allow_underdetermined,
    )
    params.update({key: value for key, value in assignment.items() if key != "regime"})
    return params


def _record_metadata(
    ds: dict[str, Any],
    *,
    args: argparse.Namespace,
    family: str,
    global_idx: int,
    dataset_seed: int,
) -> dict[str, Any]:
    n_holdout = max(1, int(ds["n_total"]) // 5)
    n_train = int(ds["n_total"]) - n_holdout
    difficulty = compute_difficulty_metadata(
        n_train=n_train,
        p_total=int(ds["p_total"]),
        noise_scale=float(ds.get("target_noise_scale", 0.0)),
        feature_noise_level=float(ds.get("feature_noise_level", 0.0)),
        sparsity_ratio=float(ds.get("sparsity_ratio", 0.0)),
        rho=float(ds.get("rho", 0.0)),
    )
    memory = estimate_generation_memory(n_train, n_holdout, int(ds["p_total"]))
    fingerprint = build_task_fingerprint(
        task_family="linear_regression",
        regime=str(ds["prior_regime"]),
        suite_family=family,
        n_total=int(ds["n_total"]),
        p_total=int(ds["p_total"]),
        covariance_type=str(ds.get("covariance_type", "iid")),
        noise_type=str(ds.get("target_noise_type", "gaussian")),
        sparsity_ratio=float(ds.get("sparsity_ratio", 0.0)),
    )
    return {
        "global_idx": global_idx,
        "dataset_seed": dataset_seed,
        "n_train_default": n_train,
        "n_holdout_default": n_holdout,
        "is_training_allowed": bool(
            REGRESSION_REGIME_REGISTRY[ds["prior_regime"]]["is_training_allowed"]
        ),
        "is_eval_only": bool(
            REGRESSION_REGIME_REGISTRY[ds["prior_regime"]]["is_eval_only"]
        ),
        "is_ood": family in {"ood", "eval_only_unseen", "hidden_holdout", "stress"},
        "is_hidden_holdout": family == "hidden_holdout",
        "task_fingerprint": fingerprint,
        **difficulty,
        **memory,
    }


def _training_audit(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    training_seeds: set[int] = set()
    training_fingerprints: set[str] = set()
    forbidden_regimes: set[str] = set()
    manifests = []
    for filename in args.training_manifest:
        manifest = json.loads(Path(filename).read_text())
        manifests.append(filename)
        for record in manifest.get("datasets", []):
            if record.get("dataset_seed") is not None:
                training_seeds.add(int(record["dataset_seed"]))
            if record.get("task_fingerprint"):
                training_fingerprints.add(str(record["task_fingerprint"]))
            if record.get("is_eval_only"):
                forbidden_regimes.add(str(record.get("prior_regime", record.get("regime"))))
    eval_seeds = {int(record["dataset_seed"]) for record in records}
    eval_fingerprints = {str(record["task_fingerprint"]) for record in records}
    return {
        "training_manifests": manifests,
        "dataset_seed_overlap": sorted(training_seeds & eval_seeds),
        "fingerprint_overlap": sorted(training_fingerprints & eval_fingerprints),
        "eval_only_regimes_in_training": sorted(forbidden_regimes),
    }


_MIXED_REG_EVAL_SEED_MAGIC = 0x4D49584E  # "MIXN"


def _run_mixed_regression(args: argparse.Namespace) -> dict[str, Any]:
    """Generate mixed-categorical regression evaluation datasets.

    Uses allocate_mixed_regression_tasks + build_mixed_regression_dataset.
    Output goes to linear/regression/mixed/eval/ with
    stage path @EVALUATION_DATASET_STAGE/linear/regression/mixed/eval/...
    """
    if args.n_datasets < 1:
        raise ValueError("--n_datasets must be >= 1")
    suite_dir = Path(args.out_dir) / "linear_regression" / "mixed"
    if args.dry_run:
        return {"suite_id": args.suite_id, "mixed_categorical": True, "dry_run": True}
    suite_dir.mkdir(parents=True, exist_ok=True)

    assignments, alloc_audit = allocate_mixed_regression_tasks(
        args.n_datasets,
        args.profile,
        args.base_seed,
        allow_underdetermined=args.allow_underdetermined,
    )

    records: list[dict[str, Any]] = []
    for idx, assignment in enumerate(assignments):
        task_seed = int(
            np.random.SeedSequence(
                [args.base_seed, idx, _MIXED_REG_EVAL_SEED_MAGIC]
            ).generate_state(1)[0]
        )
        rng = np.random.default_rng(task_seed)
        regime = str(assignment["prior_regime"])
        ds = build_mixed_regression_dataset(rng, regime, assignment)
        validate_mixed_regression_dataset(ds)

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

        eval_dir = suite_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{regime}__dataset_{idx:05d}.parquet"
        filepath = eval_dir / filename
        write_parquet_mixed_regression_dgp(ds_split, str(filepath))

        stage_path = (
            f"@EVALUATION_DATASET_STAGE/linear/regression/mixed/eval/{filename}"
        )
        records.append({
            "dataset_id":           idx,
            "suite_id":             args.suite_id,
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
            "cardinalities":        ds["cardinalities"],
            "missing_rate":         float(assignment.get("missing_rate", 0.0)),
            "target_noise_scale":   float(assignment.get("target_noise_scale", 1.0)),
            "rho":                  float(assignment.get("rho", 0.0)),
            "task_family":          ds["task_family"],
            "training_data_family": ds["training_data_family"],
            "schema_version":       ds["schema_version"],
        })

    manifest = {
        "suite_id":          args.suite_id,
        "task_family":       "mixed_categorical_regression",
        "profile":           args.profile,
        "base_seed":         args.base_seed,
        "n_datasets":        len(records),
        "mixed_categorical": True,
        "allocation_audit":  alloc_audit,
        "datasets":          records,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = suite_dir / "mixed_regression_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[mixed] Generated {len(records)} datasets → {suite_dir}\n"
        f"[mixed] Manifest: {manifest_path}"
    )
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "mixed_categorical", False) or SYNREG_IS_MIXED_CATEGORICAL:
        return _run_mixed_regression(args)
    if args.n_datasets < 1:
        raise ValueError("--n_datasets must be >= 1")
    enabled = _enabled_families(args)
    if not enabled:
        raise ValueError("at least one suite family must be enabled")
    if args.hidden_holdout_suite_id == args.suite_id:
        raise ValueError("hidden_holdout_suite_id must be a separate logical namespace")
    suite_dir = Path(args.out_dir) / "linear_regression" / "numeric"
    if suite_dir.exists() and any(suite_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output directory exists: {suite_dir}")
    if args.dry_run:
        return {"suite_id": args.suite_id, "enabled_suite_families": enabled}

    # Parse difficulty mix early for use throughout
    difficulty_mix_dict = parse_difficulty_mix(
        getattr(args, "difficulty_mix", "core=0.65,robust=0.30,stress=0.05")
    )

    # Dry-run allocation: print audit and return without writing files
    if getattr(args, "dry_run_allocation", False):
        _assignments, _alloc_audit = allocate_regression_tasks(
            args.n_datasets,
            args.profile,
            args.base_seed,
            allow_underdetermined=args.allow_underdetermined,
            n_grid=parse_int_grid(args.n_grid),
            p_signal_grid=parse_int_grid(args.p_signal_grid),
            p_noise_grid=parse_int_grid(args.p_noise_grid),
            active_s_grid=parse_int_grid(args.active_s_grid),
            rho_grid=parse_float_grid(args.rho_grid),
            target_noise_grid=parse_float_grid(args.target_noise_grid),
            feature_noise_grid=parse_float_grid(args.feature_noise_grid),
            allocation_mode=args.allocation_mode,
            min_regime_count=getattr(args, "min_regime_count", 10),
            strict_coverage=args.strict_coverage,
        )
        print(json.dumps(_alloc_audit, indent=2))
        return {"suite_id": args.suite_id, "dry_run_allocation": True}

    suite_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    global_idx = 0
    schema_version = (
        LEGACY_REGRESSION_EVAL_SCHEMA_VERSION
        if args.legacy_schema or args.profile == "legacy"
        else REGRESSION_EVAL_SCHEMA_VERSION
    )
    eval_dir = suite_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for family in enabled:
        assignments = _family_assignments(args, family, _family_count(args, family))
        for family_idx, assignment in enumerate(assignments):
            dataset_seed = _derive_seed(args, family, family_idx)
            rng = np.random.default_rng(dataset_seed)
            params = _sample_params(rng, args, assignment)
            regime = str(assignment["regime"])
            profile = args.profile if regime not in REGRESSION_EVAL_ONLY_REGIMES else "linear_stat_aware"
            ds = build_dataset_from_regime(rng, regime, profile, params)
            if family == "feature_noise":
                level = float(assignment["feature_noise_level"])
                ds["X"] = (
                    ds["X_clean"]
                    + level * rng.standard_normal(ds["X_clean"].shape)
                )
                ds["feature_noise_level"] = level
                ds["has_feature_noise"] = level > 0.0
            if family == "target_noise":
                scale = float(assignment["target_noise_scale"])
                ds["y"] = ds["betaX"] + rng.standard_normal(ds["n_total"]) * scale
                ds["target_noise_scale"] = scale
                ds["target_noise_type"] = "gaussian"
            validate_dataset(ds)
            metadata = _record_metadata(
                ds,
                args=args,
                family=family,
                global_idx=global_idx,
                dataset_seed=dataset_seed,
            )
            n_train = metadata["n_train_default"]
            teacher = compute_linear_teacher_targets(
                ds["X"][:n_train],
                ds["y"][:n_train],
                ds["X"][n_train:],
                ds["y"][n_train:],
            )
            diagnostics = compute_linear_diagnostics(
                ds["X"], ds["beta"], ds["p_signal"]
            )
            filename = f"{family}__dataset_{global_idx:05d}.parquet"
            filepath = eval_dir / filename
            payload_bytes = write_parquet_eval(
                ds,
                str(filepath),
                profile=profile,
                store_beta=args.store_beta,
                store_teacher_preds=args.store_teacher_preds,
                store_linear_moments=args.store_linear_moments,
                teacher_results=teacher,
                diagnostics=diagnostics,
                sample_complexity_bucket=_compute_sample_complexity_bucket(
                    ds["n_total"], ds["p_total"]
                ),
                suite_id=args.suite_id,
                suite_family=family,
                global_idx=global_idx,
                dataset_seed=dataset_seed,
                schema_version=schema_version,
                extra_metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in {"difficulty_components"}
                },
            )
            record = {
                "dataset_id": family_idx,
                "suite_id": args.suite_id,
                "suite_family": family,
                "stage_path": (
                    f"@EVALUATION_DATASET_STAGE/linear/regression/numeric/eval/{filename}"
                ),
                "prior_regime": regime,
                "split_seeds": [0, 1, 2, 3, 4],
                "n_total": int(ds["n_total"]),
                "p_signal": int(ds["p_signal"]),
                "p_noise": int(ds["p_noise"]),
                "p_total": int(ds["p_total"]),
                "active_s": int(ds.get("active_s", 0)),
                "sparsity_ratio": float(ds.get("sparsity_ratio", 0.0)),
                "covariance_type": str(ds.get("covariance_type", "iid")),
                "rho": float(ds.get("rho", 0.0)),
                "target_noise_scale": float(ds.get("target_noise_scale", 0.0)),
                "target_noise_type": str(ds.get("target_noise_type", "gaussian")),
                "feature_noise_level": float(ds.get("feature_noise_level", 0.0)),
                "distribution_family": str(
                    ds.get("distribution_family", "gaussian_linear")
                ),
                "training_size_anchor": False,
                "payload_bytes": payload_bytes,
                "hidden_holdout_suite_id": (
                    args.hidden_holdout_suite_id
                    if family == "hidden_holdout"
                    else None
                ),
                **metadata,
            }
            records.append(record)
            global_idx += 1

    coverage_axes = {
        "suite_family": enabled,
        "prior_regime": sorted({record["prior_regime"] for record in records}),
    }
    coverage = validate_coverage(
        records,
        coverage_axes,
        minimums=_load_coverage_config(args.coverage_config),
    )
    train_eval = _training_audit(args, records)
    hidden_seeds = {
        record["dataset_seed"]
        for record in records
        if record["suite_family"] == "hidden_holdout"
    }
    normal_seeds = {
        record["dataset_seed"]
        for record in records
        if record["suite_family"] != "hidden_holdout"
    }
    seed_audit = {
        "all_dataset_seeds_unique": len(records)
        == len({record["dataset_seed"] for record in records}),
        "hidden_normal_seed_overlap": sorted(hidden_seeds & normal_seeds),
        "hidden_holdout_suite_id": args.hidden_holdout_suite_id,
    }
    if args.strict_coverage:
        failures = []
        if not coverage["ok"]:
            failures.append(f"missing coverage: {coverage['missing']}")
        if not seed_audit["all_dataset_seeds_unique"]:
            failures.append("dataset seeds are not unique")
        if seed_audit["hidden_normal_seed_overlap"]:
            failures.append("hidden and normal seed namespaces overlap")
        for key in (
            "dataset_seed_overlap",
            "fingerprint_overlap",
            "eval_only_regimes_in_training",
        ):
            if train_eval[key]:
                failures.append(f"{key}: {train_eval[key]}")
        if failures:
            raise ValueError("strict suite validation failed: " + "; ".join(failures))

    family_counts = Counter(record["suite_family"] for record in records)
    regime_counts = Counter(record["prior_regime"] for record in records)
    memory_counts = Counter(record["memory_class"] for record in records)
    difficulty_counts = Counter(record["difficulty_tier"] for record in records)
    manifest = {
        "suite_id": args.suite_id,
        "hidden_holdout_suite_id": args.hidden_holdout_suite_id,
        "schema_version": schema_version,
        "task_family": "linear_regression",
        "profile": args.profile,
        "base_seed": args.base_seed,
        "hidden_holdout_base_seed": args.hidden_holdout_base_seed,
        "allocation_mode": args.allocation_mode,
        "strict_coverage": args.strict_coverage,
        "enabled_suite_families": enabled,
        "realized_suite_family_counts": dict(family_counts),
        "n_datasets_by_regime_group": dict(regime_counts),
        "datasets": records,
        "seed_audit": seed_audit,
        "train_eval_alignment_audit": train_eval,
        "coverage_audit": coverage,
        "memory_audit": {"counts": dict(memory_counts)},
        "difficulty_audit": {"counts": dict(difficulty_counts)},
        "alignment_audit": {"emitted": False},
        "generation_controls": {
            "task_family": "linear_regression",
            "allocation_mode": args.allocation_mode,
            "curriculum_policy": getattr(args, "curriculum_policy", "core_first"),
            "difficulty_mix": getattr(args, "difficulty_mix", "core=0.65,robust=0.30,stress=0.05"),
            "memory_guard_bytes": getattr(args, "memory_guard_bytes", DEFAULT_GPU_GUARD_BYTES),
            "min_regime_count": getattr(args, "min_regime_count", 10),
        },
        "grid_metadata": {
            "n_grid": parse_int_grid(args.n_grid),
            "p_signal_grid": parse_int_grid(args.p_signal_grid),
            "p_noise_grid": parse_int_grid(args.p_noise_grid),
            "active_s_grid": parse_int_grid(args.active_s_grid),
            "rho_grid": parse_float_grid(args.rho_grid),
            "target_noise_grid": parse_float_grid(args.target_noise_grid),
            "feature_noise_grid": parse_float_grid(args.feature_noise_grid),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = suite_dir / "linear_regression_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Optional strict manifest validation
    if getattr(args, "strict_manifest_validation", False):
        validate_controlled_manifest(manifest)

    # Optional alignment report
    if getattr(args, "emit_alignment_report", False):
        train_path = getattr(args, "train_manifest_path", None)
        eval_path = getattr(args, "eval_manifest_path", None)
        report_out = getattr(args, "alignment_report_out", None)
        if train_path and eval_path:
            train_m = json.loads(Path(train_path).read_text())
            eval_m = json.loads(Path(eval_path).read_text())
            alignment = build_train_eval_alignment_report(train_m, eval_m)
            if report_out:
                Path(report_out).write_text(json.dumps(alignment, indent=2))
            manifest["alignment_audit"] = {"emitted": True, **alignment}

    return manifest


def main() -> None:
    args = parse_args()
    manifest = run(args)
    if args.dry_run:
        print("DRY RUN")
        print(json.dumps(manifest, indent=2))
        return
    if manifest.get("dry_run_allocation"):
        return
    subdir = "linear_regression/mixed" if getattr(args, "mixed_categorical", False) else "linear_regression/numeric"
    print(
        f"Generated {len(manifest['datasets'])} datasets under "
        f"{Path(args.out_dir) / subdir}"
    )


if __name__ == "__main__":
    main()
