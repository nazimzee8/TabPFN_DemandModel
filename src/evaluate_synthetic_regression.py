"""
evaluate_synthetic_regression.py
==================================
Synthetic-regression evaluation engine.

Modes (SYNTHETIC_REGRESSION_MODE):
  deepset   -- DeepSet bounded-context ensemble (GPU)
  baselines -- sklearn + tree baselines (CPU)
  autogluon -- AutoGluon TabularPredictor (CPU)
  aggregate -- Ingest parts, rank, compute ratios, write summary CSVs
"""

from __future__ import annotations

import csv
import datetime
import gc
import json
import math
import os
import re
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from snowflake.snowpark import Session

from autogluon_models import (
    get_tabular_predictor_class,
    make_unique_autogluon_model_dir,
    predict_autogluon_timed,
)
from baseline_models import make_baseline_model, validate_baseline_dependencies
from deepset_inference import (
    deepset_gpu_memory_skip_reason,
    deepset_inference_device,
    predict_deepset_bounded_context_ensemble,
    run_permutation_tests,
    select_deepset_features_train_only,
)
from model import ModelConfig, _instantiate_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


SYNREG_SUITE_ID = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
SYNREG_NUM_SHARDS = int(os.getenv("SYNTHETIC_REGRESSION_NUM_SHARDS", "1"))
SYNREG_SHARD_INDEX = int(os.getenv("SYNTHETIC_REGRESSION_SHARD_INDEX", "0"))
SYNREG_STAGE_PREFIX = "@EVALUATION_DATASET_STAGE"
SYNREG_RESULTS_STAGE = os.getenv(
    "SYNREG_RESULTS_STAGE",
    "@EVALUATION_RESULTS_STAGE/regression",
)
SYNREG_OUTPUT_STAGE = os.getenv("SYNREG_OUTPUT_STAGE", "@EVALUATION_RESULTS_STAGE")
SYNREG_INDEX_TABLE = "SYNTHETIC_REGRESSION_DATASET_INDEX"
SYNREG_WORKER_DATA_ACCESS_MODE = os.getenv(
    "SYNREG_WORKER_DATA_ACCESS_MODE",
    "scoped_file_url",
)
SYNREG_MAX_WORK_ITEM_BYTES = int(os.getenv("SYNREG_MAX_WORK_ITEM_BYTES", "8192"))
CHECKPOINT_STAGE_PATH = "@MODEL_STAGE/checkpoints/best.pt"
SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH = os.getenv(
    "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH",
    CHECKPOINT_STAGE_PATH,   # fallback: "@MODEL_STAGE/checkpoints/best.pt"
)
SYNREG_RUN_CHECKPOINT_GATES   = os.getenv("SYNREG_RUN_CHECKPOINT_GATES",   "true").lower() == "true"
SYNREG_CHECKPOINT_GATE_STRICT = os.getenv("SYNREG_CHECKPOINT_GATE_STRICT", "true").lower() == "true"
SYNREG_GATE_MAX_RIDGE_RATIO   = float(os.getenv("SYNREG_GATE_MAX_RIDGE_RATIO", "10.0"))
SYNREG_GATE_MIN_QUERY_STD     = float(os.getenv("SYNREG_GATE_MIN_QUERY_STD",   "1e-6"))
SYNREG_GATE_OUT_DIR           = os.getenv("SYNREG_GATE_OUT_DIR", "/tmp/synreg_checkpoint_gates")
SYNREG_LOCAL_CACHE = os.getenv("SYNTHETIC_REGRESSION_LOCAL_CACHE", "/tmp/synreg_eval_data")
SYNREG_CKPT_LOCAL = os.getenv("SYNTHETIC_REGRESSION_CKPT_LOCAL", "/tmp/synreg_ckpt/best.pt")

SYNREG_MC_K = int(os.getenv("MC_K", "8"))
SYNREG_CONTEXT_SIZE = int(os.getenv("SYNTHETIC_REGRESSION_CONTEXT_SIZE", "200"))
SYNREG_CONTEXT_ENSEMBLES = int(os.getenv("SYNTHETIC_REGRESSION_CONTEXT_ENSEMBLES", "5"))
SYNREG_TEST_BATCH_SIZE = int(os.getenv("SYNTHETIC_REGRESSION_TEST_BATCH_SIZE", "128"))
SYNREG_DEEPSET_FEATURE_CAP = int(os.getenv("SYNTHETIC_REGRESSION_DEEPSET_FEATURE_CAP", "128"))
SYNREG_FEATURE_SELECTOR = os.getenv("SYNTHETIC_REGRESSION_FEATURE_SELECTOR", "train_f_regression")
TRAIN_SIZE_GRID = _parse_int_list(
    os.getenv("SYNTHETIC_REGRESSION_TRAIN_SIZE_GRID", "25,50,100,200,500,1000,2000,4832")
)
HOLDOUT_SIZE = int(os.getenv("SYNTHETIC_REGRESSION_HOLDOUT_SIZE", "1371"))
TRAIN_SIZE_ANCHOR_N = 4832

# CPU guardrails (same thresholds as evaluate.py)
SYNREG_CPU_MAX_FEATURES = int(os.getenv("BENCHMARK_CPU_MAX_PROCESSED_FEATURES", "2000"))
SYNREG_CPU_MAX_MATRIX_BYTES = int(os.getenv("BENCHMARK_CPU_MAX_MATRIX_BYTES", "536870912"))

# AutoGluon
SYNREG_AG_TIME_LIMIT = int(os.getenv("AUTOGLUON_TIME_LIMIT", "300"))
SYNREG_AG_MIN_TMP_FREE_BYTES = int(os.getenv("BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", "5368709120"))
SYNREG_AG_PRESETS = os.getenv("AUTOGLUON_PRESETS", "best_quality")
SYNREG_AG_TASK_CPUS = int(os.getenv("AUTOGLUON_TASK_CPUS", "1"))

SYNREG_BASELINE_METHODS = [
    "FixedRidgeLambda1",
    "LinearRegression",
    "Ridge",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "KNN",
    "SVR",
    "MLP",
]

SYNREG_MODE = os.getenv("SYNTHETIC_REGRESSION_MODE", "deepset")

# ---------------------------------------------------------------------------
# Aggregation constants
# ---------------------------------------------------------------------------

MIN_VALID_CSV_BYTES = 64

# All-method suites that require full shard completeness validation
ALL_METHOD_SUITE_IDS = frozenset({
    "linear_poisson_v1_recommended",
    "ood_linear_full_v1",
    "linear_all_v1",
})

# Expected shard counts for completeness validation (0 = disabled)
SYNREG_EXPECTED_DEEPSET_SHARDS  = int(os.getenv("SYNREG_EXPECTED_DEEPSET_SHARDS",  "0"))
SYNREG_EXPECTED_BASELINE_SHARDS = int(os.getenv("SYNREG_EXPECTED_BASELINE_SHARDS", "0"))
SYNREG_EXPECTED_AG_SHARDS       = int(os.getenv("SYNREG_EXPECTED_AG_SHARDS",       "0"))

SYNREG_EXPECTED_OUTPUTS = [
    "synthetic_regression_model_comparison.csv",
    "synthetic_regression_model_comparison_summary.csv",
    "synthetic_regression_summary_by_regime.csv",
    "synthetic_regression_chart_data_model_rank.csv",
]
SYNREG_CONDITIONAL_OUTPUTS = [
    "synthetic_regression_summary_by_feature_noise.csv",
    "synthetic_regression_summary_by_training_size.csv",
    "synthetic_regression_chart_data_noise_features.csv",
    "synthetic_regression_chart_data_training_size.csv",
]
SYNREG_MANIFEST_FILENAME = "synthetic_regression_aggregation_manifest.json"
SYNREG_FAILURE_FILENAME = "synthetic_regression_aggregation_failure.json"

REQUIRED_SYNREG_COLUMNS = {
    "method", "dataset_id", "split_seed", "mse_betaX",
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def safe_torch_load_with_legacy_escape_hatch(local_path: str, device) -> dict:
    """
    3-tier PyTorch load logic (mirrors evaluate.load_checkpoint_compat).
    TORCH_UNLOAD=true → sets ALLOW_UNSAFE_TORCH_LOAD=true.
    """
    if os.environ.get("TORCH_UNLOAD", "").lower() == "true":
        os.environ["ALLOW_UNSAFE_TORCH_LOAD"] = "true"
        print(
            "[WARNING] TORCH_UNLOAD=true is a temporary alias for ALLOW_UNSAFE_TORCH_LOAD=true. "
            "Use only for internally trusted best.pt checkpoints.",
            flush=True,
        )

    # Strategy 1: weights_only=True (preferred for v2 checkpoints)
    try:
        payload = torch.load(local_path, map_location=device, weights_only=True)
        return payload
    except Exception as e1:
        pass

    # Strategy 2: safe_globals([ModelConfig]) + weights_only=True
    try:
        with torch.serialization.safe_globals([ModelConfig]):
            payload = torch.load(local_path, map_location=device, weights_only=True)
        return payload
    except Exception as e2:
        pass

    # Strategy 3: weights_only=False (only if ALLOW_UNSAFE_TORCH_LOAD=true)
    if os.environ.get("ALLOW_UNSAFE_TORCH_LOAD", "").lower() == "true":
        warnings.warn(
            "Loading checkpoint with weights_only=False. Only use for trusted best.pt files.",
            UserWarning,
            stacklevel=2,
        )
        payload = torch.load(local_path, map_location=device, weights_only=False)
        return payload

    raise RuntimeError(
        f"Cannot load checkpoint at {local_path}. "
        "Set ALLOW_UNSAFE_TORCH_LOAD=true or TORCH_UNLOAD=true for legacy checkpoints."
    )


_RETIRED_FAMILIES = frozenset({"deepset", "market_aware"})

_LEGACY_CFG_FIELDS = frozenset({
    "n_sab_samp", "feature_aggregation_order", "d_sample",
    "n_sab_sample_per_feature", "sample_pool", "residual_scale_init",
})

_INDUCTIVE_REGRESSION_FAMILIES = frozenset({"market_exchangeable_icl"})


def normalize_checkpoint_cfg(payload: dict) -> "ModelConfig":
    """Extract and normalize cfg from checkpoint payload to a ModelConfig instance.

    Raises RuntimeError for retired model families (deepset, market_aware).
    Strips legacy-only cfg fields that are no longer in ModelConfig.
    """
    cfg_raw = payload.get("cfg", {})
    if isinstance(cfg_raw, ModelConfig):
        if cfg_raw.model_family in _RETIRED_FAMILIES:
            raise RuntimeError(
                f"Checkpoint uses retired model_family={cfg_raw.model_family!r}. "
                "Only 'market_exchangeable_icl' and 'market_exchangeable_completion' are supported."
            )
        return cfg_raw
    family = cfg_raw.get("model_family", "market_exchangeable_icl")
    if family in _RETIRED_FAMILIES:
        raise RuntimeError(
            f"Checkpoint uses retired model_family={family!r}. "
            "Only 'market_exchangeable_icl' and 'market_exchangeable_completion' are supported."
        )
    cfg_clean = {k: v for k, v in cfg_raw.items() if k not in _LEGACY_CFG_FIELDS}
    return ModelConfig(**cfg_clean)


def validate_checkpoint_payload(payload: dict, local_path: str) -> None:
    """Validate checkpoint payload structure and metadata consistency.

    Raises RuntimeError for hard failures.
    Warns (not raises) for missing optional fields (task_type, metadata).

    MODEL3 hard rules (format_version == 4):
    - model_arch_version must be 'model3'
    - model_design_pattern must match cfg.model_design_pattern
    - model_family must be consistent between cfg and metadata
    - task_objective must be 'inductive_regression' for synthetic regression eval
    - market_exchangeable_completion is rejected (transductive, not inductive)
    """
    import dataclasses
    if "cfg" not in payload or payload["cfg"] is None:
        raise RuntimeError(f"Checkpoint {local_path!r} missing required 'cfg' key.")
    if "state_dict" not in payload or payload["state_dict"] is None:
        raise RuntimeError(f"Checkpoint {local_path!r} missing required 'state_dict' key.")

    cfg = payload["cfg"] if isinstance(payload["cfg"], dict) else dataclasses.asdict(payload["cfg"])
    cfg_family = cfg.get("model_family", "market_exchangeable_icl")
    fmt = payload.get("checkpoint_format_version", 2)

    # Hard-reject retired families
    if cfg_family in _RETIRED_FAMILIES:
        raise RuntimeError(
            f"Checkpoint {local_path!r} uses retired model_family={cfg_family!r}. "
            "Only 'market_exchangeable_icl' and 'market_exchangeable_completion' are supported."
        )

    # Hard-reject completion checkpoints in synthetic regression evaluation
    if cfg_family == "market_exchangeable_completion":
        raise RuntimeError(
            f"Checkpoint {local_path!r} has model_family='market_exchangeable_completion'. "
            "Transductive completion checkpoints cannot be used for synthetic regression "
            "evaluation (which requires an inductive regression model). "
            "Use a checkpoint trained with model_family='market_exchangeable_icl' "
            "for synthetic regression evaluation."
        )

    meta = payload.get("metadata") or {}
    if not meta:
        warnings.warn(
            f"Checkpoint {local_path!r} has no 'metadata'. "
            "Assuming trusted internally generated checkpoint.",
            UserWarning,
        )

    meta_family = meta.get("model_family")
    if meta_family and meta_family != cfg_family:
        raise RuntimeError(
            f"Checkpoint {local_path!r}: metadata.model_family={meta_family!r} "
            f"disagrees with cfg.model_family={cfg_family!r}."
        )

    task_type = meta.get("task_type")
    if task_type is None:
        warnings.warn(
            f"Checkpoint {local_path!r} missing metadata.task_type. "
            "Assuming 'regression' for legacy checkpoint.",
            UserWarning,
        )
    elif task_type != "regression":
        raise RuntimeError(
            f"Checkpoint {local_path!r} has task_type={task_type!r}; "
            "synthetic regression evaluation requires task_type='regression'."
        )

    # MODEL3-specific hard validation
    if cfg_family in frozenset({"market_exchangeable_icl", "market_exchangeable_completion"}):
        if fmt != 4:
            raise RuntimeError(
                f"Checkpoint {local_path!r}: model_family={cfg_family!r} is a MODEL3 "
                f"family but checkpoint_format_version={fmt} (expected 4). "
                "This checkpoint was likely saved by an older version of train.py before "
                "MODEL3 format version was standardized."
            )
        meta_arch = meta.get("model_arch_version")
        cfg_arch = cfg.get("model_arch_version", "model3")
        if meta_arch and meta_arch != "model3":
            raise RuntimeError(
                f"Checkpoint {local_path!r}: MODEL3 family={cfg_family!r} but "
                f"metadata.model_arch_version={meta_arch!r} (expected 'model3')."
            )
        if cfg_arch != "model3":
            raise RuntimeError(
                f"Checkpoint {local_path!r}: MODEL3 family={cfg_family!r} but "
                f"cfg.model_arch_version={cfg_arch!r} (expected 'model3')."
            )
        meta_pattern = meta.get("model_design_pattern")
        cfg_pattern = cfg.get("model_design_pattern", "inductive_forecasting")
        if meta_pattern and meta_pattern != cfg_pattern:
            raise RuntimeError(
                f"Checkpoint {local_path!r}: metadata.model_design_pattern={meta_pattern!r} "
                f"disagrees with cfg.model_design_pattern={cfg_pattern!r}."
            )
        task_obj = meta.get("task_objective")
        if task_obj is not None and task_obj != "inductive_regression":
            raise RuntimeError(
                f"Checkpoint {local_path!r}: task_objective={task_obj!r}; "
                "synthetic regression evaluation requires task_objective='inductive_regression'."
            )


def load_best_deepset_checkpoint():
    """
    1. Download SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH to SYNREG_CKPT_LOCAL
    2. safe_torch_load_with_legacy_escape_hatch
    3. validate_checkpoint_payload
    4. normalize_checkpoint_cfg
    5. _instantiate_model(cfg).load_state_dict(payload["state_dict"])
    6. run_permutation_tests(model) — fail fast
    7. Return (model, payload)
    """
    import glob as _glob

    session = Session.builder.getOrCreate()

    ckpt_dir = str(Path(SYNREG_CKPT_LOCAL).parent)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[INFO] Checkpoint stage path: {SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH}", flush=True)
    session.file.get(SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH, ckpt_dir)

    # Resolve local checkpoint path: Snowflake may rename downloaded files.
    # Try: (1) SYNREG_CKPT_LOCAL if set explicitly, (2) filename derived from stage path,
    # (3) glob variants, (4) any .pt file in directory, (5) raise.
    if os.environ.get("SYNTHETIC_REGRESSION_CKPT_LOCAL"):
        # Explicitly set by operator — trust it.
        local_path = SYNREG_CKPT_LOCAL
    else:
        expected_filename = SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH.rsplit("/", 1)[-1]
        local_path = os.path.join(ckpt_dir, expected_filename)
        if not os.path.exists(local_path):
            # Snowflake sometimes appends suffixes; try glob variants.
            candidates = sorted(_glob.glob(local_path + "*"))
            if candidates:
                local_path = candidates[0]
        if not os.path.exists(local_path):
            # Fall back to any .pt in the directory.
            all_pts = sorted(_glob.glob(os.path.join(ckpt_dir, "*.pt")))
            if all_pts:
                local_path = all_pts[0]
        if not os.path.exists(local_path):
            raise RuntimeError(
                f"[load_best_deepset_checkpoint] Checkpoint {expected_filename!r} could not be "
                f"downloaded to {ckpt_dir!r}. "
                f"Stage path: {SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH!r}. "
                f"Verify with: LIST @MODEL_STAGE/checkpoints/;"
            )

    print(f"[INFO] Checkpoint resolved to {local_path}", flush=True)

    device = deepset_inference_device()
    payload = safe_torch_load_with_legacy_escape_hatch(local_path, device)
    validate_checkpoint_payload(payload, local_path)
    cfg = normalize_checkpoint_cfg(payload)
    print(f"[INFO] Checkpoint model_family={cfg.model_family}", flush=True)

    model = _instantiate_model(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    print("[INFO] Running permutation tests …", flush=True)
    test_results = run_permutation_tests(model)
    if isinstance(test_results, dict):
        failed = [k for k, v in test_results.items() if not v]
        if failed:
            raise RuntimeError(f"Permutation tests FAILED: {failed}")
        print(f"[INFO] All {len(test_results)} permutation tests passed.", flush=True)
    elif not test_results:
        raise RuntimeError("Permutation tests FAILED.")
    else:
        print("[INFO] All permutation tests passed.", flush=True)
    return model, payload


# ---------------------------------------------------------------------------
# Index / sharding
# ---------------------------------------------------------------------------

def _normalize_synreg_row(r) -> dict:
    """Normalise a SYNTHETIC_REGRESSION_DATASET_INDEX Snowflake row to lowercase keys.

    Snowflake Row.as_dict() may return uppercase column names (STAGE_PATH, SPLIT_SEEDS,
    etc.). Normalise to lowercase so all downstream evaluator code receives consistent keys.
    Also parses split_seeds from a JSON string to a Python list if needed.
    """
    d = {str(k).lower(): v for k, v in r.as_dict().items()}
    seeds = d.get("split_seeds")
    if isinstance(seeds, str):
        d["split_seeds"] = json.loads(seeds)
    elif seeds is None:
        d["split_seeds"] = [0]
    return d


def load_synthetic_regression_index(suite_family: str | None = None, session=None) -> list[dict]:
    """Query SYNTHETIC_REGRESSION_DATASET_INDEX for the current suite_id."""
    if session is None:
        session = Session.builder.getOrCreate()

    where = f"suite_id = '{SYNREG_SUITE_ID}'"
    if suite_family:
        where += f" AND suite_family = '{suite_family}'"

    rows = session.sql(
        f"SELECT * FROM {SYNREG_INDEX_TABLE} WHERE {where} "
        f"ORDER BY suite_id, suite_family, prior_regime, dataset_id, "
        f"feature_noise_level, target_noise_scale, stage_path"
    ).collect()
    return [_normalize_synreg_row(r) for r in rows]


def assign_synthetic_regression_shard(rows: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    """
    Modulo assignment: rows where enumerate_idx % num_shards == shard_index.
    MUST be called before any payload download.
    """
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def expand_synreg_work_items(
    rows: list[dict],
    train_size_grid: list[int] | None = None,
) -> list[dict]:
    """Expand index rows into explicit (dataset, split_seed, n_train_condition) work items.

    Each work item is a flat dict with all the row fields plus:
      - split_seed: int
      - n_train_override: int | None  (None = use dataset default)
      - is_anchor: bool

    This makes sharding and gc.collect() operate at the finest granularity,
    providing better load balance and deterministic shard assignment.
    The list is stable-ordered (dataset → seed → condition) so modulo sharding
    distributes work items across shards without inter-shard correlation.
    """
    if train_size_grid is None:
        train_size_grid = TRAIN_SIZE_GRID
    items = []
    for row in rows:
        suite_family = row.get("suite_family", "primary")
        split_seeds = row.get("split_seeds") or [0]
        if isinstance(split_seeds, str):
            split_seeds = json.loads(split_seeds)
        is_anchor = bool(row.get("training_size_anchor", False))

        if suite_family == "training_size":
            conditions = [
                {"n_train_override": nt, "is_anchor": nt == TRAIN_SIZE_ANCHOR_N}
                for nt in train_size_grid
            ]
        else:
            conditions = [{"n_train_override": None, "is_anchor": is_anchor}]

        for seed in split_seeds:
            for cond in conditions:
                item = dict(row)
                item["split_seed"] = int(seed)
                item["n_train_override"] = cond["n_train_override"]
                item["is_anchor"] = cond["is_anchor"]
                items.append(item)
    return items


SYNREG_COMPACT_WORK_ITEM_KEYS = (
    "suite_id",
    "suite_family",
    "logical_dataset_key",
    "dataset_id",
    "dataset_seed",
    "prior_regime",
    "split_seed",
    "n_train_override",
    "n_train_default",
    "n_holdout_default",
    "n_total",
    "p_signal",
    "p_noise",
    "p_total",
    "feature_noise_level",
    "target_noise_scale",
    "training_size_anchor",
    "stage_path",
)


def _json_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_compact_synreg_work_item(
    row: dict,
    *,
    session=None,
    access_mode: str | None = None,
    max_item_bytes: int | None = None,
    scoped_url_cache: dict[str, str] | None = None,
) -> dict:
    """Build the compact Ray task argument for one synthetic-regression work item.

    The dict intentionally contains metadata only. Full dataset arrays are loaded
    by the worker from the explicit dataset_access descriptor.
    """
    access_mode = access_mode or SYNREG_WORKER_DATA_ACCESS_MODE
    max_item_bytes = SYNREG_MAX_WORK_ITEM_BYTES if max_item_bytes is None else int(max_item_bytes)
    stage_path = row.get("stage_path")
    if not stage_path:
        raise RuntimeError("Cannot build compact work item without row['stage_path'].")

    item = {}
    for key in SYNREG_COMPACT_WORK_ITEM_KEYS:
        if key == "training_size_anchor":
            value = row.get("is_anchor", row.get("training_size_anchor", False))
        else:
            value = row.get(key)
        item[key] = _json_scalar(value)

    item["dataset_access"] = {
        "mode": access_mode,
        "stage_path": _json_scalar(stage_path),
    }
    if access_mode == "scoped_file_url":
        if session is None:
            raise RuntimeError(
                "build_compact_synreg_work_item requires a driver Snowpark session "
                "to derive dataset_access.scoped_url for access_mode='scoped_file_url'."
            )
        if scoped_url_cache is not None and stage_path in scoped_url_cache:
            scoped_url = scoped_url_cache[stage_path]
        else:
            scoped_url = build_scoped_dataset_file_url(session, stage_path)
            if scoped_url_cache is not None:
                scoped_url_cache[stage_path] = scoped_url
        item["dataset_access"]["scoped_url"] = scoped_url
    else:
        raise RuntimeError(
            f"Unsupported synthetic regression worker dataset access mode {access_mode!r}. "
            "Supported mode: 'scoped_file_url'."
        )

    encoded = json.dumps(item, sort_keys=True, default=str).encode("utf-8")
    if len(encoded) > max_item_bytes:
        raise RuntimeError(
            f"Compact synthetic regression work item is {len(encoded)} bytes, "
            f"exceeding SYNREG_MAX_WORK_ITEM_BYTES={max_item_bytes}. "
            "Refusing to submit unexpected large metadata through Ray."
        )
    return item


def compact_work_items_serialized_bytes(items: list[dict]) -> tuple[int, int]:
    """Return (total_bytes, max_item_bytes) for compact work-item logging."""
    if not items:
        return 0, 0
    sizes = [
        len(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))
        for item in items
    ]
    return sum(sizes), max(sizes)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet_dataset(local_path: str) -> dict:
    """Read a synthetic regression parquet file into the same dict format as np.load."""
    table = pq.read_table(local_path)
    result = {
        "X":     np.array(table["X"][0].as_py(),     dtype=np.float64),
        "y":     np.array(table["y"][0].as_py(),     dtype=np.float64),
        "betaX": np.array(table["betaX"][0].as_py(), dtype=np.float64),
    }
    for col in ("n_total", "p_signal", "p_noise", "p_total", "feature_noise_level"):
        result[col] = int(table[col][0].as_py())
    result["target_noise_scale"]   = float(table["target_noise_scale"][0].as_py())
    result["training_size_anchor"] = bool(table["training_size_anchor"][0].as_py())
    for col in ("suite_family", "prior_regime"):
        result[col] = str(table[col][0].as_py())
    return result


def _stage_path_to_local_name(stage_path: str) -> str:
    """Convert full stage path to a flat, collision-free local filename.

    @EVALUATION_DATASET_STAGE/ood_parity/E/dataset_0000.parquet
    → ood_parity__E__dataset_0000.parquet

    @EVALUATION_DATASET_STAGE/primary/dataset_0000.parquet
    → primary__dataset_0000.parquet
    """
    # Strip the @STAGE_NAME/ prefix
    clean = re.sub(r"^@[^/]+/", "", stage_path)
    # Replace path separators with double underscores
    return clean.replace("/", "__")


def _split_stage_path(stage_path: str) -> tuple[str, str]:
    if not stage_path or not str(stage_path).startswith("@"):
        raise RuntimeError(f"Expected Snowflake stage path beginning with '@'; got {stage_path!r}.")
    raw = str(stage_path)
    if "/" not in raw:
        raise RuntimeError(f"Stage path {stage_path!r} does not include a relative file path.")
    stage_name, relative_path = raw.split("/", 1)
    if not relative_path:
        raise RuntimeError(f"Stage path {stage_path!r} does not include a relative file path.")
    return stage_name, relative_path


def _sql_string_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_scoped_dataset_file_url(session, stage_path: str) -> str:
    """Driver-only helper that derives a session-free worker file descriptor."""
    stage_name, relative_path = _split_stage_path(stage_path)
    rows = session.sql(
        f"SELECT BUILD_SCOPED_FILE_URL({stage_name}, {_sql_string_literal(relative_path)}) AS SCOPED_URL"
    ).collect()
    if not rows:
        raise RuntimeError(f"BUILD_SCOPED_FILE_URL returned no rows for {stage_path!r}.")
    row = rows[0]
    if hasattr(row, "as_dict"):
        d = {str(k).lower(): v for k, v in row.as_dict().items()}
        scoped_url = d.get("scoped_url")
    else:
        scoped_url = row[0]
    if not scoped_url:
        raise RuntimeError(f"BUILD_SCOPED_FILE_URL returned an empty URL for {stage_path!r}.")
    return str(scoped_url)


def _validate_dataset_metadata(payload: dict, row: dict) -> None:
    """Raise RuntimeError if payload metadata disagrees with index row."""
    checks = [
        ("suite_family", str),
        ("prior_regime", str),
        ("n_total", int),
        ("p_signal", int),
        ("p_noise", int),
        ("p_total", int),
        ("feature_noise_level", int),
        ("target_noise_scale", float),
    ]
    mismatches = []
    for key, cast in checks:
        row_val = row.get(key)
        payload_val = payload.get(key)
        if row_val is None or payload_val is None:
            continue
        if cast(row_val) != cast(payload_val):
            mismatches.append(f"{key}: index={row_val!r}, file={payload_val!r}")
    if mismatches:
        stage_path = row.get("stage_path", "?")
        raise RuntimeError(
            f"Dataset metadata mismatch for {stage_path}: "
            + "; ".join(mismatches)
        )


def _load_prepared_synthetic_dataset_from_stage(
    row: dict,
    *,
    stage_path: str,
    local_cache_dir: str,
    session,
) -> dict:
    os.makedirs(local_cache_dir, exist_ok=True)
    safe_name = _stage_path_to_local_name(stage_path)
    local_path = os.path.join(local_cache_dir, safe_name)

    if not os.path.exists(local_path):
        with tempfile.TemporaryDirectory() as _tmp:
            session.file.get(stage_path, _tmp)
            shutil.move(os.path.join(_tmp, Path(stage_path).name), local_path)

    return _read_dataset_local_file(row, local_path)


def _read_dataset_local_file(row: dict, local_path: str) -> dict:
    if local_path.endswith(".parquet"):
        result = _load_parquet_dataset(local_path)
    else:
        data = np.load(local_path, allow_pickle=False)
        result = dict(data)
        # Inflate scalar metadata from 1-element arrays
        for key in ("n_total", "p_signal", "p_noise", "p_total", "feature_noise_level"):
            if key in result:
                result[key] = int(result[key][0])
        for key in ("target_noise_scale",):
            if key in result:
                result[key] = float(result[key][0])
        for key in ("training_size_anchor",):
            if key in result:
                result[key] = bool(result[key][0])
        for key in ("suite_family", "prior_regime"):
            if key in result:
                result[key] = str(result[key][0])

    # Validate metadata consistency between index row and file payload
    _validate_dataset_metadata(result, row)

    # Attach index row metadata
    result["n_train_default"] = int(row.get("n_train_default", result["n_total"] * 4 // 5))
    result["n_holdout_default"] = int(row.get("n_holdout_default", result["n_total"] // 5))
    result["split_seeds"] = row.get("split_seeds", [0])
    result["dataset_id"] = row.get("dataset_id")
    result["dataset_seed"] = row.get("dataset_seed")
    return result


def load_prepared_synthetic_dataset(row: dict, local_cache_dir: str = SYNREG_LOCAL_CACHE) -> dict:
    """Download dataset from row['stage_path'] using a local Snowpark session."""
    session = Session.builder.getOrCreate()
    return _load_prepared_synthetic_dataset_from_stage(
        row,
        stage_path=row["stage_path"],
        local_cache_dir=local_cache_dir,
        session=session,
    )


def load_prepared_synthetic_dataset_from_access(
    item_meta: dict,
    local_cache_dir: str = SYNREG_LOCAL_CACHE,
) -> dict:
    """Load a worker dataset through the explicit dataset_access descriptor.

    Current production mode is ``scoped_file_url``: the driver derives a scoped
    URL from the stage path, and Ray workers open the URL with SnowflakeFile
    without creating Snowpark sessions. Workers must not query
    SYNTHETIC_REGRESSION_DATASET_INDEX.
    """
    dataset_access = item_meta.get("dataset_access") or {}
    access_mode = dataset_access.get("mode", SYNREG_WORKER_DATA_ACCESS_MODE)
    stage_path = dataset_access.get("stage_path") or item_meta.get("stage_path")
    scoped_url = dataset_access.get("scoped_url")
    if not stage_path:
        raise RuntimeError("dataset_access.stage_path is required for worker dataset loading.")
    if access_mode != "scoped_file_url":
        raise RuntimeError(
            f"Unsupported synthetic regression worker dataset access mode {access_mode!r}. "
            "Supported mode: 'scoped_file_url'."
        )
    if not scoped_url:
        raise RuntimeError("dataset_access.scoped_url is required for access_mode='scoped_file_url'.")

    os.makedirs(local_cache_dir, exist_ok=True)
    dataset_key = _stage_path_to_local_name(stage_path)
    local_path = os.path.join(local_cache_dir, dataset_key)

    if not os.path.exists(local_path):
        try:
            from snowflake.snowpark.files import SnowflakeFile
        except ImportError as exc:
            raise RuntimeError(
                "snowflake.snowpark.files.SnowflakeFile is required for "
                "session-free scoped_file_url worker dataset access."
            ) from exc
        tmp_path = os.path.join(local_cache_dir, f".{dataset_key}.{os.getpid()}.tmp")
        try:
            with open(tmp_path, "wb") as dst:
                with SnowflakeFile.open(scoped_url, "rb", require_scoped_url=True) as src:
                    shutil.copyfileobj(src, dst)
            os.replace(tmp_path, local_path)
            tmp_path = None  # published; do not delete in except
        except Exception:
            try:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            if not os.path.exists(local_path):
                raise
            # Another worker published first; proceed to read their complete file

    # Pre-read validation
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        raise RuntimeError(
            f"Dataset cache file missing or empty for {stage_path!r}: {local_path}"
        )
    if local_path.endswith(".parquet"):
        try:
            pq.read_schema(local_path)
        except Exception as exc:
            raise RuntimeError(
                f"Dataset cache file is not valid Parquet for {stage_path!r}: {exc}"
            ) from exc

    return _read_dataset_local_file(item_meta, local_path)


def build_split_for_seed(data: dict, split_seed: int, n_train_override: int | None = None) -> dict:
    """
    Deterministic row split for a given split_seed.
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)
    If n_train_override: use that many rows (training_size suite).
    """
    n_total = int(data["n_total"])
    n_holdout = int(data["n_holdout_default"])

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)

    if n_train_override is not None:
        n_train = n_train_override
    else:
        n_train = n_total - n_holdout

    train_idx = perm[:n_train]
    holdout_idx = perm[n_total - n_holdout:]

    X = data["X"]
    y = data["y"]
    betaX = data["betaX"]

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "betaX_train": betaX[train_idx],
        "X_holdout": X[holdout_idx],
        "y_holdout": y[holdout_idx],
        "betaX_holdout": betaX[holdout_idx],
        "n_train": n_train,
        "n_holdout": n_holdout,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_train_only(
    X_train: np.ndarray, X_holdout: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit StandardScaler on X_train only.
    Apply same transform to X_holdout.
    Handle NaN via np.nan_to_num after transform.
    Returns (X_train_proc, X_holdout_proc) as float64.
    """
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train).astype(np.float64)
    X_holdout_proc = scaler.transform(X_holdout).astype(np.float64)
    X_train_proc = np.nan_to_num(X_train_proc, nan=0.0, posinf=0.0, neginf=0.0)
    X_holdout_proc = np.nan_to_num(X_holdout_proc, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train_proc, X_holdout_proc


def apply_deepset_feature_selection(
    X_train_p: np.ndarray,
    y_train: np.ndarray,
    X_holdout_p: np.ndarray,
    feature_cap: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Wraps the shared deepset_inference train-only feature selector.
    Only applies if X_train_p.shape[1] > feature_cap.
    Returns (X_train_sel, X_holdout_sel, meta_dict).
    selected_features <= feature_cap guaranteed.
    """
    raw_features = X_train_p.shape[1]
    meta = {
        "raw_features": raw_features,
        "signal_features": None,
        "noise_features": None,
        "processed_features": raw_features,
        "selected_features": raw_features,
        "feature_selector": "none",
        "feature_cap": feature_cap,
    }

    if raw_features <= feature_cap:
        return X_train_p, X_holdout_p, meta

    X_train_sel, X_holdout_sel, sel_meta = select_deepset_features_train_only(
        X_train_p, y_train, X_holdout_p, feature_cap, SYNREG_FEATURE_SELECTOR
    )
    meta.update(sel_meta)
    meta["processed_features"] = raw_features
    meta["selected_features"] = X_train_sel.shape[1]
    meta["feature_selector"] = SYNREG_FEATURE_SELECTOR
    meta["feature_cap"] = feature_cap
    return X_train_sel, X_holdout_sel, meta


# ---------------------------------------------------------------------------
# Memory guards
# ---------------------------------------------------------------------------

def memory_guard_matrix(X_train_p: np.ndarray, X_holdout_p: np.ndarray) -> str | None:
    """
    Returns skip_reason string or None.
    Checks SYNREG_CPU_MAX_FEATURES and SYNREG_CPU_MAX_MATRIX_BYTES thresholds.
    """
    n_feat = X_train_p.shape[1]
    if n_feat > SYNREG_CPU_MAX_FEATURES:
        return f"cpu_matrix_too_large (features={n_feat} > {SYNREG_CPU_MAX_FEATURES})"

    matrix_bytes = (X_train_p.nbytes + X_holdout_p.nbytes)
    if matrix_bytes > SYNREG_CPU_MAX_MATRIX_BYTES:
        return (
            f"cpu_matrix_too_large "
            f"(matrix_bytes={matrix_bytes} > {SYNREG_CPU_MAX_MATRIX_BYTES})"
        )
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_pred: np.ndarray,
    betaX_holdout: np.ndarray,
    y_observed: np.ndarray | None = None,
) -> dict:
    """
    Primary: mse_betaX, rmse_betaX, mae_betaX, r2_betaX
    Secondary: mse_y_observed, rmse_y_observed, mae_y_observed, r2_y_observed (NaN if y_observed not provided)
    """
    mse_b = float(mean_squared_error(betaX_holdout, y_pred))
    rmse_b = float(math.sqrt(mse_b))
    mae_b = float(mean_absolute_error(betaX_holdout, y_pred))
    r2_b = float(r2_score(betaX_holdout, y_pred))

    if y_observed is not None:
        mse_y = float(mean_squared_error(y_observed, y_pred))
        rmse_y = float(math.sqrt(mse_y))
        mae_y = float(mean_absolute_error(y_observed, y_pred))
        r2_y = float(r2_score(y_observed, y_pred))
    else:
        mse_y = rmse_y = mae_y = r2_y = float("nan")

    return {
        "mse_betaX": mse_b,
        "rmse_betaX": rmse_b,
        "mae_betaX": mae_b,
        "r2_betaX": r2_b,
        "mse_y_observed": mse_y,
        "rmse_y_observed": rmse_y,
        "mae_y_observed": mae_y,
        "r2_y_observed": r2_y,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_CANONICAL_COLUMNS = [
    # Identification
    "suite_id", "suite_family", "dataset_id", "dataset_seed", "split_seed",
    "prior_regime", "method", "logical_dataset_key",
    # Condition
    "n_total", "n_train", "n_holdout", "p_signal", "p_noise", "p_total",
    "feature_noise_level", "target_noise_scale", "training_size_anchor",
    # Feature/DeepSet metadata
    "raw_features", "signal_features", "noise_features", "processed_features",
    "selected_features", "feature_selector", "feature_cap", "d_phi", "d_rho",
    "pool", "context_windows", "context_window_size", "test_chunk_size", "mc_k",
    # Metrics
    "mse_betaX", "rmse_betaX", "mae_betaX", "r2_betaX",
    "mse_y_observed", "rmse_y_observed", "mae_y_observed", "r2_y_observed",
    # Timing
    "fit_time_s", "predict_time_s", "total_time_s",
    # Status
    "status", "skip_reason", "error_type", "error_message",
]


def _empty_row() -> dict:
    return {col: None for col in _CANONICAL_COLUMNS}


def _skipped_row(base: dict, skip_reason: str) -> dict:
    row = _empty_row()
    row.update(base)
    row["status"] = "skipped"
    row["skip_reason"] = skip_reason
    row["mse_betaX"] = float("nan")
    row["rmse_betaX"] = float("nan")
    return row


def _failed_row(base: dict, error_type: str, error_message: str) -> dict:
    row = _empty_row()
    row.update(base)
    row["status"] = "failed"
    row["error_type"] = error_type
    row["error_message"] = error_message
    row["mse_betaX"] = float("nan")
    row["rmse_betaX"] = float("nan")
    return row


def _base_autogluon_row(item_meta: dict) -> dict:
    dataset_id = item_meta.get("dataset_id")
    prior_regime = item_meta.get("prior_regime")
    suite_id = item_meta.get("suite_id")
    logical_dataset_key = item_meta.get("logical_dataset_key")
    if not logical_dataset_key and dataset_id is not None:
        logical_dataset_key = f"{suite_id}:{prior_regime}:{int(dataset_id):04d}"
    return {
        "suite_id": suite_id,
        "suite_family": item_meta.get("suite_family", "primary"),
        "dataset_id": dataset_id,
        "dataset_seed": item_meta.get("dataset_seed"),
        "split_seed": int(item_meta.get("split_seed", 0)),
        "prior_regime": prior_regime,
        "method": "AutoGluon",
        "logical_dataset_key": logical_dataset_key,
    }


def write_part_csv_to_stage(
    session,
    rows: list[dict],
    method: str,
    shard_index: int,
    num_shards: int,
) -> str:
    """
    Write rows to:
      {SYNREG_RESULTS_STAGE}/{method}_shard{shard_index}_of_{num_shards}_detailed.csv
    Returns the stage path.
    """
    if not rows:
        raise RuntimeError(
            f"[write_part_csv_to_stage] '{method}' shard {shard_index}/{num_shards}: "
            "output_rows is empty — no datasets were processed by this shard. "
            "Check that SYNTHETIC_REGRESSION_DATASET_INDEX contains rows for the "
            "configured SYNTHETIC_REGRESSION_SUITE_ID, and that the database/schema "
            "context inside the container matches where the index was written."
        )
    safe_method = method.replace(" ", "_").replace("/", "_")
    filename = f"{safe_method}_shard{shard_index}_of_{num_shards}_detailed.csv"
    local_tmp = os.path.join(tempfile.gettempdir(), filename)

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    # Canonical first, then extras alphabetically
    fieldnames = [c for c in _CANONICAL_COLUMNS if c in all_keys]
    extras = sorted(all_keys - set(fieldnames))
    fieldnames = fieldnames + extras

    with open(local_tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    session.file.put(
        local_tmp,
        SYNREG_RESULTS_STAGE,
        auto_compress=False,
        overwrite=True,
    )
    stage_path = f"{SYNREG_RESULTS_STAGE}/{filename}"
    print(f"[INFO] Wrote {len(rows)} rows to {stage_path}")
    return stage_path


# ---------------------------------------------------------------------------
# Mode: DeepSet
# ---------------------------------------------------------------------------

def run_deepset_synthetic_regression() -> None:
    """
    Run DeepSet bounded-context ensemble evaluation across all suite families.
    GPU mode; sharded.
    """
    model, payload = load_best_deepset_checkpoint()
    device = deepset_inference_device()

    if SYNREG_RUN_CHECKPOINT_GATES:
        from sanity_checks import run_checkpoint_gates, save_results as _save_gate_results
        _gate_device = next(model.parameters()).device
        _gate_results = run_checkpoint_gates(
            model=model,
            device=_gate_device,
            checkpoint_metadata=payload.get("metadata"),
            max_ridge_ratio=SYNREG_GATE_MAX_RIDGE_RATIO,
            min_query_std=SYNREG_GATE_MIN_QUERY_STD,
        )
        _save_gate_results(_gate_results, SYNREG_GATE_OUT_DIR)
        if not _gate_results.get("all_passed") and SYNREG_CHECKPOINT_GATE_STRICT:
            failed = [k for k, v in _gate_results.items()
                      if isinstance(v, dict) and not v.get("passed", True)]
            raise RuntimeError(
                f"Checkpoint quality gates FAILED before evaluation: {failed}. "
                f"Set SYNREG_CHECKPOINT_GATE_STRICT=false to downgrade to warning."
            )
        elif not _gate_results.get("all_passed"):
            print(
                f"[WARNING] Checkpoint quality gates failed (continuing): "
                f"{[k for k, v in _gate_results.items() if isinstance(v, dict) and not v.get('passed', True)]}",
                flush=True,
            )

    session = Session.builder.getOrCreate()

    all_rows_index = load_synthetic_regression_index(suite_family=None, session=session)
    # Expand to explicit (dataset, split_seed, n_train_condition) work items
    # before sharding so load is distributed at the finest granularity.
    all_work_items = expand_synreg_work_items(all_rows_index, TRAIN_SIZE_GRID)
    my_items = assign_synthetic_regression_shard(all_work_items, SYNREG_SHARD_INDEX, SYNREG_NUM_SHARDS)
    print(
        f"[INFO] MODEL3 shard {SYNREG_SHARD_INDEX}/{SYNREG_NUM_SHARDS}: "
        f"{len(all_rows_index)} index rows → {len(all_work_items)} work items → "
        f"{len(my_items)} assigned to this shard."
    )

    output_rows: list[dict] = []

    for item in my_items:
        suite_family = item.get("suite_family", "primary")
        dataset_id = item.get("dataset_id")
        dataset_seed = item.get("dataset_seed")
        prior_regime = item.get("prior_regime")
        split_seed = int(item["split_seed"])
        n_train_override = item.get("n_train_override")
        is_anchor = bool(item.get("is_anchor", False))

        # Build index row subset (without work-item expansion keys) for data loading
        row = {k: v for k, v in item.items()
               if k not in ("split_seed", "n_train_override", "is_anchor")}

        try:
            data = load_prepared_synthetic_dataset(row, SYNREG_LOCAL_CACHE)
        except Exception as e:
            base = {
                "suite_id": SYNREG_SUITE_ID, "suite_family": suite_family,
                "dataset_id": dataset_id, "dataset_seed": dataset_seed,
                "split_seed": split_seed, "prior_regime": prior_regime,
                "method": "MODEL3-ICL-MC",
            }
            output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
            del data
            gc.collect()
            continue

        n_total = int(data["n_total"])
        p_signal = int(data.get("p_signal", data["X"].shape[1]))
        p_noise = int(data.get("p_noise", 0))
        p_total = int(data.get("p_total", data["X"].shape[1]))
        feature_noise_level = int(data.get("feature_noise_level", 0))
        target_noise_scale = float(data.get("target_noise_scale", 1.0))

        try:
            split = build_split_for_seed(data, split_seed, n_train_override)
        except Exception as e:
            base = {
                "suite_id": SYNREG_SUITE_ID, "suite_family": suite_family,
                "dataset_id": dataset_id, "dataset_seed": dataset_seed,
                "split_seed": split_seed, "prior_regime": prior_regime,
                "method": "MODEL3-ICL-MC",
            }
            output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
            del data, split
            gc.collect()
            continue
        finally:
            del data

        X_train, y_train = split["X_train"], split["y_train"]
        X_holdout = split["X_holdout"]
        betaX_holdout = split["betaX_holdout"]
        y_holdout = split["y_holdout"]
        n_train = split["n_train"]
        n_holdout = split["n_holdout"]
        del split

        base = {
            "suite_id": SYNREG_SUITE_ID,
            "suite_family": suite_family,
            "dataset_id": dataset_id,
            "dataset_seed": dataset_seed,
            "split_seed": split_seed,
            "prior_regime": prior_regime,
            "method": "MODEL3-ICL-MC",
            "logical_dataset_key": (
                row.get("logical_dataset_key")
                or f"{SYNREG_SUITE_ID}:{prior_regime}:{dataset_id:04d}"
            ),
            "n_total": n_total,
            "n_train": n_train,
            "n_holdout": n_holdout,
            "p_signal": p_signal,
            "p_noise": p_noise,
            "p_total": p_total,
            "feature_noise_level": feature_noise_level,
            "target_noise_scale": target_noise_scale,
            "training_size_anchor": is_anchor,
        }

        # Preprocess
        try:
            X_train_p, X_holdout_p = preprocess_train_only(X_train, X_holdout)
        except Exception as e:
            output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
            del X_train, y_train, X_holdout, betaX_holdout, y_holdout
            gc.collect()
            continue
        del X_train, X_holdout

        # Feature selection
        feat_meta = {
            "raw_features": X_train_p.shape[1],
            "processed_features": X_train_p.shape[1],
            "selected_features": X_train_p.shape[1],
            "feature_selector": "none",
            "feature_cap": SYNREG_DEEPSET_FEATURE_CAP,
        }
        try:
            X_train_sel, X_holdout_sel, feat_meta = apply_deepset_feature_selection(
                X_train_p, y_train, X_holdout_p, SYNREG_DEEPSET_FEATURE_CAP
            )
        except Exception as e:
            output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
            del X_train_p, X_holdout_p, y_train, betaX_holdout, y_holdout
            gc.collect()
            continue
        del X_train_p, X_holdout_p

        # GPU memory check (MODEL3-aware: passes model for H-tensor estimate)
        skip_reason, _ = deepset_gpu_memory_skip_reason(
            X_train_sel, X_holdout_sel, device, model=model
        )
        if skip_reason:
            row_out = _skipped_row(base, f"gpu_oom:{skip_reason}")
            row_out.update(feat_meta)
            output_rows.append(row_out)
            del X_train_sel, X_holdout_sel, y_train, betaX_holdout, y_holdout
            gc.collect()
            continue

        # Inference
        cfg = model.cfg if hasattr(model, "cfg") else None
        d_phi = int(cfg.d_phi) if cfg and hasattr(cfg, "d_phi") else None
        d_rho = int(cfg.d_rho) if cfg and hasattr(cfg, "d_rho") else None
        pool = str(cfg.pool) if cfg and hasattr(cfg, "pool") else None

        t0 = time.perf_counter()
        try:
            y_pred = predict_deepset_bounded_context_ensemble(
                model=model,
                X_train_np=X_train_sel,
                y_train_np=y_train,
                X_test_np=X_holdout_sel,
                seed=int(split_seed),
                dataset_identity=f"synreg_{dataset_id}",
                K=SYNREG_MC_K,
                context_size=SYNREG_CONTEXT_SIZE,
                context_ensembles=SYNREG_CONTEXT_ENSEMBLES,
                test_batch_size=SYNREG_TEST_BATCH_SIZE,
                device=device,
            )
        except Exception as e:
            output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
            del X_train_sel, X_holdout_sel, y_train, betaX_holdout, y_holdout
            gc.collect()
            continue
        predict_time = time.perf_counter() - t0

        metrics = compute_regression_metrics(y_pred, betaX_holdout, y_observed=y_holdout)

        row_out = _empty_row()
        row_out.update(base)
        row_out.update(feat_meta)
        row_out.update(metrics)
        row_out.update({
            "status": "ok",
            "d_phi": d_phi,
            "d_rho": d_rho,
            "pool": pool,
            "context_windows": SYNREG_CONTEXT_ENSEMBLES,
            "context_window_size": SYNREG_CONTEXT_SIZE,
            "test_chunk_size": SYNREG_TEST_BATCH_SIZE,
            "mc_k": SYNREG_MC_K,
            "fit_time_s": float("nan"),
            "predict_time_s": predict_time,
            "total_time_s": predict_time,
            "signal_features": p_signal,
            "noise_features": p_noise,
        })
        output_rows.append(row_out)

        del X_train_sel, X_holdout_sel, y_train, betaX_holdout, y_holdout, y_pred
        gc.collect()

    write_part_csv_to_stage(
        session, output_rows, "MODEL3-ICL-MC", SYNREG_SHARD_INDEX, SYNREG_NUM_SHARDS
    )
    print(f"[INFO] MODEL3 shard {SYNREG_SHARD_INDEX} complete: {len(output_rows)} rows.")


# ---------------------------------------------------------------------------
# Mode: Baselines
# ---------------------------------------------------------------------------

def run_baseline_synthetic_regression() -> None:
    """
    Dataset-first loop. All SYNREG_BASELINE_METHODS in one pass per dataset-seed-condition.
    CPU shards: 3.
    """
    validate_baseline_dependencies(
        method for method in SYNREG_BASELINE_METHODS if method != "FixedRidgeLambda1"
    )
    session = Session.builder.getOrCreate()

    all_rows_index = load_synthetic_regression_index(suite_family=None, session=session)
    my_rows = assign_synthetic_regression_shard(all_rows_index, SYNREG_SHARD_INDEX, SYNREG_NUM_SHARDS)
    print(
        f"[INFO] Baseline shard {SYNREG_SHARD_INDEX}/{SYNREG_NUM_SHARDS}: "
        f"{len(my_rows)} index rows."
    )

    output_rows: list[dict] = []

    for row in my_rows:
        suite_family = row.get("suite_family", "primary")
        dataset_id = row.get("dataset_id")
        dataset_seed = row.get("dataset_seed")
        prior_regime = row.get("prior_regime")
        split_seeds = row.get("split_seeds") or [0]
        if isinstance(split_seeds, str):
            split_seeds = json.loads(split_seeds)

        try:
            data = load_prepared_synthetic_dataset(row, SYNREG_LOCAL_CACHE)
        except Exception as e:
            for method in SYNREG_BASELINE_METHODS:
                base = {
                    "suite_id": SYNREG_SUITE_ID, "suite_family": suite_family,
                    "dataset_id": dataset_id, "dataset_seed": dataset_seed,
                    "prior_regime": prior_regime, "method": method,
                }
                output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
            continue

        n_total = int(data["n_total"])
        p_signal = int(data.get("p_signal", data["X"].shape[1]))
        p_noise = int(data.get("p_noise", 0))
        p_total = int(data.get("p_total", data["X"].shape[1]))
        feature_noise_level = int(data.get("feature_noise_level", 0))
        target_noise_scale = float(data.get("target_noise_scale", 1.0))
        is_anchor = bool(data.get("training_size_anchor", False))

        if suite_family == "training_size":
            conditions = [
                {"n_train_override": nt, "is_anchor": nt == TRAIN_SIZE_ANCHOR_N}
                for nt in TRAIN_SIZE_GRID
            ]
        else:
            conditions = [{"n_train_override": None, "is_anchor": is_anchor}]

        for split_seed in split_seeds:
            for cond in conditions:
                n_train_override = cond["n_train_override"]

                try:
                    split = build_split_for_seed(data, split_seed, n_train_override)
                except Exception as e:
                    for method in SYNREG_BASELINE_METHODS:
                        base = {
                            "suite_id": SYNREG_SUITE_ID, "suite_family": suite_family,
                            "dataset_id": dataset_id, "dataset_seed": dataset_seed,
                            "split_seed": split_seed, "prior_regime": prior_regime,
                            "method": method,
                        }
                        output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
                    continue

                X_train, y_train = split["X_train"], split["y_train"]
                X_holdout = split["X_holdout"]
                betaX_holdout = split["betaX_holdout"]
                y_holdout = split["y_holdout"]
                n_train = split["n_train"]
                n_holdout = split["n_holdout"]

                try:
                    X_train_p, X_holdout_p = preprocess_train_only(X_train, X_holdout)
                except Exception as e:
                    for method in SYNREG_BASELINE_METHODS:
                        base = {
                            "suite_id": SYNREG_SUITE_ID, "suite_family": suite_family,
                            "dataset_id": dataset_id, "dataset_seed": dataset_seed,
                            "split_seed": split_seed, "prior_regime": prior_regime,
                            "method": method, "n_total": n_total, "n_train": n_train,
                            "n_holdout": n_holdout,
                        }
                        output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
                    continue

                skip_reason = memory_guard_matrix(X_train_p, X_holdout_p)
                if skip_reason:
                    for method in SYNREG_BASELINE_METHODS:
                        base = {
                            "suite_id": SYNREG_SUITE_ID, "suite_family": suite_family,
                            "dataset_id": dataset_id, "dataset_seed": dataset_seed,
                            "split_seed": split_seed, "prior_regime": prior_regime,
                            "method": method, "n_total": n_total, "n_train": n_train,
                            "n_holdout": n_holdout, "p_signal": p_signal,
                            "p_noise": p_noise, "p_total": p_total,
                            "feature_noise_level": feature_noise_level,
                            "target_noise_scale": target_noise_scale,
                            "training_size_anchor": is_anchor,
                        }
                        output_rows.append(_skipped_row(base, skip_reason))
                    del X_train_p, X_holdout_p
                    gc.collect()
                    continue

                # Run each baseline
                for method in SYNREG_BASELINE_METHODS:
                    base = {
                        "suite_id": SYNREG_SUITE_ID,
                        "suite_family": suite_family,
                        "dataset_id": dataset_id,
                        "dataset_seed": dataset_seed,
                        "split_seed": split_seed,
                        "prior_regime": prior_regime,
                        "method": method,
                        "logical_dataset_key": (
                        row.get("logical_dataset_key")
                        or f"{SYNREG_SUITE_ID}:{prior_regime}:{dataset_id:04d}"
                    ),
                        "n_total": n_total,
                        "n_train": n_train,
                        "n_holdout": n_holdout,
                        "p_signal": p_signal,
                        "p_noise": p_noise,
                        "p_total": p_total,
                        "feature_noise_level": feature_noise_level,
                        "target_noise_scale": target_noise_scale,
                        "training_size_anchor": is_anchor,
                    }

                    try:
                        # FixedRidgeLambda1 uses Ridge(alpha=1.0)
                        if method == "FixedRidgeLambda1":
                            reg = Ridge(alpha=1.0)
                        else:
                            reg = make_baseline_model(method, seed=int(split_seed))

                        t_fit_start = time.perf_counter()
                        reg.fit(X_train_p, y_train)
                        fit_time = time.perf_counter() - t_fit_start

                        t_pred_start = time.perf_counter()
                        y_pred = reg.predict(X_holdout_p)
                        predict_time = time.perf_counter() - t_pred_start

                        metrics = compute_regression_metrics(
                            y_pred, betaX_holdout, y_observed=y_holdout
                        )
                        row_out = _empty_row()
                        row_out.update(base)
                        row_out.update(metrics)
                        row_out.update({
                            "status": "ok",
                            "fit_time_s": fit_time,
                            "predict_time_s": predict_time,
                            "total_time_s": fit_time + predict_time,
                            "processed_features": X_train_p.shape[1],
                            "raw_features": X_train_p.shape[1],
                        })
                        output_rows.append(row_out)

                    except Exception as e:
                        output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))

                del X_train_p, X_holdout_p
                gc.collect()

        gc.collect()

    write_part_csv_to_stage(
        session, output_rows, "baselines", SYNREG_SHARD_INDEX, SYNREG_NUM_SHARDS
    )
    print(f"[INFO] Baseline shard {SYNREG_SHARD_INDEX} complete: {len(output_rows)} rows.")


# ---------------------------------------------------------------------------
# Mode: AutoGluon
# ---------------------------------------------------------------------------

def _check_tmp_free_bytes() -> int:
    """Return free bytes in /tmp."""
    usage = shutil.disk_usage("/tmp")
    return usage.free


def run_autogluon_synthetic_regression() -> None:
    """
    30 shards. One dataset at a time.
    Cleans AutoGluon /tmp artifacts after each fit.
    """
    get_tabular_predictor_class()
    session = Session.builder.getOrCreate()

    all_rows_index = load_synthetic_regression_index(suite_family=None, session=session)
    my_rows = assign_synthetic_regression_shard(all_rows_index, SYNREG_SHARD_INDEX, SYNREG_NUM_SHARDS)
    print(
        f"[INFO] AutoGluon shard {SYNREG_SHARD_INDEX}/{SYNREG_NUM_SHARDS}: "
        f"{len(my_rows)} index rows."
    )

    output_rows: list[dict] = []

    for row in my_rows:
        suite_family = row.get("suite_family", "primary")
        dataset_id = row.get("dataset_id")
        dataset_seed = row.get("dataset_seed")
        prior_regime = row.get("prior_regime")
        split_seeds = row.get("split_seeds") or [0]
        if isinstance(split_seeds, str):
            split_seeds = json.loads(split_seeds)

        try:
            data = load_prepared_synthetic_dataset(row, SYNREG_LOCAL_CACHE)
        except Exception as e:
            base = {
                "suite_id": SYNREG_SUITE_ID, "suite_family": suite_family,
                "dataset_id": dataset_id, "dataset_seed": dataset_seed,
                "prior_regime": prior_regime, "method": "AutoGluon",
            }
            output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
            continue

        n_total = int(data["n_total"])
        p_signal = int(data.get("p_signal", data["X"].shape[1]))
        p_noise = int(data.get("p_noise", 0))
        p_total = int(data.get("p_total", data["X"].shape[1]))
        feature_noise_level = int(data.get("feature_noise_level", 0))
        target_noise_scale = float(data.get("target_noise_scale", 1.0))
        is_anchor = bool(data.get("training_size_anchor", False))

        if suite_family == "training_size":
            conditions = [
                {"n_train_override": nt, "is_anchor": nt == TRAIN_SIZE_ANCHOR_N}
                for nt in TRAIN_SIZE_GRID
            ]
        else:
            conditions = [{"n_train_override": None, "is_anchor": is_anchor}]

        for split_seed in split_seeds:
            for cond in conditions:
                n_train_override = cond["n_train_override"]
                base = {
                    "suite_id": SYNREG_SUITE_ID,
                    "suite_family": suite_family,
                    "dataset_id": dataset_id,
                    "dataset_seed": dataset_seed,
                    "split_seed": split_seed,
                    "prior_regime": prior_regime,
                    "method": "AutoGluon",
                    "logical_dataset_key": (
                        row.get("logical_dataset_key")
                        or f"{SYNREG_SUITE_ID}:{prior_regime}:{dataset_id:04d}"
                    ),
                    "n_total": n_total,
                    "n_holdout": int(data["n_holdout_default"]),
                    "p_signal": p_signal,
                    "p_noise": p_noise,
                    "p_total": p_total,
                    "feature_noise_level": feature_noise_level,
                    "target_noise_scale": target_noise_scale,
                    "training_size_anchor": cond["is_anchor"],
                }

                # Check /tmp free space
                free_bytes = _check_tmp_free_bytes()
                if free_bytes < SYNREG_AG_MIN_TMP_FREE_BYTES:
                    row_out = _skipped_row(
                        base,
                        f"insufficient_tmp_space (free={free_bytes} < {SYNREG_AG_MIN_TMP_FREE_BYTES})"
                    )
                    output_rows.append(row_out)
                    continue

                try:
                    split = build_split_for_seed(data, split_seed, n_train_override)
                except Exception as e:
                    output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
                    continue

                X_train, y_train = split["X_train"], split["y_train"]
                X_holdout = split["X_holdout"]
                betaX_holdout = split["betaX_holdout"]
                y_holdout = split["y_holdout"]
                n_train = split["n_train"]
                n_holdout = split["n_holdout"]
                base["n_train"] = n_train
                base["n_holdout"] = n_holdout

                try:
                    X_train_p, X_holdout_p = preprocess_train_only(X_train, X_holdout)

                    skip_reason = memory_guard_matrix(X_train_p, X_holdout_p)
                    if skip_reason:
                        output_rows.append(_skipped_row(base, skip_reason))
                        continue

                    ag_path = make_unique_autogluon_model_dir(
                        {"dataset_id": dataset_id, "split_seed": split_seed, "prior_regime": prior_regime},
                        prefix="ag_synreg",
                    )
                    y_pred, fit_time, predict_time = predict_autogluon_timed(
                        X_train_p,
                        y_train,
                        X_holdout_p,
                        time_limit=SYNREG_AG_TIME_LIMIT,
                        presets=SYNREG_AG_PRESETS,
                        num_cpus=SYNREG_AG_TASK_CPUS,
                        num_gpus=0,
                        verbosity=0,
                        model_dir=ag_path,
                        cleanup=True,
                    )

                    metrics = compute_regression_metrics(y_pred, betaX_holdout, y_observed=y_holdout)

                    row_out = _empty_row()
                    row_out.update(base)
                    row_out.update(metrics)
                    row_out.update({
                        "status": "ok",
                        "fit_time_s": fit_time,
                        "predict_time_s": predict_time,
                        "total_time_s": fit_time + predict_time,
                        "processed_features": X_train_p.shape[1],
                        "raw_features": X_train_p.shape[1],
                    })
                    output_rows.append(row_out)

                except Exception as e:
                    output_rows.append(_failed_row(base, type(e).__name__, str(e)[:500]))
                finally:
                    # Clean up AutoGluon /tmp artifacts
                    try:
                        if "ag_path" in dir() and os.path.isdir(ag_path):
                            shutil.rmtree(ag_path, ignore_errors=True)
                    except Exception:
                        pass
                    gc.collect()

        gc.collect()

    write_part_csv_to_stage(
        session, output_rows, "AutoGluon", SYNREG_SHARD_INDEX, SYNREG_NUM_SHARDS
    )
    print(f"[INFO] AutoGluon shard {SYNREG_SHARD_INDEX} complete: {len(output_rows)} rows.")


# ---------------------------------------------------------------------------
# Mode: Aggregate
# ---------------------------------------------------------------------------

def _validate_local_csv(path: str, label: str, required_columns=(), min_bytes=MIN_VALID_CSV_BYTES) -> int:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"{label}: expected CSV does not exist: {path}")
    size = p.stat().st_size
    if size < min_bytes:
        raise RuntimeError(
            f"{label}: CSV {path} is only {size} bytes "
            f"(threshold={min_bytes}). Blank or corrupt file — refusing to continue."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"{label}: failed to parse CSV {path}: {exc}") from exc
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise RuntimeError(f"{label}: missing required columns: {missing}")
    return len(df)


def _write_failure_json(session, parts_stage, error_type, error_message, input_files, recommendation):
    payload = {
        "status": "failed",
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "error_type": error_type,
        "error_message": error_message,
        "traceback": traceback.format_exc(),
        "stage": "EVALUATION_RESULTS_STAGE",
        "input_files_discovered": input_files,
        "recommendation": recommendation,
    }
    local_path = f"/tmp/{SYNREG_FAILURE_FILENAME}"
    try:
        with open(local_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[FAILURE JSON]\n{json.dumps(payload, indent=2)}", flush=True)
        session.file.put(
            local_path,
            SYNREG_RESULTS_STAGE.replace("/regression", ""),
            auto_compress=False,
            overwrite=True,
        )
        print(f"[INFO] Failure JSON uploaded to stage.", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not upload failure JSON: {exc}", flush=True)


def _write_success_manifest(session, input_summary, output_summary, stage_root):
    payload = {
        "status": "success",
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "stage": "EVALUATION_RESULTS_STAGE",
        "input_shards": input_summary,
        "outputs": output_summary,
        "validation": {
            "required_columns_present": True,
            "empty_inputs_detected": False,
            "empty_outputs_detected": False,
        },
    }
    local_path = f"/tmp/{SYNREG_MANIFEST_FILENAME}"
    with open(local_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[MANIFEST]\n{json.dumps(payload, indent=2)}", flush=True)
    session.file.put(local_path, stage_root, auto_compress=False, overwrite=True)
    print(f"[INFO] Success manifest uploaded.", flush=True)


def _check_shard_completeness(
    found_names: list[str],
    parts_stage: str,
) -> None:
    """Verify all expected shard files are present for all-method suites."""
    if SYNREG_SUITE_ID not in ALL_METHOD_SUITE_IDS:
        return
    n_ds = SYNREG_EXPECTED_DEEPSET_SHARDS
    n_bl = SYNREG_EXPECTED_BASELINE_SHARDS
    n_ag = SYNREG_EXPECTED_AG_SHARDS
    if n_ds == 0 and n_bl == 0 and n_ag == 0:
        return  # completeness check disabled (default)

    missing = []
    for i in range(n_ds):
        expected = f"MODEL3-ICL-MC_shard{i}_of_{n_ds}_detailed.csv"
        if not any(expected in n for n in found_names):
            missing.append(expected)
    for i in range(n_bl):
        expected = f"baselines_shard{i}_of_{n_bl}_detailed.csv"
        if not any(expected in n for n in found_names):
            missing.append(expected)
    for i in range(n_ag):
        expected = f"AutoGluon_shard{i}_of_{n_ag}_detailed.csv"
        if not any(expected in n for n in found_names):
            missing.append(expected)

    if missing:
        raise RuntimeError(
            f"Shard completeness check failed for suite '{SYNREG_SUITE_ID}': "
            f"{len(missing)} expected shard file(s) missing under {parts_stage}. "
            f"Missing (first 5): {missing[:5]}"
        )


def _validate_staged_shards(session, parts_stage: str) -> list[dict]:
    """
    List staged shard files and validate each is above MIN_VALID_CSV_BYTES.
    Returns list of dicts with keys: name, size, is_baseline, is_autogluon.
    Raises RuntimeError with detailed failure JSON on first blank file detected.
    """
    list_rows = session.sql(
        f"LIST {parts_stage} PATTERN='.*_detailed[.]csv.*'"
    ).collect()

    if not list_rows:
        _write_failure_json(
            session,
            parts_stage,
            error_type="NoShardFilesFound",
            error_message=(
                f"No shard part files found under {parts_stage}. "
                "Run DeepSet, baseline, and AutoGluon evaluation phases first."
            ),
            input_files=[],
            recommendation=(
                "Call run_synthetic_regression_prep, run_synthetic_regression_deepset_evaluation, "
                "run_synthetic_regression_baseline_evaluation, and "
                "run_synthetic_regression_autogluon_evaluation before aggregation."
            ),
        )
        raise RuntimeError(
            f"No shard part files found under {parts_stage}."
        )

    blank_files = []
    valid_files = []
    for row in list_rows:
        try:
            row_dict = row.as_dict()
        except Exception:
            row_dict = dict(zip([str(i) for i in range(len(row))], row))
        row_dict_lower = {str(k).lower(): v for k, v in row_dict.items()}
        name = str(row_dict_lower.get("name") or row[0])
        size_val = row_dict_lower.get("size") or (row[1] if len(row) > 1 else 0)
        size = int(size_val) if size_val is not None else 0
        info = {
            "name": name,
            "size_bytes": size,
            "is_baseline": "baselines_shard" in name,
            "is_autogluon": "autogluon_shard" in name.lower(),
        }
        if size < MIN_VALID_CSV_BYTES:
            blank_files.append(info)
        else:
            valid_files.append(info)

    _check_shard_completeness([f["name"] for f in valid_files + blank_files], parts_stage)

    if blank_files:
        _write_failure_json(
            session,
            parts_stage,
            error_type="BlankShardFiles",
            error_message=(
                f"Detected {len(blank_files)} shard file(s) with size < "
                f"{MIN_VALID_CSV_BYTES} bytes. All such files are invalid."
            ),
            input_files=blank_files + valid_files,
            recommendation=(
                f"Blank shard files (e.g. {blank_files[0]['name']}, "
                f"size={blank_files[0]['size_bytes']} bytes) indicate that upstream "
                "evaluation shard jobs wrote empty output — most likely because "
                "SYNTHETIC_REGRESSION_DATASET_INDEX returned 0 rows. "
                "Verify: (1) CALL run_synthetic_regression_prep() completed with a "
                "non-empty index; (2) SYNTHETIC_REGRESSION_SUITE_ID matches between "
                "prep and evaluation jobs; (3) database/schema context is consistent. "
                "Then rerun the baseline and AutoGluon evaluation phases."
            ),
        )
        raise RuntimeError(
            f"Synthetic regression aggregation failed: {len(blank_files)} of "
            f"{len(list_rows)} staged shard files are invalid "
            f"(size < {MIN_VALID_CSV_BYTES} bytes). "
            f"First invalid file: {blank_files[0]['name']}, "
            f"size={blank_files[0]['size_bytes']} bytes. "
            "See synthetic_regression_aggregation_failure.json on "
            "@EVALUATION_RESULTS_STAGE for full diagnosis."
        )

    return valid_files


def _noise_feature_bucket(noise_level) -> str:
    if noise_level is None or (isinstance(noise_level, float) and math.isnan(noise_level)):
        return "unknown"
    nl = int(noise_level)
    if nl == 0:
        return "0"
    elif nl <= 25:
        return "1-25"
    elif nl <= 50:
        return "26-50"
    else:
        return "51-100"


def _training_size_bucket(n_train) -> str:
    if n_train is None or (isinstance(n_train, float) and math.isnan(n_train)):
        return "unknown"
    nt = int(n_train)
    if nt < 100:
        return "tiny(<100)"
    elif nt < 500:
        return "small(100-500)"
    elif nt < 2000:
        return "medium(500-2000)"
    else:
        return "large(>=2000)"


def _upload_csv(session, local_path: str, stage_prefix: str) -> None:
    """Upload a local CSV to stage."""
    session.file.put(local_path, stage_prefix, auto_compress=False, overwrite=True)
    print(f"[INFO] Uploaded {local_path} to {stage_prefix}")


def run_synthetic_regression_aggregation() -> None:
    """
    Ingest all synthetic_regression_parts, rank, compute ratios, write summary CSVs.
    """
    session = Session.builder.getOrCreate()

    agg_local_dir = os.path.join(tempfile.gettempdir(), "synreg_agg_output")
    os.makedirs(agg_local_dir, exist_ok=True)

    parts_stage = SYNREG_RESULTS_STAGE   # resolved from env var, includes suite_id subpath
    results_stage = SYNREG_OUTPUT_STAGE

    try:
        _run_synthetic_regression_aggregation_inner(
            session, agg_local_dir, parts_stage, results_stage
        )
    except Exception as exc:
        print(f"[FATAL] Aggregation failed: {exc}", flush=True)
        _write_failure_json(
            session,
            parts_stage,
            error_type=type(exc).__name__,
            error_message=str(exc),
            input_files=[],
            recommendation=(
                "Check synthetic_regression_aggregation_failure.json on "
                "@EVALUATION_RESULTS_STAGE for full diagnosis. "
                "Verify SYNTHETIC_REGRESSION_DATASET_INDEX has rows, and that "
                "all shard files are valid (size >= 64 bytes)."
            ),
        )
        raise


def _run_synthetic_regression_aggregation_inner(
    session, agg_local_dir: str, parts_stage: str, results_stage: str
) -> None:
    """Inner aggregation body — called by run_synthetic_regression_aggregation."""
    # ------------------------------------------------------------------ #
    # Step 0 — Validate staged shard files via LIST (size check)
    # ------------------------------------------------------------------ #
    print("[INFO] Validating staged shard files …")
    valid_shards = _validate_staged_shards(session, parts_stage)
    print(f"[INFO] {len(valid_shards)} valid shard files confirmed.")

    # ------------------------------------------------------------------ #
    # Step 1 — Ingest part files
    # ------------------------------------------------------------------ #
    print("[INFO] Listing part files …")
    list_rows = session.sql(
        f"LIST {parts_stage} PATTERN='.*_detailed[.]csv.*'"
    ).collect()
    part_files = [r["name"] for r in list_rows]

    if not part_files:
        raise RuntimeError(
            f"No part files found under {parts_stage}. "
            "Run DeepSet, baseline, and AutoGluon evaluation phases first."
        )

    print(f"[INFO] Found {len(part_files)} part files. Downloading …")
    parts_local_dir = os.path.join(agg_local_dir, "parts")
    os.makedirs(parts_local_dir, exist_ok=True)
    session.file.get(f"{parts_stage}/", parts_local_dir)

    dfs = []
    for fname in sorted(os.listdir(parts_local_dir)):
        if fname.endswith(".csv") or fname.endswith(".csv.gz"):
            fpath = os.path.join(parts_local_dir, fname)
            try:
                _validate_local_csv(fpath, f"shard:{fname}", min_bytes=MIN_VALID_CSV_BYTES)
                df_part = pd.read_csv(fpath, low_memory=False)
                dfs.append(df_part)
            except Exception as e:
                print(f"[WARN] Could not read {fname}: {e}")

    if not dfs:
        raise RuntimeError("All part files failed to parse.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total rows after concat: {len(df)}")

    # Suite-id cross-contamination guard
    if "suite_id" in df.columns:
        bad_rows = df[df["suite_id"].notna() & (df["suite_id"] != SYNREG_SUITE_ID)]
        if not bad_rows.empty:
            bad_ids = bad_rows["suite_id"].unique().tolist()
            raise RuntimeError(
                f"Aggregation found rows with suite_id != '{SYNREG_SUITE_ID}': {bad_ids}. "
                "Part files from a different suite landed in this stage prefix. "
                "Verify SYNREG_RESULTS_STAGE is suite-specific and re-run the evaluation shards."
            )

    # ------------------------------------------------------------------ #
    # Step 2 — Validate required columns
    # ------------------------------------------------------------------ #
    required_cols = {"method", "dataset_id", "split_seed", "mse_betaX"}
    missing_req = required_cols - set(df.columns)
    if missing_req:
        raise RuntimeError(f"Required columns missing from part files: {missing_req}")

    # Normalize optional columns to NaN if absent
    for col in _CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    # ------------------------------------------------------------------ #
    # Step 3 — Classify rows
    # ------------------------------------------------------------------ #
    df["mse_betaX"] = pd.to_numeric(df["mse_betaX"], errors="coerce")
    df["rmse_betaX"] = pd.to_numeric(df["rmse_betaX"], errors="coerce")
    df["mae_betaX"] = pd.to_numeric(df["mae_betaX"], errors="coerce")
    df["r2_betaX"] = pd.to_numeric(df["r2_betaX"], errors="coerce")

    is_valid_mask = (
        (df["status"] == "ok")
        & df["mse_betaX"].notna()
        & df["rmse_betaX"].notna()
        & np.isfinite(df["mse_betaX"].values)
        & np.isfinite(df["rmse_betaX"].values)
    )
    df_valid = df[is_valid_mask].copy()
    print(
        f"[INFO] Row classification: valid={is_valid_mask.sum()}, "
        f"skipped={(df['status'] == 'skipped').sum()}, "
        f"failed={(df['status'] == 'failed').sum()}"
    )

    # ------------------------------------------------------------------ #
    # Step 4 — Ranking (valid rows only)
    # ------------------------------------------------------------------ #
    rank_group_cols = [
        "suite_id", "suite_family", "dataset_id", "split_seed", "prior_regime",
        "n_train", "n_holdout", "feature_noise_level", "target_noise_scale",
        "training_size_anchor",
    ]
    # Keep only columns that exist
    rank_group_cols = [c for c in rank_group_cols if c in df_valid.columns]

    # Initialize rank columns in full df as NaN
    for rc in ["rank_mse_betaX", "rank_rmse_betaX", "rank_mae_betaX", "rank_r2_betaX", "rank_overall"]:
        df[rc] = float("nan")

    def _dense_rank_series(s: pd.Series, ascending: bool = True) -> pd.Series:
        return s.rank(method="dense", ascending=ascending, na_option="keep")

    if len(df_valid) > 0:
        df_valid = df_valid.copy()
        df_valid["rank_mse_betaX"] = df_valid.groupby(rank_group_cols, dropna=False)[
            "mse_betaX"
        ].transform(lambda x: _dense_rank_series(x, ascending=True))
        df_valid["rank_rmse_betaX"] = df_valid.groupby(rank_group_cols, dropna=False)[
            "rmse_betaX"
        ].transform(lambda x: _dense_rank_series(x, ascending=True))
        df_valid["rank_mae_betaX"] = df_valid.groupby(rank_group_cols, dropna=False)[
            "mae_betaX"
        ].transform(lambda x: _dense_rank_series(x, ascending=True))
        df_valid["rank_r2_betaX"] = df_valid.groupby(rank_group_cols, dropna=False)[
            "r2_betaX"
        ].transform(lambda x: _dense_rank_series(x, ascending=False))
        df_valid["rank_overall"] = df_valid["rank_mse_betaX"]
        # Merge ranks back into df
        rank_cols_to_merge = [
            "rank_mse_betaX", "rank_rmse_betaX", "rank_mae_betaX", "rank_r2_betaX", "rank_overall"
        ]
        df.loc[df_valid.index, rank_cols_to_merge] = df_valid[rank_cols_to_merge]

    # ------------------------------------------------------------------ #
    # Step 5 — Reference ratios
    # ------------------------------------------------------------------ #
    tree_methods = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}

    df["ratio_mse_to_fixed_ridge"] = float("nan")
    df["ratio_rmse_to_fixed_ridge"] = float("nan")
    df["ratio_mse_to_autogluon"] = float("nan")
    df["ratio_rmse_to_autogluon"] = float("nan")
    df["ratio_mse_to_best_tree"] = float("nan")
    df["ratio_rmse_to_best_tree"] = float("nan")
    df["beats_fixed_ridge"] = False
    df["beats_autogluon"] = False
    df["beats_best_tree"] = False
    df["is_best_mse"] = False
    df["is_top3_mse"] = False

    if len(df_valid) > 0:
        for group_keys, group_df in df_valid.groupby(rank_group_cols, dropna=False):
            idx = group_df.index

            def _get_ref_mse(method_name):
                ref = group_df[group_df["method"] == method_name]
                if len(ref) == 1 and np.isfinite(ref["mse_betaX"].values[0]):
                    return float(ref["mse_betaX"].values[0]), float(ref["rmse_betaX"].values[0])
                return None, None

            fr_mse, fr_rmse = _get_ref_mse("FixedRidgeLambda1")
            ag_mse, ag_rmse = _get_ref_mse("AutoGluon")

            # best_tree: minimum mse among available tree methods in this group
            tree_rows = group_df[group_df["method"].isin(tree_methods)]
            if len(tree_rows) > 0:
                bt_idx = tree_rows["mse_betaX"].idxmin()
                bt_mse = float(tree_rows.loc[bt_idx, "mse_betaX"])
                bt_rmse = float(tree_rows.loc[bt_idx, "rmse_betaX"])
                if not np.isfinite(bt_mse):
                    bt_mse = bt_rmse = None
            else:
                bt_mse = bt_rmse = None

            for i in idx:
                m = float(df.at[i, "mse_betaX"])
                rm = float(df.at[i, "rmse_betaX"])
                if not np.isfinite(m):
                    continue

                if fr_mse is not None and fr_mse > 0:
                    df.at[i, "ratio_mse_to_fixed_ridge"] = m / fr_mse
                    df.at[i, "ratio_rmse_to_fixed_ridge"] = rm / fr_rmse if fr_rmse and fr_rmse > 0 else float("nan")
                    df.at[i, "beats_fixed_ridge"] = (m / fr_mse) < 1.0

                if ag_mse is not None and ag_mse > 0:
                    df.at[i, "ratio_mse_to_autogluon"] = m / ag_mse
                    df.at[i, "ratio_rmse_to_autogluon"] = rm / ag_rmse if ag_rmse and ag_rmse > 0 else float("nan")
                    df.at[i, "beats_autogluon"] = (m / ag_mse) < 1.0

                if bt_mse is not None and bt_mse > 0:
                    df.at[i, "ratio_mse_to_best_tree"] = m / bt_mse
                    df.at[i, "ratio_rmse_to_best_tree"] = rm / bt_rmse if bt_rmse and bt_rmse > 0 else float("nan")
                    df.at[i, "beats_best_tree"] = (m / bt_mse) < 1.0

            # Win indicators
            best_mse_in_group = group_df["mse_betaX"].min()
            for i in idx:
                rank_val = df.at[i, "rank_mse_betaX"]
                if not (isinstance(rank_val, float) and math.isnan(rank_val)):
                    df.at[i, "is_best_mse"] = (rank_val == 1)
                    df.at[i, "is_top3_mse"] = (rank_val <= 3)

    # ------------------------------------------------------------------ #
    # Step 6 — Bucketing columns
    # ------------------------------------------------------------------ #
    df["noise_feature_bucket"] = df["feature_noise_level"].apply(_noise_feature_bucket)
    df["training_size_bucket"] = df["n_train"].apply(_training_size_bucket)

    # Quartiles (valid rows only, computed globally then assigned)
    for qcol, src in [("p_total_quartile", "p_total"), ("n_train_quartile", "n_train")]:
        df[qcol] = float("nan")
        if src in df_valid.columns and len(df_valid) >= 4:
            vals = pd.to_numeric(df_valid[src], errors="coerce")
            q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75]).values
            def _quartile(v):
                if pd.isna(v):
                    return float("nan")
                if v <= q25:
                    return "Q1"
                elif v <= q50:
                    return "Q2"
                elif v <= q75:
                    return "Q3"
                else:
                    return "Q4"
            df.loc[df_valid.index, qcol] = pd.to_numeric(df_valid[src], errors="coerce").apply(_quartile)

    # ------------------------------------------------------------------ #
    # Step 7 — Write canonical output
    # ------------------------------------------------------------------ #
    canonical_path = os.path.join(agg_local_dir, "synthetic_regression_model_comparison.csv")
    df.to_csv(canonical_path, index=False)
    _upload_csv(session, canonical_path, results_stage)

    # ------------------------------------------------------------------ #
    # Step 8 — Summary aggregates
    # ------------------------------------------------------------------ #
    def _safe_iqr(s):
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        return q3 - q1

    def _skip_reasons_str(grp):
        counts = grp[grp["status"] == "skipped"]["skip_reason"].value_counts()
        return "; ".join(f"{r}:{c}" for r, c in counts.items()) if len(counts) > 0 else ""

    df_v = df[is_valid_mask].copy() if len(df) > 0 else df_valid.copy()

    # --- model_comparison_summary (per method) ---
    summary_rows = []
    for method, grp in df.groupby("method"):
        grp_v = grp[grp["status"] == "ok"]
        total_valid = len(grp_v)
        wins = int(grp_v["is_best_mse"].sum()) if "is_best_mse" in grp_v.columns else 0
        top3 = int(grp_v["is_top3_mse"].sum()) if "is_top3_mse" in grp_v.columns else 0
        summary_rows.append({
            "method": method,
            "valid_count": total_valid,
            "skip_count": int((grp["status"] == "skipped").sum()),
            "failure_count": int((grp["status"] == "failed").sum()),
            "mean_mse_betaX": grp_v["mse_betaX"].mean(),
            "median_mse_betaX": grp_v["mse_betaX"].median(),
            "std_mse_betaX": grp_v["mse_betaX"].std(),
            "iqr_mse_betaX": _safe_iqr(grp_v["mse_betaX"]),
            "mean_rmse_betaX": grp_v["rmse_betaX"].mean(),
            "median_rmse_betaX": grp_v["rmse_betaX"].median(),
            "mean_mae_betaX": grp_v["mae_betaX"].mean(),
            "median_mae_betaX": grp_v["mae_betaX"].median(),
            "mean_r2_betaX": grp_v["r2_betaX"].mean(),
            "median_r2_betaX": grp_v["r2_betaX"].median(),
            "mean_rank_mse_betaX": grp_v["rank_mse_betaX"].mean(),
            "median_rank_mse_betaX": grp_v["rank_mse_betaX"].median(),
            "win_rate_mse": wins / total_valid if total_valid > 0 else float("nan"),
            "top3_rate_mse": top3 / total_valid if total_valid > 0 else float("nan"),
            "mean_ratio_mse_to_fixed_ridge": grp_v["ratio_mse_to_fixed_ridge"].mean(),
            "median_ratio_mse_to_fixed_ridge": grp_v["ratio_mse_to_fixed_ridge"].median(),
            "mean_ratio_mse_to_autogluon": grp_v["ratio_mse_to_autogluon"].mean(),
            "median_ratio_mse_to_autogluon": grp_v["ratio_mse_to_autogluon"].median(),
            "mean_ratio_mse_to_best_tree": grp_v["ratio_mse_to_best_tree"].mean(),
            "median_ratio_mse_to_best_tree": grp_v["ratio_mse_to_best_tree"].median(),
            "beats_fixed_ridge_rate": grp_v["beats_fixed_ridge"].mean() if "beats_fixed_ridge" in grp_v.columns else float("nan"),
            "beats_autogluon_rate": grp_v["beats_autogluon"].mean() if "beats_autogluon" in grp_v.columns else float("nan"),
            "beats_best_tree_rate": grp_v["beats_best_tree"].mean() if "beats_best_tree" in grp_v.columns else float("nan"),
            "mean_fit_time_s": grp_v["fit_time_s"].mean(),
            "mean_predict_time_s": grp_v["predict_time_s"].mean(),
            "mean_total_time_s": grp_v["total_time_s"].mean(),
            "skip_reasons": _skip_reasons_str(grp),
        })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(agg_local_dir, "synthetic_regression_model_comparison_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    _upload_csv(session, summary_path, results_stage)

    # --- summary_by_regime ---
    regime_group_cols = ["suite_family", "prior_regime", "method"]
    regime_group_cols = [c for c in regime_group_cols if c in df_v.columns]
    if regime_group_cols:
        df_regime = df_v.groupby(regime_group_cols).agg(
            valid_count=("mse_betaX", "count"),
            mean_mse_betaX=("mse_betaX", "mean"),
            median_mse_betaX=("mse_betaX", "median"),
            std_mse_betaX=("mse_betaX", "std"),
            mean_rmse_betaX=("rmse_betaX", "mean"),
            mean_mae_betaX=("mae_betaX", "mean"),
            mean_r2_betaX=("r2_betaX", "mean"),
            median_r2_betaX=("r2_betaX", "median"),
            mean_rank_mse_betaX=("rank_mse_betaX", "mean"),
            median_rank_mse_betaX=("rank_mse_betaX", "median"),
        ).reset_index()
        regime_path = os.path.join(agg_local_dir, "synthetic_regression_summary_by_regime.csv")
        df_regime.to_csv(regime_path, index=False)
        _upload_csv(session, regime_path, results_stage)

    # --- summary_by_feature_noise ---
    fn_cols = ["feature_noise_level", "prior_regime", "method"]
    fn_cols = [c for c in fn_cols if c in df_v.columns]
    if fn_cols:
        df_fn_base = df_v.groupby(fn_cols).agg(
            valid_count=("mse_betaX", "count"),
            mean_mse_betaX=("mse_betaX", "mean"),
            median_mse_betaX=("mse_betaX", "median"),
            std_mse_betaX=("mse_betaX", "std"),
            mean_rmse_betaX=("rmse_betaX", "mean"),
            mean_r2_betaX=("r2_betaX", "mean"),
            mean_rank_mse_betaX=("rank_mse_betaX", "mean"),
            mean_processed_features=("processed_features", "mean"),
            mean_selected_features=("selected_features", "mean"),
            mean_feature_cap=("feature_cap", "mean"),
        ).reset_index()

        # Stability columns: relative to noise_level=0
        baseline_df = df_fn_base[df_fn_base["feature_noise_level"] == 0][
            ["prior_regime", "method", "mean_mse_betaX", "mean_rmse_betaX", "mean_r2_betaX", "mean_rank_mse_betaX"]
        ].rename(columns={
            "mean_mse_betaX": "base_mse",
            "mean_rmse_betaX": "base_rmse",
            "mean_r2_betaX": "base_r2",
            "mean_rank_mse_betaX": "base_rank",
        })
        df_fn = df_fn_base.merge(baseline_df, on=["prior_regime", "method"], how="left")
        df_fn["mse_degradation_vs_noise0"] = df_fn["mean_mse_betaX"] - df_fn["base_mse"]
        df_fn["rmse_degradation_vs_noise0"] = df_fn["mean_rmse_betaX"] - df_fn.get("base_rmse", float("nan"))
        df_fn["r2_drop_vs_noise0"] = df_fn["base_r2"] - df_fn["mean_r2_betaX"]
        df_fn["rank_change_vs_noise0"] = df_fn["mean_rank_mse_betaX"] - df_fn["base_rank"]
        df_fn["relative_mse_degradation_pct"] = (
            df_fn["mse_degradation_vs_noise0"] / df_fn["base_mse"].replace(0, float("nan")) * 100
        )
        fn_path = os.path.join(agg_local_dir, "synthetic_regression_summary_by_feature_noise.csv")
        if not df_fn.empty:
            df_fn.to_csv(fn_path, index=False)
            _upload_csv(session, fn_path, results_stage)

    # --- summary_by_training_size ---
    ts_cols = ["n_train", "prior_regime", "method"]
    ts_cols = [c for c in ts_cols if c in df_v.columns]
    if ts_cols:
        df_ts_base = df_v.groupby(ts_cols).agg(
            valid_count=("mse_betaX", "count"),
            n_holdout=("n_holdout", "first"),
            mean_mse_betaX=("mse_betaX", "mean"),
            median_mse_betaX=("mse_betaX", "median"),
            std_mse_betaX=("mse_betaX", "std"),
            mean_rmse_betaX=("rmse_betaX", "mean"),
            mean_r2_betaX=("r2_betaX", "mean"),
            mean_rank_mse_betaX=("rank_mse_betaX", "mean"),
            mean_context_window_size=("context_window_size", "mean"),
            mean_context_windows=("context_windows", "mean"),
        ).reset_index()

        # Sample-efficiency columns: relative to n_train=25
        ts_baseline = df_ts_base[df_ts_base["n_train"] == 25][
            ["prior_regime", "method", "mean_mse_betaX", "mean_rmse_betaX", "mean_r2_betaX"]
        ].rename(columns={
            "mean_mse_betaX": "base25_mse",
            "mean_rmse_betaX": "base25_rmse",
            "mean_r2_betaX": "base25_r2",
        })
        df_ts = df_ts_base.merge(ts_baseline, on=["prior_regime", "method"], how="left")
        df_ts["mse_improvement_vs_25"] = df_ts["base25_mse"] - df_ts["mean_mse_betaX"]
        df_ts["rmse_improvement_vs_25"] = df_ts.get("base25_rmse", float("nan")) - df_ts["mean_rmse_betaX"]
        df_ts["r2_improvement_vs_25"] = df_ts["mean_r2_betaX"] - df_ts.get("base25_r2", float("nan"))
        df_ts["relative_mse_improvement_pct"] = (
            df_ts["mse_improvement_vs_25"] / df_ts["base25_mse"].replace(0, float("nan")) * 100
        )
        df_ts["is_tabpfn_anchor"] = (
            (df_ts["n_train"] == TRAIN_SIZE_ANCHOR_N) & (df_ts["n_holdout"] == HOLDOUT_SIZE)
        )
        ts_path = os.path.join(agg_local_dir, "synthetic_regression_summary_by_training_size.csv")
        if not df_ts.empty:
            df_ts.to_csv(ts_path, index=False)
            _upload_csv(session, ts_path, results_stage)

    # --- summary_by_regime_p_quartile_n_quartile ---
    pqnq_cols = ["suite_family", "prior_regime", "p_total_quartile", "n_train_quartile", "method"]
    pqnq_cols = [c for c in pqnq_cols if c in df_v.columns]
    if pqnq_cols:
        df_pqnq = df_v.groupby(pqnq_cols, dropna=False).agg(
            valid_count=("mse_betaX", "count"),
            mean_mse_betaX=("mse_betaX", "mean"),
            mean_r2_betaX=("r2_betaX", "mean"),
            mean_rank_mse_betaX=("rank_mse_betaX", "mean"),
        ).reset_index()
        pqnq_path = os.path.join(
            agg_local_dir,
            "synthetic_regression_summary_by_regime_p_quartile_n_quartile.csv"
        )
        df_pqnq.to_csv(pqnq_path, index=False)
        _upload_csv(session, pqnq_path, results_stage)

    # --- optional: summary_by_target_noise ---
    tn_cols = ["target_noise_scale", "prior_regime", "method"]
    has_target_noise = (
        "target_noise_scale" in df_v.columns
        and df_v["target_noise_scale"].notna().any()
        and df_v[df_v["suite_family"] == "target_noise"].shape[0] > 0
    )
    if has_target_noise:
        df_tn_only = df_v[df_v["suite_family"] == "target_noise"]
        df_tn = df_tn_only.groupby(tn_cols).agg(
            valid_count=("mse_betaX", "count"),
            mean_mse_betaX=("mse_betaX", "mean"),
            mean_r2_betaX=("r2_betaX", "mean"),
            mean_rank_mse_betaX=("rank_mse_betaX", "mean"),
        ).reset_index()
        tn_path = os.path.join(agg_local_dir, "synthetic_regression_summary_by_target_noise.csv")
        df_tn.to_csv(tn_path, index=False)
        _upload_csv(session, tn_path, results_stage)

    # ------------------------------------------------------------------ #
    # Step 9 — Chart-ready CSVs
    # ------------------------------------------------------------------ #
    # Chart: noise features
    if "feature_noise_level" in df_v.columns:
        fn_chart_cols = ["feature_noise_level", "prior_regime", "method"]
        fn_chart_cols = [c for c in fn_chart_cols if c in df_v.columns]
        df_fn_chart = df_v.groupby(fn_chart_cols).agg(
            mean_mse_betaX=("mse_betaX", "mean"),
            median_mse_betaX=("mse_betaX", "median"),
            std_mse_betaX=("mse_betaX", "std"),
            mean_rmse_betaX=("rmse_betaX", "mean"),
            mean_r2_betaX=("r2_betaX", "mean"),
            mean_rank_mse_betaX=("rank_mse_betaX", "mean"),
            mean_selected_features=("selected_features", "mean"),
            mean_processed_features=("processed_features", "mean"),
            valid_count=("mse_betaX", "count"),
        ).reset_index().rename(columns={"feature_noise_level": "x_noise_features"})
        if "fn" in dir() and hasattr(df_fn, "columns") and "relative_mse_degradation_pct" in df_fn.columns:
            df_fn_chart = df_fn_chart.merge(
                df_fn[fn_chart_cols + ["relative_mse_degradation_pct"]].rename(
                    columns={"feature_noise_level": "x_noise_features"}
                ),
                on=["x_noise_features", "prior_regime", "method"],
                how="left",
            )
        fn_chart_path = os.path.join(
            agg_local_dir, "synthetic_regression_chart_data_noise_features.csv"
        )
        if not df_fn_chart.empty:
            df_fn_chart.to_csv(fn_chart_path, index=False)
            _upload_csv(session, fn_chart_path, results_stage)

    # Chart: training size
    if "n_train" in df_v.columns:
        ts_chart_cols = ["n_train", "n_holdout", "prior_regime", "method"]
        ts_chart_cols = [c for c in ts_chart_cols if c in df_v.columns]
        df_ts_chart = df_v.groupby(ts_chart_cols).agg(
            mean_mse_betaX=("mse_betaX", "mean"),
            median_mse_betaX=("mse_betaX", "median"),
            std_mse_betaX=("mse_betaX", "std"),
            mean_rmse_betaX=("rmse_betaX", "mean"),
            mean_r2_betaX=("r2_betaX", "mean"),
            mean_rank_mse_betaX=("rank_mse_betaX", "mean"),
            mean_context_window_size=("context_window_size", "mean"),
            mean_context_windows=("context_windows", "mean"),
            valid_count=("mse_betaX", "count"),
        ).reset_index().rename(columns={"n_train": "x_n_train"})
        if "df_ts" in dir() and "relative_mse_improvement_pct" in df_ts.columns:
            df_ts_chart = df_ts_chart.merge(
                df_ts[["n_train", "prior_regime", "method", "relative_mse_improvement_pct",
                        "is_tabpfn_anchor"]].rename(columns={"n_train": "x_n_train"}),
                on=["x_n_train", "prior_regime", "method"],
                how="left",
            )
        ts_chart_path = os.path.join(
            agg_local_dir, "synthetic_regression_chart_data_training_size.csv"
        )
        if not df_ts_chart.empty:
            df_ts_chart.to_csv(ts_chart_path, index=False)
            _upload_csv(session, ts_chart_path, results_stage)

    # Chart: model rank
    rank_chart_rows = []
    for (suite_fam, method), grp in df_v.groupby(["suite_family", "method"]):
        for cond_name, cond_col in [("noise_features", "feature_noise_level"), ("n_train", "n_train")]:
            if cond_col not in grp.columns:
                continue
            for cond_val, cval_grp in grp.groupby(cond_col):
                for regime, regime_grp in cval_grp.groupby("prior_regime"):
                    rank_chart_rows.append({
                        "suite_family": suite_fam,
                        "condition_name": cond_name,
                        "condition_value": cond_val,
                        "prior_regime": regime,
                        "method": method,
                        "mean_rank_mse_betaX": regime_grp["rank_mse_betaX"].mean(),
                        "median_rank_mse_betaX": regime_grp["rank_mse_betaX"].median(),
                        "win_rate_mse": (regime_grp["rank_mse_betaX"] == 1).mean(),
                        "top3_rate_mse": (regime_grp["rank_mse_betaX"] <= 3).mean(),
                        "valid_count": len(regime_grp),
                    })
    if rank_chart_rows:
        df_rank_chart = pd.DataFrame(rank_chart_rows)
        rank_chart_path = os.path.join(
            agg_local_dir, "synthetic_regression_chart_data_model_rank.csv"
        )
        df_rank_chart.to_csv(rank_chart_path, index=False)
        _upload_csv(session, rank_chart_path, results_stage)

    # ------------------------------------------------------------------ #
    # Step 10 — Optional PNG charts
    # ------------------------------------------------------------------ #
    charts_stage = "@EVALUATION_RESULTS_STAGE/synthetic_regression_charts"
    _generate_optional_charts(df_v, agg_local_dir, session, charts_stage)

    # ------------------------------------------------------------------ #
    # Step 11 — Validate outputs and write success manifest
    # ------------------------------------------------------------------ #
    output_summary = []
    for fname in SYNREG_EXPECTED_OUTPUTS:
        local_path = os.path.join(agg_local_dir, fname)
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            output_summary.append({"name": fname, "size_bytes": size, "written": True})
        else:
            output_summary.append({"name": fname, "size_bytes": 0, "written": False})
    for fname in SYNREG_CONDITIONAL_OUTPUTS:
        local_path = os.path.join(agg_local_dir, fname)
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            output_summary.append({"name": fname, "size_bytes": size, "written": True, "conditional": True})
        else:
            output_summary.append({"name": fname, "size_bytes": 0, "written": False, "conditional": True})

    input_summary = [
        {"name": s["name"], "size_bytes": s["size_bytes"]}
        for s in valid_shards
    ]
    _write_success_manifest(session, input_summary, output_summary, results_stage)

    print("[INFO] Aggregation complete.")


def _generate_optional_charts(df_v: "pd.DataFrame", out_dir: str, session, charts_stage: str) -> None:
    """Generate optional matplotlib charts and upload to stage."""
    os.makedirs(out_dir, exist_ok=True)

    HIGHLIGHT_COLOR = "#C0392B"   # red-orange for DeepSet
    BASE_COLOR = "#2E86AB"        # steel blue for other methods
    GREY_COLOR = "#AAAAAA"        # grey for background methods and scatter dots

    def _upload_chart(chart_path: str) -> None:
        session.file.put(chart_path, charts_stage, auto_compress=False, overwrite=True)

    # Chart 1: Feature Selection Stability (noise features)
    try:
        fn_path = os.path.join(out_dir, "synthetic_regression_chart_data_noise_features.csv")
        if not os.path.exists(fn_path) or "x_noise_features" not in pd.read_csv(fn_path).columns:
            raise ValueError("noise_features chart CSV not available")
        df_fn_chart = pd.read_csv(fn_path)
        methods = sorted(df_fn_chart["method"].unique().tolist())
        n_methods = len(methods)
        ncols = min(4, n_methods)
        nrows = math.ceil(n_methods / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 sharey=False, sharex=True, squeeze=False)
        fig.suptitle("Feature Selection Stability — % RMSE Increase vs Noise Features",
                     fontsize=11, y=1.01)
        all_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
        for ax_i, method in enumerate(methods):
            ax = all_axes[ax_i]
            grp = df_fn_chart[df_fn_chart["method"] == method].copy()
            if "x_noise_features" not in grp.columns or "mean_rmse_betaX" not in grp.columns:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
                ax.set_title(method, fontsize=9, pad=3, backgroundcolor="#DDDDDD")
                continue
            grp = grp.groupby("x_noise_features")["mean_rmse_betaX"].mean().reset_index()
            grp = grp.sort_values("x_noise_features")
            base_rmse = grp.loc[grp["x_noise_features"] == grp["x_noise_features"].min(), "mean_rmse_betaX"]
            base_val = float(base_rmse.values[0]) if len(base_rmse) > 0 and float(base_rmse.values[0]) > 0 else float("nan")
            pct = (grp["mean_rmse_betaX"] - base_val) / base_val * 100 if not math.isnan(base_val) else grp["mean_rmse_betaX"]
            color = HIGHLIGHT_COLOR if "deepset" in method.lower() or "DeepSet" in method else BASE_COLOR
            ax.axhline(0, color="#27AE60", linewidth=1, linestyle="--", alpha=0.7)
            ax.plot(grp["x_noise_features"], pct, color=color, linewidth=2, marker="o", markersize=4)
            ax.set_title(method, fontsize=9, pad=3, backgroundcolor="#DDDDDD")
            ax.set_facecolor("#F5F5F5")
            ax.set_xlabel("# Noise Features", fontsize=8)
            ax.set_ylabel("% Increase in RMSE", fontsize=8)
        for ax_i in range(len(methods), len(all_axes)):
            all_axes[ax_i].set_visible(False)
        fig.tight_layout()
        chart_path = os.path.join(out_dir, "feature_selection_stability.png")
        fig.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _upload_chart(chart_path)
        print("[INFO] Chart feature_selection_stability.png uploaded.")
    except Exception as e:
        print(f"[WARN] Chart feature_selection_stability.png failed: {e}")

    # Chart 2: Stability by Training Set Size
    try:
        ts_path = os.path.join(out_dir, "synthetic_regression_chart_data_training_size.csv")
        if not os.path.exists(ts_path):
            raise ValueError("training_size chart CSV not available")
        df_ts_chart = pd.read_csv(ts_path)
        if "x_n_train" not in df_ts_chart.columns:
            raise ValueError("x_n_train column missing")
        methods_ts = sorted(df_ts_chart["method"].unique().tolist()) if "method" in df_ts_chart.columns else []
        metrics_to_plot = [
            ("mean_rmse_betaX", "RMSE (betaX)"),
            ("mean_r2_betaX", "R² (betaX)"),
            ("mean_rank_mse_betaX", "Mean Rank (MSE)"),
        ]
        metrics_available = [(m, l) for m, l in metrics_to_plot if m in df_ts_chart.columns]
        n_panels = len(metrics_available)
        if n_panels == 0:
            raise ValueError("No metrics available for training_size chart")

        fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels), sharex=True, squeeze=False)
        fig.suptitle("Stability by Training Set Size", fontsize=12, y=1.01)

        # Individual scatter from df_v
        has_scatter = "n_train" in df_v.columns and "method" in df_v.columns

        for pi, (metric_col, metric_label) in enumerate(metrics_available):
            ax = axes[pi][0]
            ax.set_facecolor("#F5F5F5")
            if has_scatter and metric_col in df_v.columns:
                scatter_grp = df_v.groupby(["n_train", "method"])[metric_col].mean().reset_index()
                for method in methods_ts:
                    m_scatter = scatter_grp[scatter_grp["method"] == method]
                    if len(m_scatter) > 0:
                        ax.scatter(m_scatter["n_train"], m_scatter[metric_col],
                                   alpha=0.25, s=12, color=GREY_COLOR, zorder=1)
            for method in methods_ts:
                m_grp = df_ts_chart[df_ts_chart["method"] == method].sort_values("x_n_train") if "method" in df_ts_chart.columns else pd.DataFrame()
                if len(m_grp) == 0:
                    continue
                if "DeepSet" in method or "deepset" in method.lower():
                    color, lw, zorder = HIGHLIGHT_COLOR, 2.5, 5
                elif method in ("Ridge", "CatBoost", "AutoGluon"):
                    color, lw, zorder = BASE_COLOR, 1.8, 4
                else:
                    color, lw, zorder = GREY_COLOR, 1.0, 3
                ax.plot(m_grp["x_n_train"], m_grp[metric_col], color=color,
                        linewidth=lw, zorder=zorder, label=method)
            ax.set_xscale("log")
            ax.set_ylabel(metric_label, fontsize=9)
            ax.set_title(metric_label, fontsize=9, pad=3, backgroundcolor="#DDDDDD")
        axes[-1][0].set_xlabel("Training Set Size (log scale)", fontsize=9)
        # Legend outside on the right from the last axis
        handles, labels = axes[0][0].get_legend_handles_labels()
        highlight_methods = [m for m in methods_ts if "DeepSet" in m or m in ("Ridge", "CatBoost", "AutoGluon")]
        h_labels = [(h, l) for h, l in zip(handles, labels) if l in highlight_methods]
        if h_labels:
            hs, ls = zip(*h_labels)
            axes[0][0].legend(hs, ls, fontsize=7, loc="upper left", framealpha=0.7)
        fig.tight_layout()
        chart_path = os.path.join(out_dir, "training_size_stability.png")
        fig.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _upload_chart(chart_path)
        print("[INFO] Chart training_size_stability.png uploaded.")
    except Exception as e:
        print(f"[WARN] Chart training_size_stability.png failed: {e}")

    # Chart 3: Per-Regime Model Comparison
    try:
        if "prior_regime" not in df_v.columns or "method" not in df_v.columns or "mse_betaX" not in df_v.columns:
            raise ValueError("Required columns missing for regime chart")
        regimes = sorted(df_v["prior_regime"].dropna().unique().tolist())
        ncols = 2
        nrows = math.ceil(len(regimes) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
        fig.suptitle("Model RMSE by Regime", fontsize=12)
        # Compute shared x max
        regime_data = {}
        for regime in regimes:
            rgrp = df_v[df_v["prior_regime"] == regime].groupby("method")["mse_betaX"].mean().reset_index()
            rgrp = rgrp.sort_values("mse_betaX", ascending=True)
            regime_data[regime] = rgrp
        x_max = max((rgrp["mse_betaX"].max() for rgrp in regime_data.values() if len(rgrp) > 0), default=1.0)
        x_max = x_max * 1.1
        for ri, regime in enumerate(regimes):
            row_i, col_i = ri // ncols, ri % ncols
            ax = axes[row_i][col_i]
            rgrp = regime_data[regime]
            colors = [HIGHLIGHT_COLOR if ("DeepSet" in m or "deepset" in m.lower()) else BASE_COLOR
                      for m in rgrp["method"]]
            ax.barh(rgrp["method"], rgrp["mse_betaX"], color=colors)
            ax.set_xlim(0, x_max)
            ax.set_title(f"Regime {regime}", fontsize=10, pad=3, backgroundcolor="#DDDDDD")
            ax.set_facecolor("#F5F5F5")
            ax.set_xlabel("Mean MSE (betaX)", fontsize=8)
        for ri in range(len(regimes), nrows * ncols):
            axes[ri // ncols][ri % ncols].set_visible(False)
        fig.tight_layout()
        chart_path = os.path.join(out_dir, "model_comparison_by_regime.png")
        fig.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _upload_chart(chart_path)
        print("[INFO] Chart model_comparison_by_regime.png uploaded.")
    except Exception as e:
        print(f"[WARN] Chart model_comparison_by_regime.png failed: {e}")

    # Chart 4: Overall Model Ranking
    try:
        if "method" not in df_v.columns or "mse_betaX" not in df_v.columns:
            raise ValueError("Required columns missing for overall ranking chart")
        overall_mse = df_v.groupby("method")["mse_betaX"].mean().reset_index().sort_values("mse_betaX", ascending=True)
        rank_col = "rank_mse_betaX" if "rank_mse_betaX" in df_v.columns else None
        overall_rank = None
        if rank_col:
            overall_rank = df_v.groupby("method")[rank_col].mean().reset_index().sort_values(rank_col, ascending=True)

        n_panels = 2 if overall_rank is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, max(4, len(overall_mse) * 0.4 + 1)),
                                 squeeze=False)
        fig.suptitle("Overall Model Comparison", fontsize=12)

        # Left: mean MSE
        ax = axes[0][0]
        colors_mse = [HIGHLIGHT_COLOR if ("DeepSet" in m or "deepset" in m.lower()) else BASE_COLOR
                      for m in overall_mse["method"]]
        ax.barh(overall_mse["method"], overall_mse["mse_betaX"], color=colors_mse)
        ax.set_xlabel("Mean MSE (betaX)", fontsize=9)
        ax.set_title("Mean MSE (betaX)", fontsize=10, pad=3, backgroundcolor="#DDDDDD")
        ax.set_facecolor("#F5F5F5")

        if overall_rank is not None:
            ax2 = axes[0][1]
            colors_rank = [HIGHLIGHT_COLOR if ("DeepSet" in m or "deepset" in m.lower()) else BASE_COLOR
                           for m in overall_rank["method"]]
            ax2.barh(overall_rank["method"], overall_rank[rank_col], color=colors_rank)
            ax2.set_xlabel("Mean Rank (MSE)", fontsize=9)
            ax2.set_title("Mean Rank (MSE)", fontsize=10, pad=3, backgroundcolor="#DDDDDD")
            ax2.set_facecolor("#F5F5F5")

        fig.tight_layout()
        chart_path = os.path.join(out_dir, "model_comparison_overall.png")
        fig.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _upload_chart(chart_path)
        print("[INFO] Chart model_comparison_overall.png uploaded.")
    except Exception as e:
        print(f"[WARN] Chart model_comparison_overall.png failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mode = SYNREG_MODE.lower()
    print(f"[INFO] evaluate_synthetic_regression starting. MODE={mode} "
          f"SHARD={SYNREG_SHARD_INDEX}/{SYNREG_NUM_SHARDS}")

    if mode == "deepset":
        run_deepset_synthetic_regression()
    elif mode == "baselines":
        run_baseline_synthetic_regression()
    elif mode == "autogluon":
        run_autogluon_synthetic_regression()
    elif mode == "aggregate":
        run_synthetic_regression_aggregation()
    else:
        raise ValueError(
            f"Unknown SYNTHETIC_REGRESSION_MODE={mode!r}. "
            "Expected: deepset | baselines | autogluon | aggregate"
        )


if __name__ == "__main__":
    main()
