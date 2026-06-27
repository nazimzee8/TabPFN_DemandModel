"""
evaluate_nonlinear_classification.py
======================================
Nonlinear synthetic classification evaluation engine (standalone fork of
evaluate_linear_classification.py with nonlinear-specific defaults).

Mode is controlled by SYNTHETIC_CLASSIFICATION_MODE env var:
  deepset    — TabPFN inference with classification checkpoint
  baselines  — sklearn/boosting baseline models
  autogluon  — AutoGluon TabularPredictor
  aggregate  — gather shard outputs, write summary CSVs and manifest JSON

Environment variables
---------------------
SYNTHETIC_CLASSIFICATION_MODE          deepset|baselines|autogluon|aggregate
SYNTHETIC_CLASSIFICATION_SUITE_ID      suite identifier (default: nonlinear_classification)
SYNCLS_RESULTS_STAGE                   stage path for output files
SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH   stage path for classification checkpoint
SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR  train_f_classif (default)
SYNTHETIC_CLASSIFICATION_NUM_SHARDS    total shard count
SYNTHETIC_CLASSIFICATION_SHARD_INDEX   this shard's index (0-based)
SYNTHETIC_CLASSIFICATION_CONTEXT_SIZE      context size (default 200)
SYNTHETIC_CLASSIFICATION_CONTEXT_ENSEMBLES ensemble count (default 5)
SYNTHETIC_CLASSIFICATION_TEST_BATCH_SIZE   test batch size (default 128)
SYNCLS_CLUSTER_SHARDS                  0 = single-node autogluon (default)
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("HOME", "/tmp")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNCLS_MODE           = os.getenv("SYNTHETIC_CLASSIFICATION_MODE", "deepset")
SYNCLS_SUITE_ID       = os.getenv("SYNTHETIC_CLASSIFICATION_SUITE_ID", "nonlinear_classification")
SYNCLS_RESULTS_STAGE  = os.getenv("SYNCLS_RESULTS_STAGE", "@EVALUATION_RESULTS_STAGE/nonlinear/classification/numeric/nonlinear_classification")
SYNCLS_CKPT_STAGE     = os.getenv("SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH", "@MODEL_STAGE/checkpoints/best_classification.pt")
SYNCLS_FEATURE_SEL    = os.getenv("SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR", "train_f_classif")
SYNCLS_BASELINE_FEATURE_MODES = tuple(
    mode.strip()
    for mode in os.getenv(
        "SYNTHETIC_EVAL_BASELINE_FEATURE_MODES", "raw,selected"
    ).split(",")
    if mode.strip() in {"raw", "selected"}
) or ("selected",)
SYNCLS_GPU_MEMORY_BYTES = int(
    os.getenv("SYNTHETIC_EVAL_GPU_MEMORY_BYTES", "0")
)
SYNCLS_GPU_MEMORY_SAFETY_FACTOR = float(
    os.getenv("SYNTHETIC_EVAL_GPU_MEMORY_SAFETY_FACTOR", "0.8")
)
SYNCLS_MIN_CONTEXT_SIZE = int(
    os.getenv("SYNTHETIC_EVAL_MIN_CONTEXT_SIZE", "32")
)
SYNCLS_MIN_QUERY_BATCH_SIZE = int(
    os.getenv("SYNTHETIC_EVAL_MIN_QUERY_BATCH_SIZE", "1")
)
SYNCLS_NUM_SHARDS     = int(os.getenv("SYNTHETIC_CLASSIFICATION_NUM_SHARDS", "10"))
SYNCLS_SHARD_INDEX    = int(os.getenv("SYNTHETIC_CLASSIFICATION_SHARD_INDEX", "0"))
SYNCLS_CONTEXT_SIZE    = int(os.getenv("SYNTHETIC_CLASSIFICATION_CONTEXT_SIZE", "200"))
SYNCLS_ENSEMBLES       = int(os.getenv("SYNTHETIC_CLASSIFICATION_CONTEXT_ENSEMBLES", "5"))
SYNCLS_TEST_BATCH      = int(os.getenv("SYNTHETIC_CLASSIFICATION_TEST_BATCH_SIZE", "128"))
SYNCLS_CLUSTER_SHARDS  = int(os.getenv("SYNCLS_CLUSTER_SHARDS", "0"))
SYNCLS_LOCAL_DIR       = os.getenv("SYNTHETIC_CLASSIFICATION_LOCAL_DIR", "/tmp/syncls_eval")
SYNCLS_INDEX_TABLE     = os.getenv("SYNCLS_INDEX_TABLE", "NONLINEAR_CLASSIFICATION_DATASET_INDEX")
SYNCLS_IS_MIXED_CAT    = os.getenv("SYNCLS_IS_MIXED_CATEGORICAL", "false").lower() in ("1", "true", "yes")
SYNCLS_PARTS_PREFIX    = f"{SYNCLS_RESULTS_STAGE}/parts"
SYNCLS_AG_TIME_LIMIT   = int(os.getenv("SYNCLS_AUTOGLUON_TIME_LIMIT", "300"))
SYNCLS_AG_PRESETS      = os.getenv("SYNCLS_AUTOGLUON_PRESETS", "high_quality")
SYNCLS_AG_TASK_CPUS    = int(os.getenv("SYNCLS_AUTOGLUON_TASK_CPUS", "1"))

# MODEL4 inference parameters (Phase 2)
SYNCLS_N_ENSEMBLES      = int(os.getenv("SYNCLS_N_ENSEMBLES", "8"))
SYNCLS_TEST_BATCH_SIZE  = int(os.getenv("SYNCLS_TEST_BATCH_SIZE", "256"))

# Checkpoint quality gates (CLS-O1).
# Reads SYNCLS_* (canonical).  Falls back to SYNREG_* for one deprecation cycle
# in case operators set the legacy regression prefix by mistake.
SYNCLS_RUN_CHECKPOINT_GATES = bool(
    os.getenv("SYNCLS_RUN_CHECKPOINT_GATES") or os.getenv("SYNREG_RUN_CHECKPOINT_GATES")
)
SYNCLS_CHECKPOINT_GATE_STRICT = bool(
    os.getenv("SYNCLS_CHECKPOINT_GATE_STRICT") or os.getenv("SYNREG_CHECKPOINT_GATE_STRICT")
)

# Metric schema version
# '1.0' = stored DGP teacher probabilities (deprecated; NOT real model inference)
# '2.0' = real MODEL4 forward pass using actual checkpoint logits
SYNCLS_METRIC_SCHEMA_VERSION = "2.0"

# Phase 3: Run-scoped evaluation identifiers
SYNCLS_EVALUATION_RUN_ID = os.getenv("SYNCLS_EVALUATION_RUN_ID", "")
SYNCLS_MIN_COVERAGE_THRESHOLD = float(os.getenv("SYNCLS_MIN_COVERAGE_THRESHOLD", "0.0"))
SYNCLS_BOOTSTRAP_SEED = int(os.getenv("SYNCLS_BOOTSTRAP_SEED", "42"))

# Phase 8: Evaluation protocol
SYNCLS_EVAL_PROTOCOL = os.getenv("SYNCLS_EVAL_PROTOCOL", "train_selected_features")


def _is_legacy_suite() -> bool:
    return bool(
        os.getenv("SYNTHETIC_EVAL_LEGACY_SUITE", "").lower()
        in {"1", "true", "yes"}
        or SYNCLS_SUITE_ID.endswith("_v1")
    )


def _task_key(ds: dict, dataset_id) -> str:
    suite_family = str(ds.get("suite_family", "primary"))
    global_idx = ds.get("global_idx")
    if global_idx is not None:
        return f"{SYNCLS_SUITE_ID}:{suite_family}:{global_idx}"
    split_seed = ds.get("split_seed", 0)
    return f"{SYNCLS_SUITE_ID}:{suite_family}:{dataset_id}:{split_seed}"


def _result_identity(ds: dict, dataset_id, *, prediction_source: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "suite_id": SYNCLS_SUITE_ID,
        "suite_family": ds.get("suite_family", "primary"),
        "global_idx": ds.get("global_idx"),
        "task_key": _task_key(ds, dataset_id),
        "prediction_source": prediction_source,
        "result_status": "success",
    }

# Guard: result path must contain /nonlinear/classification/
assert "/nonlinear/classification/" in SYNCLS_RESULTS_STAGE, (
    f"SYNCLS_RESULTS_STAGE must contain '/nonlinear/classification/', got: {SYNCLS_RESULTS_STAGE!r}"
)

# ---------------------------------------------------------------------------
# Snowflake session helpers
# ---------------------------------------------------------------------------

def _get_session():
    from snowflake.snowpark import Session
    return Session.builder.getOrCreate()


def _download_from_stage(session, stage_path: str, local_dir: str) -> Path:
    """Download a file from Snowflake stage to local_dir."""
    os.makedirs(local_dir, exist_ok=True)
    session.file.get(stage_path, local_dir)
    filename = stage_path.rsplit("/", 1)[-1]
    local_path = Path(local_dir) / filename
    if not local_path.exists():
        candidates = list(Path(local_dir).glob(filename))
        if candidates:
            local_path = candidates[0]
    return local_path


def _upload_to_stage(session, local_path: str, stage_dir: str) -> str:
    """Upload local file to Snowflake stage directory."""
    result = session.file.put(
        str(local_path),
        stage_dir,
        auto_compress=False,
        overwrite=True,
    )
    filename = Path(local_path).name
    return f"{stage_dir}/{filename}"


def _list_stage_files(session, stage_prefix: str, pattern: str = "") -> list[str]:
    """List files on stage, optionally filtered by pattern substring."""
    try:
        rows = session.sql(f"LIST {stage_prefix}").collect()
        paths = [r["name"] for r in rows]
        if pattern:
            paths = [p for p in paths if pattern in p]
        return paths
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Data loading from LINEAR_CLASSIFICATION_DATASET_INDEX
# ---------------------------------------------------------------------------

def _load_datasets_for_shard(session) -> list[dict]:
    """Load this shard's slice of datasets from the classification index."""
    rows = session.sql(
        f"SELECT * FROM {SYNCLS_INDEX_TABLE} "
        f"WHERE suite_id = '{SYNCLS_SUITE_ID}' "
        f"ORDER BY dataset_id"
    ).collect()

    all_rows = [dict(r.as_dict()) if hasattr(r, "as_dict") else dict(r) for r in rows]
    all_rows = [{k.lower(): v for k, v in r.items()} for r in all_rows]

    shard_rows = [
        r for i, r in enumerate(all_rows)
        if i % SYNCLS_NUM_SHARDS == SYNCLS_SHARD_INDEX
    ]
    print(
        f"[INFO] Shard {SYNCLS_SHARD_INDEX}/{SYNCLS_NUM_SHARDS}: "
        f"{len(shard_rows)}/{len(all_rows)} datasets",
        flush=True,
    )
    return shard_rows


def _download_parquet_dataset(session, stage_path: str, local_dir: str):
    """Download and load a classification parquet dataset.

    Returns a task dict produced by load_classification_eval_task() with keys:
    X_train, y_train, X_test, y_test, num_classes, n_train, n_test, n_features,
    suite_id, suite_family, label_noise_rate, feature_noise_level, margin_level,
    task_seed, regime, profile, dataset_idx, global_idx, is_tabpfn_anchor,
    task_family, task_objective, distribution_family, covariance_type, rho,
    temperature, imbalance_type, y_clean_test, label_noise_mask_test,
    dgp_teacher_probs_test.
    """
    import sys as _sys
    import os as _os
    _src = _os.path.join(_os.path.dirname(__file__), "..", "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        import pyarrow.parquet as pq

    from dgp_helpers import load_classification_eval_task

    local_path = _download_from_stage(session, stage_path, local_dir)
    tbl = pq.read_table(str(local_path))
    return load_classification_eval_task(tbl)


def _download_mixed_parquet_dataset(session, stage_path: str, local_dir: str):
    """Download and load a mixed-categorical classification parquet dataset.

    Returns task dict with standard keys PLUS X_cat_train, X_cat_test,
    categorical_cardinalities, p_num, p_cat, max_cardinality.
    """
    import sys as _sys
    import os as _os
    _src = _os.path.join(_os.path.dirname(__file__), "..", "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        import pyarrow.parquet as pq

    from dgp_helpers import load_mixed_classification_eval_task

    local_path = _download_from_stage(session, stage_path, local_dir)
    tbl = pq.read_table(str(local_path))
    return load_mixed_classification_eval_task(tbl)


# ---------------------------------------------------------------------------
# Mixed-categorical feature encoding for baselines
# ---------------------------------------------------------------------------

def _prepare_mixed_features(X_num, X_cat, cat_cardinalities, model_family):
    """Encode categorical features for a specific baseline model family.

    Parameters
    ----------
    X_num : np.ndarray, shape (n, p_num) — numeric features
    X_cat : np.ndarray, shape (n, p_cat) — entity-embed-encoded categorical features
    cat_cardinalities : array-like — real cardinality per categorical feature
    model_family : str — 'linear', 'tree_ordinal', 'catboost', or 'autogluon'

    Returns
    -------
    X_combined : np.ndarray
    """
    import numpy as np
    import sys as _sys
    import os as _os
    _src = _os.path.join(_os.path.dirname(__file__), "..", "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from constants import ENTITY_EMBED_FIRST_REAL_ID

    # Convert from entity-embed IDs to 0-based ordinal
    X_cat_ordinal = np.asarray(X_cat, dtype=np.int64) - ENTITY_EMBED_FIRST_REAL_ID
    X_cat_ordinal = np.clip(X_cat_ordinal, 0, None)  # clamp special tokens to 0

    if model_family == "linear":
        # One-hot encode for linear models (Ridge, Lasso, LogisticRegression, etc.)
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            categories=[list(range(int(c))) for c in cat_cardinalities],
        )
        X_cat_oh = enc.fit_transform(X_cat_ordinal)
        return np.hstack([X_num, X_cat_oh])

    # tree_ordinal, catboost, autogluon — all use ordinal encoding
    return np.hstack([X_num, X_cat_ordinal.astype(np.float64)])


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _apply_feature_selector(X_train, X_test, y_train, feature_selector: str):
    """Apply feature selection to reduce irrelevant features."""
    try:
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    except ImportError:
        return X_train, X_test

    import numpy as np
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)

    n, p = X_train.shape
    if p <= 1:
        return X_train, X_test

    k = min(p, max(1, int(math.sqrt(p * n / max(n, 1)))))
    score_func = f_classif if "classif" in feature_selector else f_regression

    try:
        sel = SelectKBest(score_func=score_func, k=k)
        X_train_sel = sel.fit_transform(X_train, y_train)
        X_test_sel = sel.transform(X_test)
        return X_train_sel, X_test_sel
    except Exception:
        return X_train, X_test


def _feature_variants(X_train, X_test, y_train):
    import numpy as np

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    for mode in SYNCLS_BASELINE_FEATURE_MODES:
        if mode == "raw":
            yield "raw", X_train, X_test, "none"
        else:
            X_train_sel, X_test_sel = _apply_feature_selector(
                X_train, X_test, y_train, "train_f_classif"
            )
            yield "selected", X_train_sel, X_test_sel, "train_f_classif"


def _variant_model_name(base_name: str, feature_mode: str) -> str:
    if _is_legacy_suite():
        return base_name
    suffix = "raw" if feature_mode == "raw" else "train_f_classif"
    return f"{base_name}_{suffix}"


# ---------------------------------------------------------------------------
# Phase 6: Probability-class alignment
# ---------------------------------------------------------------------------

def _align_proba_to_canonical(proba, model_classes, num_classes: int):
    """Align a probability matrix to canonical class IDs 0..num_classes-1.

    Parameters
    ----------
    proba : array-like of shape (n_samples, len(model_classes))
    model_classes : array-like of class labels (e.g., model.classes_ or column names)
    num_classes : int — expected number of classes K

    Returns
    -------
    aligned : np.ndarray of shape (n_samples, num_classes)
        Aligned probability matrix. Missing classes get epsilon (1e-10).
    has_missing_support_class : bool
        True if any canonical class was absent from model_classes.
    missing_class_ids : list[int]
        IDs of absent classes (empty when has_missing_support_class is False).
    """
    import numpy as np
    epsilon = 1e-10
    proba = np.asarray(proba, dtype=np.float64)
    model_classes = [int(c) for c in model_classes]
    n_samples = proba.shape[0]
    aligned = np.full((n_samples, num_classes), epsilon, dtype=np.float64)
    for col_idx, cls_id in enumerate(model_classes):
        if 0 <= cls_id < num_classes:
            aligned[:, cls_id] = proba[:, col_idx]
    missing_classes = sorted(set(range(num_classes)) - set(model_classes))
    has_missing = len(missing_classes) > 0
    # Renormalize
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    aligned /= row_sums
    return aligned, has_missing, missing_classes


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _compute_ece(y_true, proba, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    import numpy as np
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    if proba.ndim == 2:
        # Multiclass: use max probability
        confidences = proba.max(axis=1)
        predictions = proba.argmax(axis=1)
        correctness = (predictions == y_true).astype(float)
    else:
        # Binary
        confidences = np.where(proba >= 0.5, proba, 1 - proba)
        predictions = (proba >= 0.5).astype(int)
        correctness = (predictions == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def _compute_classification_metrics(
    y_true, y_pred, proba, num_classes: int, model_name: str,
    *,
    has_missing_support: bool = False,
    missing_support_classes: list | None = None,
) -> dict:
    """Compute all classification metrics for a single model.

    Parameters
    ----------
    has_missing_support : bool
        True when the aligned probability matrix had at least one class absent
        (epsilon-padded). Multiclass AUC is suppressed in this case.
    missing_support_classes : list[int] | None
        IDs of classes absent from the model output (used for structured warnings).
    """
    import numpy as np
    from sklearn import metrics as skm

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    canonical_labels = list(range(num_classes))

    result = {
        "model_name": model_name,
        "num_classes": num_classes,
    }
    metric_warnings = []

    # Label-only metrics
    result["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
    result["balanced_accuracy"] = float(skm.balanced_accuracy_score(y_true, y_pred))
    result["macro_f1"] = float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
    result["weighted_f1"] = float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0))
    result["macro_precision"] = float(
        skm.precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result["macro_recall"] = float(
        skm.recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result["mcc"] = float(skm.matthews_corrcoef(y_true, y_pred))
    result["cohen_kappa"] = float(skm.cohen_kappa_score(y_true, y_pred))

    # F4: metric status fields
    result["has_missing_support"] = bool(has_missing_support)
    result["missing_support_classes"] = list(missing_support_classes or [])

    # Probability-based metrics
    if proba is not None:
        proba = np.asarray(proba)
        # F4-A: pass labels= so log_loss handles absent classes correctly.
        try:
            result["log_loss"] = float(
                skm.log_loss(y_true, proba, labels=canonical_labels)
            )
            result["log_loss_valid"] = True
        except Exception as e:
            result["log_loss"] = float("nan")
            result["log_loss_valid"] = False
            metric_warnings.append(f"log_loss: {e}")

        roc_auc_valid = True
        try:
            if num_classes == 2:
                p_pos = proba[:, 1] if proba.ndim == 2 else proba
                result["roc_auc_ovr"] = float(skm.roc_auc_score(y_true, p_pos))
                result["roc_auc_ovo"] = result["roc_auc_ovr"]
                result["brier_score"] = float(skm.brier_score_loss(y_true, p_pos))
                result["average_precision"] = float(
                    skm.average_precision_score(y_true, p_pos)
                )
            else:
                # F4-D: skip multiclass AUC when epsilon-padded columns are present.
                if has_missing_support:
                    result["roc_auc_ovr"] = float("nan")
                    result["roc_auc_ovo"] = float("nan")
                    roc_auc_valid = False
                    metric_warnings.append(
                        f"roc_auc skipped: missing_support_classes={missing_support_classes} "
                        "(epsilon-padded columns would produce misleading AUC)"
                    )
                else:
                    result["roc_auc_ovr"] = float(skm.roc_auc_score(
                        y_true, proba, multi_class="ovr", average="macro"
                    ))
                    result["roc_auc_ovo"] = float(skm.roc_auc_score(
                        y_true, proba, multi_class="ovo", average="macro"
                    ))
                one_hot = np.eye(num_classes)[y_true.astype(int)]
                result["brier_score"] = float(
                    np.mean(np.sum((proba - one_hot) ** 2, axis=1))
                )
                result["average_precision"] = float(
                    skm.average_precision_score(
                        one_hot, proba, average="macro"
                    )
                )
        except Exception as e:
            result["roc_auc_ovr"] = float("nan")
            result["roc_auc_ovo"] = float("nan")
            result["brier_score"] = float("nan")
            result["average_precision"] = float("nan")
            roc_auc_valid = False
            metric_warnings.append(f"probability_metrics: {e}")

        result["roc_auc_valid"] = roc_auc_valid
        result["brier_score_valid"] = "brier_score" in result and not (
            isinstance(result.get("brier_score"), float) and
            not (result["brier_score"] == result["brier_score"])  # nan check
        )

        # ECE
        try:
            result["expected_calibration_error"] = _compute_ece(y_true, proba)
        except Exception:
            result["expected_calibration_error"] = float("nan")

        # F4-B: pass labels= so top_k_accuracy handles absent classes correctly.
        for k in (2, 3):
            if num_classes <= k:
                result[f"top_{k}_accuracy"] = float("nan")
                continue
            try:
                result[f"top_{k}_accuracy"] = float(
                    skm.top_k_accuracy_score(
                        y_true, proba, k=k, labels=canonical_labels
                    )
                )
            except Exception as exc:
                result[f"top_{k}_accuracy"] = float("nan")
                metric_warnings.append(f"top_{k}_accuracy: {exc}")
    else:
        # No proba — NaN for all proba metrics
        for m in ["log_loss", "roc_auc_ovr", "roc_auc_ovo", "brier_score",
                  "expected_calibration_error", "average_precision",
                  "top_2_accuracy", "top_3_accuracy"]:
            result[m] = float("nan")
        result["log_loss_valid"] = False
        result["roc_auc_valid"] = False
        result["brier_score_valid"] = False

    result["metric_warning"] = "; ".join(metric_warnings) if metric_warnings else ""

    return result


# ---------------------------------------------------------------------------
# MODEL4 inference helper
# ---------------------------------------------------------------------------

def _guard_inference_sizes(
    n_train: int,
    n_test: int,
    p_selected: int,
    context_size: int,
    query_batch_size: int,
) -> tuple[int, int, str | None, int]:
    """Downshift query batch first, then context, using the suite memory formula."""
    budget = SYNCLS_GPU_MEMORY_BYTES
    if budget <= 0:
        try:
            import torch
            if torch.cuda.is_available():
                budget = int(torch.cuda.mem_get_info()[0])
        except Exception:
            budget = 0
    if budget <= 0:
        return context_size, query_batch_size, None, 0
    usable = int(budget * SYNCLS_GPU_MEMORY_SAFETY_FACTOR)

    def estimate(context: int, query: int) -> int:
        return int(
            2.0
            * 4
            * 128
            * min(max(1, p_selected), 256)
            * (min(context, 1024) + min(query, 256))
        )

    context = min(context_size, n_train)
    query = min(query_batch_size, max(1, n_test))
    while estimate(context, query) > usable and query > SYNCLS_MIN_QUERY_BATCH_SIZE:
        query = max(SYNCLS_MIN_QUERY_BATCH_SIZE, query // 2)
    while estimate(context, query) > usable and context > SYNCLS_MIN_CONTEXT_SIZE:
        context = max(SYNCLS_MIN_CONTEXT_SIZE, context // 2)
    estimated = estimate(context, query)
    reason = None
    if estimated > usable:
        reason = (
            f"gpu_memory_guard:estimated_bytes={estimated}:"
            f"usable_bytes={usable}"
        )
    return context, query, reason, estimated

def _run_model4_inference(
    ckpt_dict, X_train, y_train, X_test, num_classes,
    context_size=None, n_ensembles=8, test_batch_size=256,
    *,
    X_cat_train=None, X_cat_test=None, categorical_cardinalities=None,
):
    """Run actual MODEL4 forward pass and return probabilities plus selector metadata.

    Uses checkpoint weights directly — does NOT use stored DGP teacher probabilities.
    """
    import numpy as np
    import torch

    try:
        from model import build_model_from_config
        from deepset_inference import (
            select_deepset_classification_features_train_only,
            select_deepset_context_indices,
            resolve_deepset_feature_cap,
        )
    except ImportError as exc:
        raise ImportError(
            f"Required inference modules not importable: {exc}. "
            "Ensure model.py and deepset_inference.py are on the Python path."
        ) from exc

    cfg = ckpt_dict.get("cfg", {})
    if isinstance(cfg, dict):
        cfg.setdefault("model_family", "market_exchangeable_icl")
        cfg.setdefault("task_objective", "inductive_classification")
        cfg.setdefault("model_design_pattern", "inductive_forecasting")
    model = build_model_from_config(cfg)
    state_dict = ckpt_dict.get("state_dict") or ckpt_dict.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(
            "Checkpoint missing both 'state_dict' (canonical) and 'model_state_dict' (legacy)."
        )
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Fail-fast permutation contract gates
    _run_perm_gates = os.getenv("CLASSIFICATION_RUN_PERMUTATION_GATES", "true").lower() == "true"
    _perm_gate_strict = os.getenv("CLASSIFICATION_PERMUTATION_GATE_STRICT", "true").lower() == "true"
    if _run_perm_gates:
        try:
            from permutation_contracts import run_all
            gate_results = run_all(model)
            # Only enforce F1-F4 (support, query, feature, coefficient); skip F5 (class-label)
            failures = [
                r for r in gate_results
                if not r.passed and "skipped" not in r.failure_reason
                and not r.check_type.startswith("F5_")
            ]
            if failures:
                msgs = [f"  {r.check_type}: {r.failure_reason}" for r in failures]
                msg = f"Permutation gate failures ({len(failures)}):\n" + "\n".join(msgs)
                if _perm_gate_strict:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}", flush=True)
            else:
                print(f"[INFO] All {len(gate_results)} permutation gates passed.", flush=True)
        except ImportError:
            print("[WARN] permutation_contracts not available; skipping gates.", flush=True)

    feature_cap = resolve_deepset_feature_cap(model)
    X_tr_sel, X_te_sel, selector_metadata = select_deepset_classification_features_train_only(
        X_train, y_train, X_test, feature_cap
    )
    n_test = X_te_sel.shape[0]
    ctx_size = context_size or min(len(y_train), cfg.get("max_n_train", 1024))
    if ctx_size == 0:
        ctx_size = len(y_train)
    ctx_size, test_batch_size, skip_reason, estimated_bytes = _guard_inference_sizes(
        len(y_train), n_test, X_tr_sel.shape[1], ctx_size, test_batch_size
    )
    if skip_reason:
        return None, {
            "feature_selector": "train_f_classif",
            "selected_features": int(X_tr_sel.shape[1]),
            "selector_metadata": selector_metadata,
            "gpu_memory_estimated_bytes": estimated_bytes,
            "skip_reason": skip_reason,
        }
    all_probs = np.zeros((n_test, num_classes), dtype=np.float64)

    # Pre-compute categorical tensors (mixed-cat mode)
    _has_cat = X_cat_train is not None and X_cat_test is not None
    if _has_cat:
        _cat_cards_t = torch.as_tensor(
            np.asarray(categorical_cardinalities, dtype=np.int64),
            dtype=torch.long, device=device,
        )

    with torch.no_grad():
        for ens_idx in range(n_ensembles):
            ctx_idx = select_deepset_context_indices(
                len(y_train), ctx_size, seed=0,
                context_index=ens_idx, context_ensembles=n_ensembles,
            )
            X_ctx = torch.tensor(X_tr_sel[ctx_idx], dtype=torch.float32, device=device)
            y_ctx = torch.tensor(y_train[ctx_idx], dtype=torch.long, device=device)

            # Categorical context tensors (sliced by ctx_idx)
            cat_fwd_ctx = {}
            if _has_cat:
                cat_fwd_ctx["X_cat_train"] = torch.as_tensor(
                    X_cat_train[ctx_idx], dtype=torch.long, device=device,
                )
                cat_fwd_ctx["categorical_cardinalities"] = _cat_cards_t

            for b in range(0, n_test, test_batch_size):
                X_b = torch.tensor(
                    X_te_sel[b:b + test_batch_size], dtype=torch.float32, device=device
                )

                # Categorical query tensors
                cat_fwd_batch = {}
                if _has_cat:
                    cat_fwd_batch["X_cat_test"] = torch.as_tensor(
                        X_cat_test[b:b + test_batch_size], dtype=torch.long, device=device,
                    )

                out = model(
                    X_ctx, y_ctx, X_b,
                    task_objective="inductive_classification",
                    num_classes=num_classes,
                    **cat_fwd_ctx,
                    **cat_fwd_batch,
                )
                logits = out["logits"].float().cpu().numpy()
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                all_probs[b:b + len(e)] += e / e.sum(axis=1, keepdims=True)

    return all_probs / n_ensembles, {
        "feature_selector": "train_f_classif",
        "selected_features": int(X_tr_sel.shape[1]),
        "selector_metadata": selector_metadata,
        "gpu_memory_estimated_bytes": estimated_bytes,
        "context_size_used": ctx_size,
        "query_batch_size_used": test_batch_size,
    }


# ---------------------------------------------------------------------------
# DeepSet mode
# ---------------------------------------------------------------------------

def _run_deepset_mode(session) -> str:
    """Run TabPFN inference on this shard's datasets."""
    import numpy as np
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        import pyarrow as pa
        import pyarrow.parquet as pq

    # Validate checkpoint
    local_ckpt_dir = os.path.join(SYNCLS_LOCAL_DIR, "ckpt")
    os.makedirs(local_ckpt_dir, exist_ok=True)
    ckpt_local = _download_from_stage(session, SYNCLS_CKPT_STAGE, local_ckpt_dir)
    print(f"[INFO] Checkpoint downloaded: {ckpt_local}", flush=True)

    try:
        from sanity_checks import validate_classification_checkpoint
        validate_classification_checkpoint(str(ckpt_local))
        print("[INFO] Checkpoint validated as classification checkpoint.", flush=True)
    except ValueError as exc:
        raise RuntimeError(f"Checkpoint validation failed: {exc}") from exc
    except Exception as exc:
        print(f"[WARN] Checkpoint validation skipped: {exc}", flush=True)

    # Load model
    import torch
    ckpt = torch.load(str(ckpt_local), map_location="cpu", weights_only=False)
    print(f"[INFO] task_objective in checkpoint: {ckpt.get('task_objective')}", flush=True)

    datasets = _load_datasets_for_shard(session)
    if not datasets:
        print("[INFO] No datasets for this shard.", flush=True)
        return f"deepset_shard_{SYNCLS_SHARD_INDEX}: 0 datasets"

    result_rows = []
    local_data_dir = os.path.join(SYNCLS_LOCAL_DIR, "data")
    os.makedirs(local_data_dir, exist_ok=True)

    for ds in datasets:
        dataset_id = ds.get("dataset_id")
        stage_path = ds.get("stage_path")
        num_classes = int(ds.get("num_classes") or 2)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if SYNCLS_IS_MIXED_CAT:
                    data = _download_mixed_parquet_dataset(session, stage_path, tmp_dir)
                else:
                    data = _download_parquet_dataset(session, stage_path, tmp_dir)

                X_train = data["X_train"]
                y_train = data["y_train"]
                X_test  = data["X_test"]
                y_test  = data["y_test"]

                # Build categorical kwargs for mixed-cat datasets
                cat_kwargs = {}
                if SYNCLS_IS_MIXED_CAT and "X_cat_train" in data:
                    cat_kwargs["X_cat_train"] = data["X_cat_train"]
                    cat_kwargs["X_cat_test"] = data["X_cat_test"]
                    cat_kwargs["categorical_cardinalities"] = data["categorical_cardinalities"]

                # Real MODEL4 inference — uses checkpoint weights, not stored DGP probs.
                # Ground-truth probs_test are available in data for Bayes-reference diagnostics only.
                probs_test, selector_meta = _run_model4_inference(
                    ckpt, X_train, y_train, X_test, num_classes,
                    context_size=SYNCLS_CONTEXT_SIZE,
                    n_ensembles=SYNCLS_N_ENSEMBLES,
                    test_batch_size=SYNCLS_TEST_BATCH_SIZE,
                    **cat_kwargs,
                )
                if probs_test is None:
                    result_rows.append({
                        **_result_identity(
                            ds, dataset_id, prediction_source="model4_checkpoint"
                        ),
                        "shard_index": SYNCLS_SHARD_INDEX,
                        "mode": "deepset",
                        "model_name": "MODEL-ICL-MC",
                        "result_status": "skipped",
                        "skip_reason": selector_meta["skip_reason"],
                        "selected_features": selector_meta["selected_features"],
                    })
                    continue
                y_pred = probs_test.argmax(axis=1)

                metrics = _compute_classification_metrics(
                    y_test, y_pred, probs_test, num_classes, "MODEL-ICL-MC"
                )
                metrics.update(_result_identity(
                    ds, dataset_id, prediction_source="model4_checkpoint"
                ))
                metrics["shard_index"] = SYNCLS_SHARD_INDEX
                metrics["mode"] = "deepset"
                metrics["metric_schema_version"] = SYNCLS_METRIC_SCHEMA_VERSION
                metrics["feature_selector"] = selector_meta["feature_selector"]
                metrics["selected_features"] = selector_meta["selected_features"]
                metrics["gpu_memory_estimated_bytes"] = selector_meta[
                    "gpu_memory_estimated_bytes"
                ]

                # DGP-teacher diagnostic: stored DGP teacher probs (NOT model predictions,
                # NOT Bayes-optimal — teacher at finite temperature is not the Bayes predictor).
                bayes_probs = data.get("dgp_teacher_probs_test")
                if bayes_probs is not None and np.asarray(bayes_probs).size > 0:
                    try:
                        from sklearn.metrics import log_loss
                        metrics["dgp_teacher_log_loss"] = float(log_loss(y_test, bayes_probs))
                    except Exception:
                        pass

                # Copy relevant index metadata
                for field in ["prior_regime", "classification_regime", "suite_family",
                               "feature_noise_level", "label_noise_rate", "num_classes",
                               "sample_complexity_bucket", "margin_level"]:
                    if field in ds:
                        metrics[field] = ds[field]
                result_rows.append(metrics)

        except Exception as exc:
            print(f"[ERROR] dataset_id={dataset_id}: {exc}", flush=True)
            result_rows.append({
                **_result_identity(
                    ds, dataset_id, prediction_source="model4_checkpoint"
                ),
                "shard_index": SYNCLS_SHARD_INDEX,
                "mode": "deepset",
                "model_name": "MODEL-ICL-MC",
                "result_status": "failed",
                "error": str(exc),
            })

    # Write shard parquet
    os.makedirs(SYNCLS_LOCAL_DIR, exist_ok=True)
    out_name = f"deepset_shard_{SYNCLS_SHARD_INDEX}.parquet"
    out_path = os.path.join(SYNCLS_LOCAL_DIR, out_name)
    _write_results_parquet(result_rows, out_path)
    _upload_to_stage(session, out_path, SYNCLS_PARTS_PREFIX)
    print(f"[INFO] Deepset shard {SYNCLS_SHARD_INDEX} done: {len(result_rows)} rows.", flush=True)
    return f"deepset_shard_{SYNCLS_SHARD_INDEX}: {len(result_rows)} datasets"


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def _get_baseline_models(num_classes: int):
    """Return list of (name, model) tuples for baseline evaluation."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    models = [
        ("DummyClassifier_most_frequent", DummyClassifier(strategy="most_frequent")),
        ("DummyClassifier_stratified",    DummyClassifier(strategy="stratified")),
        ("LogisticRegression",            LogisticRegression(max_iter=1000)),
        ("RidgeClassifier",               RidgeClassifier()),
        ("LinearSVC",                     LinearSVC(max_iter=2000)),
        ("RandomForestClassifier",        RandomForestClassifier(n_estimators=100)),
    ]

    try:
        from xgboost import XGBClassifier
        models.append(("XGBClassifier", XGBClassifier(eval_metric="logloss", verbosity=0)))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models.append(("LGBMClassifier", LGBMClassifier(verbosity=-1)))
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models.append(("CatBoostClassifier", CatBoostClassifier(verbose=0)))
    except ImportError:
        pass

    models.append(("KNeighborsClassifier", KNeighborsClassifier()))
    models.append(("MLPClassifier", MLPClassifier(max_iter=500)))

    return models


def _has_predict_proba(model) -> bool:
    """Return True if model supports predict_proba."""
    return hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))


def _run_baselines_mode(session) -> str:
    """Run all baseline models on this shard's datasets."""
    import numpy as np
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        import pyarrow as pa
        import pyarrow.parquet as pq

    datasets = _load_datasets_for_shard(session)
    if not datasets:
        print("[INFO] No datasets for this shard.", flush=True)
        return f"baselines_shard_{SYNCLS_SHARD_INDEX}: 0 datasets"

    result_rows = []

    for ds in datasets:
        dataset_id = ds.get("dataset_id")
        stage_path = ds.get("stage_path")
        num_classes = int(ds.get("num_classes") or 2)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if SYNCLS_IS_MIXED_CAT:
                    data = _download_mixed_parquet_dataset(session, stage_path, tmp_dir)
                else:
                    data = _download_parquet_dataset(session, stage_path, tmp_dir)

                X_train = data["X_train"]
                y_train = data["y_train"]
                X_test  = data["X_test"]
                y_test  = data["y_test"]

                # Mixed-cat: combine numeric + encoded categorical for baselines
                if SYNCLS_IS_MIXED_CAT and "X_cat_train" in data:
                    _cat_cards = data["categorical_cardinalities"]
                    _X_cat_train = data["X_cat_train"]
                    _X_cat_test = data["X_cat_test"]
                else:
                    _cat_cards = None
                    _X_cat_train = None
                    _X_cat_test = None

                for feature_mode, X_train_used, X_test_used, selector in _feature_variants(
                    X_train, X_test, y_train
                ):
                    # For mixed-cat, combine numeric selected features with encoded categoricals
                    if _cat_cards is not None:
                        # Determine encoding family from model name
                        _linear_baselines = {
                            "logistic_regression", "ridge_classifier", "lasso",
                            "elastic_net", "knn", "svr", "mlp",
                        }
                        _tree_baselines = {
                            "random_forest", "xgboost", "lightgbm",
                        }

                        def _get_model_family(name):
                            name_lower = name.lower()
                            if "catboost" in name_lower:
                                return "catboost"
                            for t in _tree_baselines:
                                if t in name_lower:
                                    return "tree_ordinal"
                            return "linear"

                    baseline_models = _get_baseline_models(num_classes)
                    for base_name, model in baseline_models:
                        model_name = _variant_model_name(base_name, feature_mode)
                        try:
                            if _cat_cards is not None:
                                mfam = _get_model_family(base_name)
                                X_train_fit = _prepare_mixed_features(
                                    X_train_used, _X_cat_train, _cat_cards, mfam
                                )
                                X_test_fit = _prepare_mixed_features(
                                    X_test_used, _X_cat_test, _cat_cards, mfam
                                )
                            else:
                                X_train_fit = X_train_used
                                X_test_fit = X_test_used
                            model.fit(X_train_fit, y_train)
                            y_pred = model.predict(X_test_fit)

                            proba = None
                            has_missing_support = False
                            missing_support_cls: list[int] = []
                            if _has_predict_proba(model):
                                try:
                                    raw_proba = model.predict_proba(X_test_fit)
                                    if hasattr(model, "classes_") and len(model.classes_) < num_classes:
                                        proba, has_missing_support, missing_support_cls = (
                                            _align_proba_to_canonical(
                                                raw_proba, model.classes_, num_classes
                                            )
                                        )
                                    else:
                                        proba = raw_proba
                                except Exception:
                                    proba = None

                            metrics = _compute_classification_metrics(
                                y_test, y_pred, proba, num_classes, model_name,
                                has_missing_support=has_missing_support,
                                missing_support_classes=missing_support_cls,
                            )
                            metrics["has_missing_support_class"] = has_missing_support
                            metrics.update(_result_identity(
                                ds, dataset_id, prediction_source="fitted_baseline"
                            ))
                            metrics["shard_index"] = SYNCLS_SHARD_INDEX
                            metrics["mode"] = "baselines"
                            metrics["feature_mode"] = feature_mode
                            metrics["feature_selector"] = selector
                            metrics["selected_features"] = int(X_train_used.shape[1])
                            metrics["metric_schema_version"] = SYNCLS_METRIC_SCHEMA_VERSION
                            for field in ["prior_regime", "classification_regime", "suite_family",
                                           "feature_noise_level", "label_noise_rate", "num_classes",
                                           "sample_complexity_bucket", "margin_level"]:
                                if field in ds:
                                    metrics[field] = ds[field]
                            result_rows.append(metrics)
                        except Exception as exc:
                            print(f"[ERROR] dataset_id={dataset_id} model={model_name}: {exc}", flush=True)
                            result_rows.append({
                                **_result_identity(
                                    ds, dataset_id, prediction_source="fitted_baseline"
                                ),
                                "shard_index": SYNCLS_SHARD_INDEX,
                                "mode": "baselines",
                                "model_name": model_name,
                                "feature_mode": feature_mode,
                                "feature_selector": selector,
                                "result_status": "failed",
                                "error": str(exc),
                            })

        except Exception as exc:
            print(f"[ERROR] dataset_id={dataset_id}: {exc}", flush=True)
            result_rows.append({
                **_result_identity(
                    ds, dataset_id, prediction_source="fitted_baseline"
                ),
                "shard_index": SYNCLS_SHARD_INDEX,
                "mode": "baselines",
                "model_name": "ERROR",
                "result_status": "failed",
                "error": str(exc),
            })

    # Write shard parquet
    out_name = f"baselines_shard_{SYNCLS_SHARD_INDEX}.parquet"
    out_path = os.path.join(SYNCLS_LOCAL_DIR, out_name)
    os.makedirs(SYNCLS_LOCAL_DIR, exist_ok=True)
    _write_results_parquet(result_rows, out_path)
    _upload_to_stage(session, out_path, SYNCLS_PARTS_PREFIX)
    print(f"[INFO] Baselines shard {SYNCLS_SHARD_INDEX} done: {len(result_rows)} rows.", flush=True)
    return f"baselines_shard_{SYNCLS_SHARD_INDEX}: {len(result_rows)} results"


# ---------------------------------------------------------------------------
# AutoGluon mode
# ---------------------------------------------------------------------------

def _create_tabular_predictor(label: str, problem_type: str, path: str, verbosity: int = 0):
    """
    Factory that returns a TabularPredictor instance.
    Extracted to a module-level function so tests can patch it without requiring
    autogluon to be installed.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "autogluon.tabular is required for AutoGluon evaluation mode. "
            "Install with: pip install autogluon.tabular"
        ) from exc
    return TabularPredictor(label=label, problem_type=problem_type, path=path, verbosity=verbosity)


def _run_autogluon_mode(session) -> str:
    """Run AutoGluon on this shard's datasets (single-node by default)."""
    import numpy as np
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        import pyarrow as pa
        import pyarrow.parquet as pq

    import pandas as pd

    datasets = _load_datasets_for_shard(session)
    if not datasets:
        print("[INFO] No datasets for this shard.", flush=True)
        return f"autogluon_shard_{SYNCLS_SHARD_INDEX}: 0 datasets"

    result_rows = []

    for ds in datasets:
        dataset_id = ds.get("dataset_id")
        stage_path = ds.get("stage_path")
        num_classes = int(ds.get("num_classes") or 2)
        problem_type = "binary" if num_classes == 2 else "multiclass"

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if SYNCLS_IS_MIXED_CAT:
                    data = _download_mixed_parquet_dataset(session, stage_path, tmp_dir)
                else:
                    data = _download_parquet_dataset(session, stage_path, tmp_dir)

                X_train = data["X_train"]
                y_train = data["y_train"]
                X_test  = data["X_test"]
                y_test  = data["y_test"]

                for feature_mode, X_train_used, X_test_used, selector in _feature_variants(
                    X_train, X_test, y_train
                ):
                    # Build DataFrame — mixed-cat: add ordinal cat columns with .astype("category")
                    if SYNCLS_IS_MIXED_CAT and "X_cat_train" in data:
                        import sys as _sys2
                        import os as _os2
                        _src2 = _os2.path.join(_os2.path.dirname(__file__), "..", "src")
                        if _src2 not in _sys2.path:
                            _sys2.path.insert(0, _src2)
                        from constants import ENTITY_EMBED_FIRST_REAL_ID
                        _X_cat_train_ord = np.asarray(data["X_cat_train"], dtype=np.int64) - ENTITY_EMBED_FIRST_REAL_ID
                        _X_cat_test_ord = np.asarray(data["X_cat_test"], dtype=np.int64) - ENTITY_EMBED_FIRST_REAL_ID
                        _X_cat_train_ord = np.clip(_X_cat_train_ord, 0, None)
                        _X_cat_test_ord = np.clip(_X_cat_test_ord, 0, None)
                        p_num = X_train_used.shape[1]
                        p_cat = _X_cat_train_ord.shape[1]
                        col_names = [f"f{i}" for i in range(p_num)] + [f"cat{j}" for j in range(p_cat)]
                        X_combined_train = np.hstack([X_train_used, _X_cat_train_ord.astype(np.float64)])
                        X_combined_test = np.hstack([X_test_used, _X_cat_test_ord.astype(np.float64)])
                        df_train = pd.DataFrame(X_combined_train, columns=col_names)
                        for j in range(p_cat):
                            df_train[f"cat{j}"] = df_train[f"cat{j}"].astype(int).astype("category")
                        df_train["target"] = y_train
                        df_test = pd.DataFrame(X_combined_test, columns=col_names)
                        for j in range(p_cat):
                            df_test[f"cat{j}"] = df_test[f"cat{j}"].astype(int).astype("category")
                    else:
                        _, p = X_train_used.shape
                        col_names = [f"f{i}" for i in range(p)]
                        df_train = pd.DataFrame(X_train_used, columns=col_names)
                        df_train["target"] = y_train
                        df_test = pd.DataFrame(X_test_used, columns=col_names)

                    ag_dir = os.path.join(tmp_dir, f"ag_{dataset_id}_{feature_mode}")
                    predictor = _create_tabular_predictor(
                        label="target",
                        problem_type=problem_type,
                        path=ag_dir,
                        verbosity=0,
                    )
                    predictor.fit(
                        df_train,
                        time_limit=SYNCLS_AG_TIME_LIMIT,
                        presets=SYNCLS_AG_PRESETS,
                        num_cpus=SYNCLS_AG_TASK_CPUS,
                    )

                    y_pred = np.asarray(predictor.predict(df_test))
                    try:
                        proba = predictor.predict_proba(df_test).values
                    except Exception:
                        proba = None

                    model_name = _variant_model_name("autogluon", feature_mode)
                    metrics = _compute_classification_metrics(
                        y_test, y_pred, proba, num_classes, model_name
                    )
                    metrics.update(_result_identity(
                        ds, dataset_id, prediction_source="autogluon_predictor"
                    ))
                    metrics["shard_index"] = SYNCLS_SHARD_INDEX
                    metrics["mode"] = "autogluon"
                    metrics["feature_mode"] = feature_mode
                    metrics["feature_selector"] = selector
                    metrics["selected_features"] = int(p)
                    metrics["ag_problem_type"] = problem_type
                    metrics["metric_schema_version"] = SYNCLS_METRIC_SCHEMA_VERSION
                    for field in ["prior_regime", "classification_regime", "suite_family",
                                   "feature_noise_level", "label_noise_rate", "num_classes",
                                   "sample_complexity_bucket", "margin_level"]:
                        if field in ds:
                            metrics[field] = ds[field]
                    result_rows.append(metrics)

        except Exception as exc:
            print(f"[ERROR] dataset_id={dataset_id}: {exc}", flush=True)
            result_rows.append({
                **_result_identity(
                    ds, dataset_id, prediction_source="autogluon_predictor"
                ),
                "shard_index": SYNCLS_SHARD_INDEX,
                "mode": "autogluon",
                "model_name": "autogluon",
                "result_status": "failed",
                "error": str(exc),
            })

    # Write shard parquet
    out_name = f"autogluon_shard_{SYNCLS_SHARD_INDEX}.parquet"
    out_path = os.path.join(SYNCLS_LOCAL_DIR, out_name)
    os.makedirs(SYNCLS_LOCAL_DIR, exist_ok=True)
    _write_results_parquet(result_rows, out_path)
    _upload_to_stage(session, out_path, SYNCLS_PARTS_PREFIX)
    print(f"[INFO] AutoGluon shard {SYNCLS_SHARD_INDEX} done: {len(result_rows)} rows.", flush=True)
    return f"autogluon_shard_{SYNCLS_SHARD_INDEX}: {len(result_rows)} results"


# ---------------------------------------------------------------------------
# Parquet I/O helpers
# ---------------------------------------------------------------------------

def _write_results_parquet(rows: list[dict], out_path: str) -> None:
    """Write a list of result dicts to a parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        import pyarrow as pa
        import pyarrow.parquet as pq
    import pandas as pd

    if not rows:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(rows)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), out_path)


def _read_parquet_df(path: str):
    """Read a parquet file into a pandas DataFrame."""
    import pandas as pd
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pandas()
    except Exception:
        return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Aggregate mode
# ---------------------------------------------------------------------------

_REQUIRED_OUTPUT_FILES = [
    "synthetic_classification_model_comparison.csv",
    "synthetic_classification_model_comparison_summary.csv",
    "synthetic_classification_summary_by_regime.csv",
    "synthetic_classification_summary_by_suite_family.csv",
    "synthetic_classification_summary_by_feature_noise.csv",
    "synthetic_classification_summary_by_label_noise.csv",
    "synthetic_classification_summary_by_training_size.csv",
    "synthetic_classification_summary_by_class_imbalance.csv",
    "synthetic_classification_summary_by_margin.csv",
    "synthetic_classification_summary_by_num_classes.csv",
    "synthetic_classification_summary_by_sample_complexity.csv",
    "synthetic_classification_summary_by_p_quartile_n_quartile.csv",
    "synthetic_classification_chart_data_model_rank.csv",
    "synthetic_classification_chart_data_noise_features.csv",
    "synthetic_classification_chart_data_label_noise.csv",
    "synthetic_classification_chart_data_training_size.csv",
    "synthetic_classification_chart_data_calibration.csv",
    "synthetic_classification_aggregation_manifest.json",
]

_REQUIRED_METRIC_COLS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
_OPTIONAL_METRIC_COLS = [
    "log_loss", "roc_auc_ovr", "roc_auc_ovo", "brier_score",
    "expected_calibration_error", "mcc", "cohen_kappa",
    "macro_precision", "macro_recall", "average_precision",
    "top_2_accuracy", "top_3_accuracy",
]


def _add_rank_metrics(df):
    """Add comparative/rank metrics to the comparison dataframe."""
    import numpy as np
    import pandas as pd

    task_col = "task_key" if "task_key" in df.columns else "dataset_id"
    if "log_loss" in df.columns:
        df["rank_by_log_loss"] = df.groupby(task_col)["log_loss"].rank(method="min")
    if "accuracy" in df.columns:
        df["rank_by_accuracy"] = df.groupby(task_col)["accuracy"].rank(
            method="min", ascending=False
        )
    if "macro_f1" in df.columns:
        df["rank_by_macro_f1"] = df.groupby(task_col)["macro_f1"].rank(
            method="min", ascending=False
        )

    # is_best / top-3
    if "rank_by_log_loss" in df.columns:
        df["is_best_log_loss"]  = df["rank_by_log_loss"]  == 1
        df["is_top3_log_loss"]  = df["rank_by_log_loss"]  <= 3
    if "rank_by_accuracy" in df.columns:
        df["is_best_accuracy"]  = df["rank_by_accuracy"]  == 1
        df["is_top3_accuracy"]  = df["rank_by_accuracy"]  <= 3

    # Beats-baseline comparisons
    ref_models = {
        "LogisticRegression":         "beats_logistic_regression",
        "RidgeClassifier":            "beats_ridge_classifier",
        "autogluon":                  "beats_autogluon",
    }
    tree_models = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"}

    for ref_model, col_name in ref_models.items():
        if "model_name" not in df.columns or "accuracy" not in df.columns:
            continue
        ref_acc = df[df["model_name"].astype(str).str.startswith(ref_model)].groupby(task_col)["accuracy"].mean()
        df["_ref_acc"] = df[task_col].map(ref_acc)
        # NaN reference → NaN beat metric (not False). Use .where(condition) to replace
        # entries where reference is unavailable with NaN.
        beats = (df["accuracy"] > df["_ref_acc"])
        df[col_name] = beats.where(df["_ref_acc"].notna())  # NaN where ref unavailable
        df.drop(columns=["_ref_acc"], inplace=True)

    # beats_best_tree
    if "model_name" in df.columns and "accuracy" in df.columns:
        tree_df = df[
            df["model_name"].astype(str).str.startswith(tuple(tree_models))
        ]
        best_tree_acc = tree_df.groupby(task_col)["accuracy"].max()
        df["_best_tree"] = df[task_col].map(best_tree_acc)
        # NaN reference → NaN beat metric (not False).
        beats_tree = (df["accuracy"] > df["_best_tree"])
        df["beats_best_tree"] = beats_tree.where(df["_best_tree"].notna())
        df.drop(columns=["_best_tree"], inplace=True)

    # Ratio metrics
    if "model_name" in df.columns and "log_loss" in df.columns:
        for ref_model, suffix in [
            ("LogisticRegression", "logistic_regression"),
            ("autogluon",          "autogluon"),
        ]:
            ref_ll = df[df["model_name"].astype(str).str.startswith(ref_model)].groupby(task_col)["log_loss"].mean()
            df[f"_ref_ll_{suffix}"] = df[task_col].map(ref_ll)
            df[f"ratio_log_loss_to_{suffix}"] = df["log_loss"] / df[f"_ref_ll_{suffix}"].replace(0, float("nan"))
            df.drop(columns=[f"_ref_ll_{suffix}"], inplace=True)

    if "model_name" in df.columns and "accuracy" in df.columns:
        ref_acc = df[df["model_name"].astype(str).str.startswith("LogisticRegression")].groupby(task_col)["accuracy"].mean()
        df["_ref_acc_lr"] = df[task_col].map(ref_acc)
        df["ratio_error_rate_to_logistic_regression"] = (
            (1 - df["accuracy"]) / (1 - df["_ref_acc_lr"]).replace(0, float("nan"))
        )
        df["accuracy_delta_vs_logistic_regression"] = df["accuracy"] - df["_ref_acc_lr"]
        df.drop(columns=["_ref_acc_lr"], inplace=True)

    if "model_name" in df.columns and "macro_f1" in df.columns:
        ref_f1 = df[df["model_name"].astype(str).str.startswith("LogisticRegression")].groupby(task_col)["macro_f1"].mean()
        df["_ref_f1"] = df[task_col].map(ref_f1)
        df["macro_f1_delta_vs_logistic_regression"] = df["macro_f1"] - df["_ref_f1"]
        df.drop(columns=["_ref_f1"], inplace=True)

    return df


def _safe_groupby_summary(df, group_col: str, metric_cols: list) -> object:
    """Compute mean metric summary grouped by group_col, returning empty df if column missing."""
    import pandas as pd
    if group_col not in df.columns:
        return pd.DataFrame()
    available_metrics = [c for c in metric_cols if c in df.columns]
    if not available_metrics:
        return pd.DataFrame()
    # Avoid duplicate groupby columns when group_col is 'model_name'
    if "model_name" in df.columns and group_col != "model_name":
        group_by = ["model_name", group_col]
    else:
        group_by = [group_col]
    return df.groupby(group_by)[available_metrics].mean().reset_index()


def _task_bootstrap_ci(
    df,
    metric_cols: list[str],
    *,
    replicates: int = 2000,
    seed: int = 20260607,
):
    import numpy as np
    import pandas as pd

    if "task_key" not in df.columns or "model_name" not in df.columns:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(seed)
    for model_name, model_df in df.groupby("model_name"):
        tasks = model_df["task_key"].dropna().unique()
        if len(tasks) == 0:
            continue
        for metric in metric_cols:
            if metric not in model_df.columns:
                continue
            task_values = model_df.groupby("task_key")[metric].mean().reindex(tasks)
            values = task_values.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            samples = rng.choice(values, size=(replicates, values.size), replace=True)
            means = samples.mean(axis=1)
            rows.append({
                "model_name": model_name,
                "metric": metric,
                "mean": float(values.mean()),
                "ci_lower": float(np.quantile(means, 0.025)),
                "ci_upper": float(np.quantile(means, 0.975)),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
            })
    return pd.DataFrame(rows)


def _run_aggregate_mode(session) -> str:
    """Gather all shard parquets and produce summary CSVs + manifest."""
    import numpy as np
    import pandas as pd

    expected_deepset_shards  = int(os.getenv("SYNCLS_EXPECTED_DEEPSET_SHARDS",  str(SYNCLS_NUM_SHARDS)))
    expected_baseline_shards = int(os.getenv("SYNCLS_EXPECTED_BASELINE_SHARDS", str(SYNCLS_NUM_SHARDS)))
    expected_ag_shards       = int(os.getenv("SYNCLS_EXPECTED_AG_SHARDS",       str(SYNCLS_CLUSTER_SHARDS)))

    print(
        f"[INFO] Aggregate mode: expected deepset={expected_deepset_shards} "
        f"baseline={expected_baseline_shards} ag={expected_ag_shards}",
        flush=True,
    )

    # Discover shard parquets
    all_stage_files = _list_stage_files(session, SYNCLS_PARTS_PREFIX)
    deepset_files  = sorted([f for f in all_stage_files if "deepset_shard_"  in f])
    baseline_files = sorted([f for f in all_stage_files if "baselines_shard_" in f])
    ag_files       = sorted([f for f in all_stage_files if "autogluon_shard_" in f])

    missing_shards = []
    if len(deepset_files)  < expected_deepset_shards:
        missing_shards.append(f"deepset: {len(deepset_files)}/{expected_deepset_shards}")
    if len(baseline_files) < expected_baseline_shards:
        missing_shards.append(f"baseline: {len(baseline_files)}/{expected_baseline_shards}")
    if expected_ag_shards > 0 and len(ag_files) < expected_ag_shards:
        missing_shards.append(f"autogluon: {len(ag_files)}/{expected_ag_shards}")

    if missing_shards:
        raise FileNotFoundError(
            f"Missing shards for aggregation: {', '.join(missing_shards)}. "
            f"Ensure all evaluation jobs completed before running aggregation."
        )

    # Download and combine
    local_agg_dir = os.path.join(SYNCLS_LOCAL_DIR, "aggregate")
    os.makedirs(local_agg_dir, exist_ok=True)

    all_dfs = []
    input_shard_list = []
    for stage_path in deepset_files + baseline_files + ag_files:
        try:
            local_path = _download_from_stage(session, stage_path, local_agg_dir)
            df_shard = _read_parquet_df(str(local_path))
            all_dfs.append(df_shard)
            input_shard_list.append(stage_path)
        except Exception as exc:
            print(f"[WARN] Could not load shard {stage_path}: {exc}", flush=True)

    if not all_dfs:
        raise FileNotFoundError("No shard data loaded for aggregation.")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"[INFO] Combined {len(df)} rows from {len(all_dfs)} shards.", flush=True)
    if "task_key" not in df.columns:
        family = df.get("suite_family", pd.Series("primary", index=df.index))
        global_idx = df.get("global_idx", pd.Series(np.nan, index=df.index))
        split_seed = df.get("split_seed", pd.Series(0, index=df.index))
        df["task_key"] = [
            (
                f"{SYNCLS_SUITE_ID}:{fam}:{int(gidx)}"
                if pd.notna(gidx)
                else f"{SYNCLS_SUITE_ID}:{fam}:{dataset_id}:{seed}"
            )
            for fam, gidx, dataset_id, seed in zip(
                family, global_idx, df.get("dataset_id"), split_seed
            )
        ]
    if "result_status" not in df.columns:
        df["result_status"] = np.where(
            df.get("error", pd.Series(index=df.index, dtype=object)).notna(),
            "failed",
            "success",
        )

    # Guard: no regression metrics
    for bad_col in ["r2", "mse", "rmse", "mae"]:
        if bad_col in df.columns:
            df.drop(columns=[bad_col], inplace=True)
            print(f"[WARN] Dropped unexpected regression metric column: {bad_col}", flush=True)

    # Add rank/comparative metrics
    df = _add_rank_metrics(df)

    # --- Output files ---
    output_files = []
    metric_cols = _REQUIRED_METRIC_COLS + [c for c in _OPTIONAL_METRIC_COLS if c in df.columns]

    def _write_csv(sub_df, name: str):
        path = os.path.join(local_agg_dir, name)
        sub_df.to_csv(path, index=False)
        _upload_to_stage(session, path, SYNCLS_RESULTS_STAGE)
        output_files.append(name)
        return path

    # Model comparison (all rows)
    _write_csv(df, "synthetic_classification_model_comparison.csv")

    successful = df[df["result_status"] == "success"].copy()
    development = successful[
        successful.get("suite_family", pd.Series("primary", index=successful.index))
        != "hidden_holdout"
    ]

    # Headline development summary excludes hidden holdout rows.
    available_metric_cols = [c for c in metric_cols if c in df.columns]
    if "model_name" in df.columns and available_metric_cols:
        summary_df = development.groupby("model_name")[available_metric_cols].mean().reset_index()
        _write_csv(summary_df, "synthetic_classification_model_comparison_summary.csv")
    else:
        _write_csv(pd.DataFrame(), "synthetic_classification_model_comparison_summary.csv")

    # Summaries by dimension
    dim_summaries = [
        ("classification_regime",    "synthetic_classification_summary_by_regime.csv"),
        ("suite_family",             "synthetic_classification_summary_by_suite_family.csv"),
        ("feature_noise_level",      "synthetic_classification_summary_by_feature_noise.csv"),
        ("label_noise_rate",         "synthetic_classification_summary_by_label_noise.csv"),
        ("n_train_default",          "synthetic_classification_summary_by_training_size.csv"),
        ("class_imbalance_type",     "synthetic_classification_summary_by_class_imbalance.csv"),
        ("margin_level",             "synthetic_classification_summary_by_margin.csv"),
        ("num_classes",              "synthetic_classification_summary_by_num_classes.csv"),
        ("sample_complexity_bucket", "synthetic_classification_summary_by_sample_complexity.csv"),
    ]
    if SYNCLS_IS_MIXED_CAT:
        dim_summaries.extend([
            ("p_num",            "synthetic_classification_summary_by_p_num.csv"),
            ("p_cat",            "synthetic_classification_summary_by_p_cat.csv"),
            ("max_cardinality",  "synthetic_classification_summary_by_max_cardinality.csv"),
            ("cat_effect_scale", "synthetic_classification_summary_by_cat_effect_scale.csv"),
            ("missing_rate",     "synthetic_classification_summary_by_missing_rate.csv"),
        ])
    for group_col, fname in dim_summaries:
        _write_csv(_safe_groupby_summary(successful, group_col, metric_cols), fname)

    for family, filename in [
        ("hidden_holdout", "synthetic_classification_hidden_summary.csv"),
        ("eval_only_unseen", "synthetic_classification_eval_only_summary.csv"),
    ]:
        _write_csv(
            _safe_groupby_summary(
                successful[successful.get("suite_family") == family],
                "model_name",
                metric_cols,
            ),
            filename,
        )
    if "suite_family" in successful.columns:
        distribution_group = np.where(
            successful["suite_family"] == "hidden_holdout",
            "hidden",
            np.where(
                successful["suite_family"].isin(
                    ["ood", "eval_only_unseen", "stress"]
                ),
                "ood",
                "id",
            ),
        )
        successful["distribution_group"] = distribution_group
        _write_csv(
            _safe_groupby_summary(
                successful, "distribution_group", metric_cols
            ),
            "synthetic_classification_id_ood_hidden_summary.csv",
        )
        degradation_source = (
            successful.groupby(["model_name", "distribution_group"])[
                [c for c in ["accuracy", "log_loss", "macro_f1"] if c in successful.columns]
            ]
            .mean()
            .reset_index()
        )
        degradation = degradation_source.pivot(
            index="model_name", columns="distribution_group"
        )
        degradation.columns = [
            f"{metric}_{group}" for metric, group in degradation.columns
        ]
        degradation = degradation.reset_index()
        for metric in ["accuracy", "macro_f1"]:
            if f"{metric}_id" in degradation and f"{metric}_ood" in degradation:
                degradation[f"{metric}_degradation_id_to_ood"] = (
                    degradation[f"{metric}_id"] - degradation[f"{metric}_ood"]
                )
        if "log_loss_id" in degradation and "log_loss_ood" in degradation:
            degradation["log_loss_degradation_id_to_ood"] = (
                degradation["log_loss_ood"] - degradation["log_loss_id"]
            )
        _write_csv(
            degradation,
            "synthetic_classification_distribution_degradation.csv",
        )

    # p/n quartile summary
    if "p_total" in df.columns and "n_total" in df.columns:
        df["p_quartile"] = pd.qcut(df["p_total"].fillna(0), q=4, labels=False, duplicates="drop")
        df["n_quartile"] = pd.qcut(df["n_total"].fillna(0), q=4, labels=False, duplicates="drop")
        _write_csv(
            df.groupby(
                ["model_name", "p_quartile", "n_quartile"], dropna=False
            )[[c for c in metric_cols if c in df.columns]].mean().reset_index(),
            "synthetic_classification_summary_by_p_quartile_n_quartile.csv",
        )
    else:
        _write_csv(pd.DataFrame(), "synthetic_classification_summary_by_p_quartile_n_quartile.csv")

    # Chart data files
    _write_csv(
        df[["model_name", "dataset_id"] + [c for c in ["rank_by_log_loss", "rank_by_accuracy"] if c in df.columns]],
        "synthetic_classification_chart_data_model_rank.csv",
    )
    _write_csv(
        _safe_groupby_summary(df, "feature_noise_level", metric_cols),
        "synthetic_classification_chart_data_noise_features.csv",
    )
    _write_csv(
        _safe_groupby_summary(df, "label_noise_rate", metric_cols),
        "synthetic_classification_chart_data_label_noise.csv",
    )
    _write_csv(
        _safe_groupby_summary(df, "n_train_default", metric_cols),
        "synthetic_classification_chart_data_training_size.csv",
    )
    if "expected_calibration_error" in df.columns:
        ece_df = df.groupby("model_name")[["expected_calibration_error"]].mean().reset_index()
        _write_csv(ece_df, "synthetic_classification_chart_data_calibration.csv")
    else:
        _write_csv(pd.DataFrame(), "synthetic_classification_chart_data_calibration.csv")

    completeness = (
        df.groupby("model_name", dropna=False)
        .agg(
            expected_task_count=("task_key", "nunique"),
            successful_task_count=(
                "task_key",
                lambda values: df.loc[values.index]
                .query("result_status == 'success'")["task_key"]
                .nunique(),
            ),
            failed_or_skipped_rows=(
                "result_status", lambda values: int((values != "success").sum())
            ),
        )
        .reset_index()
    )
    completeness["completion_rate"] = (
        completeness["successful_task_count"]
        / completeness["expected_task_count"].replace(0, np.nan)
    )
    _write_csv(completeness, "synthetic_classification_method_completeness_audit.csv")
    _write_csv(
        _task_bootstrap_ci(successful, available_metric_cols),
        "synthetic_classification_bootstrap_confidence_intervals.csv",
    )

    # F7: per-model ranking summary
    try:
        from evaluation_metrics import build_ranking_summary
        task_col = "task_key" if "task_key" in successful.columns else "dataset_id"
        if "model_name" in successful.columns and task_col in successful.columns:
            rank_frames = []
            for metric, higher in [
                ("log_loss",  False),
                ("accuracy",  True),
                ("macro_f1",  True),
            ]:
                if metric in successful.columns:
                    rdf = build_ranking_summary(
                        successful, metric_col=metric,
                        task_col=task_col, model_col="model_name",
                        higher_is_better=higher,
                    )
                    rdf.columns = [
                        c if c == "model_name" else f"{metric}_{c}"
                        for c in rdf.columns
                    ]
                    rank_frames.append(rdf)
            if rank_frames:
                import functools
                ranking_summary = functools.reduce(
                    lambda a, b: a.merge(b, on="model_name", how="outer"),
                    rank_frames,
                )
                _write_csv(
                    ranking_summary,
                    "synthetic_classification_ranking_summary.csv",
                )
                # Per-K breakdown
                if "num_classes" in successful.columns:
                    for k_val in sorted(successful["num_classes"].dropna().unique()):
                        sub = successful[successful["num_classes"] == k_val]
                        k_frames = []
                        for metric, higher in [("log_loss", False), ("accuracy", True)]:
                            if metric in sub.columns:
                                rdf = build_ranking_summary(
                                    sub, metric_col=metric,
                                    task_col=task_col, model_col="model_name",
                                    higher_is_better=higher,
                                )
                                rdf.columns = [
                                    c if c == "model_name" else f"{metric}_{c}"
                                    for c in rdf.columns
                                ]
                                k_frames.append(rdf)
                        if k_frames:
                            import functools as _ft
                            k_summary = _ft.reduce(
                                lambda a, b: a.merge(b, on="model_name", how="outer"),
                                k_frames,
                            )
                            _write_csv(
                                k_summary,
                                f"synthetic_classification_ranking_summary_k{int(k_val)}.csv",
                            )
                # Per-suite-family breakdown
                if "suite_family" in successful.columns:
                    for fam in sorted(successful["suite_family"].dropna().unique()):
                        sub = successful[successful["suite_family"] == fam]
                        fam_frames = []
                        for metric, higher in [("log_loss", False), ("accuracy", True)]:
                            if metric in sub.columns:
                                rdf = build_ranking_summary(
                                    sub, metric_col=metric,
                                    task_col=task_col, model_col="model_name",
                                    higher_is_better=higher,
                                )
                                rdf.columns = [
                                    c if c == "model_name" else f"{metric}_{c}"
                                    for c in rdf.columns
                                ]
                                fam_frames.append(rdf)
                        if fam_frames:
                            import functools as _ft2
                            fam_summary = _ft2.reduce(
                                lambda a, b: a.merge(b, on="model_name", how="outer"),
                                fam_frames,
                            )
                            safe_fam = fam.replace("/", "_").replace(" ", "_")
                            _write_csv(
                                fam_summary,
                                f"synthetic_classification_ranking_summary_{safe_fam}.csv",
                            )
    except Exception as _rank_exc:
        print(f"[WARN] build_ranking_summary failed: {_rank_exc}", flush=True)

    # Validation status
    required_metrics_ok = all(c in df.columns for c in _REQUIRED_METRIC_COLS)

    # Manifest
    manifest = {
        "suite_id": SYNCLS_SUITE_ID,
        "task_family": "nonlinear_classification",
        "task_objective": "inductive_classification",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "evaluation_run_id": SYNCLS_EVALUATION_RUN_ID,
        "suite_schema_version": "nonlinear_classification_eval_v1",
        "metric_schema_version": SYNCLS_METRIC_SCHEMA_VERSION,
        "input_shard_list": input_shard_list,
        "expected_deepset_shards": expected_deepset_shards,
        "expected_baseline_shards": expected_baseline_shards,
        "expected_ag_shards": expected_ag_shards,
        "actual_deepset_shards": len(deepset_files),
        "actual_baseline_shards": len(baseline_files),
        "actual_ag_shards": len(ag_files),
        "missing_shards": missing_shards,
        "output_file_list": output_files,
        "result_stage_root": SYNCLS_RESULTS_STAGE,
        "checkpoint_path": SYNCLS_CKPT_STAGE,
        "feature_selector": SYNCLS_FEATURE_SEL,
        "feature_selector_protocol": SYNCLS_EVAL_PROTOCOL,
        "baseline_feature_modes": list(SYNCLS_BASELINE_FEATURE_MODES),
        "method_completeness_audit": completeness.to_dict(orient="records"),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": SYNCLS_BOOTSTRAP_SEED,
        "expected_shard_topology": {
            "deepset": expected_deepset_shards,
            "baseline": expected_baseline_shards,
            "autogluon": expected_ag_shards,
        },
        "actual_shard_topology": {
            "deepset": len(deepset_files),
            "baseline": len(baseline_files),
            "autogluon": len(ag_files),
        },
        "runtime_variables": {
            "SYNTHETIC_CLASSIFICATION_SUITE_ID": SYNCLS_SUITE_ID,
            "SYNCLS_RESULTS_STAGE": SYNCLS_RESULTS_STAGE,
            "SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH": SYNCLS_CKPT_STAGE,
            "SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR": SYNCLS_FEATURE_SEL,
            "SYNCLS_CLUSTER_SHARDS": SYNCLS_CLUSTER_SHARDS,
        },
        "validation_status": "ok" if required_metrics_ok else "missing_required_metrics",
        "total_datasets": int(df["dataset_id"].nunique()) if "dataset_id" in df.columns else None,
        "total_rows": len(df),
    }

    manifest_path = os.path.join(local_agg_dir, "synthetic_classification_aggregation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    _upload_to_stage(session, manifest_path, SYNCLS_RESULTS_STAGE)
    output_files.append("synthetic_classification_aggregation_manifest.json")

    print(
        f"[INFO] Aggregation done. {len(output_files)} output files. "
        f"total_datasets={manifest['total_datasets']} total_rows={manifest['total_rows']}",
        flush=True,
    )
    return (
        f"aggregate: ok suite_id={SYNCLS_SUITE_ID} "
        f"total_datasets={manifest['total_datasets']} "
        f"output_files={len(output_files)}"
    )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main() -> None:
    session = _get_session()
    mode = SYNCLS_MODE.lower()
    print(f"[INFO] evaluate_nonlinear_classification mode={mode} suite_id={SYNCLS_SUITE_ID}", flush=True)

    if mode == "deepset":
        result = _run_deepset_mode(session)
    elif mode == "baselines":
        result = _run_baselines_mode(session)
    elif mode == "autogluon":
        result = _run_autogluon_mode(session)
    elif mode == "aggregate":
        result = _run_aggregate_mode(session)
    else:
        raise ValueError(
            f"Unknown SYNTHETIC_CLASSIFICATION_MODE={mode!r}. "
            "Expected: deepset | baselines | autogluon | aggregate"
        )
    print(f"[RESULT] {result}", flush=True)


if __name__ == "__main__":
    main()
