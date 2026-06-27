"""Shared AutoGluon tabular regression helpers."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid

import pandas as pd

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_IMPORT_ERROR = None
except ImportError as exc:
    TabularPredictor = None
    AUTOGLUON_IMPORT_ERROR = exc


def autogluon_import_error_message():
    return (
        "AutoGluon is not available in AUTOGLUON_RUNTIME_ENVIRONMENT. "
        "Use autogluon.tabular==1.3.0 or the validated compatible "
        "AutoGluon package set for this runtime image."
    )


def get_tabular_predictor_class():
    if TabularPredictor is None:
        tabular_module = sys.modules.get("autogluon.tabular")
        patched_predictor = getattr(tabular_module, "TabularPredictor", None)
        if patched_predictor is not None:
            return patched_predictor
        raise RuntimeError(autogluon_import_error_message()) from AUTOGLUON_IMPORT_ERROR
    return TabularPredictor


def make_autogluon_dataframes(X_train_np, y_train_np, X_test_np, label="target"):
    feature_cols = [f"f{i}" for i in range(X_train_np.shape[1])]
    train_df = pd.DataFrame(X_train_np, columns=feature_cols)
    train_df[label] = y_train_np
    test_df = pd.DataFrame(X_test_np, columns=feature_cols)
    return train_df, test_df


def predict_autogluon(
    X_train_np,
    y_train_np,
    X_test_np,
    *,
    time_limit=300,
    presets="best_quality",
    num_cpus=1,
    num_gpus=0,
    verbosity=0,
    model_dir=None,
    cleanup=True,
    predictor_cls=None,
):
    """Fit an AutoGluon stacked ensemble and return predictions as a numpy array."""
    preds, _, _ = predict_autogluon_timed(
        X_train_np,
        y_train_np,
        X_test_np,
        time_limit=time_limit,
        presets=presets,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        verbosity=verbosity,
        model_dir=model_dir,
        cleanup=cleanup,
        predictor_cls=predictor_cls,
    )
    return preds


def predict_autogluon_timed(
    X_train_np,
    y_train_np,
    X_test_np,
    *,
    time_limit=300,
    presets="best_quality",
    num_cpus=1,
    num_gpus=0,
    verbosity=0,
    model_dir=None,
    cleanup=True,
    predictor_cls=None,
):
    """Fit an AutoGluon regressor and return (predictions, fit_time_s, predict_time_s)."""
    TabularPredictor = predictor_cls or get_tabular_predictor_class()
    train_df, test_df = make_autogluon_dataframes(X_train_np, y_train_np, X_test_np)
    if model_dir is None:
        model_dir = tempfile.mkdtemp(prefix="autogluon_", dir="/tmp")
    else:
        os.makedirs(model_dir, exist_ok=True)

    try:
        predictor = TabularPredictor(
            label="target",
            path=model_dir,
            problem_type="regression",
            verbosity=verbosity,
        )
        fit_start = time.perf_counter()
        predictor.fit(
            train_df,
            presets=presets,
            time_limit=time_limit,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            verbosity=verbosity,
        )
        fit_time_s = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        preds = predictor.predict(test_df)
        predict_time_s = time.perf_counter() - predict_start
        if hasattr(preds, "to_numpy"):
            pred_values = preds.to_numpy()
        else:
            pred_values = preds.values
        return pred_values, fit_time_s, predict_time_s
    finally:
        if cleanup:
            shutil.rmtree(model_dir, ignore_errors=True)


def predict_autogluon_classification_timed(
    X_train_np,
    y_train_np,
    X_test_np,
    *,
    time_limit=300,
    presets="best_quality",
    num_cpus=1,
    num_gpus=0,
    verbosity=0,
    model_dir=None,
    cleanup=True,
    predictor_cls=None,
):
    """Fit an AutoGluon classifier and return (class probabilities, fit time, predict time)."""
    TabularPredictor = predictor_cls or get_tabular_predictor_class()
    train_df, test_df = make_autogluon_dataframes(X_train_np, y_train_np, X_test_np)
    if model_dir is None:
        model_dir = tempfile.mkdtemp(prefix="autogluon_classification_", dir="/tmp")
    else:
        os.makedirs(model_dir, exist_ok=True)

    try:
        problem_type = "binary" if len(pd.unique(y_train_np)) == 2 else "multiclass"
        predictor = TabularPredictor(
            label="target",
            path=model_dir,
            problem_type=problem_type,
            eval_metric="log_loss",
            verbosity=verbosity,
        )
        fit_start = time.perf_counter()
        predictor.fit(
            train_df,
            presets=presets,
            time_limit=time_limit,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            verbosity=verbosity,
        )
        fit_time_s = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        probabilities = predictor.predict_proba(test_df)
        predict_time_s = time.perf_counter() - predict_start
        if hasattr(probabilities, "to_numpy"):
            probability_values = probabilities.to_numpy()
        else:
            probability_values = probabilities.values
        return probability_values, fit_time_s, predict_time_s
    finally:
        if cleanup:
            shutil.rmtree(model_dir, ignore_errors=True)


def make_unique_autogluon_model_dir(item_meta=None, *, base_dir=None, prefix="ag_ray"):
    """Create a unique temporary model directory with human-readable context."""
    item_meta = item_meta or {}
    base_dir = base_dir or tempfile.gettempdir()
    parts = [
        prefix,
        f"shard{item_meta.get('shard_index', 'na')}",
        str(item_meta.get("prior_regime", "regime_na")),
        f"d{item_meta.get('dataset_id', 'na')}",
        f"s{item_meta.get('split_seed', 'na')}",
        f"pid{os.getpid()}",
        uuid.uuid4().hex[:12],
    ]
    safe_prefix = "_".join(
        "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in part)
        for part in parts
    )
    return tempfile.mkdtemp(prefix=f"{safe_prefix}_", dir=base_dir)


def tmp_free_bytes(tmp_dir="/tmp"):
    check_dir = tmp_dir if os.path.exists(tmp_dir) else tempfile.gettempdir()
    return int(shutil.disk_usage(check_dir).free)


def tmp_free_space_skip_reason(
    tmp_dir="/tmp",
    *,
    min_free_bytes=5368709120,
    env_name="BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES",
):
    check_dir = tmp_dir if os.path.exists(tmp_dir) else tempfile.gettempdir()
    free_bytes = int(shutil.disk_usage(check_dir).free)
    if free_bytes < int(min_free_bytes):
        return (
            f"{check_dir} free_bytes={free_bytes} below "
            f"{env_name}={int(min_free_bytes)}"
        )
    return None
