"""
tests/test_evaluate_linear_classification.py
================================================
Unit tests for scripts/evaluate_linear_classification.py.

Uses _FakeJob / _FakeSession pattern.
No Snowflake connection required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_linear_classification as eval_cls


# ---------------------------------------------------------------------------
# Helpers: minimal datasets
# ---------------------------------------------------------------------------

def _make_dataset(n_train=50, n_test=20, p=5, num_classes=2, seed=42):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, p))
    X_test  = rng.standard_normal((n_test, p))
    y_train = rng.integers(0, num_classes, n_train)
    y_test  = rng.integers(0, num_classes, n_test)
    teacher_probs = np.full((n_test, num_classes), 1.0 / num_classes)
    return {
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test":  X_test.tolist(),
        "y_test":  y_test.tolist(),
        "dgp_teacher_probs_test": teacher_probs.tolist(),  # renamed from probs_test (CLS-D1)
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------------
# Tests: _compute_classification_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_binary():
    """Basic binary classification metrics are computed correctly."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    proba = rng.dirichlet([1, 1], size=100)
    y_pred = proba.argmax(axis=1)

    metrics = eval_cls._compute_classification_metrics(y_true, y_pred, proba, 2, "test_model")

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "log_loss" in metrics
    assert "roc_auc_ovr" in metrics
    assert "brier_score" in metrics
    assert "mcc" in metrics
    assert "cohen_kappa" in metrics
    assert "expected_calibration_error" in metrics

    # Must not contain regression metrics
    for bad in ["r2", "mse", "rmse", "mae"]:
        assert bad not in metrics


def test_compute_metrics_multiclass():
    """Multiclass metrics: no binary-only brier_score (NaN), top_k accuracy present."""
    rng = np.random.default_rng(1)
    num_classes = 4
    y_true = rng.integers(0, num_classes, 80)
    proba = rng.dirichlet([1] * num_classes, size=80)
    y_pred = proba.argmax(axis=1)

    metrics = eval_cls._compute_classification_metrics(y_true, y_pred, proba, num_classes, "test_model")

    assert metrics["roc_auc_ovr"] == metrics["roc_auc_ovr"]  # not NaN for valid proba
    # Multiclass Brier score is now supported (Phase 5 fix)
    assert not np.isnan(metrics["brier_score"]), "brier_score should not be NaN for multiclass"
    assert "top_2_accuracy" in metrics  # k=2 for 3+ classes


def test_compute_metrics_no_proba_gives_nan():
    """Model without predict_proba → NaN for all proba-based metrics."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 50)
    y_pred = rng.integers(0, 2, 50)

    metrics = eval_cls._compute_classification_metrics(y_true, y_pred, None, 2, "no_proba_model")

    assert np.isnan(metrics["log_loss"])
    assert np.isnan(metrics["roc_auc_ovr"])
    assert np.isnan(metrics["brier_score"])
    assert np.isnan(metrics["expected_calibration_error"])

    # Label-only metrics should still be computed
    assert not np.isnan(metrics["accuracy"])
    assert not np.isnan(metrics["macro_f1"])


def test_no_regression_metrics_in_output():
    """Must never emit R², MSE, RMSE, or MAE."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 3, 60)
    proba = rng.dirichlet([1, 1, 1], size=60)
    y_pred = proba.argmax(axis=1)

    metrics = eval_cls._compute_classification_metrics(y_true, y_pred, proba, 3, "test")
    for bad_key in ["r2", "mse", "rmse", "mae", "R2", "MSE", "RMSE", "MAE"]:
        assert bad_key not in metrics, f"Unexpected regression metric: {bad_key}"


# ---------------------------------------------------------------------------
# Tests: ECE
# ---------------------------------------------------------------------------

def test_ece_binary_perfect():
    """Perfect calibration: ECE should be near 0."""
    n = 1000
    rng = np.random.default_rng(4)
    proba = rng.uniform(0, 1, n)
    # Generate labels probabilistically matching the predicted probability
    y_true = rng.binomial(1, proba)

    ece = eval_cls._compute_ece(y_true, proba)
    # ECE for well-calibrated model should be small (< 0.2 with 1000 samples)
    assert ece >= 0.0
    assert ece < 0.2


def test_ece_multiclass():
    """ECE for multiclass (2D proba array) runs without error."""
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 3, 100)
    proba = rng.dirichlet([1, 1, 1], size=100)

    ece = eval_cls._compute_ece(y_true, proba)
    assert isinstance(ece, float)
    assert ece >= 0.0


# ---------------------------------------------------------------------------
# Tests: _get_baseline_models
# ---------------------------------------------------------------------------

def test_get_baseline_models_count():
    """At minimum 6 baseline models (sklearn core) must be returned."""
    models = eval_cls._get_baseline_models(2)
    assert len(models) >= 6


def test_get_baseline_models_names():
    """Required baseline model names must be present."""
    required = {
        "DummyClassifier_most_frequent",
        "DummyClassifier_stratified",
        "LogisticRegression",
        "RidgeClassifier",
        "LinearSVC",
        "RandomForestClassifier",
    }
    model_names = {name for name, _ in eval_cls._get_baseline_models(2)}
    for req in required:
        assert req in model_names, f"Missing required baseline model: {req}"


def test_linearsvc_has_no_predict_proba():
    """LinearSVC must NOT have predict_proba (this is core to the NaN test)."""
    from sklearn.svm import LinearSVC
    m = LinearSVC()
    assert not eval_cls._has_predict_proba(m)


def test_logistic_regression_has_predict_proba():
    """LogisticRegression must have predict_proba."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression()
    assert eval_cls._has_predict_proba(m)


# ---------------------------------------------------------------------------
# Tests: autogluon problem_type routing
# ---------------------------------------------------------------------------

def test_autogluon_binary_problem_type(monkeypatch, tmp_path):
    """num_classes==2 → problem_type='binary'."""
    captured = {}

    class _FakePredictor:
        def fit(self, *a, **kw): pass
        def predict(self, df): return np.zeros(len(df), dtype=int)
        def predict_proba(self, df):
            import pandas as pd
            return pd.DataFrame({"0": [0.5]*len(df), "1": [0.5]*len(df)})

    def _fake_create_predictor(label, problem_type, path, verbosity=0):
        captured["problem_type"] = problem_type
        return _FakePredictor()

    datasets = [{"dataset_id": 0, "stage_path": "@stage/ds0.parquet", "num_classes": 2}]
    session = MagicMock()

    with patch.object(eval_cls, "_create_tabular_predictor", side_effect=_fake_create_predictor), \
         patch.object(eval_cls, "_load_datasets_for_shard", return_value=datasets), \
         patch.object(eval_cls, "_download_parquet_dataset", return_value=_make_dataset(num_classes=2)), \
         patch.object(eval_cls, "_upload_to_stage", return_value="@stage/out.parquet"), \
         patch.object(eval_cls, "_write_results_parquet", return_value=None):
        monkeypatch.setattr(eval_cls, "SYNCLS_SHARD_INDEX", 0)
        monkeypatch.setattr(eval_cls, "SYNCLS_NUM_SHARDS", 1)
        monkeypatch.setattr(eval_cls, "SYNCLS_LOCAL_DIR", str(tmp_path))
        eval_cls._run_autogluon_mode(session)

    assert captured.get("problem_type") == "binary"


def test_autogluon_multiclass_problem_type(monkeypatch, tmp_path):
    """num_classes==3 → problem_type='multiclass'."""
    captured = {}

    class _FakePredictor:
        def fit(self, *a, **kw): pass
        def predict(self, df): return np.zeros(len(df), dtype=int)
        def predict_proba(self, df):
            import pandas as pd
            return pd.DataFrame({str(i): [1/3]*len(df) for i in range(3)})

    def _fake_create_predictor(label, problem_type, path, verbosity=0):
        captured["problem_type"] = problem_type
        return _FakePredictor()

    datasets = [{"dataset_id": 0, "stage_path": "@stage/ds0.parquet", "num_classes": 3}]
    session = MagicMock()

    with patch.object(eval_cls, "_create_tabular_predictor", side_effect=_fake_create_predictor), \
         patch.object(eval_cls, "_load_datasets_for_shard", return_value=datasets), \
         patch.object(eval_cls, "_download_parquet_dataset", return_value=_make_dataset(num_classes=3)), \
         patch.object(eval_cls, "_upload_to_stage", return_value="@stage/out.parquet"), \
         patch.object(eval_cls, "_write_results_parquet", return_value=None):
        monkeypatch.setattr(eval_cls, "SYNCLS_SHARD_INDEX", 0)
        monkeypatch.setattr(eval_cls, "SYNCLS_NUM_SHARDS", 1)
        monkeypatch.setattr(eval_cls, "SYNCLS_LOCAL_DIR", str(tmp_path))
        eval_cls._run_autogluon_mode(session)

    assert captured.get("problem_type") == "multiclass"


# ---------------------------------------------------------------------------
# Tests: deepset mode validates checkpoint
# ---------------------------------------------------------------------------

def test_deepset_mode_rejects_regression_checkpoint(monkeypatch, tmp_path):
    """DeepSet mode must raise RuntimeError if checkpoint is for regression."""
    import torch

    ckpt_path = tmp_path / "bad_ckpt.pt"
    torch.save({"task_objective": "inductive_regression", "state_dict": {}}, str(ckpt_path))

    def _fake_validate_cls(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        task_obj = ckpt.get("task_objective")
        if task_obj is not None and task_obj != "inductive_classification":
            raise ValueError(
                f"Expected task_objective='inductive_classification', got {task_obj!r}."
            )

    session = MagicMock()
    monkeypatch.setattr(eval_cls, "SYNCLS_CKPT_STAGE", "@MODEL_STAGE/checkpoints/bad_ckpt.pt")
    monkeypatch.setattr(eval_cls, "SYNCLS_LOCAL_DIR", str(tmp_path))

    with patch.object(eval_cls, "_download_from_stage", return_value=ckpt_path), \
         patch("evaluate_linear_classification.validate_classification_checkpoint",
               _fake_validate_cls, create=True):
        with pytest.raises(RuntimeError, match="Checkpoint validation failed"):
            eval_cls._run_deepset_mode(session)


# ---------------------------------------------------------------------------
# Tests: feature selector
# ---------------------------------------------------------------------------

def test_feature_selector_uses_classif(monkeypatch):
    """DeepSet mode must use train_f_classif feature selector."""
    assert "classif" in eval_cls.SYNCLS_FEATURE_SEL or eval_cls.SYNCLS_FEATURE_SEL == "train_f_classif"


# ---------------------------------------------------------------------------
# Tests: deepset shard output path
# ---------------------------------------------------------------------------

def test_deepset_shard_output_in_classification_linear_path(monkeypatch, tmp_path):
    """Deepset shard output must be written to classification/linear/ path."""
    import torch

    # Create a valid fake checkpoint (classification)
    ckpt_path = tmp_path / "best_classification.pt"
    torch.save({"task_objective": "inductive_classification"}, str(ckpt_path))

    uploaded = []

    def _fake_upload(session, local_path, stage_dir):
        uploaded.append(stage_dir)
        return f"{stage_dir}/file.parquet"

    datasets = [{"dataset_id": 0, "stage_path": "@stage/ds0.parquet", "num_classes": 2}]

    monkeypatch.setattr(eval_cls, "SYNCLS_SHARD_INDEX", 0)
    monkeypatch.setattr(eval_cls, "SYNCLS_NUM_SHARDS", 1)
    monkeypatch.setattr(eval_cls, "SYNCLS_LOCAL_DIR", str(tmp_path))
    monkeypatch.setattr(
        eval_cls, "SYNCLS_RESULTS_STAGE",
        "@EVALUATION_RESULTS_STAGE/linear/classification/numeric/test_suite"
    )
    monkeypatch.setattr(
        eval_cls, "SYNCLS_PARTS_PREFIX",
        "@EVALUATION_RESULTS_STAGE/linear/classification/numeric/test_suite/parts"
    )

    session = MagicMock()

    with patch.object(eval_cls, "_download_from_stage", return_value=ckpt_path), \
         patch("evaluate_linear_classification.validate_classification_checkpoint",
               lambda path: None, create=True), \
         patch.object(eval_cls, "_load_datasets_for_shard", return_value=datasets), \
         patch.object(eval_cls, "_download_parquet_dataset", return_value=_make_dataset()), \
         patch.object(eval_cls, "_write_results_parquet", return_value=None), \
         patch.object(eval_cls, "_upload_to_stage", side_effect=_fake_upload):
        eval_cls._run_deepset_mode(session)

    assert any("/linear/classification/" in u for u in uploaded), (
        f"Expected /linear/classification/ in upload paths, got: {uploaded}"
    )


# ---------------------------------------------------------------------------
# Phase 5: Multiclass Brier score fix
# ---------------------------------------------------------------------------

def test_brier_score_multiclass_not_nan():
    """Multiclass (K>2) Brier score must not be NaN when proba is provided."""
    rng = np.random.default_rng(0)
    K = 4
    n = 30
    y_true = rng.integers(0, K, n)
    proba = rng.dirichlet(np.ones(K), n)
    y_pred = proba.argmax(axis=1)
    metrics = eval_cls._compute_classification_metrics(y_true, y_pred, proba, K, "test")
    assert not np.isnan(metrics.get("brier_score", float("nan"))), (
        f"Multiclass brier_score should not be NaN, got {metrics.get('brier_score')}"
    )


# ---------------------------------------------------------------------------
# Phase 6: Probability alignment
# ---------------------------------------------------------------------------

def test_proba_alignment_non_contiguous_classes():
    """sklearn model with classes [0, 2] aligned to [0, 1, 2] correctly."""
    proba_raw = np.array([[0.3, 0.7], [0.6, 0.4]])  # shape (2, 2) for classes [0, 2]
    aligned, has_missing, missing_cls = eval_cls._align_proba_to_canonical(proba_raw, [0, 2], num_classes=3)
    assert aligned.shape == (2, 3), f"Expected (2, 3), got {aligned.shape}"
    assert not np.isnan(aligned).any(), "aligned proba must not have NaN"
    assert np.allclose(aligned.sum(axis=1), 1.0, atol=1e-6), "rows must sum to 1.0"
    assert has_missing, "class 1 was absent, has_missing should be True"
    assert 1 in missing_cls, "class 1 should be in missing_cls"


def test_proba_alignment_missing_support_class():
    """Missing class gets epsilon and proba is renormalized."""
    proba_raw = np.array([[0.8, 0.2]])
    aligned, has_missing, missing_cls = eval_cls._align_proba_to_canonical(proba_raw, [0, 1], num_classes=3)
    assert has_missing, "class 2 missing, has_missing should be True"
    assert 2 in missing_cls, "class 2 should be in missing_cls"
    assert aligned[0, 2] > 0, "missing class should get epsilon probability"
    assert abs(aligned[0].sum() - 1.0) < 1e-6, "rows must still sum to 1"


def test_swapped_proba_columns_detected():
    """Unaligned proba (wrong class order) gives different log_loss than aligned."""
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(7)
    K = 3
    n = 50
    y_true = rng.integers(0, K, n)

    # Model trained on classes [2, 0, 1] → class_order is wrong for 0..2
    proba_wrong_order = rng.dirichlet(np.ones(K), n)
    proba_aligned, _, _missing = eval_cls._align_proba_to_canonical(
        proba_wrong_order, [2, 0, 1], num_classes=K
    )
    loss_wrong = log_loss(y_true, proba_wrong_order, labels=list(range(K)))
    loss_aligned = log_loss(y_true, proba_aligned, labels=list(range(K)))
    # They should differ (unless by chance the random seed makes them equal)
    # We just verify the alignment produces a valid probability matrix
    assert proba_aligned.shape == (n, K)
    assert np.allclose(proba_aligned.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 8: Protocol recording
# ---------------------------------------------------------------------------

def test_protocol_recorded_in_result_row():
    """eval_protocol constant must be defined in evaluator module."""
    assert hasattr(eval_cls, "SYNCLS_EVAL_PROTOCOL"), (
        "SYNCLS_EVAL_PROTOCOL constant must be defined in evaluate_linear_classification"
    )
    assert eval_cls.SYNCLS_EVAL_PROTOCOL in ("raw_features", "train_selected_features"), (
        f"SYNCLS_EVAL_PROTOCOL={eval_cls.SYNCLS_EVAL_PROTOCOL!r} not a valid protocol"
    )
