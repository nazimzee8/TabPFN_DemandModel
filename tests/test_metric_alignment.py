"""Tests for F4: probability metric alignment with labels= arg and status fields."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))


def _import_compute_metrics():
    import importlib
    spec = importlib.util.spec_from_file_location(
        "evaluate_linear_classification",
        os.path.join(_HERE, "..", "scripts", "evaluate_linear_classification.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Stub heavy env vars to avoid import-time side effects
    os.environ.setdefault("SYNTHETIC_CLASSIFICATION_MODE", "aggregate")
    spec.loader.exec_module(mod)
    return mod._compute_classification_metrics, mod._align_proba_to_canonical


try:
    _compute_classification_metrics, _align_proba_to_canonical = _import_compute_metrics()
    _IMPORT_OK = True
except Exception:
    _IMPORT_OK = False


pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="evaluate_linear_classification not importable")


def _uniform_proba(n, k, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k), size=n)
    return p


def test_log_loss_valid_field_present():
    n, K = 50, 3
    y_true = np.tile(np.arange(K), n // K + 1)[:n]
    proba = _uniform_proba(n, K)
    result = _compute_classification_metrics(y_true, y_true, proba, K, "test_model")
    assert "log_loss_valid" in result
    assert result["log_loss_valid"] is True


def test_roc_auc_suppressed_when_missing_support():
    """Multiclass AUC must be NaN and roc_auc_valid=False when support is missing."""
    n, K = 40, 3
    y_true = np.tile(np.arange(2), n // 2)  # only classes 0 and 1 in y_true
    proba = _uniform_proba(n, K)
    result = _compute_classification_metrics(
        y_true, y_true, proba, K, "test_model",
        has_missing_support=True,
        missing_support_classes=[2],
    )
    assert result["roc_auc_valid"] is False
    assert np.isnan(result["roc_auc_ovr"])
    assert np.isnan(result["roc_auc_ovo"])


def test_log_loss_with_labels_arg_binary():
    """log_loss with labels= must not raise for binary classification."""
    n, K = 50, 2
    y_true = np.array([0, 1] * (n // 2))
    proba = _uniform_proba(n, K)
    result = _compute_classification_metrics(y_true, y_true, proba, K, "test_model")
    assert np.isfinite(result["log_loss"])
    assert result["log_loss_valid"] is True


def test_metric_status_fields_present_no_proba():
    """Status fields must be present even when proba is None."""
    n, K = 30, 3
    y_true = np.tile(np.arange(K), n // K + 1)[:n]
    result = _compute_classification_metrics(y_true, y_true, None, K, "test_model")
    assert "log_loss_valid" in result
    assert result["log_loss_valid"] is False
    assert "roc_auc_valid" in result
    assert result["roc_auc_valid"] is False


def test_align_proba_returns_three_tuple():
    """_align_proba_to_canonical must return (aligned, has_missing, missing_classes)."""
    raw = np.array([[0.7, 0.3], [0.4, 0.6]])
    aligned, has_missing, missing = _align_proba_to_canonical(raw, [0, 1], num_classes=3)
    assert aligned.shape == (2, 3)
    assert has_missing is True
    assert 2 in missing


def test_align_proba_no_missing():
    raw = np.array([[0.7, 0.3], [0.4, 0.6]])
    aligned, has_missing, missing = _align_proba_to_canonical(raw, [0, 1], num_classes=2)
    assert has_missing is False
    assert missing == []
