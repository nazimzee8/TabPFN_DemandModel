"""Tests for F3: minimum support count policy in load_classification_parquet."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import pyarrow as pa
import pyarrow.parquet as pq

from train import load_classification_parquet


def _write_parquet(tmp_path, X_train, y_train, X_test, y_test, num_classes):
    path = os.path.join(tmp_path, "task.parquet")
    table = pa.table({
        "X_train": [X_train.tolist()],
        "y_train": [y_train.tolist()],
        "X_test":  [X_test.tolist()],
        "y_test":  [y_test.tolist()],
        "num_classes": [num_classes],
    })
    pq.write_table(table, path)
    return path


def test_missing_support_raises_by_default():
    """Classes with 0 training samples must raise when min_support_per_class=1.

    Scenario: num_classes=3 from parquet, but y_train only has classes 0 and 1.
    Class 2 has 0 training examples → support check should raise.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, p = 30, 10, 3
    X_train = rng.standard_normal((n_train, p)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  p)).astype(np.float32)
    # y_train has classes 0 and 1 only; class 2 is absent.
    y_train = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    y_test  = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(tmp, X_train, y_train, X_test, y_test, num_classes=3)
        with pytest.raises(ValueError, match="have 0 support"):
            load_classification_parquet(path)


def test_missing_support_allowed_with_zero_threshold():
    """min_support_per_class=0 must suppress the error (stress evaluation regime)."""
    rng = np.random.default_rng(1)
    n_train, n_test, p = 30, 10, 3
    X_train = rng.standard_normal((n_train, p)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  p)).astype(np.float32)
    # y_train has classes 0 and 1 only; class 2 is absent.
    y_train = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    y_test  = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(tmp, X_train, y_train, X_test, y_test, num_classes=3)
        result = load_classification_parquet(path, min_support_per_class=0)

    assert "missing_support_classes" in result
    assert 2 in result["missing_support_classes"]


def test_support_counts_returned():
    """support_class_counts must be present in the returned dict."""
    rng = np.random.default_rng(2)
    n_train, n_test, p = 30, 10, 3
    X_train = rng.standard_normal((n_train, p)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  p)).astype(np.float32)
    y_train = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    y_test  = np.array([0, 1] * 5, dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(tmp, X_train, y_train, X_test, y_test, num_classes=2)
        result = load_classification_parquet(path)

    sc = result["support_class_counts"]
    assert sc[0] == 15
    assert sc[1] == 15
    assert result["missing_support_classes"] == []


def test_error_message_includes_global_idx():
    """Error message must include the supplied global_idx for debugging."""
    rng = np.random.default_rng(3)
    n_train, n_test, p = 30, 10, 3
    X_train = rng.standard_normal((n_train, p)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  p)).astype(np.float32)
    # num_classes=3 but y_train only has classes 0 and 1.
    y_train = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    y_test  = np.zeros(n_test, dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(tmp, X_train, y_train, X_test, y_test, num_classes=3)
        with pytest.raises(ValueError, match="global_idx=999"):
            load_classification_parquet(path, global_idx=999)
