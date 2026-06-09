"""Tests for F8: query-label remapping uses only y_train for the canonical label map."""

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
import torch

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


def test_unique_labels_derived_from_y_train_only():
    """Canonical label map must not incorporate y_test-only classes.

    When y_train has contiguous classes [0, 1] but num_classes=3 (from parquet),
    class 2 appears only in y_test and must be recorded as an unseen query class.
    num_classes is preserved from the parquet specification (3).
    """
    rng = np.random.default_rng(0)
    n_train, n_test, p = 40, 10, 4
    X_train = rng.standard_normal((n_train, p)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  p)).astype(np.float32)
    # Training sees classes 0 and 1 only; query introduces class 2 (unseen).
    y_train = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    y_test  = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(tmp, X_train, y_train, X_test, y_test, num_classes=3)
        result = load_classification_parquet(path, min_support_per_class=0)

    # num_classes is preserved from the parquet spec (not reduced to y_train unique count)
    assert result["num_classes"] == 3, (
        f"num_classes should be preserved from parquet (3), got {result['num_classes']}"
    )
    # Unseen query class 2 must be recorded
    assert 2 in result["unseen_query_classes"], (
        f"Class 2 should appear in unseen_query_classes: {result['unseen_query_classes']}"
    )


def test_no_query_leakage_in_contiguous_case():
    """When labels are already contiguous, ensure y_test OOD labels are recorded."""
    rng = np.random.default_rng(1)
    n_train, n_test, p = 30, 10, 3
    X_train = rng.standard_normal((n_train, p)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  p)).astype(np.float32)
    y_train = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    y_test  = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(tmp, X_train, y_train, X_test, y_test, num_classes=2)
        result = load_classification_parquet(path)

    assert result["num_classes"] == 2
    assert result["unseen_query_classes"] == []


def test_remapping_preserves_y_train_classes():
    """After remapping, y_train values must be in [0, num_classes)."""
    rng = np.random.default_rng(2)
    n_train, n_test, p = 30, 10, 3
    X_train = rng.standard_normal((n_train, p)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  p)).astype(np.float32)
    # Sparse labels — only classes 0 and 3 in train.
    y_train = np.array([0] * 15 + [3] * 15, dtype=np.int64)
    y_test  = np.array([0, 3, 0, 3, 0, 3, 0, 3, 0, 3], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_parquet(tmp, X_train, y_train, X_test, y_test, num_classes=4)
        result = load_classification_parquet(path, min_support_per_class=0)

    yt = result["y_train"]
    assert int(yt.min()) >= 0
    assert int(yt.max()) < result["num_classes"]
