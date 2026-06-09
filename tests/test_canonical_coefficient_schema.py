"""
tests/test_canonical_coefficient_schema.py
==========================================
Tests for canonical W_true/b_true coefficient schema (Issue 1) and
feature_noise_level float64 storage contract (Issue 2).

These tests verify:
1. Binary (K=2) datasets emit W_true as (p, K) ndarray
2. Multiclass (K>2) datasets emit W_true as (p, K) ndarray
3. w_true deprecated alias equals W_true[:, 1] for all K
4. feature_noise_level is float64 in the training parquet writer
5. feature_noise_level is float64 in the eval parquet writer
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dgp_helpers import (
    generate_classification_dataset,
    split_classification_dataset,
    write_classification_parquet,
    write_parquet_classification_eval,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal dataset generation
# ---------------------------------------------------------------------------

def _make_binary_ds(seed: int = 0):
    """Generate a minimal K=2 classification dataset."""
    rng = np.random.default_rng(seed)
    assignment = {
        "classification_regime": "A_iid_dense_logistic",
        "n": 60,
        "n_train": 40,
        "n_test": 20,
        "p_signal": 4,
        "p_noise": 2,
        "num_classes": 2,
        "class_imbalance_type": "balanced",
        "label_noise_rate": 0.0,
        "feature_noise_level": 0.05,
        "margin_level": "medium",
        "temperature": 1.0,
    }
    grids = {
        "n_grid": [60],
        "p_signal_grid": [4],
        "p_noise_grid": [2],
        "active_s_grid": [2],
        "rho_grid": [0.0],
        "feature_noise_grid": [0.05],
        "num_classes_grid": [2],
        "temperature_grid": [1.0],
        "label_noise_grid": [0.0],
        "class_imbalance_grid": ["balanced"],
        "margin_grid": ["medium"],
        "coefficient_scale_grid": [1.0],
        "intercept_scale_grid": [0.0],
    }
    ds = generate_classification_dataset(
        rng, "A_iid_dense_logistic", assignment, grids, task_seed=seed
    )
    return ds


def _make_multiclass_ds(seed: int = 1, num_classes: int = 3):
    """Generate a minimal K>2 classification dataset."""
    rng = np.random.default_rng(seed)
    assignment = {
        "classification_regime": "A_iid_dense_logistic",
        "n": 90,
        "n_train": 60,
        "n_test": 30,
        "p_signal": 4,
        "p_noise": 2,
        "num_classes": num_classes,
        "class_imbalance_type": "balanced",
        "label_noise_rate": 0.0,
        "feature_noise_level": 0.10,
        "margin_level": "medium",
        "temperature": 1.0,
    }
    grids = {
        "n_grid": [90],
        "p_signal_grid": [4],
        "p_noise_grid": [2],
        "active_s_grid": [2],
        "rho_grid": [0.0],
        "feature_noise_grid": [0.10],
        "num_classes_grid": [num_classes],
        "temperature_grid": [1.0],
        "label_noise_grid": [0.0],
        "class_imbalance_grid": ["balanced"],
        "margin_grid": ["medium"],
        "coefficient_scale_grid": [1.0],
        "intercept_scale_grid": [0.0],
    }
    ds = generate_classification_dataset(
        rng, "A_iid_dense_logistic", assignment, grids, task_seed=seed
    )
    return ds


# ---------------------------------------------------------------------------
# Test 1: Binary (K=2) W_true always present as (p, K)
# ---------------------------------------------------------------------------

def test_binary_w_true_canonical_shape():
    """K=2: W_true must be (p, K) ndarray, not None."""
    ds = _make_binary_ds()
    assert ds is not None, "generate_classification_dataset returned None for K=2"
    assert "W_true" in ds, "W_true key missing from K=2 dataset dict"
    W = ds["W_true"]
    assert W is not None, "W_true is None for K=2; must be (p, K) ndarray"
    p = ds["p_total"]
    k = ds["num_classes"]
    assert W.shape == (p, k), (
        f"W_true.shape={W.shape}, expected ({p}, {k}) for K=2"
    )


# ---------------------------------------------------------------------------
# Test 2: Multiclass (K>2) W_true always present as (p, K)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_classes", [3, 4, 5])
def test_multiclass_w_true_canonical_shape(num_classes):
    """K>2: W_true must be (p, K) ndarray."""
    ds = _make_multiclass_ds(num_classes=num_classes)
    assert ds is not None, f"generate_classification_dataset returned None for K={num_classes}"
    assert "W_true" in ds, f"W_true key missing from K={num_classes} dataset dict"
    W = ds["W_true"]
    assert W is not None, f"W_true is None for K={num_classes}"
    p = ds["p_total"]
    assert W.shape == (p, num_classes), (
        f"W_true.shape={W.shape}, expected ({p}, {num_classes})"
    )


# ---------------------------------------------------------------------------
# Test 3: w_true deprecated alias equals W_true[:, 1] for all K
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_fn,label", [
    (_make_binary_ds, "K=2"),
    (lambda: _make_multiclass_ds(num_classes=3), "K=3"),
])
def test_w_true_deprecated_alias_equals_w_true_col1(make_fn, label):
    """w_true (deprecated) must equal W_true[:, 1] for all K."""
    ds = make_fn()
    assert ds is not None, f"dataset generation failed for {label}"
    assert "w_true" in ds, f"w_true alias missing for {label}"
    assert "W_true" in ds, f"W_true missing for {label}"
    np.testing.assert_array_equal(
        ds["w_true"], ds["W_true"][:, 1],
        err_msg=f"w_true != W_true[:, 1] for {label}"
    )


# ---------------------------------------------------------------------------
# Test 4: feature_noise_level is float64 in training parquet
# ---------------------------------------------------------------------------

def test_feature_noise_level_float64_in_training_parquet():
    """write_classification_parquet must store feature_noise_level as float64."""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        pytest.skip("pyarrow not installed")

    ds_raw = _make_binary_ds()
    assert ds_raw is not None
    ds = split_classification_dataset(ds_raw)

    with tempfile.TemporaryDirectory() as tmp:
        fpath = str(Path(tmp) / "train_test.parquet")
        teacher = {
            "teacher_available": False,
            "teacher_type": "none",
            "teacher_failure_reason": "test",
            "teacher_regularization": 1.0,
        }
        write_classification_parquet(ds, fpath, teacher_results=teacher)
        tbl = pq.read_table(fpath)
        field = tbl.schema.field("feature_noise_level")
        assert field.type == pa.float64(), (
            f"feature_noise_level in training parquet must be float64, got {field.type}"
        )


# ---------------------------------------------------------------------------
# Test 5: feature_noise_level is float64 in eval parquet
# ---------------------------------------------------------------------------

def test_feature_noise_level_float64_in_eval_parquet():
    """write_parquet_classification_eval must store feature_noise_level as float64."""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        pytest.skip("pyarrow not installed")

    ds_raw = _make_binary_ds()
    assert ds_raw is not None
    ds_split = split_classification_dataset(ds_raw)

    with tempfile.TemporaryDirectory() as tmp:
        fpath = str(Path(tmp) / "eval_test.parquet")
        write_parquet_classification_eval(
            ds_split,
            fpath,
            suite_id="test_suite",
            suite_family="primary",
            dataset_idx=0,
            global_idx=0,
            profile="linear_classification_eval",
            regime="A_iid_dense_logistic",
            task_seed=12345,
        )
        tbl = pq.read_table(fpath)
        field = tbl.schema.field("feature_noise_level")
        assert field.type == pa.float64(), (
            f"feature_noise_level in eval parquet must be float64, got {field.type}"
        )
        # Also verify task_seed column is present and correct
        assert "task_seed" in tbl.schema.names, "task_seed column missing from eval parquet"
        task_seed_col = tbl["task_seed"].to_pylist()
        assert all(v == 12345 for v in task_seed_col), (
            f"task_seed values should all be 12345, got {task_seed_col[:5]}"
        )
