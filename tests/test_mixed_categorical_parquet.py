"""Tests for mixed-categorical Parquet round-trip (Step 9 — 6 tests)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from constants import (
    ENTITY_EMBED_FIRST_REAL_ID,
    ENTITY_EMBED_MISSING_ID,
    MIXED_CAT_REGRESSION_TRAINING_FAMILY,
    MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
    MIXED_REG_DGP_SCHEMA_VERSION,
    MIXED_CLS_DGP_SCHEMA_VERSION,
)
from dgp_helpers import (
    build_mixed_regression_dataset,
    build_mixed_classification_dataset,
    mark_unseen_query_categories,
    write_parquet_mixed_regression_dgp,
    write_parquet_mixed_classification_dgp,
)


def _make_regression_ds(seed=42):
    """Build a split regression dataset dict ready for parquet writing."""
    params = {
        "n": 60, "p_num_signal": 3, "p_num_noise": 1,
        "p_cat_signal": 2, "p_cat_noise": 1,
        "cardinalities": [3, 5, 10],
        "cat_effect_scale": 1.0, "missing_rate": 0.05,
    }
    ds = build_mixed_regression_dataset(np.random.default_rng(seed), "A_iid_dense", params)
    n = ds["n"]
    n_train = int(0.8 * n)
    n_test = n - n_train
    unknown_mask = mark_unseen_query_categories(
        ds["X_cat"][:n_train], ds["X_cat"][n_train:]
    )
    ds_split = {
        "X_num_train": ds["X_num"][:n_train],
        "X_num_test":  ds["X_num"][n_train:],
        "y_train":     ds["y"][:n_train],
        "y_test":      ds["y"][n_train:],
        "X_cat_train": ds["X_cat"][:n_train],
        "X_cat_test":  ds["X_cat"][n_train:],
        "cat_missing_mask_train": ds["missing_mask"][:n_train],
        "cat_missing_mask_test":  ds["missing_mask"][n_train:],
        "cat_unknown_mask_test":  unknown_mask,
        "categorical_cardinalities": np.array(params["cardinalities"], dtype=np.int64),
        "n": n,
        "p_num": 4,
        "p_cat": 3,
        "n_train": n_train,
        "n_test": n_test,
        "prior_regime": "A_iid_dense",
        "schema_version": MIXED_REG_DGP_SCHEMA_VERSION,
        "task_family": "linear_regression",
        "training_data_family": MIXED_CAT_REGRESSION_TRAINING_FAMILY,
        "task_objective": "inductive_regression",
        "beta_num": ds["beta_num"],
        "numeric_support_mask": ds["numeric_support_mask"],
        "cat_support_mask": ds["cat_support_mask"],
        "cat_effects": ds["cat_effects"],
    }
    return ds_split


def _make_classification_ds(seed=42):
    """Build a split classification dataset dict ready for parquet writing."""
    params = {
        "n": 100, "p_num_signal": 3, "p_num_noise": 1,
        "p_cat_signal": 2, "p_cat_noise": 0,
        "cardinalities": [3, 5],
        "num_classes": 3,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(np.random.default_rng(seed), "A_iid_dense_logistic", params)
    n = ds["n"]
    n_train = int(0.8 * n)
    n_test = n - n_train
    unknown_mask = mark_unseen_query_categories(
        ds["X_cat"][:n_train], ds["X_cat"][n_train:]
    )
    ds_split = {
        "X_num_train": ds["X_num"][:n_train],
        "X_num_test":  ds["X_num"][n_train:],
        "y_train":     ds["y"][:n_train],
        "y_test":      ds["y"][n_train:],
        "X_cat_train": ds["X_cat"][:n_train],
        "X_cat_test":  ds["X_cat"][n_train:],
        "cat_missing_mask_train": np.zeros((n_train, 2), dtype=bool),
        "cat_missing_mask_test":  np.zeros((n_test, 2), dtype=bool),
        "cat_unknown_mask_test":  unknown_mask,
        "categorical_cardinalities": np.array(params["cardinalities"], dtype=np.int64),
        "n": n,
        "p_num": 4,
        "p_cat": 2,
        "n_train": n_train,
        "n_test": n_test,
        "num_classes": 3,
        "prior_regime": "A_iid_dense_logistic",
        "schema_version": MIXED_CLS_DGP_SCHEMA_VERSION,
        "task_family": "linear_classification",
        "training_data_family": MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
        "task_objective": "inductive_classification",
        "W_num": ds["W_num"],
        "b": ds["b"],
        "numeric_support_mask": ds["numeric_support_mask"],
        "cat_support_mask": ds["cat_support_mask"],
        "cat_class_effects": ds["cat_class_effects"],
    }
    return ds_split


# Test 9: Write and read mixed regression parquet round-trip
def test_write_and_read_mixed_regression_parquet():
    ds_split = _make_regression_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "task_0001.parquet")
        write_parquet_mixed_regression_dgp(ds_split, filepath, store_cat_params=True)
        assert Path(filepath).exists()
        table = pq.read_table(filepath)
        schema_names = set(table.schema.names)
        # Core fields present
        for field in ("X_num_train", "X_cat_train", "y_train", "X_cat_test",
                      "cat_missing_mask_train", "cat_unknown_mask_test",
                      "categorical_cardinalities", "n", "p_num", "p_cat",
                      "schema_version", "training_data_family"):
            assert field in schema_names, f"Missing field: {field}"
        # Param fields present when store_cat_params=True
        for field in ("beta_num", "cat_effects", "cat_support_mask"):
            assert field in schema_names, f"Missing param field: {field}"


# Test 10: Regression parquet schema version correct
def test_mixed_regression_parquet_schema_version():
    ds_split = _make_regression_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "task_0002.parquet")
        write_parquet_mixed_regression_dgp(ds_split, filepath)
        table = pq.read_table(filepath)
        d = table.to_pydict()
        assert d["schema_version"][0] == MIXED_REG_DGP_SCHEMA_VERSION


# Test 11: Classification parquet schema version correct
def test_mixed_classification_parquet_schema_version():
    ds_split = _make_classification_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "task_cls_0001.parquet")
        write_parquet_mixed_classification_dgp(ds_split, filepath)
        table = pq.read_table(filepath)
        d = table.to_pydict()
        assert d["schema_version"][0] == MIXED_CLS_DGP_SCHEMA_VERSION


# Test 12: Context-only split enforced (80/20)
def test_context_only_split_enforced():
    ds_split = _make_regression_ds()
    assert ds_split["n_train"] == int(0.8 * ds_split["n"])
    assert ds_split["n_test"] == ds_split["n"] - ds_split["n_train"]
    assert ds_split["X_num_train"].shape[0] == ds_split["n_train"]
    assert ds_split["X_num_test"].shape[0] == ds_split["n_test"]
    assert ds_split["X_cat_train"].shape[0] == ds_split["n_train"]
    assert ds_split["X_cat_test"].shape[0] == ds_split["n_test"]


# Test 13: Cat stats unaffected by y_test changes (parquet-level check)
def test_no_y_test_in_parquet_cat_stats():
    """Verify that changing y_test doesn't affect cat stat-related fields."""
    ds1 = _make_regression_ds(seed=99)
    ds2 = _make_regression_ds(seed=99)
    # Modify y_test in ds2
    ds2["y_test"] = np.zeros_like(ds2["y_test"]) + 999.0
    with tempfile.TemporaryDirectory() as tmpdir:
        fp1 = str(Path(tmpdir) / "task_a.parquet")
        fp2 = str(Path(tmpdir) / "task_b.parquet")
        write_parquet_mixed_regression_dgp(ds1, fp1)
        write_parquet_mixed_regression_dgp(ds2, fp2)
        t1 = pq.read_table(fp1).to_pydict()
        t2 = pq.read_table(fp2).to_pydict()
        # cat_effects should be identical (only depends on DGP, not y_test)
        assert t1["cat_effects"] == t2["cat_effects"]
        # X_cat arrays should be identical
        assert t1["X_cat_train"] == t2["X_cat_train"]
        assert t1["X_cat_test"] == t2["X_cat_test"]


# Test 14: Classification parquet has required classification fields
def test_classification_parquet_has_required_fields():
    ds_split = _make_classification_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "task_cls_0002.parquet")
        write_parquet_mixed_classification_dgp(ds_split, filepath, store_class_cat_params=True)
        table = pq.read_table(filepath)
        schema_names = set(table.schema.names)
        for field in ("num_classes", "W_num", "b", "cat_class_effects",
                      "cat_support_mask", "temperature", "label_noise_rate"):
            assert field in schema_names, f"Missing classification field: {field}"
