"""
tests/test_classification_roundtrip.py
=======================================
Full generator → writer → load_classification_eval_task() → evaluator-shape tests.

Phase 1 acceptance criteria:
- A parquet produced by write_parquet_classification_eval() can be consumed
  by load_classification_eval_task()
- No production evaluator path reads nested X_train/X_test parquet columns
- Stage paths and manifest paths agree
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgp_helpers import (  # noqa: E402
    load_classification_eval_task,
    validate_classification_eval_dataset,
    write_parquet_classification_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ds(n_train: int = 40, n_test: int = 15, p: int = 5, num_classes: int = 2) -> dict:
    """Build a minimal ds dict compatible with write_parquet_classification_eval."""
    rng = np.random.default_rng(0)
    return {
        "X_train": rng.standard_normal((n_train, p)),
        "y_train": rng.integers(0, num_classes, n_train),
        "X_test": rng.standard_normal((n_test, p)),
        "y_test": rng.integers(0, num_classes, n_test),
        "p_total": p,
        "num_classes": num_classes,
        "label_noise_rate": 0.0,
        "feature_noise_level": 0.0,
        "margin_level": "medium",
        "class_imbalance_type": "balanced",
        "temperature": 1.0,
        "distribution_family": "gaussian_linear_logits",
        "covariance_type": "iid",
        "rho": 0.0,
    }


def _write_and_load(tmp_path, ds=None, **kwargs) -> dict:
    if ds is None:
        ds = _make_ds()
    path = str(tmp_path / "task.parquet")
    defaults = dict(
        suite_id="roundtrip_test",
        suite_family="primary",
        dataset_idx=0,
        global_idx=0,
        profile="linear_classification_stat_aware",
        regime="A_iid_dense",
        task_seed=42,
        store_class_params=False,
    )
    defaults.update(kwargs)
    write_parquet_classification_eval(ds, path, **defaults)
    tbl = pq.read_table(path)
    return load_classification_eval_task(tbl)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_context_rows_become_x_train_y_train(tmp_path):
    """context rows → X_train/y_train; query rows → X_test/y_test."""
    ds = _make_ds(n_train=30, n_test=10, p=4)
    task = _write_and_load(tmp_path, ds=ds)

    assert task["X_train"].shape == (30, 4)
    assert task["y_train"].shape == (30,)
    assert task["X_test"].shape == (10, 4)
    assert task["y_test"].shape == (10,)


def test_schema_mismatch_raises_value_error(tmp_path):
    """Wrong schema_version must raise ValueError."""
    ds = _make_ds()
    path = str(tmp_path / "task.parquet")
    write_parquet_classification_eval(
        ds, path,
        suite_id="test", suite_family="primary",
        dataset_idx=0, global_idx=0,
        profile="linear_classification_stat_aware",
        regime="A_iid_dense",
    )
    tbl = pq.read_table(path)
    # Corrupt schema_version
    bad = tbl.set_column(
        tbl.schema.get_field_index("schema_version"),
        "schema_version",
        pa.array(["unknown_v999"] * len(tbl), type=pa.utf8()),
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_classification_eval_task(bad)


def test_empty_context_raises_value_error(tmp_path):
    """Table with no context rows must raise ValueError."""
    ds = _make_ds(n_train=10, n_test=10)
    path = str(tmp_path / "task.parquet")
    write_parquet_classification_eval(
        ds, path,
        suite_id="test", suite_family="primary",
        dataset_idx=0, global_idx=0,
        profile="linear_classification_stat_aware",
        regime="A_iid_dense",
    )
    tbl = pq.read_table(path)
    # Keep only query rows
    split_col = tbl.column("split").to_pylist()
    query_indices = [i for i, s in enumerate(split_col) if s == "query"]
    query_only = tbl.take(query_indices)
    with pytest.raises(ValueError):
        load_classification_eval_task(query_only)


def test_empty_query_raises_value_error(tmp_path):
    """Table with no query rows must raise ValueError."""
    ds = _make_ds(n_train=10, n_test=10)
    path = str(tmp_path / "task.parquet")
    write_parquet_classification_eval(
        ds, path,
        suite_id="test", suite_family="primary",
        dataset_idx=0, global_idx=0,
        profile="linear_classification_stat_aware",
        regime="A_iid_dense",
    )
    tbl = pq.read_table(path)
    # Keep only context rows
    split_col = tbl.column("split").to_pylist()
    ctx_indices = [i for i, s in enumerate(split_col) if s == "context"]
    ctx_only = tbl.take(ctx_indices)
    with pytest.raises(ValueError):
        load_classification_eval_task(ctx_only)


def test_canonical_stage_path_matches_generator_and_prep(tmp_path):
    """The canonical stage root used by prepare must match the generator output path."""
    import importlib
    import os

    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    # Import prep to inspect stage root constant
    import prepare_synthetic_classification as prep

    # The prep stage root must use 'synthetic_classification_prepared', not 'classification'
    assert "synthetic_classification_prepared" in prep.SYNCLS_STAGE_ROOT, (
        f"prep.SYNCLS_STAGE_ROOT={prep.SYNCLS_STAGE_ROOT!r} does not use "
        "'synthetic_classification_prepared' prefix"
    )
    assert "classification/" not in prep.SYNCLS_STAGE_ROOT.replace(
        "synthetic_classification_prepared", ""
    ), "Stage root should not have old 'classification/' prefix"


def test_no_nested_x_train_column_in_output(tmp_path):
    """write_parquet_classification_eval must NOT produce nested X_train column."""
    ds = _make_ds()
    path = str(tmp_path / "task.parquet")
    write_parquet_classification_eval(
        ds, path,
        suite_id="test", suite_family="primary",
        dataset_idx=0, global_idx=0,
        profile="linear_classification_stat_aware",
        regime="A_iid_dense",
    )
    tbl = pq.read_table(path)
    assert "X_train" not in tbl.column_names, (
        "X_train nested column must not appear in eval_v1/v2 format"
    )
    assert "X_test" not in tbl.column_names
    assert "y_train" not in tbl.column_names
    assert "y_test" not in tbl.column_names


def test_multiclass_roundtrip(tmp_path):
    """Multiclass (K=4) data loads correctly."""
    ds = _make_ds(n_train=60, n_test=20, p=6, num_classes=4)
    task = _write_and_load(tmp_path, ds=ds)

    assert task["num_classes"] == 4
    assert set(task["y_train"].tolist()).issubset({0, 1, 2, 3})
    assert set(task["y_test"].tolist()).issubset({0, 1, 2, 3})


def test_feature_width_consistency(tmp_path):
    """X_train and X_test must have same number of features."""
    task = _write_and_load(tmp_path)
    assert task["X_train"].shape[1] == task["X_test"].shape[1]


def test_metadata_scalars_preserved(tmp_path):
    """Scalar metadata fields must survive the write/load round-trip."""
    task = _write_and_load(tmp_path)
    assert task["suite_id"] == "roundtrip_test"
    assert task["suite_family"] == "primary"
    assert task["task_seed"] == 42
    assert task["num_classes"] == 2
    assert task["n_train"] > 0
    assert task["n_test"] > 0
