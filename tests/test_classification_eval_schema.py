"""
test_classification_eval_schema.py
====================================
Unit tests for write_parquet_classification_eval() and
validate_classification_eval_dataset() in src/dgp_helpers.py.

Tests exercise the eval writer/validator in isolation without
invoking the top-level generation script.
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
    CLASSIFICATION_EVAL_SCHEMA_VERSION,
    CLASSIFICATION_EVAL_SUITE_FAMILIES,
    _REGRESSION_ONLY_FIELDS,
    generate_classification_dataset,
    split_classification_dataset,
    validate_classification_eval_dataset,
    write_parquet_classification_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ds(n_train: int = 50, n_test: int = 20, p: int = 5, num_classes: int = 2) -> dict:
    rng = np.random.default_rng(99)
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


_WRITE_DEFAULTS = dict(
    suite_id="test_suite",
    suite_family="primary",
    dataset_idx=0,
    global_idx=0,
    profile="linear_classification_stat_aware",
    regime="A_iid_dense",
    task_seed=0,
    store_class_params=False,
)


def _write(tmp_path, ds=None, **kwargs) -> Path:
    if ds is None:
        ds = _make_ds()
    path = tmp_path / "eval.parquet"
    kw = {**_WRITE_DEFAULTS, **kwargs}
    write_parquet_classification_eval(ds, str(path), **kw)
    return path


# ---------------------------------------------------------------------------
# Tests: writer output structure
# ---------------------------------------------------------------------------

def test_write_returns_row_count(tmp_path):
    """write_parquet_classification_eval returns n_train + n_test."""
    ds = _make_ds(n_train=30, n_test=12)
    path = tmp_path / "eval.parquet"
    n = write_parquet_classification_eval(ds, str(path), **_WRITE_DEFAULTS)
    assert n == 42


def test_output_has_per_row_format(tmp_path):
    """Output parquet must have per-row format (not nested X_train/X_test columns)."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    assert "X_train" not in t.column_names
    assert "y_train" not in t.column_names
    assert "X_test" not in t.column_names
    assert "y_test" not in t.column_names


def test_output_has_split_column(tmp_path):
    """Output parquet must have split column with 'context'/'query' values."""
    path = _write(tmp_path, ds=_make_ds(n_train=10, n_test=5))
    t = pq.read_table(str(path))
    assert "split" in t.column_names
    splits = set(t.column("split").to_pylist())
    assert splits == {"context", "query"}


def test_output_context_query_counts(tmp_path):
    """Context row count = n_train, query row count = n_test."""
    n_train, n_test = 15, 8
    path = _write(tmp_path, ds=_make_ds(n_train=n_train, n_test=n_test))
    t = pq.read_table(str(path))
    splits = t.column("split").to_pylist()
    assert splits.count("context") == n_train
    assert splits.count("query") == n_test


def test_schema_version_in_output(tmp_path):
    """Output parquet must have schema_version column with known value."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    assert "schema_version" in t.column_names
    versions = set(t.column("schema_version").to_pylist())
    assert len(versions) == 1
    assert list(versions)[0] == CLASSIFICATION_EVAL_SCHEMA_VERSION


def test_feature_vector_column_shape(tmp_path):
    """feature_vector column must be list<float64> with correct width."""
    p = 7
    path = _write(tmp_path, ds=_make_ds(p=p))
    t = pq.read_table(str(path))
    assert "feature_vector" in t.column_names
    fv = t.column("feature_vector")
    # Each element should be a list of length p
    first = fv[0].as_py()
    assert len(first) == p


def test_label_column_has_integer_dtype(tmp_path):
    """label column must be integer dtype."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    assert "label" in t.column_names
    assert pa.types.is_integer(t.schema.field("label").type)


def test_no_regression_fields_in_output(tmp_path):
    """Regression-only fields must not appear in classification eval parquet."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    for field in _REGRESSION_ONLY_FIELDS:
        assert field not in t.column_names, f"Regression-only field {field!r} found in output"


def test_invalid_suite_family_raises(tmp_path):
    """Unknown suite_family raises ValueError."""
    ds = _make_ds()
    path = tmp_path / "eval.parquet"
    with pytest.raises(ValueError, match="suite_family"):
        write_parquet_classification_eval(
            ds, str(path), suite_family="invalid_family_xyz",
            suite_id="test", dataset_idx=0, global_idx=0,
            profile="linear_classification_stat_aware", regime="A",
        )


def test_teacher_probs_written_when_requested(tmp_path):
    """Teacher probability columns are written when store_teacher_preds=True."""
    ds = _make_ds(n_train=20, n_test=10, num_classes=2)
    teacher = {
        "teacher_available": True,
        "teacher_probs_train": np.full((20, 2), 0.5),
        "teacher_probs_test": np.full((10, 2), 0.5),
        "teacher_converged": True,
        "teacher_n_iter": 100,
        "teacher_C": 1.0,
    }
    path = tmp_path / "eval.parquet"
    write_parquet_classification_eval(
        ds, str(path),
        store_teacher_preds=True, teacher_results=teacher,
        **_WRITE_DEFAULTS,
    )
    t = pq.read_table(str(path))
    assert "teacher_prob_0" in t.column_names
    assert "teacher_prob_1" in t.column_names


def test_suite_id_propagated(tmp_path):
    """suite_id must be present and consistent across all rows."""
    sid = "my_test_suite_v42"
    path = _write(tmp_path, suite_id=sid)
    t = pq.read_table(str(path))
    assert "suite_id" in t.column_names
    suite_ids = set(t.column("suite_id").to_pylist())
    assert suite_ids == {sid}


def test_num_classes_column(tmp_path):
    """num_classes column must be present and match the dataset."""
    path = _write(tmp_path, ds=_make_ds(num_classes=4))
    t = pq.read_table(str(path))
    assert "num_classes" in t.column_names
    vals = set(t.column("num_classes").to_pylist())
    assert vals == {4}


# ---------------------------------------------------------------------------
# Tests: validator
# ---------------------------------------------------------------------------

def test_valid_parquet_passes_validator(tmp_path):
    """Valid parquet written by write_parquet_classification_eval passes validator."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    errors = validate_classification_eval_dataset(t, strict=False)
    assert errors == []


def test_missing_split_column_returns_error(tmp_path):
    """Table missing split column must return an error."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    t_no_split = t.remove_column(t.schema.get_field_index("split"))
    errors = validate_classification_eval_dataset(t_no_split, strict=False)
    assert any("split" in e for e in errors)


def test_wrong_task_objective_returns_error(tmp_path):
    """Wrong task_objective must be detected."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    bad = t.set_column(
        t.schema.get_field_index("task_objective"),
        "task_objective",
        pa.array(["regression"] * len(t), type=pa.utf8()),
    )
    errors = validate_classification_eval_dataset(bad, strict=False)
    assert any("task_objective" in e for e in errors)


def test_label_out_of_range_returns_error(tmp_path):
    """Labels >= num_classes must be detected."""
    path = _write(tmp_path, ds=_make_ds(num_classes=2))
    t = pq.read_table(str(path))
    # Set all labels to 2 (out of range for K=2)
    bad = t.set_column(
        t.schema.get_field_index("label"),
        "label",
        pa.array([2] * len(t), type=pa.int64()),
    )
    errors = validate_classification_eval_dataset(bad, strict=False)
    assert any("label" in e.lower() or "range" in e for e in errors)


def test_strict_raises_on_first_error(tmp_path):
    """strict=True must raise ValueError on the first error found."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    bad = t.set_column(
        t.schema.get_field_index("task_objective"),
        "task_objective",
        pa.array(["regression"] * len(t), type=pa.utf8()),
    )
    with pytest.raises(ValueError):
        validate_classification_eval_dataset(bad, strict=True)


def test_valid_parquet_no_strict_raises(tmp_path):
    """Valid parquet in strict mode must not raise."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    errors = validate_classification_eval_dataset(t, strict=True)
    assert errors == []


def test_wrong_schema_version_returns_error(tmp_path):
    """Wrong schema_version returns error from validator."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    bad = t.set_column(
        t.schema.get_field_index("schema_version"),
        "schema_version",
        pa.array(["wrong_version_v99"] * len(t), type=pa.utf8()),
    )
    errors = validate_classification_eval_dataset(bad, strict=False)
    assert any("schema_version" in e for e in errors)


def test_regression_field_in_table_returns_error(tmp_path):
    """Regression field in table returns error from validator."""
    path = _write(tmp_path)
    t = pq.read_table(str(path))
    contaminated = t.append_column(
        "betaX",
        pa.array([[0.0] * 8] * len(t), type=pa.list_(pa.float64())),
    )
    errors = validate_classification_eval_dataset(contaminated, strict=False)
    assert any("betaX" in e or "Regression" in e for e in errors)


# ---------------------------------------------------------------------------
# Phase 1: load_classification_eval_task round-trip tests
# ---------------------------------------------------------------------------

def test_load_classification_eval_task_roundtrip(tmp_path):
    """write_parquet_classification_eval() output can be loaded by load_classification_eval_task()."""
    from dgp_helpers import load_classification_eval_task

    path = _write(tmp_path)
    t = pq.read_table(str(path))
    task = load_classification_eval_task(t)

    assert task["X_train"] is not None
    assert task["y_train"] is not None
    assert task["X_test"] is not None
    assert task["y_test"] is not None
    assert task["n_train"] == task["X_train"].shape[0]
    assert task["n_test"] == task["X_test"].shape[0]
    assert task["X_train"].shape[1] == task["X_test"].shape[1]
    assert task["num_classes"] >= 2
    assert task["suite_id"] is not None
    all_families = {"primary"} | CLASSIFICATION_EVAL_SUITE_FAMILIES
    assert task["suite_family"] in all_families


def test_load_classification_eval_task_wrong_schema_raises(tmp_path):
    """load_classification_eval_task raises ValueError on wrong schema_version."""
    from dgp_helpers import load_classification_eval_task

    path = _write(tmp_path)
    t = pq.read_table(str(path))
    bad = t.set_column(
        t.schema.get_field_index("schema_version"),
        "schema_version",
        pa.array(["bad_version_v0"] * len(t), type=pa.utf8()),
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_classification_eval_task(bad)
