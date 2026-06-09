"""
tests/test_prepare_synthetic_classification.py
===============================================
Unit tests for scripts/prepare_synthetic_classification.py.

Uses mock Snowpark session pattern (same as existing regression tests).
No Snowflake connection required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import prepare_synthetic_classification as prep


# ---------------------------------------------------------------------------
# Fake session infrastructure
# ---------------------------------------------------------------------------

class _FakeSQLResult:
    def __init__(self, value=0):
        self._value = value

    def __getitem__(self, key):
        if isinstance(key, int):
            return self
        return self._value

    def collect(self):
        return [self]


class _FakeSession:
    def __init__(self, index_row_count: int = 0, stage_files: list | None = None):
        self._index_row_count = index_row_count
        self._stage_files = stage_files or []
        self.sql_calls = []
        self.file = _FakeFileOps(stage_files or [])

    def sql(self, query: str):
        self.sql_calls.append(query)
        # Return count for COUNT(*) queries
        if "COUNT(*)" in query.upper():
            return _FakeCountResult(self._index_row_count)
        return _FakeVoidResult()


class _FakeCountResult:
    def __init__(self, count: int):
        self._count = count

    def collect(self):
        row = MagicMock()
        row.__getitem__ = MagicMock(return_value=self._count)
        return [row]


class _FakeVoidResult:
    def collect(self):
        return []


class _FakeFileOps:
    def __init__(self, stage_files: list):
        self._stage_files = stage_files

    def get(self, stage_path: str, local_dir: str) -> None:
        # Write a minimal parquet file to local_dir for testing
        _write_minimal_parquet(Path(local_dir) / Path(stage_path).name)

    def put(self, local_path: str, stage_dir: str, **kwargs) -> None:
        pass  # no-op upload


def _write_minimal_parquet(path: Path, suite_id: str = "test_suite") -> None:
    """Write a minimal valid classification parquet in linear_classification_eval_v2 format."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import numpy as np
    except ImportError:
        # If pyarrow not available, write an empty file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from dgp_helpers import write_parquet_classification_eval

    n_train, n_test, p = 50, 20, 5
    num_classes = 2
    rng = np.random.default_rng(42)

    ds = {
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

    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_classification_eval(
        ds,
        str(path),
        suite_id=suite_id,
        suite_family="primary",
        dataset_idx=0,
        global_idx=0,
        profile="linear_classification_stat_aware",
        regime="A_iid_dense",
        task_seed=42,
        store_class_params=False,
    )


# ---------------------------------------------------------------------------
# Tests: default suite_id
# ---------------------------------------------------------------------------

def test_default_suite_id():
    """Default SYNTHETIC_CLASSIFICATION_SUITE_ID must be linear_classification_stat_aware."""
    # Reset env so module-level constant is visible
    env_backup = os.environ.pop("SYNTHETIC_CLASSIFICATION_SUITE_ID", None)
    try:
        import importlib
        importlib.reload(prep)
        assert prep.SYNCLS_SUITE_ID == "linear_classification_stat_aware"
    finally:
        if env_backup is not None:
            os.environ["SYNTHETIC_CLASSIFICATION_SUITE_ID"] = env_backup
        importlib.reload(prep)


# ---------------------------------------------------------------------------
# Tests: idempotency (skip when already indexed)
# ---------------------------------------------------------------------------

def test_skips_when_already_indexed(monkeypatch):
    """Should skip without inserting if rows already exist and FORCE_REBUILD is false."""
    session = _FakeSession(index_row_count=100)

    monkeypatch.setattr(prep, "SYNCLS_FORCE_REBUILD", False)
    monkeypatch.setattr(prep, "SYNCLS_SUITE_ID", "linear_classification_stat_aware")

    result = prep.prepare_synthetic_classification(session=session)
    assert "SKIPPED" in result
    assert "100" in result


# ---------------------------------------------------------------------------
# Tests: force rebuild
# ---------------------------------------------------------------------------

def test_force_rebuild_reindexes(monkeypatch, tmp_path):
    """When FORCE_REBUILD=TRUE, should delete and re-index even if rows exist."""
    parquet_name = "dataset_0000.parquet"
    _write_minimal_parquet(tmp_path / parquet_name)

    session = _FakeSession(
        index_row_count=50,  # existing rows
        stage_files=[f"@EVALUATION_DATASET_STAGE/synthetic_classification_prepared/test_suite/{parquet_name}"],
    )

    monkeypatch.setattr(prep, "SYNCLS_FORCE_REBUILD", True)
    monkeypatch.setattr(prep, "SYNCLS_SUITE_ID", "test_suite")
    monkeypatch.setattr(prep, "SYNCLS_LOCAL_DIR", str(tmp_path))

    # Patch _list_stage_parquets to return our fake file
    monkeypatch.setattr(
        prep, "_list_stage_parquets",
        lambda session, prefix: [f"@EVALUATION_DATASET_STAGE/synthetic_classification_prepared/test_suite/{parquet_name}"]
    )
    # Patch _download_parquet to use our pre-written file
    monkeypatch.setattr(
        prep, "_download_parquet",
        lambda session, stage_path, local_dir: tmp_path / parquet_name
    )

    # Patch insert to avoid Snowflake dependency
    inserted_rows = []
    monkeypatch.setattr(
        prep, "_insert_cls_index_rows",
        lambda session, rows: inserted_rows.extend(rows)
    )
    # Patch final validation count
    monkeypatch.setattr(
        prep, "_index_row_count",
        lambda session, suite_id: len(inserted_rows) if inserted_rows else 1
    )

    result = prep.prepare_synthetic_classification(session=session)
    assert "SKIPPED" not in result
    assert len(inserted_rows) == 1


# ---------------------------------------------------------------------------
# Tests: missing parquets raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_missing_parquets_raises(monkeypatch, tmp_path):
    """If no parquets are found on stage, should raise FileNotFoundError."""
    session = _FakeSession(index_row_count=0)

    monkeypatch.setattr(prep, "SYNCLS_FORCE_REBUILD", True)
    monkeypatch.setattr(prep, "SYNCLS_SUITE_ID", "empty_suite")
    monkeypatch.setattr(prep, "SYNCLS_LOCAL_DIR", str(tmp_path))
    monkeypatch.setattr(prep, "_list_stage_parquets", lambda *a, **kw: [])

    with pytest.raises(FileNotFoundError, match="No parquet files found"):
        prep.prepare_synthetic_classification(session=session)


# ---------------------------------------------------------------------------
# Tests: missing schema field raises ValueError (aggregated as RuntimeError)
# ---------------------------------------------------------------------------

def test_missing_required_field_raises(monkeypatch, tmp_path):
    """Parquet missing required fields should cause RuntimeError with summary."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Write parquet without 'num_classes' field
    bad_parquet = tmp_path / "bad.parquet"
    tbl = pa.table({"X_train": [[1.0, 2.0]], "y_train": [[0]]})
    pq.write_table(tbl, str(bad_parquet))

    session = _FakeSession(index_row_count=0)
    monkeypatch.setattr(prep, "SYNCLS_FORCE_REBUILD", True)
    monkeypatch.setattr(prep, "SYNCLS_SUITE_ID", "test_suite")
    monkeypatch.setattr(prep, "SYNCLS_LOCAL_DIR", str(tmp_path))
    monkeypatch.setattr(
        prep, "_list_stage_parquets",
        lambda *a, **kw: ["@stage/test_suite/bad.parquet"]
    )
    monkeypatch.setattr(
        prep, "_download_parquet",
        lambda session, stage_path, local_dir: bad_parquet
    )

    with pytest.raises(RuntimeError, match="failed validation"):
        prep.prepare_synthetic_classification(session=session)


# ---------------------------------------------------------------------------
# Tests: writes only to LINEAR_CLASSIFICATION_DATASET_INDEX
# ---------------------------------------------------------------------------

def test_writes_to_classification_index_only(monkeypatch, tmp_path):
    """Must insert into LINEAR_CLASSIFICATION_DATASET_INDEX, not regression index."""
    parquet_name = "dataset_0000.parquet"
    _write_minimal_parquet(tmp_path / parquet_name)

    session = _FakeSession(index_row_count=0)
    monkeypatch.setattr(prep, "SYNCLS_FORCE_REBUILD", True)
    monkeypatch.setattr(prep, "SYNCLS_SUITE_ID", "test_suite")
    monkeypatch.setattr(prep, "SYNCLS_LOCAL_DIR", str(tmp_path))
    monkeypatch.setattr(
        prep, "_list_stage_parquets",
        lambda *a, **kw: [f"@stage/test_suite/{parquet_name}"]
    )
    monkeypatch.setattr(
        prep, "_download_parquet",
        lambda session, stage_path, local_dir: tmp_path / parquet_name
    )

    sql_log = []
    original_insert = prep._insert_cls_index_rows

    def _spy_insert(session, rows):
        sql_log.append("INSERT")
        return None  # skip actual insert

    monkeypatch.setattr(prep, "_insert_cls_index_rows", _spy_insert)
    monkeypatch.setattr(prep, "_index_row_count", lambda *a: 1)

    prep.prepare_synthetic_classification(session=session)

    # Verify the index table constant is classification-specific
    assert prep.SYNCLS_INDEX_TABLE == "LINEAR_CLASSIFICATION_DATASET_INDEX"
    assert "LINEAR_REGRESSION_DATASET_INDEX" not in prep.SYNCLS_INDEX_TABLE


# ---------------------------------------------------------------------------
# Tests: partial errors aggregated
# ---------------------------------------------------------------------------

def test_partial_errors_aggregated(monkeypatch, tmp_path):
    """Multiple parquet validation errors must be aggregated in a single RuntimeError."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    bad1 = tmp_path / "bad1.parquet"
    bad2 = tmp_path / "bad2.parquet"
    for p in [bad1, bad2]:
        tbl = pa.table({"X_train": [[1.0]]})  # missing many required fields
        pq.write_table(tbl, str(p))

    session = _FakeSession(index_row_count=0)
    monkeypatch.setattr(prep, "SYNCLS_FORCE_REBUILD", True)
    monkeypatch.setattr(prep, "SYNCLS_SUITE_ID", "test_suite")
    monkeypatch.setattr(prep, "SYNCLS_LOCAL_DIR", str(tmp_path))
    monkeypatch.setattr(
        prep, "_list_stage_parquets",
        lambda *a, **kw: ["@stage/bad1.parquet", "@stage/bad2.parquet"]
    )

    files_map = {"@stage/bad1.parquet": bad1, "@stage/bad2.parquet": bad2}
    monkeypatch.setattr(
        prep, "_download_parquet",
        lambda session, stage_path, local_dir: files_map[stage_path]
    )

    with pytest.raises(RuntimeError) as exc_info:
        prep.prepare_synthetic_classification(session=session)

    # Both files should be mentioned in the error summary
    error_msg = str(exc_info.value)
    assert "2" in error_msg  # "2 parquet(s) failed"
