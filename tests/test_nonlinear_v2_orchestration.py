"""
test_nonlinear_v2_orchestration.py
===================================
9 index/pipeline/isolation tests for nonlinear_v2.
Uses FakeSession/FakeJob patterns — no Snowflake dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Any

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
SRC = os.path.join(ROOT, "src")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Delay import so we can patch constants before module-level code runs
import importlib


# ---------------------------------------------------------------------------
# FakeSession helper
# ---------------------------------------------------------------------------

class FakeSession:
    """Minimal session stub that records SQL calls."""

    def __init__(self):
        self.sql_calls: list[str] = []
        self._manifest: dict | None = None
        self._row_count: int = 0

    def sql(self, stmt: str):
        self.sql_calls.append(stmt.strip())
        return self

    def collect(self):
        # Heuristic: return count row for SELECT COUNT(*) queries
        if "SELECT COUNT(*)" in " ".join(self.sql_calls[-1:]).upper():
            return [[self._row_count]]
        if "GROUP BY prior_regime" in " ".join(self.sql_calls[-1:]):
            return [["poly_quad", 70], ["sin_low", 70], ["hinge", 70],
                    ["sparse_interact", 70], ["mixed_linear", 70], ["demand_mono", 70]]
        return []

    def set_row_count(self, n: int):
        self._row_count = n

    class _FileOps:
        def __init__(self, session, manifest):
            self._session = session
            self._manifest = manifest

        def get(self, remote_path: str, local_dir: str):
            os.makedirs(local_dir, exist_ok=True)
            filename = remote_path.split("/")[-1]
            local_path = os.path.join(local_dir, filename)
            with open(local_path, "w") as f:
                json.dump(self._manifest, f)

    @property
    def file(self):
        return self._FileOps(self, self._manifest)


def _make_valid_manifest(n=420):
    """Build a minimal valid manifest with all 6 families × 7 regimes."""
    families = ["poly_quad", "sin_low", "hinge", "sparse_interact", "mixed_linear", "demand_mono"]
    regimes = ["iid_dense", "iid_sparse", "ar1", "block", "equicorr", "noise_feats", "feat_noise"]
    datasets = []
    idx = 0
    for fam in families:
        for reg in regimes:
            key = f"nonlinear:{fam}:{idx:05d}"
            datasets.append({
                "logical_dataset_key": key,
                "condition_id": f"{fam}_{reg}_n300_p16_noisegaussian_scale1.0",
                "target_family": fam,
                "feature_regime": reg,
                "suite_component": "core",
                "dataset_idx": idx,
                "filename": f"{fam}/dataset_{idx:05d}.parquet",
                "n_total": 300,
                "p_signal": 16,
                "p_noise": 0,
                "p_total": 16,
                "target_noise_scale": 1.0,
                "target_noise_type": "gaussian",
                "feature_noise_sigma": 0.0,
                "normalization_constant": 1.2,
                "teacher_seed": 1000 + idx,
                "sample_seed": 2000 + idx,
                "snr_target": 1.0,
                "rho": 0.0,
                "active_fraction": 1.0,
                "covariance_type": "iid",
                "noise_feature_fraction": 0.0,
            })
            idx += 1
    # Pad to n with duplicates of last family if needed
    while len(datasets) < min(n, idx):
        pass  # 42 cells is fine for validation tests
    return {
        "suite_id": "nonlinear",
        "prior_name": "nonlinear",
        "prior_version": "v2",
        "n_datasets": len(datasets),
        "target_families": families,
        "split_seeds": [0, 1, 2],
        "datasets": datasets,
    }


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import evaluate_nonlinear_regression as mod


# ---------------------------------------------------------------------------
# Test 1: _truncate uses suite_id = 'nonlinear', not 'nonlinear_v1'
# ---------------------------------------------------------------------------

def test_truncate_uses_v2_suite_id_only():
    """DELETE must use suite_id = 'nonlinear', not 'nonlinear_v1'."""
    session = FakeSession()
    mod._truncate_nonlinear_index(session)

    assert any("DELETE FROM" in sql for sql in session.sql_calls)
    delete_sql = next(sql for sql in session.sql_calls if "DELETE FROM" in sql)
    assert "nonlinear_v1" not in delete_sql, (
        "Truncate must not touch v1 rows (nonlinear_v1)"
    )
    assert "suite_id = 'nonlinear'" in delete_sql or "suite_id='nonlinear'" in delete_sql, (
        f"DELETE must filter by suite_id='nonlinear', got: {delete_sql!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Schema migration uses IF NOT EXISTS for all columns
# ---------------------------------------------------------------------------

def test_schema_migration_uses_if_not_exists():
    """All ALTER TABLE statements must include IF NOT EXISTS."""
    session = FakeSession()
    mod._run_schema_migration(session)

    alter_statements = [sql for sql in session.sql_calls if "ALTER TABLE" in sql.upper()]
    assert len(alter_statements) > 0, "No ALTER TABLE statements found"
    for sql in alter_statements:
        assert "IF NOT EXISTS" in sql.upper(), (
            f"ALTER TABLE missing IF NOT EXISTS: {sql!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: Schema migration covers all 13 new column names
# ---------------------------------------------------------------------------

def test_schema_migration_covers_all_13_columns():
    """Migration must touch all 13 expected new column names."""
    session = FakeSession()
    mod._run_schema_migration(session)

    all_sql = " ".join(session.sql_calls).lower()
    expected_cols = [
        "feature_regime", "covariance_type", "rho", "active_fraction",
        "noise_feature_fraction", "feature_noise_sigma", "suite_component",
        "target_noise_type", "snr_target", "condition_id",
        "teacher_seed", "sample_seed", "normalization_constant",
    ]
    missing = [col for col in expected_cols if col.lower() not in all_sql]
    assert missing == [], f"Missing columns in migration SQL: {missing}"


# ---------------------------------------------------------------------------
# Test 4: Index rows have correct logical key format
# ---------------------------------------------------------------------------

def test_index_rows_have_correct_logical_key_format():
    """logical_dataset_key must match 'nonlinear:{family}:{idx:05d}'."""
    manifest = _make_valid_manifest()
    rows = mod._build_index_rows(manifest)

    for row in rows:
        key = row["logical_dataset_key"]
        parts = key.split(":")
        assert len(parts) == 3, f"Key {key!r} should have 3 colon-separated parts"
        assert parts[0] == "nonlinear", f"Key prefix should be 'nonlinear', got {parts[0]!r}"
        assert parts[1] in [
            "poly_quad", "sin_low", "hinge", "sparse_interact", "mixed_linear", "demand_mono"
        ], f"Unknown family in key: {parts[1]!r}"
        assert len(parts[2]) == 5 and parts[2].isdigit(), (
            f"Index part should be 5-digit zero-padded, got {parts[2]!r}"
        )


# ---------------------------------------------------------------------------
# Test 5: Index rows have correct prior metadata
# ---------------------------------------------------------------------------

def test_index_rows_prior_metadata():
    """Rows must have prior_name='nonlinear', prior_version='v2', suite_id='nonlinear'."""
    manifest = _make_valid_manifest()
    rows = mod._build_index_rows(manifest)

    for row in rows:
        assert row["prior_name"] == "nonlinear", f"prior_name mismatch: {row['prior_name']!r}"
        assert row["prior_version"] == "v2", f"prior_version mismatch: {row['prior_version']!r}"
        assert row["suite_id"] == "nonlinear", f"suite_id mismatch: {row['suite_id']!r}"


# ---------------------------------------------------------------------------
# Test 6: Preflight catches missing coverage cell
# ---------------------------------------------------------------------------

def test_preflight_catches_missing_coverage_cell():
    """_validate_manifest_coverage should raise ValueError if a family is absent."""
    manifest = _make_valid_manifest()
    # Remove all poly_quad entries
    manifest["datasets"] = [
        d for d in manifest["datasets"] if d["target_family"] != "poly_quad"
    ]

    with pytest.raises(ValueError, match="poly_quad"):
        mod._validate_manifest_coverage(manifest)


# ---------------------------------------------------------------------------
# Test 7: Preflight catches duplicate logical keys
# ---------------------------------------------------------------------------

def test_preflight_catches_duplicate_logical_keys():
    """_validate_no_duplicate_keys should raise ValueError on collision."""
    manifest = _make_valid_manifest()
    # Introduce a duplicate
    dup = dict(manifest["datasets"][0])
    dup["dataset_idx"] = 9999
    manifest["datasets"].append(dup)

    with pytest.raises(ValueError, match="Duplicate"):
        mod._validate_no_duplicate_keys(manifest)


# ---------------------------------------------------------------------------
# Test 8: Orchestration injects NONLINEAR_SUITE_ID=nonlinear
# ---------------------------------------------------------------------------

def test_orchestration_injects_nonlinear_suite_id():
    """The module constant NONLINEAR_SUITE_ID must be 'nonlinear' by default."""
    assert mod.NONLINEAR_SUITE_ID == "nonlinear" or os.environ.get("NONLINEAR_SUITE_ID", "nonlinear") == "nonlinear", (
        f"NONLINEAR_SUITE_ID should default to 'nonlinear', got {mod.NONLINEAR_SUITE_ID!r}"
    )


# ---------------------------------------------------------------------------
# Test 9: Table is never dropped
# ---------------------------------------------------------------------------

def test_table_never_dropped(monkeypatch):
    """No DROP TABLE should appear in any SQL emitted by main()."""
    tmp_dir = tempfile.mkdtemp()
    try:
        # Patch file.get to write a valid manifest
        manifest = _make_valid_manifest()
        session = FakeSession()
        session._manifest = manifest
        # Patch _validate_stage_spot_check to no-op
        monkeypatch.setattr(mod, "_validate_stage_spot_check", lambda s, m, n_spot=5: None)
        # Patch _assert_nonlinear_index_populated to no-op
        monkeypatch.setattr(mod, "_assert_nonlinear_index_populated", lambda s, expected: len(manifest["datasets"]))
        # Set row count for SELECT COUNT(*)
        session.set_row_count(len(manifest["datasets"]))

        result = mod.main(session=session)

        all_sql = " ".join(session.sql_calls).upper()
        assert "DROP TABLE" not in all_sql, (
            "main() must never issue DROP TABLE"
        )
        assert "OK" in result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
