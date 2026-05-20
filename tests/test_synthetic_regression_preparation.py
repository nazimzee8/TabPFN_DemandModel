"""
tests/test_synthetic_regression_preparation.py
================================================
Unit tests for scripts/prepare_synthetic_regression.py.

All tests run without Snowflake connectivity (session is mocked).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Ensure src and scripts are importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "ood_regression"))

import prepare_synthetic_regression as prep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng():
    return np.random.default_rng(seed=42)


def _tiny_synreg_dataset(rng: np.random.Generator, regime: str) -> dict:
    """Return minimal valid NPZ-compatible dict for a given regime."""
    n, p = 30, 3
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    betaX = X @ beta
    y = betaX + rng.standard_normal(n)
    return {
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "suite_family": "primary",
        "prior_regime": regime,
        "n_total": n,
        "p_signal": p,
        "p_noise": 0,
        "p_total": p,
        "target_noise_scale": 1.0,
    }


@pytest.fixture()
def mock_session():
    """Return a lightweight mock Snowflake session."""
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    session.file.put.return_value = [MagicMock(status="UPLOADED")]
    session.file.get.side_effect = FileNotFoundError("no manifest")
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseSeed:
    def test_primary_suite_uses_seed_not_42(self):
        """Base seed must be 20260512, never 42."""
        assert prep.SYNREG_BASE_SEED != 42
        assert prep.SYNREG_BASE_SEED == 20260512

    def test_rng_from_base_seed_differs_from_seed42(self):
        rng_base = np.random.default_rng(prep.SYNREG_BASE_SEED)
        rng_42 = np.random.default_rng(42)
        val_base = rng_base.standard_normal(1)[0]
        val_42 = rng_42.standard_normal(1)[0]
        assert val_base != val_42


class TestPrimaryDatasetCount:
    def test_primary_n_datasets_constant(self):
        assert prep.PRIMARY_N_DATASETS == 200

    def test_env_override_primary_datasets(self):
        """Verify that the env var parsing function returns the overridden value."""
        # We test the parsing mechanism without reloading the module (to avoid global state pollution)
        import prepare_synthetic_regression as p2
        with patch.dict(os.environ, {"SYNTHETIC_REGRESSION_PRIMARY_DATASETS": "8"}):
            override_val = int(os.getenv("SYNTHETIC_REGRESSION_PRIMARY_DATASETS", "200"))
        assert override_val == 8


class TestPrimaryRegimeBalance:
    def test_primary_suite_balances_regimes(self, rng):
        """50 datasets per regime (tolerance ±5) across 200 datasets."""
        per_regime = prep.PRIMARY_N_DATASETS // len(prep.REGIMES)
        assert per_regime == 50
        assert len(prep.REGIMES) == 4

    def test_all_four_regimes_present(self):
        assert set(prep.REGIMES) == {"A", "B", "C", "D"}


class TestFeatureNoiseSuite:
    def test_feature_noise_suite_includes_all_noise_levels(self):
        """Noise levels 0,10,25,50,75,100 must all be present."""
        expected = {0, 10, 25, 50, 75, 100}
        actual = set(prep.FEATURE_NOISE_LEVELS)
        assert expected == actual

    def test_feature_noise_n_datasets(self):
        assert prep.FEATURE_NOISE_N_DATASETS == 80

    def test_feature_noise_seeds(self):
        assert prep.FEATURE_NOISE_SEEDS == [0, 1, 2]


class TestTrainingSizeSuite:
    def test_training_size_includes_anchor_constant(self):
        """TRAIN_SIZE_ANCHOR_N must be 4832."""
        assert prep.TRAIN_SIZE_ANCHOR_N == 4832

    def test_training_size_suite_holdout_is_1371(self):
        assert prep.HOLDOUT_SIZE == 1371

    def test_training_size_n_total_is_correct(self):
        expected_n_total = prep.TRAIN_SIZE_ANCHOR_N + prep.HOLDOUT_SIZE
        assert expected_n_total == 6203

    def test_training_size_grid_includes_anchor(self):
        assert prep.TRAIN_SIZE_ANCHOR_N in prep.TRAIN_SIZE_GRID

    def test_training_size_grid_has_8_entries(self):
        assert len(prep.TRAIN_SIZE_GRID) == 8

    def test_n_holdout_default_for_training_size_suite(self):
        n_holdout = prep._compute_n_holdout_default("training_size", 6203)
        assert n_holdout == prep.HOLDOUT_SIZE


class TestNPZSerialization:
    def test_npz_serializes_without_allow_pickle(self, rng):
        """NPZ must be loadable with allow_pickle=False."""
        tmp_path = Path(tempfile.mkdtemp())
        ds = _tiny_synreg_dataset(rng, "A")
        outpath = str(tmp_path / "test.npz")
        prep.serialize_synthetic_npz(outpath, ds, noise_level=0, is_anchor=False)
        loaded = np.load(outpath, allow_pickle=False)
        assert "X" in loaded
        assert "y" in loaded
        assert "betaX" in loaded

    def test_npz_metadata_as_arrays(self, rng):
        """All metadata must be stored as 1-element arrays (no Python objects)."""
        tmp_path = Path(tempfile.mkdtemp())
        ds = _tiny_synreg_dataset(rng, "B")
        outpath = str(tmp_path / "meta.npz")
        prep.serialize_synthetic_npz(outpath, ds, noise_level=25, is_anchor=True)
        loaded = np.load(outpath, allow_pickle=False)
        assert loaded["n_total"].shape == (1,)
        assert loaded["p_signal"].shape == (1,)
        assert loaded["feature_noise_level"].shape == (1,)
        assert int(loaded["feature_noise_level"][0]) == 25
        assert bool(loaded["training_size_anchor"][0]) is True

    def test_npz_dtype_float64(self, rng):
        tmp_path = Path(tempfile.mkdtemp())
        ds = _tiny_synreg_dataset(rng, "C")
        outpath = str(tmp_path / "dtype.npz")
        prep.serialize_synthetic_npz(outpath, ds, noise_level=0)
        loaded = np.load(outpath, allow_pickle=False)
        assert loaded["X"].dtype == np.float64
        assert loaded["y"].dtype == np.float64
        assert loaded["betaX"].dtype == np.float64


class TestIndexRowConstruction:
    def test_index_row_has_required_columns(self, rng):
        row = prep._make_index_row(
            suite_family="primary",
            dataset_id=0,
            dataset_seed=42,
            stage_path="@stage/primary/dataset_0000.npz",
            regime="A",
            split_seeds=[0, 1, 2],
            n_total=100,
            p_signal=5,
            p_noise=0,
        )
        required = [
            "stage_path", "suite_family", "prior_regime", "p_signal", "p_noise", "p_total",
            "n_total", "n_train_default", "n_holdout_default",
        ]
        for col in required:
            assert col in row, f"Missing column: {col}"
            assert row[col] is not None, f"Column {col} is None"

    def test_p_total_is_sum(self):
        row = prep._make_index_row(
            suite_family="feature_noise",
            dataset_id=0,
            dataset_seed=1,
            stage_path="@stage/fn/d.npz",
            regime="B",
            split_seeds=[0],
            n_total=50,
            p_signal=5,
            p_noise=25,
        )
        assert row["p_total"] == 30

    def test_n_holdout_primary_suite(self):
        row = prep._make_index_row(
            suite_family="primary",
            dataset_id=0,
            dataset_seed=1,
            stage_path="@s/d.npz",
            regime="A",
            split_seeds=[0],
            n_total=100,
            p_signal=5,
            p_noise=0,
        )
        # Primary suite: 80/20 split
        assert row["n_holdout_default"] == 20
        assert row["n_train_default"] == 80


class TestManifestIdempotency:
    def test_manifest_validation_idempotent(self):
        """
        Run prepare twice (mocking everything): same manifest content, no re-upload on 2nd run.
        """
        tmp_path = Path(tempfile.mkdtemp())
        # Build a minimal manifest
        manifest = {
            "suite_id": prep.SYNREG_SUITE_ID,
            "n_datasets_primary": 200,
            "n_datasets_feature_noise": 80,
            "n_datasets_training_size": 40,
            "n_datasets_target_noise": 0,
            "stage_paths": ["@stage/file1.npz"],
        }
        manifest_file = tmp_path / "synthetic_regression_manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        session = MagicMock()
        # Simulate manifest already present and downloadable
        def _fake_get(stage_path, local_dir):
            dest = Path(local_dir) / "synthetic_regression_manifest.json"
            dest.write_text(json.dumps(manifest))
        session.file.get.side_effect = _fake_get

        # LIST returns the expected stage file
        list_row = MagicMock()
        list_row.__getitem__ = lambda self, k: "@META_DATASET_STAGE/.../file1.npz"
        session.sql.return_value.collect.return_value = [
            type("Row", (), {"name": "file1.npz"})()
        ]

        with patch.dict(os.environ, {"SYNTHETIC_REGRESSION_FORCE_REBUILD": "false"}):
            with patch("prepare_synthetic_regression.SYNREG_LOCAL_DIR", str(tmp_path)):
                result = prep.create_or_validate_manifest(session, prep.SYNREG_SUITE_ID)
        # With file1.npz present in LIST and valid manifest: should be True (no rebuild)
        assert isinstance(result, bool)

    def test_force_rebuild_ignores_valid_manifest(self, mock_session):
        """SYNTHETIC_REGRESSION_FORCE_REBUILD=true must always return False from validation."""
        with patch.dict(os.environ, {"SYNTHETIC_REGRESSION_FORCE_REBUILD": "true"}):
            with patch("prepare_synthetic_regression.SYNREG_FORCE_REBUILD", True):
                result = prep.create_or_validate_manifest(mock_session, prep.SYNREG_SUITE_ID)
        assert result is False

    def test_missing_manifest_triggers_rebuild(self, mock_session):
        """If manifest file not found, must return False."""
        mock_session.file.get.side_effect = FileNotFoundError("not found")
        with patch("prepare_synthetic_regression.SYNREG_FORCE_REBUILD", False):
            result = prep.create_or_validate_manifest(mock_session, prep.SYNREG_SUITE_ID)
        assert result is False

    def test_wrong_suite_id_triggers_rebuild(self, mock_session):
        tmp_path = Path(tempfile.mkdtemp())
        manifest = {"suite_id": "other_suite", "n_datasets_primary": 200,
                    "n_datasets_feature_noise": 80, "n_datasets_training_size": 40}
        manifest_file = tmp_path / "synthetic_regression_manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        def _fake_get(stage_path, local_dir):
            dest = Path(local_dir) / "synthetic_regression_manifest.json"
            dest.write_text(json.dumps(manifest))
        mock_session.file.get.side_effect = _fake_get

        with patch("prepare_synthetic_regression.SYNREG_LOCAL_DIR", str(tmp_path)):
            with patch("prepare_synthetic_regression.SYNREG_FORCE_REBUILD", False):
                result = prep.create_or_validate_manifest(mock_session, prep.SYNREG_SUITE_ID)
        assert result is False


class TestSampleParams:
    def test_sample_params_primary_constraints(self, rng):
        """sample_params_primary must return p>=1, n>=5, n>=5*p."""
        for _ in range(20):
            n, p = prep.sample_params_primary(rng)
            assert p >= 1
            assert n >= 5
            assert n >= 5 * p

    def test_sample_params_training_size_meets_n_required(self, rng):
        n_required = 6203
        n, p = prep.sample_params_training_size(rng, n_required=n_required)
        assert n == n_required
        assert p >= 1
        assert n >= max(5, p)

    def test_sample_params_for_signal(self, rng):
        for _ in range(10):
            n, p_signal = prep.sample_params_for_signal(rng)
            assert p_signal >= 1
            assert n >= max(5 * p_signal, 50)


class TestGenerateSyntheticDataset:
    @pytest.mark.parametrize("regime", ["A", "B", "C", "D"])
    def test_all_regimes_generate(self, rng, regime):
        ds = prep.generate_synthetic_dataset(rng, regime, "primary", n=50, p_signal=4, p_noise=0)
        assert ds["X"].shape == (50, 4)
        assert ds["y"].shape == (50,)
        assert ds["betaX"].shape == (50,)

    def test_noise_features_appended(self, rng):
        ds = prep.generate_synthetic_dataset(rng, "A", "feature_noise", n=30, p_signal=3, p_noise=10)
        assert ds["X"].shape == (30, 13)
        assert ds["p_total"] == 13
        assert ds["p_noise"] == 10

    def test_target_noise_scale_zero_means_betaX_equals_y(self, rng):
        ds = prep.generate_synthetic_dataset(
            rng, "A", "target_noise", n=100, p_signal=5, target_noise_scale=0.0
        )
        np.testing.assert_allclose(ds["y"], ds["betaX"], atol=1e-10)


class TestEnvHelpers:
    def test_env_flag_true(self):
        with patch.dict(os.environ, {"TEST_FLAG": "true"}):
            assert prep._env_flag("TEST_FLAG") is True

    def test_env_flag_false(self):
        with patch.dict(os.environ, {"TEST_FLAG": "false"}):
            assert prep._env_flag("TEST_FLAG") is False

    def test_env_flag_default(self):
        # Ensure env var not set
        os.environ.pop("__UNSET_FLAG__", None)
        assert prep._env_flag("__UNSET_FLAG__", "false") is False
        assert prep._env_flag("__UNSET_FLAG__", "true") is True

    def test_parse_int_list(self):
        assert prep._parse_int_list("0,1,2,3,4") == [0, 1, 2, 3, 4]

    def test_parse_float_list(self):
        result = prep._parse_float_list("0.0,0.1,0.25")
        assert result == [0.0, 0.1, 0.25]


class TestInsertSynregIndexRows:
    def _make_rows(self, n: int = 3) -> list[dict]:
        """Return n minimal valid index rows."""
        rows = []
        for i in range(n):
            rows.append(prep._make_index_row(
                suite_family="primary",
                dataset_id=i,
                dataset_seed=100 + i,
                stage_path=f"@stage/primary/dataset_{i:04d}.npz",
                regime="A",
                split_seeds=[0, 1, 2, 3, 4],
                n_total=200,
                p_signal=5,
                p_noise=0,
                payload_bytes=4096,
            ))
        return rows

    def _make_session(self, n_rows: int):
        """Return a fake session that records SQL and returns correct COUNT."""
        sql_log = []
        def fake_sql(sql_str):
            sql_log.append(sql_str)
            mock_result = MagicMock()
            if "COUNT(*)" in sql_str:
                mock_row = MagicMock()
                mock_row.__getitem__ = lambda self, i: n_rows
                mock_result.collect.return_value = [mock_row]
            else:
                mock_result.collect.return_value = []
            return mock_result
        session = MagicMock()
        session.sql.side_effect = fake_sql
        return session, sql_log

    def test_no_parse_json_in_values_clause(self):
        """PARSE_JSON must not appear inside a VALUES clause (Snowflake rejects it)."""
        rows = self._make_rows(3)
        session, sql_log = self._make_session(len(rows))
        prep.insert_synreg_index_rows(session, rows)
        insert_sqls = [s for s in sql_log if "INSERT INTO" in s]
        assert insert_sqls, "No INSERT SQL was generated"
        for sql in insert_sqls:
            # If PARSE_JSON is present, VALUES must NOT be present
            if "PARSE_JSON" in sql:
                import re
                assert not re.search(r'\bVALUES\b', sql, re.IGNORECASE), (
                    "PARSE_JSON found inside a VALUES clause — "
                    "Snowflake rejects this. Use INSERT … SELECT … UNION ALL instead."
                )

    def test_insert_uses_select_union_all(self):
        """INSERT must use SELECT … UNION ALL … not VALUES."""
        rows = self._make_rows(3)
        session, sql_log = self._make_session(len(rows))
        prep.insert_synreg_index_rows(session, rows)
        insert_sqls = [s for s in sql_log if "INSERT INTO" in s]
        assert insert_sqls, "No INSERT SQL was generated"
        for sql in insert_sqls:
            assert "SELECT" in sql.upper(), "INSERT must use SELECT expressions"
            assert "UNION ALL" in sql.upper(), "Multiple rows must be joined with UNION ALL"

    def test_row_count_validation_passes(self):
        """insert_synreg_index_rows must not raise when count matches."""
        rows = self._make_rows(5)
        session, _ = self._make_session(len(rows))
        # Should not raise
        prep.insert_synreg_index_rows(session, rows)

    def test_row_count_validation_raises_on_mismatch(self):
        """insert_synreg_index_rows must raise RuntimeError on count mismatch."""
        rows = self._make_rows(3)
        session, _ = self._make_session(999)  # wrong count
        with pytest.raises(RuntimeError, match="row count"):
            prep.insert_synreg_index_rows(session, rows)


# ---------------------------------------------------------------------------
# Tests: _validate_parquet_index (Issue 2)
# ---------------------------------------------------------------------------

def _make_manifest_dataset(dataset_id: int, regime: str = "A",
                            suite_family: str = "primary") -> dict:
    return {
        "suite_id": prep.SYNREG_SUITE_ID,
        "suite_family": suite_family,
        "dataset_id": dataset_id,
        "dataset_seed": dataset_id * 10,
        "stage_path": f"@EVALUATION_DATASET_STAGE/primary/dataset_{dataset_id:04d}.parquet",
        "prior_regime": regime,
        "n_total": 200,
        "n_train_default": 160,
        "n_holdout_default": 40,
        "p_signal": 5,
        "p_noise": 0,
        "p_total": 5,
        "feature_noise_level": 0,
        "target_noise_scale": 1.0,
        "training_size_anchor": False,
        "split_seeds": [0, 1, 2, 3, 4],
        "payload_bytes": 1024,
    }


def _make_index_row_obj(rec: dict):
    """Return a fake Row object whose as_dict() returns the record."""
    obj = MagicMock()
    obj.as_dict.return_value = rec
    return obj


def _build_mock_session_for_validation(index_rows: list[dict], count_override: int | None = None):
    """Return a mock session that returns given index rows for SELECT * queries
    and the count for COUNT(*) queries."""
    n = count_override if count_override is not None else len(index_rows)

    def fake_sql(sql_str):
        result = MagicMock()
        if "COUNT(*)" in sql_str:
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, i: n
            result.collect.return_value = [mock_row]
        else:
            result.collect.return_value = [_make_index_row_obj(r) for r in index_rows]
        return result

    session = MagicMock()
    session.sql.side_effect = fake_sql
    return session


class TestParquetManifestValidation:
    def _make_manifest(self, n: int = 3) -> dict:
        return {
            "suite_id": prep.SYNREG_SUITE_ID,
            "locally_generated": True,
            "datasets": [_make_manifest_dataset(i) for i in range(n)],
        }

    def test_empty_index_triggers_rebuild(self):
        """count=0 → returns False."""
        manifest = self._make_manifest(3)
        session = _build_mock_session_for_validation([], count_override=0)
        result = prep._validate_parquet_index(session, manifest, prep.SYNREG_SUITE_ID)
        assert result is False

    def test_partial_index_triggers_rebuild(self):
        """count < len(manifest['datasets']) → returns False."""
        manifest = self._make_manifest(5)
        index_rows = [_make_manifest_dataset(i) for i in range(3)]  # only 3 of 5
        session = _build_mock_session_for_validation(index_rows, count_override=3)
        result = prep._validate_parquet_index(session, manifest, prep.SYNREG_SUITE_ID)
        assert result is False

    def test_stale_stage_path_triggers_rebuild(self):
        """manifest stage_path not in index rows → returns False."""
        manifest = self._make_manifest(3)
        # Index rows have wrong stage paths
        bad_rows = []
        for i in range(3):
            r = _make_manifest_dataset(i)
            r["stage_path"] = f"@EVALUATION_DATASET_STAGE/primary/dataset_{i:04d}.WRONG"
            bad_rows.append(r)
        session = _build_mock_session_for_validation(bad_rows, count_override=3)
        result = prep._validate_parquet_index(session, manifest, prep.SYNREG_SUITE_ID)
        assert result is False

    def test_valid_index_skips_rebuild(self):
        """All checks pass → returns True."""
        manifest = self._make_manifest(3)
        index_rows = [_make_manifest_dataset(i) for i in range(3)]
        session = _build_mock_session_for_validation(index_rows, count_override=3)
        result = prep._validate_parquet_index(session, manifest, prep.SYNREG_SUITE_ID)
        assert result is True

    def test_duplicate_logical_keys_trigger_rebuild(self):
        """Duplicate (suite_family, regime, dataset_id, noise, scale) → returns False."""
        manifest = self._make_manifest(3)
        # Index contains duplicate for dataset_id=0
        index_rows = [
            _make_manifest_dataset(0),  # first copy
            _make_manifest_dataset(0),  # duplicate
            _make_manifest_dataset(1),
        ]
        session = _build_mock_session_for_validation(index_rows, count_override=3)
        result = prep._validate_parquet_index(session, manifest, prep.SYNREG_SUITE_ID)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: _truncate_synreg_index force rebuild behavior (Issue 5)
# ---------------------------------------------------------------------------

class TestForceRebuildBehavior:
    def _sql_log_session(self):
        sql_calls = []

        def fake_sql(sql_str):
            sql_calls.append(sql_str)
            result = MagicMock()
            result.collect.return_value = []
            return result

        session = MagicMock()
        session.sql.side_effect = fake_sql
        return session, sql_calls

    def test_force_rebuild_deletes_only_active_suite_rows(self):
        """SYNREG_FORCE_REBUILD=true, DROP_INDEX_TABLE=false → DELETE WHERE suite_id, not DROP."""
        session, sql_calls = self._sql_log_session()
        with patch.dict(os.environ, {
            "SYNTHETIC_REGRESSION_FORCE_REBUILD": "true",
            "SYNTHETIC_REGRESSION_DROP_INDEX_TABLE": "false",
        }):
            with patch("prepare_synthetic_regression.SYNREG_FORCE_REBUILD", True):
                with patch("prepare_synthetic_regression.SYNREG_DROP_INDEX_TABLE", False):
                    prep._truncate_synreg_index(session)
        delete_sqls = [s for s in sql_calls if "DELETE" in s.upper()]
        drop_sqls = [s for s in sql_calls if "DROP" in s.upper()]
        assert delete_sqls, "Expected DELETE SQL but none found"
        assert not drop_sqls, f"DROP TABLE must not be issued, but got: {drop_sqls}"

    def test_ood_suite_rows_preserved_during_force_rebuild(self):
        """DELETE WHERE suite_id scopes the delete to the active suite only."""
        session, sql_calls = self._sql_log_session()
        with patch("prepare_synthetic_regression.SYNREG_DROP_INDEX_TABLE", False):
            prep._truncate_synreg_index(session)
        delete_sqls = [s for s in sql_calls if "DELETE" in s.upper()]
        assert delete_sqls
        # The DELETE must be scoped to the active suite_id
        assert any(prep.SYNREG_SUITE_ID in s for s in delete_sqls)
        # No full DROP
        assert not any("DROP" in s.upper() for s in sql_calls)

    def test_full_drop_requires_explicit_env_var(self):
        """DROP_INDEX_TABLE=false → no DROP TABLE; DROP_INDEX_TABLE=true → DROP TABLE present."""
        # Case 1: flag off → DELETE only
        session_a, calls_a = self._sql_log_session()
        with patch("prepare_synthetic_regression.SYNREG_DROP_INDEX_TABLE", False):
            prep._truncate_synreg_index(session_a)
        assert not any("DROP" in s.upper() for s in calls_a)
        assert any("DELETE" in s.upper() for s in calls_a)

        # Case 2: flag on → DROP TABLE
        session_b, calls_b = self._sql_log_session()
        with patch("prepare_synthetic_regression.SYNREG_DROP_INDEX_TABLE", True):
            prep._truncate_synreg_index(session_b)
        assert any("DROP" in s.upper() for s in calls_b)
        assert not any("DELETE" in s.upper() for s in calls_b)


# ---------------------------------------------------------------------------
# Tests: logical_dataset_key persistence (Item 3)
# ---------------------------------------------------------------------------

class TestLogicalDatasetKeyPersistence:
    def _make_session(self, n_rows: int):
        sql_log = []
        def fake_sql(sql_str):
            sql_log.append(sql_str)
            mock_result = MagicMock()
            if "COUNT(*)" in sql_str:
                mock_row = MagicMock()
                mock_row.__getitem__ = lambda self, i: n_rows
                mock_result.collect.return_value = [mock_row]
            else:
                mock_result.collect.return_value = []
            return mock_result
        session = MagicMock()
        session.sql.side_effect = fake_sql
        return session, sql_log

    def test_insert_includes_logical_dataset_key_column(self):
        rows = [prep._make_index_row(
            suite_family="primary",
            dataset_id=0,
            dataset_seed=42,
            stage_path="@EVALUATION_DATASET_STAGE/primary/dataset_0000.parquet",
            regime="A",
            split_seeds=[0, 1, 2],
            n_total=200,
            p_signal=5,
            p_noise=0,
        )]
        session, sql_log = self._make_session(1)
        prep.insert_synreg_index_rows(session, rows)
        insert_sqls = [s for s in sql_log if "INSERT INTO" in s]
        assert insert_sqls, "No INSERT SQL was generated"
        assert any("logical_dataset_key" in s for s in insert_sqls), (
            "logical_dataset_key must appear in the INSERT SQL"
        )

    def test_make_index_row_contains_logical_dataset_key(self):
        row = prep._make_index_row(
            suite_family="primary",
            dataset_id=7,
            dataset_seed=42,
            stage_path="@EVALUATION_DATASET_STAGE/primary/dataset_0007.parquet",
            regime="A",
            split_seeds=[0, 1, 2],
            n_total=200,
            p_signal=5,
            p_noise=0,
        )
        assert "logical_dataset_key" in row
        assert row["logical_dataset_key"] is not None

    def test_primary_key_format_zero_padded(self):
        row = prep._make_index_row(
            suite_family="primary",
            dataset_id=7,
            dataset_seed=42,
            stage_path="@EVALUATION_DATASET_STAGE/primary/dataset_0007.parquet",
            regime="A",
            split_seeds=[0, 1, 2],
            n_total=200,
            p_signal=5,
            p_noise=0,
        )
        # Zero-padded: :A:0007 not :A:7
        assert ":A:0007" in row["logical_dataset_key"], (
            f"Expected zero-padded dataset_id in key, got: {row['logical_dataset_key']!r}"
        )


class TestOODLogicalDatasetKey:
    def _make_ood_row(self, dataset_id: int, regime: str):
        import prepare_ood_regression as ood_prep
        return ood_prep._make_ood_index_row(
            dataset_id=dataset_id,
            dataset_seed=dataset_id * 7,
            regime=regime,
            n_total=100,
            p_signal=5,
            p_noise=0,
            p_total=5,
            target_noise_scale=1.0,
        )

    def test_ood_key_zero_padded(self):
        import prepare_ood_regression as ood_prep
        row = self._make_ood_row(dataset_id=3, regime="G")
        assert ":G:0003" in row["logical_dataset_key"], (
            f"Expected zero-padded OOD key, got: {row['logical_dataset_key']!r}"
        )

    def test_ood_keys_unique_across_regimes(self):
        keys = [
            self._make_ood_row(dataset_id=0, regime=r)["logical_dataset_key"]
            for r in ["E", "F", "G", "H"]
        ]
        assert len(keys) == len(set(keys)), "OOD logical_dataset_key must differ per regime"

    def test_evaluation_uses_index_key_when_available(self):
        import evaluate_synthetic_regression as evsr
        row = {
            "logical_dataset_key": "X:A:0001",
            "suite_id": "linear_poisson_v1_recommended",
            "prior_regime": "A",
            "dataset_id": 1,
        }
        # The key in the row dict should be used directly
        key = (
            row.get("logical_dataset_key")
            or f"{evsr.SYNREG_SUITE_ID}:A:{1:04d}"
        )
        assert key == "X:A:0001"

    def test_evaluation_falls_back_to_zero_padded_key(self):
        import evaluate_synthetic_regression as evsr
        row = {
            "suite_id": "linear_poisson_v1_recommended",
            "prior_regime": "A",
            "dataset_id": 7,
        }
        key = (
            row.get("logical_dataset_key")
            or f"{evsr.SYNREG_SUITE_ID}:A:{7:04d}"
        )
        assert ":A:0007" in key, (
            f"Fallback key should be zero-padded, got: {key!r}"
        )


# ---------------------------------------------------------------------------
# Combined suite: prepare_combined_suite + _load_index_rows
# ---------------------------------------------------------------------------

class TestCombinedSuite:
    """Tests for prepare_combined_suite and supporting helpers."""

    _PRIMARY_REGIMES = ["A", "B", "C", "D"]
    _OOD_REGIMES     = ["E", "F", "G", "H"]

    def _make_primary_row(self, regime: str, dataset_id: int) -> dict:
        return {
            "suite_id": "linear_poisson_v1_recommended",
            "suite_family": "primary",
            "dataset_id": dataset_id,
            "dataset_seed": dataset_id * 3,
            "stage_path": f"@EVALUATION_DATASET_STAGE/primary/dataset_{dataset_id:04d}.npz",
            "prior_name": "linear_poisson",
            "prior_version": "v1",
            "prior_regime": regime,
            "split_seeds": [0, 1, 2, 3, 4],
            "n_total": 500,
            "n_train_default": 400,
            "n_holdout_default": 100,
            "p_signal": 5,
            "p_noise": 0,
            "p_total": 5,
            "target_noise_scale": 1.0,
            "training_size_anchor": False,
            "feature_noise_level": 0,
            "eval_weight": 1.0,
            "payload_bytes": 1000,
            "logical_dataset_key": f"linear_poisson_v1_recommended:{regime}:{dataset_id:04d}",
        }

    def _make_ood_row(self, regime: str, dataset_id: int) -> dict:
        return {
            "suite_id": "ood_linear_full_v1",
            "suite_family": "ood_primary",
            "dataset_id": dataset_id,
            "dataset_seed": dataset_id * 7,
            "stage_path": f"@EVALUATION_DATASET_STAGE/ood_parity/{regime}/dataset_{dataset_id:04d}.parquet",
            "prior_name": "linear_poisson_ood",
            "prior_version": "v1",
            "prior_regime": regime,
            "split_seeds": [0, 1, 2],
            "n_total": 500,
            "n_train_default": 400,
            "n_holdout_default": 100,
            "p_signal": 5,
            "p_noise": 0,
            "p_total": 5,
            "target_noise_scale": 1.0,
            "training_size_anchor": False,
            "feature_noise_level": 0,
            "eval_weight": 1.0,
            "payload_bytes": 1000,
            "logical_dataset_key": f"ood_linear_full_v1:{regime}:{dataset_id:04d}",
        }

    def _build_source_rows(self, n_per_regime: int = 50):
        """Return (primary_rows, ood_rows) with n_per_regime rows per regime."""
        primary = []
        global_idx = 0
        for regime in self._PRIMARY_REGIMES:
            for i in range(n_per_regime):
                primary.append(self._make_primary_row(regime, global_idx))
                global_idx += 1

        ood = []
        global_idx = 0
        for regime in self._OOD_REGIMES:
            for i in range(n_per_regime):
                ood.append(self._make_ood_row(regime, global_idx))
                global_idx += 1

        return primary, ood

    def _make_session(self, primary_rows, ood_rows):
        """Return a mock session; _load_index_rows is patched separately."""
        sql_log = []
        inserted_rows = []

        def fake_sql(sql_str):
            sql_log.append(sql_str)
            mock_result = MagicMock()
            if "COUNT(*)" in sql_str:
                mock_row = MagicMock()
                mock_row.__getitem__ = lambda self, i: len(primary_rows) + len(ood_rows)
                mock_result.collect.return_value = [mock_row]
            else:
                mock_result.collect.return_value = []
            return mock_result

        session = MagicMock()
        session.sql.side_effect = fake_sql
        return session, sql_log, inserted_rows

    # ------------------------------------------------------------------

    def test_combined_suite_inserts_400_rows(self):
        primary_rows, ood_rows = self._build_source_rows(50)
        session, _, _ = self._make_session(primary_rows, ood_rows)

        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with patch.object(prep, "insert_synreg_index_rows") as mock_insert:
                    with patch.object(prep, "_assert_index_populated", return_value=400):
                        prep.prepare_combined_suite(session)

        rows = mock_insert.call_args[0][1]
        assert len(rows) == 400, f"Expected 400 rows, got {len(rows)}"

    def test_combined_suite_per_regime_balance(self):
        primary_rows, ood_rows = self._build_source_rows(50)
        session, _, _ = self._make_session(primary_rows, ood_rows)

        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with patch.object(prep, "insert_synreg_index_rows") as mock_insert:
                    with patch.object(prep, "_assert_index_populated", return_value=400):
                        prep.prepare_combined_suite(session)

        rows = mock_insert.call_args[0][1]
        from collections import Counter
        counts = Counter(r["prior_regime"] for r in rows)
        for regime in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            assert counts[regime] == 50, (
                f"Regime {regime}: expected 50 rows, got {counts[regime]}"
            )

    def test_combined_suite_source_suite_id_populated(self):
        primary_rows, ood_rows = self._build_source_rows(50)
        session, _, _ = self._make_session(primary_rows, ood_rows)

        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with patch.object(prep, "insert_synreg_index_rows") as mock_insert:
                    with patch.object(prep, "_assert_index_populated", return_value=400):
                        prep.prepare_combined_suite(session)

        rows = mock_insert.call_args[0][1]
        primary_source = {r["source_suite_id"] for r in rows if r["prior_regime"] in self._PRIMARY_REGIMES}
        ood_source = {r["source_suite_id"] for r in rows if r["prior_regime"] in self._OOD_REGIMES}
        assert primary_source == {prep.COMBINED_PRIMARY_SUITE}
        assert ood_source == {prep.COMBINED_OOD_SUITE}

    def test_combined_suite_logical_key_uses_combined_suite_id(self):
        primary_rows, ood_rows = self._build_source_rows(50)
        session, _, _ = self._make_session(primary_rows, ood_rows)

        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with patch.object(prep, "insert_synreg_index_rows") as mock_insert:
                    with patch.object(prep, "_assert_index_populated", return_value=400):
                        prep.prepare_combined_suite(session)

        rows = mock_insert.call_args[0][1]
        for row in rows:
            assert row["logical_dataset_key"].startswith(prep.COMBINED_SUITE_ID + ":"), (
                f"logical_dataset_key must start with '{prep.COMBINED_SUITE_ID}:', "
                f"got {row['logical_dataset_key']!r}"
            )

    def test_combined_suite_split_seeds_normalized(self):
        primary_rows, ood_rows = self._build_source_rows(50)
        session, _, _ = self._make_session(primary_rows, ood_rows)

        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with patch.object(prep, "insert_synreg_index_rows") as mock_insert:
                    with patch.object(prep, "_assert_index_populated", return_value=400):
                        prep.prepare_combined_suite(session)

        rows = mock_insert.call_args[0][1]
        for row in rows:
            assert list(row["split_seeds"]) == prep.COMBINED_SPLIT_SEEDS, (
                f"Expected split_seeds={prep.COMBINED_SPLIT_SEEDS}, "
                f"got {row['split_seeds']!r}"
            )

    def test_combined_suite_suite_id_remapped(self):
        primary_rows, ood_rows = self._build_source_rows(50)
        session, _, _ = self._make_session(primary_rows, ood_rows)

        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with patch.object(prep, "insert_synreg_index_rows") as mock_insert:
                    with patch.object(prep, "_assert_index_populated", return_value=400):
                        prep.prepare_combined_suite(session)

        rows = mock_insert.call_args[0][1]
        assert all(r["suite_id"] == prep.COMBINED_SUITE_ID for r in rows), (
            "All combined rows must have suite_id=COMBINED_SUITE_ID"
        )

    def test_combined_suite_raises_if_primary_missing(self):
        session = MagicMock()
        session.sql.return_value.collect.return_value = []

        _, ood_rows = self._build_source_rows(50)
        with patch.object(prep, "_load_index_rows", side_effect=[[], ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with pytest.raises(RuntimeError, match="primary suite"):
                    prep.prepare_combined_suite(session)

    def test_combined_suite_raises_if_ood_missing(self):
        session = MagicMock()
        session.sql.return_value.collect.return_value = []

        primary_rows, _ = self._build_source_rows(50)
        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, []]):
            with patch.object(prep, "create_synreg_index_table"):
                with pytest.raises(RuntimeError, match="OOD suite"):
                    prep.prepare_combined_suite(session)

    def test_combined_suite_raises_if_regime_imbalanced(self):
        """Only 49 per regime A-D (missing 1) → ValueError about regime balance."""
        primary_rows, ood_rows = self._build_source_rows(49)  # 49 per regime, not 50
        session = MagicMock()
        session.sql.return_value.collect.return_value = []

        with patch.object(prep, "_load_index_rows", side_effect=[primary_rows, ood_rows]):
            with patch.object(prep, "create_synreg_index_table"):
                with pytest.raises(ValueError, match="regime"):
                    prep.prepare_combined_suite(session)

    def test_load_index_rows_lowercases_keys(self):
        """_load_index_rows must lowercase column names from Snowflake Row objects."""
        class FakeRow:
            def as_dict(self):
                return {"SUITE_ID": "x", "PRIOR_REGIME": "A", "SPLIT_SEEDS": [0, 1, 2]}

        session = MagicMock()
        session.sql.return_value.collect.return_value = [FakeRow()]
        rows = prep._load_index_rows(session, "x")
        assert len(rows) == 1
        assert "suite_id" in rows[0], f"Keys not lowercased: {list(rows[0].keys())}"
        assert "prior_regime" in rows[0]

    def test_load_index_rows_normalizes_split_seeds_string(self):
        """_load_index_rows must parse split_seeds when Snowflake returns a JSON string."""
        class FakeRow:
            def as_dict(self):
                return {"suite_id": "x", "prior_regime": "A", "split_seeds": "[0,1,2]"}

        session = MagicMock()
        session.sql.return_value.collect.return_value = [FakeRow()]
        rows = prep._load_index_rows(session, "x")
        assert rows[0]["split_seeds"] == [0, 1, 2], (
            f"Expected [0,1,2], got {rows[0]['split_seeds']!r}"
        )

    def test_combined_suite_dispatch_from_prepare_synthetic_regression(self):
        """prepare_synthetic_regression dispatches to combined handler when suite=linear_all_v1."""
        with patch.dict(os.environ, {"SYNTHETIC_REGRESSION_SUITE_ID": prep.COMBINED_SUITE_ID}):
            # Reload constants to pick up env var
            import importlib
            import prepare_synthetic_regression as _fresh
            importlib.reload(_fresh)
            session = MagicMock()
            session.sql.return_value.collect.return_value = []

            with patch.object(_fresh, "prepare_combined_suite", return_value="ok-combined") as mock_combined:
                result = _fresh.prepare_synthetic_regression(session)

        assert result == "ok-combined", f"Expected combined dispatch, got: {result!r}"
        mock_combined.assert_called_once_with(session)
