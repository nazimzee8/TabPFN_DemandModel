"""Tests for dynamic dataset-index count parsing and split derivation.

Covers Area 1: _parse_expected_total, _derive_split_counts, and stage-count
validation logic in both the linear and nonlinear index builders.
"""

import os
import sys
from unittest.mock import MagicMock
import pytest

# Allow importing from src/ without an installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock snowflake.snowpark before builder modules are imported (they use it at module level).
_sf_mock = MagicMock()
_sf_mock.BooleanType = type("BooleanType", (), {})
_sf_mock.FloatType   = type("FloatType",   (), {})
_sf_mock.IntegerType = type("IntegerType", (), {})
_sf_mock.StringType  = type("StringType",  (), {})
_sf_mock.StructField = type("StructField", (), {})
_sf_mock.StructType  = type("StructType",  (), {})
sys.modules.setdefault("snowflake",                   MagicMock())
sys.modules.setdefault("snowflake.snowpark",          _sf_mock)
sys.modules.setdefault("snowflake.snowpark.context",  _sf_mock)
sys.modules.setdefault("snowflake.snowpark.types",    _sf_mock)


# ---------------------------------------------------------------------------
# Helpers under test (import after path fix)
# ---------------------------------------------------------------------------

def _import_linear():
    import importlib
    import build_meta_dataset_index as m
    return m


def _import_nonlinear():
    import importlib
    import build_meta_nonlinear_dataset_index as m
    return m


# ---------------------------------------------------------------------------
# _parse_expected_total
# ---------------------------------------------------------------------------

class TestParseExpectedTotal:
    def test_default_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("META_DATASET_EXPECTED_TOTAL", raising=False)
        m = _import_linear()
        assert m._parse_expected_total("META_DATASET_EXPECTED_TOTAL", default=1000) == 1000

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("META_DATASET_EXPECTED_TOTAL", "500")
        m = _import_linear()
        assert m._parse_expected_total("META_DATASET_EXPECTED_TOTAL", default=1000) == 500

    def test_custom_default(self, monkeypatch):
        monkeypatch.delenv("META_DATASET_EXPECTED_TOTAL", raising=False)
        m = _import_linear()
        assert m._parse_expected_total("META_DATASET_EXPECTED_TOTAL", default=2000) == 2000

    def test_nonlinear_env_var(self, monkeypatch):
        monkeypatch.setenv("META_NONLINEAR_DATASET_EXPECTED_TOTAL", "800")
        m = _import_nonlinear()
        assert m._parse_expected_total("META_NONLINEAR_DATASET_EXPECTED_TOTAL", default=1000) == 800


# ---------------------------------------------------------------------------
# _derive_split_counts
# ---------------------------------------------------------------------------

class TestDeriveSplitCounts:
    def test_default_1000_backward_compat(self):
        m = _import_linear()
        counts = m._derive_split_counts(1000)
        assert counts == {"train": 800, "val": 100, "test": 100}

    def test_total_is_preserved(self):
        m = _import_linear()
        for total in [100, 500, 1000, 2000]:
            counts = m._derive_split_counts(total)
            assert counts["train"] + counts["val"] + counts["test"] == total

    def test_formula(self):
        m = _import_linear()
        counts = m._derive_split_counts(500)
        assert counts["train"] == int(0.8 * 500)   # 400
        assert counts["val"]   == int(0.1 * 500)   # 50
        assert counts["test"]  == 500 - 400 - 50   # 50

    def test_nonlinear_builder_has_same_formula(self):
        lin = _import_linear()
        nl  = _import_nonlinear()
        for total in [500, 1000]:
            assert lin._derive_split_counts(total) == nl._derive_split_counts(total)


# ---------------------------------------------------------------------------
# Module-level EXPECTED_COUNTS respects env var
# ---------------------------------------------------------------------------

class TestModuleLevelExpectedCounts:
    def test_linear_default(self, monkeypatch):
        monkeypatch.delenv("META_DATASET_EXPECTED_TOTAL", raising=False)
        # Re-import so module-level code runs with fresh env
        import importlib
        import build_meta_dataset_index
        importlib.reload(build_meta_dataset_index)
        assert build_meta_dataset_index.EXPECTED_COUNTS == {"train": 800, "val": 100, "test": 100}

    def test_nonlinear_default(self, monkeypatch):
        monkeypatch.delenv("META_NONLINEAR_DATASET_EXPECTED_TOTAL", raising=False)
        import importlib
        import build_meta_nonlinear_dataset_index
        importlib.reload(build_meta_nonlinear_dataset_index)
        assert build_meta_nonlinear_dataset_index.EXPECTED_COUNTS == {"train": 800, "val": 100, "test": 100}
