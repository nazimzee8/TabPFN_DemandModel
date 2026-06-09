"""Tests for META_REGRESSION_DATASET_INDEX schema consistency (Area 2).

Checks that:
- build_meta_dataset_index.py OPTIONAL_METADATA_DEFAULTS covers all 14 MODEL4 columns
- The SQL DDL in run_training_job.sql declares all 23 expected columns
- The migration script adds all 14 MODEL4 columns (including the 4 previously missing)
"""

import os
import re
import sys
from unittest.mock import MagicMock
import pytest

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

SQL_DIR = os.path.join(os.path.dirname(__file__), "..", "sql")

# The 14 MODEL4 metadata columns that must be present in the linear index
MODEL4_METADATA_COLUMNS = {
    "profile",
    "suite_id",
    "p_signal",
    "p_noise",
    "p_total",
    "active_s",
    "sparsity_ratio",
    "covariance_type",
    "rho",
    "target_noise_scale",
    "feature_noise_level",
    "sample_complexity_bucket",
    "has_noise_features",
    "has_feature_noise",
}

# The 9 runtime columns
RUNTIME_COLUMNS = {
    "split",
    "task_id",
    "stage_path",
    "n",
    "p",
    "n_train",
    "n_test",
    "prior_regime",
    "hpo_bucket",
}

ALL_EXPECTED_COLUMNS = RUNTIME_COLUMNS | MODEL4_METADATA_COLUMNS


class TestBuilderSchema:
    def test_optional_defaults_covers_all_model4_columns(self):
        import build_meta_dataset_index as m
        defaults_keys = set(m.OPTIONAL_METADATA_DEFAULTS.keys())
        missing = MODEL4_METADATA_COLUMNS - defaults_keys
        assert not missing, f"OPTIONAL_METADATA_DEFAULTS missing MODEL4 columns: {missing}"

    def test_write_index_schema_covers_all_columns(self):
        """The StructType in _write_index must include all 23 columns."""
        import inspect
        import build_meta_dataset_index as m
        source = inspect.getsource(m._write_index)
        for col in ALL_EXPECTED_COLUMNS:
            assert col in source, f"Column {col!r} missing from _write_index StructType"


class TestSQLDDL:
    @pytest.fixture
    def run_training_sql(self):
        path = os.path.join(SQL_DIR, "synthetic_linear_pipeline.sql")
        with open(path) as f:
            return f.read()

    def test_meta_dataset_index_ddl_has_all_columns(self, run_training_sql):
        # Extract the CREATE TABLE block for META_REGRESSION_DATASET_INDEX
        match = re.search(
            r"CREATE TRANSIENT TABLE IF NOT EXISTS META_REGRESSION_DATASET_INDEX\s*\((.+?)\)",
            run_training_sql,
            re.DOTALL | re.IGNORECASE,
        )
        assert match, "META_REGRESSION_DATASET_INDEX CREATE TABLE not found in run_training_job.sql"
        ddl_block = match.group(1).lower()
        for col in ALL_EXPECTED_COLUMNS:
            assert col.lower() in ddl_block, (
                f"Column {col!r} missing from META_REGRESSION_DATASET_INDEX DDL in run_training_job.sql"
            )

    def test_suite_id_is_declared_nullable(self, run_training_sql):
        """suite_id must be present but must NOT have NOT NULL constraint."""
        match = re.search(
            r"CREATE TRANSIENT TABLE IF NOT EXISTS META_REGRESSION_DATASET_INDEX\s*\((.+?)\)",
            run_training_sql,
            re.DOTALL | re.IGNORECASE,
        )
        assert match
        ddl_block = match.group(1).lower()
        assert "suite_id" in ddl_block
        # Verify NOT NULL is not applied to suite_id line
        for line in ddl_block.splitlines():
            if "suite_id" in line:
                assert "not null" not in line, "suite_id must be nullable"

    def test_one_arg_overload_exists(self, run_training_sql):
        """A 1-arg build_meta_dataset_index(EXPECTED_TOTAL INTEGER) overload must exist."""
        assert "build_meta_dataset_index_with_total" in run_training_sql or (
            "EXPECTED_TOTAL INTEGER" in run_training_sql
            and "build_meta_dataset_index" in run_training_sql
        )


class TestMigrationScript:
    @pytest.fixture
    def migration_sql(self):
        path = os.path.join(SQL_DIR, "migrate_meta_dataset_index_v2.sql")
        with open(path) as f:
            return f.read()

    def test_migration_adds_all_14_model4_columns(self, migration_sql):
        content = migration_sql.lower()
        for col in MODEL4_METADATA_COLUMNS:
            assert col.lower() in content, (
                f"Migration script missing ALTER TABLE for column {col!r}"
            )
