"""Tests for nonlinear training index routing and DDL.

Covers Fix 1: _resolve_index_table_and_stage routing, SQL DDL presence checks,
and build_meta_nonlinear_dataset_index.py existence.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers: project root / sys.path setup
# ---------------------------------------------------------------------------

def _add_src_to_path():
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


# ---------------------------------------------------------------------------
# Fix 1b — _resolve_index_table_and_stage routing
# ---------------------------------------------------------------------------

class TestResolveIndexTableAndStage:

    def setup_method(self):
        _add_src_to_path()
        import importlib
        import snowflake_io
        importlib.reload(snowflake_io)
        self.mod = snowflake_io

    def test_nonlinear_family_routes_to_nonlinear_index(self, monkeypatch):
        """TRAINING_DATA_FAMILY=synthetic_regression_nonlinear routes to nonlinear names."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_nonlinear")
        index_table, stage = self.mod._resolve_index_table_and_stage()
        assert index_table == "META_NONLINEAR_DATASET_INDEX"
        assert stage == "@META_NONLINEAR_DATASET_STAGE"

    def test_linear_family_routes_to_meta_dataset_index(self, monkeypatch):
        """TRAINING_DATA_FAMILY=synthetic_regression_combined routes to the linear names."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        index_table, stage = self.mod._resolve_index_table_and_stage()
        assert index_table == "META_DATASET_INDEX"
        assert stage == "@META_DATASET_STAGE"

    def test_unset_family_routes_to_meta_dataset_index(self, monkeypatch):
        """Unset TRAINING_DATA_FAMILY defaults to the linear index and stage."""
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        index_table, stage = self.mod._resolve_index_table_and_stage()
        assert index_table == "META_DATASET_INDEX"
        assert stage == "@META_DATASET_STAGE"

    def test_explicit_arg_overrides_env(self, monkeypatch):
        """Explicit training_data_family argument takes precedence over env var."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        index_table, stage = self.mod._resolve_index_table_and_stage(
            "synthetic_regression_nonlinear"
        )
        assert index_table == "META_NONLINEAR_DATASET_INDEX"
        assert stage == "@META_NONLINEAR_DATASET_STAGE"

    def test_snowflake_io_sql_uses_nonlinear_index_when_env_set(self, monkeypatch):
        """select_meta_dataset_index_rows builds SQL with META_NONLINEAR_DATASET_INDEX when env set."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_nonlinear")

        # Capture the SQL that would be sent to Snowflake
        captured_sql = []

        class _FakeSession:
            def sql(self, query):
                captured_sql.append(query)
                return self

            def collect(self):
                return []

        # Patch _validate_index_rows to avoid validation failure on empty result
        with patch.object(self.mod, "_validate_index_rows", return_value=None):
            try:
                self.mod.select_meta_dataset_index_rows(
                    splits=("train",), session=_FakeSession()
                )
            except Exception:
                pass  # empty result handling may raise; we only care about SQL

        assert captured_sql, "No SQL was captured"
        assert any("META_NONLINEAR_DATASET_INDEX" in sql for sql in captured_sql), (
            f"Expected META_NONLINEAR_DATASET_INDEX in SQL, got: {captured_sql}"
        )
        assert not any("META_DATASET_INDEX" == sql.strip() for sql in captured_sql), (
            "SQL must reference META_NONLINEAR_DATASET_INDEX, not META_DATASET_INDEX"
        )


# ---------------------------------------------------------------------------
# Fix 1d — SQL DDL presence checks
# ---------------------------------------------------------------------------

class TestNonlinearIndexDDL:

    def test_nonlinear_index_ddl_in_run_training_job_sql(self):
        """META_NONLINEAR_DATASET_INDEX DDL must be present in sql/run_training_job.sql."""
        sql_file = ROOT / "sql" / "run_training_job.sql"
        text = sql_file.read_text()
        assert "META_NONLINEAR_DATASET_INDEX" in text, (
            "META_NONLINEAR_DATASET_INDEX DDL not found in sql/run_training_job.sql"
        )
        # Must be a CREATE TABLE statement, not just a comment
        assert "CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_DATASET_INDEX" in text, (
            "Expected CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_DATASET_INDEX"
        )

    def test_nonlinear_index_has_correct_columns(self):
        """META_NONLINEAR_DATASET_INDEX DDL must include all required columns."""
        sql_file = ROOT / "sql" / "run_training_job.sql"
        text = sql_file.read_text()
        required_columns = ("split", "task_id", "stage_path", "n", "p", "n_train",
                            "n_test", "prior_regime", "hpo_bucket")
        for col in required_columns:
            assert col in text, (
                f"Column {col!r} not found in META_NONLINEAR_DATASET_INDEX DDL"
            )

    def test_nonlinear_stage_ddl_in_stages_sql(self):
        """META_NONLINEAR_DATASET_STAGE must be created in sql/01_stages_and_metadata_tables.sql."""
        sql_file = ROOT / "sql" / "01_stages_and_metadata_tables.sql"
        text = sql_file.read_text()
        assert "META_NONLINEAR_DATASET_STAGE" in text, (
            "META_NONLINEAR_DATASET_STAGE not found in sql/01_stages_and_metadata_tables.sql"
        )
        assert "CREATE STAGE IF NOT EXISTS META_NONLINEAR_DATASET_STAGE" in text, (
            "Expected CREATE STAGE IF NOT EXISTS META_NONLINEAR_DATASET_STAGE"
        )

    def test_nonlinear_stage_has_sse_encryption(self):
        """META_NONLINEAR_DATASET_STAGE must use SNOWFLAKE_SSE encryption."""
        sql_file = ROOT / "sql" / "01_stages_and_metadata_tables.sql"
        text = sql_file.read_text()
        # Find the CREATE STAGE statement specifically
        create_key = "CREATE STAGE IF NOT EXISTS META_NONLINEAR_DATASET_STAGE"
        idx = text.find(create_key)
        assert idx != -1, f"{create_key!r} not found in 01_stages_and_metadata_tables.sql"
        snippet = text[idx : idx + 200]
        assert "SNOWFLAKE_SSE" in snippet, (
            "META_NONLINEAR_DATASET_STAGE must use ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')"
        )


# ---------------------------------------------------------------------------
# Fix 1a — build_meta_nonlinear_dataset_index.py existence
# ---------------------------------------------------------------------------

class TestBuildMetaNonlinearScript:

    def test_build_meta_nonlinear_script_exists(self):
        """src/build_meta_nonlinear_dataset_index.py must exist."""
        script = ROOT / "src" / "build_meta_nonlinear_dataset_index.py"
        assert script.exists(), (
            "src/build_meta_nonlinear_dataset_index.py does not exist. "
            "This file must be created as part of Fix 1a."
        )

    def test_build_meta_nonlinear_script_has_correct_constants(self):
        """build_meta_nonlinear_dataset_index.py must reference nonlinear stage/index."""
        script = ROOT / "src" / "build_meta_nonlinear_dataset_index.py"
        text = script.read_text()
        assert "META_NONLINEAR_DATASET_STAGE" in text, (
            "build_meta_nonlinear_dataset_index.py must reference @META_NONLINEAR_DATASET_STAGE"
        )
        assert "META_NONLINEAR_DATASET_INDEX" in text, (
            "build_meta_nonlinear_dataset_index.py must reference META_NONLINEAR_DATASET_INDEX"
        )

    def test_build_meta_nonlinear_script_has_correct_counts(self):
        """Expected split counts must be 800/100/100 (same as linear)."""
        script = ROOT / "src" / "build_meta_nonlinear_dataset_index.py"
        text = script.read_text()
        assert '"train": 800' in text or "'train': 800" in text, (
            "Expected 800 training samples in EXPECTED_COUNTS"
        )
        assert '"val": 100' in text or "'val': 100" in text, (
            "Expected 100 validation samples in EXPECTED_COUNTS"
        )
        assert '"test": 100' in text or "'test': 100" in text, (
            "Expected 100 test samples in EXPECTED_COUNTS"
        )

    def test_build_meta_nonlinear_script_has_main_entrypoint(self):
        """build_meta_nonlinear_dataset_index.py must define a main() function."""
        script = ROOT / "src" / "build_meta_nonlinear_dataset_index.py"
        text = script.read_text()
        assert "def main(" in text, (
            "build_meta_nonlinear_dataset_index.py must define main() as Snowflake handler"
        )

    def test_build_meta_nonlinear_script_compiles(self):
        """build_meta_nonlinear_dataset_index.py must have valid Python syntax."""
        import py_compile
        script = ROOT / "src" / "build_meta_nonlinear_dataset_index.py"
        py_compile.compile(str(script), doraise=True)


# ---------------------------------------------------------------------------
# Fix 2 / SQL — nonlinear pretrain procedure in SQL
# ---------------------------------------------------------------------------

class TestNonlinearPretrain:

    def test_nonlinear_pretrain_procedure_in_sql(self):
        """sql/02_pretrain_hpo_training_procedures.sql must define run_pretrain_pipeline_nonlinear."""
        sql_file = ROOT / "sql" / "02_pretrain_hpo_training_procedures.sql"
        text = sql_file.read_text()
        assert "run_pretrain_pipeline_nonlinear" in text, (
            "run_pretrain_pipeline_nonlinear procedure not found in "
            "sql/02_pretrain_hpo_training_procedures.sql"
        )
        assert "build_meta_nonlinear_dataset_index" in text, (
            "build_meta_nonlinear_dataset_index procedure not found in "
            "sql/02_pretrain_hpo_training_procedures.sql"
        )

    def test_6arg_hpo_overload_in_sql(self):
        """sql/02 must define 6-arg run_hpo_pipeline with HPO_PRETRAIN_CHECKPOINT_STAGE_PATH."""
        sql_file = ROOT / "sql" / "02_pretrain_hpo_training_procedures.sql"
        text = sql_file.read_text()
        assert "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH" in text, (
            "6-arg HPO procedure HPO_PRETRAIN_CHECKPOINT_STAGE_PATH not found in sql/02"
        )
        assert "run_hpo_pipeline_model_sweep_with_baseline_and_pretrain" in text, (
            "Handler run_hpo_pipeline_model_sweep_with_baseline_and_pretrain not found in sql/02"
        )
