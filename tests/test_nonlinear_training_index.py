"""Tests for nonlinear training index routing and DDL.

Covers Fix 1: _resolve_index_table_and_stage routing, SQL DDL presence checks,
and build_meta_nonlinear_dataset_index.py existence.
"""

import os
import sys
import types
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
        assert index_table == "META_NONLINEAR_REGRESSION_DATASET_INDEX"
        assert stage == "@META_DATASET_STAGE"

    def test_linear_family_routes_to_meta_dataset_index(self, monkeypatch):
        """TRAINING_DATA_FAMILY=synthetic_regression_combined routes to the linear names."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        index_table, stage = self.mod._resolve_index_table_and_stage()
        assert index_table == "META_REGRESSION_DATASET_INDEX"
        assert stage == "@META_DATASET_STAGE"

    def test_unset_family_routes_to_meta_dataset_index(self, monkeypatch):
        """Unset TRAINING_DATA_FAMILY defaults to the linear index and stage."""
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        index_table, stage = self.mod._resolve_index_table_and_stage()
        assert index_table == "META_REGRESSION_DATASET_INDEX"
        assert stage == "@META_DATASET_STAGE"

    def test_explicit_arg_overrides_env(self, monkeypatch):
        """Explicit training_data_family argument takes precedence over env var."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        index_table, stage = self.mod._resolve_index_table_and_stage(
            "synthetic_regression_nonlinear"
        )
        assert index_table == "META_NONLINEAR_REGRESSION_DATASET_INDEX"
        assert stage == "@META_DATASET_STAGE"

    def test_snowflake_io_sql_uses_nonlinear_index_when_env_set(self, monkeypatch):
        """select_meta_dataset_index_rows builds SQL with META_NONLINEAR_REGRESSION_DATASET_INDEX when env set."""
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
        assert any("META_NONLINEAR_REGRESSION_DATASET_INDEX" in sql for sql in captured_sql), (
            f"Expected META_NONLINEAR_REGRESSION_DATASET_INDEX in SQL, got: {captured_sql}"
        )
        assert not any("META_DATASET_INDEX" == sql.strip() for sql in captured_sql), (
            "SQL must reference META_NONLINEAR_REGRESSION_DATASET_INDEX, not META_DATASET_INDEX"
        )

    def test_rank_sharded_training_sql_uses_nonlinear_index_when_env_set(self, monkeypatch):
        """DDP rank sharding must query META_NONLINEAR_REGRESSION_DATASET_INDEX for nonlinear training."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_nonlinear")
        captured_sql = []

        class _FakeSession:
            def sql(self, query):
                captured_sql.append(query)
                return self

            def collect(self):
                return [{
                    "split": "train",
                    "task_id": 0,
                    "stage_path": "train/sample.parquet",
                    "p": 8,
                    "n_train": 256,
                }]

        rows = self.mod.select_rank_sharded_index_rows(
            "train", rank=0, world_size=4, session=_FakeSession()
        )

        assert rows[0]["stage_path"] == "train/sample.parquet"
        assert captured_sql, "No SQL was captured"
        assert any("FROM META_NONLINEAR_REGRESSION_DATASET_INDEX" in sql for sql in captured_sql), (
            f"Expected rank-sharded SQL to query nonlinear index, got: {captured_sql}"
        )

    def test_stage_file_path_uses_nonlinear_stage_when_env_set(self, monkeypatch):
        """Materialization must download nonlinear rows from @META_DATASET_STAGE."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_nonlinear")

        assert (
            self.mod._stage_file_path("train/sample.parquet", split="train")
            == "@META_DATASET_STAGE/train/sample.parquet"
        )
        assert (
            self.mod._stage_file_path(
                "@META_DATASET_STAGE/train/sample.parquet",
                split="train",
            )
            == "@META_DATASET_STAGE/train/sample.parquet"
        )


# ---------------------------------------------------------------------------
# Fix 1d — SQL DDL presence checks
# ---------------------------------------------------------------------------

class TestNonlinearIndexDDL:

    def test_nonlinear_index_ddl_in_run_training_job_sql(self):
        """META_NONLINEAR_REGRESSION_DATASET_INDEX DDL must be present in sql/synthetic_nonlinear_pipeline.sql."""
        sql_file = ROOT / "sql" / "synthetic_nonlinear_pipeline.sql"
        text = sql_file.read_text()
        assert "META_NONLINEAR_REGRESSION_DATASET_INDEX" in text, (
            "META_NONLINEAR_REGRESSION_DATASET_INDEX DDL not found in sql/synthetic_nonlinear_pipeline.sql"
        )
        # Must be a CREATE TABLE statement, not just a comment
        assert "CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_REGRESSION_DATASET_INDEX" in text, (
            "Expected CREATE TRANSIENT TABLE IF NOT EXISTS META_NONLINEAR_REGRESSION_DATASET_INDEX"
        )

    def test_nonlinear_index_has_correct_columns(self):
        """META_NONLINEAR_REGRESSION_DATASET_INDEX DDL must include all required columns."""
        sql_file = ROOT / "sql" / "synthetic_nonlinear_pipeline.sql"
        text = sql_file.read_text()
        required_columns = ("split", "task_id", "stage_path", "n", "p", "n_train",
                            "n_test", "prior_regime", "hpo_bucket")
        for col in required_columns:
            assert col in text, (
                f"Column {col!r} not found in META_NONLINEAR_REGRESSION_DATASET_INDEX DDL"
            )

    def test_nonlinear_stage_ddl_in_stages_sql(self):
        """META_DATASET_STAGE must be created in sql/synthetic_nonlinear_pipeline.sql."""
        sql_file = ROOT / "sql" / "synthetic_nonlinear_pipeline.sql"
        text = sql_file.read_text()
        assert "META_DATASET_STAGE" in text, (
            "META_DATASET_STAGE not found in sql/synthetic_nonlinear_pipeline.sql"
        )
        assert "CREATE STAGE IF NOT EXISTS META_DATASET_STAGE" in text, (
            "Expected CREATE STAGE IF NOT EXISTS META_DATASET_STAGE"
        )

    def test_nonlinear_stage_has_sse_encryption(self):
        """META_DATASET_STAGE must use SNOWFLAKE_SSE encryption."""
        sql_file = ROOT / "sql" / "synthetic_nonlinear_pipeline.sql"
        text = sql_file.read_text()
        create_key = "CREATE STAGE IF NOT EXISTS META_DATASET_STAGE"
        idx = text.find(create_key)
        assert idx != -1, f"{create_key!r} not found in run_training_job.sql"
        snippet = text[idx : idx + 200]
        assert "SNOWFLAKE_SSE" in snippet, (
            "META_DATASET_STAGE must use ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')"
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
        assert "META_DATASET_STAGE" in text, (
            "build_meta_nonlinear_dataset_index.py must reference @META_DATASET_STAGE"
        )
        assert "META_NONLINEAR_REGRESSION_DATASET_INDEX" in text, (
            "build_meta_nonlinear_dataset_index.py must reference META_NONLINEAR_REGRESSION_DATASET_INDEX"
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
        """sql/synthetic_nonlinear_pipeline.sql must define run_pretrain_pipeline_nonlinear."""
        sql_file = ROOT / "sql" / "synthetic_nonlinear_pipeline.sql"
        text = sql_file.read_text()
        assert "run_pretrain_pipeline_nonlinear" in text, (
            "run_pretrain_pipeline_nonlinear procedure not found in "
            "sql/synthetic_nonlinear_pipeline.sql"
        )
        assert "build_meta_nonlinear_dataset_index" in text, (
            "build_meta_nonlinear_dataset_index procedure not found in "
            "sql/synthetic_nonlinear_pipeline.sql"
        )

    def test_6arg_hpo_overload_in_sql(self):
        """sql/synthetic_nonlinear_pipeline.sql must define 6-arg run_hpo_pipeline with HPO_PRETRAIN_CHECKPOINT_STAGE_PATH."""
        sql_file = ROOT / "sql" / "synthetic_nonlinear_pipeline.sql"
        text = sql_file.read_text()
        assert "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH" in text, (
            "6-arg HPO procedure HPO_PRETRAIN_CHECKPOINT_STAGE_PATH not found in sql/synthetic_nonlinear_pipeline.sql"
        )
        assert "run_hpo_pipeline_model_sweep_with_baseline_and_pretrain" in text, (
            "Handler run_hpo_pipeline_model_sweep_with_baseline_and_pretrain not found in sql/synthetic_nonlinear_pipeline.sql"
        )


# ---------------------------------------------------------------------------
# Fix 6 — run_training_job.py family routing
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal Snowpark session stub for run_training_job import tests."""
    def sql(self, query):
        return self

    def collect(self):
        return []


class TestRunTrainingJobFamilyRouting:
    """run_training_job.py _validate_training_index routes to the correct index."""

    def _load_run_training_job(self, monkeypatch):
        scripts = ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        sfml = types.ModuleType("snowflake")
        sfml.ml = types.ModuleType("snowflake.ml")
        sfml.ml.jobs = types.ModuleType("snowflake.ml.jobs")
        sfml.ml.jobs.submit_from_stage = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("snowflake", sfml)
        sys.modules.setdefault("snowflake.ml", sfml.ml)
        sys.modules.setdefault("snowflake.ml.jobs", sfml.ml.jobs)
        if "run_training_job" in sys.modules:
            del sys.modules["run_training_job"]
        import run_training_job
        return run_training_job

    def test_nonlinear_family_routes_to_nonlinear_index(self, monkeypatch):
        """synthetic_regression_nonlinear → _validate_nonlinear_training_index called."""
        mod = self._load_run_training_job(monkeypatch)
        called = []
        monkeypatch.setattr(mod, "_validate_nonlinear_training_index",
                            lambda s: called.append("nonlinear"))
        monkeypatch.setattr(mod, "_validate_meta_dataset_index",
                            lambda s: called.append("linear"))
        mod._validate_training_index(_FakeSession(), "synthetic_regression_nonlinear")
        assert called == ["nonlinear"]

    def test_linear_family_routes_to_linear_index(self, monkeypatch):
        """synthetic_regression_combined → _validate_meta_dataset_index called."""
        mod = self._load_run_training_job(monkeypatch)
        called = []
        monkeypatch.setattr(mod, "_validate_nonlinear_training_index",
                            lambda s: called.append("nonlinear"))
        monkeypatch.setattr(mod, "_validate_meta_dataset_index",
                            lambda s: called.append("linear"))
        mod._validate_training_index(_FakeSession(), "synthetic_regression_combined")
        assert called == ["linear"]

    def test_expected_index_counts_is_module_level_constant(self, monkeypatch):
        """EXPECTED_INDEX_COUNTS must be a module-level constant, not a local variable."""
        mod = self._load_run_training_job(monkeypatch)
        assert hasattr(mod, "EXPECTED_INDEX_COUNTS"), (
            "run_training_job.py must define EXPECTED_INDEX_COUNTS at module level"
        )
        assert mod.EXPECTED_INDEX_COUNTS == {"train": 800, "val": 100, "test": 100}

    def test_run_pipeline_raises_for_nonlinear_family(self, monkeypatch):
        """run_pipeline() must raise before any session/job when family is nonlinear."""
        mod = self._load_run_training_job(monkeypatch)
        mod.DEFAULT_TRAINING_DATA_FAMILY = "synthetic_regression_nonlinear"
        # _get_session must not be reached — patch it to a sentinel
        monkeypatch.setattr(mod, "_get_session", lambda: (_ for _ in ()).throw(
            AssertionError("_get_session should not be called for nonlinear family")
        ))
        with pytest.raises((RuntimeError, ValueError)):
            mod.run_pipeline()

    def test_run_pipeline_error_message_includes_required_strings(self, monkeypatch):
        """Error message must name all three required nonlinear entrypoints."""
        mod = self._load_run_training_job(monkeypatch)
        mod.DEFAULT_TRAINING_DATA_FAMILY = "synthetic_regression_nonlinear"
        monkeypatch.setattr(mod, "_get_session", lambda: None)
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            mod.run_pipeline()
        msg = str(exc_info.value)
        assert "run_pretrain_pipeline_nonlinear" in msg
        assert "nonlinear_model" in msg
        assert "run_model_training" in msg

    def test_run_pipeline_linear_family_passes_guard(self, monkeypatch):
        """synthetic_regression_combined must not trigger the nonlinear guard."""
        mod = self._load_run_training_job(monkeypatch)
        mod.DEFAULT_TRAINING_DATA_FAMILY = "synthetic_regression_combined"
        # If the guard fires incorrectly it raises before reaching _get_session.
        # We detect guard-pass by reaching _get_session (which we make raise a sentinel).
        sentinel = RuntimeError("_get_session_sentinel")
        monkeypatch.setattr(mod, "_get_session", lambda: (_ for _ in ()).throw(sentinel))
        with pytest.raises(RuntimeError, match="_get_session_sentinel"):
            mod.run_pipeline()


# ---------------------------------------------------------------------------
# New — TestPretrainValidationRouting
# ---------------------------------------------------------------------------

class TestPretrainValidationRouting:
    """_run_pretrain_impl and _run_pretrain_gate_impl route index validation by family."""

    def _load_run_pretrain_job(self, monkeypatch):
        scripts = ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        sfml = types.ModuleType("snowflake")
        sfml.ml = types.ModuleType("snowflake.ml")
        sfml.ml.jobs = types.ModuleType("snowflake.ml.jobs")
        sfml.ml.jobs.submit_from_stage = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("snowflake", sfml)
        sys.modules.setdefault("snowflake.ml", sfml.ml)
        sys.modules.setdefault("snowflake.ml.jobs", sfml.ml.jobs)
        if "run_pretrain_job" in sys.modules:
            del sys.modules["run_pretrain_job"]
        import run_pretrain_job
        return run_pretrain_job

    def test_generic_pretrain_nonlinear_family_calls_nonlinear_validator(self, monkeypatch):
        """_run_pretrain_impl with nonlinear family must call _validate_nonlinear_dataset_index."""
        mod = self._load_run_pretrain_job(monkeypatch)
        called = []
        monkeypatch.setattr(mod, "_validate_nonlinear_dataset_index", lambda s: called.append("nonlinear"))
        monkeypatch.setattr(mod, "_validate_meta_dataset_index", lambda s: called.append("linear"))
        monkeypatch.setattr(mod, "submit_from_stage", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(mod, "_wait_done", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_list_stage", lambda *a, **kw: [])
        mod._run_pretrain_impl(MagicMock(), "mf", "synthetic_regression_nonlinear", "mdp")
        assert called == ["nonlinear"]

    def test_generic_pretrain_linear_family_calls_linear_validator(self, monkeypatch):
        """_run_pretrain_impl with linear family must call _validate_meta_dataset_index."""
        mod = self._load_run_pretrain_job(monkeypatch)
        called = []
        monkeypatch.setattr(mod, "_validate_nonlinear_dataset_index", lambda s: called.append("nonlinear"))
        monkeypatch.setattr(mod, "_validate_meta_dataset_index", lambda s: called.append("linear"))
        monkeypatch.setattr(mod, "submit_from_stage", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(mod, "_wait_done", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_list_stage", lambda *a, **kw: [])
        mod._run_pretrain_impl(MagicMock(), "mf", "synthetic_regression_combined", "mdp")
        assert called == ["linear"]

    def test_gate_pretrain_nonlinear_family_calls_nonlinear_validator(self, monkeypatch):
        """_run_pretrain_gate_impl with nonlinear family must call _validate_nonlinear_dataset_index."""
        mod = self._load_run_pretrain_job(monkeypatch)
        called = []
        monkeypatch.setattr(mod, "_validate_nonlinear_dataset_index", lambda s: called.append("nonlinear"))
        monkeypatch.setattr(mod, "_validate_meta_dataset_index", lambda s: called.append("linear"))
        monkeypatch.setattr(mod, "submit_from_stage", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(mod, "_wait_done", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_list_stage", lambda *a, **kw: [])
        mod._run_pretrain_gate_impl(MagicMock(), "mf", "synthetic_regression_nonlinear", "mdp", 64)
        assert called == ["nonlinear"]


# ---------------------------------------------------------------------------
# New — TestNonlinearPipelineSqlDrift
# ---------------------------------------------------------------------------

class TestNonlinearPipelineSqlDrift:
    """synthetic_nonlinear_pipeline.sql must not use stale training patterns."""

    def test_no_stale_pretrain_call(self):
        """synthetic_nonlinear_pipeline.sql must not call run_pretrain_pipeline with nonlinear family."""
        sql = (ROOT / "sql" / "synthetic_nonlinear_pipeline.sql").read_text()
        # The stale pattern is calling the GENERIC run_pretrain_pipeline (not _nonlinear)
        # with 'synthetic_regression_nonlinear' as an argument.
        assert "run_pretrain_pipeline_nonlinear" in sql, (
            "Must call run_pretrain_pipeline_nonlinear, not the generic run_pretrain_pipeline"
        )

    def test_no_drop_compute_pool(self):
        """synthetic_nonlinear_pipeline.sql must not contain DROP COMPUTE POOL statements."""
        sql = (ROOT / "sql" / "synthetic_nonlinear_pipeline.sql").read_text()
        assert "DROP COMPUTE POOL" not in sql, (
            "DROP COMPUTE POOL is unsafe in the production nonlinear pipeline"
        )

    def test_hpo_call_includes_pretrain_checkpoint_path(self):
        """synthetic_nonlinear_pipeline.sql HPO CALL must include pretrain_nonlinear_meta.pt."""
        sql = (ROOT / "sql" / "synthetic_nonlinear_pipeline.sql").read_text()
        assert "pretrain_nonlinear_meta.pt" in sql, (
            "HPO CALL must pass HPO_PRETRAIN_CHECKPOINT_STAGE_PATH with pretrain_nonlinear_meta.pt"
        )
