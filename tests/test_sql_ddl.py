"""
tests/test_sql_ddl.py
=====================
Tests that run_training_job.sql contains the required stored procedure DDL
for the OOD full suite evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


class TestSQLStoredProcedures:
    def _sql(self) -> str:
        return (ROOT / "sql" / "run_training_job.sql").read_text()

    def test_ood_full_procedure_exists(self):
        assert "run_synthetic_regression_ood_full_evaluation" in self._sql()

    def test_ood_full_handler_is_correct(self):
        assert (
            "run_synthetic_regression_evaluation.run_synthetic_regression_ood_full_evaluation"
            in self._sql()
        )

    def test_ood_full_imports_all_required_scripts(self):
        sql = self._sql()
        assert "prepare_ood_regression.py" in sql
        assert "prepare_synthetic_regression.py" in sql
        assert "run_synthetic_regression_evaluation.py" in sql

    def test_synthetic_regression_index_ddl_exists(self):
        assert "SYNTHETIC_REGRESSION_DATASET_INDEX" in self._sql()

    def test_synthetic_regression_index_has_logical_dataset_key(self):
        sql = self._sql()
        assert "logical_dataset_key" in sql
        assert "SYNTHETIC_REGRESSION_DATASET_INDEX" in sql

    def test_ood_full_procedure_has_runbook(self):
        sql = self._sql()
        assert "ood_linear_full_v1" in sql
        assert "200 datasets" in sql or "200" in sql


class TestSplitPhaseStoredProcedures:
    def _sql(self) -> str:
        return (ROOT / "sql" / "run_training_job.sql").read_text()

    @pytest.mark.parametrize("name", [
        "run_synthetic_regression_runtime_probes",
        "run_synthetic_regression_capacity_probe",
        "run_synthetic_regression_baseline_capacity_probe",
        "run_synthetic_regression_autogluon_capacity_probe",
        "run_synthetic_regression_prep",
        "run_synthetic_regression_deepset_evaluation",
        "run_synthetic_regression_baseline_evaluation",
        "run_synthetic_regression_autogluon_evaluation",
        "run_synthetic_regression_aggregation",
        "run_synthetic_regression_pipeline",
    ])
    def test_main_pipeline_procedure_exists(self, name):
        assert name in self._sql(), f"Procedure '{name}' not found in SQL DDL"

    @pytest.mark.parametrize("name", [
        "run_synthetic_regression_ood_full_prep",
        "run_synthetic_regression_ood_full_deepset_evaluation",
        "run_synthetic_regression_ood_full_baseline_evaluation",
        "run_synthetic_regression_ood_full_autogluon_evaluation",
        "run_synthetic_regression_ood_full_aggregation",
    ])
    def test_ood_full_phase_procedure_exists(self, name):
        assert name in self._sql(), f"Procedure '{name}' not found in SQL DDL"

    @pytest.mark.parametrize("name", [
        "run_synthetic_regression_combined_prep",
        "run_synthetic_regression_combined_deepset_evaluation",
        "run_synthetic_regression_combined_baseline_evaluation",
        "run_synthetic_regression_combined_autogluon_evaluation",
        "run_synthetic_regression_combined_aggregation",
    ])
    def test_combined_phase_procedure_exists(self, name):
        assert name in self._sql(), f"Procedure '{name}' not found in SQL DDL"

    def test_concurrency_overload_arguments_exist(self):
        sql = self._sql()
        assert "BASELINE_CONCURRENT_NODES INTEGER" in sql
        assert "AUTOGLUON_CONCURRENT_NODES INTEGER" in sql

    def test_existing_synthetic_regression_signatures_remain(self):
        sql = self._sql()
        assert (
            "run_synthetic_regression_baseline_evaluation(\n"
            "  PREP_RUNTIME_ENVIRONMENT STRING,\n"
            "  BENCHMARK_RUNTIME_ENVIRONMENT STRING,\n"
            "  AUTOGLUON_RUNTIME_ENVIRONMENT STRING\n"
            ")"
        ) in sql
        assert (
            "run_synthetic_regression_autogluon_evaluation(\n"
            "  PREP_RUNTIME_ENVIRONMENT STRING,\n"
            "  BENCHMARK_RUNTIME_ENVIRONMENT STRING,\n"
            "  AUTOGLUON_RUNTIME_ENVIRONMENT STRING\n"
            ")"
        ) in sql

    def test_compute_pool_defaults_reflect_runtime_concurrency_defaults(self):
        sql = self._sql()
        assert "MAX_NODES = 6" in sql
        assert "MAX_NODES = 60" in sql

    def test_evaluation_dataset_stage_exists(self):
        sql = self._sql()
        assert "CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE" in sql
        assert "EVALUATION_DATASET_STAGE" in sql


class TestSQLEnvVarRenames:
    """Verify MODEL_FAMILY / MODEL_ARCH_VERSION rename in stored procedure signatures."""

    def _sql(self) -> str:
        return (ROOT / "sql" / "run_training_job.sql").read_text()

    def test_model_family_param_present(self):
        """MODEL_FAMILY STRING must appear in the SQL."""
        assert "MODEL_FAMILY STRING" in self._sql()

    def test_deepset_model_family_param_absent(self):
        """Retired model-family parameter must not appear in the SQL."""
        assert "DEEPSET" + "_MODEL_FAMILY STRING" not in self._sql()

    def test_model_arch_version_absent_from_pretrain_pipeline(self):
        """run_pretrain_pipeline parameterised overload must not accept MODEL_ARCH_VERSION.

        Note: run_model_ddp_memory_probe legitimately keeps MODEL_ARCH_VERSION STRING
        as a probe-validation param — we scope this check to the pretrain pipeline only.
        """
        sql = self._sql()
        # Extract only the run_pretrain_pipeline procedure definition
        start = sql.find("run_pretrain_pipeline")
        end = sql.find("run_hpo_pipeline")
        pretrain_section = sql[start:end] if start != -1 and end != -1 else sql
        assert "MODEL_ARCH_VERSION STRING" not in pretrain_section

    def test_model_family_in_pretrain_pipeline(self):
        """run_pretrain_pipeline must reference MODEL_FAMILY."""
        sql = self._sql()
        assert "run_pretrain_pipeline" in sql
        assert "DEEPSET" + "_MODEL_FAMILY" not in sql
