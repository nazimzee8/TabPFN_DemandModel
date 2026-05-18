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
