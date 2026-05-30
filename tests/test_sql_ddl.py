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
        "run_synthetic_regression_combined_autogluon_worker_access_probe",
        "run_synthetic_regression_combined_aggregation",
    ])
    def test_combined_phase_procedure_exists(self, name):
        assert name in self._sql(), f"Procedure '{name}' not found in SQL DDL"

    def test_concurrency_overload_arguments_exist(self):
        sql = self._sql()
        assert "BASELINE_CONCURRENT_NODES INTEGER" in sql
        assert "AUTOGLUON_CONCURRENT_NODES INTEGER" in sql

    def test_baseline_shards_overload_argument_exists(self):
        """BASELINE_SHARDS INTEGER must appear in the SQL for the new overloads."""
        assert "BASELINE_SHARDS INTEGER" in self._sql()

    def test_combined_baseline_capacity_probe_4arg_overload_exists(self):
        """4-arg combined baseline capacity probe overload (BASELINE_SHARDS, BASELINE_CONCURRENT_NODES) must exist."""
        sql = self._sql()
        assert "run_synthetic_regression_combined_baseline_capacity_probe_with_shards" in sql

    def test_combined_baseline_evaluation_4arg_overload_exists(self):
        """4-arg combined baseline evaluation overload (BASELINE_SHARDS, BASELINE_CONCURRENT_NODES) must exist."""
        sql = self._sql()
        assert "run_synthetic_regression_combined_baseline_evaluation_with_shards" in sql

    def test_combined_evaluation_with_baseline_shards_overload_exists(self):
        """Full combined evaluation overload with BASELINE_SHARDS must exist."""
        sql = self._sql()
        assert "run_synthetic_regression_combined_evaluation_with_baseline_shards" in sql

    def test_old_combined_baseline_capacity_probe_2arg_still_exists(self):
        """Backward-compatible 2-arg combined baseline capacity probe must remain."""
        sql = self._sql()
        assert "run_synthetic_regression_combined_baseline_capacity_probe_default" in sql

    def test_old_combined_baseline_evaluation_2arg_still_exists(self):
        """Backward-compatible 2-arg combined baseline evaluation must remain."""
        sql = self._sql()
        assert "run_synthetic_regression_combined_baseline_evaluation_default" in sql

    def test_combined_autogluon_worker_access_probe_handlers_exist(self):
        sql = self._sql()
        assert (
            "run_synthetic_regression_evaluation."
            "run_synthetic_regression_combined_autogluon_worker_access_probe_default"
        ) in sql
        assert (
            "run_synthetic_regression_evaluation."
            "run_synthetic_regression_combined_autogluon_worker_access_probe"
        ) in sql

    def test_ray_readiness_overloads_exist(self):
        sql = self._sql()
        assert "RAY_READY_TIMEOUT_SECONDS INTEGER" in sql
        assert "RAY_READY_POLL_SECONDS INTEGER" in sql
        assert "SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS" in sql
        assert "SYNREG_RAY_CLUSTER_READY_POLL_SECONDS" in sql

    def test_combined_autogluon_runbook_orders_capacity_worker_access_then_eval(self):
        sql = self._sql()
        capacity_pos = sql.index("CALL run_synthetic_regression_combined_autogluon_capacity_probe")
        worker_access_pos = sql.index(
            "CALL run_synthetic_regression_combined_autogluon_worker_access_probe"
        )
        evaluation_pos = sql.index("CALL run_synthetic_regression_combined_autogluon_evaluation")
        assert capacity_pos < worker_access_pos < evaluation_pos

    def test_concurrency_comments_document_single_wave_rejection(self):
        sql = self._sql()
        assert "Single-wave execution is enforced" in sql
        assert "Lower values fail fast" in sql or "Lower values are rejected" in sql
        assert "AUTOGLUON_CONCURRENT_CLUSTERS must equal AUTOGLUON_CLUSTER_SHARDS" in sql

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

    def test_no_stray_characters_after_string_literals_in_imports(self):
        """IMPORTS blocks must not contain stray characters immediately after string literals (e.g. trailing 'F')."""
        import re
        sql = self._sql()
        # Extract all IMPORTS blocks: IMPORTS = ( ... )
        imports_blocks = re.findall(r'IMPORTS\s*=\s*\([^)]*\)', sql, re.DOTALL)
        assert imports_blocks, "No IMPORTS blocks found in SQL — check DDL structure"
        for block in imports_blocks:
            # Flag any string literal immediately followed by a letter (no space/comma/newline between)
            bad = re.findall(r"'[^'\n]*'([A-Za-z])", block)
            assert not bad, (
                f"Stray character(s) after string literal in IMPORTS block: {bad!r}\n"
                f"Block: {block!r}\n"
                "A trailing letter (e.g. 'F') causes a SQL syntax error."
            )


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

    def test_gate_hidden_dim_overload_exists(self):
        """run_pretrain_pipeline 4-arg overload with GATE_HIDDEN_DIM INTEGER must exist."""
        sql = self._sql()
        assert "GATE_HIDDEN_DIM INTEGER" in sql, (
            "SQL must define the 4-arg run_pretrain_pipeline overload with GATE_HIDDEN_DIM INTEGER"
        )
        assert "run_pretrain_pipeline_model_gate" in sql, (
            "SQL handler must be run_pretrain_pipeline_model_gate"
        )

    def test_gate_pretrain_runbook_pattern_exists(self):
        """Runbook section must show three CALL run_pretrain_pipeline(..., <N>); examples."""
        sql = self._sql()
        # Each gate dim should appear in a pretrain call
        for gate_dim in (32, 64, 128):
            assert f"'inductive_forecasting', {gate_dim}" in sql, (
                f"Runbook must include CALL run_pretrain_pipeline(..., {gate_dim})"
            )

    def test_architecture_sweep_enabled_comment(self):
        """SQL must document that architecture HPO is enabled and describe two-sweep strategy."""
        sql = self._sql()
        assert "architecture" in sql, (
            "SQL must reference architecture sweep mode"
        )
        # architecture sweep is now enabled — no 'NotImplementedError' guard should exist
        assert "NotImplementedError" not in sql, (
            "SQL must not reference NotImplementedError for architecture sweep "
            "(architecture sweep is now enabled)"
        )

    def test_architecture_recommended_as_two_sweep_path(self):
        """SQL must describe two-sweep HPO (ridge_residual → architecture) as recommended."""
        sql = self._sql()
        # Two-sweep HPO is now the recommended path
        assert "Two-sweep HPO" in sql or "two-sweep" in sql.lower(), (
            "SQL must describe two-sweep HPO strategy"
        )
        assert "architecture" in sql, (
            "SQL must reference architecture sweep as part of the recommended path"
        )

    def test_pretrain_pt_not_required_for_hpo(self):
        """SQL HPO comments must not state pretrain.pt is required (gate-specific files are)."""
        sql = self._sql()
        # Find the hpo section comment block
        hpo_start = sql.find("run_hpo_pipeline() launches")
        hpo_end = sql.find("CREATE OR REPLACE PROCEDURE run_hpo_pipeline()", hpo_start)
        hpo_section = sql[hpo_start:hpo_end] if hpo_start != -1 and hpo_end != -1 else ""
        assert "pretrain.pt" not in hpo_section, (
            "HPO comment block must not claim pretrain.pt (not gate-specific) is required"
        )


class TestAutogluonImportTimingProbeSQLDDL:
    """Tests that run_training_job.sql contains the import timing probe DDL."""

    def _sql(self) -> str:
        return (ROOT / "sql" / "run_training_job.sql").read_text()

    def test_default_handler_exists(self):
        assert (
            "run_synthetic_regression_evaluation."
            "run_synthetic_regression_autogluon_import_timing_probe_default"
        ) in self._sql()

    def test_full_handler_exists(self):
        assert (
            "run_synthetic_regression_evaluation."
            "run_synthetic_regression_autogluon_import_timing_probe"
        ) in self._sql()

    def test_with_pip_boolean_param_exists(self):
        assert "WITH_PIP BOOLEAN" in self._sql()

    def test_probe_count_integer_param_exists(self):
        assert "PROBE_COUNT INTEGER" in self._sql()

    def test_staging_instructions_include_probe_script(self):
        sql = self._sql()
        assert "autogluon_import_timing_probe.py" in sql


class TestSPCSSQLDDL:
    """Tests that run_training_job.sql and related SQL files contain SPCS DDL."""

    def _sql(self) -> str:
        return (ROOT / "sql" / "run_training_job.sql").read_text()

    def _repo_sql(self) -> str:
        return (ROOT / "sql" / "create_autogluon_spcs_image_repository.sql").read_text()

    def test_image_repository_ddl_exists(self):
        assert "AUTOGLUON_IMAGE_REPOSITORY" in self._repo_sql()

    def test_image_repository_create_statement(self):
        assert "CREATE IMAGE REPOSITORY" in self._repo_sql()

    def test_image_repository_show_command(self):
        assert "SHOW IMAGE REPOSITORIES" in self._repo_sql()

    def test_spcs_import_probe_procedure_exists(self):
        assert "run_synthetic_regression_autogluon_spcs_import_probe" in self._sql()

    def test_spcs_session_probe_procedure_exists(self):
        """run_synthetic_regression_autogluon_spcs_session_probe must be defined in SQL."""
        assert "run_synthetic_regression_autogluon_spcs_session_probe" in self._sql()

    def test_spcs_session_probe_handler_correct(self):
        assert (
            "run_synthetic_regression_evaluation."
            "run_synthetic_regression_autogluon_spcs_session_probe"
        ) in self._sql()

    def test_spcs_spec_snowflake_service_documented_in_sql(self):
        """SQL runbook or SPCS comments must reference snowflakeService token injection."""
        sql = self._sql()
        # Either in a comment or in the procedure DDL context
        assert "snowflakeService" in sql or "session/token" in sql or "OAuth" in sql

    def test_spcs_capacity_probe_procedure_exists(self):
        assert "run_synthetic_regression_combined_autogluon_spcs_capacity_probe" in self._sql()

    def test_spcs_worker_access_probe_procedure_exists(self):
        assert "run_synthetic_regression_combined_autogluon_spcs_worker_access_probe" in self._sql()

    def test_spcs_evaluation_procedure_exists(self):
        assert "run_synthetic_regression_combined_autogluon_spcs_evaluation" in self._sql()

    def test_spcs_procedures_have_correct_handlers(self):
        sql = self._sql()
        assert "run_synthetic_regression_evaluation.run_synthetic_regression_autogluon_spcs_import_probe" in sql
        assert "run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_capacity_probe" in sql
        assert "run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_worker_access_probe" in sql
        assert "run_synthetic_regression_evaluation.run_synthetic_regression_combined_autogluon_spcs_evaluation" in sql

    def test_spcs_procedures_no_pip_requirements(self):
        sql = self._sql()
        # Find SPCS procedure sections and verify they don't mention pip_requirements
        import re
        spcs_blocks = re.findall(
            r'CREATE OR REPLACE PROCEDURE run_synthetic_regression_(?:combined_autogluon_spcs|autogluon_spcs)[^\n]*.*?HANDLER\s*=\s*[^\n]+',
            sql, re.DOTALL
        )
        assert spcs_blocks, "No SPCS procedure blocks found"
        for block in spcs_blocks:
            assert "pip_requirements" not in block.lower()

    def test_docs_mention_autogluon_image_repository(self):
        training_md = (ROOT / "docs" / "Snowflake_Training.md").read_text(encoding="utf-8")
        assert "AUTOGLUON_IMAGE_REPOSITORY" in training_md

    def test_docs_mention_no_mljob_runtime_for_spcs(self):
        training_md = (ROOT / "docs" / "Snowflake_Training.md").read_text(encoding="utf-8")
        # Docs must explain that SPCS path does not use MLJob runtime_environment
        assert "runtime_environment" in training_md
        assert "spcs" in training_md.lower() or "SPCS" in training_md

    def test_docs_mention_self_managed_ray(self):
        training_md = (ROOT / "docs" / "Snowflake_Training.md").read_text(encoding="utf-8")
        assert "self-managed" in training_md.lower() or "self managed" in training_md.lower()

    def test_docs_mention_spcs_resource_profiles_and_container_counts(self):
        training_md = (ROOT / "docs" / "Snowflake_Training.md").read_text(encoding="utf-8")
        assert "30 containers" in training_md
        assert "24 worker" in training_md
        assert "SYNREG_SPCS_RAY_COORDINATOR_CPU" in training_md
        assert "SYNREG_SPCS_RAY_WORKER_CPU" in training_md

    def test_sql_runbook_mentions_spcs_resource_profiles(self):
        sql = self._sql()
        assert "30 containers" in sql
        assert "24 worker" in sql
        assert "SYNREG_SPCS_RAY_COORDINATOR_*" in sql
        assert "SYNREG_SPCS_RAY_WORKER_*" in sql


class TestCombined04SPCSSQLDDL:
    def _sql(self) -> str:
        return (ROOT / "sql" / "04_synthetic_regression_evaluation_pipeline.sql").read_text()

    def test_spcs_session_probe_procedure_exists(self):
        assert "run_synthetic_regression_autogluon_spcs_session_probe" in self._sql()

    def test_spcs_procedure_signatures_use_explicit_image_arg(self):
        import re

        sql = self._sql()
        procedure_names = [
            "run_synthetic_regression_autogluon_spcs_import_probe",
            "run_synthetic_regression_autogluon_spcs_session_probe",
            "run_synthetic_regression_combined_autogluon_spcs_capacity_probe",
            "run_synthetic_regression_combined_autogluon_spcs_worker_access_probe",
            "run_synthetic_regression_combined_autogluon_spcs_evaluation",
        ]
        for procedure_name in procedure_names:
            assert re.search(
                rf"CREATE OR REPLACE PROCEDURE {procedure_name}\(\s*AUTOGLUON_SPCS_IMAGE STRING",
                sql,
            ), f"{procedure_name} must take AUTOGLUON_SPCS_IMAGE as its first argument"

    def test_spcs_runbook_calls_pass_image_ref_first(self):
        import re

        sql = self._sql()
        assert "CALL run_synthetic_regression_autogluon_spcs_import_probe('<image_ref>', 1);" in sql
        assert "CALL run_synthetic_regression_autogluon_spcs_session_probe('<image_ref>', 1);" in sql
        assert (
            "CALL run_synthetic_regression_combined_autogluon_spcs_capacity_probe('<image_ref>', 0, 1, 6);"
            in sql
        )
        assert (
            "CALL run_synthetic_regression_combined_autogluon_spcs_worker_access_probe('<image_ref>', 0, 1, 6);"
            in sql
        )
        assert re.search(
            r"CALL run_synthetic_regression_combined_autogluon_spcs_evaluation\(\s*'<image_ref>'",
            sql,
        )
        assert not re.search(
            r"CALL run_synthetic_regression_[^(]*spcs[^(]*\(\s*'spcs_job'",
            sql,
        )
        assert "\nALTER SESSION SET SYNREG_AUTOGLUON_SPCS_IMAGE" not in sql

    def test_spcs_evaluation_9arg_overload_exists(self):
        sql = self._sql()
        import re
        matches = re.findall(
            r"CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_evaluation\b[^;]+;",
            sql,
            re.DOTALL,
        )
        nine_arg = [m for m in matches if "RAY_READY_TIMEOUT_SECONDS" in m and "WORKER_SUBMIT_STAGGER_SECONDS" in m]
        assert nine_arg, (
            "Expected a 9-argument overload of run_synthetic_regression_combined_autogluon_spcs_evaluation "
            "with RAY_READY_TIMEOUT_SECONDS and WORKER_SUBMIT_STAGGER_SECONDS in sql/04_..."
        )

    def test_production_spcs_evaluation_sql_does_not_expose_keep_support_jobs(self):
        sql = self._sql()
        import re
        procs = re.findall(
            r"CREATE OR REPLACE PROCEDURE run_synthetic_regression_combined_autogluon_spcs_evaluation\b[^;]+;",
            sql,
            re.DOTALL,
        )
        for proc in procs:
            assert "KEEP_SUPPORT_JOBS_ON_FAILURE" not in proc, (
                "KEEP_SUPPORT_JOBS_ON_FAILURE must not appear in the production evaluation "
                "procedure signature — it is an env-only diagnostic escape hatch."
            )
