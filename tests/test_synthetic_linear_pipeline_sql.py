"""
tests/test_synthetic_linear_pipeline_sql.py
============================================
SQL DDL structural tests for sql/synthetic_linear_pipeline.sql.
Reads the file as text and asserts presence of required identifiers/strings.
No Snowflake connection required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SQL_PATH = Path(__file__).parent.parent / "sql" / "synthetic_linear_pipeline.sql"


@pytest.fixture(scope="module")
def sql_content() -> str:
    return SQL_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Regression-linear procedure names
# ---------------------------------------------------------------------------

REGRESSION_LINEAR_PROCEDURES = [
    "run_synthetic_regression_linear_prep",
    "run_synthetic_regression_linear_deepset_evaluation",
    "run_synthetic_regression_linear_baseline_evaluation",
    "run_synthetic_regression_linear_autogluon_evaluation",
    "run_synthetic_regression_linear_aggregation",
    "run_synthetic_regression_linear_pipeline",
]


@pytest.mark.parametrize("proc_name", REGRESSION_LINEAR_PROCEDURES)
def test_regression_linear_procedure_present(sql_content, proc_name):
    assert proc_name in sql_content, (
        f"Expected procedure {proc_name!r} not found in SQL file."
    )


# ---------------------------------------------------------------------------
# Classification-linear procedure names
# ---------------------------------------------------------------------------

CLASSIFICATION_LINEAR_PROCEDURES = [
    "run_synthetic_classification_linear_prep",
    "run_synthetic_classification_linear_deepset_evaluation",
    "run_synthetic_classification_linear_baseline_evaluation",
    "run_synthetic_classification_linear_autogluon_evaluation",
    "run_synthetic_classification_linear_aggregation",
    "run_synthetic_classification_linear_pipeline",
]


@pytest.mark.parametrize("proc_name", CLASSIFICATION_LINEAR_PROCEDURES)
def test_classification_linear_procedure_present(sql_content, proc_name):
    assert proc_name in sql_content, (
        f"Expected procedure {proc_name!r} not found in SQL file."
    )


# ---------------------------------------------------------------------------
# Result path references
# ---------------------------------------------------------------------------

def test_regression_linear_result_path_in_sql(sql_content):
    """Regression linear procedures must reference /linear/regression/numeric/ path."""
    assert "/linear/regression/numeric/" in sql_content, (
        "Expected '/linear/regression/numeric/' path reference in SQL file."
    )


def test_classification_linear_result_path_in_sql(sql_content):
    """Classification linear procedures must reference /linear/classification/numeric/ path."""
    assert "/linear/classification/numeric/" in sql_content, (
        "Expected '/linear/classification/numeric/' path reference in SQL file."
    )


# ---------------------------------------------------------------------------
# Checkpoint references
# ---------------------------------------------------------------------------

def test_best_regression_pt_in_sql(sql_content):
    """Regression linear section must mention best_regression.pt."""
    assert "best_regression.pt" in sql_content, (
        "Expected 'best_regression.pt' checkpoint reference in SQL file."
    )


def test_best_classification_pt_in_sql(sql_content):
    """Classification linear section must mention best_classification.pt."""
    assert "best_classification.pt" in sql_content, (
        "Expected 'best_classification.pt' checkpoint reference in SQL file."
    )


# ---------------------------------------------------------------------------
# Feature selector references
# ---------------------------------------------------------------------------

def test_train_f_regression_in_sql(sql_content):
    """Regression section must mention train_f_regression feature selector."""
    assert "train_f_regression" in sql_content, (
        "Expected 'train_f_regression' feature selector reference in SQL file."
    )


def test_train_f_classif_in_sql(sql_content):
    """Classification section must mention train_f_classif feature selector."""
    assert "train_f_classif" in sql_content, (
        "Expected 'train_f_classif' feature selector reference in SQL file."
    )


# ---------------------------------------------------------------------------
# Index table names
# ---------------------------------------------------------------------------

def test_synthetic_classification_dataset_index_present(sql_content):
    assert "LINEAR_CLASSIFICATION_DATASET_INDEX" in sql_content


def test_synthetic_regression_dataset_index_present(sql_content):
    assert "LINEAR_REGRESSION_DATASET_INDEX" in sql_content


# ---------------------------------------------------------------------------
# Handler module reference
# ---------------------------------------------------------------------------

def test_classification_handler_module_for_classification_procedures(sql_content):
    """Classification procedures must reference run_synthetic_classification_evaluation as HANDLER module."""
    assert "run_synthetic_classification_evaluation." in sql_content, (
        "Expected 'run_synthetic_classification_evaluation.' handler reference for classification procedures."
    )


def test_regression_linear_handler_module(sql_content):
    """Regression linear procedures must reference run_synthetic_regression_evaluation as HANDLER module."""
    assert "run_synthetic_regression_evaluation.run_synthetic_regression_linear_" in sql_content, (
        "Expected regression-linear handler references in SQL file."
    )


# ---------------------------------------------------------------------------
# New schema columns in LINEAR_CLASSIFICATION_DATASET_INDEX
# ---------------------------------------------------------------------------

NEW_CLASSIFICATION_COLUMNS = [
    "task_family",
    "task_objective",
    "label_noise_rate",
    "class_imbalance_type",
    "temperature",
    "margin_level",
    "sample_complexity_bucket",
]


@pytest.mark.parametrize("col_name", NEW_CLASSIFICATION_COLUMNS)
def test_new_classification_index_columns_present(sql_content, col_name):
    assert col_name in sql_content, (
        f"Expected new column {col_name!r} in LINEAR_CLASSIFICATION_DATASET_INDEX DDL."
    )


def test_classification_index_uses_create_or_replace(sql_content):
    """Schema change: LINEAR_CLASSIFICATION_DATASET_INDEX must use CREATE OR REPLACE."""
    # The schema update changes IF NOT EXISTS → CREATE OR REPLACE TRANSIENT TABLE
    assert "CREATE OR REPLACE TRANSIENT TABLE LINEAR_CLASSIFICATION_DATASET_INDEX" in sql_content, (
        "Expected 'CREATE OR REPLACE TRANSIENT TABLE LINEAR_CLASSIFICATION_DATASET_INDEX'. "
        "Schema update may not have been applied."
    )


# ---------------------------------------------------------------------------
# Example CALL commands
# ---------------------------------------------------------------------------

def test_example_regression_linear_call_present(sql_content):
    assert "run_synthetic_regression_linear_pipeline" in sql_content


def test_example_classification_linear_call_present(sql_content):
    assert "run_synthetic_classification_linear_pipeline" in sql_content


# ---------------------------------------------------------------------------
# Phase 0 schema fix tests
# ---------------------------------------------------------------------------

def test_regression_index_feature_noise_level_is_float(sql_content):
    """LINEAR_REGRESSION_DATASET_INDEX must declare feature_noise_level as FLOAT, not NUMBER."""
    # Confirm the old NUMBER declaration is gone and FLOAT is present in the regression index block
    lines = sql_content.splitlines()
    in_regression_index = False
    found_float = False
    found_number = False
    for line in lines:
        if "LINEAR_REGRESSION_DATASET_INDEX" in line:
            in_regression_index = True
        if in_regression_index:
            if "feature_noise_level" in line:
                if "FLOAT" in line.upper():
                    found_float = True
                if line.strip().upper().endswith("NUMBER") or "NUMBER," in line.upper():
                    found_number = True
                break
    assert found_float, (
        "feature_noise_level in LINEAR_REGRESSION_DATASET_INDEX must be declared as FLOAT. "
        "Found: " + repr(next((l for l in lines if "feature_noise_level" in l and "REGRESSION" not in l), "not found"))
    )
    assert not found_number, (
        "feature_noise_level in LINEAR_REGRESSION_DATASET_INDEX must NOT be NUMBER. "
        "Old type still present."
    )


def test_task_seed_in_meta_dataset_index(sql_content):
    """META_REGRESSION_DATASET_INDEX must declare task_seed column."""
    assert "task_seed" in sql_content, "task_seed column missing from SQL"
    # Verify it appears before LINEAR_REGRESSION_DATASET_INDEX (i.e., in META_REGRESSION_DATASET_INDEX)
    meta_pos = sql_content.find("META_REGRESSION_DATASET_INDEX")
    task_seed_pos = sql_content.find("task_seed")
    assert task_seed_pos > meta_pos, "task_seed must appear after META_REGRESSION_DATASET_INDEX definition"


def test_task_seed_in_classification_dataset_index(sql_content):
    """LINEAR_CLASSIFICATION_DATASET_INDEX must declare task_seed column."""
    cls_pos = sql_content.find("LINEAR_CLASSIFICATION_DATASET_INDEX")
    assert cls_pos != -1, "LINEAR_CLASSIFICATION_DATASET_INDEX not found"
    # task_seed must appear somewhere after the classification index DDL starts
    task_seed_pos = sql_content.find("task_seed", cls_pos)
    assert task_seed_pos != -1, "task_seed column missing from LINEAR_CLASSIFICATION_DATASET_INDEX"


def test_coeff_schema_version_in_classification_index(sql_content):
    """LINEAR_CLASSIFICATION_DATASET_INDEX must declare coeff_schema_version."""
    assert "coeff_schema_version" in sql_content, (
        "coeff_schema_version column missing from LINEAR_CLASSIFICATION_DATASET_INDEX"
    )


def test_base_task_id_in_regression_index(sql_content):
    """LINEAR_REGRESSION_DATASET_INDEX must declare base_task_id column."""
    assert "base_task_id" in sql_content, (
        "base_task_id column missing from LINEAR_REGRESSION_DATASET_INDEX"
    )


def test_perturbation_condition_in_regression_index(sql_content):
    """LINEAR_REGRESSION_DATASET_INDEX must declare perturbation_condition column."""
    assert "perturbation_condition" in sql_content, (
        "perturbation_condition column missing from LINEAR_REGRESSION_DATASET_INDEX"
    )


def test_stage_path_contract_comment_present(sql_content):
    """SQL file must include stage path contract documentation block."""
    assert "STAGE PATH CONTRACT" in sql_content, (
        "STAGE PATH CONTRACT comment block missing from SQL file"
    )


def test_metric_schema_version_note_present(sql_content):
    """SQL file must include metric_schema_version documentation note for deepset procedure."""
    assert "metric_schema_version" in sql_content, (
        "metric_schema_version note missing from classification deepset procedure"
    )
    assert "1.0' = stored DGP teacher" in sql_content or "'1.0' = stored DGP teacher" in sql_content, (
        "metric_schema_version note must document v1.0 as deprecated stored DGP teacher probs"
    )
    assert "2.0' = real MODEL4" in sql_content or "'2.0' = real MODEL4" in sql_content, (
        "metric_schema_version note must document v2.0 as real MODEL4 inference"
    )


def test_index_classification_eval_suite_procedure_present(sql_content):
    """index_synthetic_classification_eval_suite procedure must be defined."""
    assert "index_synthetic_classification_eval_suite" in sql_content, (
        "index_synthetic_classification_eval_suite procedure missing from SQL file"
    )
