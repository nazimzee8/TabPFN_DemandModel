"""
schema_contract.py — Column-contract validation for eval/prep INSERT builders.

Validates that a Python INSERT row dict's keys are a subset of the live Snowflake
DDL columns. Call validate_insert_columns() early in any prep/eval stage function
to fail fast with a clear diagnostic rather than a silent NULL or Snowflake error.
"""
from __future__ import annotations

import os
import warnings
from typing import Iterable, Mapping


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def validate_insert_columns(
    row: Mapping[str, object],
    ddl_columns: Iterable[str],
    table_name: str = "<unknown>",
    *,
    strict: bool = False,
) -> list[str]:
    """
    Assert that all keys in `row` are known DDL columns.

    Parameters
    ----------
    row : Mapping[str, object]
        A representative INSERT row dict (keys = column names).
    ddl_columns : Iterable[str]
        Exhaustive list of column names from the DDL or live SHOW COLUMNS output.
    table_name : str
        Used in error messages only.
    strict : bool
        If True, raise ValueError on any unknown key.  If False (default),
        emit a RuntimeWarning so the job doesn't abort in production.

    Returns
    -------
    list[str]
        List of unknown column names (empty = all OK).
    """
    ddl_set = set(c.upper() for c in ddl_columns)
    unknown = [k for k in row if k.upper() not in ddl_set]
    if unknown:
        msg = (
            f"[schema_contract] Table {table_name!r}: INSERT row contains "
            f"column(s) not in DDL: {unknown}.  "
            f"This will cause a Snowflake 'invalid identifier' error.  "
            f"Known DDL columns: {sorted(ddl_set)}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return unknown


def validate_required_columns(
    row: Mapping[str, object],
    required: Iterable[str],
    table_name: str = "<unknown>",
) -> list[str]:
    """
    Warn if any required column is missing or None in the INSERT row.

    Returns list of missing/None column names (empty = all OK).
    """
    missing = [c for c in required if row.get(c) is None]
    if missing:
        warnings.warn(
            f"[schema_contract] Table {table_name!r}: required columns are None/missing: {missing}.  "
            f"Downstream aggregation or eval row-selection may silently fail.",
            RuntimeWarning,
            stacklevel=2,
        )
    return missing


# ---------------------------------------------------------------------------
# Per-table DDL column registries
# ---------------------------------------------------------------------------

# Keep in sync with the SQL DDL in sql/*_pipeline.sql.
# These are the authoritative column lists for each eval index table,
# derived directly from the CREATE TABLE statements in each pipeline file.

# ---------------------------------------------------------------------------
# LINEAR_REGRESSION_DATASET_INDEX
# Source: sql/linear_regression_numeric_pipeline.sql
# ---------------------------------------------------------------------------
LINEAR_REGRESSION_DATASET_INDEX_COLS = [
    "suite_id", "suite_family", "dataset_id", "dataset_seed", "global_idx",
    "stage_path", "prior_name", "prior_version", "prior_regime", "split_seeds",
    "n_total", "n_train_default", "n_holdout_default", "p_signal", "p_noise",
    "p_total", "target_noise_scale", "training_size_anchor", "feature_noise_level",
    "eval_weight", "payload_bytes", "created_at", "logical_dataset_key",
    "source_suite_id", "base_task_id", "perturbation_condition", "task_seed",
    "schema_version", "task_family", "distribution_family", "covariance_type",
    "rho", "target_noise_type", "is_training_allowed", "is_eval_only", "is_ood",
    "is_hidden_holdout", "difficulty_score", "difficulty_tier", "difficulty_reasons",
    "estimated_memory_bytes", "estimated_memory_gib", "memory_class",
    "hidden_holdout_suite_id", "task_fingerprint", "training_data_family",
    "task_objective",
]

# ---------------------------------------------------------------------------
# LINEAR_MIXED_REGRESSION_DATASET_INDEX
# Source: sql/linear_regression_mixed_pipeline.sql
# = LINEAR_REGRESSION_DATASET_INDEX_COLS + mixed-categorical columns
# ---------------------------------------------------------------------------
LINEAR_MIXED_REGRESSION_DATASET_INDEX_COLS = LINEAR_REGRESSION_DATASET_INDEX_COLS + [
    "p_num", "p_cat", "categorical_cardinalities", "max_cardinality",
    "cat_effect_scale", "missing_rate",
]

# ---------------------------------------------------------------------------
# LINEAR_CLASSIFICATION_DATASET_INDEX
# Source: sql/linear_classification_numeric_pipeline.sql
# Note: does NOT include prior_name / prior_version (classification uses
# prior_regime only). Does include global_idx, sample_complexity_bucket,
# coeff_schema_version, distribution_family, and difficulty/memory/holdout cols.
# ---------------------------------------------------------------------------
LINEAR_CLASSIFICATION_DATASET_INDEX_COLS = [
    "suite_id", "suite_family", "dataset_id", "dataset_seed", "global_idx",
    "stage_path", "prior_regime", "classification_regime", "split_seeds",
    "n_total", "n_train_default", "n_holdout_default", "p_signal", "p_noise",
    "p_total", "num_classes", "feature_noise_level", "training_size_anchor",
    "eval_weight", "payload_bytes", "created_at", "logical_dataset_key",
    "source_suite_id", "task_family", "task_objective", "label_noise_rate",
    "class_imbalance_type", "temperature", "margin_level", "sample_complexity_bucket",
    "task_seed", "schema_version", "coeff_schema_version", "distribution_family",
    "covariance_type", "rho", "is_training_allowed", "is_eval_only", "is_ood",
    "is_hidden_holdout", "difficulty_score", "difficulty_tier", "difficulty_reasons",
    "estimated_memory_bytes", "estimated_memory_gib", "memory_class",
    "hidden_holdout_suite_id", "task_fingerprint",
]

# ---------------------------------------------------------------------------
# LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX
# Source: sql/linear_classification_mixed_pipeline.sql
# = LINEAR_CLASSIFICATION_DATASET_INDEX_COLS + mixed cols + training_data_family
# ---------------------------------------------------------------------------
LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX_COLS = LINEAR_CLASSIFICATION_DATASET_INDEX_COLS + [
    "p_num", "p_cat", "categorical_cardinalities", "max_cardinality",
    "cat_effect_scale", "missing_rate", "training_data_family",
]

# ---------------------------------------------------------------------------
# NONLINEAR_REGRESSION_DATASET_INDEX
# Source: sql/nonlinear_regression_numeric_pipeline.sql
# ---------------------------------------------------------------------------
NONLINEAR_REGRESSION_DATASET_INDEX_COLS = [
    "suite_id", "suite_family", "dataset_id", "dataset_seed", "stage_path",
    "prior_name", "prior_version", "prior_regime", "split_seeds",
    "n_total", "n_train_default", "n_holdout_default", "p_signal", "p_noise",
    "p_total", "target_noise_scale", "training_size_anchor", "feature_noise_level",
    "eval_weight", "payload_bytes", "created_at", "logical_dataset_key",
    "source_suite_id", "feature_regime", "covariance_type", "rho",
    "active_fraction", "noise_feature_fraction", "feature_noise_sigma",
    "suite_component", "target_noise_type", "snr_target", "condition_id",
    "teacher_seed", "sample_seed", "normalization_constant",
    # These 4 columns are present in BOTH numeric and mixed tables (NULL for pure-numeric rows).
    # The numeric DDL creates them; the INSERT always includes them. The guard in
    # prepare_nonlinear_regression.py emits them unconditionally so they must be in this list.
    "p_num", "p_cat", "categorical_cardinalities", "max_cardinality",
]

# ---------------------------------------------------------------------------
# NONLINEAR_MIXED_REGRESSION_DATASET_INDEX
# Source: sql/nonlinear_regression_mixed_pipeline.sql
# = NONLINEAR_REGRESSION_DATASET_INDEX_COLS + mixed-only columns
# (p_num/p_cat/categorical_cardinalities/max_cardinality are already in the base list)
# ---------------------------------------------------------------------------
NONLINEAR_MIXED_REGRESSION_DATASET_INDEX_COLS = NONLINEAR_REGRESSION_DATASET_INDEX_COLS + [
    "cat_effect_scale", "missing_rate",
]

# ---------------------------------------------------------------------------
# NONLINEAR_CLASSIFICATION_DATASET_INDEX
# Source: sql/nonlinear_classification_numeric_pipeline.sql
# Note: shares the nonlinear-regression base columns (prior_name/version,
# feature_regime, suite_component, condition_id, teacher_seed, sample_seed)
# but replaces regression-specific cols (active_fraction, noise_feature_fraction,
# feature_noise_sigma, snr_target, normalization_constant, target_noise_type)
# with classification-specific cols (num_classes, label_noise_rate, etc.).
# ---------------------------------------------------------------------------
NONLINEAR_CLASSIFICATION_DATASET_INDEX_COLS = [
    "suite_id", "suite_family", "dataset_id", "dataset_seed", "stage_path",
    "prior_name", "prior_version", "prior_regime", "split_seeds",
    "n_total", "n_train_default", "n_holdout_default", "p_signal", "p_noise",
    "p_total", "target_noise_scale", "training_size_anchor", "feature_noise_level",
    "eval_weight", "payload_bytes", "created_at", "logical_dataset_key",
    "source_suite_id", "feature_regime", "covariance_type", "rho",
    "suite_component", "condition_id", "teacher_seed", "sample_seed",
    "num_classes", "label_noise_rate", "class_imbalance_type", "margin_level",
    "temperature", "realized_num_classes", "realized_label_noise_rate",
    "realized_margin_level",
    # These 4 columns are present in BOTH numeric and mixed tables (NULL for pure-numeric rows).
    # The numeric DDL creates them; the INSERT always includes them. The guard in
    # prepare_nonlinear_classification.py emits them unconditionally so they must be in this list.
    "p_num", "p_cat", "categorical_cardinalities", "max_cardinality",
]

# ---------------------------------------------------------------------------
# NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX
# Source: sql/nonlinear_classification_mixed_pipeline.sql
# = NONLINEAR_CLASSIFICATION_DATASET_INDEX_COLS + mixed-only columns
# (p_num/p_cat/categorical_cardinalities/max_cardinality are already in the base list)
# ---------------------------------------------------------------------------
NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX_COLS = NONLINEAR_CLASSIFICATION_DATASET_INDEX_COLS + [
    "cat_effect_scale", "missing_rate",
]

# Registry mapping table name → DDL column list
TABLE_COLUMN_REGISTRY: dict[str, list[str]] = {
    "LINEAR_REGRESSION_DATASET_INDEX": LINEAR_REGRESSION_DATASET_INDEX_COLS,
    "LINEAR_MIXED_REGRESSION_DATASET_INDEX": LINEAR_MIXED_REGRESSION_DATASET_INDEX_COLS,
    "LINEAR_CLASSIFICATION_DATASET_INDEX": LINEAR_CLASSIFICATION_DATASET_INDEX_COLS,
    "LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX": LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX_COLS,
    "NONLINEAR_REGRESSION_DATASET_INDEX": NONLINEAR_REGRESSION_DATASET_INDEX_COLS,
    "NONLINEAR_MIXED_REGRESSION_DATASET_INDEX": NONLINEAR_MIXED_REGRESSION_DATASET_INDEX_COLS,
    "NONLINEAR_CLASSIFICATION_DATASET_INDEX": NONLINEAR_CLASSIFICATION_DATASET_INDEX_COLS,
    "NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX": NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX_COLS,
}


def validate_row_against_registry(
    row: Mapping[str, object],
    table_name: str,
    *,
    strict: bool = False,
) -> list[str]:
    """Convenience wrapper using the built-in registry."""
    cols = TABLE_COLUMN_REGISTRY.get(table_name.upper())
    if cols is None:
        warnings.warn(
            f"[schema_contract] Table {table_name!r} not in registry; skipping validation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    return validate_insert_columns(row, cols, table_name, strict=strict)
