"""Tests for src/validation/schema_contract.py."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from validation.schema_contract import (
    validate_insert_columns,
    validate_required_columns,
    validate_row_against_registry,
    TABLE_COLUMN_REGISTRY,
)


class TestValidateInsertColumns:
    def test_valid_row_passes(self):
        ddl = ["id", "name", "value"]
        row = {"id": 1, "name": "foo"}
        unknown = validate_insert_columns(row, ddl, "test_table")
        assert unknown == []

    def test_unknown_column_warns(self):
        ddl = ["id", "name"]
        row = {"id": 1, "typo_col": "x"}
        with pytest.warns(RuntimeWarning, match="typo_col"):
            unknown = validate_insert_columns(row, ddl, "test_table")
        assert "typo_col" in unknown

    def test_unknown_column_strict_raises(self):
        ddl = ["id"]
        row = {"id": 1, "bad": "x"}
        with pytest.raises(ValueError, match="bad"):
            validate_insert_columns(row, ddl, strict=True)

    def test_case_insensitive(self):
        ddl = ["DATASET_ID", "STAGE_PATH"]
        row = {"dataset_id": 1, "stage_path": "x/y.parquet"}
        unknown = validate_insert_columns(row, ddl)
        assert unknown == []

    def test_old_nonlinear_reg_column_names_detected(self):
        """Regression: the old wrong column names (dataset_idx, filename, target_family)
        must NOT appear in the registry — they would have caused INSERT failures."""
        cols = TABLE_COLUMN_REGISTRY.get("NONLINEAR_REGRESSION_DATASET_INDEX", [])
        col_upper = {c.upper() for c in cols}
        assert "DATASET_IDX" not in col_upper, "dataset_idx is wrong name — should be dataset_id"
        assert "FILENAME" not in col_upper, "filename is wrong name — should be stage_path"
        assert "TARGET_FAMILY" not in col_upper, "target_family doesn't exist in DDL"
        assert "DATASET_ID" in col_upper
        assert "STAGE_PATH" in col_upper


class TestValidateRequiredColumns:
    def test_missing_stage_path_warns(self):
        row = {"dataset_id": 1, "stage_path": None, "n_total": 100}
        with pytest.warns(RuntimeWarning, match="stage_path"):
            missing = validate_required_columns(row, ["dataset_id", "stage_path"], "test")
        assert "stage_path" in missing

    def test_all_present_passes(self):
        row = {"dataset_id": 1, "stage_path": "x/y.parquet"}
        missing = validate_required_columns(row, ["dataset_id", "stage_path"])
        assert missing == []


class TestRegistry:
    def test_all_eight_tables_registered(self):
        expected = {
            "LINEAR_REGRESSION_DATASET_INDEX",
            "LINEAR_MIXED_REGRESSION_DATASET_INDEX",
            "LINEAR_CLASSIFICATION_DATASET_INDEX",
            "LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX",
            "NONLINEAR_REGRESSION_DATASET_INDEX",
            "NONLINEAR_MIXED_REGRESSION_DATASET_INDEX",
            "NONLINEAR_CLASSIFICATION_DATASET_INDEX",
            "NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX",
        }
        assert expected.issubset(set(TABLE_COLUMN_REGISTRY.keys()))

    def test_mixed_tables_have_more_cols_than_numeric(self):
        assert len(TABLE_COLUMN_REGISTRY["LINEAR_MIXED_REGRESSION_DATASET_INDEX"]) > \
               len(TABLE_COLUMN_REGISTRY["LINEAR_REGRESSION_DATASET_INDEX"])
        assert len(TABLE_COLUMN_REGISTRY["NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX"]) > \
               len(TABLE_COLUMN_REGISTRY["NONLINEAR_CLASSIFICATION_DATASET_INDEX"])
