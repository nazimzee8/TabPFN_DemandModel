"""Build META_NONLINEAR_CLASSIFICATION_DATASET_INDEX from staged nonlinear classification parquet.

Same pattern as build_meta_classification_dataset_index.py but for nonlinear classification
tasks. Reads from @META_DATASET_STAGE/nonlinear/classification/numeric/{split}/ and writes to
META_NONLINEAR_CLASSIFICATION_DATASET_INDEX.
"""

from __future__ import annotations

import logging
import os
import re
import shutil

import pyarrow.parquet as pq
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)


STAGE = "@META_DATASET_STAGE"
_DATA_SUBDIR = "nonlinear/classification/numeric"
INDEX_TABLE = "META_NONLINEAR_CLASSIFICATION_DATASET_INDEX"
SPLITS = ("train", "val", "test")
HPO_BUCKETS = 40
LOCAL_ROOT = "/tmp/meta_nonlinear_classification_dataset_index"


def _expected_total() -> int:
    return int(os.getenv("META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL", "1000"))


def _expected_counts(total: int) -> dict[str, int]:
    train = int(0.8 * total)
    val = int(0.1 * total)
    return {"train": train, "val": val, "test": total - train - val}


def _row_value(row, key: str, index: int):
    if hasattr(row, "as_dict"):
        values = row.as_dict()
        return values.get(key) or values.get(key.upper()) or values.get(key.lower())
    return row[index]


def _task_id(stage_name: str) -> str:
    name = os.path.basename(stage_name.rstrip("/"))
    if not name.endswith(".parquet"):
        raise ValueError(f"Expected parquet file, got {stage_name!r}.")
    return name[:-len(".parquet")]


def _hpo_bucket(task_id: str) -> int:
    match = re.search(r"(\d+)$", task_id)
    if not match:
        raise ValueError(f"Task id has no numeric suffix: {task_id!r}.")
    return int(match.group(1)) % HPO_BUCKETS


def _list_files(session, split: str) -> list[str]:
    rows = session.sql(f"LIST {STAGE}/{_DATA_SUBDIR}/{split}/").collect()
    return sorted(
        str(_row_value(row, "name", 0)).replace("\\", "/")
        for row in rows
        if str(_row_value(row, "name", 0)).endswith(".parquet")
    )


def _download(session, split: str, task_id: str) -> str:
    local_dir = os.path.join(LOCAL_ROOT, split)
    os.makedirs(local_dir, exist_ok=True)
    filename = f"{task_id}.parquet"
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        os.remove(local_path)
    session.file.get(f"{STAGE}/{_DATA_SUBDIR}/{split}/{filename}", local_dir)
    if os.path.exists(local_path):
        return local_path
    candidates = sorted(
        os.path.join(local_dir, name)
        for name in os.listdir(local_dir)
        if name.startswith(filename)
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Downloaded file not found for {split}/{filename}.")


def _fallback_int(data: dict, key: str, default: int, local_path: str) -> int:
    """Return int(data[key][0]) if present; otherwise log a WARNING and return default."""
    if key in data:
        return int(data[key][0])
    logger.warning(
        "Nonlinear classification parquet %s is missing optional column %r — "
        "falling back to default value %d. "
        "This indicates a stale parquet written before the column was added. "
        "Regenerate the dataset to eliminate this fallback.",
        local_path, key, default,
    )
    return default


def _read_metadata(local_path: str) -> dict:
    required = {
        "n", "p", "n_train", "n_test", "prior_regime",
        "num_classes", "classification_regime", "task_objective",
    }
    optional = {
        "p_signal", "p_noise", "p_total", "class_imbalance_type",
        "label_noise_rate", "feature_noise_level", "feature_noise_level_float",
    }
    parquet = pq.ParquetFile(local_path)
    available = set(parquet.schema_arrow.names)
    missing = required - available
    if missing:
        raise ValueError(
            f"Nonlinear classification parquet {local_path} is missing metadata: {sorted(missing)}"
        )
    table = pq.read_table(local_path, columns=sorted(required | (optional & available)))
    data = table.slice(0, 1).to_pydict()
    if table.num_rows < 1:
        raise ValueError(f"Nonlinear classification parquet has no rows: {local_path}")
    if str(data["task_objective"][0]) != "inductive_classification":
        raise ValueError(
            f"Invalid task_objective in {local_path}: {data['task_objective'][0]!r}."
        )
    p = int(data["p"][0])
    feature_noise = data.get(
        "feature_noise_level_float", data.get("feature_noise_level", [0.0])
    )[0]
    return {
        "n": int(data["n"][0]),
        "p": p,
        "n_train": int(data["n_train"][0]),
        "n_test": int(data["n_test"][0]),
        "prior_regime": str(data["prior_regime"][0]),
        "num_classes": int(data["num_classes"][0]),
        "classification_regime": str(data["classification_regime"][0]),
        "task_objective": str(data["task_objective"][0]),
        "p_signal": _fallback_int(data, "p_signal", p, local_path),
        "p_noise": _fallback_int(data, "p_noise", 0, local_path),
        "p_total": _fallback_int(data, "p_total", p, local_path),
        "class_imbalance_type": str(
            data.get("class_imbalance_type", ["unknown"])[0]
        ),
        "label_noise_rate": float(data.get("label_noise_rate", [0.0])[0]),
        "feature_noise_level": float(feature_noise),
    }


def _build_rows(session) -> list[dict]:
    if os.path.exists(LOCAL_ROOT):
        shutil.rmtree(LOCAL_ROOT)
    os.makedirs(LOCAL_ROOT, exist_ok=True)
    rows = []
    for split in SPLITS:
        for stage_name in _list_files(session, split):
            task_id = _task_id(stage_name)
            metadata = _read_metadata(_download(session, split, task_id))
            rows.append({
                "split": split,
                "task_id": task_id,
                "stage_path": f"{_DATA_SUBDIR}/{split}/{task_id}.parquet",
                "hpo_bucket": _hpo_bucket(task_id),
                **metadata,
            })
    return rows


def _validate_counts(rows: list[dict]) -> dict[str, int]:
    expected = _expected_counts(_expected_total())
    actual = {split: 0 for split in SPLITS}
    for row in rows:
        actual[row["split"]] += 1
    mismatches = {
        split: {"expected": expected[split], "actual": actual[split]}
        for split in SPLITS if actual[split] != expected[split]
    }
    if mismatches:
        raise ValueError(f"{INDEX_TABLE} split count validation failed: {mismatches}")
    return actual


def _write(session, rows: list[dict]) -> None:
    schema = StructType([
        StructField("split", StringType(), nullable=False),
        StructField("task_id", StringType(), nullable=False),
        StructField("stage_path", StringType(), nullable=False),
        StructField("n", IntegerType()),
        StructField("p", IntegerType()),
        StructField("n_train", IntegerType()),
        StructField("n_test", IntegerType()),
        StructField("prior_regime", StringType()),
        StructField("hpo_bucket", IntegerType()),
        StructField("num_classes", IntegerType()),
        StructField("classification_regime", StringType()),
        StructField("task_objective", StringType()),
        StructField("p_signal", IntegerType()),
        StructField("p_noise", IntegerType()),
        StructField("p_total", IntegerType()),
        StructField("class_imbalance_type", StringType()),
        StructField("label_noise_rate", FloatType()),
        StructField("feature_noise_level", FloatType()),
    ])
    ordered = [
        tuple(row[field.name] for field in schema.fields)
        for row in rows
    ]
    live_cols = [r[0].upper() for r in session.sql(f"DESCRIBE TABLE {INDEX_TABLE}").collect()]
    expected_cols = [f.name.upper() for f in schema.fields]
    if len(live_cols) != len(expected_cols):
        raise RuntimeError(
            f"{INDEX_TABLE} has {len(live_cols)} columns but writer expects {len(expected_cols)}. "
            f"Re-deploy the DDL in sql/nonlinear_classification_numeric_pipeline.sql before running this script. "
            f"Table: {live_cols}. Writer: {expected_cols}."
        )
    session.sql(f"TRUNCATE TABLE {INDEX_TABLE}").collect()
    session.create_dataframe(ordered, schema=schema).write.mode("append").save_as_table(
        INDEX_TABLE
    )


def _preflight_k_coverage(rows: list[dict]) -> None:
    """Warn (not fail) if required K values are missing from the index rows."""
    try:
        from dgp_helpers import validate_per_split_k_coverage
    except ImportError:
        return
    required_k = [2, 3, 5, 10]
    result = validate_per_split_k_coverage(rows, required_k=required_k, strict_coverage=False)
    if not result["coverage_passed"]:
        import warnings
        warnings.warn(
            f"[build_meta_nonlinear_classification_dataset_index] K coverage incomplete. "
            f"Missing K values: {result['missing_k']}. "
            f"Realized K counts: {result['realized_k_counts']}. "
            "Consider regenerating the dataset with the missing K values covered.",
            UserWarning,
            stacklevel=2,
        )


def main() -> None:
    session = get_active_session()
    rows = _build_rows(session)
    _preflight_k_coverage(rows)
    counts = _validate_counts(rows)
    _write(session, rows)
    print(
        f"Rebuilt {INDEX_TABLE}: {counts} "
        f"(META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL={_expected_total()})"
    )


if __name__ == "__main__":
    main()
