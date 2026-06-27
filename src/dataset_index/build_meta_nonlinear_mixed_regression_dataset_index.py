"""Build META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX from staged nonlinear mixed-categorical
regression parquet files.

Same pattern as build_meta_mixed_regression_dataset_index.py but for nonlinear mixed-categorical
regression tasks. Reads from @META_DATASET_STAGE/nonlinear/regression/mixed/{split}/ and writes
to META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX.
"""

from __future__ import annotations

import json
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
    VariantType,
)


STAGE = "@META_DATASET_STAGE"
_DATA_SUBDIR = "nonlinear/regression/mixed"
INDEX_TABLE = "META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX"
SPLITS = ("train", "val", "test")
HPO_BUCKETS = 40
LOCAL_ROOT = "/tmp/meta_nonlinear_mixed_regression_dataset_index"


def _expected_total() -> int:
    return int(os.getenv("META_NONLINEAR_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL", "1000"))


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


def _read_metadata(local_path: str) -> dict:
    required = {"n", "p_num", "p_cat", "n_train", "n_test", "prior_regime"}
    parquet = pq.ParquetFile(local_path)
    available = set(parquet.schema_arrow.names)
    missing = required - available
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)} in {local_path}")
    table = parquet.read()
    d = table.to_pydict()
    p_num = int(d["p_num"][0])
    p_cat = int(d["p_cat"][0])
    return {
        "n": int(d["n"][0]),
        "p": p_num + p_cat,
        "p_num": p_num,
        "p_cat": p_cat,
        "n_train": int(d["n_train"][0]),
        "n_test": int(d["n_test"][0]),
        "prior_regime": str(d["prior_regime"][0]),
        "schema_version": str(d.get("schema_version", [None])[0] or ""),
        "task_family": str(d.get("task_family", ["synthetic_nonlinear_regression_mixed_categorical"])[0]),
        "training_data_family": str(d.get("training_data_family", [""])[0]),
        "task_objective": str(d.get("task_objective", ["inductive_regression"])[0]),
        "categorical_cardinalities": (
            list(d["categorical_cardinalities"][0])
            if "categorical_cardinalities" in d else []
        ),
    }


def build_index(session=None, expected_total: int | None = None) -> str:
    """Build META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX."""
    if session is None:
        session = get_active_session()
    total = expected_total or _expected_total()

    if os.path.exists(LOCAL_ROOT):
        shutil.rmtree(LOCAL_ROOT)
    os.makedirs(LOCAL_ROOT, exist_ok=True)

    # Guard: fail fast if the live table column count drifts from the writer's schema.
    # The writer inserts 16 explicit columns; any DDL change must be reflected here.
    _EXPECTED_COL_COUNT = 16
    _live_cols = session.sql(f"DESCRIBE TABLE {INDEX_TABLE}").collect()
    if len(_live_cols) != _EXPECTED_COL_COUNT:
        raise RuntimeError(
            f"{INDEX_TABLE} has {len(_live_cols)} columns but writer expects "
            f"{_EXPECTED_COL_COUNT} — re-deploy the DDL in "
            "sql/nonlinear_regression_mixed_pipeline.sql before running this script."
        )

    session.sql(f"DELETE FROM {INDEX_TABLE}").collect()

    rows_inserted = 0
    for split in SPLITS:
        files = _list_files(session, split)
        for stage_name in files:
            tid = _task_id(stage_name)
            local_path = _download(session, split, tid)
            meta = _read_metadata(local_path)
            stage_path = f"{_DATA_SUBDIR}/{split}/{tid}.parquet"
            bucket = _hpo_bucket(tid)
            cat_cards_json = json.dumps(meta["categorical_cardinalities"])
            session.sql(
                f"INSERT INTO {INDEX_TABLE} "
                "(split, task_id, stage_path, n, p, p_num, p_cat, n_train, n_test, "
                "prior_regime, hpo_bucket, schema_version, "
                "task_family, training_data_family, task_objective, "
                "categorical_cardinalities) "
                f"SELECT '{split}', '{tid}', '{stage_path}', "
                f"{meta['n']}, {meta['p']}, {meta['p_num']}, {meta['p_cat']}, "
                f"{meta['n_train']}, {meta['n_test']}, "
                f"'{meta['prior_regime']}', {bucket}, "
                f"'{meta['schema_version']}', "
                f"'{meta['task_family']}', '{meta['training_data_family']}', "
                f"'{meta['task_objective']}', "
                f"PARSE_JSON('{cat_cards_json}')"
            ).collect()
            rows_inserted += 1

    return f"Indexed {rows_inserted} nonlinear mixed-regression tasks across {SPLITS}."
