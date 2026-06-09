"""Build META_MIXED_CATEGORICAL_DATASET_INDEX from staged mixed-categorical parquet files.

Same pattern as build_meta_classification_dataset_index.py but for mixed-categorical
classification tasks. Reads from @META_CLASSIFICATION_DATASET_STAGE/mixed/{split}/
and writes to META_MIXED_CATEGORICAL_DATASET_INDEX.
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


STAGE = "@META_CLASSIFICATION_DATASET_STAGE"
_MIXED_SUBDIR = "mixed"
INDEX_TABLE = "META_MIXED_CATEGORICAL_DATASET_INDEX"
SPLITS = ("train", "val", "test")
HPO_BUCKETS = 40
LOCAL_ROOT = "/tmp/meta_mixed_categorical_dataset_index"


def _expected_total() -> int:
    return int(os.getenv("META_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL", "1000"))


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
    rows = session.sql(f"LIST {STAGE}/{_MIXED_SUBDIR}/{split}/").collect()
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
    session.file.get(f"{STAGE}/{_MIXED_SUBDIR}/{split}/{filename}", local_dir)
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
    required = {
        "n", "p_num", "p_cat", "n_train", "n_test", "prior_regime",
        "num_classes", "task_objective",
    }
    parquet = pq.ParquetFile(local_path)
    available = set(parquet.schema_arrow.names)
    missing = required - available
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)} in {local_path}")
    table = parquet.read()
    d = table.to_pydict()

    def _get(key, default=None):
        if key in d and d[key] and d[key][0] is not None:
            return d[key][0]
        return default

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
        "num_classes": int(d["num_classes"][0]),
        "task_objective": str(d["task_objective"][0]),
        "classification_regime": str(_get("prior_regime", "")),
        "schema_version": str(_get("schema_version", "")),
        "task_family": str(_get("task_family", "linear_classification")),
        "training_data_family": str(_get("training_data_family", "")),
        "class_imbalance_type": str(_get("class_imbalance_type", "balanced")),
        "label_noise_rate": float(_get("label_noise_rate", 0.0)),
        "feature_noise_level": float(_get("feature_noise_level", 0.0)),
        "temperature": float(_get("temperature", 1.0)),
        "categorical_cardinalities": (
            list(d["categorical_cardinalities"][0])
            if "categorical_cardinalities" in d else []
        ),
    }


def build_index(session=None, expected_total: int | None = None) -> str:
    """Build META_MIXED_CATEGORICAL_DATASET_INDEX."""
    if session is None:
        session = get_active_session()
    total = expected_total or _expected_total()
    counts = _expected_counts(total)

    if os.path.exists(LOCAL_ROOT):
        shutil.rmtree(LOCAL_ROOT)
    os.makedirs(LOCAL_ROOT, exist_ok=True)

    session.sql(f"DELETE FROM {INDEX_TABLE}").collect()

    rows_inserted = 0
    for split in SPLITS:
        files = _list_files(session, split)
        for stage_name in files:
            tid = _task_id(stage_name)
            local_path = _download(session, split, tid)
            meta = _read_metadata(local_path)
            stage_path = f"{STAGE}/{_MIXED_SUBDIR}/{split}/{tid}.parquet"
            bucket = _hpo_bucket(tid)
            cat_cards_json = json.dumps(meta["categorical_cardinalities"])
            session.sql(
                f"INSERT INTO {INDEX_TABLE} "
                "(split, task_id, stage_path, n, p, p_num, p_cat, n_train, n_test, "
                "prior_regime, hpo_bucket, num_classes, classification_regime, "
                "task_objective, class_imbalance_type, label_noise_rate, "
                "feature_noise_level, temperature, schema_version, "
                "task_family, training_data_family, "
                "categorical_cardinalities) "
                f"SELECT '{split}', '{tid}', '{stage_path}', "
                f"{meta['n']}, {meta['p']}, {meta['p_num']}, {meta['p_cat']}, "
                f"{meta['n_train']}, {meta['n_test']}, "
                f"'{meta['prior_regime']}', {bucket}, "
                f"{meta['num_classes']}, '{meta['classification_regime']}', "
                f"'{meta['task_objective']}', '{meta['class_imbalance_type']}', "
                f"{meta['label_noise_rate']}, {meta['feature_noise_level']}, "
                f"{meta['temperature']}, "
                f"'{meta['schema_version']}', "
                f"'{meta['task_family']}', '{meta['training_data_family']}', "
                f"PARSE_JSON('{cat_cards_json}')"
            ).collect()
            rows_inserted += 1

    return f"Indexed {rows_inserted} mixed-classification tasks across {SPLITS}."
