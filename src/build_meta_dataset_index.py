"""Build META_DATASET_INDEX from staged synthetic parquet files.

This script is intended to run inside a Snowflake MLJob. It lists parquet files
under @META_DATASET_STAGE/{train,val,test}/, reads scalar metadata from each
payload inside the container, and rebuilds META_DATASET_INDEX in Snowflake.
"""

import os
import re
import shutil

import pyarrow.parquet as pq
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)


META_DATASET_STAGE = "@META_DATASET_STAGE"
META_DATASET_INDEX = "META_DATASET_INDEX"
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 800, "val": 100, "test": 100}
REQUIRED_METADATA_COLUMNS = ("n", "p", "n_train", "n_test", "prior_regime")
HPO_BUCKETS = 40
LOCAL_ROOT = "/tmp/meta_dataset_index"


def _row_value(row, key, index):
    if hasattr(row, "as_dict"):
        data = row.as_dict()
        return data.get(key) or data.get(key.upper()) or data.get(key.lower())
    if isinstance(row, dict):
        return row.get(key) or row.get(key.upper()) or row.get(key.lower())
    return row[index]


def _list_split_parquet(session, split):
    rows = session.sql(f"LIST {META_DATASET_STAGE}/{split}/").collect()
    names = []
    for row in rows:
        name = str(_row_value(row, "name", 0)).replace("\\", "/")
        if name.endswith(".parquet"):
            names.append(name)
    return sorted(names)


def _task_id_from_name(stage_name):
    basename = os.path.basename(stage_name.rstrip("/"))
    if not basename.endswith(".parquet"):
        raise ValueError(f"Expected parquet file, got {stage_name!r}")
    return basename[:-len(".parquet")]


def _hpo_bucket(task_id):
    match = re.search(r"(\d+)$", task_id)
    if not match:
        raise ValueError(f"Cannot compute hpo_bucket; task_id has no numeric suffix: {task_id!r}")
    return int(match.group(1)) % HPO_BUCKETS


def _download_stage_file(session, split, task_id):
    local_dir = os.path.join(LOCAL_ROOT, split)
    os.makedirs(local_dir, exist_ok=True)
    filename = f"{task_id}.parquet"
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        os.remove(local_path)
    session.file.get(f"{META_DATASET_STAGE}/{split}/{filename}", local_dir)
    if os.path.exists(local_path):
        return local_path
    candidates = [
        os.path.join(local_dir, name)
        for name in os.listdir(local_dir)
        if name.startswith(filename)
    ]
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(f"Downloaded file not found for {split}/{filename}")


def _read_metadata(local_path):
    table = pq.read_table(local_path, columns=list(REQUIRED_METADATA_COLUMNS))
    if table.num_rows < 1:
        raise ValueError(f"Parquet file has no rows: {local_path}")
    data = table.slice(0, 1).to_pydict()
    return {
        "n": int(data["n"][0]),
        "p": int(data["p"][0]),
        "n_train": int(data["n_train"][0]),
        "n_test": int(data["n_test"][0]),
        "prior_regime": str(data["prior_regime"][0]),
    }


def _build_rows(session):
    rows = []
    if os.path.exists(LOCAL_ROOT):
        shutil.rmtree(LOCAL_ROOT)
    os.makedirs(LOCAL_ROOT, exist_ok=True)

    for split in SPLITS:
        for stage_name in _list_split_parquet(session, split):
            task_id = _task_id_from_name(stage_name)
            local_path = _download_stage_file(session, split, task_id)
            metadata = _read_metadata(local_path)
            rows.append({
                "split": split,
                "task_id": task_id,
                "stage_path": f"{split}/{task_id}.parquet",
                "n": metadata["n"],
                "p": metadata["p"],
                "n_train": metadata["n_train"],
                "n_test": metadata["n_test"],
                "prior_regime": metadata["prior_regime"],
                "hpo_bucket": _hpo_bucket(task_id),
            })
    return rows


def _validate_counts(rows):
    counts = {split: 0 for split in SPLITS}
    for row in rows:
        counts[row["split"]] += 1
    mismatches = {
        split: {"expected": expected, "actual": counts.get(split, 0)}
        for split, expected in EXPECTED_COUNTS.items()
        if counts.get(split, 0) != expected
    }
    if mismatches:
        raise ValueError(f"{META_DATASET_INDEX} split count validation failed: {mismatches}")
    return counts


def _write_index(session, rows):
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
    ])
    ordered_rows = [
        (
            row["split"],
            row["task_id"],
            row["stage_path"],
            row["n"],
            row["p"],
            row["n_train"],
            row["n_test"],
            row["prior_regime"],
            row["hpo_bucket"],
        )
        for row in rows
    ]
    session.sql(f"TRUNCATE TABLE {META_DATASET_INDEX}").collect()
    session.create_dataframe(ordered_rows, schema=schema).write.mode("append").save_as_table(
        META_DATASET_INDEX
    )


def main():
    session = get_active_session()
    rows = _build_rows(session)
    counts = _validate_counts(rows)
    _write_index(session, rows)
    print(f"Rebuilt {META_DATASET_INDEX}: {counts}")


if __name__ == "__main__":
    main()
