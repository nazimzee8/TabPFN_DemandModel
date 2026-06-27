"""Build META_NONLINEAR_REGRESSION_DATASET_INDEX from staged synthetic nonlinear parquet files.

This script is intended to run inside a Snowflake MLJob. It lists parquet files
under @META_DATASET_STAGE/nonlinear/regression/numeric/{train,val,test}/, reads scalar metadata
from each payload inside the container, and rebuilds META_NONLINEAR_REGRESSION_DATASET_INDEX in
Snowflake.
"""

import os
import re
import shutil

import pyarrow.parquet as pq
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


META_DATASET_STAGE = "@META_DATASET_STAGE"
META_DATASET_INDEX = "META_NONLINEAR_REGRESSION_DATASET_INDEX"
_DATA_SUBDIR = "nonlinear/regression/numeric"
SPLITS = ("train", "val", "test")


def _parse_expected_total(env_var: str, default: int = 1000) -> int:
    """Read expected total dataset count from env var, falling back to default."""
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return int(raw)
    return default


def _derive_split_counts(total: int) -> dict:
    """Derive per-split counts from total using 80/10/10 formula.

    train = int(0.8 * total), val = int(0.1 * total), test = total - train - val.
    For total=1000: train=800, val=100, test=100.
    """
    train = int(0.8 * total)
    val = int(0.1 * total)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


EXPECTED_TOTAL = _parse_expected_total("META_NONLINEAR_DATASET_EXPECTED_TOTAL", default=1000)
EXPECTED_COUNTS = _derive_split_counts(EXPECTED_TOTAL)
REQUIRED_METADATA_COLUMNS = ("n", "p", "n_train", "n_test", "prior_regime")
OPTIONAL_METADATA_DEFAULTS = {
    "profile": "legacy",
    "feature_regime": "legacy_gaussian",
    "p_signal": None,
    "p_noise": 0,
    "p_total": None,
    "active_s": None,
    "sparsity_ratio": None,
    "covariance_type": "iid",
    "rho": 0.0,
    "target_noise_scale": 1.0,
    "feature_noise_level": 0.0,
    "sample_complexity_bucket": "legacy",
    "has_noise_features": False,
    "has_feature_noise": False,
}
HPO_BUCKETS = 40
LOCAL_ROOT = "/tmp/meta_nonlinear_dataset_index"


def _row_value(row, key, index):
    if hasattr(row, "as_dict"):
        data = row.as_dict()
        return data.get(key) or data.get(key.upper()) or data.get(key.lower())
    if isinstance(row, dict):
        return row.get(key) or row.get(key.upper()) or row.get(key.lower())
    return row[index]


def _list_split_parquet(session, split):
    rows = session.sql(f"LIST {META_DATASET_STAGE}/{_DATA_SUBDIR}/{split}/").collect()
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
    session.file.get(f"{META_DATASET_STAGE}/{_DATA_SUBDIR}/{split}/{filename}", local_dir)
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
    parquet_file = pq.ParquetFile(local_path)
    available_columns = set(parquet_file.schema_arrow.names)
    requested_columns = [
        column
        for column in REQUIRED_METADATA_COLUMNS + tuple(OPTIONAL_METADATA_DEFAULTS)
        if column in available_columns
    ]
    missing_required = [
        column for column in REQUIRED_METADATA_COLUMNS if column not in available_columns
    ]
    if missing_required:
        raise ValueError(
            f"Parquet file {local_path} is missing required metadata columns: "
            f"{missing_required}"
        )
    table = pq.read_table(local_path, columns=requested_columns)
    if table.num_rows < 1:
        raise ValueError(f"Parquet file has no rows: {local_path}")
    data = table.slice(0, 1).to_pydict()
    p = int(data["p"][0])
    p_signal = data.get("p_signal", [None])[0]
    p_noise = data.get("p_noise", [0])[0]
    p_total = data.get("p_total", [None])[0]
    active_s = data.get("active_s", [None])[0]
    if p_signal is None:
        p_signal = p
    if p_total is None:
        p_total = p
    if active_s is None:
        active_s = p_signal
    sparsity_ratio = data.get("sparsity_ratio", [None])[0]
    if sparsity_ratio is None:
        sparsity_ratio = float(active_s) / float(max(1, int(p_signal)))

    return {
        "n": int(data["n"][0]),
        "p": p,
        "n_train": int(data["n_train"][0]),
        "n_test": int(data["n_test"][0]),
        "prior_regime": str(data["prior_regime"][0]),
        "profile": str(data.get("profile", [OPTIONAL_METADATA_DEFAULTS["profile"]])[0]),
        "feature_regime": str(data.get(
            "feature_regime", [OPTIONAL_METADATA_DEFAULTS["feature_regime"]]
        )[0]),
        "p_signal": int(p_signal),
        "p_noise": int(p_noise),
        "p_total": int(p_total),
        "active_s": int(active_s),
        "sparsity_ratio": float(sparsity_ratio),
        "covariance_type": str(data.get(
            "covariance_type", [OPTIONAL_METADATA_DEFAULTS["covariance_type"]]
        )[0]),
        "rho": float(data.get("rho", [OPTIONAL_METADATA_DEFAULTS["rho"]])[0]),
        "target_noise_scale": float(data.get(
            "target_noise_scale", [OPTIONAL_METADATA_DEFAULTS["target_noise_scale"]]
        )[0]),
        "feature_noise_level": float(data.get(
            "feature_noise_level", [OPTIONAL_METADATA_DEFAULTS["feature_noise_level"]]
        )[0]),
        "sample_complexity_bucket": str(data.get(
            "sample_complexity_bucket",
            [OPTIONAL_METADATA_DEFAULTS["sample_complexity_bucket"]],
        )[0]),
        "has_noise_features": bool(data.get(
            "has_noise_features", [int(p_noise) > 0]
        )[0]),
        "has_feature_noise": bool(data.get(
            "has_feature_noise",
            [float(data.get("feature_noise_level", [0.0])[0]) > 0.0],
        )[0]),
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
                "stage_path": f"{_DATA_SUBDIR}/{split}/{task_id}.parquet",
                "n": metadata["n"],
                "p": metadata["p"],
                "n_train": metadata["n_train"],
                "n_test": metadata["n_test"],
                "prior_regime": metadata["prior_regime"],
                "hpo_bucket": _hpo_bucket(task_id),
                "profile": metadata["profile"],
                "feature_regime": metadata["feature_regime"],
                "p_signal": metadata["p_signal"],
                "p_noise": metadata["p_noise"],
                "p_total": metadata["p_total"],
                "active_s": metadata["active_s"],
                "sparsity_ratio": metadata["sparsity_ratio"],
                "covariance_type": metadata["covariance_type"],
                "rho": metadata["rho"],
                "target_noise_scale": metadata["target_noise_scale"],
                "feature_noise_level": metadata["feature_noise_level"],
                "sample_complexity_bucket": metadata["sample_complexity_bucket"],
                "has_noise_features": metadata["has_noise_features"],
                "has_feature_noise": metadata["has_feature_noise"],
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
        StructField("profile", StringType()),
        StructField("feature_regime", StringType()),
        StructField("p_signal", IntegerType()),
        StructField("p_noise", IntegerType()),
        StructField("p_total", IntegerType()),
        StructField("active_s", IntegerType()),
        StructField("sparsity_ratio", FloatType()),
        StructField("covariance_type", StringType()),
        StructField("rho", FloatType()),
        StructField("target_noise_scale", FloatType()),
        StructField("feature_noise_level", FloatType()),
        StructField("sample_complexity_bucket", StringType()),
        StructField("has_noise_features", BooleanType()),
        StructField("has_feature_noise", BooleanType()),
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
            row["profile"],
            row["feature_regime"],
            row["p_signal"],
            row["p_noise"],
            row["p_total"],
            row["active_s"],
            row["sparsity_ratio"],
            row["covariance_type"],
            row["rho"],
            row["target_noise_scale"],
            row["feature_noise_level"],
            row["sample_complexity_bucket"],
            row["has_noise_features"],
            row["has_feature_noise"],
        )
        for row in rows
    ]
    # Guard: verify live table schema matches writer before we truncate
    live_cols = [r[0].upper() for r in session.sql(f"DESCRIBE TABLE {META_DATASET_INDEX}").collect()]
    expected_cols = [f.name.upper() for f in schema.fields]
    if len(live_cols) != len(expected_cols):
        raise RuntimeError(
            f"{META_DATASET_INDEX} has {len(live_cols)} columns but writer expects {len(expected_cols)}. "
            f"Re-deploy the DDL in sql/nonlinear_regression_numeric_pipeline.sql before running this script. "
            f"Table: {live_cols}. Writer: {expected_cols}."
        )
    session.sql(f"TRUNCATE TABLE {META_DATASET_INDEX}").collect()
    session.create_dataframe(ordered_rows, schema=schema).write.mode("append").save_as_table(
        META_DATASET_INDEX
    )


def main():
    session = get_active_session()
    rows = _build_rows(session)
    counts = _validate_counts(rows)
    _write_index(session, rows)
    print(
        f"Rebuilt {META_DATASET_INDEX}: {counts} "
        f"(META_NONLINEAR_DATASET_EXPECTED_TOTAL={EXPECTED_TOTAL})"
    )


if __name__ == "__main__":
    main()
