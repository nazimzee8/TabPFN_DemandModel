"""Snowflake-only stage materialization helpers."""

import glob
import os
import posixpath

META_DATASET_STAGE = "@META_DATASET_STAGE"
META_DATASET_INDEX = "META_DATASET_INDEX"
META_NONLINEAR_DATASET_STAGE = "@META_NONLINEAR_DATASET_STAGE"
META_NONLINEAR_DATASET_INDEX = "META_NONLINEAR_DATASET_INDEX"
NONLINEAR_TRAINING_FAMILY = "synthetic_regression_nonlinear"
DEFAULT_SPLITS = ("train", "val", "test")
RUNTIME_INDEX_COLUMNS = ("split", "task_id", "stage_path", "p", "n_train")
HPO_INDEX_COLUMNS = RUNTIME_INDEX_COLUMNS + ("hpo_bucket", "prior_regime")


def _resolve_index_table_and_stage(training_data_family=None):
    """Return (index_table, stage) based on TRAINING_DATA_FAMILY env var or explicit arg."""
    family = training_data_family or os.getenv("TRAINING_DATA_FAMILY", "")
    if family == NONLINEAR_TRAINING_FAMILY:
        return META_NONLINEAR_DATASET_INDEX, META_NONLINEAR_DATASET_STAGE
    return META_DATASET_INDEX, META_DATASET_STAGE


def _get_active_session_or_none():
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        pass
    try:
        from snowflake.snowpark import Session
        return Session.builder.getOrCreate()
    except Exception:
        return None


def materialize_meta_dataset_stage(local_root="/tmp/data", splits=DEFAULT_SPLITS):
    """
    Materialize staged meta-dataset parquet splits inside a Snowflake MLJob.

    This intentionally does nothing outside Snowflake. It must not be used to pull
    stage contents onto a developer workstation.

    Routes to META_NONLINEAR_DATASET_STAGE when TRAINING_DATA_FAMILY=synthetic_regression_nonlinear.
    """
    _index_table, _stage = _resolve_index_table_and_stage()
    session = _get_active_session_or_none()
    if session is None:
        print(
            "[INFO] No active Snowflake session; not downloading "
            f"{_stage}. Using existing local files under {local_root}."
        )
        return local_root

    for split in splits:
        split_dir = os.path.join(local_root, split)
        os.makedirs(split_dir, exist_ok=True)
        session.file.get(f"{_stage}/{split}/", split_dir)
        parquet_files = [
            name for name in os.listdir(split_dir)
            if name.endswith(".parquet")
        ]
        if not parquet_files:
            raise FileNotFoundError(
                f"No .parquet files materialized for split {split!r} in {split_dir}"
            )
        print(f"Materialized {len(parquet_files)} {split} parquet files to {split_dir}")

    return local_root


def _quote_sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def _row_to_dict(row):
    if hasattr(row, "as_dict"):
        return {str(k).lower(): v for k, v in row.as_dict().items()}
    if isinstance(row, dict):
        return {str(k).lower(): v for k, v in row.items()}
    raise TypeError(f"Unsupported Snowflake row type: {type(row)!r}")


def _local_split_files(local_root, split, limit=None):
    files = sorted(glob.glob(os.path.join(local_root, split, "*.parquet")))
    if limit is not None:
        files = files[: int(limit)]
    return files


def _counts_by_split(rows):
    counts = {}
    for row in rows:
        if not isinstance(row, dict):
            row = _row_to_dict(row)
        split = row.get("split")
        split = "<missing>" if split is None or split == "" else str(split)
        counts[split] = counts.get(split, 0) + 1
    return counts


def _local_index_rows(local_root, splits, split_limits=None):
    split_limits = dict(split_limits or {})
    rows = []
    for split in splits:
        for i, path in enumerate(_local_split_files(local_root, split, split_limits.get(split))):
            p = 1
            n_train = 1
            try:
                import pyarrow.parquet as pq

                values = pq.read_table(path, columns=["p", "n_train"]).to_pydict()
                p = int(values["p"][0])
                n_train = int(values["n_train"][0])
            except Exception:
                pass
            rows.append({
                "split": split,
                "task_id": i,
                "stage_path": f"{split}/{os.path.basename(path)}",
                "local_path": path,
                "p": p,
                "n_train": n_train,
                "hpo_bucket": i,
                "prior_regime": "",
            })
    if split_limits:
        _validate_index_rows(rows, RUNTIME_INDEX_COLUMNS, split_limits=split_limits)
    return rows


def _normalize_stage_path(stage_path, split=None):
    if stage_path is None:
        raise ValueError("META_DATASET_INDEX row has NULL stage_path")
    path = str(stage_path).replace("\\", "/").strip()
    prefix = f"{META_DATASET_STAGE}/"
    if path.upper().startswith(prefix.upper()):
        path = path[len(prefix):]
    path = path.lstrip("/")
    if not path:
        raise ValueError("META_DATASET_INDEX row has empty stage_path")
    if split and not path.startswith(f"{split}/"):
        path = posixpath.join(str(split), posixpath.basename(path))
    return path


def _stage_file_path(stage_path, split=None):
    return f"{META_DATASET_STAGE}/{_normalize_stage_path(stage_path, split=split)}"


def _validate_index_rows(rows, required_columns, split_limits=None):
    if not rows:
        raise ValueError(f"{META_DATASET_INDEX} returned no rows")

    missing = []
    invalid = []
    counts = {}
    for i, row in enumerate(rows):
        split = row.get("split")
        if split:
            counts[split] = counts.get(split, 0) + 1
        for column in required_columns:
            if column not in row:
                missing.append(column)
            elif row[column] is None or row[column] == "":
                invalid.append((i, column))
        for column in ("p", "n_train"):
            if column in row and row[column] is not None and int(row[column]) <= 0:
                invalid.append((i, column))

    if missing:
        raise ValueError(
            f"{META_DATASET_INDEX} query is missing required columns: "
            f"{sorted(set(missing))}"
        )
    if invalid:
        preview = ", ".join(f"row {i} {col}" for i, col in invalid[:10])
        raise ValueError(f"{META_DATASET_INDEX} returned invalid required fields: {preview}")
    if split_limits:
        too_few = {
            split: {"expected": int(limit), "actual": counts.get(split, 0)}
            for split, limit in split_limits.items()
            if counts.get(split, 0) < int(limit)
        }
        if too_few:
            raise ValueError(f"{META_DATASET_INDEX} returned too few rows: {too_few}")


def select_meta_dataset_index_rows(
    splits=("train", "val"),
    split_limits=None,
    hpo_subset=False,
    session=None,
):
    """
    Select deterministic runtime rows from META_DATASET_INDEX (or META_NONLINEAR_DATASET_INDEX
    when TRAINING_DATA_FAMILY=synthetic_regression_nonlinear).

    Outside Snowflake, returns synthetic rows for existing local parquet files so
    developer runs never download stage contents to the workstation.
    """
    _index_table, _stage = _resolve_index_table_and_stage()
    session = session or _get_active_session_or_none()
    split_limits = dict(split_limits or {})
    splits = tuple(splits)
    required_columns = HPO_INDEX_COLUMNS if hpo_subset else RUNTIME_INDEX_COLUMNS

    if session is None:
        return _local_index_rows("/tmp/data", splits, split_limits=split_limits)

    split_sql = ", ".join(_quote_sql_string(split) for split in splits)
    columns_sql = ", ".join(required_columns)
    if hpo_subset:
        limit_clauses = [
            f"(split = {_quote_sql_string(split)} AND split_rank <= {int(limit)})"
            for split, limit in split_limits.items()
        ]
        if not limit_clauses:
            raise ValueError("hpo_subset=True requires split_limits")
        sql = f"""
WITH ranked AS (
    SELECT {columns_sql},
           ROW_NUMBER() OVER (
               PARTITION BY split, hpo_bucket
               ORDER BY prior_regime, p, n_train, task_id
           ) AS bucket_rank
    FROM {_index_table}
    WHERE split IN ({split_sql})
),
ordered AS (
    SELECT {columns_sql}, bucket_rank,
           ROW_NUMBER() OVER (
               PARTITION BY split
               ORDER BY bucket_rank, hpo_bucket, prior_regime, p, n_train, task_id
           ) AS split_rank
    FROM ranked
)
SELECT {columns_sql}
FROM ordered
WHERE {" OR ".join(limit_clauses)}
ORDER BY split, split_rank
"""
    else:
        sql = f"""
SELECT {columns_sql}
FROM {_index_table}
WHERE split IN ({split_sql})
ORDER BY split, task_id
"""

    try:
        rows = [_row_to_dict(row) for row in session.sql(sql).collect()]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query {_index_table}; create and populate the index "
            "before launching Snowflake training jobs."
        ) from exc

    _validate_index_rows(rows, required_columns, split_limits=split_limits)
    return rows


def select_rank_sharded_index_rows(split, rank, world_size, session=None):
    """
    Select the META_DATASET_INDEX rows owned by one DDP rank.

    This is the production training sharding path. It avoids
    ShardedDataConnector worker-side shard conversion and lets Snowflake do the
    deterministic split assignment with ROW_NUMBER() and MOD().
    """
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive; got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}); got {rank}")

    session = session or _get_active_session_or_none()
    split_sql = _quote_sql_string(split)

    if session is None:
        rows = _local_index_rows("/tmp/data", (split,))
        selected = [
            row for i, row in enumerate(sorted(rows, key=lambda row: row["task_id"]))
            if i % world_size == rank
        ]
        _validate_index_rows(selected, RUNTIME_INDEX_COLUMNS)
        return selected

    sql = f"""
SELECT split, task_id, stage_path, p, n_train
FROM (
  SELECT
    split, task_id, stage_path, p, n_train,
    ROW_NUMBER() OVER (PARTITION BY split ORDER BY task_id) - 1 AS rn
  FROM {META_DATASET_INDEX}
  WHERE split = {split_sql}
)
WHERE MOD(rn, {world_size}) = {rank}
ORDER BY task_id
"""
    try:
        rows = [_row_to_dict(row) for row in session.sql(sql).collect()]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query rank-sharded {META_DATASET_INDEX} rows for "
            f"split={split!r}, rank={rank}, world_size={world_size}."
        ) from exc

    _validate_index_rows(rows, RUNTIME_INDEX_COLUMNS)
    return rows


def materialize_indexed_meta_dataset(
    local_root="/tmp/data",
    splits=("train", "val"),
    split_limits=None,
    hpo_subset=False,
    rows=None,
):
    """
    Materialize selected META_DATASET_INDEX rows into DATA_DIR/<split>/.

    Returns a dict mapping each split to sorted local parquet paths.
    """
    session = _get_active_session_or_none()
    split_limits = dict(split_limits or {})
    splits = tuple(splits)
    provided_rows = None if rows is None else list(rows)

    if session is None:
        if provided_rows is not None:
            raise RuntimeError(
                "Snowflake stage materialization for explicit "
                f"{META_DATASET_INDEX} rows requires an active Snowflake session "
                "inside the Ray worker. Local fallback is disabled when "
                "rows=... is provided because the selected rows reference staged "
                "parquet files, not developer-local files. "
                f"local_root={local_root!r}, requested_splits={splits!r}, "
                f"split_limits={split_limits!r}, "
                f"provided_counts_by_split={_counts_by_split(provided_rows)!r}. "
                "Previous misleading symptom: Local fallback needs at least 200 "
                "train parquet files under /tmp/data/train; found 0."
            )
        result = {
            split: _local_split_files(local_root, split, split_limits.get(split))
            for split in splits
        }
        for split, limit in split_limits.items():
            if len(result.get(split, [])) < int(limit):
                raise FileNotFoundError(
                    f"Local fallback needs at least {limit} {split} parquet files under "
                    f"{os.path.join(local_root, split)}; found {len(result.get(split, []))}"
                )
        return result

    if provided_rows is not None:
        selected_rows = provided_rows
    else:
        selected_rows = select_meta_dataset_index_rows(
            splits=splits,
            split_limits=split_limits,
            hpo_subset=hpo_subset,
            session=session,
        )
    selected_by_split = {split: [] for split in splits}
    for row in selected_rows:
        split = str(row["split"])
        normalized = _normalize_stage_path(row["stage_path"], split=split)
        local_path = os.path.join(local_root, *normalized.split("/"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            session.file.get(_stage_file_path(row["stage_path"], split=split), os.path.dirname(local_path))
        if not os.path.exists(local_path):
            candidates = glob.glob(os.path.join(os.path.dirname(local_path), os.path.basename(local_path) + "*"))
            if candidates:
                local_path = sorted(candidates)[0]
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Failed to materialize indexed parquet {_stage_file_path(row['stage_path'], split=split)}"
            )
        selected_by_split.setdefault(split, []).append(local_path)

    for split in selected_by_split:
        selected_by_split[split] = sorted(selected_by_split[split])
    return selected_by_split


def materialize_connector_shard(shard, local_root, split):
    """
    Download Parquet files for a single ShardedDataConnector shard to local_root.

    shard : DataConnector returned by dataset_map[split].get_shard()
    Returns sorted list of local file paths.
    """
    session = _get_active_session_or_none()
    if session is None:
        raise RuntimeError(
            "materialize_connector_shard requires an active Snowflake session."
        )
    df = shard.to_pandas()
    # Snowflake returns uppercase column names; normalise to lowercase for safety
    df.columns = [c.lower() for c in df.columns]

    local_files = []
    for _, row in df.iterrows():
        stage_path_val = str(row["stage_path"])
        normalized = _normalize_stage_path(stage_path_val, split=split)
        local_path = os.path.join(local_root, *normalized.split("/"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            session.file.get(
                _stage_file_path(stage_path_val, split=split),
                os.path.dirname(local_path),
            )
        if not os.path.exists(local_path):
            candidates = glob.glob(local_path + "*")
            if candidates:
                local_path = sorted(candidates)[0]
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"[materialize_connector_shard] rank shard file not found: {stage_path_val}"
            )
        local_files.append(local_path)
    return sorted(local_files)
