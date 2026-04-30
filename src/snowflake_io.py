"""Snowflake-only stage materialization helpers."""

import os

META_DATASET_STAGE = "@META_DATASET_STAGE"
DEFAULT_SPLITS = ("train", "val", "test")


def _get_active_session_or_none():
    try:
        from snowflake.snowpark.context import get_active_session

        return get_active_session()
    except Exception:
        return None


def materialize_meta_dataset_stage(local_root="/tmp/data", splits=DEFAULT_SPLITS):
    """
    Materialize staged meta-dataset parquet splits inside a Snowflake MLJob.

    This intentionally does nothing outside Snowflake. It must not be used to pull
    @META_DATASET_STAGE contents onto a developer workstation.
    """
    session = _get_active_session_or_none()
    if session is None:
        print(
            "[INFO] No active Snowflake session; not downloading "
            f"{META_DATASET_STAGE}. Using existing local files under {local_root}."
        )
        return local_root

    for split in splits:
        split_dir = os.path.join(local_root, split)
        os.makedirs(split_dir, exist_ok=True)
        session.file.get(f"{META_DATASET_STAGE}/{split}/", split_dir)
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
