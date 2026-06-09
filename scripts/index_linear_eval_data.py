"""Index pre-staged linear parquet files into LINEAR_REGRESSION_DATASET_INDEX.

Upload the 1200-dataset linear evaluation suite to Snowflake stage first:
    PUT file:///<local_dir>/*.parquet
        @EVALUATION_DATASET_STAGE/linear/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

Then call from Snowflake:
    CALL run_synthetic_regression_linear_index();
"""

import os
import re

from prepare_synthetic_regression import (
    create_synreg_index_table,
    insert_synreg_index_rows,
)

# ---------------------------------------------------------------------------
# Constants — all overridable via environment variables
# ---------------------------------------------------------------------------

LINEAR_STAGE_PREFIX = "@EVALUATION_DATASET_STAGE/linear"
SUITE_ID = os.getenv(
    "SYNTHETIC_REGRESSION_SUITE_ID", "linear_stat_aware_all_regimes_v1"
)
SPLIT_SEEDS_DEFAULT = [0, 1, 2, 3, 4]
PRIOR_NAME = "linear_stat_aware"
PRIOR_VERSION = "v1"
LOCAL_DIR = "/tmp/linear_index"
FORCE_REBUILD = (
    os.getenv("SYNTHETIC_REGRESSION_FORCE_REBUILD", "false").lower()
    in ("1", "true", "yes")
)

SYNREG_INDEX_TABLE = "LINEAR_REGRESSION_DATASET_INDEX"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_linear_parquets(session) -> list:
    """Return sorted list of stage paths ending in .parquet under LINEAR_STAGE_PREFIX."""
    rows = session.sql(f"LIST {LINEAR_STAGE_PREFIX}").collect()
    paths = []
    for row in rows:
        # The LIST result has a 'name' column with the stage path
        name = row["name"] if "name" in row else str(row[0])
        # name looks like: evaluation_dataset_stage/linear/dataset_0000.parquet
        # We reconstruct the @-prefixed stage path
        if name.endswith(".parquet"):
            # Strip any leading bucket/stage path prefix to extract the relative part
            # and reconstruct as @EVALUATION_DATASET_STAGE/linear/<filename>
            filename = name.split("/")[-1]
            paths.append(f"{LINEAR_STAGE_PREFIX}/{filename}")
    return sorted(paths)


def _read_parquet_metadata(local_path: str) -> dict:
    """Read metadata columns from a parquet file using pyarrow."""
    import pyarrow.parquet as pq

    table = pq.read_table(local_path)
    schema_names = set(table.schema.names)

    def _col_scalar(name, cast=None, aliases=None):
        """Return the first value of a column, or None if absent."""
        candidates = [name] + (aliases or [])
        for col in candidates:
            if col in schema_names:
                val = table.column(col)[0].as_py()
                if val is not None and cast is not None:
                    val = cast(val)
                return val
        return None

    required_cols = ["n", "p", "n_train", "n_test", "prior_regime"]
    for col in required_cols:
        if col not in schema_names:
            raise ValueError(
                f"Required column '{col}' missing from parquet file: {local_path}"
            )

    meta = {
        "n": _col_scalar("n", int),
        "p": _col_scalar("p", int),
        "n_train": _col_scalar("n_train", int),
        "n_test": _col_scalar("n_test", int),
        "prior_regime": _col_scalar("prior_regime"),
        "profile": _col_scalar("profile"),
        "p_signal": _col_scalar("p_signal", int),
        "p_noise": _col_scalar("p_noise", int),
        "p_total": _col_scalar("p_total", int),
        "target_noise_scale": _col_scalar("target_noise_scale", float),
        "feature_noise_level": _col_scalar(
            "feature_noise_level",
            float,
            aliases=["feature_noise_level_float"],
        ),
        "covariance_type": _col_scalar("covariance_type"),
        "rho": _col_scalar("rho", float),
        "sample_complexity_bucket": _col_scalar("sample_complexity_bucket"),
    }
    return meta


def _dataset_idx_from_filename(filename: str) -> int:
    """Extract integer index from 'dataset_XXXX.parquet'."""
    m = re.search(r"dataset_(\d+)\.parquet$", filename)
    if not m:
        raise ValueError(f"Cannot parse dataset index from filename: {filename}")
    return int(m.group(1))


def _build_index_row(stage_path: str, filename: str, dataset_idx: int, meta: dict) -> dict:
    """Construct the full row dict expected by insert_synreg_index_rows."""
    n_total = meta["n"]
    n_holdout = max(1, n_total // 5)
    return {
        "suite_id": SUITE_ID,
        "suite_family": "primary",
        "dataset_id": dataset_idx,
        "dataset_seed": 0,
        "stage_path": stage_path,
        "prior_name": PRIOR_NAME,
        "prior_version": PRIOR_VERSION,
        "prior_regime": meta["prior_regime"],
        "split_seeds": SPLIT_SEEDS_DEFAULT,
        "n_total": n_total,
        "n_train_default": n_total - n_holdout,
        "n_holdout_default": n_holdout,
        "p_signal": meta.get("p_signal"),
        "p_noise": meta.get("p_noise"),
        "p_total": meta.get("p_total") or meta["p"],
        "target_noise_scale": meta.get("target_noise_scale"),
        "training_size_anchor": False,
        "feature_noise_level": meta.get("feature_noise_level"),  # float or None
        "eval_weight": 1.0,
        "payload_bytes": None,
        "logical_dataset_key": f"{SUITE_ID}_{dataset_idx:04d}",
        "source_suite_id": SUITE_ID,
    }


# ---------------------------------------------------------------------------
# Stored-procedure entry point
# ---------------------------------------------------------------------------


def index_linear_eval_data(session=None) -> str:
    """Index pre-staged linear parquet files into LINEAR_REGRESSION_DATASET_INDEX.

    Designed to run as a Snowpark stored-procedure handler.  Can also be called
    directly from Python with an existing Snowpark session.

    Returns:
        "SKIPPED"                     — suite already indexed and FORCE_REBUILD=false
        "OK suite_id=... total_rows=N" — successful run
    """
    # 1. Obtain session
    if session is None:
        from snowflake.snowpark import Session

        session = Session.builder.getOrCreate()

    # 2. Idempotency check
    if not FORCE_REBUILD:
        count_rows = session.sql(
            f"SELECT COUNT(*) AS cnt FROM {SYNREG_INDEX_TABLE} "
            f"WHERE suite_id = '{SUITE_ID}'"
        ).collect()
        existing_count = count_rows[0]["CNT"] if count_rows else 0
        if existing_count > 0:
            print(
                f"[INFO] {SYNREG_INDEX_TABLE} already contains {existing_count} rows "
                f"for suite_id='{SUITE_ID}'. Set SYNTHETIC_REGRESSION_FORCE_REBUILD=true "
                f"to overwrite. Returning SKIPPED."
            )
            return "SKIPPED"

    # 3. List parquet files
    parquet_paths = _list_linear_parquets(session)

    # 4. Guard: fail fast if nothing staged
    if not parquet_paths:
        raise RuntimeError(
            f"No .parquet files found under {LINEAR_STAGE_PREFIX}. "
            "Stage the evaluation datasets before calling this procedure."
        )
    print(f"[INFO] Found {len(parquet_paths)} parquet files under {LINEAR_STAGE_PREFIX}.")

    # 5. Ensure index table exists
    create_synreg_index_table(session)

    # 6. Delete any existing rows for this suite (safe for re-runs when FORCE_REBUILD=true)
    session.sql(
        f"DELETE FROM {SYNREG_INDEX_TABLE} WHERE suite_id = '{SUITE_ID}'"
    ).collect()
    print(f"[INFO] Cleared existing rows for suite_id='{SUITE_ID}'.")

    # 7. Download each parquet, read metadata, build index row
    os.makedirs(LOCAL_DIR, exist_ok=True)
    rows = []
    for stage_path in parquet_paths:
        filename = stage_path.split("/")[-1]
        dataset_idx = _dataset_idx_from_filename(filename)

        # Download to LOCAL_DIR
        session.file.get(stage_path, LOCAL_DIR)
        local_path = os.path.join(LOCAL_DIR, filename)

        meta = _read_parquet_metadata(local_path)
        row = _build_index_row(stage_path, filename, dataset_idx, meta)
        rows.append(row)

    print(f"[INFO] Built {len(rows)} index rows.")

    # 8. Insert
    insert_synreg_index_rows(session, rows)

    # 9. Validate
    count_rows = session.sql(
        f"SELECT COUNT(*) AS cnt FROM {SYNREG_INDEX_TABLE} "
        f"WHERE suite_id = '{SUITE_ID}'"
    ).collect()
    inserted_count = count_rows[0]["CNT"] if count_rows else 0
    if inserted_count != len(rows):
        raise RuntimeError(
            f"Row count mismatch after insert: expected {len(rows)}, "
            f"found {inserted_count} in {SYNREG_INDEX_TABLE}."
        )

    result = f"OK suite_id={SUITE_ID} total_rows={inserted_count}"
    print(f"[INFO] {result}")
    return result
