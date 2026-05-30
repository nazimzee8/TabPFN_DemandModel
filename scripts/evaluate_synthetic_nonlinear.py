"""
evaluate_synthetic_nonlinear.py
================================
MLJob entrypoint that creates and populates SYNTHETIC_NONLINEAR_DATASET_INDEX
from a staged nonlinear evaluation manifest.

Purpose
-------
Reads nonlinear_manifest.json from @EVALUATION_DATASET_STAGE/nonlinear/, selects
NONLINEAR_N_DATASETS datasets (balanced across regimes I/J/K/L), and inserts rows
into SYNTHETIC_NONLINEAR_DATASET_INDEX isolated by suite_id = NONLINEAR_SUITE_ID.

Safety invariants
-----------------
* SYNTHETIC_NONLINEAR_DATASET_INDEX is NEVER DROPped.
  _truncate_nonlinear_index() uses DELETE WHERE suite_id = NONLINEAR_SUITE_ID only.
* Other suite rows are never touched.
* create_nonlinear_index_table() is always called before _truncate_nonlinear_index()
  to ensure the table exists before any DELETE.

Environment variables
---------------------
NONLINEAR_SUITE_ID      override suite_id (default: nonlinear_v1)
NONLINEAR_N_DATASETS    number of datasets to index (default: 400)
NONLINEAR_LOCAL_DIR     local scratch dir (default: /tmp/nonlinear_prep)
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: supports both Snowflake flat staging and local dev nested layout.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))          # Snowflake: same staging dir
sys.path.insert(0, str(_HERE.parent))   # local dev: scripts/ parent

from prepare_synthetic_regression import (
    _compute_n_train_default,
    _compute_n_holdout_default,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NONLINEAR_SUITE_ID      = os.getenv("NONLINEAR_SUITE_ID", "nonlinear_v1")
NONLINEAR_N_DATASETS    = int(os.getenv("NONLINEAR_N_DATASETS", "400"))
NONLINEAR_REGIMES       = ["I", "J", "K", "L"]
NONLINEAR_SPLIT_SEEDS   = [0, 1, 2]
NONLINEAR_PRIOR_NAME    = "nonlinear"
NONLINEAR_PRIOR_VERSION = "v1"
NONLINEAR_INDEX_TABLE   = "SYNTHETIC_NONLINEAR_DATASET_INDEX"
EVAL_STAGE_PREFIX       = "@EVALUATION_DATASET_STAGE/nonlinear"
NONLINEAR_MANIFEST_PATH = f"{EVAL_STAGE_PREFIX}/nonlinear_manifest.json"
NONLINEAR_LOCAL_DIR     = os.getenv("NONLINEAR_LOCAL_DIR", "/tmp/nonlinear_prep")

if NONLINEAR_N_DATASETS <= 0:
    raise ValueError(
        f"NONLINEAR_N_DATASETS must be positive, got {NONLINEAR_N_DATASETS}."
    )
if NONLINEAR_N_DATASETS % 4 != 0:
    raise ValueError(
        f"NONLINEAR_N_DATASETS={NONLINEAR_N_DATASETS} is not divisible by 4 "
        "(one per nonlinear regime I/J/K/L)."
    )


# ---------------------------------------------------------------------------
# Index table DDL
# ---------------------------------------------------------------------------

def create_nonlinear_index_table(session) -> None:
    """Create SYNTHETIC_NONLINEAR_DATASET_INDEX if not already present.

    Schema is identical to SYNTHETIC_REGRESSION_DATASET_INDEX.
    """
    ddl = f"""
    CREATE TRANSIENT TABLE IF NOT EXISTS {NONLINEAR_INDEX_TABLE} (
      suite_id             STRING,
      suite_family         STRING,
      dataset_id           NUMBER,
      dataset_seed         NUMBER,
      stage_path           STRING,
      prior_name           STRING,
      prior_version        STRING,
      prior_regime         STRING,
      split_seeds          ARRAY,
      n_total              NUMBER,
      n_train_default      NUMBER,
      n_holdout_default    NUMBER,
      p_signal             NUMBER,
      p_noise              NUMBER,
      p_total              NUMBER,
      target_noise_scale   FLOAT,
      training_size_anchor BOOLEAN,
      feature_noise_level  NUMBER,
      eval_weight          FLOAT,
      payload_bytes        NUMBER,
      created_at           TIMESTAMP_NTZ,
      logical_dataset_key  STRING,
      source_suite_id      STRING
    ) DATA_RETENTION_TIME_IN_DAYS = 0
    """
    session.sql(ddl).collect()
    print(f"[INFO] Index table {NONLINEAR_INDEX_TABLE} ready.", flush=True)


# ---------------------------------------------------------------------------
# Row truncation (never drops table)
# ---------------------------------------------------------------------------

def _truncate_nonlinear_index(session) -> None:
    """Delete nonlinear suite rows from SYNTHETIC_NONLINEAR_DATASET_INDEX.

    Uses DELETE WHERE suite_id = NONLINEAR_SUITE_ID only. The table is never destroyed.
    """
    sql = (
        f"DELETE FROM {NONLINEAR_INDEX_TABLE} "
        f"WHERE suite_id = '{NONLINEAR_SUITE_ID}'"
    )
    session.sql(sql).collect()
    print(
        f"[INFO] Truncated rows (suite_id={NONLINEAR_SUITE_ID}) "
        f"from {NONLINEAR_INDEX_TABLE}.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Manifest download
# ---------------------------------------------------------------------------

def _download_manifest(session) -> dict:
    """Download nonlinear_manifest.json from Snowflake stage and parse it."""
    os.makedirs(NONLINEAR_LOCAL_DIR, exist_ok=True)
    session.file.get(NONLINEAR_MANIFEST_PATH, NONLINEAR_LOCAL_DIR)
    manifest_local = os.path.join(NONLINEAR_LOCAL_DIR, "nonlinear_manifest.json")
    with open(manifest_local, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    print(
        f"[INFO] Downloaded manifest: {manifest.get('n_datasets')} datasets, "
        f"regimes={manifest.get('regimes')}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# Index row builder
# ---------------------------------------------------------------------------

def _build_index_rows(manifest: dict) -> list[dict]:
    """Build index row dicts for the first NONLINEAR_N_DATASETS entries."""
    n_per_regime = NONLINEAR_N_DATASETS // len(NONLINEAR_REGIMES)
    rows: list[dict] = []

    for regime in NONLINEAR_REGIMES:
        regime_entries = [
            d for d in manifest["datasets"] if d["regime"] == regime
        ]
        if len(regime_entries) < n_per_regime:
            raise ValueError(
                f"Regime {regime!r}: manifest has {len(regime_entries)} entries, "
                f"need {n_per_regime} for NONLINEAR_N_DATASETS={NONLINEAR_N_DATASETS}"
            )
        for entry in regime_entries[:n_per_regime]:
            dataset_id = entry["dataset_id"]
            n_total    = entry["n_total"]
            p_signal   = entry["p_signal"]
            n_train    = _compute_n_train_default("primary", n_total)
            n_holdout  = _compute_n_holdout_default("primary", n_total)
            stage_path = (
                f"{EVAL_STAGE_PREFIX}/{regime}/dataset_{dataset_id:04d}.parquet"
            )
            rows.append({
                "suite_id":             NONLINEAR_SUITE_ID,
                "suite_family":         "primary",
                "dataset_id":           dataset_id,
                "dataset_seed":         entry["dataset_seed"],
                "stage_path":           stage_path,
                "prior_name":           NONLINEAR_PRIOR_NAME,
                "prior_version":        NONLINEAR_PRIOR_VERSION,
                "prior_regime":         regime,
                "split_seeds":          NONLINEAR_SPLIT_SEEDS,
                "n_total":              n_total,
                "n_train_default":      n_train,
                "n_holdout_default":    n_holdout,
                "p_signal":             p_signal,
                "p_noise":              entry.get("p_noise", 0),
                "p_total":              entry.get("p_total", p_signal),
                "target_noise_scale":   float(entry.get("target_noise_scale", 1.0)),
                "training_size_anchor": False,
                "feature_noise_level":  0,
                "eval_weight":          1.0,
                "payload_bytes":        entry.get("payload_bytes", 0),
                "logical_dataset_key":  f"{NONLINEAR_SUITE_ID}:{regime}:{dataset_id:04d}",
                "source_suite_id":      None,
            })

    return rows


# ---------------------------------------------------------------------------
# Row insertion (targets NONLINEAR_INDEX_TABLE directly)
# ---------------------------------------------------------------------------

def _insert_nonlinear_rows(session, rows: list[dict]) -> None:
    """Insert index rows into SYNTHETIC_NONLINEAR_DATASET_INDEX in chunks of 100.

    Mirrors insert_synreg_index_rows() from prepare_synthetic_regression.py but
    targets NONLINEAR_INDEX_TABLE instead of SYNTHETIC_REGRESSION_DATASET_INDEX.
    """
    if not rows:
        return

    def _sql_str(v) -> str:
        if v is None:
            return "NULL"
        return "'" + str(v).replace("'", "''") + "'"

    def _sql_num(v) -> str:
        return str(int(v)) if v is not None else "NULL"

    def _sql_float(v) -> str:
        return str(float(v)) if v is not None else "NULL"

    def _sql_bool(v) -> str:
        return "TRUE" if v else "FALSE"

    now_sql = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    select_strings = []
    for r in rows:
        split_seeds_json = json.dumps(list(r["split_seeds"]))
        select_strings.append(
            "SELECT "
            f"{_sql_str(r['suite_id'])}, "
            f"{_sql_str(r['suite_family'])}, "
            f"{_sql_num(r['dataset_id'])}, "
            f"{_sql_num(r['dataset_seed'])}, "
            f"{_sql_str(r['stage_path'])}, "
            f"{_sql_str(r.get('prior_name'))}, "
            f"{_sql_str(r.get('prior_version'))}, "
            f"{_sql_str(r.get('prior_regime'))}, "
            f"PARSE_JSON({_sql_str(split_seeds_json)}), "
            f"{_sql_num(r.get('n_total'))}, "
            f"{_sql_num(r.get('n_train_default'))}, "
            f"{_sql_num(r.get('n_holdout_default'))}, "
            f"{_sql_num(r.get('p_signal'))}, "
            f"{_sql_num(r.get('p_noise'))}, "
            f"{_sql_num(r.get('p_total'))}, "
            f"{_sql_float(r.get('target_noise_scale'))}, "
            f"{_sql_bool(r.get('training_size_anchor', False))}, "
            f"{_sql_num(r.get('feature_noise_level'))}, "
            f"{_sql_float(r.get('eval_weight'))}, "
            f"{_sql_num(r.get('payload_bytes'))}, "
            f"TO_TIMESTAMP_NTZ({_sql_str(now_sql)}), "
            f"{_sql_str(r.get('logical_dataset_key'))}, "
            f"{_sql_str(r.get('source_suite_id'))}"
        )

    col_list = (
        "suite_id, suite_family, dataset_id, dataset_seed, stage_path, "
        "prior_name, prior_version, prior_regime, split_seeds, "
        "n_total, n_train_default, n_holdout_default, "
        "p_signal, p_noise, p_total, target_noise_scale, "
        "training_size_anchor, feature_noise_level, "
        "eval_weight, payload_bytes, created_at, logical_dataset_key, source_suite_id"
    )

    chunk_size = 100
    for start in range(0, len(select_strings), chunk_size):
        chunk = select_strings[start : start + chunk_size]
        sql = (
            f"INSERT INTO {NONLINEAR_INDEX_TABLE} ({col_list})\n"
            + "\nUNION ALL\n".join(chunk)
        )
        session.sql(sql).collect()
        print(
            f"[INFO] Inserted rows {start}–{start + len(chunk) - 1} "
            f"into {NONLINEAR_INDEX_TABLE}.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Row count validation
# ---------------------------------------------------------------------------

def _assert_nonlinear_index_populated(session, expected: int) -> int:
    """Assert that NONLINEAR_INDEX_TABLE has the expected row count for the suite."""
    count_row = session.sql(
        f"SELECT COUNT(*) AS n FROM {NONLINEAR_INDEX_TABLE} "
        f"WHERE suite_id = '{NONLINEAR_SUITE_ID}'"
    ).collect()
    actual = int(count_row[0][0]) if count_row else 0
    if actual != expected:
        raise RuntimeError(
            f"{NONLINEAR_INDEX_TABLE} row count {actual} does not match "
            f"expected {expected} after insert."
        )
    print(
        f"[INFO] Row count validated: {actual} rows in {NONLINEAR_INDEX_TABLE}.",
        flush=True,
    )
    return actual


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(session=None) -> str:
    """Index nonlinear evaluation datasets into SYNTHETIC_NONLINEAR_DATASET_INDEX.

    Steps
    -----
    1. create_nonlinear_index_table — CREATE TABLE IF NOT EXISTS (before DELETE)
    2. _download_manifest           — GET manifest from stage
    3. Validate manifest has >= NONLINEAR_N_DATASETS entries
    4. _build_index_rows            — select rows per regime
    5. _truncate_nonlinear_index    — DELETE WHERE suite_id (never DROP)
    6. _insert_nonlinear_rows       — INSERT into SYNTHETIC_NONLINEAR_DATASET_INDEX
    7. _assert_nonlinear_index_populated — SELECT COUNT(*) check
    """
    if session is None:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()

    # Step 1: Ensure table exists
    create_nonlinear_index_table(session)

    # Step 2: Download manifest
    manifest = _download_manifest(session)

    # Step 3: Validate manifest
    if manifest.get("n_datasets", 0) < NONLINEAR_N_DATASETS:
        raise ValueError(
            f"Manifest has {manifest.get('n_datasets')} datasets, "
            f"need at least {NONLINEAR_N_DATASETS}."
        )

    # Step 4: Build index rows
    rows = _build_index_rows(manifest)
    print(
        f"[INFO] Built {len(rows)} index rows "
        f"({NONLINEAR_N_DATASETS // len(NONLINEAR_REGIMES)} per regime × "
        f"{len(NONLINEAR_REGIMES)} regimes).",
        flush=True,
    )

    # Step 5: Truncate existing suite rows (DELETE WHERE suite_id only)
    _truncate_nonlinear_index(session)

    # Step 6: Insert rows
    _insert_nonlinear_rows(session, rows)

    # Step 7: Validate row count
    _assert_nonlinear_index_populated(session, expected=len(rows))

    status = (
        f"OK suite_id={NONLINEAR_SUITE_ID} "
        f"n_indexed={len(rows)} "
        f"split_seeds={NONLINEAR_SPLIT_SEEDS}"
    )
    print(f"[INFO] evaluate_synthetic_nonlinear complete: {status}", flush=True)
    return status


# ---------------------------------------------------------------------------
# Script entry point (MLJob runs this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = main()
    print(f"[RESULT] {result}")
