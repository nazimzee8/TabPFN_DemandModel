"""
prepare_synthetic_nonlinear_regression.py
==========================================
MLJob entrypoint that creates and populates NONLINEAR_REGRESSION_DATASET_INDEX
from the staged nonlinear_v2 manifest.

Purpose
-------
Downloads nonlinear_v2_manifest.json from @EVALUATION_DATASET_STAGE/nonlinear_v2/,
validates coverage (all 6 families × 7 regimes), runs schema migration for 13 new
columns, then inserts 420 rows with suite_id = 'nonlinear'.

Canonical prep handler for the nonlinear regression evaluation pipeline.
The stored procedure HANDLER is:
  run_synthetic_nonlinear_evaluation.prepare_synthetic_nonlinear_regression

Safety invariants
-----------------
* NONLINEAR_REGRESSION_DATASET_INDEX is NEVER DROPped.
* _truncate_nonlinear_index() uses DELETE WHERE suite_id = NONLINEAR_SUITE_ID only.
* create_nonlinear_index_table() is always called before any DELETE.

Environment variables
---------------------
NONLINEAR_SUITE_ID    override suite_id (default: nonlinear)
NONLINEAR_LOCAL_DIR   local scratch dir (default: /tmp/nonlinear_prep)
"""

from __future__ import annotations

import datetime
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from prepare_synthetic_regression import (
    _compute_n_train_default,
    _compute_n_holdout_default,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NONLINEAR_SUITE_ID      = os.getenv("NONLINEAR_SUITE_ID", "nonlinear")
# SYNREG_INDEX_TABLE overrides the default so the same prep script can populate either
# NONLINEAR_REGRESSION_DATASET_INDEX (standard) or
# NONLINEAR_MIXED_REGRESSION_DATASET_INDEX (mixed-categorical).
NONLINEAR_INDEX_TABLE   = os.getenv("SYNREG_INDEX_TABLE", "NONLINEAR_REGRESSION_DATASET_INDEX")
NONLINEAR_N_DATASETS    = 420
NONLINEAR_PRIOR_NAME    = "nonlinear"
NONLINEAR_PRIOR_VERSION = "v2"
EVAL_STAGE_PREFIX       = "@EVALUATION_DATASET_STAGE/nonlinear_v2"
NONLINEAR_MANIFEST_PATH = f"{EVAL_STAGE_PREFIX}/nonlinear_v2_manifest.json"
NONLINEAR_LOCAL_DIR     = os.getenv("NONLINEAR_LOCAL_DIR", "/tmp/nonlinear_prep")
NONLINEAR_SPLIT_SEEDS   = [0, 1, 2]

_V2_TARGET_FAMILIES = [
    "poly_quad", "sin_low", "hinge", "sparse_interact", "mixed_linear", "demand_mono",
]
_V2_FEATURE_REGIMES = [
    "iid_dense", "iid_sparse", "ar1", "block", "equicorr", "noise_feats", "feat_noise",
]

# 13 new metadata columns added by schema migration
_V2_NEW_COLUMNS = [
    "feature_regime",
    "covariance_type",
    "rho",
    "active_fraction",
    "noise_feature_fraction",
    "feature_noise_sigma",
    "suite_component",
    "target_noise_type",
    "snr_target",
    "condition_id",
    "teacher_seed",
    "sample_seed",
    "normalization_constant",
]


# ---------------------------------------------------------------------------
# Index table DDL
# ---------------------------------------------------------------------------

def create_nonlinear_index_table(session) -> None:
    """CREATE TABLE IF NOT EXISTS NONLINEAR_REGRESSION_DATASET_INDEX (36 columns)."""
    ddl = f"""
    CREATE TRANSIENT TABLE IF NOT EXISTS {NONLINEAR_INDEX_TABLE} (
      suite_id               STRING,
      suite_family           STRING,
      dataset_id             NUMBER,
      dataset_seed           NUMBER,
      stage_path             STRING,
      prior_name             STRING,
      prior_version          STRING,
      prior_regime           STRING,
      split_seeds            ARRAY,
      n_total                NUMBER,
      n_train_default        NUMBER,
      n_holdout_default      NUMBER,
      p_signal               NUMBER,
      p_noise                NUMBER,
      p_total                NUMBER,
      target_noise_scale     FLOAT,
      training_size_anchor   BOOLEAN,
      feature_noise_level    NUMBER,
      eval_weight            FLOAT,
      payload_bytes          NUMBER,
      created_at             TIMESTAMP_NTZ,
      logical_dataset_key    STRING,
      source_suite_id        STRING,
      feature_regime         STRING,
      covariance_type        STRING,
      rho                    FLOAT,
      active_fraction        FLOAT,
      noise_feature_fraction FLOAT,
      feature_noise_sigma    FLOAT,
      suite_component        STRING,
      target_noise_type      STRING,
      snr_target             FLOAT,
      condition_id           STRING,
      teacher_seed           NUMBER,
      sample_seed            NUMBER,
      normalization_constant FLOAT
    ) DATA_RETENTION_TIME_IN_DAYS = 0
    """
    session.sql(ddl).collect()
    print(f"[INFO] Index table {NONLINEAR_INDEX_TABLE} ready.", flush=True)


# ---------------------------------------------------------------------------
# Schema migration — 13 idempotent ADD COLUMN IF NOT EXISTS
# ---------------------------------------------------------------------------

def _run_schema_migration(session) -> None:
    """Add 13 new metadata columns to NONLINEAR_REGRESSION_DATASET_INDEX.

    All statements use IF NOT EXISTS — safe to re-run.
    """
    migrations = [
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS feature_regime         STRING",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS covariance_type        STRING",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS rho                    FLOAT",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS active_fraction        FLOAT",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS noise_feature_fraction FLOAT",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS feature_noise_sigma    FLOAT",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS suite_component        STRING",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS target_noise_type      STRING",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS snr_target             FLOAT",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS condition_id           STRING",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS teacher_seed           NUMBER",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS sample_seed            NUMBER",
        f"ALTER TABLE {NONLINEAR_INDEX_TABLE} ADD COLUMN IF NOT EXISTS normalization_constant FLOAT",
    ]
    for sql in migrations:
        session.sql(sql).collect()
    print(
        f"[INFO] Schema migration complete: {len(migrations)} columns added/verified.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Manifest download
# ---------------------------------------------------------------------------

def _download_manifest(session) -> dict:
    """Download nonlinear_v2_manifest.json from Snowflake stage and parse it."""
    os.makedirs(NONLINEAR_LOCAL_DIR, exist_ok=True)
    session.file.get(NONLINEAR_MANIFEST_PATH, NONLINEAR_LOCAL_DIR)
    manifest_local = os.path.join(NONLINEAR_LOCAL_DIR, "nonlinear_v2_manifest.json")
    with open(manifest_local, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    print(
        f"[INFO] Downloaded manifest: {manifest.get('n_datasets')} datasets, "
        f"families={manifest.get('target_families')}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_manifest_coverage(manifest: dict) -> None:
    """Raise ValueError if any (family × regime) cell is absent."""
    datasets = manifest.get("datasets", [])
    covered_families = {d["target_family"] for d in datasets}
    covered_regimes = {d["feature_regime"] for d in datasets}

    missing_families = set(_V2_TARGET_FAMILIES) - covered_families
    if missing_families:
        raise ValueError(
            f"Manifest missing target families: {sorted(missing_families)}"
        )

    missing_regimes = set(_V2_FEATURE_REGIMES) - covered_regimes
    if missing_regimes:
        raise ValueError(
            f"Manifest missing feature regimes: {sorted(missing_regimes)}"
        )

    print(
        f"[INFO] Coverage validated: {len(covered_families)} families × "
        f"{len(covered_regimes)} regimes.",
        flush=True,
    )


def _validate_no_duplicate_keys(manifest: dict) -> None:
    """Raise ValueError if any logical_dataset_key appears more than once."""
    datasets = manifest.get("datasets", [])
    keys = [d["logical_dataset_key"] for d in datasets]
    seen: set[str] = set()
    duplicates: list[str] = []
    for k in keys:
        if k in seen:
            duplicates.append(k)
        seen.add(k)
    if duplicates:
        raise ValueError(
            f"Duplicate logical_dataset_key in manifest: {duplicates[:5]}"
        )
    print(f"[INFO] Key uniqueness validated: {len(keys)} keys, no duplicates.", flush=True)


def _validate_stage_spot_check(session, manifest: dict, n_spot: int = 5) -> None:
    """LIST n_spot random stage files to verify they exist."""
    datasets = manifest.get("datasets", [])
    if not datasets:
        return
    sample = random.sample(datasets, min(n_spot, len(datasets)))
    for entry in sample:
        stage_path = f"{EVAL_STAGE_PREFIX}/{entry['filename']}"
        path_parts = stage_path.rsplit("/", 1)
        stage_dir = path_parts[0] + "/"
        results = session.sql(f"LIST {stage_dir}").collect()
        found = any(
            entry["filename"].split("/")[-1] in str(row)
            for row in results
        )
        if not found:
            raise RuntimeError(
                f"Stage spot-check failed: {stage_path!r} not found via LIST {stage_dir}"
            )
    print(f"[INFO] Stage spot-check OK ({n_spot} files verified).", flush=True)


# ---------------------------------------------------------------------------
# Index row builder
# ---------------------------------------------------------------------------

def _build_index_rows(manifest: dict) -> list[dict]:
    """Build 36-column index row dicts from manifest entries."""
    rows: list[dict] = []
    datasets = manifest.get("datasets", [])

    for entry in datasets:
        idx = int(entry["dataset_idx"])
        target_family = entry["target_family"]
        n_total = int(entry["n_total"])
        p_signal = int(entry["p_signal"])
        n_train = _compute_n_train_default("primary", n_total)
        n_holdout = _compute_n_holdout_default("primary", n_total)

        stage_path = f"{EVAL_STAGE_PREFIX}/{entry['filename']}"

        rows.append({
            # 23 original columns
            "suite_id":             NONLINEAR_SUITE_ID,
            "suite_family":         "primary",
            "dataset_id":           idx,
            "dataset_seed":         int(entry.get("sample_seed", 0)),
            "stage_path":           stage_path,
            "prior_name":           NONLINEAR_PRIOR_NAME,
            "prior_version":        NONLINEAR_PRIOR_VERSION,
            "prior_regime":         target_family,
            "split_seeds":          NONLINEAR_SPLIT_SEEDS,
            "n_total":              n_total,
            "n_train_default":      n_train,
            "n_holdout_default":    n_holdout,
            "p_signal":             p_signal,
            "p_noise":              int(entry.get("p_noise", 0)),
            "p_total":              int(entry.get("p_total", p_signal)),
            "target_noise_scale":   float(entry.get("target_noise_scale", 1.0)),
            "training_size_anchor": False,
            "feature_noise_level":  float(entry.get("feature_noise_sigma", 0.0)),
            "eval_weight":          1.0,
            "payload_bytes":        0,
            "logical_dataset_key":  entry["logical_dataset_key"],
            "source_suite_id":      None,
            # 13 new metadata columns
            "feature_regime":           entry.get("feature_regime", "iid_dense"),
            "covariance_type":          entry.get("covariance_type", "iid"),
            "rho":                      float(entry.get("rho", 0.0)),
            "active_fraction":          float(entry.get("active_fraction", 1.0)),
            "noise_feature_fraction":   float(entry.get("noise_feature_fraction", 0.0)),
            "feature_noise_sigma":      float(entry.get("feature_noise_sigma", 0.0)),
            "suite_component":          entry.get("suite_component", "core"),
            "target_noise_type":        entry.get("target_noise_type", "gaussian"),
            "snr_target":               float(entry.get("snr_target", 0.0)),
            "condition_id":             entry.get("condition_id", ""),
            "teacher_seed":             int(entry.get("teacher_seed", 0)),
            "sample_seed":              int(entry.get("sample_seed", 0)),
            "normalization_constant":   float(entry.get("normalization_constant", 1.0)),
        })

    return rows


# ---------------------------------------------------------------------------
# Truncation (never drops table)
# ---------------------------------------------------------------------------

def _truncate_nonlinear_index(session) -> None:
    """DELETE rows WHERE suite_id = NONLINEAR_SUITE_ID. Never drops the table."""
    sql = (
        f"DELETE FROM {NONLINEAR_INDEX_TABLE} "
        f"WHERE suite_id = '{NONLINEAR_SUITE_ID}'"
    )
    session.sql(sql).collect()
    print(
        f"[INFO] Truncated rows (suite_id={NONLINEAR_SUITE_ID!r}) "
        f"from {NONLINEAR_INDEX_TABLE}.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Row insertion (36 columns, 100-row chunks)
# ---------------------------------------------------------------------------

def _insert_nonlinear_rows(session, rows: list[dict]) -> None:
    """Insert index rows into NONLINEAR_REGRESSION_DATASET_INDEX in chunks of 100."""
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
            f"{_sql_float(r.get('feature_noise_level'))}, "
            f"{_sql_float(r.get('eval_weight'))}, "
            f"{_sql_num(r.get('payload_bytes'))}, "
            f"TO_TIMESTAMP_NTZ({_sql_str(now_sql)}), "
            f"{_sql_str(r.get('logical_dataset_key'))}, "
            f"{_sql_str(r.get('source_suite_id'))}, "
            f"{_sql_str(r.get('feature_regime'))}, "
            f"{_sql_str(r.get('covariance_type'))}, "
            f"{_sql_float(r.get('rho'))}, "
            f"{_sql_float(r.get('active_fraction'))}, "
            f"{_sql_float(r.get('noise_feature_fraction'))}, "
            f"{_sql_float(r.get('feature_noise_sigma'))}, "
            f"{_sql_str(r.get('suite_component'))}, "
            f"{_sql_str(r.get('target_noise_type'))}, "
            f"{_sql_float(r.get('snr_target'))}, "
            f"{_sql_str(r.get('condition_id'))}, "
            f"{_sql_num(r.get('teacher_seed'))}, "
            f"{_sql_num(r.get('sample_seed'))}, "
            f"{_sql_float(r.get('normalization_constant'))}"
        )

    col_list = (
        "suite_id, suite_family, dataset_id, dataset_seed, stage_path, "
        "prior_name, prior_version, prior_regime, split_seeds, "
        "n_total, n_train_default, n_holdout_default, "
        "p_signal, p_noise, p_total, target_noise_scale, "
        "training_size_anchor, feature_noise_level, "
        "eval_weight, payload_bytes, created_at, logical_dataset_key, source_suite_id, "
        + ", ".join(_V2_NEW_COLUMNS)
    )

    chunk_size = 100
    for start in range(0, len(select_strings), chunk_size):
        chunk = select_strings[start: start + chunk_size]
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
# Post-insert assertion
# ---------------------------------------------------------------------------

def _assert_nonlinear_index_populated(session, expected: int) -> int:
    """Assert row count and per-family count for suite_id=NONLINEAR_SUITE_ID."""
    count_row = session.sql(
        f"SELECT COUNT(*) AS n FROM {NONLINEAR_INDEX_TABLE} "
        f"WHERE suite_id = '{NONLINEAR_SUITE_ID}'"
    ).collect()
    actual = int(count_row[0][0]) if count_row else 0

    if actual != expected:
        raise RuntimeError(
            f"{NONLINEAR_INDEX_TABLE} row count {actual} != expected {expected}."
        )

    per_family_rows = session.sql(
        f"SELECT prior_regime, COUNT(*) AS n "
        f"FROM {NONLINEAR_INDEX_TABLE} "
        f"WHERE suite_id = '{NONLINEAR_SUITE_ID}' "
        f"GROUP BY prior_regime ORDER BY prior_regime"
    ).collect()
    for row in per_family_rows:
        fam = row[0]
        cnt = int(row[1])
        if cnt == 0:
            raise RuntimeError(f"Family {fam!r} has 0 rows in index.")

    print(
        f"[INFO] Row count validated: {actual} rows in {NONLINEAR_INDEX_TABLE}.",
        flush=True,
    )
    return actual


# ---------------------------------------------------------------------------
# Entry point / Snowflake stored-procedure handler
# ---------------------------------------------------------------------------

def prepare_synthetic_nonlinear_regression(session=None) -> str:
    """Snowflake stored-procedure handler.

    HANDLER = 'run_synthetic_nonlinear_evaluation.prepare_synthetic_nonlinear_regression'

    Steps
    -----
    1. create_nonlinear_index_table — CREATE TABLE IF NOT EXISTS (36 columns)
    2. _run_schema_migration        — 13 idempotent ADD COLUMN IF NOT EXISTS
    3. _download_manifest           — GET manifest from stage
    4. _validate_manifest_coverage  — all 6 families × 7 regimes present
    5. _validate_no_duplicate_keys  — no key collisions
    6. _validate_stage_spot_check   — LIST 5 random stage files
    7. _build_index_rows            — 36 columns per row
    8. _truncate_nonlinear_index    — DELETE WHERE suite_id (never DROP)
    9. _insert_nonlinear_rows       — INSERT in 100-row chunks
    10. _assert_nonlinear_index_populated — row count + per-family check
    """
    if session is None:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()

    create_nonlinear_index_table(session)
    _run_schema_migration(session)
    manifest = _download_manifest(session)
    _validate_manifest_coverage(manifest)
    _validate_no_duplicate_keys(manifest)
    _validate_stage_spot_check(session, manifest)

    rows = _build_index_rows(manifest)
    print(f"[INFO] Built {len(rows)} index rows.", flush=True)

    _truncate_nonlinear_index(session)
    _insert_nonlinear_rows(session, rows)
    _assert_nonlinear_index_populated(session, expected=len(rows))

    status = (
        f"OK suite_id={NONLINEAR_SUITE_ID} "
        f"n_indexed={len(rows)} "
        f"split_seeds={NONLINEAR_SPLIT_SEEDS}"
    )
    print(f"[INFO] prepare_synthetic_nonlinear_regression complete: {status}", flush=True)
    return status


# Alias for backward compatibility and direct invocation
main = prepare_synthetic_nonlinear_regression


if __name__ == "__main__":
    result = main()
    print(f"[RESULT] {result}")
