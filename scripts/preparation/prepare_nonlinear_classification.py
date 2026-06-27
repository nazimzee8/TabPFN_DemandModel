"""
prepare_nonlinear_classification.py
=====================================
MLJob entrypoint that creates and populates
NONLINEAR_CLASSIFICATION_DATASET_INDEX from the staged
nonlinear_classification_v1 manifest.

Purpose
-------
Downloads nonlinear_classification_v1_manifest.json from
@EVALUATION_DATASET_STAGE/nonlinear/classification/{numeric|mixed}/, validates coverage
(11 classification families × feature regimes), runs schema migration for
classification-specific columns, then inserts rows with
suite_id = 'nonlinear_classification'.

Canonical prep handler for the nonlinear classification evaluation pipeline.
The stored procedure HANDLER is:
  run_nonlinear_classification_evaluation.prepare_nonlinear_classification

Safety invariants
-----------------
* NONLINEAR_CLASSIFICATION_DATASET_INDEX is NEVER DROPped.
* _truncate_cls_index() uses DELETE WHERE suite_id = CLS_SUITE_ID only.
* create_cls_index_table() is always called before any DELETE.

Environment variables
---------------------
NONLINEAR_CLS_SUITE_ID    override suite_id (default: nonlinear_classification)
NONLINEAR_CLS_LOCAL_DIR   local scratch dir (default: /tmp/nonlinear_cls_prep)
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

CLS_SUITE_ID       = os.getenv("NONLINEAR_CLS_SUITE_ID", "nonlinear_classification")
# SYNCLS_INDEX_TABLE overrides the default so the same prep script can populate either
# NONLINEAR_CLASSIFICATION_DATASET_INDEX (standard) or
# NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX (mixed-categorical).
CLS_INDEX_TABLE    = os.getenv("SYNCLS_INDEX_TABLE", "NONLINEAR_CLASSIFICATION_DATASET_INDEX")
CLS_PRIOR_NAME     = "nonlinear_classification"
CLS_PRIOR_VERSION  = "v1"
_IS_MIXED = os.environ.get("SYNCLS_IS_MIXED_CATEGORICAL", "false").lower() in ("true", "1")

if _IS_MIXED:
    _EVAL_STAGE_PREFIX = "@EVALUATION_DATASET_STAGE/nonlinear/classification/mixed/eval"
else:
    _EVAL_STAGE_PREFIX = "@EVALUATION_DATASET_STAGE/nonlinear/classification/numeric/eval"

# Backward-compatible alias used by _validate_stage_spot_check and other functions
EVAL_STAGE_PREFIX    = _EVAL_STAGE_PREFIX
CLS_MANIFEST_PATH    = f"{_EVAL_STAGE_PREFIX}/nonlinear_classification_v1_manifest.json"
CLS_LOCAL_DIR      = os.getenv("NONLINEAR_CLS_LOCAL_DIR", "/tmp/nonlinear_cls_prep")
CLS_SPLIT_SEEDS    = [0, 1, 2]

_V1_CLASSIFICATION_FAMILIES = [
    "nonlinear_binary_logistic",
    "sparse_interaction_logistic",
    "radial_decision_boundary",
    "piecewise_margin",
    "multiclass_smooth_softmax",
    "multiclass_sparse_highdim",
    "correlated_nonlinear_softmax",
    "low_margin_overlap",
    "label_noise_nonlinear",
    "class_imbalance_nonlinear",
    "mixed_categorical_nonlinear",
]

# Classification-specific metadata columns added by schema migration
_CLS_NEW_COLUMNS = [
    "feature_regime",
    "covariance_type",
    "rho",
    "suite_component",
    "condition_id",
    "teacher_seed",
    "sample_seed",
    "num_classes",
    "label_noise_rate",
    "class_imbalance_type",
    "margin_level",
    "temperature",
    "realized_num_classes",
    "realized_label_noise_rate",
    "realized_margin_level",
    # Mixed-categorical columns (NULL for pure-numeric entries)
    "p_num",
    "p_cat",
    "categorical_cardinalities",
    "max_cardinality",
]


# ---------------------------------------------------------------------------
# Index table DDL
# ---------------------------------------------------------------------------

def create_cls_index_table(session) -> None:
    """CREATE TABLE IF NOT EXISTS NONLINEAR_CLASSIFICATION_DATASET_INDEX."""
    ddl = f"""
    CREATE TRANSIENT TABLE IF NOT EXISTS {CLS_INDEX_TABLE} (
      suite_id                   STRING,
      suite_family               STRING,
      dataset_id                 NUMBER,
      dataset_seed               NUMBER,
      stage_path                 STRING,
      prior_name                 STRING,
      prior_version              STRING,
      prior_regime               STRING,
      split_seeds                ARRAY,
      n_total                    NUMBER,
      n_train_default            NUMBER,
      n_holdout_default          NUMBER,
      p_signal                   NUMBER,
      p_noise                    NUMBER,
      p_total                    NUMBER,
      target_noise_scale         FLOAT,
      training_size_anchor       BOOLEAN,
      feature_noise_level        NUMBER,
      eval_weight                FLOAT,
      payload_bytes              NUMBER,
      created_at                 TIMESTAMP_NTZ,
      logical_dataset_key        STRING,
      source_suite_id            STRING,
      feature_regime             STRING,
      covariance_type            STRING,
      rho                        FLOAT,
      suite_component            STRING,
      condition_id               STRING,
      teacher_seed               NUMBER,
      sample_seed                NUMBER,
      num_classes                NUMBER,
      label_noise_rate           FLOAT,
      class_imbalance_type       STRING,
      margin_level               STRING,
      temperature                FLOAT,
      realized_num_classes       NUMBER,
      realized_label_noise_rate  FLOAT,
      realized_margin_level      STRING,
      p_num                      NUMBER,
      p_cat                      NUMBER,
      categorical_cardinalities  ARRAY,
      max_cardinality            NUMBER
    ) DATA_RETENTION_TIME_IN_DAYS = 0
    """
    session.sql(ddl).collect()
    print(f"[INFO] Index table {CLS_INDEX_TABLE} ready.", flush=True)


# ---------------------------------------------------------------------------
# Schema migration — idempotent ADD COLUMN IF NOT EXISTS
# ---------------------------------------------------------------------------

def _run_schema_migration(session) -> None:
    """Add classification-specific metadata columns to CLS_INDEX_TABLE."""
    migrations = [
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS feature_regime             STRING",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS covariance_type            STRING",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS rho                        FLOAT",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS suite_component            STRING",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS condition_id               STRING",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS teacher_seed               NUMBER",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS sample_seed                NUMBER",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS num_classes                NUMBER",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS label_noise_rate           FLOAT",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS class_imbalance_type       STRING",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS margin_level               STRING",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS temperature                FLOAT",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS realized_num_classes       NUMBER",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS realized_label_noise_rate  FLOAT",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS realized_margin_level      STRING",
        # Mixed-categorical columns
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS p_num                     NUMBER",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS p_cat                     NUMBER",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS categorical_cardinalities ARRAY",
        f"ALTER TABLE {CLS_INDEX_TABLE} ADD COLUMN IF NOT EXISTS max_cardinality           NUMBER",
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
    """Download manifest from Snowflake stage and parse it."""
    os.makedirs(CLS_LOCAL_DIR, exist_ok=True)
    session.file.get(CLS_MANIFEST_PATH, CLS_LOCAL_DIR)
    local_path = os.path.join(CLS_LOCAL_DIR, "nonlinear_classification_v1_manifest.json")
    with open(local_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    print(
        f"[INFO] Downloaded manifest: {manifest.get('n_datasets')} datasets, "
        f"is_mixed_categorical={manifest.get('is_mixed_categorical', False)}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_manifest_coverage(manifest: dict) -> None:
    """Raise ValueError if manifest contains unrecognised families."""
    datasets = manifest.get("datasets", [])
    if not datasets:
        raise ValueError("Manifest contains no datasets.")

    present_families = {d["nonlinear_family"] for d in datasets}
    unknown = present_families - set(_V1_CLASSIFICATION_FAMILIES)
    if unknown:
        raise ValueError(
            f"Manifest contains unrecognised classification families: {sorted(unknown)}"
        )

    print(
        f"[INFO] Coverage validated: {len(present_families)} families, "
        f"{len(datasets)} total datasets.",
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
        stage_dir = stage_path.rsplit("/", 1)[0] + "/"
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
    """Build 38-column index row dicts from manifest entries."""
    rows: list[dict] = []
    datasets = manifest.get("datasets", [])

    for entry in datasets:
        idx = int(entry["dataset_idx"])
        family = entry["nonlinear_family"]
        n_total = int(entry["n_total"])
        p_signal = int(entry["p_signal"])
        n_train = _compute_n_train_default("primary", n_total)
        n_holdout = _compute_n_holdout_default("primary", n_total)
        stage_path = f"{_EVAL_STAGE_PREFIX}/{entry['filename']}"

        rows.append({
            # 23 base columns
            "suite_id":             CLS_SUITE_ID,
            "suite_family":         "primary",
            "dataset_id":           idx,
            "dataset_seed":         int(entry.get("sample_seed", 0)),
            "stage_path":           stage_path,
            "prior_name":           CLS_PRIOR_NAME,
            "prior_version":        CLS_PRIOR_VERSION,
            "prior_regime":         family,
            "split_seeds":          CLS_SPLIT_SEEDS,
            "n_total":              n_total,
            "n_train_default":      n_train,
            "n_holdout_default":    n_holdout,
            "p_signal":             p_signal,
            "p_noise":              int(entry.get("p_noise", 0)),
            "p_total":              int(entry.get("p_total", p_signal)),
            "target_noise_scale":   0.0,
            "training_size_anchor": False,
            "feature_noise_level":  0.0,
            "eval_weight":          1.0,
            "payload_bytes":        0,
            "logical_dataset_key":  entry["logical_dataset_key"],
            "source_suite_id":      None,
            # 15 classification-specific metadata columns
            "feature_regime":             entry.get("feature_regime", "iid_dense"),
            "covariance_type":            entry.get("covariance_type", "iid"),
            "rho":                        float(entry.get("rho", 0.0)),
            "suite_component":            entry.get("suite_component", "core"),
            "condition_id":               entry.get("condition_id", ""),
            "teacher_seed":               int(entry.get("teacher_seed", 0)),
            "sample_seed":                int(entry.get("sample_seed", 0)),
            "num_classes":                int(entry.get("num_classes", 2)),
            "label_noise_rate":           float(entry.get("label_noise_rate", 0.0)),
            "class_imbalance_type":       entry.get("class_imbalance_type", "balanced"),
            "margin_level":               entry.get("margin_level", "medium"),
            "temperature":                float(entry.get("temperature", 1.0)),
            "realized_num_classes":       int(entry.get("realized_num_classes", entry.get("num_classes", 2))),
            "realized_label_noise_rate":  float(entry.get("realized_label_noise_rate", entry.get("label_noise_rate", 0.0))),
            "realized_margin_level":      entry.get("realized_margin_level", entry.get("margin_level", "medium")),
            # Mixed-categorical columns (populated from manifest for mixed entries; defaults for numeric)
            "p_num":                      int(entry.get("p_num", int(entry.get("p_signal", 0)) + int(entry.get("p_noise", 0)))),
            "p_cat":                      int(entry.get("p_cat", 0)),
            "categorical_cardinalities":  [int(c) for c in entry.get("categorical_cardinalities", [])],
            "max_cardinality":            max(entry.get("categorical_cardinalities", [0]) or [0]),
        })

    return rows


# ---------------------------------------------------------------------------
# Truncation (never drops table)
# ---------------------------------------------------------------------------

def _truncate_cls_index(session) -> None:
    """DELETE rows WHERE suite_id = CLS_SUITE_ID. Never drops the table."""
    sql = (
        f"DELETE FROM {CLS_INDEX_TABLE} "
        f"WHERE suite_id = '{CLS_SUITE_ID}'"
    )
    session.sql(sql).collect()
    print(
        f"[INFO] Truncated rows (suite_id={CLS_SUITE_ID!r}) from {CLS_INDEX_TABLE}.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Row insertion (38 columns, 100-row chunks)
# ---------------------------------------------------------------------------

def _insert_cls_rows(session, rows: list[dict]) -> None:
    """Insert index rows into CLS_INDEX_TABLE in chunks of 100."""
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
        cat_cards_json = json.dumps(r.get("categorical_cardinalities", []))
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
            f"{_sql_str(r.get('suite_component'))}, "
            f"{_sql_str(r.get('condition_id'))}, "
            f"{_sql_num(r.get('teacher_seed'))}, "
            f"{_sql_num(r.get('sample_seed'))}, "
            f"{_sql_num(r.get('num_classes'))}, "
            f"{_sql_float(r.get('label_noise_rate'))}, "
            f"{_sql_str(r.get('class_imbalance_type'))}, "
            f"{_sql_str(r.get('margin_level'))}, "
            f"{_sql_float(r.get('temperature'))}, "
            f"{_sql_num(r.get('realized_num_classes'))}, "
            f"{_sql_float(r.get('realized_label_noise_rate'))}, "
            f"{_sql_str(r.get('realized_margin_level'))}, "
            f"{_sql_num(r.get('p_num'))}, "
            f"{_sql_num(r.get('p_cat'))}, "
            f"PARSE_JSON({_sql_str(cat_cards_json)}), "
            f"{_sql_num(r.get('max_cardinality'))}"
        )

    col_list = (
        "suite_id, suite_family, dataset_id, dataset_seed, stage_path, "
        "prior_name, prior_version, prior_regime, split_seeds, "
        "n_total, n_train_default, n_holdout_default, "
        "p_signal, p_noise, p_total, target_noise_scale, "
        "training_size_anchor, feature_noise_level, "
        "eval_weight, payload_bytes, created_at, logical_dataset_key, source_suite_id, "
        + ", ".join(_CLS_NEW_COLUMNS)
    )

    # Runtime schema alignment guard.
    try:
        from schema_contract import validate_row_against_registry
        _sample_row = {c.strip(): None for c in col_list.split(",")}
        validate_row_against_registry(_sample_row, CLS_INDEX_TABLE.upper(), strict=True)
        print(f"[INFO] Schema alignment validated: col_list matches {CLS_INDEX_TABLE}.", flush=True)
    except ImportError:
        pass

    chunk_size = 100
    for start in range(0, len(select_strings), chunk_size):
        chunk = select_strings[start: start + chunk_size]
        sql = (
            f"INSERT INTO {CLS_INDEX_TABLE} ({col_list})\n"
            + "\nUNION ALL\n".join(chunk)
        )
        session.sql(sql).collect()
        print(
            f"[INFO] Inserted rows {start}–{start + len(chunk) - 1} into {CLS_INDEX_TABLE}.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Post-insert assertion
# ---------------------------------------------------------------------------

def _assert_cls_index_populated(session, expected: int) -> int:
    """Assert row count matches expected for suite_id=CLS_SUITE_ID."""
    count_row = session.sql(
        f"SELECT COUNT(*) AS n FROM {CLS_INDEX_TABLE} "
        f"WHERE suite_id = '{CLS_SUITE_ID}'"
    ).collect()
    actual = int(count_row[0][0]) if count_row else 0

    if actual != expected:
        raise RuntimeError(
            f"{CLS_INDEX_TABLE} row count {actual} != expected {expected}."
        )

    per_family_rows = session.sql(
        f"SELECT prior_regime, COUNT(*) AS n "
        f"FROM {CLS_INDEX_TABLE} "
        f"WHERE suite_id = '{CLS_SUITE_ID}' "
        f"GROUP BY prior_regime ORDER BY prior_regime"
    ).collect()
    for row in per_family_rows:
        fam = row[0]
        cnt = int(row[1])
        if cnt == 0:
            raise RuntimeError(f"Family {fam!r} has 0 rows in index.")

    print(
        f"[INFO] Row count validated: {actual} rows in {CLS_INDEX_TABLE}.",
        flush=True,
    )
    return actual


# ---------------------------------------------------------------------------
# Entry point / Snowflake stored-procedure handler
# ---------------------------------------------------------------------------

def prepare_nonlinear_classification(session=None) -> str:
    """Snowflake stored-procedure handler.

    HANDLER = 'run_nonlinear_classification_evaluation.prepare_nonlinear_classification'

    Steps
    -----
    1. create_cls_index_table      — CREATE TABLE IF NOT EXISTS (38 columns)
    2. _run_schema_migration       — idempotent ADD COLUMN IF NOT EXISTS
    3. _download_manifest          — GET manifest from stage
    4. _validate_manifest_coverage — families within recognised set
    5. _validate_no_duplicate_keys — no key collisions
    6. _validate_stage_spot_check  — LIST 5 random stage files
    7. _build_index_rows           — 38 columns per row
    8. _truncate_cls_index         — DELETE WHERE suite_id (never DROP)
    9. _insert_cls_rows            — INSERT in 100-row chunks
    10. _assert_cls_index_populated — row count + per-family check
    """
    if session is None:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()

    create_cls_index_table(session)
    _run_schema_migration(session)
    manifest = _download_manifest(session)
    _validate_manifest_coverage(manifest)
    _validate_no_duplicate_keys(manifest)
    _validate_stage_spot_check(session, manifest)

    rows = _build_index_rows(manifest)
    print(f"[INFO] Built {len(rows)} index rows.", flush=True)

    _truncate_cls_index(session)
    _insert_cls_rows(session, rows)
    _assert_cls_index_populated(session, expected=len(rows))

    status = (
        f"OK suite_id={CLS_SUITE_ID} "
        f"n_indexed={len(rows)} "
        f"split_seeds={CLS_SPLIT_SEEDS}"
    )
    print(f"[INFO] prepare_nonlinear_classification complete: {status}", flush=True)
    return status


main = prepare_nonlinear_classification


if __name__ == "__main__":
    result = main()
    print(f"[RESULT] {result}")
