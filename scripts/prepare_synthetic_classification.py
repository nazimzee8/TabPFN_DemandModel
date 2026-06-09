"""
prepare_synthetic_classification.py
=====================================
Classification equivalent of prepare_synthetic_regression.py.

Reads parquet files from
  @EVALUATION_DATASET_STAGE/synthetic_classification_prepared/<suite_id>/
validates them against the linear_classification_eval_v1/v2 schema, and
upserts records into LINEAR_CLASSIFICATION_DATASET_INDEX.

Idempotent by default; full rebuild when SYNTHETIC_CLASSIFICATION_FORCE_REBUILD=true.

Parquet format
--------------
Per-row format (linear_classification_eval_v1/v2): one row per sample.
  Required columns: feature_vector, label, split, schema_version, suite_id,
    suite_family, dataset_idx, global_idx, task_objective, n_train, n_test,
    n_features, num_classes, is_tabpfn_anchor
  Optional columns: y_clean, label_noise_mask, teacher_prob_0..K-1, etc.

Optional metadata fields (new columns, null-filled if absent)
-----------------------
  label_noise_rate, class_imbalance_type, temperature,
  margin_level, sample_complexity_bucket
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HOME", "/tmp")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Constants (env-overridable)
# ---------------------------------------------------------------------------

SYNCLS_SUITE_ID = os.getenv("SYNTHETIC_CLASSIFICATION_SUITE_ID", "linear_classification_stat_aware")
SYNCLS_FORCE_REBUILD = _env_flag("SYNTHETIC_CLASSIFICATION_FORCE_REBUILD", "false")
SYNCLS_STAGE_PREFIX = "@EVALUATION_DATASET_STAGE"
SYNCLS_INDEX_TABLE = os.getenv("SYNCLS_INDEX_TABLE", "LINEAR_CLASSIFICATION_DATASET_INDEX")
SYNCLS_LOCAL_DIR = os.getenv("SYNTHETIC_CLASSIFICATION_LOCAL_DIR", "/tmp/syncls_prep")
SYNCLS_IS_MIXED_CATEGORICAL = _env_flag("SYNCLS_IS_MIXED_CATEGORICAL", "false")
# Canonical stage root: generator writes to synthetic_classification_prepared/{suite_id}/{family}/
SYNCLS_STAGE_ROOT = f"{SYNCLS_STAGE_PREFIX}/synthetic_classification_prepared"
SYNCLS_MANIFEST_PATH = f"{SYNCLS_STAGE_ROOT}/synthetic_classification_manifest_{SYNCLS_SUITE_ID}.json"

# Optional metadata fields → index columns (new columns added in schema update)
_OPTIONAL_META_FIELDS = {
    "label_noise_rate", "class_imbalance_type", "temperature",
    "margin_level", "sample_complexity_bucket",
}


# ---------------------------------------------------------------------------
# Snowflake stage helpers
# ---------------------------------------------------------------------------

def _list_stage_parquets(session, stage_prefix: str) -> list[str]:
    """Return list of stage paths for parquet files under prefix (recursive)."""
    try:
        # LIST with PATTERN recursively covers all subdirectories
        rows = session.sql(f"LIST {stage_prefix}").collect()
        return [r["name"] for r in rows if str(r["name"]).endswith(".parquet")]
    except Exception as exc:
        print(f"[WARN] LIST {stage_prefix} failed: {exc}", flush=True)
        return []


def _download_parquet(session, stage_path: str, local_dir: str) -> Path:
    """Download a parquet file from stage to local_dir. Returns local path."""
    os.makedirs(local_dir, exist_ok=True)
    session.file.get(stage_path, local_dir)
    filename = stage_path.rsplit("/", 1)[-1]
    # Snowflake may strip leading @stage/ prefix from name
    local_path = Path(local_dir) / filename
    if not local_path.exists():
        # Try without stage prefix (Snowflake naming)
        candidates = list(Path(local_dir).glob(filename))
        if candidates:
            local_path = candidates[0]
    return local_path


# ---------------------------------------------------------------------------
# Parquet validation
# ---------------------------------------------------------------------------

def _validate_parquet_fields(parquet_path: Path, required_fields: set = None) -> dict:
    """
    Read and validate a parquet file in linear_classification_eval_v1/v2 format.
    Uses load_classification_eval_task() as the canonical loader.
    Returns a metadata dict extracted from the loaded task.
    Raises ValueError if the parquet does not conform to the schema.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from dgp_helpers import validate_classification_eval_dataset, load_classification_eval_task

    table = pq.read_table(str(parquet_path))

    # Fail-fast schema validation
    validate_classification_eval_dataset(table, strict=True)

    # Load task via canonical loader
    task = load_classification_eval_task(table)

    # Build metadata dict from task fields (compatible with _build_index_record)
    meta = {k: v for k, v in task.items() if not hasattr(v, "__len__") or isinstance(v, str)}
    # Also expose array shapes as scalar hints
    for arr_key, shape_key in [("X_train", "n_train"), ("X_test", "n_test")]:
        arr = task.get(arr_key)
        if arr is not None and hasattr(arr, "shape"):
            meta[shape_key] = arr.shape[0]

    return meta


# ---------------------------------------------------------------------------
# Index table helpers
# ---------------------------------------------------------------------------

def _create_cls_index_table(session) -> None:
    """Create LINEAR_CLASSIFICATION_DATASET_INDEX if not already present."""
    ddl = f"""
    CREATE TRANSIENT TABLE IF NOT EXISTS {SYNCLS_INDEX_TABLE} (
      suite_id STRING,
      suite_family STRING,
      dataset_id NUMBER,
      dataset_seed NUMBER,
      global_idx NUMBER,
      stage_path STRING,
      prior_regime STRING,
      classification_regime STRING,
      split_seeds ARRAY,
      n_total NUMBER,
      n_train_default NUMBER,
      n_holdout_default NUMBER,
      p_signal NUMBER,
      p_noise NUMBER,
      p_total NUMBER,
      num_classes NUMBER,
      feature_noise_level FLOAT,
      training_size_anchor BOOLEAN,
      eval_weight FLOAT,
      payload_bytes NUMBER,
      created_at TIMESTAMP_NTZ,
      logical_dataset_key STRING,
      source_suite_id STRING,
      task_family STRING,
      task_objective STRING,
      label_noise_rate FLOAT,
      class_imbalance_type STRING,
      temperature FLOAT,
      margin_level STRING,
      sample_complexity_bucket STRING,
      schema_version STRING,
      distribution_family STRING,
      covariance_type STRING,
      rho FLOAT,
      is_training_allowed BOOLEAN,
      is_eval_only BOOLEAN,
      is_ood BOOLEAN,
      is_hidden_holdout BOOLEAN,
      difficulty_score FLOAT,
      difficulty_tier STRING,
      difficulty_reasons STRING,
      estimated_memory_bytes NUMBER,
      estimated_memory_gib FLOAT,
      memory_class STRING,
      hidden_holdout_suite_id STRING,
      task_fingerprint STRING
    ) DATA_RETENTION_TIME_IN_DAYS = 0
    """
    session.sql(ddl).collect()
    print(f"[INFO] Index table {SYNCLS_INDEX_TABLE} ready.", flush=True)


def _index_row_count(session, suite_id: str) -> int:
    """Return current row count for suite_id in LINEAR_CLASSIFICATION_DATASET_INDEX."""
    try:
        rows = session.sql(
            f"SELECT COUNT(*) AS n FROM {SYNCLS_INDEX_TABLE} WHERE suite_id = '{suite_id}'"
        ).collect()
        return int(rows[0][0]) if rows else 0
    except Exception:
        return 0


def _delete_cls_index_rows(session, suite_id: str) -> None:
    """Remove stale rows for the current suite_id before a rebuild."""
    session.sql(
        f"DELETE FROM {SYNCLS_INDEX_TABLE} WHERE suite_id = '{suite_id}'"
    ).collect()
    print(f"[INFO] Deleted rows for suite_id='{suite_id}' from {SYNCLS_INDEX_TABLE}.", flush=True)


def _sql_str(v) -> str:
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


def _sql_num(v) -> str:
    if v is None:
        return "NULL"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "NULL"


def _sql_float(v) -> str:
    if v is None:
        return "NULL"
    try:
        return str(float(v))
    except (TypeError, ValueError):
        return "NULL"


def _sql_bool(v) -> str:
    return "TRUE" if v else "FALSE"


def _insert_cls_index_rows(session, rows: list[dict]) -> None:
    """Insert classification index rows in chunks using raw SQL INSERT."""
    if not rows:
        return

    now_sql = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    col_list = (
        "suite_id, suite_family, dataset_id, dataset_seed, global_idx, stage_path, "
        "prior_regime, classification_regime, split_seeds, "
        "n_total, n_train_default, n_holdout_default, "
        "p_signal, p_noise, p_total, num_classes, feature_noise_level, "
        "training_size_anchor, eval_weight, payload_bytes, created_at, "
        "logical_dataset_key, source_suite_id, "
        "task_family, task_objective, label_noise_rate, class_imbalance_type, "
        "temperature, margin_level, sample_complexity_bucket, schema_version, "
        "distribution_family, covariance_type, rho, is_training_allowed, "
        "is_eval_only, is_ood, is_hidden_holdout, difficulty_score, difficulty_tier, "
        "difficulty_reasons, estimated_memory_bytes, estimated_memory_gib, memory_class, "
        "hidden_holdout_suite_id, task_fingerprint"
    )

    select_strings = []
    for r in rows:
        split_seeds_json = json.dumps(list(r.get("split_seeds") or []))
        select_strings.append(
            "SELECT "
            f"{_sql_str(r.get('suite_id'))}, "
            f"{_sql_str(r.get('suite_family'))}, "
            f"{_sql_num(r.get('dataset_id'))}, "
            f"{_sql_num(r.get('dataset_seed'))}, "
            f"{_sql_num(r.get('global_idx'))}, "
            f"{_sql_str(r.get('stage_path'))}, "
            f"{_sql_str(r.get('prior_regime'))}, "
            f"{_sql_str(r.get('classification_regime'))}, "
            f"PARSE_JSON({_sql_str(split_seeds_json)}), "
            f"{_sql_num(r.get('n_total'))}, "
            f"{_sql_num(r.get('n_train_default'))}, "
            f"{_sql_num(r.get('n_holdout_default'))}, "
            f"{_sql_num(r.get('p_signal'))}, "
            f"{_sql_num(r.get('p_noise'))}, "
            f"{_sql_num(r.get('p_total'))}, "
            f"{_sql_num(r.get('num_classes'))}, "
            f"{_sql_float(r.get('feature_noise_level'))}, "
            f"{_sql_bool(r.get('training_size_anchor', False))}, "
            f"{_sql_float(r.get('eval_weight', 1.0))}, "
            f"{_sql_num(r.get('payload_bytes', 0))}, "
            f"TO_TIMESTAMP_NTZ({_sql_str(now_sql)}), "
            f"{_sql_str(r.get('logical_dataset_key'))}, "
            f"{_sql_str(r.get('source_suite_id'))}, "
            f"{_sql_str(r.get('task_family'))}, "
            f"{_sql_str(r.get('task_objective'))}, "
            f"{_sql_float(r.get('label_noise_rate'))}, "
            f"{_sql_str(r.get('class_imbalance_type'))}, "
            f"{_sql_float(r.get('temperature'))}, "
            f"{_sql_str(r.get('margin_level'))}, "
            f"{_sql_str(r.get('sample_complexity_bucket'))}, "
            f"{_sql_str(r.get('schema_version'))}, "
            f"{_sql_str(r.get('distribution_family'))}, "
            f"{_sql_str(r.get('covariance_type'))}, "
            f"{_sql_float(r.get('rho'))}, "
            f"{_sql_bool(r.get('is_training_allowed', True))}, "
            f"{_sql_bool(r.get('is_eval_only', False))}, "
            f"{_sql_bool(r.get('is_ood', False))}, "
            f"{_sql_bool(r.get('is_hidden_holdout', False))}, "
            f"{_sql_float(r.get('difficulty_score'))}, "
            f"{_sql_str(r.get('difficulty_tier'))}, "
            f"{_sql_str(json.dumps(r.get('difficulty_reasons', [])))}, "
            f"{_sql_num(r.get('estimated_memory_bytes'))}, "
            f"{_sql_float(r.get('estimated_memory_gib'))}, "
            f"{_sql_str(r.get('memory_class'))}, "
            f"{_sql_str(r.get('hidden_holdout_suite_id'))}, "
            f"{_sql_str(r.get('task_fingerprint'))}"
        )

    chunk_size = 100
    for start in range(0, len(select_strings), chunk_size):
        chunk = select_strings[start: start + chunk_size]
        sql = (
            f"INSERT INTO {SYNCLS_INDEX_TABLE} ({col_list})\n"
            + "\nUNION ALL\n".join(chunk)
        )
        session.sql(sql).collect()
        print(
            f"[INFO] Inserted rows {start}–{start + len(chunk) - 1} into {SYNCLS_INDEX_TABLE}.",
            flush=True,
        )

    # Validate row count
    actual = _index_row_count(session, rows[0]["suite_id"])
    print(f"[INFO] Row count validated: {actual} rows in {SYNCLS_INDEX_TABLE}.", flush=True)


# ---------------------------------------------------------------------------
# Record builder from parquet metadata
# ---------------------------------------------------------------------------

def _build_index_record(
    suite_id: str,
    dataset_id: int,
    stage_path: str,
    meta: dict,
    payload_bytes: int = 0,
) -> dict:
    """Build an index row dict from parquet metadata (as returned by _validate_parquet_fields)."""
    # Shape info — read directly from canonical task fields (no array[N] string heuristics)
    n_train_default = meta.get("n_train")
    n_holdout_default = meta.get("n_test")
    n_total = meta.get("n_total")
    if n_total is None and n_train_default is not None and n_holdout_default is not None:
        n_total = n_train_default + n_holdout_default

    p_signal = meta.get("p_signal")
    p_noise = meta.get("p_noise")
    p_total = meta.get("p_total") or meta.get("n_features")

    num_classes = meta.get("num_classes")
    if isinstance(num_classes, str):
        try:
            num_classes = int(num_classes)
        except ValueError:
            num_classes = None

    task_family = meta.get("task_family")
    if isinstance(task_family, list) and task_family:
        task_family = task_family[0]
    if task_family not in {None, "linear_classification"}:
        raise ValueError(
            "Classification preparer refuses task_family="
            f"{task_family!r}"
        )
    task_objective = meta.get("task_objective")
    if isinstance(task_objective, list) and task_objective:
        task_objective = task_objective[0]

    # Classification regime from parquet or inferred
    classification_regime = meta.get("classification_regime")
    prior_regime = meta.get("prior_regime")
    suite_family = meta.get("suite_family", "primary")

    # Optional new fields
    label_noise_rate = meta.get("label_noise_rate")
    class_imbalance_type = meta.get("class_imbalance_type")
    temperature = meta.get("temperature")
    margin_level = meta.get("margin_level")
    sample_complexity_bucket = meta.get("sample_complexity_bucket")

    feature_noise_level = meta.get("feature_noise_level", 0.0)

    logical_dataset_key = (
        meta.get("logical_dataset_key")
        or f"{suite_id}:{classification_regime or prior_regime}:{dataset_id:04d}"
    )

    split_seeds_raw = meta.get("split_seeds")
    if split_seeds_raw is None:
        # CLS-S2: Do not silently invent a default; store NULL so consumers can
        # distinguish "genuinely used [0,1,2]" from "metadata absent".
        task_seed_val = meta.get("task_seed")
        if task_seed_val is not None:
            # Derive deterministically from task_seed rather than using a fake default.
            ts = int(task_seed_val)
            split_seeds = [ts % 2**16, (ts >> 16) % 2**16, (ts >> 32) % 2**16]
            logging.warning(
                "split_seeds absent from parquet metadata; derived from task_seed=%d: %s",
                ts, split_seeds,
            )
        else:
            split_seeds = None
            logging.warning(
                "split_seeds absent from parquet metadata and task_seed unavailable; "
                "storing NULL. Dataset may not be reproducible without re-generation."
            )
    else:
        split_seeds = split_seeds_raw
        if isinstance(split_seeds, str):
            try:
                split_seeds = json.loads(split_seeds)
            except Exception:
                logging.warning(
                    "split_seeds could not be parsed from string %r; storing NULL.",
                    split_seeds,
                )
                split_seeds = None

    return {
        "suite_id": suite_id,
        "suite_family": suite_family,
        "dataset_id": dataset_id,
        "dataset_seed": meta.get("dataset_seed"),
        "global_idx": meta.get("global_idx"),
        "stage_path": stage_path,
        "prior_regime": prior_regime,
        "classification_regime": classification_regime,
        "split_seeds": split_seeds,
        "n_total": n_total,
        "n_train_default": n_train_default,
        "n_holdout_default": n_holdout_default,
        "p_signal": p_signal,
        "p_noise": p_noise,
        "p_total": p_total,
        "num_classes": num_classes,
        "feature_noise_level": feature_noise_level,
        "training_size_anchor": meta.get("training_size_anchor", False),
        "eval_weight": meta.get("eval_weight", 1.0),
        "payload_bytes": payload_bytes,
        "logical_dataset_key": logical_dataset_key,
        "source_suite_id": meta.get("source_suite_id"),
        "task_family": task_family,
        "task_objective": task_objective,
        "label_noise_rate": label_noise_rate,
        "class_imbalance_type": class_imbalance_type,
        "temperature": temperature,
        "margin_level": margin_level,
        "sample_complexity_bucket": sample_complexity_bucket,
        "schema_version": meta.get("schema_version"),
        "distribution_family": meta.get("distribution_family"),
        "covariance_type": meta.get("covariance_type"),
        "rho": meta.get("rho"),
        "is_training_allowed": meta.get("is_training_allowed", True),
        "is_eval_only": meta.get("is_eval_only", False),
        "is_ood": meta.get("is_ood", False),
        "is_hidden_holdout": meta.get("is_hidden_holdout", False),
        "difficulty_score": meta.get("difficulty_score"),
        "difficulty_tier": meta.get("difficulty_tier"),
        "difficulty_reasons": meta.get("difficulty_reasons", []),
        "estimated_memory_bytes": meta.get("estimated_memory_bytes"),
        "estimated_memory_gib": meta.get("estimated_memory_gib"),
        "memory_class": meta.get("memory_class"),
        "hidden_holdout_suite_id": meta.get("hidden_holdout_suite_id"),
        "task_fingerprint": meta.get("task_fingerprint"),
    }


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------

def _should_skip(session, suite_id: str) -> bool:
    """Return True if the index already has rows for this suite_id and FORCE_REBUILD is false."""
    if SYNCLS_FORCE_REBUILD:
        print("[INFO] SYNTHETIC_CLASSIFICATION_FORCE_REBUILD=true — rebuilding.", flush=True)
        return False
    count = _index_row_count(session, suite_id)
    if count > 0:
        print(
            f"[INFO] {SYNCLS_INDEX_TABLE} already has {count} rows for suite_id={suite_id!r}. "
            "Skipping (set SYNTHETIC_CLASSIFICATION_FORCE_REBUILD=true to force rebuild).",
            flush=True,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Mixed-categorical classification preparation
# ---------------------------------------------------------------------------

def _create_mixed_cls_index_table(session) -> None:
    """Create LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX if not already present."""
    ddl = f"""
    CREATE TRANSIENT TABLE IF NOT EXISTS {SYNCLS_INDEX_TABLE} (
      suite_id STRING,
      suite_family STRING,
      dataset_id NUMBER,
      dataset_seed NUMBER,
      global_idx NUMBER,
      stage_path STRING,
      prior_regime STRING,
      classification_regime STRING,
      split_seeds ARRAY,
      n_total NUMBER,
      n_train_default NUMBER,
      n_holdout_default NUMBER,
      p_signal NUMBER,
      p_noise NUMBER,
      p_total NUMBER,
      p_num NUMBER,
      p_cat NUMBER,
      categorical_cardinalities VARIANT,
      max_cardinality NUMBER,
      cat_effect_scale FLOAT,
      missing_rate FLOAT,
      num_classes NUMBER,
      feature_noise_level FLOAT,
      training_size_anchor BOOLEAN,
      eval_weight FLOAT,
      payload_bytes NUMBER,
      created_at TIMESTAMP_NTZ,
      logical_dataset_key STRING,
      source_suite_id STRING,
      task_family STRING,
      task_objective STRING,
      label_noise_rate FLOAT,
      class_imbalance_type STRING,
      temperature FLOAT,
      margin_level STRING,
      sample_complexity_bucket STRING,
      task_seed NUMBER,
      schema_version STRING,
      coeff_schema_version STRING,
      distribution_family STRING,
      covariance_type STRING,
      rho FLOAT,
      is_training_allowed BOOLEAN,
      is_eval_only BOOLEAN,
      is_ood BOOLEAN,
      is_hidden_holdout BOOLEAN,
      difficulty_score FLOAT,
      difficulty_tier STRING,
      difficulty_reasons STRING,
      estimated_memory_bytes NUMBER,
      estimated_memory_gib FLOAT,
      memory_class STRING,
      hidden_holdout_suite_id STRING,
      task_fingerprint STRING,
      training_data_family STRING
    ) DATA_RETENTION_TIME_IN_DAYS = 0
    """
    session.sql(ddl).collect()
    print(f"[INFO] Mixed-cat classification index table {SYNCLS_INDEX_TABLE} ready.", flush=True)


def _validate_mixed_cls_parquet_fields(parquet_path) -> dict:
    """Read scalar metadata from a mixed-cat classification parquet file."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(str(parquet_path))
    d = tbl.to_pydict()

    meta = {}
    for field in [
        "n", "p_num", "p_cat", "n_train", "n_test", "num_classes",
        "prior_regime", "schema_version", "task_family",
        "training_data_family", "task_objective",
        "temperature", "label_noise_rate", "class_imbalance_type",
    ]:
        if field in d:
            meta[field] = d[field][0]

    if "categorical_cardinalities" in d:
        meta["categorical_cardinalities"] = list(d["categorical_cardinalities"][0])

    return meta


def _build_mixed_cls_index_record(
    suite_id: str,
    dataset_id: int,
    stage_path: str,
    meta: dict,
    payload_bytes: int = 0,
) -> dict:
    """Build an index row dict for a mixed-categorical classification task."""
    cat_cards = list(meta.get("categorical_cardinalities", []))
    p_num = int(meta.get("p_num", 0))
    p_cat = int(meta.get("p_cat", 0))
    max_card = max(cat_cards) if cat_cards else 0
    n_train = int(meta.get("n_train", 0))
    n_test = int(meta.get("n_test", 0))
    n_total = n_train + n_test
    return {
        "suite_id": suite_id,
        "suite_family": meta.get("suite_family", "primary"),
        "dataset_id": dataset_id,
        "dataset_seed": meta.get("dataset_seed"),
        "global_idx": meta.get("global_idx", dataset_id),
        "stage_path": stage_path,
        "prior_regime": meta.get("prior_regime"),
        "classification_regime": meta.get("classification_regime"),
        "split_seeds": meta.get("split_seeds", [0]),
        "n_total": n_total,
        "n_train_default": n_train,
        "n_holdout_default": n_test,
        "p_signal": meta.get("p_signal", p_num),
        "p_noise": meta.get("p_noise", 0),
        "p_total": p_num + p_cat,
        "p_num": p_num,
        "p_cat": p_cat,
        "categorical_cardinalities": cat_cards,
        "max_cardinality": max_card,
        "cat_effect_scale": meta.get("cat_effect_scale"),
        "missing_rate": meta.get("missing_rate"),
        "num_classes": meta.get("num_classes", 2),
        "feature_noise_level": meta.get("feature_noise_level"),
        "training_size_anchor": meta.get("training_size_anchor", False),
        "eval_weight": meta.get("eval_weight", 1.0),
        "payload_bytes": payload_bytes,
        "logical_dataset_key": f"{suite_id}:mixed_cls:{dataset_id:04d}",
        "source_suite_id": meta.get("source_suite_id"),
        "task_family": meta.get("task_family", "linear_classification"),
        "task_objective": meta.get("task_objective", "inductive_classification"),
        "label_noise_rate": meta.get("label_noise_rate"),
        "class_imbalance_type": meta.get("class_imbalance_type"),
        "temperature": meta.get("temperature"),
        "margin_level": meta.get("margin_level"),
        "sample_complexity_bucket": meta.get("sample_complexity_bucket"),
        "task_seed": meta.get("task_seed"),
        "schema_version": meta.get("schema_version"),
        "coeff_schema_version": meta.get("coeff_schema_version"),
        "distribution_family": meta.get("distribution_family"),
        "covariance_type": meta.get("covariance_type"),
        "rho": meta.get("rho"),
        "is_training_allowed": meta.get("is_training_allowed", True),
        "is_eval_only": meta.get("is_eval_only", False),
        "is_ood": meta.get("is_ood", False),
        "is_hidden_holdout": meta.get("is_hidden_holdout", False),
        "difficulty_score": meta.get("difficulty_score"),
        "difficulty_tier": meta.get("difficulty_tier"),
        "difficulty_reasons": meta.get("difficulty_reasons", []),
        "estimated_memory_bytes": meta.get("estimated_memory_bytes"),
        "estimated_memory_gib": meta.get("estimated_memory_gib"),
        "memory_class": meta.get("memory_class"),
        "hidden_holdout_suite_id": meta.get("hidden_holdout_suite_id"),
        "task_fingerprint": meta.get("task_fingerprint"),
        "training_data_family": meta.get(
            "training_data_family", "synthetic_linear_classification_mixed_categorical"
        ),
    }


def _insert_mixed_cls_index_rows(session, rows: list[dict]) -> None:
    """Insert mixed-cat classification index rows using raw SQL INSERT."""
    if not rows:
        return

    now_sql = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    col_list = (
        "suite_id, suite_family, dataset_id, dataset_seed, global_idx, stage_path, "
        "prior_regime, classification_regime, split_seeds, "
        "n_total, n_train_default, n_holdout_default, "
        "p_signal, p_noise, p_total, "
        "p_num, p_cat, categorical_cardinalities, max_cardinality, "
        "cat_effect_scale, missing_rate, "
        "num_classes, feature_noise_level, "
        "training_size_anchor, eval_weight, payload_bytes, created_at, "
        "logical_dataset_key, source_suite_id, "
        "task_family, task_objective, label_noise_rate, class_imbalance_type, "
        "temperature, margin_level, sample_complexity_bucket, "
        "task_seed, schema_version, coeff_schema_version, "
        "distribution_family, covariance_type, rho, is_training_allowed, "
        "is_eval_only, is_ood, is_hidden_holdout, difficulty_score, difficulty_tier, "
        "difficulty_reasons, estimated_memory_bytes, estimated_memory_gib, memory_class, "
        "hidden_holdout_suite_id, task_fingerprint, training_data_family"
    )

    select_strings = []
    for r in rows:
        split_seeds_json = json.dumps(list(r.get("split_seeds") or []))
        cat_cards_json = json.dumps(list(r.get("categorical_cardinalities") or []))
        diff_reasons = r.get("difficulty_reasons", [])
        if isinstance(diff_reasons, list):
            diff_reasons = json.dumps(diff_reasons)
        select_strings.append(
            "SELECT "
            f"{_sql_str(r.get('suite_id'))}, "
            f"{_sql_str(r.get('suite_family'))}, "
            f"{_sql_num(r.get('dataset_id'))}, "
            f"{_sql_num(r.get('dataset_seed'))}, "
            f"{_sql_num(r.get('global_idx'))}, "
            f"{_sql_str(r.get('stage_path'))}, "
            f"{_sql_str(r.get('prior_regime'))}, "
            f"{_sql_str(r.get('classification_regime'))}, "
            f"PARSE_JSON({_sql_str(split_seeds_json)}), "
            f"{_sql_num(r.get('n_total'))}, "
            f"{_sql_num(r.get('n_train_default'))}, "
            f"{_sql_num(r.get('n_holdout_default'))}, "
            f"{_sql_num(r.get('p_signal'))}, "
            f"{_sql_num(r.get('p_noise'))}, "
            f"{_sql_num(r.get('p_total'))}, "
            f"{_sql_num(r.get('p_num'))}, "
            f"{_sql_num(r.get('p_cat'))}, "
            f"PARSE_JSON({_sql_str(cat_cards_json)}), "
            f"{_sql_num(r.get('max_cardinality'))}, "
            f"{_sql_float(r.get('cat_effect_scale'))}, "
            f"{_sql_float(r.get('missing_rate'))}, "
            f"{_sql_num(r.get('num_classes'))}, "
            f"{_sql_float(r.get('feature_noise_level'))}, "
            f"{_sql_bool(r.get('training_size_anchor', False))}, "
            f"{_sql_float(r.get('eval_weight', 1.0))}, "
            f"{_sql_num(r.get('payload_bytes', 0))}, "
            f"TO_TIMESTAMP_NTZ({_sql_str(now_sql)}), "
            f"{_sql_str(r.get('logical_dataset_key'))}, "
            f"{_sql_str(r.get('source_suite_id'))}, "
            f"{_sql_str(r.get('task_family'))}, "
            f"{_sql_str(r.get('task_objective'))}, "
            f"{_sql_float(r.get('label_noise_rate'))}, "
            f"{_sql_str(r.get('class_imbalance_type'))}, "
            f"{_sql_float(r.get('temperature'))}, "
            f"{_sql_str(r.get('margin_level'))}, "
            f"{_sql_str(r.get('sample_complexity_bucket'))}, "
            f"{_sql_num(r.get('task_seed'))}, "
            f"{_sql_str(r.get('schema_version'))}, "
            f"{_sql_str(r.get('coeff_schema_version'))}, "
            f"{_sql_str(r.get('distribution_family'))}, "
            f"{_sql_str(r.get('covariance_type'))}, "
            f"{_sql_float(r.get('rho'))}, "
            f"{_sql_bool(r.get('is_training_allowed', True))}, "
            f"{_sql_bool(r.get('is_eval_only', False))}, "
            f"{_sql_bool(r.get('is_ood', False))}, "
            f"{_sql_bool(r.get('is_hidden_holdout', False))}, "
            f"{_sql_float(r.get('difficulty_score'))}, "
            f"{_sql_str(r.get('difficulty_tier'))}, "
            f"{_sql_str(diff_reasons)}, "
            f"{_sql_num(r.get('estimated_memory_bytes'))}, "
            f"{_sql_float(r.get('estimated_memory_gib'))}, "
            f"{_sql_str(r.get('memory_class'))}, "
            f"{_sql_str(r.get('hidden_holdout_suite_id'))}, "
            f"{_sql_str(r.get('task_fingerprint'))}, "
            f"{_sql_str(r.get('training_data_family'))}"
        )

    chunk_size = 100
    for start in range(0, len(select_strings), chunk_size):
        chunk = select_strings[start: start + chunk_size]
        sql = (
            f"INSERT INTO {SYNCLS_INDEX_TABLE} ({col_list})\n"
            + "\nUNION ALL\n".join(chunk)
        )
        session.sql(sql).collect()
        print(
            f"[INFO] Inserted rows {start}\u2013{start + len(chunk) - 1} into {SYNCLS_INDEX_TABLE}.",
            flush=True,
        )


def _prepare_mixed_classification(session) -> str:
    """Prepare mixed-categorical classification evaluation index.

    Discovers parquet files on stage, validates metadata, and populates
    LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX.
    """
    suite_id = SYNCLS_SUITE_ID
    print(
        f"[INFO] _prepare_mixed_classification START suite_id={suite_id} "
        f"index_table={SYNCLS_INDEX_TABLE}",
        flush=True,
    )

    stage_prefix = f"@EVALUATION_DATASET_STAGE/mixed_classification_prepared/{suite_id}"
    parquet_paths = _list_stage_parquets(session, stage_prefix)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found under {stage_prefix}. "
            "Ensure mixed-categorical classification data has been generated and staged."
        )
    print(f"[INFO] Found {len(parquet_paths)} parquet files under {stage_prefix}.", flush=True)

    os.makedirs(SYNCLS_LOCAL_DIR, exist_ok=True)
    _create_mixed_cls_index_table(session)

    # Delete existing rows for this suite
    session.sql(
        f"DELETE FROM {SYNCLS_INDEX_TABLE} WHERE suite_id = '{suite_id}'"
    ).collect()

    index_rows = []
    errors = []

    for dataset_id, stage_path in enumerate(sorted(parquet_paths)):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = _download_parquet(session, stage_path, tmp_dir)
                meta = _validate_mixed_cls_parquet_fields(local_path)
                payload_bytes = local_path.stat().st_size if local_path.exists() else 0
                record = _build_mixed_cls_index_record(
                    suite_id=suite_id,
                    dataset_id=dataset_id,
                    stage_path=stage_path,
                    meta=meta,
                    payload_bytes=payload_bytes,
                )
                index_rows.append(record)
        except Exception as exc:
            errors.append(f"  [{stage_path}]: {exc}")
            print(f"[ERROR] {stage_path}: {exc}", flush=True)

    if errors:
        summary = "\n".join(errors)
        raise RuntimeError(
            f"_prepare_mixed_classification: {len(errors)} parquet(s) failed:\n{summary}"
        )

    _insert_mixed_cls_index_rows(session, index_rows)

    actual = _index_row_count(session, suite_id)
    if actual != len(index_rows):
        raise RuntimeError(
            f"{SYNCLS_INDEX_TABLE} row count {actual} does not match "
            f"expected {len(index_rows)} after insert."
        )

    print(
        f"[INFO] _prepare_mixed_classification DONE. {actual} rows for suite_id={suite_id!r}.",
        flush=True,
    )
    return f"OK suite_id={suite_id} total_rows={actual} (mixed_categorical)"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def prepare_synthetic_classification(session=None) -> str:
    """Stored procedure handler and standalone entry point."""
    if session is None:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()

    # Dispatch to mixed-categorical handler
    if SYNCLS_IS_MIXED_CATEGORICAL:
        return _prepare_mixed_classification(session)

    suite_id = SYNCLS_SUITE_ID
    print(
        f"[INFO] prepare_synthetic_classification START "
        f"suite_id={suite_id} force_rebuild={SYNCLS_FORCE_REBUILD}",
        flush=True,
    )

    if _should_skip(session, suite_id):
        count = _index_row_count(session, suite_id)
        return f"SKIPPED ({count} rows already indexed for suite_id={suite_id!r})"

    # Discover parquet files on stage (canonical path: synthetic_classification_prepared/{suite_id})
    stage_prefix = f"{SYNCLS_STAGE_ROOT}/{suite_id}"
    parquet_paths = _list_stage_parquets(session, stage_prefix)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found under {stage_prefix}. "
            "Ensure classification data has been generated and staged before running prep."
        )
    print(f"[INFO] Found {len(parquet_paths)} parquet files under {stage_prefix}.", flush=True)

    os.makedirs(SYNCLS_LOCAL_DIR, exist_ok=True)
    _create_cls_index_table(session)
    _delete_cls_index_rows(session, suite_id)

    index_rows = []
    errors = []

    for dataset_id, stage_path in enumerate(sorted(parquet_paths)):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = _download_parquet(session, stage_path, tmp_dir)
                meta = _validate_parquet_fields(local_path)
                payload_bytes = local_path.stat().st_size if local_path.exists() else 0
                record = _build_index_record(
                    suite_id=suite_id,
                    dataset_id=dataset_id,
                    stage_path=stage_path,
                    meta=meta,
                    payload_bytes=payload_bytes,
                )
                index_rows.append(record)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"  [{stage_path}]: {exc}")
            print(f"[ERROR] {stage_path}: {exc}", flush=True)
        except Exception as exc:
            errors.append(f"  [{stage_path}]: unexpected error: {exc}")
            print(f"[ERROR] {stage_path}: unexpected error: {exc}", flush=True)

    if errors:
        summary = "\n".join(errors)
        raise RuntimeError(
            f"prepare_synthetic_classification: {len(errors)} parquet(s) failed validation:\n{summary}"
        )

    _insert_cls_index_rows(session, index_rows)

    # Final validation
    actual = _index_row_count(session, suite_id)
    if actual != len(index_rows):
        raise RuntimeError(
            f"{SYNCLS_INDEX_TABLE} row count {actual} does not match "
            f"expected {len(index_rows)} after insert."
        )

    print(
        f"[INFO] prepare_synthetic_classification DONE. "
        f"Total index rows: {actual} for suite_id={suite_id!r}.",
        flush=True,
    )
    return f"OK suite_id={suite_id} total_rows={actual}"


def main() -> None:
    result = prepare_synthetic_classification(session=None)
    print(f"[RESULT] {result}")


if __name__ == "__main__":
    main()
