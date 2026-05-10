"""
prepare_benchmark_datasets.py

Standalone MLJob script that fetches all benchmark datasets exactly once,
stages them as .npz files with no pickle, and writes benchmark_manifest.json.

Run before any benchmark shard jobs:
    CALL prepare_benchmark_datasets();
    LIST @META_DATASET_STAGE/benchmark_prepared/;

Or let run_evaluation_pipeline() handle it automatically.

Idempotent: if benchmark_manifest.json already exists on stage and
BENCHMARK_FORCE_REBUILD=false, validates and exits early.
"""

import datetime
import json
import os

import numpy as np

# SPCS home guard — redirect ~ to writable path before any Snowflake imports
os.environ.setdefault("HOME", "/tmp")

# ---------------------------------------------------------------------------
# Constants (read from env vars with defaults)
# ---------------------------------------------------------------------------

BENCHMARK_PREPARED_STAGE = os.environ.get(
    "BENCHMARK_PREPARED_STAGE", "@META_DATASET_STAGE/benchmark_prepared/"
)
BENCHMARK_FORCE_REBUILD = os.environ.get("BENCHMARK_FORCE_REBUILD", "false").lower() == "true"
BENCHMARK_INCLUDE_OPENML = os.environ.get("BENCHMARK_INCLUDE_OPENML", "true").lower() == "true"
BENCHMARK_INCLUDE_KAGGLE = os.environ.get("BENCHMARK_INCLUDE_KAGGLE", "true").lower() == "true"
BENCHMARK_REQUIRE_KAGGLE = os.environ.get("BENCHMARK_REQUIRE_KAGGLE", "false").lower() == "true"
OPENML_TARGET_N = int(os.environ.get("BENCHMARK_OPENML_TARGET_N", "28"))
EVAL_RESULTS_STAGE = os.environ.get("EVAL_RESULTS_STAGE", "@EVALUATION_RESULTS_STAGE")
BENCHMARK_DATASET_INDEX_TABLE = os.environ.get(
    "BENCHMARK_DATASET_INDEX_TABLE", "BENCHMARK_DATASET_INDEX"
)

KAGGLE_STAGE_PATH = "@META_DATASET_STAGE/kaggle/"
LOCAL_PREP_DIR = "/tmp/benchmark_prep"
MANIFEST_FILENAME = "benchmark_manifest.json"

# Dataset filtering constants (must match evaluate.py)
MAX_SAMPLES = 10_000
MAX_FEATURES = 500
MAX_CAT_FRACTION = 0.3
TRAIN_FRACTION = 0.9
SEEDS = list(range(10))


# ---------------------------------------------------------------------------
# Snowflake session helper
# ---------------------------------------------------------------------------

def _get_session():
    from snowflake.snowpark import Session
    return Session.builder.getOrCreate()


def _stage_file_exists(session, stage_prefix, filename):
    """Return True if filename exists under stage_prefix."""
    try:
        rows = session.sql(f"LIST {stage_prefix}").collect()
        return any(str(row[0]).rstrip("/").endswith(f"/{filename}") for row in rows)
    except Exception as exc:
        print(f"  [WARN] LIST {stage_prefix} failed: {exc}")
        return False


def _sql_literal(value):
    """Return a simple Snowflake SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _benchmark_weight(n_samples, n_features):
    """Default cost proxy for benchmark sharding."""
    return int(n_samples) * int(n_features)


def _utc_now_iso():
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def _manifest_entry(dataset_index, source, task_id, dataset_id, name, stage_path,
                    X, categorical_indicator, local_path, created_at_utc):
    estimated_bytes = os.path.getsize(local_path) if os.path.exists(local_path) else None
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    return {
        "dataset_index":       int(dataset_index),
        "source":              str(source),
        "task_id":             str(task_id),
        "dataset_id":          str(dataset_id),
        "name":                str(name),
        "stage_path":          str(stage_path),
        "n_samples":           n_samples,
        "n_features":          n_features,
        "estimated_bytes":     int(estimated_bytes) if estimated_bytes is not None else None,
        "benchmark_weight":    _benchmark_weight(n_samples, n_features),
        "created_at_utc":      created_at_utc,
        "categorical_indicator": [bool(v) for v in categorical_indicator],
    }


def _create_benchmark_dataset_index_table(session):
    """Create the manifest metadata index table; do not optimize staged .npz files."""
    session.sql(f"""
CREATE TRANSIENT TABLE IF NOT EXISTS {BENCHMARK_DATASET_INDEX_TABLE} (
  dataset_index NUMBER NOT NULL,
  source STRING,
  task_id STRING,
  dataset_id STRING,
  name STRING,
  stage_path STRING,
  n_samples NUMBER,
  n_features NUMBER,
  estimated_bytes NUMBER,
  benchmark_weight NUMBER,
  created_at_utc TIMESTAMP_NTZ
)
DATA_RETENTION_TIME_IN_DAYS = 0
""").collect()


def _write_benchmark_dataset_index(session, manifest_datasets, created_at_utc):
    """
    Rebuild BENCHMARK_DATASET_INDEX from manifest metadata.

    This table is a lightweight query/index layer. The prepared .npz files stay
    as opaque staged binary payloads and are not split, clustered, or inspected.
    """
    _create_benchmark_dataset_index_table(session)
    session.sql(f"DELETE FROM {BENCHMARK_DATASET_INDEX_TABLE}").collect()
    if not manifest_datasets:
        return

    created_sql = created_at_utc.rstrip("Z").replace("T", " ")
    values = []
    for ds in manifest_datasets:
        estimated = ds.get("estimated_bytes")
        values.append(
            "("
            f"{int(ds['dataset_index'])}, "
            f"{_sql_literal(ds['source'])}, "
            f"{_sql_literal(ds['task_id'])}, "
            f"{_sql_literal(ds.get('dataset_id', ''))}, "
            f"{_sql_literal(ds['name'])}, "
            f"{_sql_literal(ds['stage_path'])}, "
            f"{int(ds['n_samples'])}, "
            f"{int(ds['n_features'])}, "
            f"{int(estimated) if estimated is not None else 'NULL'}, "
            f"{int(ds.get('benchmark_weight') or _benchmark_weight(ds['n_samples'], ds['n_features']))}, "
            f"TO_TIMESTAMP_NTZ({_sql_literal(created_sql)})"
            ")"
        )

    for start in range(0, len(values), 100):
        chunk = values[start:start + 100]
        session.sql(f"""
INSERT INTO {BENCHMARK_DATASET_INDEX_TABLE} (
  dataset_index, source, task_id, dataset_id, name, stage_path,
  n_samples, n_features, estimated_bytes, benchmark_weight, created_at_utc
)
VALUES
{", ".join(chunk)}
""").collect()

    count = session.sql(
        f"SELECT COUNT(*) AS N FROM {BENCHMARK_DATASET_INDEX_TABLE}"
    ).collect()[0][0]
    if int(count) != len(manifest_datasets):
        raise RuntimeError(
            f"{BENCHMARK_DATASET_INDEX_TABLE} row count {count} does not match "
            f"manifest dataset count {len(manifest_datasets)}"
        )


def _validate_and_backfill_existing_manifest(manifest):
    """
    Validate required payload metadata and backfill metadata-only fields.

    The staged .npz files stay unchanged; this only updates manifest/index
    fields needed for deterministic benchmark sharding.
    """
    if "datasets" not in manifest or not manifest["datasets"]:
        raise ValueError("existing manifest has no datasets")

    changed = False
    created_at_utc = manifest.get("created_at_utc")
    if not created_at_utc:
        created_at_utc = _utc_now_iso()
        manifest["created_at_utc"] = created_at_utc
        changed = True

    seen = set()
    payload_fields = (
        "dataset_index", "source", "task_id", "name", "stage_path",
        "n_samples", "n_features", "categorical_indicator",
    )
    for ds in manifest["datasets"]:
        for field in payload_fields:
            if field not in ds:
                raise ValueError(f"existing manifest missing payload field {field!r}")

        dataset_index = int(ds["dataset_index"])
        if dataset_index in seen:
            raise ValueError(f"existing manifest has duplicate dataset_index={dataset_index}")
        seen.add(dataset_index)

        if "estimated_bytes" not in ds:
            ds["estimated_bytes"] = None
            changed = True
        if "benchmark_weight" not in ds or ds["benchmark_weight"] is None:
            ds["benchmark_weight"] = _benchmark_weight(ds["n_samples"], ds["n_features"])
            changed = True
        if "created_at_utc" not in ds or not ds["created_at_utc"]:
            ds["created_at_utc"] = created_at_utc
            changed = True

    return manifest, changed


def _upload_manifest(session, manifest, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    manifest_local = os.path.join(local_dir, MANIFEST_FILENAME)
    with open(manifest_local, "w") as f:
        json.dump(manifest, f, indent=2)
    session.file.put(
        manifest_local,
        BENCHMARK_PREPARED_STAGE,
        overwrite=True,
        auto_compress=False,
    )


# ---------------------------------------------------------------------------
# OpenML fetching and benchmark dataset selection.
#
# This module is the OpenML owner. Evaluation shards consume only the prepared
# manifest and .npz files written here; evaluate.py must not import or call
# OpenML APIs.
# ---------------------------------------------------------------------------

def _fetch_openml_datasets(target_n=OPENML_TARGET_N,
                            max_samples=MAX_SAMPLES,
                            max_features=MAX_FEATURES,
                            max_cat_fraction=MAX_CAT_FRACTION):
    """
    Fetch regression benchmark datasets from OpenML studies 271/269/218 + 353.
    Filtering criteria are owned here and mirrored only as prepared manifest
    metadata consumed by evaluate.py.

    Returns list of dicts:
        {task_id, dataset_id, name, X (float64 ndarray), y (float64 ndarray),
         categorical_indicator (list of bool)}
    """
    import openml

    openml.config.cache_directory = "/tmp/openml_cache"
    os.makedirs("/tmp/openml_cache", exist_ok=True)

    task_ids = set()

    # 1. AutoML Benchmark regression study (try known IDs; fall back to tag search)
    for study_id in [271, 269, 218]:
        try:
            suite = openml.study.get_suite(study_id)
            if suite.tasks:
                task_ids.update(suite.tasks)
                print(f"  AutoML Benchmark study {study_id}: {len(suite.tasks)} tasks")
                break
        except Exception:
            continue

    # 2. OpenML-CTR23 regression study (study 353)
    try:
        ctr23 = openml.study.get_suite(353)
        task_ids.update(ctr23.tasks)
        print(f"  OpenML-CTR23 study 353: {len(ctr23.tasks)} tasks")
    except Exception as exc:
        print(f"  [WARN] Could not load CTR23 study 353: {exc}")

    datasets = []
    seen_did = set()

    for task_id in sorted(task_ids):
        if len(datasets) >= target_n:
            break
        try:
            task = openml.tasks.get_task(task_id)
            if "Regression" not in str(task.task_type):
                continue
            dataset = task.get_dataset()
            # Check metadata before fetching full data where possible
            if hasattr(dataset, "qualities"):
                try:
                    n_rows = dataset.qualities.get("NumberOfInstances")
                    n_cols = dataset.qualities.get("NumberOfFeatures")
                    if n_rows and int(n_rows) > max_samples:
                        continue
                    if n_cols and int(n_cols) > max_features:
                        continue
                except Exception:
                    pass

            X, y, categorical_indicator, _ = dataset.get_data(
                dataset_format="array", target=task.target_name
            )
            if X is None or y is None:
                continue
            if X.shape[0] > max_samples or X.shape[1] > max_features:
                continue
            if categorical_indicator is not None:
                cat_frac = sum(categorical_indicator) / max(len(categorical_indicator), 1)
                if cat_frac > max_cat_fraction:
                    continue
            did = dataset.dataset_id
            if did in seen_did:
                continue
            seen_did.add(did)
            if categorical_indicator is None:
                categorical_indicator = [False] * X.shape[1]
            datasets.append({
                "task_id":               task_id,
                "dataset_id":            did,
                "name":                  dataset.name,
                "X":                     X.astype(np.float64),
                "y":                     y.astype(np.float64),
                "categorical_indicator": list(categorical_indicator),
            })
            print(f"  Fetched OpenML: {dataset.name} "
                  f"(tid={task_id}, did={did}, n={X.shape[0]}, p={X.shape[1]})")
        except Exception as exc:
            print(f"  [SKIP] task {task_id}: {exc}")

    print(f"Fetched {len(datasets)}/{target_n} OpenML benchmark datasets.")
    return datasets


# ---------------------------------------------------------------------------
# Kaggle fetching
# ---------------------------------------------------------------------------

def _fetch_kaggle_datasets(session,
                            stage_path=KAGGLE_STAGE_PATH,
                            local_dir=None):
    """
    Download .npz files from the Kaggle stage prefix.
    Validates required arrays; normalises into the returned list.
    Returns list of dicts or raises if BENCHMARK_REQUIRE_KAGGLE=true and stage is empty.
    """
    local_dir = local_dir or os.path.join(LOCAL_PREP_DIR, "kaggle_raw")
    os.makedirs(local_dir, exist_ok=True)

    required_fields = {"X", "y", "categorical_indicator", "task_type", "slug", "dataset_name"}

    try:
        rows = session.sql(f"LIST {stage_path}").collect()
        if not rows:
            msg = f"Kaggle stage {stage_path} is empty."
            if BENCHMARK_REQUIRE_KAGGLE:
                raise RuntimeError(msg)
            print(f"  [WARN] {msg}  Continuing with OpenML only.")
            return []
        for row in rows:
            fname = os.path.basename(row["name"])
            local_path = os.path.join(local_dir, fname)
            if not os.path.exists(local_path):
                session.file.get(f"{stage_path}{fname}", local_dir)
        print(f"  Downloaded {len(rows)} Kaggle file(s) from {stage_path}")
    except RuntimeError:
        raise
    except Exception as exc:
        msg = f"Kaggle stage unavailable ({exc})."
        if BENCHMARK_REQUIRE_KAGGLE:
            raise RuntimeError(msg) from exc
        print(f"  [WARN] {msg}  Continuing with OpenML only.")
        return []

    datasets = []
    for fname in sorted(os.listdir(local_dir)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(local_dir, fname)
        try:
            data = np.load(fpath, allow_pickle=True)
            missing = required_fields - set(data.files)
            if missing:
                print(f"  [SKIP Kaggle] {fname}: missing fields {sorted(missing)}")
                continue
            task_type = str(data["task_type"][0])
            if task_type != "regression":
                print(f"  [SKIP Kaggle] {fname}: task_type={task_type!r} (want regression)")
                continue
            cat_mask = np.asarray(data["categorical_indicator"])
            if cat_mask.any():
                slug = str(data["slug"][0])
                print(f"  [SKIP Kaggle] {slug}: {cat_mask.sum()} categorical feature(s) "
                      f"— model trained on numeric DGPs only")
                continue
            X = data["X"].astype(np.float64)
            y = data["y"].astype(np.float64)
            slug = str(data["slug"][0])
            name = str(data["dataset_name"][0])
            datasets.append({
                "slug":                  slug,
                "name":                  name,
                "X":                     X,
                "y":                     y,
                "categorical_indicator": cat_mask.tolist(),
            })
            print(f"  Loaded Kaggle: {name} (slug={slug}, n={X.shape[0]}, p={X.shape[1]})")
        except Exception as exc:
            print(f"  [SKIP Kaggle] {fname}: {exc}")

    print(f"Loaded {len(datasets)} Kaggle regression dataset(s).")
    return datasets


# ---------------------------------------------------------------------------
# Dataset serialization
# ---------------------------------------------------------------------------

def _serialize_dataset(local_path, source, task_id, dataset_id, name, X, y,
                        categorical_indicator):
    """Write a single prepared .npz dataset; no pickle."""
    np.savez_compressed(
        local_path,
        X=X.astype(np.float64),
        y=y.astype(np.float64),
        categorical_indicator=np.asarray(categorical_indicator, dtype=bool),
        source=np.array([source]),
        task_id=np.array([str(task_id)]),
        dataset_id=np.array([str(dataset_id)]),
        dataset_name=np.array([str(name)]),
        problem_type=np.array(["regression"]),
    )


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------

def _validate_existing_manifest(session):
    """
    Download and validate the existing manifest from stage.
    Returns True if valid, False if it should be rebuilt.
    """
    local_dir = "/tmp/benchmark_manifest_check"
    os.makedirs(local_dir, exist_ok=True)
    try:
        manifest_stage_path = (
            BENCHMARK_PREPARED_STAGE.rstrip("/") + "/" + MANIFEST_FILENAME
        )
        session.file.get(manifest_stage_path, local_dir)
        local_path = os.path.join(local_dir, MANIFEST_FILENAME)
        with open(local_path) as f:
            manifest = json.load(f)
        try:
            manifest, changed = _validate_and_backfill_existing_manifest(manifest)
        except ValueError as exc:
            print(f"  Existing manifest invalid ({exc}) - will rebuild.")
            return False

        created_at_utc = manifest["created_at_utc"]
        if changed:
            print("  Existing manifest needed metadata backfill; rewriting manifest.")
            _upload_manifest(session, manifest, local_dir)

        _write_benchmark_dataset_index(session, manifest["datasets"], created_at_utc)
        print(f"  Existing manifest valid: {len(manifest['datasets'])} datasets. "
              f"{BENCHMARK_DATASET_INDEX_TABLE} refreshed. "
              f"Exiting early (set BENCHMARK_FORCE_REBUILD=true to rebuild).")
        return True
        """
        if "datasets" not in manifest or not manifest["datasets"]:
            print("  Existing manifest has no datasets — will rebuild.")
            return False
        for ds in manifest["datasets"]:
            for field in ("dataset_index", "source", "task_id", "name",
                          "stage_path", "n_samples", "n_features",
                          "estimated_bytes", "benchmark_weight",
                          "categorical_indicator"):
                if field not in ds:
                    print(f"  Existing manifest missing field {field!r} — will rebuild.")
                    return False
        created_at_utc = manifest.get("created_at_utc")
        if not created_at_utc:
            print("  Existing manifest missing created_at_utc - will rebuild.")
            return False
        _write_benchmark_dataset_index(session, manifest["datasets"], created_at_utc)
        print(f"  Existing manifest valid: {len(manifest['datasets'])} datasets. "
              f"{BENCHMARK_DATASET_INDEX_TABLE} refreshed. "
              f"Exiting early (set BENCHMARK_FORCE_REBUILD=true to rebuild).")
        return True
        """
    except Exception as exc:
        print(f"  Could not validate existing manifest ({exc}) — will rebuild.")
        return False


# ---------------------------------------------------------------------------
# Main preparation logic
# ---------------------------------------------------------------------------

def prepare_datasets(session=None):
    """
    Prepare all benchmark datasets and write to stage.
    Can be called as a Snowpark stored procedure handler or as main().
    """
    import traceback

    print("=" * 70)
    print("prepare_benchmark_datasets — benchmark dataset preparation")
    print(f"  BENCHMARK_PREPARED_STAGE = {BENCHMARK_PREPARED_STAGE}")
    print(f"  BENCHMARK_FORCE_REBUILD  = {BENCHMARK_FORCE_REBUILD}")
    print(f"  BENCHMARK_INCLUDE_OPENML = {BENCHMARK_INCLUDE_OPENML}")
    print(f"  BENCHMARK_INCLUDE_KAGGLE = {BENCHMARK_INCLUDE_KAGGLE}")
    print(f"  OPENML_TARGET_N          = {OPENML_TARGET_N}")
    print("=" * 70, flush=True)

    if session is None:
        session = _get_session()

    # --- Idempotency check ---
    if not BENCHMARK_FORCE_REBUILD:
        manifest_exists = _stage_file_exists(session, BENCHMARK_PREPARED_STAGE, MANIFEST_FILENAME)
        if manifest_exists:
            if _validate_existing_manifest(session):
                return f"Benchmark datasets already prepared. Use BENCHMARK_FORCE_REBUILD=true to rebuild."

    # --- Local working directory ---
    openml_local_dir = os.path.join(LOCAL_PREP_DIR, "openml")
    kaggle_local_dir = os.path.join(LOCAL_PREP_DIR, "kaggle")
    os.makedirs(openml_local_dir, exist_ok=True)
    os.makedirs(kaggle_local_dir, exist_ok=True)

    try:
        manifest_datasets = []
        dataset_index = 0
        created_at_utc = datetime.datetime.utcnow().isoformat() + "Z"

        # --- OpenML ---
        if BENCHMARK_INCLUDE_OPENML:
            print("\nFetching OpenML benchmark datasets...", flush=True)
            try:
                import openml  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "openml package is not installed. "
                    "Add 'openml' to pip_requirements in _submit_dataset_prep()."
                )
            openml_datasets = _fetch_openml_datasets(target_n=OPENML_TARGET_N)

            for ds in openml_datasets:
                task_id = ds["task_id"]
                dataset_id = ds["dataset_id"]
                name = ds["name"]
                X = ds["X"]
                y = ds["y"]
                cat_ind = ds["categorical_indicator"]

                fname = f"openml_{task_id}_{dataset_id}.npz"
                local_path = os.path.join(openml_local_dir, fname)
                stage_path = (
                    BENCHMARK_PREPARED_STAGE.rstrip("/")
                    + "/openml/"
                    + fname
                )

                _serialize_dataset(
                    local_path, "openml", task_id, dataset_id, name, X, y, cat_ind
                )

                print(f"  Uploading {fname} ...", flush=True)
                session.file.put(
                    local_path,
                    BENCHMARK_PREPARED_STAGE.rstrip("/") + "/openml/",
                    overwrite=True,
                    auto_compress=False,
                )

                manifest_datasets.append(
                    _manifest_entry(
                        dataset_index, "openml", task_id, dataset_id, name,
                        stage_path, X, cat_ind, local_path, created_at_utc
                    )
                )
                dataset_index += 1

        else:
            print("BENCHMARK_INCLUDE_OPENML=false — skipping OpenML.", flush=True)

        # --- Kaggle ---
        if BENCHMARK_INCLUDE_KAGGLE:
            print("\nFetching Kaggle benchmark datasets...", flush=True)
            kaggle_datasets = _fetch_kaggle_datasets(session)

            for ds in kaggle_datasets:
                slug = ds["slug"]
                name = ds["name"]
                X = ds["X"]
                y = ds["y"]
                cat_ind = ds["categorical_indicator"]

                fname = f"{slug}.npz"
                local_path = os.path.join(kaggle_local_dir, fname)
                stage_path = (
                    BENCHMARK_PREPARED_STAGE.rstrip("/")
                    + "/kaggle/"
                    + fname
                )

                _serialize_dataset(
                    local_path, "kaggle", slug, slug, name, X, y, cat_ind
                )

                print(f"  Uploading {fname} ...", flush=True)
                session.file.put(
                    local_path,
                    BENCHMARK_PREPARED_STAGE.rstrip("/") + "/kaggle/",
                    overwrite=True,
                    auto_compress=False,
                )

                manifest_datasets.append(
                    _manifest_entry(
                        dataset_index, "kaggle", slug, slug, name,
                        stage_path, X, cat_ind, local_path, created_at_utc
                    )
                )
                dataset_index += 1

        else:
            print("BENCHMARK_INCLUDE_KAGGLE=false — skipping Kaggle.", flush=True)

        if not manifest_datasets:
            raise RuntimeError(
                "No benchmark datasets were prepared. "
                "Check OpenML connectivity and Kaggle stage contents."
            )

        # --- Write manifest ---
        manifest = {
            "manifest_format_version": 1,
            "created_at_utc": created_at_utc,
            "benchmark_protocol": {
                "train_fraction":     TRAIN_FRACTION,
                "seeds":              SEEDS,
                "max_samples":        MAX_SAMPLES,
                "max_features":       MAX_FEATURES,
                "openml_target_n":    OPENML_TARGET_N,
            },
            "datasets": manifest_datasets,
        }

        print(f"\nUploading {MANIFEST_FILENAME} ...", flush=True)
        _upload_manifest(session, manifest, LOCAL_PREP_DIR)
        print(f"Refreshing {BENCHMARK_DATASET_INDEX_TABLE} ...", flush=True)
        _write_benchmark_dataset_index(session, manifest_datasets, created_at_utc)

        summary = (
            f"Benchmark dataset preparation complete: "
            f"{len(manifest_datasets)} datasets staged to {BENCHMARK_PREPARED_STAGE}"
        )
        print(summary, flush=True)
        return summary

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[PREP FAILURE]\n{tb}", flush=True)
        _upload_failure_artifact(session, exc, tb)
        raise


def _upload_failure_artifact(session, exc, tb):
    """Write and upload a failure JSON artifact."""
    import json as _json

    failure = {
        "job":           "benchmark_dataset_preparation",
        "error_type":    type(exc).__name__,
        "error_message": str(exc),
        "traceback":     tb,
    }
    local_path = "/tmp/benchmark_dataset_preparation_failure.json"
    with open(local_path, "w") as f:
        _json.dump(failure, f, indent=2)
    try:
        session.file.put(
            local_path,
            f"{EVAL_RESULTS_STAGE}/failures/",
            overwrite=True,
            auto_compress=False,
        )
        print(f"Uploaded failure artifact to {EVAL_RESULTS_STAGE}/failures/", flush=True)
    except Exception as upload_exc:
        print(f"[WARN] Could not upload failure artifact: {upload_exc}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    prepare_datasets()


if __name__ == "__main__":
    main()
