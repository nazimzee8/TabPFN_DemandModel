"""
prepare_synthetic_regression.py
================================
Synthetic-regression equivalent of src/prepare_benchmark_datasets.py.

Generates and stages all synthetic payloads for the split-phase evaluation suite;
creates/refreshes SYNTHETIC_REGRESSION_DATASET_INDEX.

Idempotent by default; full rebuild when SYNTHETIC_REGRESSION_FORCE_REBUILD=true.

Suite families
--------------
* primary          – 200 datasets × 5 split seeds
* feature_noise    – 80 datasets × 3 split seeds × 6 noise levels
* training_size    – 40 datasets × 3 split seeds × 8 n_train_grid values
* target_noise     – 40 datasets × 3 split seeds × 5 noise scales  (optional)
"""

from __future__ import annotations

import os
# SPCS home guard — redirect ~ to writable path before any Snowflake imports.
# Mirrors prepare_benchmark_datasets.py line 24.
os.environ.setdefault("HOME", "/tmp")

import gc
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Constants (env-overridable)
# ---------------------------------------------------------------------------

SYNREG_SUITE_ID = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
SYNREG_BASE_SEED = int(os.getenv("SYNTHETIC_REGRESSION_BASE_SEED", "20260512"))
SYNREG_FORCE_REBUILD = _env_flag("SYNTHETIC_REGRESSION_FORCE_REBUILD", "false")
SYNREG_DROP_INDEX_TABLE = _env_flag("SYNTHETIC_REGRESSION_DROP_INDEX_TABLE", "false")
SYNREG_STAGE_PREFIX = "@EVALUATION_DATASET_STAGE"
SYNREG_MANIFEST_PATH = f"{SYNREG_STAGE_PREFIX}/synthetic_regression_manifest.json"
SYNREG_INDEX_TABLE = "SYNTHETIC_REGRESSION_DATASET_INDEX"
SYNREG_LOCAL_DIR = os.getenv("SYNTHETIC_REGRESSION_LOCAL_DIR", "/tmp/synreg_prep")

# Primary suite
PRIMARY_N_DATASETS = int(os.getenv("SYNTHETIC_REGRESSION_PRIMARY_DATASETS", "200"))
PRIMARY_SPLIT_SEEDS = _parse_int_list(os.getenv("SYNTHETIC_REGRESSION_PRIMARY_SPLIT_SEEDS", "0,1,2,3,4"))

# Feature-noise suite
FEATURE_NOISE_N_DATASETS = int(os.getenv("SYNTHETIC_REGRESSION_FEATURE_NOISE_DATASETS", "80"))
FEATURE_NOISE_SEEDS = _parse_int_list(os.getenv("SYNTHETIC_REGRESSION_FEATURE_NOISE_SEEDS", "0,1,2"))
FEATURE_NOISE_LEVELS = _parse_int_list(
    os.getenv("SYNTHETIC_REGRESSION_FEATURE_NOISE_LEVELS", "0,10,25,50,75,100")
)

# Training-size suite
TRAIN_SIZE_N_DATASETS = int(os.getenv("SYNTHETIC_REGRESSION_TRAIN_SIZE_DATASETS", "40"))
TRAIN_SIZE_SEEDS = _parse_int_list(os.getenv("SYNTHETIC_REGRESSION_TRAIN_SIZE_SEEDS", "0,1,2"))
TRAIN_SIZE_GRID = _parse_int_list(
    os.getenv("SYNTHETIC_REGRESSION_TRAIN_SIZE_GRID", "25,50,100,200,500,1000,2000,4832")
)
HOLDOUT_SIZE = int(os.getenv("SYNTHETIC_REGRESSION_HOLDOUT_SIZE", "1371"))
TRAIN_SIZE_ANCHOR_N = 4832  # hard-coded TabPFN anchor

# Target-noise suite (optional)
ENABLE_TARGET_NOISE_SUITE = _env_flag("SYNTHETIC_REGRESSION_ENABLE_TARGET_NOISE_SUITE", "false")
TARGET_NOISE_N_DATASETS = int(os.getenv("SYNTHETIC_REGRESSION_TARGET_NOISE_DATASETS", "40"))
TARGET_NOISE_SEEDS = _parse_int_list(os.getenv("SYNTHETIC_REGRESSION_TARGET_NOISE_SEEDS", "0,1,2"))
TARGET_NOISE_SCALES = _parse_float_list(
    os.getenv("SYNTHETIC_REGRESSION_TARGET_NOISE_SCALES", "0.0,0.1,0.25,0.5,1.0")
)

# Feature caps
SYNREG_DEEPSET_FEATURE_CAP = int(os.getenv("SYNTHETIC_REGRESSION_DEEPSET_FEATURE_CAP", "128"))

REGIMES = ["A", "B", "C", "D"]
PRIOR_NAME = "linear_poisson"
PRIOR_VERSION = "v1"

# Combined suite (index-level composition of primary in-distribution + OOD full datasets)
COMBINED_SUITE_ID       = "linear_all_v1"
COMBINED_SPLIT_SEEDS    = [0, 1, 2]
COMBINED_N_DATASETS     = 400  # 200 primary (A/B/C/D) + 200 OOD (E/F/G/H)
COMBINED_REGIMES        = ["A", "B", "C", "D", "E", "F", "G", "H"]
COMBINED_N_PER_REGIME   = COMBINED_N_DATASETS // len(COMBINED_REGIMES)  # 50
COMBINED_PRIMARY_SUITE  = os.getenv("COMBINED_PRIMARY_SUITE_ID", "linear_poisson_v1_recommended")
COMBINED_OOD_SUITE      = os.getenv("COMBINED_OOD_SUITE_ID", "ood_linear_full_v1")


# ---------------------------------------------------------------------------
# DGP helpers (mirrors generate_dgp.py logic)
# ---------------------------------------------------------------------------

def _generate_X_regime_A(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Standard normal features."""
    return rng.standard_normal((n, p))


def _generate_X_regime_D(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """AR(1) features with rho=0.6."""
    X = np.zeros((n, p))
    X[0] = rng.standard_normal(p)
    for t in range(1, n):
        X[t] = 0.6 * X[t - 1] + math.sqrt(0.64) * rng.standard_normal(p)
    return X


def _generate_features(rng: np.random.Generator, n: int, p: int, regime: str) -> np.ndarray:
    if regime == "D":
        return _generate_X_regime_D(rng, n, p)
    return _generate_X_regime_A(rng, n, p)


def _generate_beta(rng: np.random.Generator, p: int, regime: str) -> np.ndarray:
    if regime == "B":
        beta = rng.normal(0, 2, p)
        mask = rng.random(p) < 0.70
        beta[mask] = 0.0
    else:
        beta = rng.standard_normal(p)
    return beta


def _generate_noise(rng: np.random.Generator, n: int, regime: str) -> np.ndarray:
    if regime == "C":
        return rng.standard_t(df=3, size=n)
    return rng.standard_normal(n)


def _generate_core_dataset(
    rng: np.random.Generator, n: int, p: int, regime: str, target_noise_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, betaX, y). X shape (n, p)."""
    X = _generate_features(rng, n, p, regime)
    beta = _generate_beta(rng, p, regime)
    betaX = X @ beta
    eps = _generate_noise(rng, n, regime)
    y = betaX + target_noise_scale * eps
    return X, betaX, y


# ---------------------------------------------------------------------------
# sample_params variants
# ---------------------------------------------------------------------------

def sample_params_primary(rng: np.random.Generator) -> tuple[int, int]:
    """
    Sample (n, p) for the primary suite using same constraints as generate_dgp.sample_params.
    n ~ Poisson(200), p ~ Poisson(10), p >= 1, n >= 5, n >= 5*p.
    """
    while True:
        n = int(rng.poisson(200))
        p = int(rng.poisson(10))
        if p >= 1 and n >= 5 and n >= 5 * p:
            return n, p


def sample_params_training_size(rng: np.random.Generator, n_required: int) -> tuple[int, int]:
    """
    Rejection sample (n, p) for training-size suite.
    n >= n_required, p >= 1, n >= max(5, p).
    """
    while True:
        p = int(rng.poisson(10))
        if p < 1:
            continue
        n = n_required  # fixed at n_required for this suite
        if n >= max(5, p):
            return n, p


def sample_params_for_signal(rng: np.random.Generator) -> tuple[int, int, int]:
    """
    Sample (n, p_signal, n_total) for feature-noise suite.
    p_signal ~ Poisson(10), n >= max(5*p_signal, 50).
    Returns (n, p_signal).
    """
    while True:
        p_signal = int(rng.poisson(10))
        if p_signal < 1:
            continue
        n = int(rng.poisson(200))
        n = max(n, 5 * p_signal, 50)
        if n >= 5:
            return n, p_signal


# ---------------------------------------------------------------------------
# Dataset generation wrappers
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    rng: np.random.Generator,
    regime: str,
    suite_family: str,
    n: int,
    p_signal: int,
    p_noise: int = 0,
    target_noise_scale: float = 1.0,
) -> dict:
    """Generate a single synthetic dataset and return a dict of arrays."""
    p_total = p_signal + p_noise
    X_signal, betaX, y = _generate_core_dataset(rng, n, p_signal, regime, target_noise_scale)

    if p_noise > 0:
        X_noise = rng.standard_normal((n, p_noise))
        X = np.concatenate([X_signal, X_noise], axis=1)
    else:
        X = X_signal

    return {
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "suite_family": suite_family,
        "prior_regime": regime,
        "n_total": n,
        "p_signal": p_signal,
        "p_noise": p_noise,
        "p_total": p_total,
        "target_noise_scale": target_noise_scale,
    }


def serialize_synthetic_npz(outpath: str, arrays_dict: dict, noise_level: int = 0,
                              is_anchor: bool = False) -> None:
    """
    Serialize a synthetic dataset dict to a compressed NPZ file.
    All metadata stored as numeric 1-element arrays (no Python objects).
    """
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outpath,
        X=arrays_dict["X"].astype(np.float64),
        y=arrays_dict["y"].astype(np.float64),
        betaX=arrays_dict["betaX"].astype(np.float64),
        suite_family=np.array([arrays_dict["suite_family"]]),
        prior_regime=np.array([arrays_dict["prior_regime"]]),
        n_total=np.array([arrays_dict["n_total"]], dtype=np.int64),
        p_signal=np.array([arrays_dict["p_signal"]], dtype=np.int64),
        p_noise=np.array([arrays_dict["p_noise"]], dtype=np.int64),
        p_total=np.array([arrays_dict["p_total"]], dtype=np.int64),
        target_noise_scale=np.array([arrays_dict["target_noise_scale"]], dtype=np.float64),
        training_size_anchor=np.array([is_anchor], dtype=bool),
        feature_noise_level=np.array([noise_level], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Snowflake stage helpers
# ---------------------------------------------------------------------------

def upload_file_to_stage(session, local_path: str, stage_dir: str) -> str:
    """Upload a local file to a Snowflake stage directory. Returns the stage path."""
    result = session.file.put(
        local_path,
        stage_dir,
        auto_compress=False,
        overwrite=True,
    )
    filename = Path(local_path).name
    stage_path = f"{stage_dir}/{filename}"
    return stage_path


def _stage_file_list(session, stage_prefix: str) -> set[str]:
    """Return set of filenames currently present under a stage prefix."""
    try:
        rows = session.sql(f"LIST {stage_prefix}").collect()
        return {r["name"] for r in rows}
    except Exception:
        return set()


def create_or_validate_manifest(session, suite_id: str, manifest_dict: dict | None = None) -> bool:
    """
    Download and validate the manifest from stage.
    Returns True if manifest is valid and all stage_path values are present (no rebuild needed).
    SYNREG_FORCE_REBUILD=true always returns False.
    """
    if SYNREG_FORCE_REBUILD:
        print("[INFO] SYNTHETIC_REGRESSION_FORCE_REBUILD=true — skipping manifest validation.")
        return False

    # Try to download the manifest
    local_manifest = os.path.join(SYNREG_LOCAL_DIR, "synthetic_regression_manifest.json")
    os.makedirs(SYNREG_LOCAL_DIR, exist_ok=True)
    try:
        session.file.get(SYNREG_MANIFEST_PATH, SYNREG_LOCAL_DIR)
    except Exception as e:
        print(f"[INFO] No existing manifest found ({e}). Will generate fresh.")
        return False

    try:
        with open(local_manifest) as f:
            existing = json.load(f)
    except Exception as e:
        print(f"[INFO] Could not parse existing manifest: {e}. Will regenerate.")
        return False

    if existing.get("suite_id") != suite_id:
        print(f"[INFO] Manifest suite_id mismatch ({existing.get('suite_id')} != {suite_id}). Rebuilding.")
        return False

    # Check required count fields
    required_keys = ["n_datasets_primary", "n_datasets_feature_noise", "n_datasets_training_size"]
    for k in required_keys:
        if k not in existing:
            print(f"[INFO] Manifest missing key '{k}'. Rebuilding.")
            return False

    # Check stage paths
    stage_paths = existing.get("stage_paths", [])
    if not stage_paths:
        print("[INFO] Manifest has no stage_paths. Rebuilding.")
        return False

    present_files = _stage_file_list(session, SYNREG_STAGE_PREFIX)
    missing = [p for p in stage_paths if not any(p.endswith(f) or f.endswith(p) for f in present_files)]
    if missing:
        print(f"[INFO] Manifest stage_paths missing from stage ({len(missing)} missing). Rebuilding.")
        return False

    # Check index table has rows for this suite_id
    try:
        count_row = session.sql(
            f"SELECT COUNT(*) AS n FROM {SYNREG_INDEX_TABLE} "
            f"WHERE suite_id = '{suite_id}'"
        ).collect()
        index_row_count = int(count_row[0][0]) if count_row else 0
        if index_row_count == 0:
            print(
                f"[INFO] Manifest and stage files are valid, but "
                f"{SYNREG_INDEX_TABLE} has 0 rows for suite_id={suite_id}. "
                "Forcing rebuild to repopulate index."
            )
            return False
        print(f"[INFO] Index has {index_row_count} rows for suite_id={suite_id}. Skipping rebuild.")
    except Exception as exc:
        print(f"[INFO] Could not query {SYNREG_INDEX_TABLE}: {exc}. Rebuilding to be safe.")
        return False

    print(f"[INFO] Valid manifest found with {len(stage_paths)} staged files. Skipping regeneration.")
    return True


def write_manifest_to_stage(session, manifest_dict: dict) -> None:
    """Write the manifest JSON to the stage."""
    os.makedirs(SYNREG_LOCAL_DIR, exist_ok=True)
    local_path = os.path.join(SYNREG_LOCAL_DIR, "synthetic_regression_manifest.json")
    with open(local_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    upload_file_to_stage(session, local_path, SYNREG_STAGE_PREFIX)
    print(f"[INFO] Manifest written to {SYNREG_MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Index table DDL + insertion
# ---------------------------------------------------------------------------

def create_synreg_index_table(session) -> None:
    """Create SYNTHETIC_REGRESSION_DATASET_INDEX if not already present."""
    ddl = f"""
    CREATE TRANSIENT TABLE IF NOT EXISTS {SYNREG_INDEX_TABLE} (
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
    print(f"[INFO] Index table {SYNREG_INDEX_TABLE} ready.")


def insert_synreg_index_rows(session, rows: list[dict]) -> None:
    """Insert index rows into SYNTHETIC_REGRESSION_DATASET_INDEX in chunks.
    Uses raw SQL INSERT to avoid Snowpark type-inference issues with ARRAY columns.
    Mirrors prepare_benchmark_datasets._write_benchmark_dataset_index()."""
    if not rows:
        return

    import datetime

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
            f"INSERT INTO {SYNREG_INDEX_TABLE} ({col_list})\n"
            + "\nUNION ALL\n".join(chunk)
        )
        session.sql(sql).collect()
        print(
            f"[INFO] Inserted rows {start}–{start + len(chunk) - 1} "
            f"into {SYNREG_INDEX_TABLE}.",
            flush=True,
        )

    count_row = session.sql(
        f"SELECT COUNT(*) AS n FROM {SYNREG_INDEX_TABLE} "
        f"WHERE suite_id = '{rows[0]['suite_id']}'"
    ).collect()
    actual = int(count_row[0][0]) if count_row else 0
    if actual != len(rows):
        raise RuntimeError(
            f"{SYNREG_INDEX_TABLE} row count {actual} does not match "
            f"expected {len(rows)} after insert."
        )
    print(f"[INFO] Row count validated: {actual} rows in {SYNREG_INDEX_TABLE}.", flush=True)


def _truncate_synreg_index(session) -> None:
    """Remove stale rows for the current suite_id before a rebuild.
    Drops the table only when SYNTHETIC_REGRESSION_DROP_INDEX_TABLE=true is explicitly set,
    preserving rows belonging to other suites (e.g. OOD) in all other cases."""
    if SYNREG_DROP_INDEX_TABLE:
        session.sql(f"DROP TABLE IF EXISTS {SYNREG_INDEX_TABLE}").collect()
        print(f"[INFO] Dropped {SYNREG_INDEX_TABLE} (SYNTHETIC_REGRESSION_DROP_INDEX_TABLE=true).")
    else:
        session.sql(
            f"DELETE FROM {SYNREG_INDEX_TABLE} WHERE suite_id = '{SYNREG_SUITE_ID}'"
        ).collect()
        print(
            f"[INFO] Deleted rows for suite_id='{SYNREG_SUITE_ID}' from {SYNREG_INDEX_TABLE} "
            f"(force_rebuild={SYNREG_FORCE_REBUILD}). Other suite rows preserved."
        )


# ---------------------------------------------------------------------------
# Suite preparation helpers
# ---------------------------------------------------------------------------

def _compute_n_train_default(suite_family: str, n_total: int) -> int:
    if suite_family == "training_size":
        # Training-size suite: holdout is fixed at HOLDOUT_SIZE
        return n_total - HOLDOUT_SIZE
    # Primary / feature_noise / target_noise: 80/20 split
    return int(round(n_total * 0.8))


def _compute_n_holdout_default(suite_family: str, n_total: int) -> int:
    if suite_family == "training_size":
        return HOLDOUT_SIZE
    return n_total - _compute_n_train_default(suite_family, n_total)


def _make_index_row(
    suite_family: str,
    dataset_id: int,
    dataset_seed: int,
    stage_path: str,
    regime: str,
    split_seeds: list[int],
    n_total: int,
    p_signal: int,
    p_noise: int,
    target_noise_scale: float = 1.0,
    noise_level: int = 0,
    is_anchor: bool = False,
    payload_bytes: int = 0,
) -> dict:
    n_train_default = _compute_n_train_default(suite_family, n_total)
    n_holdout_default = _compute_n_holdout_default(suite_family, n_total)
    return {
        "suite_id": SYNREG_SUITE_ID,
        "suite_family": suite_family,
        "dataset_id": dataset_id,
        "dataset_seed": dataset_seed,
        "stage_path": stage_path,
        "prior_name": PRIOR_NAME,
        "prior_version": PRIOR_VERSION,
        "prior_regime": regime,
        "split_seeds": split_seeds,
        "n_total": n_total,
        "n_train_default": n_train_default,
        "n_holdout_default": n_holdout_default,
        "p_signal": p_signal,
        "p_noise": p_noise,
        "p_total": p_signal + p_noise,
        "target_noise_scale": target_noise_scale,
        "training_size_anchor": is_anchor,
        "feature_noise_level": noise_level,
        "eval_weight": 1.0,
        "payload_bytes": payload_bytes,
        "created_at": None,  # Snowflake will fill CURRENT_TIMESTAMP
        "logical_dataset_key": f"{SYNREG_SUITE_ID}:{regime}:{dataset_id:04d}",
        "source_suite_id": None,
    }


def _is_locally_generated_manifest(manifest: dict) -> bool:
    return (
        manifest.get("generated_locally") is True
        and manifest.get("format") == "parquet"
        and isinstance(manifest.get("datasets"), list)
        and len(manifest["datasets"]) > 0
    )


def _download_manifest_json(session) -> dict | None:
    """Download manifest from stage; None on failure."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            session.file.get(SYNREG_MANIFEST_PATH, tmp)
            p = Path(tmp) / "synthetic_regression_manifest.json"
            if p.exists():
                return json.load(open(p))
    except Exception:
        pass
    return None


def _build_index_from_manifest_datasets(session, manifest: dict) -> list[dict]:
    """Construct index rows from manifest.datasets (no NPZ generation needed)."""
    suite_id = manifest["suite_id"]
    rows = []
    for rec in manifest["datasets"]:
        rows.append({
            "suite_id": suite_id,
            "suite_family": rec["suite_family"],
            "dataset_id": rec["dataset_id"],
            "dataset_seed": rec["dataset_seed"],
            "stage_path": rec["stage_path"],
            "prior_name": PRIOR_NAME,
            "prior_version": PRIOR_VERSION,
            "prior_regime": rec["prior_regime"],
            "split_seeds": rec["split_seeds"],
            "n_total": rec["n_total"],
            "n_train_default": rec["n_train_default"],
            "n_holdout_default": rec["n_holdout_default"],
            "p_signal": rec["p_signal"],
            "p_noise": rec["p_noise"],
            "p_total": rec["p_total"],
            "target_noise_scale": rec["target_noise_scale"],
            "training_size_anchor": rec["training_size_anchor"],
            "feature_noise_level": rec["feature_noise_level"],
            "eval_weight": 1.0,
            "payload_bytes": rec.get("payload_bytes", 0),
            "created_at": None,
            "logical_dataset_key": (
                rec.get("logical_dataset_key")
                or f"{suite_id}:{rec['prior_regime']}:{rec['dataset_id']:04d}"
            ),
            "source_suite_id": rec.get("source_suite_id"),
        })
    return rows


def prepare_primary_suite(rng: np.random.Generator, local_dir: str, session) -> list[dict]:
    """
    Generate PRIMARY_N_DATASETS datasets (balanced across 4 regimes).
    Returns index rows for all datasets.
    """
    print(f"[INFO] Preparing primary suite: {PRIMARY_N_DATASETS} datasets × {len(PRIMARY_SPLIT_SEEDS)} seeds")
    index_rows = []
    per_regime = PRIMARY_N_DATASETS // len(REGIMES)
    stage_dir = f"{SYNREG_STAGE_PREFIX}/primary"
    os.makedirs(os.path.join(local_dir, "primary"), exist_ok=True)

    global_idx = 0
    for regime in REGIMES:
        for i in range(per_regime):
            dataset_id = global_idx
            n, p = sample_params_primary(rng)
            ds = generate_synthetic_dataset(
                rng=rng,
                regime=regime,
                suite_family="primary",
                n=n,
                p_signal=p,
                p_noise=0,
            )
            outfile = os.path.join(local_dir, "primary", f"dataset_{dataset_id:04d}.npz")
            serialize_synthetic_npz(outfile, ds, noise_level=0, is_anchor=False)
            payload_bytes = os.path.getsize(outfile)
            stage_path = upload_file_to_stage(session, outfile, stage_dir)
            row = _make_index_row(
                suite_family="primary",
                dataset_id=dataset_id,
                dataset_seed=int(rng.integers(0, 2**31)),
                stage_path=stage_path,
                regime=regime,
                split_seeds=PRIMARY_SPLIT_SEEDS,
                n_total=n,
                p_signal=p,
                p_noise=0,
                noise_level=0,
                is_anchor=False,
                payload_bytes=payload_bytes,
            )
            index_rows.append(row)
            global_idx += 1
            if (dataset_id + 1) % 20 == 0:
                print(f"[INFO]   primary: {dataset_id + 1}/{PRIMARY_N_DATASETS} generated")
            gc.collect()

    print(f"[INFO] Primary suite done: {len(index_rows)} index rows.")
    return index_rows


def prepare_feature_noise_suite(rng: np.random.Generator, local_dir: str, session) -> list[dict]:
    """
    Generate FEATURE_NOISE_N_DATASETS base datasets, pre-baked per noise level.
    Each NPZ contains both signal + noise features pre-embedded.
    Returns index rows.
    """
    print(
        f"[INFO] Preparing feature-noise suite: {FEATURE_NOISE_N_DATASETS} base datasets × "
        f"{len(FEATURE_NOISE_LEVELS)} noise levels"
    )
    index_rows = []
    per_regime = FEATURE_NOISE_N_DATASETS // len(REGIMES)
    os.makedirs(os.path.join(local_dir, "feature_noise"), exist_ok=True)
    stage_dir = f"{SYNREG_STAGE_PREFIX}/feature_noise"

    global_idx = 0
    for regime in REGIMES:
        for i in range(per_regime):
            dataset_id = global_idx
            n, p_signal = sample_params_for_signal(rng)

            for noise_level in FEATURE_NOISE_LEVELS:
                p_noise = noise_level  # noise_level is treated as count of noise features
                p_total = p_signal + p_noise

                ds = generate_synthetic_dataset(
                    rng=rng,
                    regime=regime,
                    suite_family="feature_noise",
                    n=n,
                    p_signal=p_signal,
                    p_noise=p_noise,
                )
                filename = f"dataset_{dataset_id:04d}_noise{noise_level:03d}.npz"
                outfile = os.path.join(local_dir, "feature_noise", filename)
                serialize_synthetic_npz(outfile, ds, noise_level=noise_level, is_anchor=False)
                payload_bytes = os.path.getsize(outfile)
                stage_path = upload_file_to_stage(session, outfile, stage_dir)

                row = _make_index_row(
                    suite_family="feature_noise",
                    dataset_id=dataset_id,
                    dataset_seed=int(rng.integers(0, 2**31)),
                    stage_path=stage_path,
                    regime=regime,
                    split_seeds=FEATURE_NOISE_SEEDS,
                    n_total=n,
                    p_signal=p_signal,
                    p_noise=p_noise,
                    noise_level=noise_level,
                    is_anchor=False,
                    payload_bytes=payload_bytes,
                )
                index_rows.append(row)

            global_idx += 1
            if (dataset_id + 1) % 10 == 0:
                print(f"[INFO]   feature_noise: {dataset_id + 1}/{FEATURE_NOISE_N_DATASETS} base datasets")
            gc.collect()

    print(f"[INFO] Feature-noise suite done: {len(index_rows)} index rows.")
    return index_rows


def prepare_training_size_suite(rng: np.random.Generator, local_dir: str, session) -> list[dict]:
    """
    Generate TRAIN_SIZE_N_DATASETS large datasets (n_total = TRAIN_SIZE_ANCHOR_N + HOLDOUT_SIZE).
    One NPZ per dataset; multiple n_train_grid values evaluated from same file at eval time.
    """
    n_required = TRAIN_SIZE_ANCHOR_N + HOLDOUT_SIZE  # 4832 + 1371 = 6203
    print(
        f"[INFO] Preparing training-size suite: {TRAIN_SIZE_N_DATASETS} datasets, "
        f"n_total={n_required}, grid={TRAIN_SIZE_GRID}"
    )
    index_rows = []
    per_regime = TRAIN_SIZE_N_DATASETS // len(REGIMES)
    os.makedirs(os.path.join(local_dir, "training_size"), exist_ok=True)
    stage_dir = f"{SYNREG_STAGE_PREFIX}/training_size"

    global_idx = 0
    for regime in REGIMES:
        for i in range(per_regime):
            dataset_id = global_idx
            n, p = sample_params_training_size(rng, n_required=n_required)

            ds = generate_synthetic_dataset(
                rng=rng,
                regime=regime,
                suite_family="training_size",
                n=n,
                p_signal=p,
                p_noise=0,
            )
            filename = f"dataset_{dataset_id:04d}.npz"
            outfile = os.path.join(local_dir, "training_size", filename)
            is_anchor = True  # this whole suite has an anchor entry at n_train=TRAIN_SIZE_ANCHOR_N
            serialize_synthetic_npz(outfile, ds, noise_level=0, is_anchor=is_anchor)
            payload_bytes = os.path.getsize(outfile)
            stage_path = upload_file_to_stage(session, outfile, stage_dir)

            row = _make_index_row(
                suite_family="training_size",
                dataset_id=dataset_id,
                dataset_seed=int(rng.integers(0, 2**31)),
                stage_path=stage_path,
                regime=regime,
                split_seeds=TRAIN_SIZE_SEEDS,
                n_total=n,
                p_signal=p,
                p_noise=0,
                noise_level=0,
                is_anchor=True,
                payload_bytes=payload_bytes,
            )
            index_rows.append(row)
            global_idx += 1
            if (dataset_id + 1) % 5 == 0:
                print(f"[INFO]   training_size: {dataset_id + 1}/{TRAIN_SIZE_N_DATASETS} generated")
            gc.collect()

    print(f"[INFO] Training-size suite done: {len(index_rows)} index rows.")
    return index_rows


def prepare_target_noise_suite(rng: np.random.Generator, local_dir: str, session) -> list[dict]:
    """
    Optional target-noise suite.
    Generate TARGET_NOISE_N_DATASETS × len(TARGET_NOISE_SCALES) datasets.
    """
    if not ENABLE_TARGET_NOISE_SUITE:
        print("[INFO] Target-noise suite disabled (SYNTHETIC_REGRESSION_ENABLE_TARGET_NOISE_SUITE=false).")
        return []

    print(
        f"[INFO] Preparing target-noise suite: {TARGET_NOISE_N_DATASETS} datasets × "
        f"{len(TARGET_NOISE_SCALES)} noise scales"
    )
    index_rows = []
    per_regime = TARGET_NOISE_N_DATASETS // len(REGIMES)
    os.makedirs(os.path.join(local_dir, "target_noise"), exist_ok=True)
    stage_dir = f"{SYNREG_STAGE_PREFIX}/target_noise"

    global_idx = 0
    for regime in REGIMES:
        for i in range(per_regime):
            dataset_id = global_idx
            n, p = sample_params_primary(rng)

            for scale in TARGET_NOISE_SCALES:
                ds = generate_synthetic_dataset(
                    rng=rng,
                    regime=regime,
                    suite_family="target_noise",
                    n=n,
                    p_signal=p,
                    p_noise=0,
                    target_noise_scale=scale,
                )
                scale_str = f"{scale:.2f}".replace(".", "p")
                filename = f"dataset_{dataset_id:04d}_scale{scale_str}.npz"
                outfile = os.path.join(local_dir, "target_noise", filename)
                serialize_synthetic_npz(outfile, ds, noise_level=0, is_anchor=False)
                payload_bytes = os.path.getsize(outfile)
                stage_path = upload_file_to_stage(session, outfile, stage_dir)

                row = _make_index_row(
                    suite_family="target_noise",
                    dataset_id=dataset_id,
                    dataset_seed=int(rng.integers(0, 2**31)),
                    stage_path=stage_path,
                    regime=regime,
                    split_seeds=TARGET_NOISE_SEEDS,
                    n_total=n,
                    p_signal=p,
                    p_noise=0,
                    target_noise_scale=scale,
                    noise_level=0,
                    is_anchor=False,
                    payload_bytes=payload_bytes,
                )
                index_rows.append(row)

            global_idx += 1
            gc.collect()

    print(f"[INFO] Target-noise suite done: {len(index_rows)} index rows.")
    return index_rows


# ---------------------------------------------------------------------------
# Idempotency + main orchestration
# ---------------------------------------------------------------------------

def _build_manifest(index_rows: list[dict]) -> dict:
    """Construct manifest dict from index rows."""
    stage_paths = [r["stage_path"] for r in index_rows]
    n_primary = sum(1 for r in index_rows if r["suite_family"] == "primary")
    n_feature_noise = sum(1 for r in index_rows if r["suite_family"] == "feature_noise")
    n_training_size = sum(1 for r in index_rows if r["suite_family"] == "training_size")
    n_target_noise = sum(1 for r in index_rows if r["suite_family"] == "target_noise")
    return {
        "suite_id": SYNREG_SUITE_ID,
        "base_seed": SYNREG_BASE_SEED,
        "n_datasets_primary": n_primary,
        "n_datasets_feature_noise": n_feature_noise,
        "n_datasets_training_size": n_training_size,
        "n_datasets_target_noise": n_target_noise,
        "stage_paths": stage_paths,
        "regimes": REGIMES,
        "prior_name": PRIOR_NAME,
        "prior_version": PRIOR_VERSION,
        "primary_split_seeds": PRIMARY_SPLIT_SEEDS,
        "feature_noise_levels": FEATURE_NOISE_LEVELS,
        "train_size_grid": TRAIN_SIZE_GRID,
        "holdout_size": HOLDOUT_SIZE,
        "train_size_anchor_n": TRAIN_SIZE_ANCHOR_N,
        "target_noise_scales": TARGET_NOISE_SCALES,
        "target_noise_enabled": ENABLE_TARGET_NOISE_SUITE,
    }


def _refresh_index_from_manifest(session, manifest: dict) -> None:
    """Rebuild the index table from a valid existing manifest (idempotent refresh)."""
    # In idempotent mode, we just ensure the table exists and rows from this suite_id are present.
    create_synreg_index_table(session)
    print(
        f"[INFO] Index refresh skipped (manifest valid). "
        f"Existing {SYNREG_INDEX_TABLE} rows retained."
    )


def _assert_index_populated(session, suite_id: str) -> int:
    """Raise RuntimeError if index has 0 rows after prep."""
    try:
        count_row = session.sql(
            f"SELECT COUNT(*) AS n FROM {SYNREG_INDEX_TABLE} "
            f"WHERE suite_id = '{suite_id}'"
        ).collect()
        n = int(count_row[0][0]) if count_row else 0
    except Exception as exc:
        raise RuntimeError(
            f"[FATAL] Could not verify {SYNREG_INDEX_TABLE} row count: {exc}"
        ) from exc
    if n == 0:
        raise RuntimeError(
            f"[FATAL] {SYNREG_INDEX_TABLE} has 0 rows for suite_id={suite_id} "
            "after prepare_synthetic_regression completed. "
            "insert_synreg_index_rows() likely failed silently. "
            "Check the container logs and retry with SYNTHETIC_REGRESSION_FORCE_REBUILD=true."
        )
    print(f"[INFO] Index validation passed: {n} rows for suite_id={suite_id}.")
    return n


def _validate_parquet_index(session, manifest: dict, suite_id: str) -> bool:
    """Validate that the index table is consistent with the parquet manifest.

    Checks performed:
    1. Row count matches len(manifest["datasets"])
    2. No duplicate logical keys: (suite_id, suite_family, prior_regime, dataset_id,
       feature_noise_level, target_noise_scale)
    3. Every stage_path in the manifest exists in the index
    4. Spot-checks key metadata fields from index rows against manifest rows
    Returns True if all checks pass (skip rebuild), False if any check fails (trigger rebuild).
    """
    datasets = manifest.get("datasets", [])
    expected_count = len(datasets)

    # Check 1: row count
    try:
        count_row = session.sql(
            f"SELECT COUNT(*) AS n FROM {SYNREG_INDEX_TABLE} "
            f"WHERE suite_id = '{suite_id}'"
        ).collect()
        actual_count = int(count_row[0][0]) if count_row else 0
    except Exception:
        return False

    if actual_count != expected_count:
        print(
            f"[INFO] Parquet index validation failed: count {actual_count} != "
            f"expected {expected_count}. Triggering rebuild."
        )
        return False

    # Fetch all index rows for further checks
    try:
        index_rows = session.sql(
            f"SELECT * FROM {SYNREG_INDEX_TABLE} WHERE suite_id = '{suite_id}'"
        ).collect()
        index_dicts = [r.as_dict() for r in index_rows]
    except Exception:
        return False

    # Check 2: no duplicate logical keys
    logical_keys = [
        (
            str(r.get("suite_id")),
            str(r.get("suite_family")),
            str(r.get("prior_regime")),
            str(r.get("dataset_id")),
            str(r.get("feature_noise_level")),
            str(r.get("target_noise_scale")),
        )
        for r in index_dicts
    ]
    if len(logical_keys) != len(set(logical_keys)):
        print("[INFO] Parquet index validation failed: duplicate logical keys found. Triggering rebuild.")
        return False

    # Check 3: every stage_path in manifest exists in index
    index_stage_paths = {str(r.get("stage_path")) for r in index_dicts}
    for rec in datasets:
        if str(rec.get("stage_path")) not in index_stage_paths:
            print(
                f"[INFO] Parquet index validation failed: stage_path "
                f"'{rec.get('stage_path')}' missing from index. Triggering rebuild."
            )
            return False

    # Check 4: spot-check key metadata fields
    index_by_path = {str(r.get("stage_path")): r for r in index_dicts}
    spot_fields = [
        "suite_id", "suite_family", "dataset_id", "prior_regime",
        "n_total", "p_signal", "p_noise", "p_total",
        "feature_noise_level", "target_noise_scale",
    ]
    for rec in datasets:
        path = str(rec.get("stage_path"))
        if path not in index_by_path:
            continue
        idx_row = index_by_path[path]
        for field in spot_fields:
            manifest_val = rec.get(field)
            index_val = idx_row.get(field)
            if manifest_val is None or index_val is None:
                continue
            try:
                if str(manifest_val) != str(index_val):
                    print(
                        f"[INFO] Parquet index validation failed: field '{field}' mismatch "
                        f"for {path}: manifest={manifest_val!r} index={index_val!r}. "
                        "Triggering rebuild."
                    )
                    return False
            except Exception:
                pass

    print(
        f"[INFO] Parquet index validation passed: {actual_count} rows, "
        "no duplicates, all stage_paths present, metadata spot-check ok."
    )
    return True


def _load_index_rows(session, suite_id: str, suite_family: str | None = None) -> list[dict]:
    """Load index rows for a given suite_id as plain lowercase-keyed dicts.

    Parameters
    ----------
    session       : Snowpark session
    suite_id      : Filter rows to this suite
    suite_family  : If given, also filter by suite_family (e.g. "primary")
    """
    family_clause = f" AND suite_family = '{suite_family}'" if suite_family else ""
    sql = (
        f"SELECT suite_id, suite_family, dataset_id, dataset_seed, stage_path, "
        f"prior_name, prior_version, prior_regime, split_seeds, "
        f"n_total, n_train_default, n_holdout_default, "
        f"p_signal, p_noise, p_total, target_noise_scale, "
        f"training_size_anchor, feature_noise_level, eval_weight, payload_bytes, "
        f"logical_dataset_key "
        f"FROM {SYNREG_INDEX_TABLE} "
        f"WHERE suite_id = '{suite_id}'{family_clause}"
    )
    rows = session.sql(sql).collect()
    result = []
    for r in rows:
        d = dict(r.as_dict()) if hasattr(r, "as_dict") else dict(r)
        d = {k.lower(): v for k, v in d.items()}
        # Normalize split_seeds: Snowflake ARRAY may come back as JSON string or list
        ss = d.get("split_seeds")
        if isinstance(ss, str):
            d["split_seeds"] = json.loads(ss)
        elif ss is None:
            d["split_seeds"] = []
        result.append(d)
    return result


def prepare_combined_suite(session) -> str:
    """Build the linear_all_v1 combined suite by index-level composition.

    Copies index rows from:
      - COMBINED_PRIMARY_SUITE (regimes A/B/C/D, suite_family='primary', 200 rows)
      - COMBINED_OOD_SUITE     (regimes E/F/G/H, all families,           200 rows)

    Each copied row is remapped:
      - suite_id           → COMBINED_SUITE_ID ('linear_all_v1')
      - source_suite_id    → original suite_id (lineage field)
      - split_seeds        → COMBINED_SPLIT_SEEDS ([0, 1, 2])
      - logical_dataset_key → '{COMBINED_SUITE_ID}:{regime}:{dataset_id:04d}'

    Validates exactly 400 total rows and 50 per regime A-H.
    No parquet files are merged or rewritten.
    """
    from collections import Counter

    create_synreg_index_table(session)

    session.sql(
        f"DELETE FROM {SYNREG_INDEX_TABLE} WHERE suite_id = '{COMBINED_SUITE_ID}'"
    ).collect()
    print(f"[INFO] Cleared existing rows for suite_id='{COMBINED_SUITE_ID}'.")

    print(f"[INFO] Loading primary rows from '{COMBINED_PRIMARY_SUITE}' (family=primary) …")
    primary_rows = _load_index_rows(session, COMBINED_PRIMARY_SUITE, suite_family="primary")

    print(f"[INFO] Loading OOD rows from '{COMBINED_OOD_SUITE}' …")
    ood_rows = _load_index_rows(session, COMBINED_OOD_SUITE)

    if not primary_rows:
        raise RuntimeError(
            f"No rows found for primary suite '{COMBINED_PRIMARY_SUITE}' "
            f"(suite_family='primary'). Run prepare_synthetic_regression first."
        )
    if not ood_rows:
        raise RuntimeError(
            f"No rows found for OOD suite '{COMBINED_OOD_SUITE}'. "
            f"Run prepare_ood_regression first."
        )

    combined_rows = []
    for src_rows, src_suite_id in [
        (primary_rows, COMBINED_PRIMARY_SUITE),
        (ood_rows, COMBINED_OOD_SUITE),
    ]:
        for r in src_rows:
            row = dict(r)
            dataset_id = row["dataset_id"]
            regime = row["prior_regime"]
            row["suite_id"] = COMBINED_SUITE_ID
            row["source_suite_id"] = src_suite_id
            row["split_seeds"] = list(COMBINED_SPLIT_SEEDS)
            row["logical_dataset_key"] = f"{COMBINED_SUITE_ID}:{regime}:{dataset_id:04d}"
            combined_rows.append(row)

    # Validate per-regime balance before inserting
    regime_counts = Counter(r["prior_regime"] for r in combined_rows)
    for regime in COMBINED_REGIMES:
        count = regime_counts.get(regime, 0)
        if count != COMBINED_N_PER_REGIME:
            raise ValueError(
                f"Combined suite regime '{regime}': expected {COMBINED_N_PER_REGIME} "
                f"datasets, found {count}. "
                f"Ensure both source suites are fully indexed before running combined prep."
            )

    if len(combined_rows) != COMBINED_N_DATASETS:
        raise ValueError(
            f"Combined suite has {len(combined_rows)} rows; expected {COMBINED_N_DATASETS}."
        )

    print(f"[INFO] Inserting {len(combined_rows)} combined rows into '{COMBINED_SUITE_ID}' …")
    insert_synreg_index_rows(session, combined_rows)
    _assert_index_populated(session, COMBINED_SUITE_ID)

    print(f"[INFO] prepare_combined_suite DONE. {len(combined_rows)} rows indexed.")
    return (
        f"prepare_combined_suite: ok suite_id={COMBINED_SUITE_ID} "
        f"total={len(combined_rows)} regimes={COMBINED_REGIMES}"
    )


def prepare_synthetic_regression(session=None) -> str:
    """
    Stored procedure handler and main function.
    Returns status string on completion.
    """
    if session is None:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()

    # Dispatch to combined-suite handler when SYNTHETIC_REGRESSION_SUITE_ID=linear_all_v1
    if SYNREG_SUITE_ID == COMBINED_SUITE_ID:
        return prepare_combined_suite(session)

    os.makedirs(SYNREG_LOCAL_DIR, exist_ok=True)

    print(
        f"[INFO] prepare_synthetic_regression START "
        f"suite_id={SYNREG_SUITE_ID} base_seed={SYNREG_BASE_SEED} "
        f"force_rebuild={SYNREG_FORCE_REBUILD}"
    )

    try:
        # --- Parquet workflow (locally-generated manifest) ---
        manifest = _download_manifest_json(session)
        if manifest is not None and _is_locally_generated_manifest(manifest):
            print("[INFO] Locally-generated parquet manifest detected.")
            if not SYNREG_FORCE_REBUILD:
                if _validate_parquet_index(session, manifest, SYNREG_SUITE_ID):
                    _assert_index_populated(session, SYNREG_SUITE_ID)
                    try:
                        count_row = session.sql(
                            f"SELECT COUNT(*) AS n FROM {SYNREG_INDEX_TABLE} "
                            f"WHERE suite_id = '{SYNREG_SUITE_ID}'"
                        ).collect()
                        n = int(count_row[0][0]) if count_row else 0
                    except Exception:
                        n = len(manifest.get("datasets", []))
                    return f"SKIPPED (parquet manifest, {n} rows)"
            _truncate_synreg_index(session)
            create_synreg_index_table(session)
            index_rows = _build_index_from_manifest_datasets(session, manifest)
            insert_synreg_index_rows(session, index_rows)
            _assert_index_populated(session, SYNREG_SUITE_ID)
            return f"prepare_synthetic_regression: ok (parquet, {len(index_rows)} rows)"

        # --- Existing NPZ workflow (unchanged below) ---
        # --- Idempotency check ---
        if not SYNREG_FORCE_REBUILD:
            is_valid = create_or_validate_manifest(session, SYNREG_SUITE_ID)
            if is_valid:
                _refresh_index_from_manifest(session, {})
                _assert_index_populated(session, SYNREG_SUITE_ID)
                return f"SKIPPED (valid manifest for suite_id={SYNREG_SUITE_ID})"

        # --- Full generation ---
        rng = np.random.default_rng(seed=SYNREG_BASE_SEED)
        all_index_rows: list[dict] = []

        _truncate_synreg_index(session)
        create_synreg_index_table(session)

        all_index_rows += prepare_primary_suite(rng, SYNREG_LOCAL_DIR, session)
        all_index_rows += prepare_feature_noise_suite(rng, SYNREG_LOCAL_DIR, session)
        all_index_rows += prepare_training_size_suite(rng, SYNREG_LOCAL_DIR, session)
        all_index_rows += prepare_target_noise_suite(rng, SYNREG_LOCAL_DIR, session)

        # --- Write manifest ---
        manifest = _build_manifest(all_index_rows)
        write_manifest_to_stage(session, manifest)

        # --- Insert index rows ---
        insert_synreg_index_rows(session, all_index_rows)

        total = len(all_index_rows)
        print(f"[INFO] prepare_synthetic_regression DONE. Total index rows: {total}")
        _assert_index_populated(session, SYNREG_SUITE_ID)
        return f"OK suite_id={SYNREG_SUITE_ID} total_rows={total}"

    except Exception as exc:
        import traceback as _tb
        import datetime
        payload = {
            "status": "failed",
            "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": _tb.format_exc(),
            "suite_id": SYNREG_SUITE_ID,
        }
        local_path = os.path.join(SYNREG_LOCAL_DIR, "synreg_prep_failure.json")
        try:
            with open(local_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[PREP FAILURE JSON]\n{json.dumps(payload, indent=2)}", flush=True)
            session.file.put(
                local_path,
                "@EVALUATION_RESULTS_STAGE",
                auto_compress=False,
                overwrite=True,
            )
            print("[INFO] Prep failure JSON uploaded to @EVALUATION_RESULTS_STAGE.", flush=True)
        except Exception as upload_exc:
            print(f"[WARN] Could not upload prep failure JSON: {upload_exc}", flush=True)
        raise


def main() -> None:
    result = prepare_synthetic_regression(session=None)
    print(f"[RESULT] {result}")


if __name__ == "__main__":
    main()
