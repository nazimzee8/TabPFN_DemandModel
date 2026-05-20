"""
prepare_ood_regression.py
=========================
MLJob entrypoint that indexes OOD datasets into
SYNTHETIC_REGRESSION_DATASET_INDEX.

Reads the manifest from @EVALUATION_DATASET_STAGE/ood_parity/ood_manifest.json,
selects the first OOD_N_DATASETS // 4 datasets per regime (deterministic), and
inserts rows into the shared SYNTHETIC_REGRESSION_DATASET_INDEX table, isolated
by suite_id = OOD_SUITE_ID.

Safety invariants
-----------------
* Shared table SYNTHETIC_REGRESSION_DATASET_INDEX is NEVER DROPped.
  _truncate_ood_index() uses DELETE WHERE suite_id = OOD_SUITE_ID only.
* Production suite rows (suite_id != OOD_SUITE_ID) are never touched.
* OOD data lives on @EVALUATION_DATASET_STAGE; @META_DATASET_STAGE is untouched.
* create_synreg_index_table() is always called before _truncate_ood_index()
  to ensure the table exists before any DELETE.

Environment variables
---------------------
OOD_REGRESSION_SUITE_ID    override suite_id (default: ood_linear_pilot_v1)
OOD_REGRESSION_N_DATASETS  number of datasets to index (preferred; default: 80)
OOD_REGRESSION_N_PILOT     legacy fallback for OOD_REGRESSION_N_DATASETS (default: 80)
OOD_REGRESSION_LOCAL_DIR   local scratch dir (default: /tmp/ood_reg_prep)

Dataset counts
--------------
Source pool:    200 staged parquet files (50 per regime E/F/G/H)
Pilot indexed:   80 (20/regime) under ood_linear_pilot_v1
Full suite:     200 (50/regime) under ood_linear_full_v1
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: supports both Snowflake flat staging and local dev nested layout.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))          # Snowflake: same staging dir as prepare_synthetic_regression.py
sys.path.insert(0, str(_HERE.parent))   # local dev: scripts/

from prepare_synthetic_regression import (
    insert_synreg_index_rows,
    create_synreg_index_table,
    _assert_index_populated,
    _compute_n_train_default,
    _compute_n_holdout_default,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OOD_SUITE_ID = os.getenv("OOD_REGRESSION_SUITE_ID", "ood_linear_pilot_v1")
OOD_BASE_SEED = 20260513
OOD_PRIOR_NAME = "ood_linear"
OOD_PRIOR_VERSION = "v1"
OOD_REGIMES = ["E", "F", "G", "H"]
# OOD_REGRESSION_N_DATASETS is the preferred env var.
# OOD_REGRESSION_N_PILOT is kept as a backward-compatible fallback.
OOD_N_DATASETS = int(
    os.getenv("OOD_REGRESSION_N_DATASETS")
    or os.getenv("OOD_REGRESSION_N_PILOT")
    or "80"
)
if OOD_N_DATASETS <= 0:
    raise ValueError(
        f"OOD_N_DATASETS must be positive, got {OOD_N_DATASETS}. "
        "Set OOD_REGRESSION_N_DATASETS (or legacy OOD_REGRESSION_N_PILOT) to a positive "
        "integer divisible by 4 (one per regime)."
    )
if OOD_N_DATASETS % 4 != 0:
    raise ValueError(
        f"OOD_N_DATASETS={OOD_N_DATASETS} is not divisible by 4 (one per OOD regime). "
        "Set OOD_REGRESSION_N_DATASETS to a multiple of 4 (e.g. 80 for pilot, 200 for full suite)."
    )
OOD_SPLIT_SEEDS = [0, 1, 2]
OOD_INDEX_TABLE = "SYNTHETIC_REGRESSION_DATASET_INDEX"
EVAL_STAGE_PREFIX = "@EVALUATION_DATASET_STAGE/ood_parity"
OOD_MANIFEST_PATH = f"{EVAL_STAGE_PREFIX}/ood_manifest.json"
OOD_LOCAL_DIR = os.getenv("OOD_REGRESSION_LOCAL_DIR", "/tmp/ood_reg_prep")

# ---------------------------------------------------------------------------
# OOD-specific helpers (never touches production suite rows)
# ---------------------------------------------------------------------------


def _truncate_ood_index(session) -> None:
    """Delete OOD suite rows from SYNTHETIC_REGRESSION_DATASET_INDEX.

    Uses DELETE WHERE suite_id = OOD_SUITE_ID only — never DROP TABLE.
    """
    sql = (
        f"DELETE FROM {OOD_INDEX_TABLE} "
        f"WHERE suite_id = '{OOD_SUITE_ID}'"
    )
    session.sql(sql).collect()
    print(
        f"[INFO] Truncated OOD rows (suite_id={OOD_SUITE_ID}) "
        f"from {OOD_INDEX_TABLE}.",
        flush=True,
    )


def _make_ood_index_row(
    dataset_id: int,
    dataset_seed: int,
    regime: str,
    n_total: int,
    p_signal: int,
    p_noise: int,
    p_total: int,
    target_noise_scale: float,
) -> dict:
    """Build a single index row for the OOD pilot suite."""
    n_train = _compute_n_train_default("ood_primary", n_total)
    n_holdout = _compute_n_holdout_default("ood_primary", n_total)
    stage_path = (
        f"{EVAL_STAGE_PREFIX}/{regime}/dataset_{dataset_id:04d}.parquet"
    )
    return {
        "suite_id": OOD_SUITE_ID,
        "suite_family": "ood_primary",
        "dataset_id": dataset_id,
        "dataset_seed": dataset_seed,
        "stage_path": stage_path,
        "prior_name": OOD_PRIOR_NAME,
        "prior_version": OOD_PRIOR_VERSION,
        "prior_regime": regime,
        "split_seeds": OOD_SPLIT_SEEDS,
        "n_total": n_total,
        "n_train_default": n_train,
        "n_holdout_default": n_holdout,
        "p_signal": p_signal,
        "p_noise": p_noise,
        "p_total": p_total,
        "target_noise_scale": target_noise_scale,
        "training_size_anchor": False,
        "feature_noise_level": 0,
        "eval_weight": 1.0,
        "payload_bytes": None,
        "logical_dataset_key": f"{OOD_SUITE_ID}:{regime}:{dataset_id:04d}",
    }


# ---------------------------------------------------------------------------
# Manifest download
# ---------------------------------------------------------------------------


def _download_manifest(session) -> dict:
    """Download ood_manifest.json from @EVALUATION_DATASET_STAGE and parse it."""
    local_dir = OOD_LOCAL_DIR
    os.makedirs(local_dir, exist_ok=True)
    session.file.get(OOD_MANIFEST_PATH, local_dir)
    manifest_local = os.path.join(local_dir, "ood_manifest.json")
    with open(manifest_local, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    print(
        f"[INFO] Downloaded manifest: {manifest.get('n_datasets')} datasets, "
        f"regimes={manifest.get('regimes')}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def prepare_ood_regression(session=None) -> str:
    """Index the OOD pilot subset into SYNTHETIC_REGRESSION_DATASET_INDEX.

    Steps
    -----
    1. Download ood_manifest.json from @EVALUATION_DATASET_STAGE.
    2. Select first OOD_N_DATASETS // 4 datasets per regime (deterministic).
    3. CREATE TABLE IF NOT EXISTS (ensure table exists before any DELETE).
    4. DELETE existing OOD suite rows (suite_id-scoped only).
    5. Build and insert index rows.
    6. Assert index row count equals OOD_N_DATASETS.
    7. Return status string.

    Parameters
    ----------
    session : snowflake.snowpark.Session, optional
        If None, a session is obtained via Session.builder.getOrCreate().

    Returns
    -------
    str
        Status message.
    """
    if session is None:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()

    # Step 1: Download manifest
    manifest = _download_manifest(session)

    # Step 2: Select subset — first n_per_regime datasets per regime
    n_per_regime = OOD_N_DATASETS // len(OOD_REGIMES)
    selected_datasets: list[dict] = []
    for regime in OOD_REGIMES:
        regime_entries = [
            d for d in manifest["datasets"] if d["regime"] == regime
        ]
        if len(regime_entries) < n_per_regime:
            raise ValueError(
                f"Regime {regime!r}: manifest has {len(regime_entries)} entries, "
                f"need {n_per_regime} for OOD_N_DATASETS={OOD_N_DATASETS}"
            )
        # Deterministic: take first n_per_regime by order in manifest
        selected_datasets.extend(regime_entries[:n_per_regime])

    print(
        f"[INFO] Selected {len(selected_datasets)} datasets "
        f"({n_per_regime} per regime × {len(OOD_REGIMES)} regimes).",
        flush=True,
    )

    # Step 3: CREATE TABLE IF NOT EXISTS (ensure table exists before DELETE)
    create_synreg_index_table(session)

    # Step 4: Truncate existing OOD rows (DELETE WHERE suite_id only)
    _truncate_ood_index(session)

    # Step 5: Build index rows and insert
    rows = [
        _make_ood_index_row(
            dataset_id=entry["dataset_id"],
            dataset_seed=entry["dataset_seed"],
            regime=entry["regime"],
            n_total=entry["n_total"],
            p_signal=entry["p_signal"],
            p_noise=entry["p_noise"],
            p_total=entry["p_total"],
            target_noise_scale=entry["target_noise_scale"],
        )
        for entry in selected_datasets
    ]
    insert_synreg_index_rows(session, rows)

    # Step 6: Assert index row count equals OOD_N_DATASETS
    actual = _assert_index_populated(session, OOD_SUITE_ID)
    if actual != OOD_N_DATASETS:
        raise RuntimeError(
            f"Expected {OOD_N_DATASETS} indexed rows for {OOD_SUITE_ID}, "
            f"got {actual}"
        )

    status = (
        f"OK suite_id={OOD_SUITE_ID} "
        f"n_indexed={len(rows)} "
        f"split_seeds={OOD_SPLIT_SEEDS}"
    )
    print(f"[INFO] prepare_ood_regression complete: {status}", flush=True)
    return status


# ---------------------------------------------------------------------------
# Script entry point (MLJob runs this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = prepare_ood_regression()
    print(f"[RESULT] {result}")
