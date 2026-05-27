"""autogluon_ray.py

Distributed AutoGluon evaluation entrypoint for Snowflake MLJob multi-instance clusters.

Strategy: Ray work-item distribution (SYNREG_AUTOGLUON_DISTRIBUTED_MODE=ray_work_items)
  - Snowflake submits N logical cluster shards, each as a Snowflake MLJob with
    target_instances=SYNREG_AUTOGLUON_WORKERS_PER_SHARD.
  - This file is the entrypoint for each MLJob. It starts Ray, then the driver process
    assigns itself one shard index (SYNTHETIC_REGRESSION_SHARD_INDEX) and distributes
    its work items across Ray tasks.
  - Each Ray task receives a compact metadata/item dict plus an explicit
    dataset_access descriptor. The driver does NOT put dataset payloads into the Ray object store. This eliminates driver-side
    object-store memory pressure and reduces per-task object-store pinning. Each worker
    trades extra local I/O for lower Ray object-store and driver memory pressure.
  - Each Ray task runs one bounded local AutoGluon fit.
  - The DRIVER is the only process that writes the output CSV. Worker tasks return
    canonical row dicts; they never query SYNTHETIC_REGRESSION_DATASET_INDEX or write
    stage files. Current worker dataset access mode is scoped_file_url:
    the driver derives scoped URLs and workers use SnowflakeFile without sessions.
  - Fallback to single-node mode is DISABLED to prevent duplicate shard file writes.

Invariants:
  - One output file per MLJob: AutoGluon_shard{i}_of_{N}_detailed.csv
  - SYNREG_AUTOGLUON_DISTRIBUTED_MODE must equal "ray_work_items"
  - If Ray init fails, the entrypoint aborts with a clear RuntimeError before writing CSV.
  - MAX_IN_FLIGHT bounds the number of concurrently executing worker-loaded AutoGluon fits.
"""

import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback

print("[ag_ray] entered Python", flush=True)

# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------

def _require_env(name: str, description: str = "") -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"[ag_ray] Required environment variable {name!r} is not set. {description}"
        )
    return val.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[ag_ray] Environment variable {name!r} must be an integer; got {raw!r}."
        ) from exc


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    if value <= 0:
        raise RuntimeError(
            f"[ag_ray] Environment variable {name!r} must be positive; got {value!r}."
        )
    return value


SUITE_ID = _require_env(
    "SYNTHETIC_REGRESSION_SUITE_ID",
    "Set to 'linear_all_v1' for combined suite.",
)
NUM_SHARDS = _env_int("SYNTHETIC_REGRESSION_NUM_SHARDS", 6)
SHARD_INDEX = _env_int("SYNTHETIC_REGRESSION_SHARD_INDEX", 0)
RESULTS_STAGE = _require_env(
    "SYNREG_RESULTS_STAGE",
    "Example: @EVALUATION_RESULTS_STAGE/regression/linear_all_v1",
)
DISTRIBUTED_MODE = os.getenv("SYNREG_AUTOGLUON_DISTRIBUTED_MODE", "ray_work_items")
CLUSTER_SHARDS = _env_int("SYNREG_AUTOGLUON_CLUSTER_SHARDS", NUM_SHARDS)
WORKERS_PER_SHARD = _env_int("SYNREG_AUTOGLUON_WORKERS_PER_SHARD", 1)
TASK_CPUS = _env_int("AUTOGLUON_TASK_CPUS", 1)
TIME_LIMIT = _env_int("AUTOGLUON_TIME_LIMIT", 300)
PRESETS = os.getenv("AUTOGLUON_PRESETS", "best_quality")
MIN_TMP_FREE_BYTES = _env_int("BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", 5368709120)
MAX_DATASET_BYTES = _env_int("BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES", 2147483648)
LOCAL_CACHE = os.getenv("SYNTHETIC_REGRESSION_LOCAL_CACHE", "/tmp/synreg_cache")
WORKER_DATA_ACCESS_MODE = os.getenv(
    "SYNREG_WORKER_DATA_ACCESS_MODE",
    "driver_presigned_url",
)
MAX_WORK_ITEM_BYTES = _env_int("SYNREG_MAX_WORK_ITEM_BYTES", 8192)
RAY_READY_TIMEOUT_SECONDS = _env_positive_int(
    "SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS",
    600,
)
RAY_READY_POLL_SECONDS = _env_positive_int(
    "SYNREG_RAY_CLUSTER_READY_POLL_SECONDS",
    10,
)
RAY_ADDRESS_MODE = os.getenv("SYNREG_RAY_ADDRESS_MODE", "auto")
RAY_HEAD_ADDRESS = os.getenv("RAY_HEAD_ADDRESS", "auto")
# Cluster identity (Finding 6): set by orchestrator to verify driver joined the right shard's head
SPCS_RAY_RUN_ID = os.getenv("SPCS_RAY_RUN_ID", "")
SPCS_RAY_SHARD_INDEX = os.getenv("SPCS_RAY_SHARD_INDEX", "")

print(
    f"[ag_ray] resolved env: suite_id={SUITE_ID} shard={SHARD_INDEX}/{NUM_SHARDS} "
    f"cluster_shards={CLUSTER_SHARDS} workers_per_shard={WORKERS_PER_SHARD} "
    f"task_cpus={TASK_CPUS} time_limit={TIME_LIMIT} presets={PRESETS!r} "
    f"distributed_mode={DISTRIBUTED_MODE!r} results_stage={RESULTS_STAGE!r} "
    f"worker_data_access_mode={WORKER_DATA_ACCESS_MODE!r} "
    f"max_work_item_bytes={MAX_WORK_ITEM_BYTES} "
    f"ray_ready_timeout_seconds={RAY_READY_TIMEOUT_SECONDS}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if DISTRIBUTED_MODE != "ray_work_items":
    raise RuntimeError(
        f"[ag_ray] SYNREG_AUTOGLUON_DISTRIBUTED_MODE={DISTRIBUTED_MODE!r} but this "
        "entrypoint only implements 'ray_work_items'. "
        "Use evaluate_synthetic_regression.py for single-node mode."
    )

if NUM_SHARDS != CLUSTER_SHARDS:
    raise RuntimeError(
        f"[ag_ray] SYNTHETIC_REGRESSION_NUM_SHARDS={NUM_SHARDS} != "
        f"SYNREG_AUTOGLUON_CLUSTER_SHARDS={CLUSTER_SHARDS}. "
        "These must match for distributed work-item assignment to be consistent."
    )

if not (0 <= SHARD_INDEX < NUM_SHARDS):
    raise RuntimeError(
        f"[ag_ray] SYNTHETIC_REGRESSION_SHARD_INDEX={SHARD_INDEX} out of range "
        f"[0, {NUM_SHARDS}). Check orchestration env vars."
    )

if TASK_CPUS < 1:
    raise RuntimeError(f"[ag_ray] AUTOGLUON_TASK_CPUS={TASK_CPUS} must be >= 1.")

if TIME_LIMIT < 1:
    raise RuntimeError(f"[ag_ray] AUTOGLUON_TIME_LIMIT={TIME_LIMIT} must be >= 1.")

if WORKERS_PER_SHARD < 1:
    raise RuntimeError(
        f"[ag_ray] SYNREG_AUTOGLUON_WORKERS_PER_SHARD={WORKERS_PER_SHARD} must be >= 1."
    )

# ---------------------------------------------------------------------------
# Imports (after env validation to surface env errors first)
# ---------------------------------------------------------------------------

import numpy as np

# Shared helpers from evaluate_synthetic_regression.py (co-located in @MODEL_STAGE/scripts/)
from evaluate_synthetic_regression import (
    create_snowpark_session,
    load_synthetic_regression_index,
    assign_synthetic_regression_shard,
    expand_synreg_work_items,
    build_compact_synreg_work_item,
    compact_work_items_serialized_bytes,
    build_split_for_seed,
    preprocess_train_only,
    compute_regression_metrics,
    write_part_csv_to_stage,
    memory_guard_matrix,
    _base_autogluon_row,
    _empty_row,
    _skipped_row,
    _failed_row,
    _check_tmp_free_bytes,
)
from autogluon_models import (
    get_tabular_predictor_class,
    make_unique_autogluon_model_dir,
    predict_autogluon_timed,
)

print("[ag_ray] imports complete", flush=True)

# ---------------------------------------------------------------------------
# Ray initialisation — fail fast, no single-node fallback
# ---------------------------------------------------------------------------

try:
    import ray
except ImportError as exc:
    raise RuntimeError(
        "[ag_ray] Ray is not installed in this environment. "
        "The AutoGluon distributed work-item entrypoint requires Ray. "
        "Install it or use evaluate_synthetic_regression.py for single-node mode."
    ) from exc


def _wait_for_ray_capacity(*, expected_nodes: int, expected_cpus_min: int) -> tuple[int, int, dict]:
    deadline = time.monotonic() + RAY_READY_TIMEOUT_SECONDS
    last_state: tuple[int, int] | None = None

    while True:
        cluster_resources = ray.cluster_resources()
        available_cpus = int(cluster_resources.get("CPU", 0))
        live_nodes = [node for node in ray.nodes() if node.get("Alive")]
        state = (len(live_nodes), available_cpus)
        if state != last_state:
            print(
                "[ag_ray] ray readiness: "
                f"live_nodes={len(live_nodes)}/{expected_nodes} "
                f"available_cpus={available_cpus}/{expected_cpus_min} "
                f"cluster_resources={dict(cluster_resources)}",
                flush=True,
            )
            last_state = state

        if len(live_nodes) >= expected_nodes and available_cpus >= expected_cpus_min:
            return len(live_nodes), available_cpus, dict(cluster_resources)

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "[ag_ray] Ray cluster did not reach requested capacity before readiness timeout: "
                f"live_nodes={len(live_nodes)}/{expected_nodes}, "
                f"available_cpus={available_cpus}/{expected_cpus_min}, "
                f"timeout_seconds={RAY_READY_TIMEOUT_SECONDS}, "
                f"cluster_resources={dict(cluster_resources)}. "
                "Snowflake may still be starting target_instances, or account/compute-pool "
                "concurrency may be below the requested Ray capacity envelope."
            )

        time.sleep(RAY_READY_POLL_SECONDS)


# Resolve Ray address based on mode
if RAY_ADDRESS_MODE == "explicit":
    _ray_address = RAY_HEAD_ADDRESS
    if not _ray_address or _ray_address == "auto":
        raise RuntimeError(
            "[ag_ray] SYNREG_RAY_ADDRESS_MODE=explicit requires RAY_HEAD_ADDRESS "
            "to be set to the head node address (e.g. 'host:6379'). "
            "In SPCS, this is the SPCS service DNS name of the Ray head container."
        )
elif RAY_ADDRESS_MODE == "auto":
    _ray_address = "auto"
else:
    raise RuntimeError(
        f"[ag_ray] SYNREG_RAY_ADDRESS_MODE={RAY_ADDRESS_MODE!r} is not supported. "
        "Supported modes: 'auto' (Snowflake MLJob), 'explicit' (SPCS self-managed Ray)."
    )

print(
    f"[ag_ray] ray init starting: address_mode={RAY_ADDRESS_MODE!r} address={_ray_address!r}",
    flush=True,
)
try:
    ray.init(
        address=_ray_address,
        ignore_reinit_error=True,
        log_to_driver=True,
        include_dashboard=False,
    )
    # Finding 5: In SPCS (explicit) mode, the Ray head starts with --num-cpus=0, so the
    # head is a live node that contributes zero CPUs. Expected topology:
    #   live_nodes  = workers_per_shard + 1  (head + workers)
    #   available_cpus = workers_per_shard * TASK_CPUS  (only workers have CPUs)
    # In MLJob (auto) mode, Snowflake manages Ray and the head is not separately counted,
    # so expected_nodes = WORKERS_PER_SHARD and expected_cpus_min = max(TASK_CPUS, WORKERS_PER_SHARD).
    if RAY_ADDRESS_MODE == "explicit":
        expected_nodes = WORKERS_PER_SHARD + 1
        expected_cpus_min = WORKERS_PER_SHARD * TASK_CPUS
    else:
        expected_nodes = WORKERS_PER_SHARD
        expected_cpus_min = max(TASK_CPUS, WORKERS_PER_SHARD)
    live_node_count, available_cpus, cluster_resources = _wait_for_ray_capacity(
        expected_nodes=expected_nodes,
        expected_cpus_min=expected_cpus_min,
    )
    print(
        f"[ag_ray] ray ready  cluster_cpus={available_cpus} "
        f"live_nodes={live_node_count} cluster_resources={dict(cluster_resources)} "
        f"address_mode={RAY_ADDRESS_MODE!r} expected_nodes={expected_nodes} "
        f"expected_cpus_min={expected_cpus_min}",
        flush=True,
    )
    # Finding 6: Cluster identity check — verify this driver connected to its own shard's head.
    # The head injects a custom resource "spcs_cluster_id_<run_id>_<shard_index>": 1 at startup.
    if RAY_ADDRESS_MODE == "explicit" and SPCS_RAY_RUN_ID and SPCS_RAY_SHARD_INDEX != "":
        cluster_id_key = f"spcs_cluster_id_{SPCS_RAY_RUN_ID}_{SPCS_RAY_SHARD_INDEX}"
        if cluster_id_key not in cluster_resources:
            raise RuntimeError(
                f"[ag_ray] Cluster identity check FAILED: expected custom resource "
                f"{cluster_id_key!r} not found in cluster resources {dict(cluster_resources)}. "
                f"Driver (shard={SHARD_INDEX}) may be connected to the wrong Ray head. "
                "Check SPCS_RAY_HEAD_DNS_SUFFIX configuration and confirm each shard's "
                "head container started successfully."
            )
        print(
            f"[ag_ray] cluster identity verified: {cluster_id_key!r} present in cluster resources.",
            flush=True,
        )
except Exception as exc:
    raise RuntimeError(
        "AutoGluon distributed work-item mode requires a Ray-backed cluster. "
        "Ray initialization failed before any output CSV was written. "
        "Falling back to single-node mode is disabled for this entrypoint because it can "
        "create duplicate shard files."
    ) from exc

print(f"[ag_ray] driver owns shard {SHARD_INDEX}/{NUM_SHARDS}", flush=True)
print(
    f"[ag_ray] dataset loading mode={WORKER_DATA_ACCESS_MODE} "
    "driver_metadata_only=true no_driver_ray_put=true",
    flush=True,
)

# ---------------------------------------------------------------------------
# Snowpark session (driver only)
# ---------------------------------------------------------------------------

session = create_snowpark_session()

# ---------------------------------------------------------------------------
# Work-item assignment
# ---------------------------------------------------------------------------

print(f"[ag_ray] loading index suite_id={SUITE_ID!r} …", flush=True)
all_rows_index = load_synthetic_regression_index(suite_family=None, session=session)
all_work_items = expand_synreg_work_items(all_rows_index, train_size_grid=None)
assigned_work_items = assign_synthetic_regression_shard(all_work_items, SHARD_INDEX, NUM_SHARDS)
dataset_url_cache: dict[str, str] = {}
my_work_items = [
    build_compact_synreg_work_item(
        row,
        session=session,
        access_mode=WORKER_DATA_ACCESS_MODE,
        max_item_bytes=MAX_WORK_ITEM_BYTES,
        dataset_url_cache=dataset_url_cache,
    )
    for row in assigned_work_items
]
items_total_bytes, items_max_bytes = compact_work_items_serialized_bytes(my_work_items)

print(
    f"[ag_ray] shard {SHARD_INDEX}/{NUM_SHARDS}: "
    f"{len(all_rows_index)} index rows → {len(all_work_items)} work items → "
    f"{len(my_work_items)} compact items assigned to this shard "
    f"unique_dataset_access_urls={len(dataset_url_cache)} "
    f"serialized_total_bytes={items_total_bytes} "
    f"serialized_max_item_bytes={items_max_bytes}",
    flush=True,
)

# Atomic work-item uniqueness guard: detect duplicates before submitting to Ray.
# Each (suite_id, stage_path, split_seed, n_train_override) must be unique within this shard.
_seen_keys: set[tuple] = set()
for _item in my_work_items:
    _key = (
        _item.get("suite_id"),
        _item.get("stage_path") or _item.get("dataset_id"),
        _item.get("split_seed"),
        _item.get("n_train_override"),
    )
    if _key in _seen_keys:
        raise RuntimeError(
            f"[ag_ray] Duplicate atomic work item detected before Ray submission: "
            f"{_key!r}. This indicates a bug in shard assignment or work-item expansion. "
            f"Aborting to prevent double-evaluation."
        )
    _seen_keys.add(_key)
del _seen_keys

if not my_work_items:
    raise RuntimeError(
        f"[ag_ray] Shard {SHARD_INDEX}/{NUM_SHARDS} has zero work items. "
        f"Check that SYNTHETIC_REGRESSION_DATASET_INDEX has rows for suite_id={SUITE_ID!r}."
    )

# Finding 7: Presigned URL expiry validation.
# Strict by default so late workers do not silently hit expired URLs.
if WORKER_DATA_ACCESS_MODE == "driver_presigned_url":
    _expiry_seconds = int(os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_SECONDS", "86400"))
    # Conservative upper bound: all items run sequentially at max time_limit.
    _estimated_max_shard_seconds = len(my_work_items) * TIME_LIMIT
    _buffer_seconds = int(os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_BUFFER_SECONDS", "3600"))
    _policy = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_POLICY", "strict").strip().lower()
    if _policy not in {"strict", "warn"}:
        raise RuntimeError(
            "[ag_ray] SYNREG_PRESIGNED_URL_EXPIRY_POLICY must be 'strict' or 'warn'; "
            f"got {_policy!r}."
        )
    if _expiry_seconds < _estimated_max_shard_seconds + _buffer_seconds:
        if _policy == "strict":
            raise RuntimeError(
                f"[ag_ray] presigned URL expiry={_expiry_seconds}s may be insufficient: "
                f"estimated_max_shard_runtime={_estimated_max_shard_seconds}s "
                f"for {len(my_work_items)} items at {TIME_LIMIT}s plus buffer={_buffer_seconds}s. "
                "Increase SYNREG_PRESIGNED_URL_EXPIRY_SECONDS or set "
                "SYNREG_PRESIGNED_URL_EXPIRY_POLICY=warn to accept late-worker HTTP 403 risk."
            )
        print(
            f"[ag_ray] WARNING presigned URL expiry={_expiry_seconds}s may be insufficient: "
            f"estimated_max_shard_runtime={_estimated_max_shard_seconds}s "
            f"({len(my_work_items)} items × {TIME_LIMIT}s) + buffer={_buffer_seconds}s. "
            "Consider increasing SYNREG_PRESIGNED_URL_EXPIRY_SECONDS to avoid "
            "HTTP 403 errors on late-running workers.",
            flush=True,
        )
    else:
        print(
            f"[ag_ray] presigned URL expiry ok: expiry={_expiry_seconds}s "
            f">= estimated_max={_estimated_max_shard_seconds}s + buffer={_buffer_seconds}s.",
            flush=True,
        )

# Pre-fetch AutoGluon predictor class to trigger import errors early.
get_tabular_predictor_class()


# ---------------------------------------------------------------------------
# Ray remote task
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=TASK_CPUS)
def _autogluon_work_item(item_meta: dict) -> dict:
    """Execute one AutoGluon fit+predict for a single (dataset, seed, condition) triple.

    Parameters
    ----------
    item_meta : dict
        Work-item metadata (suite_id, dataset_id, split_seed, etc.).
        The worker loads its own dataset from staged metadata — no dataset payload
        is passed through the Ray object store.

    Returns
    -------
    dict
        Canonical output row (status='ok', 'skipped', or 'failed').
    """
    import gc
    import os
    import shutil

    import numpy as np
    from autogluon_models import (
        get_tabular_predictor_class,
        make_unique_autogluon_model_dir,
        predict_autogluon_timed,
    )
    from evaluate_synthetic_regression import (
        load_prepared_synthetic_dataset_from_access,
        build_split_for_seed,
        preprocess_train_only,
        compute_regression_metrics,
        memory_guard_matrix,
        _base_autogluon_row,
        _empty_row,
        _skipped_row,
        _failed_row,
        _check_tmp_free_bytes,
        SYNREG_AG_MIN_TMP_FREE_BYTES,
    )

    suite_id     = item_meta["suite_id"]
    suite_family = item_meta.get("suite_family", "primary")
    dataset_id   = item_meta.get("dataset_id")
    dataset_seed = item_meta.get("dataset_seed")
    prior_regime = item_meta.get("prior_regime")
    split_seed   = int(item_meta["split_seed"])
    n_train_override = item_meta.get("n_train_override")

    task_cpus        = int(os.getenv("AUTOGLUON_TASK_CPUS", "1"))
    time_limit       = int(os.getenv("AUTOGLUON_TIME_LIMIT", "300"))
    presets          = os.getenv("AUTOGLUON_PRESETS", "best_quality")
    max_dataset_bytes = int(os.getenv("BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES", "2147483648"))
    local_cache      = os.getenv("SYNTHETIC_REGRESSION_LOCAL_CACHE", "/tmp/synreg_cache")

    # Partial base row — data-derived fields appended after successful load
    base = _base_autogluon_row(item_meta)

    # Guard: /tmp free space (before loading dataset)
    free_bytes = _check_tmp_free_bytes()
    if free_bytes < SYNREG_AG_MIN_TMP_FREE_BYTES:
        return _skipped_row(
            base,
            f"insufficient_tmp_space (free={free_bytes} < {SYNREG_AG_MIN_TMP_FREE_BYTES})",
        )

    ag_path = None
    data = None
    X_train = y_train = X_holdout = betaX_holdout = y_holdout = None
    X_train_p = X_holdout_p = None
    try:
        data = load_prepared_synthetic_dataset_from_access(item_meta, local_cache)

        # Guard: dataset payload size (computed locally, no ray.put)
        dataset_bytes = sum(int(getattr(v, "nbytes", 0)) for v in data.values())
        if dataset_bytes > max_dataset_bytes:
            return _skipped_row(
                base,
                (
                    "autogluon_dataset_too_large "
                    f"(dataset_bytes={dataset_bytes} > {max_dataset_bytes})"
                ),
            )

        n_total = int(data["n_total"])
        p_signal = int(data.get("p_signal", data["X"].shape[1]))
        p_noise  = int(data.get("p_noise", 0))
        p_total  = int(data.get("p_total", data["X"].shape[1]))
        feature_noise_level = int(data.get("feature_noise_level", 0))
        target_noise_scale  = float(data.get("target_noise_scale", 1.0))
        is_anchor = bool(data.get("training_size_anchor", False))

        base.update({
            "n_total": n_total,
            "n_holdout": int(data.get("n_holdout_default", n_total // 5)),
            "p_signal": p_signal,
            "p_noise": p_noise,
            "p_total": p_total,
            "feature_noise_level": feature_noise_level,
            "target_noise_scale": target_noise_scale,
            "training_size_anchor": bool(item_meta.get("training_size_anchor", is_anchor)),
        })

        split = build_split_for_seed(data, split_seed, n_train_override)
        X_train, y_train = split["X_train"], split["y_train"]
        X_holdout    = split["X_holdout"]
        betaX_holdout = split["betaX_holdout"]
        y_holdout    = split["y_holdout"]
        n_train      = split["n_train"]
        n_holdout    = split["n_holdout"]
        base["n_train"]   = n_train
        base["n_holdout"] = n_holdout

        X_train_p, X_holdout_p = preprocess_train_only(X_train, X_holdout)

        skip_reason = memory_guard_matrix(X_train_p, X_holdout_p)
        if skip_reason:
            return _skipped_row(base, skip_reason)

        get_tabular_predictor_class()

        ag_path = make_unique_autogluon_model_dir(
            {
                **item_meta,
                "shard_index": os.getenv("SYNTHETIC_REGRESSION_SHARD_INDEX", "na"),
            },
            prefix="ag_ray",
        )
        y_pred, fit_time, predict_time = predict_autogluon_timed(
            X_train_p,
            y_train,
            X_holdout_p,
            time_limit=time_limit,
            presets=presets,
            num_cpus=task_cpus,
            num_gpus=0,
            verbosity=0,
            model_dir=ag_path,
            cleanup=True,
        )

        metrics = compute_regression_metrics(y_pred, betaX_holdout, y_observed=y_holdout)

        row_out = _empty_row()
        row_out.update(base)
        row_out.update(metrics)
        row_out.update({
            "status": "ok",
            "fit_time_s": fit_time,
            "predict_time_s": predict_time,
            "total_time_s": fit_time + predict_time,
            "processed_features": X_train_p.shape[1],
            "raw_features": X_train_p.shape[1],
        })
        return row_out

    except Exception as exc:
        return _failed_row(base, type(exc).__name__, str(exc)[:500])

    finally:
        if ag_path is not None:
            try:
                if os.path.isdir(ag_path):
                    shutil.rmtree(ag_path, ignore_errors=True)
            except Exception:
                pass
        # Explicitly release large objects to reduce worker memory pressure
        try: del data
        except NameError: pass
        try: del X_train
        except NameError: pass
        try: del y_train
        except NameError: pass
        try: del X_holdout
        except NameError: pass
        try: del betaX_holdout
        except NameError: pass
        try: del y_holdout
        except NameError: pass
        try: del X_train_p
        except NameError: pass
        try: del X_holdout_p
        except NameError: pass
        gc.collect()


# ---------------------------------------------------------------------------
# Main distributed evaluation loop
# ---------------------------------------------------------------------------

MAX_IN_FLIGHT = _env_int("SYNREG_AUTOGLUON_MAX_IN_FLIGHT", WORKERS_PER_SHARD)
if MAX_IN_FLIGHT < 1:
    raise RuntimeError(
        f"[ag_ray] SYNREG_AUTOGLUON_MAX_IN_FLIGHT={MAX_IN_FLIGHT} must be >= 1."
    )
if MAX_IN_FLIGHT > WORKERS_PER_SHARD:
    raise RuntimeError(
        f"[ag_ray] SYNREG_AUTOGLUON_MAX_IN_FLIGHT={MAX_IN_FLIGHT} exceeds "
        f"SYNREG_AUTOGLUON_WORKERS_PER_SHARD={WORKERS_PER_SHARD}. "
        f"MAX_IN_FLIGHT must not exceed WORKERS_PER_SHARD to avoid scheduling "
        f"starvation (more in-flight futures than available workers). "
        f"Reduce SYNREG_AUTOGLUON_MAX_IN_FLIGHT or increase SYNREG_AUTOGLUON_WORKERS_PER_SHARD."
    )

print(
    f"[ag_ray] starting distributed evaluation: "
    f"{len(my_work_items)} work items "
    f"max_in_flight={MAX_IN_FLIGHT} task_cpus={TASK_CPUS}",
    flush=True,
)

expected_work_items = len(my_work_items)
output_rows: list[dict] = []
pending: list[ray.ObjectRef] = []  # futures only; workers load datasets independently
future_to_item: dict[ray.ObjectRef, dict] = {}
item_iter = iter(my_work_items)
items_exhausted = False
submitted = 0
completed = 0

while True:
    # Refill the pending pool up to MAX_IN_FLIGHT
    while not items_exhausted and len(pending) < MAX_IN_FLIGHT:
        item = next(item_iter, None)
        if item is None:
            items_exhausted = True
            break
        # Workers load their own datasets; driver passes only small item metadata
        future = _autogluon_work_item.remote(item)
        pending.append(future)
        future_to_item[future] = item
        submitted += 1

    if not pending:
        break  # all work done

    # Collect one completed result (blocking)
    done_futures, _ = ray.wait(pending, num_returns=1, timeout=None)
    if done_futures:
        done_future = done_futures[0]
        pending.remove(done_future)
        item_meta = future_to_item.pop(done_future)
        try:
            row = ray.get(done_future)
            output_rows.append(row)
        except Exception as exc:
            # Ray task raised unexpectedly outside the worker's handled path.
            print(f"[ag_ray] unexpected Ray task error; emitting failed row: {exc}", flush=True)
            output_rows.append(
                _failed_row(
                    _base_autogluon_row(item_meta),
                    type(exc).__name__,
                    str(exc)[:500],
                )
            )
        del done_future
        completed += 1
        gc.collect()
        if completed % 10 == 0:
            print(
                f"[ag_ray] progress: submitted={submitted} completed={completed} "
                f"in_flight={len(pending)} rows_so_far={len(output_rows)}",
                flush=True,
            )

print(
    f"[ag_ray] work items done: submitted={submitted} completed={completed} "
    f"output_rows={len(output_rows)}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

if not output_rows:
    raise RuntimeError(
        f"[ag_ray] Shard {SHARD_INDEX}/{NUM_SHARDS} produced zero output rows. "
        "This is fatal — refusing to write an empty CSV to stage. "
        "Check that the combined suite index is populated and all datasets are accessible."
    )

if (
    submitted != expected_work_items
    or completed != expected_work_items
    or len(output_rows) != expected_work_items
):
    raise RuntimeError(
        f"[ag_ray] Incomplete Ray work item accounting for shard {SHARD_INDEX}/{NUM_SHARDS}: "
        f"expected={expected_work_items} submitted={submitted} completed={completed} "
        f"output_rows={len(output_rows)}. Refusing to write a partial shard CSV."
    )

ok_count = sum(1 for r in output_rows if r.get("status") == "ok")
skip_count = sum(1 for r in output_rows if r.get("status") == "skipped")
fail_count = sum(1 for r in output_rows if r.get("status") == "failed")
print(
    f"[ag_ray] row summary: ok={ok_count} skipped={skip_count} failed={fail_count} "
    f"total={len(output_rows)}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Write output — driver only, exactly one file
# ---------------------------------------------------------------------------

# SYNREG_RESULTS_STAGE was captured at module import time from the required
# SYNREG_RESULTS_STAGE env var; write_part_csv_to_stage uses that constant directly.
write_part_csv_to_stage(
    session,
    output_rows,
    "AutoGluon",
    SHARD_INDEX,
    NUM_SHARDS,
)

print(
    f"[ag_ray] shard {SHARD_INDEX}/{NUM_SHARDS} complete: "
    f"{len(output_rows)} rows written to {RESULTS_STAGE}",
    flush=True,
)
