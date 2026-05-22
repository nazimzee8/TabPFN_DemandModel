"""evaluate_synthetic_regression_autogluon_ray.py

Distributed AutoGluon evaluation entrypoint for Snowflake MLJob multi-instance clusters.

Strategy: Ray work-item distribution (SYNREG_AUTOGLUON_DISTRIBUTED_MODE=ray_work_items)
  - Snowflake submits N logical cluster shards, each as a Snowflake MLJob with
    target_instances=SYNREG_AUTOGLUON_WORKERS_PER_SHARD.
  - This file is the entrypoint for each MLJob. It starts Ray, then the driver process
    assigns itself one shard index (SYNTHETIC_REGRESSION_SHARD_INDEX) and distributes
    its work items across Ray tasks.
  - Each Ray task runs one bounded local AutoGluon fit.
  - The DRIVER is the only process that writes the output CSV. Worker tasks return
    canonical row dicts; they never open Snowpark sessions or write stage files.
  - Fallback to single-node mode is DISABLED to prevent duplicate shard file writes.

Invariants:
  - One output file per MLJob: AutoGluon_shard{i}_of_{N}_detailed.csv
  - SYNREG_AUTOGLUON_DISTRIBUTED_MODE must equal "ray_work_items"
  - If Ray init fails, the entrypoint aborts with a clear RuntimeError before writing CSV.
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
CPU_MAX_FEATURES = _env_int("BENCHMARK_CPU_MAX_PROCESSED_FEATURES", 512)
CPU_MAX_MATRIX_BYTES = _env_int("BENCHMARK_CPU_MAX_MATRIX_BYTES", 2147483648)

print(
    f"[ag_ray] resolved env: suite_id={SUITE_ID} shard={SHARD_INDEX}/{NUM_SHARDS} "
    f"cluster_shards={CLUSTER_SHARDS} workers_per_shard={WORKERS_PER_SHARD} "
    f"task_cpus={TASK_CPUS} time_limit={TIME_LIMIT} presets={PRESETS!r} "
    f"distributed_mode={DISTRIBUTED_MODE!r} results_stage={RESULTS_STAGE!r}",
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
    load_synthetic_regression_index,
    assign_synthetic_regression_shard,
    expand_synreg_work_items,
    load_prepared_synthetic_dataset,
    build_split_for_seed,
    preprocess_train_only,
    compute_regression_metrics,
    write_part_csv_to_stage,
    memory_guard_matrix,
    _empty_row,
    _skipped_row,
    _failed_row,
    _check_tmp_free_bytes,
)
from autogluon_models import get_tabular_predictor_class, predict_autogluon_timed

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

print("[ag_ray] ray init starting", flush=True)
try:
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        log_to_driver=True,
        include_dashboard=False,
    )
    cluster_resources = ray.cluster_resources()
    available_cpus = int(cluster_resources.get("CPU", 0))
    live_nodes = [node for node in ray.nodes() if node.get("Alive")]
    print(
        f"[ag_ray] ray init complete  cluster_cpus={available_cpus} "
        f"live_nodes={len(live_nodes)} cluster_resources={dict(cluster_resources)}",
        flush=True,
    )
    if len(live_nodes) < WORKERS_PER_SHARD:
        raise RuntimeError(
            f"[ag_ray] Ray cluster has only {len(live_nodes)} live nodes but "
            f"SYNREG_AUTOGLUON_WORKERS_PER_SHARD={WORKERS_PER_SHARD}. "
            "Snowflake did not attach the requested target_instances to this MLJob."
        )
    if available_cpus < TASK_CPUS:
        raise RuntimeError(
            f"[ag_ray] Ray cluster has only {available_cpus} CPUs but "
            f"AUTOGLUON_TASK_CPUS={TASK_CPUS} per task. "
            "The Snowflake MLJob multi-instance environment did not provide enough CPU resources."
        )
except Exception as exc:
    raise RuntimeError(
        "AutoGluon distributed work-item mode requires a Ray-backed Snowflake MLJob cluster. "
        "Ray initialization failed before any output CSV was written. "
        "Falling back to single-node mode is disabled for this entrypoint because it can "
        "create duplicate shard files."
    ) from exc

print(f"[ag_ray] driver owns shard {SHARD_INDEX}/{NUM_SHARDS}", flush=True)

# ---------------------------------------------------------------------------
# Snowpark session (driver only)
# ---------------------------------------------------------------------------

from snowflake.snowpark import Session

session = Session.builder.getOrCreate()

# ---------------------------------------------------------------------------
# Work-item assignment
# ---------------------------------------------------------------------------

print(f"[ag_ray] loading index suite_id={SUITE_ID!r} …", flush=True)
all_rows_index = load_synthetic_regression_index(suite_family=None, session=session)
all_work_items = expand_synreg_work_items(all_rows_index, train_size_grid=None)
my_work_items = assign_synthetic_regression_shard(all_work_items, SHARD_INDEX, NUM_SHARDS)

print(
    f"[ag_ray] shard {SHARD_INDEX}/{NUM_SHARDS}: "
    f"{len(all_rows_index)} index rows → {len(all_work_items)} work items → "
    f"{len(my_work_items)} assigned to this shard",
    flush=True,
)

if not my_work_items:
    raise RuntimeError(
        f"[ag_ray] Shard {SHARD_INDEX}/{NUM_SHARDS} has zero work items. "
        f"Check that SYNTHETIC_REGRESSION_DATASET_INDEX has rows for suite_id={SUITE_ID!r}."
    )

# Pre-fetch AutoGluon predictor class to trigger import errors early.
get_tabular_predictor_class()

# ---------------------------------------------------------------------------
# Ray remote task
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=TASK_CPUS)
def _autogluon_work_item(item_meta: dict, dataset_payload: dict) -> dict:
    """Execute one AutoGluon fit+predict for a single (dataset, seed, condition) triple.

    Parameters
    ----------
    item_meta : dict
        Work-item metadata (suite_id, dataset_id, split_seed, etc.).
    dataset_payload : dict
        Pre-loaded dataset dict with keys X, y, betaX, n_total, etc.
        Passed through Ray object store; driver downloads each dataset once.

    Returns
    -------
    dict
        Canonical output row (status='ok', 'skipped', or 'failed').
    """
    import gc
    import os
    import shutil
    import tempfile

    import numpy as np
    from autogluon_models import get_tabular_predictor_class, predict_autogluon_timed
    from evaluate_synthetic_regression import (
        build_split_for_seed,
        preprocess_train_only,
        compute_regression_metrics,
        memory_guard_matrix,
        _empty_row,
        _skipped_row,
        _failed_row,
        _check_tmp_free_bytes,
        SYNREG_AG_MIN_TMP_FREE_BYTES,
    )

    suite_id    = item_meta["suite_id"]
    suite_family = item_meta.get("suite_family", "primary")
    dataset_id  = item_meta.get("dataset_id")
    dataset_seed = item_meta.get("dataset_seed")
    prior_regime = item_meta.get("prior_regime")
    split_seed  = int(item_meta["split_seed"])
    n_train_override = item_meta.get("n_train_override")

    task_cpus   = int(os.getenv("AUTOGLUON_TASK_CPUS", "1"))
    time_limit  = int(os.getenv("AUTOGLUON_TIME_LIMIT", "300"))
    presets     = os.getenv("AUTOGLUON_PRESETS", "best_quality")
    min_tmp     = int(os.getenv("BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", "5368709120"))
    cpu_max_feat = int(os.getenv("BENCHMARK_CPU_MAX_PROCESSED_FEATURES", "512"))
    cpu_max_mat  = int(os.getenv("BENCHMARK_CPU_MAX_MATRIX_BYTES", "2147483648"))

    data = dataset_payload

    n_total = int(data["n_total"])
    p_signal = int(data.get("p_signal", data["X"].shape[1]))
    p_noise  = int(data.get("p_noise", 0))
    p_total  = int(data.get("p_total", data["X"].shape[1]))
    feature_noise_level  = int(data.get("feature_noise_level", 0))
    target_noise_scale   = float(data.get("target_noise_scale", 1.0))
    is_anchor = bool(data.get("training_size_anchor", False))

    base = {
        "suite_id": suite_id,
        "suite_family": suite_family,
        "dataset_id": dataset_id,
        "dataset_seed": dataset_seed,
        "split_seed": split_seed,
        "prior_regime": prior_regime,
        "method": "AutoGluon",
        "logical_dataset_key": (
            item_meta.get("logical_dataset_key")
            or f"{suite_id}:{prior_regime}:{dataset_id:04d}"
        ),
        "n_total": n_total,
        "n_holdout": int(data.get("n_holdout_default", n_total // 5)),
        "p_signal": p_signal,
        "p_noise": p_noise,
        "p_total": p_total,
        "feature_noise_level": feature_noise_level,
        "target_noise_scale": target_noise_scale,
        "training_size_anchor": (
            n_train_override == item_meta.get("_anchor_n") if n_train_override is not None
            else is_anchor
        ),
    }

    # Guard: /tmp free space
    free_bytes = _check_tmp_free_bytes()
    if free_bytes < min_tmp:
        return _skipped_row(
            base,
            f"insufficient_tmp_space (free={free_bytes} < {min_tmp})",
        )

    ag_path = None
    try:
        split = build_split_for_seed(data, split_seed, n_train_override)
        X_train, y_train = split["X_train"], split["y_train"]
        X_holdout = split["X_holdout"]
        betaX_holdout = split["betaX_holdout"]
        y_holdout = split["y_holdout"]
        n_train   = split["n_train"]
        n_holdout = split["n_holdout"]
        base["n_train"]   = n_train
        base["n_holdout"] = n_holdout

        X_train_p, X_holdout_p = preprocess_train_only(X_train, X_holdout)

        # Guard: CPU matrix size
        n_feat = X_train_p.shape[1]
        matrix_bytes = X_train_p.nbytes + X_holdout_p.nbytes
        if n_feat > cpu_max_feat:
            return _skipped_row(
                base,
                f"cpu_matrix_too_large (features={n_feat} > {cpu_max_feat})",
            )
        if matrix_bytes > cpu_max_mat:
            return _skipped_row(
                base,
                f"cpu_matrix_too_large (matrix_bytes={matrix_bytes} > {cpu_max_mat})",
            )

        get_tabular_predictor_class()

        ag_path = os.path.join(
            tempfile.gettempdir(),
            f"ag_ray_{dataset_id}_{split_seed}_{n_train_override or 'def'}"
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
        gc.collect()


# ---------------------------------------------------------------------------
# Dataset loading helpers (driver only, bounded in-flight)
# ---------------------------------------------------------------------------

def _load_dataset_for_item(item: dict) -> dict:
    """Download and return the dataset for a work item (driver-side)."""
    return load_prepared_synthetic_dataset(item, LOCAL_CACHE)


def _dataset_payload_nbytes(payload: dict) -> int:
    total = 0
    for value in payload.values():
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            total += int(nbytes)
    return total


def _base_row_for_item(item: dict) -> dict:
    return {
        "suite_id": SUITE_ID,
        "suite_family": item.get("suite_family", "primary"),
        "dataset_id": item.get("dataset_id"),
        "dataset_seed": item.get("dataset_seed"),
        "split_seed": int(item.get("split_seed", 0)),
        "prior_regime": item.get("prior_regime"),
        "method": "AutoGluon",
        "logical_dataset_key": (
            item.get("logical_dataset_key")
            or f"{SUITE_ID}:{item.get('prior_regime')}:{item.get('dataset_id', 0):04d}"
        ),
    }


# ---------------------------------------------------------------------------
# Main distributed evaluation loop
# ---------------------------------------------------------------------------

MAX_IN_FLIGHT = _env_int("SYNREG_AUTOGLUON_MAX_IN_FLIGHT", WORKERS_PER_SHARD)
if MAX_IN_FLIGHT < 1:
    raise RuntimeError(
        f"[ag_ray] SYNREG_AUTOGLUON_MAX_IN_FLIGHT={MAX_IN_FLIGHT} must be >= 1."
    )

print(
    f"[ag_ray] starting distributed evaluation: "
    f"{len(my_work_items)} work items "
    f"max_in_flight={MAX_IN_FLIGHT} task_cpus={TASK_CPUS}",
    flush=True,
)

output_rows: list[dict] = []
pending: list[tuple[ray.ObjectRef, dict]] = []  # (future, item_meta)
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
        # Driver downloads dataset; passed to worker via Ray object store
        try:
            dataset_payload = _load_dataset_for_item(item)
        except Exception as exc:
            # Dataset load failure → emit failed row without submitting Ray task
            base = _base_row_for_item(item)
            output_rows.append(_failed_row(base, type(exc).__name__, str(exc)[:500]))
            print(
                f"[ag_ray] dataset load failed for item "
                f"dataset_id={item.get('dataset_id')} split_seed={item.get('split_seed')}: {exc}",
                flush=True,
            )
            continue

        dataset_bytes = _dataset_payload_nbytes(dataset_payload)
        if dataset_bytes > MAX_DATASET_BYTES:
            output_rows.append(
                _skipped_row(
                    _base_row_for_item(item),
                    (
                        "autogluon_dataset_too_large "
                        f"(dataset_bytes={dataset_bytes} > {MAX_DATASET_BYTES})"
                    ),
                )
            )
            del dataset_payload
            continue

        payload_ref = ray.put(dataset_payload)
        future = _autogluon_work_item.remote(item, payload_ref)
        pending.append((future, item))
        submitted += 1
        del dataset_payload  # release local reference; worker gets it via object store

    if not pending:
        break  # all work done

    # Collect one completed result (blocking)
    done_futures, remaining_futures = ray.wait(
        [f for f, _ in pending],
        num_returns=1,
        timeout=None,
    )
    if done_futures:
        done_future = done_futures[0]
        pending = [(f, m) for f, m in pending if f != done_future]
        try:
            row = ray.get(done_future)
            output_rows.append(row)
        except Exception as exc:
            # Ray task raised unexpectedly (not caught inside the task)
            print(f"[ag_ray] unexpected Ray task error: {exc}", flush=True)
        completed += 1
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

# Override SYNREG_RESULTS_STAGE so write_part_csv_to_stage uses the correct stage path.
os.environ["SYNREG_RESULTS_STAGE"] = RESULTS_STAGE

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
