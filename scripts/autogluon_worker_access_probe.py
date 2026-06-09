"""Lightweight worker dataset-access probe for combined AutoGluon MLJobs.

This probe validates the production worker-local dataset access path without
running AutoGluon training and without writing evaluation shard outputs.
Current production access mode is driver_presigned_url: the driver derives
presigned HTTPS URLs, and workers download with urllib without creating
Snowpark sessions.

Ray mode:
  - attach to the Snowflake Ray cluster
  - validate expected node/CPU visibility
  - load LINEAR_REGRESSION_DATASET_INDEX metadata in the driver only
  - pass small item dicts to Ray worker tasks
  - workers resolve the dataset_access descriptor and load one dataset locally

Single-node mode:
  - load metadata and validate one local dataset access path without Ray
"""

from __future__ import annotations

import json
import os
import time


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer; got {raw!r}.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive; got {raw!r}.")
    return value


def _env_non_neg_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer; got {raw!r}.") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be non-negative; got {raw!r}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_env(name: str, description: str = "") -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required. {description}")
    return value.strip()


def _wait_for_ray_capacity(ray, *, expected_nodes: int, expected_cpus_min: int) -> tuple[int, int, dict]:
    ready_timeout_seconds = _env_int("SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS", 300)
    poll_seconds = _env_int("SYNREG_RAY_CLUSTER_READY_POLL_SECONDS", 10)
    deadline = time.monotonic() + ready_timeout_seconds
    last_state: tuple[int, int] | None = None

    while True:
        live_nodes = [node for node in ray.nodes() if node.get("Alive")]
        cluster_resources = ray.cluster_resources()
        available_cpus = int(cluster_resources.get("CPU", 0))
        state = (len(live_nodes), available_cpus)
        if state != last_state:
            print(
                "[ag_worker_access_probe] ray readiness: "
                f"live_nodes={len(live_nodes)}/{expected_nodes} "
                f"available_cpus={available_cpus}/{expected_cpus_min} "
                f"resources={dict(cluster_resources)}",
                flush=True,
            )
            last_state = state

        if len(live_nodes) >= expected_nodes and available_cpus >= expected_cpus_min:
            return len(live_nodes), available_cpus, dict(cluster_resources)

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Ray cluster did not reach requested capacity before readiness timeout: "
                f"live_nodes={len(live_nodes)}/{expected_nodes}, "
                f"available_cpus={available_cpus}/{expected_cpus_min}, "
                f"timeout_seconds={ready_timeout_seconds}, "
                f"resources={dict(cluster_resources)}. "
                "Snowflake may still be starting target_instances, or account/compute-pool "
                "concurrency may be below the requested Ray capacity envelope."
            )

        time.sleep(poll_seconds)


def _load_probe_items(*, shard_index: int, num_shards: int, max_items: int) -> list[dict]:
    from evaluate_linear_regression import (
        assign_synthetic_regression_shard,
        build_compact_synreg_work_item,
        compact_work_items_serialized_bytes,
        create_snowpark_session,
        expand_synreg_work_items,
        load_synthetic_regression_index,
    )

    session = create_snowpark_session()
    rows = load_synthetic_regression_index(suite_family=None, session=session)
    if not rows:
        suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "<unset>")
        raise RuntimeError(
            "LINEAR_REGRESSION_DATASET_INDEX returned zero rows for "
            f"suite_id={suite_id!r}."
        )

    work_items = expand_synreg_work_items(rows, train_size_grid=None)
    assigned = assign_synthetic_regression_shard(work_items, shard_index, num_shards)
    if not assigned:
        raise RuntimeError(
            f"Shard {shard_index}/{num_shards} has zero metadata work items. "
            "Check SYNTHETIC_REGRESSION_NUM_SHARDS and SYNTHETIC_REGRESSION_SHARD_INDEX."
        )

    access_mode = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE", "driver_presigned_url")
    max_item_bytes = _env_int("SYNREG_MAX_WORK_ITEM_BYTES", 8192)
    dataset_url_cache: dict[str, str] = {}
    items = [
        build_compact_synreg_work_item(
            row,
            session=session,
            access_mode=access_mode,
            max_item_bytes=max_item_bytes,
            dataset_url_cache=dataset_url_cache,
        )
        for row in assigned[:max_items]
    ]
    total_bytes, max_bytes = compact_work_items_serialized_bytes(items)
    print(
        "[ag_worker_access_probe] driver loaded metadata only: "
        f"index_rows={len(rows)} work_items={len(work_items)} "
        f"assigned={len(assigned)} probe_items={len(items)} "
        f"unique_dataset_access_urls={len(dataset_url_cache)} "
        f"serialized_total_bytes={total_bytes} serialized_max_item_bytes={max_bytes}",
        flush=True,
    )
    return items


def _validate_item_access(item_meta: dict) -> dict:
    """Worker-side validation. Raises with clear messages on access failure."""
    import os

    from evaluate_linear_regression import load_prepared_synthetic_dataset_from_access

    dataset_access = item_meta.get("dataset_access") or {}
    access_mode = dataset_access.get("mode", "driver_presigned_url")
    stage_path = dataset_access.get("stage_path") or item_meta.get("stage_path")
    if not stage_path:
        raise RuntimeError("Worker item is missing dataset stage_path.")

    local_cache = os.getenv("SYNTHETIC_REGRESSION_LOCAL_CACHE", "/tmp/synreg_worker_access_probe")
    try:
        data = load_prepared_synthetic_dataset_from_access(item_meta, local_cache)
    except Exception as exc:
        raise RuntimeError(
            "Worker failed to resolve dataset access via "
            f"{access_mode} for stage_path={stage_path!r}: {type(exc).__name__}: {exc}"
        ) from exc

    x = data.get("X")
    y = data.get("y")
    beta_x = data.get("betaX")
    x_shape = tuple(getattr(x, "shape", ()))
    y_shape = tuple(getattr(y, "shape", ()))
    beta_shape = tuple(getattr(beta_x, "shape", ()))
    if len(x_shape) != 2 or not y_shape or not beta_shape:
        raise RuntimeError(
            "Worker loaded dataset but found invalid array shapes: "
            f"X={x_shape} y={y_shape} betaX={beta_shape}."
        )

    return {
        "status": "ok",
        "access_mode": access_mode,
        "stage_path": stage_path,
        "dataset_id": item_meta.get("dataset_id"),
        "split_seed": item_meta.get("split_seed"),
        "x_shape": x_shape,
        "y_shape": y_shape,
        "beta_x_shape": beta_shape,
    }


def _run_single_node() -> None:
    num_shards = _env_int("SYNTHETIC_REGRESSION_NUM_SHARDS", 1)
    shard_index = _env_non_neg_int("SYNTHETIC_REGRESSION_SHARD_INDEX", 0)
    max_items = _env_int("SYNREG_WORKER_ACCESS_PROBE_ITEMS", 1)
    if not (0 <= shard_index < num_shards):
        raise RuntimeError(
            f"SYNTHETIC_REGRESSION_SHARD_INDEX={shard_index} out of range [0, {num_shards})."
        )

    items = _load_probe_items(
        shard_index=shard_index,
        num_shards=num_shards,
        max_items=max_items,
    )
    results = [_validate_item_access(item) for item in items]
    print(
        "[ag_worker_access_probe] single-node validation complete: "
        f"results={json.dumps(results, default=str)}",
        flush=True,
    )


def _run_ray() -> None:
    expected_nodes = _env_int("EXPECTED_RAY_NODES", 1)
    expected_cpus_min = _env_int("EXPECTED_RAY_CPUS_MIN", expected_nodes)
    workers_per_shard = _env_int("SYNREG_AUTOGLUON_WORKERS_PER_SHARD", expected_nodes)
    num_shards = _env_int("SYNTHETIC_REGRESSION_NUM_SHARDS", 1)
    shard_index = _env_non_neg_int("SYNTHETIC_REGRESSION_SHARD_INDEX", 0)
    task_cpus = _env_int("AUTOGLUON_TASK_CPUS", 1)
    max_items = _env_int(
        "SYNREG_WORKER_ACCESS_PROBE_ITEMS",
        max(1, workers_per_shard),
    )
    if not (0 <= shard_index < num_shards):
        raise RuntimeError(
            f"SYNTHETIC_REGRESSION_SHARD_INDEX={shard_index} out of range [0, {num_shards})."
        )

    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not importable in this runtime. The worker-access probe in "
            "ray mode requires the Snowflake Ray runtime."
        ) from exc

    ray_address_mode = os.getenv("SYNREG_RAY_ADDRESS_MODE", "auto")
    if ray_address_mode == "explicit":
        ray_address = _require_env(
            "RAY_HEAD_ADDRESS",
            "SYNREG_RAY_ADDRESS_MODE=explicit requires a Ray head address.",
        )
    elif ray_address_mode == "auto":
        ray_address = "auto"
    else:
        raise RuntimeError(
            f"Unsupported SYNREG_RAY_ADDRESS_MODE={ray_address_mode!r}; "
            "expected 'auto' or 'explicit'."
        )

    ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True, include_dashboard=False)
    live_node_count, available_cpus, cluster_resources = _wait_for_ray_capacity(
        ray,
        expected_nodes=expected_nodes,
        expected_cpus_min=expected_cpus_min,
    )
    print(
        "[ag_worker_access_probe] ray ready: "
        f"live_nodes={live_node_count} available_cpus={available_cpus} "
        f"resources={dict(cluster_resources)}",
        flush=True,
    )

    items = _load_probe_items(
        shard_index=shard_index,
        num_shards=num_shards,
        max_items=max_items,
    )
    print(
        "[ag_worker_access_probe] submitting small item dicts to workers: "
        f"items={len(items)} task_cpus={task_cpus} no_ray_put=true",
        flush=True,
    )

    @ray.remote(num_cpus=task_cpus)
    def _worker_access_probe_item(item_meta: dict) -> dict:
        return _validate_item_access(item_meta)

    futures = [_worker_access_probe_item.remote(item) for item in items]
    results = ray.get(futures)
    print(
        "[ag_worker_access_probe] ray worker validation complete: "
        f"results={json.dumps(results, default=str)}",
        flush=True,
    )


def main() -> None:
    _require_env(
        "SYNTHETIC_REGRESSION_SUITE_ID",
        "Set to the combined suite id, usually 'linear_all_v1'.",
    )
    use_ray = _env_bool("SYNREG_WORKER_ACCESS_PROBE_USE_RAY", False)
    access_mode = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE", "driver_presigned_url")
    print(
        "[ag_worker_access_probe] started: "
        f"use_ray={use_ray} suite_id={os.getenv('SYNTHETIC_REGRESSION_SUITE_ID')!r} "
        f"access_mode={access_mode!r}",
        flush=True,
    )
    if use_ray:
        _run_ray()
    else:
        _run_single_node()
    print("[ag_worker_access_probe] complete", flush=True)


if __name__ == "__main__":
    main()
