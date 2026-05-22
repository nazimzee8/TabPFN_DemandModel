"""Lightweight Snowflake multi-node Ray capacity probe."""

from __future__ import annotations

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


expected_nodes = _env_int("EXPECTED_RAY_NODES", 1)
expected_cpus_min = _env_int("EXPECTED_RAY_CPUS_MIN", expected_nodes)
sleep_seconds = _env_int("CAPACITY_PROBE_SLEEP_SECONDS", 30)
label = os.getenv("CAPACITY_PROBE_LABEL", "ray_capacity_probe")

print(
    f"[ray_capacity_probe] started: label={label!r} "
    f"expected_nodes={expected_nodes} expected_cpus_min={expected_cpus_min}",
    flush=True,
)

try:
    import ray
except ImportError as exc:
    raise RuntimeError(
        "Ray is not importable in this Snowflake runtime. Multi-node AutoGluon "
        "requires the Snowflake Ray runtime to be available."
    ) from exc

ray.init(address="auto", ignore_reinit_error=True, include_dashboard=False)
live_nodes = [node for node in ray.nodes() if node.get("Alive")]
cluster_resources = ray.cluster_resources()
available_cpus = int(cluster_resources.get("CPU", 0))

print(
    f"[ray_capacity_probe] ray attached: live_nodes={len(live_nodes)} "
    f"available_cpus={available_cpus} resources={dict(cluster_resources)}",
    flush=True,
)

if len(live_nodes) < expected_nodes:
    raise RuntimeError(
        f"Ray cluster has {len(live_nodes)} live nodes, expected at least "
        f"{expected_nodes}. Snowflake did not attach all requested target_instances."
    )

if available_cpus < expected_cpus_min:
    raise RuntimeError(
        f"Ray cluster has {available_cpus} CPUs, expected at least "
        f"{expected_cpus_min}."
    )

print(f"[ray_capacity_probe] sleeping {sleep_seconds}s to hold allocation", flush=True)
time.sleep(sleep_seconds)
print("[ray_capacity_probe] complete", flush=True)
