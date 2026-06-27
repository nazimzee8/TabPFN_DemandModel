"""Lightweight Snowflake multi-node Ray capacity probe."""

from __future__ import annotations

import json
import os
import time


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


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


def _wait_for_ray_capacity(ray, *, expected_nodes: int, expected_cpus_min: int) -> tuple[int, int, dict]:
    ready_timeout_seconds = _env_int("SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS", 900)
    poll_seconds = _env_int("SYNREG_RAY_CLUSTER_READY_POLL_SECONDS", 10)
    deadline = time.monotonic() + ready_timeout_seconds
    last_state: tuple[int, int] | None = None
    live_nodes: list = []
    available_cpus: int = 0
    cluster_resources: dict = {}

    while True:
        live_nodes = [node for node in ray.nodes() if node.get("Alive")]
        cluster_resources = ray.cluster_resources()
        available_cpus = int(cluster_resources.get("CPU", 0))
        state = (len(live_nodes), available_cpus)
        if state != last_state:
            _log(
                "ray_capacity_probe_readiness",
                live_nodes=len(live_nodes),
                expected_nodes=expected_nodes,
                available_cpus=available_cpus,
                expected_cpus_min=expected_cpus_min,
                resources=dict(cluster_resources),
            )
            last_state = state

        if len(live_nodes) >= expected_nodes and available_cpus >= expected_cpus_min:
            return len(live_nodes), available_cpus, dict(cluster_resources)

        if time.monotonic() >= deadline:
            _log(
                "ray_capacity_probe_timeout",
                expected_nodes=expected_nodes,
                expected_cpus_min=expected_cpus_min,
                live_nodes=len(live_nodes),
                available_cpus=available_cpus,
                timeout_seconds=ready_timeout_seconds,
                poll_seconds=poll_seconds,
                resources=dict(cluster_resources),
                hint="set SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE=true to leave worker jobs running",
            )
            raise RuntimeError(
                "Ray cluster did not reach requested capacity before readiness timeout. "
                f"live_nodes={len(live_nodes)}/{expected_nodes}, "
                f"available_cpus={available_cpus}/{expected_cpus_min}, "
                f"timeout_seconds={ready_timeout_seconds}, "
                f"resources={dict(cluster_resources)}. "
                "Hint: set SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE=true to leave worker jobs "
                "running after a probe failure so their logs can be inspected manually."
            )

        time.sleep(poll_seconds)


expected_nodes = _env_int("EXPECTED_RAY_NODES", 1)
expected_cpus_min = _env_int("EXPECTED_RAY_CPUS_MIN", expected_nodes)
sleep_seconds = _env_int("CAPACITY_PROBE_SLEEP_SECONDS", 30)
label = os.getenv("CAPACITY_PROBE_LABEL", "ray_capacity_probe")
ray_address_mode = os.getenv("SYNREG_RAY_ADDRESS_MODE", "auto")
if ray_address_mode == "explicit":
    ray_address = os.getenv("RAY_HEAD_ADDRESS")
    if not ray_address:
        raise RuntimeError(
            "SYNREG_RAY_ADDRESS_MODE=explicit requires RAY_HEAD_ADDRESS."
        )
elif ray_address_mode == "auto":
    ray_address = "auto"
else:
    raise RuntimeError(
        f"Unsupported SYNREG_RAY_ADDRESS_MODE={ray_address_mode!r}; "
        "expected 'auto' or 'explicit'."
    )

_log(
    "ray_capacity_probe_started",
    label=label,
    expected_nodes=expected_nodes,
    expected_cpus_min=expected_cpus_min,
    sleep_seconds=sleep_seconds,
    ray_address=ray_address,
    ready_timeout_seconds=_env_int("SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS", 900),
)

try:
    import ray
except ImportError as exc:
    raise RuntimeError(
        "Ray is not importable in this Snowflake runtime. Multi-node AutoGluon "
        "requires the Snowflake Ray runtime to be available."
    ) from exc

ray.init(address=ray_address, ignore_reinit_error=True, include_dashboard=False)
live_node_count, available_cpus, cluster_resources = _wait_for_ray_capacity(
    ray,
    expected_nodes=expected_nodes,
    expected_cpus_min=expected_cpus_min,
)

_log(
    "ray_capacity_probe_ready",
    label=label,
    live_nodes=live_node_count,
    available_cpus=available_cpus,
    resources=cluster_resources,
)

_log("ray_capacity_probe_sleeping", label=label, sleep_seconds=sleep_seconds)
time.sleep(sleep_seconds)
_log("ray_capacity_probe_complete", label=label)
