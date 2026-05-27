"""spcs_ray_head.py

Starts a Ray head node in an SPCS container and keeps alive until a sentinel file
or SIGTERM is received. Used in the SPCS self-managed Ray distributed AutoGluon backend.

Environment variables:
  SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT   (default 6379)
  SPCS_RAY_HEAD_KEEPALIVE_SECONDS       (default 7200 = 2h)
  SPCS_RAY_RUN_ID                        Run ID injected by the orchestrator for cluster
                                          identity verification (Finding 6).
  SPCS_RAY_SHARD_INDEX                   Shard index injected by the orchestrator.

Design notes (Finding 5 — Ray node count semantics):
  The head starts with --num-cpus=0 so that it does NOT contribute CPU capacity to the
  cluster. This means:
  - Workers provide all schedulable CPUs (N workers × TASK_CPUS each).
  - The live node count equals workers_per_shard + 1 (workers + head).
  - The driver's _wait_for_ray_capacity() uses expected_nodes=workers+1 in SPCS mode
    so that readiness is not declared prematurely.

Design notes (Finding 6 — Cluster identity):
  When SPCS_RAY_RUN_ID and SPCS_RAY_SHARD_INDEX are set, a custom Ray resource
  "spcs_cluster_id_<run_id>_<shard_index>" with value 1 is added to the head.
  The driver (autogluon_ray.py) verifies this resource exists after ray.init()
  to confirm it joined the correct shard's cluster, not a neighbour's.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_SENTINEL_FILE = "/tmp/spcs_ray_head_done"

head_port = int(os.getenv("SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT", "6379"))
keepalive_seconds = int(os.getenv("SPCS_RAY_HEAD_KEEPALIVE_SECONDS", "7200"))

# Cluster identity (Finding 6)
_run_id = os.getenv("SPCS_RAY_RUN_ID", "")
_shard_index = os.getenv("SPCS_RAY_SHARD_INDEX", "")


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


_log(
    "spcs_ray_head_starting",
    head_port=head_port,
    run_id=_run_id,
    shard_index=_shard_index,
)

# Build cluster identity resource string for --resources flag.
# The custom resource lets the driver verify it connected to the right shard's head.
_cluster_id_resources: str | None = None
if _run_id and _shard_index != "":
    _cluster_id_key = f"spcs_cluster_id_{_run_id}_{_shard_index}"
    _cluster_id_resources = f'{{"{_cluster_id_key}": 1}}'
    _log("spcs_ray_head_cluster_id", cluster_id_key=_cluster_id_key)

cmd = [
    "ray", "start", "--head",
    f"--port={head_port}",
    "--include-dashboard=false",
    # Finding 5: head contributes zero CPUs — only worker nodes provide schedulable capacity.
    # This prevents the readiness check from counting the head as a worker node.
    "--num-cpus=0",
    "--block",
]
if _cluster_id_resources:
    cmd.insert(-1, f"--resources={_cluster_id_resources}")

_log("spcs_ray_head_cmd", cmd=cmd)

try:
    proc = subprocess.Popen(cmd)
    _log("spcs_ray_head_started", pid=proc.pid)
except Exception as exc:
    _log("spcs_ray_head_start_failed", error=str(exc))
    raise

# Keep alive: wait for sentinel or timeout
deadline = time.monotonic() + keepalive_seconds
poll_seconds = 5
while time.monotonic() < deadline:
    if os.path.exists(_SENTINEL_FILE):
        _log("spcs_ray_head_sentinel_received")
        break
    ret = proc.poll()
    if ret is not None:
        _log("spcs_ray_head_ray_exited", returncode=ret)
        sys.exit(ret if ret != 0 else 0)
    time.sleep(poll_seconds)
else:
    _log("spcs_ray_head_keepalive_expired", keepalive_seconds=keepalive_seconds)

proc.terminate()
try:
    proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    proc.kill()
_log("spcs_ray_head_done")
