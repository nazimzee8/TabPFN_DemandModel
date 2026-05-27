"""spcs_ray_worker.py

Starts a Ray worker node that connects to a self-managed Ray head in an SPCS container.
Keeps alive until a sentinel file or timeout.

Environment variables:
  RAY_HEAD_ADDRESS                              required, e.g. "spcs-ray-coord-abc123-0.myschema.mydb.snowflakecomputing.internal:6379"
  AUTOGLUON_TASK_CPUS                           (default 1) — logical CPUs advertised to Ray scheduler
  SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES  (default 2000000000 = ~2 GB)
  SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS       (default 300)
  SPCS_RAY_WORKER_KEEPALIVE_SECONDS             (default 7200)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

_SENTINEL_FILE = "/tmp/spcs_ray_worker_done"


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Required env var {name!r} is not set.")
    return val.strip()


ray_head_address = _require_env("RAY_HEAD_ADDRESS")
connect_timeout = int(os.getenv("SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS", "300"))
keepalive_seconds = int(os.getenv("SPCS_RAY_WORKER_KEEPALIVE_SECONDS", "7200"))
task_cpus = int(os.getenv("AUTOGLUON_TASK_CPUS", "1"))
obj_store_bytes = int(
    os.getenv("SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES", "2000000000")
)

_log(
    "spcs_ray_worker_starting",
    ray_head_address=ray_head_address,
    task_cpus=task_cpus,
    obj_store_bytes=obj_store_bytes,
)

# Parse host:port for reachability check
_parts = ray_head_address.rsplit(":", 1)
_head_host = _parts[0]
_head_port = int(_parts[1]) if len(_parts) == 2 else 6379


def _wait_head_reachable(host: str, port: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError:
            time.sleep(5)
    raise RuntimeError(
        f"Ray head {host}:{port} not reachable after {timeout_seconds}s. "
        "Check that the head container started successfully."
    )


_log("spcs_ray_worker_waiting_for_head", host=_head_host, port=_head_port)
_wait_head_reachable(_head_host, _head_port, connect_timeout)
_log("spcs_ray_worker_head_reachable")

cmd = [
    "ray", "start",
    f"--address={ray_head_address}",
    f"--num-cpus={task_cpus}",
    f"--object-store-memory={obj_store_bytes}",
    "--block",
]
_log("spcs_ray_worker_cmd", cmd=cmd)

try:
    proc = subprocess.Popen(cmd)
    _log("spcs_ray_worker_started", pid=proc.pid)
except Exception as exc:
    _log("spcs_ray_worker_start_failed", error=str(exc))
    raise

deadline = time.monotonic() + keepalive_seconds
poll_seconds = 5
while time.monotonic() < deadline:
    if os.path.exists(_SENTINEL_FILE):
        _log("spcs_ray_worker_sentinel_received")
        break
    ret = proc.poll()
    if ret is not None:
        _log("spcs_ray_worker_ray_exited", returncode=ret)
        sys.exit(ret if ret != 0 else 0)
    time.sleep(poll_seconds)
else:
    _log("spcs_ray_worker_keepalive_expired", keepalive_seconds=keepalive_seconds)

proc.terminate()
try:
    proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    proc.kill()
_log("spcs_ray_worker_done")
