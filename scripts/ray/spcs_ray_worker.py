"""spcs_ray_worker.py

Starts a Ray worker node that connects to a self-managed Ray head in an SPCS container.
Keeps alive until a sentinel file or timeout.

Environment variables:
  RAY_HEAD_ADDRESS                              required, e.g. "spcs-ray-coord-abc123-0.myschema.mydb.snowflakecomputing.internal:6379"
  AUTOGLUON_TASK_CPUS                           (default 1) — logical CPUs advertised to Ray scheduler
  SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES  (default 268435456 = 256 MiB)
  SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS       (default 300)
  SPCS_RAY_WORKER_KEEPALIVE_SECONDS             (default 7200)
  SYNREG_SPCS_RAY_NODE_MANAGER_PORT             (default 6380) — deterministic raylet port
  SYNREG_SPCS_RAY_OBJECT_MANAGER_PORT           (default 6381) — deterministic object manager port
  SYNREG_SPCS_RAY_RUNTIME_ENV_AGENT_PORT        (default 6382) — deterministic runtime env agent port
  SYNREG_SPCS_RAY_MIN_WORKER_PORT               (default 10002) — min port for Ray worker processes
  SYNREG_SPCS_RAY_MAX_WORKER_PORT               (default 10010) — max port for Ray worker processes
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time

START_MONOTONIC = time.monotonic()

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
    os.getenv("SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES", "268435456")
)
node_manager_port = int(os.getenv("SYNREG_SPCS_RAY_NODE_MANAGER_PORT", "6380"))
object_manager_port = int(os.getenv("SYNREG_SPCS_RAY_OBJECT_MANAGER_PORT", "6381"))
runtime_env_agent_port = int(os.getenv("SYNREG_SPCS_RAY_RUNTIME_ENV_AGENT_PORT", "6382"))
min_worker_port = int(os.getenv("SYNREG_SPCS_RAY_MIN_WORKER_PORT", "10002"))
max_worker_port = int(os.getenv("SYNREG_SPCS_RAY_MAX_WORKER_PORT", "10010"))

_ray_proc: subprocess.Popen | None = None  # set to proc after Popen; read by signal handler

_RAY_LOG_FILES = [
    "/tmp/ray/session_latest/logs/raylet.err",
    "/tmp/ray/session_latest/logs/raylet.out",
    "/tmp/ray/session_latest/logs/gcs_server.err",
    "/tmp/ray/session_latest/logs/gcs_server.out",
    "/tmp/ray/session_latest/logs/dashboard_agent.err",
    "/tmp/ray/session_latest/logs/dashboard_agent.out",
]
_RAY_LOG_TAIL_BYTES = 32768  # 32 KB per file


def _dump_ray_logs() -> None:
    for path in _RAY_LOG_FILES:
        try:
            if not os.path.exists(path):
                _log("ray_log_file", path=path, missing=True)
                continue
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - _RAY_LOG_TAIL_BYTES))
                tail = f.read().decode("utf-8", errors="replace")
            _log("ray_log_file", path=path, bytes=min(size, _RAY_LOG_TAIL_BYTES), content=tail)
        except Exception as exc:
            _log("ray_log_file_error", path=path, error=str(exc))


_log(
    "spcs_ray_worker_starting",
    ray_head_address=ray_head_address,
    task_cpus=task_cpus,
    obj_store_bytes=obj_store_bytes,
    node_manager_port=node_manager_port,
    object_manager_port=object_manager_port,
    runtime_env_agent_port=runtime_env_agent_port,
    min_worker_port=min_worker_port,
    max_worker_port=max_worker_port,
)

# Parse host:port for reachability check
_parts = ray_head_address.rsplit(":", 1)
_head_host = _parts[0]
_head_port = int(_parts[1]) if len(_parts) == 2 else 6379


def _handle_signal(signum: int, frame) -> None:  # noqa: ANN001
    try:
        sig_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        sig_name = str(signum)
    uptime = round(time.monotonic() - START_MONOTONIC, 2)
    extra: dict = {}
    for _k, _e in (("run_id", "SPCS_RAY_RUN_ID"), ("shard_index", "SPCS_RAY_SHARD_INDEX"),
                   ("worker_index", "SPCS_RAY_WORKER_INDEX")):
        _v = os.getenv(_e)
        if _v is not None:
            extra[_k] = _v
    _log(
        "spcs_ray_worker_signal_received",
        signal_number=signum,
        signal_name=sig_name,
        uptime_seconds=uptime,
        ray_head_address=ray_head_address,
        task_cpus=task_cpus,
        obj_store_bytes=obj_store_bytes,
        node_manager_port=node_manager_port,
        object_manager_port=object_manager_port,
        runtime_env_agent_port=runtime_env_agent_port,
        min_worker_port=min_worker_port,
        max_worker_port=max_worker_port,
        ray_proc_pid=_ray_proc.pid if _ray_proc is not None else None,
        ray_proc_returncode=_ray_proc.poll() if _ray_proc is not None else None,
        **extra,
    )
    if os.path.isdir("/tmp/ray/session_latest/logs"):
        _dump_ray_logs()
    if _ray_proc is not None and _ray_proc.poll() is None:
        _ray_proc.terminate()
        try:
            _ray_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _ray_proc.kill()
    exit_code = 128 + signum
    _log(
        "spcs_ray_worker_exit_after_signal",
        final_ray_returncode=_ray_proc.poll() if _ray_proc is not None else None,
        exit_code=exit_code,
    )
    sys.exit(exit_code)

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


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
    f"--node-manager-port={node_manager_port}",
    f"--object-manager-port={object_manager_port}",
    f"--runtime-env-agent-port={runtime_env_agent_port}",
    f"--min-worker-port={min_worker_port}",
    f"--max-worker-port={max_worker_port}",
    f"--num-cpus={task_cpus}",
    f"--object-store-memory={obj_store_bytes}",
    "--block",
]
_log("spcs_ray_worker_cmd", cmd=cmd)

try:
    proc = subprocess.Popen(cmd)
    _ray_proc = proc
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
        _log("spcs_ray_worker_ray_exited", returncode=ret, uptime_seconds=round(time.monotonic() - START_MONOTONIC, 2))
        if ret != 0:
            _dump_ray_logs()
        sys.exit(ret if ret != 0 else 0)
    time.sleep(poll_seconds)
else:
    _log("spcs_ray_worker_keepalive_expired", keepalive_seconds=keepalive_seconds)

proc.terminate()
try:
    proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    proc.kill()
_log("spcs_ray_worker_done", uptime_seconds=round(time.monotonic() - START_MONOTONIC, 2))
