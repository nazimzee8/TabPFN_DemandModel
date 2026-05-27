"""spcs_ray_coordinator.py

Coordinator container for the SPCS self-managed Ray distributed AutoGluon backend.

This script merges the roles of the Ray head node and the AutoGluon driver into a
single SPCS container, reducing the required container count from 36 to 30 for a
6-shard × 4-worker setup.

Execution sequence:
  1. Start a Ray head subprocess with --num-cpus=0 and --object-store-memory.
  2. Poll localhost:{head_port} until reachable (same socket-poll approach as
     spcs_ray_worker.py).
  3. Run the configured driver script as a child subprocess, passing
     SYNREG_RAY_ADDRESS_MODE=explicit and RAY_HEAD_ADDRESS=localhost:{head_port}
     so the driver connects to the local head.
  4. Forward the driver script's exit code.
  5. In finally: terminate the Ray head gracefully.

Environment variables (read by this script):
  SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT           (default 6379)
  SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES  (default 500000000 = ~500 MB)
  SPCS_RAY_HEAD_CONNECT_TIMEOUT_SECONDS          (default 300)
  SPCS_RAY_RUN_ID                                Run ID for cluster identity (Finding 6)
  SPCS_RAY_SHARD_INDEX                           Shard index for cluster identity
  SPCS_RAY_DRIVER_SCRIPT                         Path to driver script (default: autogluon_ray.py)
  SYNREG_SPCS_RAY_NODE_MANAGER_PORT             (default 6380) — deterministic raylet port
  SYNREG_SPCS_RAY_OBJECT_MANAGER_PORT           (default 6381) — deterministic object manager port
  SYNREG_SPCS_RAY_RUNTIME_ENV_AGENT_PORT        (default 6382) — deterministic runtime env agent port
  SYNREG_SPCS_RAY_MIN_WORKER_PORT               (default 10002) — min port for Ray worker processes
  SYNREG_SPCS_RAY_MAX_WORKER_PORT               (default 10010) — max port for Ray worker processes

All other env vars (SYNTHETIC_REGRESSION_*, SYNREG_*, AUTOGLUON_*, BENCHMARK_*,
HOME, SNOWFLAKE_*) are inherited by the driver script subprocess unchanged.

Design notes:
  - The coordinator sets SYNREG_RAY_ADDRESS_MODE=explicit and
    RAY_HEAD_ADDRESS=localhost:{head_port} in the subprocess environment so that
    autogluon_ray.py does not rely on "auto" discovery.
  - SPCS_RAY_RUN_ID and SPCS_RAY_SHARD_INDEX are already in the container's env
    (set by the orchestrator); they flow through to autogluon_ray.py unchanged.
  - The head starts with --num-cpus=0 so it does NOT contribute CPU capacity.
    Workers provide all schedulable CPUs (N workers × AUTOGLUON_TASK_CPUS each).
  - --object-store-memory is set explicitly to avoid Ray's heuristic default,
    which can over-allocate on memory-constrained containers.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTOGLUON_RAY_PY = os.path.join(_SCRIPT_DIR, "autogluon_ray.py")
_DRIVER_SCRIPT = os.getenv("SPCS_RAY_DRIVER_SCRIPT", str(_AUTOGLUON_RAY_PY))

head_port = int(os.getenv("SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT", "6379"))
obj_store_bytes = int(
    os.getenv("SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES", "500000000")
)
head_connect_timeout = int(os.getenv("SPCS_RAY_HEAD_CONNECT_TIMEOUT_SECONDS", "300"))
node_manager_port = int(os.getenv("SYNREG_SPCS_RAY_NODE_MANAGER_PORT", "6380"))
object_manager_port = int(os.getenv("SYNREG_SPCS_RAY_OBJECT_MANAGER_PORT", "6381"))
runtime_env_agent_port = int(os.getenv("SYNREG_SPCS_RAY_RUNTIME_ENV_AGENT_PORT", "6382"))
min_worker_port = int(os.getenv("SYNREG_SPCS_RAY_MIN_WORKER_PORT", "10002"))
max_worker_port = int(os.getenv("SYNREG_SPCS_RAY_MAX_WORKER_PORT", "10010"))

_run_id = os.getenv("SPCS_RAY_RUN_ID", "")
_shard_index = os.getenv("SPCS_RAY_SHARD_INDEX", "")


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


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
    "spcs_ray_coordinator_starting",
    head_port=head_port,
    obj_store_bytes=obj_store_bytes,
    node_manager_port=node_manager_port,
    object_manager_port=object_manager_port,
    runtime_env_agent_port=runtime_env_agent_port,
    min_worker_port=min_worker_port,
    max_worker_port=max_worker_port,
    run_id=_run_id,
    shard_index=_shard_index,
)

# Build cluster identity resource string (Finding 6).
# The custom resource lets the driver (autogluon_ray.py) verify it connected to the
# correct shard's head, not a neighbour's.
_cluster_id_resources: str | None = None
if _run_id and _shard_index != "":
    _cluster_id_key = f"spcs_cluster_id_{_run_id}_{_shard_index}"
    _cluster_id_resources = f'{{"{_cluster_id_key}": 1}}'
    _log("spcs_ray_coordinator_cluster_id", cluster_id_key=_cluster_id_key)

# ---------------------------------------------------------------------------
# Step 1: Start Ray head subprocess (non-blocking)
# ---------------------------------------------------------------------------
ray_head_cmd = [
    "ray", "start", "--head",
    f"--port={head_port}",
    f"--node-manager-port={node_manager_port}",
    f"--object-manager-port={object_manager_port}",
    f"--runtime-env-agent-port={runtime_env_agent_port}",
    f"--min-worker-port={min_worker_port}",
    f"--max-worker-port={max_worker_port}",
    "--include-dashboard=false",
    # Finding 5: head contributes zero CPUs so workers provide all schedulable capacity.
    "--num-cpus=0",
    f"--object-store-memory={obj_store_bytes}",
    "--block",
]
if _cluster_id_resources:
    ray_head_cmd.insert(-1, f"--resources={_cluster_id_resources}")

_log("spcs_ray_coordinator_head_cmd", cmd=ray_head_cmd)

ray_head_proc: subprocess.Popen | None = None
driver_rc = 1  # default to failure
try:
    try:
        ray_head_proc = subprocess.Popen(ray_head_cmd)
        _log("spcs_ray_coordinator_head_started", pid=ray_head_proc.pid)
    except Exception as exc:
        _log("spcs_ray_coordinator_head_start_failed", error=str(exc))
        raise

    # ---------------------------------------------------------------------------
    # Step 2: Poll localhost:{head_port} until reachable
    # ---------------------------------------------------------------------------
    _log("spcs_ray_coordinator_waiting_for_head", port=head_port, timeout=head_connect_timeout)
    deadline = time.monotonic() + head_connect_timeout
    head_reachable = False
    while time.monotonic() < deadline:
        # Check if head process already exited (startup failure)
        rc = ray_head_proc.poll()
        if rc is not None:
            _log("spcs_ray_coordinator_head_exited_early", returncode=rc)
            _dump_ray_logs()
            sys.exit(1)
        try:
            with socket.create_connection(("localhost", head_port), timeout=5):
                head_reachable = True
                break
        except OSError:
            time.sleep(5)

    if not head_reachable:
        raise RuntimeError(
            f"Ray head did not become reachable on localhost:{head_port} "
            f"within {head_connect_timeout}s."
        )
    _log("spcs_ray_coordinator_head_reachable")

    # ---------------------------------------------------------------------------
    # Step 3: Run autogluon_ray.py as child subprocess
    # ---------------------------------------------------------------------------
    driver_env = os.environ.copy()
    # Override address mode so the driver connects to our local head, not "auto".
    driver_env["SYNREG_RAY_ADDRESS_MODE"] = "explicit"
    driver_env["RAY_HEAD_ADDRESS"] = f"localhost:{head_port}"

    driver_cmd = [sys.executable, _DRIVER_SCRIPT]
    _log("spcs_ray_coordinator_driver_starting", cmd=driver_cmd, driver_script=_DRIVER_SCRIPT)

    driver_result = subprocess.run(driver_cmd, env=driver_env)
    driver_rc = driver_result.returncode
    _log("spcs_ray_coordinator_driver_finished", returncode=driver_rc)

finally:
    # ---------------------------------------------------------------------------
    # Step 4: Terminate Ray head gracefully
    # ---------------------------------------------------------------------------
    if ray_head_proc is not None:
        head_rc = ray_head_proc.poll()
        if head_rc is not None and head_rc != 0:
            _dump_ray_logs()
        if head_rc is None:
            _log("spcs_ray_coordinator_terminating_head")
            ray_head_proc.terminate()
            try:
                ray_head_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                ray_head_proc.kill()
            _log("spcs_ray_coordinator_head_terminated")

_log("spcs_ray_coordinator_done", driver_returncode=driver_rc)
sys.exit(driver_rc)
