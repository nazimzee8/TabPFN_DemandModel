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

_run_id = os.getenv("SPCS_RAY_RUN_ID", "")
_shard_index = os.getenv("SPCS_RAY_SHARD_INDEX", "")


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


_log(
    "spcs_ray_coordinator_starting",
    head_port=head_port,
    obj_store_bytes=obj_store_bytes,
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
    if ray_head_proc is not None and ray_head_proc.poll() is None:
        _log("spcs_ray_coordinator_terminating_head")
        ray_head_proc.terminate()
        try:
            ray_head_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            ray_head_proc.kill()
        _log("spcs_ray_coordinator_head_terminated")

_log("spcs_ray_coordinator_done", driver_returncode=driver_rc)
sys.exit(driver_rc)
