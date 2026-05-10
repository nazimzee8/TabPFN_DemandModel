"""
Lightweight runtime probe for Snowflake MLJob / Ray startup diagnostics.

This script is intentionally minimal. It verifies whether the Snowflake MLJob
container can reach Python execution before model training, PyTorchDistributor,
dataset materialization, or DDP/NCCL logic is involved.
"""

import os
import importlib
import socket
import subprocess
import sys

print("[runtime_probe] entered Python", flush=True)
print("[runtime_probe] host:", socket.gethostname(), flush=True)
print("[runtime_probe] python:", sys.version, flush=True)
print("[runtime_probe] cwd:", os.getcwd(), flush=True)

print("[runtime_probe] env TRAIN_NUM_NODES:", os.environ.get("TRAIN_NUM_NODES", ""), flush=True)
print("[runtime_probe] env EXPECTED_TRAIN_WORLD_SIZE:", os.environ.get("EXPECTED_TRAIN_WORLD_SIZE", ""), flush=True)
print("[runtime_probe] env STRICT_WORLD_SIZE_CHECK:", os.environ.get("STRICT_WORLD_SIZE_CHECK", ""), flush=True)
print("[runtime_probe] env CHECKPOINT_OUTPUT_NAME:", os.environ.get("CHECKPOINT_OUTPUT_NAME", ""), flush=True)
print("[runtime_probe] env EVAL_RUNTIME_ENVIRONMENT:", os.environ.get("EVAL_RUNTIME_ENVIRONMENT", ""), flush=True)
print("[runtime_probe] env RUNTIME_PROBE_LABEL:", os.environ.get("RUNTIME_PROBE_LABEL", ""), flush=True)
print("[runtime_probe] env REQUIRED_IMPORTS:", os.environ.get("REQUIRED_IMPORTS", ""), flush=True)
print("[runtime_probe] env REQUIRE_CUDA:", os.environ.get("REQUIRE_CUDA", ""), flush=True)


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _check_required_imports():
    required = [
        name.strip()
        for name in os.environ.get("REQUIRED_IMPORTS", "").split(",")
        if name.strip()
    ]
    if not required:
        print("[runtime_probe] no REQUIRED_IMPORTS configured", flush=True)
        return

    missing = []
    for module_name in required:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "")
            version_suffix = f" version={version}" if version else ""
            print(f"[runtime_probe] required import ok: {module_name}{version_suffix}", flush=True)
        except Exception as exc:
            missing.append(f"{module_name} ({type(exc).__name__}: {exc})")
            print(f"[runtime_probe] required import failed: {module_name}: {exc!r}", flush=True)
    if missing:
        raise RuntimeError(
            "Runtime image is missing required import(s): " + "; ".join(missing)
        )

try:
    import ray
    print("[runtime_probe] ray version:", ray.__version__, flush=True)
except Exception as exc:
    print("[runtime_probe] ray import failed:", repr(exc), flush=True)

try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    print("[runtime_probe] torch version:", torch.__version__, flush=True)
    print("[runtime_probe] cuda available:", cuda_available, flush=True)
    print("[runtime_probe] cuda device count:", cuda_device_count, flush=True)
except Exception as exc:
    print("[runtime_probe] torch import failed:", repr(exc), flush=True)
    if _env_flag("REQUIRE_CUDA"):
        raise
else:
    if _env_flag("REQUIRE_CUDA") and (not cuda_available or cuda_device_count <= 0):
        raise RuntimeError(
            "REQUIRE_CUDA=true but torch reports CUDA unavailable "
            f"(available={cuda_available}, device_count={cuda_device_count})."
        )

_check_required_imports()

try:
    result = subprocess.run(
        ["df", "-h"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    print("[runtime_probe] df -h stdout:", result.stdout, flush=True)
    print("[runtime_probe] df -h stderr:", result.stderr, flush=True)
except Exception as exc:
    print("[runtime_probe] df -h failed:", repr(exc), flush=True)

try:
    result = subprocess.run(
        ["mount"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    print("[runtime_probe] mount stdout:", result.stdout, flush=True)
    print("[runtime_probe] mount stderr:", result.stderr, flush=True)
except Exception as exc:
    print("[runtime_probe] mount failed:", repr(exc), flush=True)

print("[runtime_probe] completed", flush=True)
