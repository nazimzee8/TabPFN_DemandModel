"""
Lightweight runtime probe for Snowflake MLJob / Ray startup diagnostics.

This script is intentionally minimal. It verifies whether the Snowflake MLJob
container can reach Python execution before model training, PyTorchDistributor,
dataset materialization, or DDP/NCCL logic is involved.
"""

import os
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

try:
    import ray
    print("[runtime_probe] ray version:", ray.__version__, flush=True)
except Exception as exc:
    print("[runtime_probe] ray import failed:", repr(exc), flush=True)

try:
    import torch
    print("[runtime_probe] torch version:", torch.__version__, flush=True)
    print("[runtime_probe] cuda available:", torch.cuda.is_available(), flush=True)
    print("[runtime_probe] cuda device count:", torch.cuda.device_count(), flush=True)
except Exception as exc:
    print("[runtime_probe] torch import failed:", repr(exc), flush=True)

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
