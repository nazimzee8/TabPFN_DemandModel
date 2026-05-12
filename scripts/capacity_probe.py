"""
capacity_probe.py — lightweight Snowflake node allocation probe.
Prints CAPACITY_PROBE_LABEL and EVAL_RUNTIME_ENVIRONMENT, sleeps 30 s, exits 0.
"""
import os, sys, time

label   = os.environ.get("CAPACITY_PROBE_LABEL", "capacity_probe")
runtime = os.environ.get("EVAL_RUNTIME_ENVIRONMENT", "unknown")
print(f"[capacity_probe] started: label={label!r} runtime={runtime!r}", flush=True)
print("[capacity_probe] sleeping 30s to hold node allocation ...", flush=True)
time.sleep(30)
print(f"[capacity_probe] complete: {label!r}", flush=True)
