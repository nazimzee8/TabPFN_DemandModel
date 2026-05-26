"""Lightweight AutoGluon dependency timing probe for Snowflake MLJobs.

Measures how long the Snowflake scheduling + container startup + pip install
takes before the Python entrypoint starts, and how long autogluon.tabular and
ray take to import after the environment is ready.

Interpretation:
- Time from MLJob submission to first JSON log line approximates scheduling +
  image pull + pip install (when SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE="pip").
- autogluon_import_complete.import_seconds measures pure import overhead after
  the environment is ready.
- Compare pip vs no_pip_baseline probe waves to estimate dependency bootstrap
  overhead under concurrency. no_pip_baseline intentionally skips AutoGluon/Ray
  imports so it measures startup-to-entrypoint without requiring dependencies.
- preinstalled mode validates a custom image by importing AutoGluon/Ray without
  pip installation.

Does not connect to Snowflake, call ray cluster init, query data, or write files.
"""

from __future__ import annotations

import json
import os
import time

# ---------------------------------------------------------------------------
# Capture entrypoint start times immediately — before any other imports.
# ---------------------------------------------------------------------------
_ENTRYPOINT_WALL_TIME = time.time()
_ENTRYPOINT_PERF_COUNTER = time.perf_counter()


def _log(event: str, **fields) -> None:
    record = {"event": event, **fields}
    print(json.dumps(record, default=str), flush=True)


def _elapsed() -> float:
    return time.perf_counter() - _ENTRYPOINT_PERF_COUNTER


# ---------------------------------------------------------------------------
# Resolve configuration from environment.
# ---------------------------------------------------------------------------
deps_mode = os.getenv("SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE", "pip")
probe_label = os.getenv("SYNREG_AG_IMPORT_PROBE_LABEL", "autogluon_import_timing_probe")
runtime_environment = os.getenv("EVAL_RUNTIME_ENVIRONMENT")
_BASELINE_MODES = {"baseline", "no_pip", "no_pip_baseline", "startup_baseline"}
import_dependencies = deps_mode not in _BASELINE_MODES

# ---------------------------------------------------------------------------
# Event 1: python_entrypoint_started — emitted immediately.
# ---------------------------------------------------------------------------
_log(
    "python_entrypoint_started",
    wall_time_epoch_s=_ENTRYPOINT_WALL_TIME,
    perf_counter_s=_ENTRYPOINT_PERF_COUNTER,
    runtime_environment=runtime_environment,
    deps_mode=deps_mode,
    probe_label=probe_label,
)

# ---------------------------------------------------------------------------
# Event 2: autogluon_import_complete, autogluon_import_skipped, or import_failed.
# ---------------------------------------------------------------------------
if import_dependencies:
    _t0 = time.perf_counter()
    try:
        import autogluon.tabular  # noqa: F401
        _ag_import_s = time.perf_counter() - _t0
        try:
            import autogluon.version  # type: ignore[import]
            _ag_version = autogluon.version.__version__
        except Exception:
            try:
                _ag_version = autogluon.tabular.__version__  # type: ignore[attr-defined]
            except Exception:
                _ag_version = None
        _log(
            "autogluon_import_complete",
            import_seconds=round(_ag_import_s, 4),
            elapsed_since_entrypoint_seconds=round(_elapsed(), 4),
            autogluon_version=_ag_version,
        )
    except Exception as exc:
        _log(
            "import_failed",
            module="autogluon.tabular",
            error_type=type(exc).__name__,
            error_message=str(exc),
            elapsed_since_entrypoint_seconds=round(_elapsed(), 4),
        )
        raise
else:
    _log(
        "autogluon_import_skipped",
        reason="no_pip_baseline",
        elapsed_since_entrypoint_seconds=round(_elapsed(), 4),
    )

# ---------------------------------------------------------------------------
# Event 3: ray_import_complete, ray_import_skipped, or import_failed.
# ---------------------------------------------------------------------------
if import_dependencies:
    _t0 = time.perf_counter()
    try:
        import ray  # noqa: F401
        _ray_import_s = time.perf_counter() - _t0
        try:
            _ray_version = ray.__version__
        except Exception:
            _ray_version = None
        _log(
            "ray_import_complete",
            import_seconds=round(_ray_import_s, 4),
            elapsed_since_entrypoint_seconds=round(_elapsed(), 4),
            ray_version=_ray_version,
        )
    except Exception as exc:
        _log(
            "import_failed",
            module="ray",
            error_type=type(exc).__name__,
            error_message=str(exc),
            elapsed_since_entrypoint_seconds=round(_elapsed(), 4),
        )
        raise
else:
    _log(
        "ray_import_skipped",
        reason="no_pip_baseline",
        elapsed_since_entrypoint_seconds=round(_elapsed(), 4),
    )
