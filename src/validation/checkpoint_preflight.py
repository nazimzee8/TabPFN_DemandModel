"""
checkpoint_preflight.py — Verify checkpoints and HPO configs exist before eval starts.

Call check_checkpoint() in eval orchestrators before submitting compute jobs to catch
missing files early, before wasting GPU hours on a job that will crash at load time.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Optional


def check_hpo_config(stage_path: str, *, strict: bool = False) -> bool:
    """
    Verify that the HPO best_config.json exists and is parseable at `stage_path`.
    Uses the Snowpark session if available via snowflake_io; falls back to a no-op
    warning if running outside Snowflake.

    Returns True if OK, False if missing/invalid.
    """
    # Try to use snowflake_io if available
    try:
        from snowflake_io import _get_active_session_or_none
        session = _get_active_session_or_none()
        if session is not None:
            rows = session.sql(f"LIST {stage_path}").collect()
            if not rows:
                msg = f"[checkpoint_preflight] HPO config not found: {stage_path}"
                if strict:
                    raise FileNotFoundError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return False
            return True
    except ImportError:
        pass

    # Local fallback (useful in tests)
    local_path = stage_path.replace("@MODEL_STAGE/", "").replace("@", "")
    if os.path.exists(local_path):
        try:
            with open(local_path) as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, OSError) as e:
            msg = f"[checkpoint_preflight] HPO config invalid at {local_path}: {e}"
            if strict:
                raise
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return False

    warnings.warn(
        f"[checkpoint_preflight] Cannot verify HPO config (no session, no local file): {stage_path}",
        RuntimeWarning, stacklevel=2,
    )
    return True  # Don't block if we can't check


def check_model_checkpoint(stage_path: str, *, strict: bool = False) -> bool:
    """
    Verify that the model checkpoint (.pt) exists at `stage_path` before eval.

    Returns True if OK, False if missing.
    """
    try:
        from snowflake_io import _get_active_session_or_none
        session = _get_active_session_or_none()
        if session is not None:
            rows = session.sql(f"LIST {stage_path}").collect()
            if not rows:
                msg = f"[checkpoint_preflight] Model checkpoint not found: {stage_path}"
                if strict:
                    raise FileNotFoundError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return False
            return True
    except ImportError:
        pass

    warnings.warn(
        f"[checkpoint_preflight] Cannot verify checkpoint (no session): {stage_path}",
        RuntimeWarning, stacklevel=2,
    )
    return True
