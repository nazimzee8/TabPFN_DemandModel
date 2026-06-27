"""
_bootstrap.py — recursive sys.path installer for TabPFN_DemandModel.

Adds every subdirectory of src/ and scripts/ (and those top-level dirs themselves)
to sys.path, making all flat-import modules resolvable regardless of which
subdirectory a script or test lives in.

Usage:
    import _bootstrap  # anywhere — install() is called on import

For standalone scripts that run outside pytest (where conftest.py is not loaded),
find the repo root via sentinel and then import this module:

    from pathlib import Path as _Path
    _p = _Path(__file__).resolve()
    for _anc in _p.parents:
        if (_anc / "_bootstrap.py").exists():
            import sys; sys.path.insert(0, str(_anc)); break
    import _bootstrap  # noqa: E402

This module is intentionally NOT PUT to @MODEL_STAGE/scripts/ — Snowflake uses
a flat stage and resolves modules by bare filename without a bootstrap.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _iter_code_dirs():
    """Yield src/, scripts/, and every non-__pycache__ subdir within each."""
    for top_name in ("src", "scripts"):
        base = _REPO_ROOT / top_name
        if not base.is_dir():
            continue
        yield base
        for child in sorted(base.rglob("*")):
            if child.is_dir() and "__pycache__" not in child.parts:
                yield child


def install() -> Path:
    """Insert every code directory into sys.path (idempotent)."""
    for d in _iter_code_dirs():
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
    return _REPO_ROOT


install()
