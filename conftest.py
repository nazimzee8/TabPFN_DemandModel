"""
conftest.py — root-level pytest configuration.

Redirects pytest's tmp_path fixture to the local tmp_pytest/ directory
to avoid Windows permission issues with AppData/Local/Temp/pytest-of-<user>.

PYTEST_DEBUG_TEMPROOT is read lazily by TempPathFactory.getbasetemp(), so
setting it at module-import time (before any test runs) takes effect.
"""
import os
from pathlib import Path

# Set before TempPathFactory.getbasetemp() is called (it reads this lazily).
_local_tmp = Path(__file__).parent / "tmp_pytest"
_local_tmp.mkdir(exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_local_tmp))
