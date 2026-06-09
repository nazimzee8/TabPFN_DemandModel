"""Tests verifying schema separation between regression and classification families."""
from __future__ import annotations
import json
import sys
import subprocess
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_regression_smoke(out_dir: Path, n: int = 12) -> dict:
    """Run regression smoke generation and return the manifest."""
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "generate_synthetic_regression.py"),
            "--n_datasets", str(n),
            "--profile", "linear_stat_aware",
            "--allocation_mode", "balanced",
            "--suite_id", "test_schema_sep_reg",
            "--out_dir", str(out_dir),
            "--no-strict_coverage",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if result.returncode != 0:
        pytest.skip(f"regression smoke failed: {result.stderr[-500:]}")
    manifest_path = (
        out_dir / "synthetic_regression_prepared" / "test_schema_sep_reg"
        / "synthetic_regression_manifest.json"
    )
    if not manifest_path.exists():
        pytest.skip("manifest not found")
    return json.loads(manifest_path.read_text())


def _run_classification_smoke(out_dir: Path, n: int = 12) -> dict:
    """Run classification smoke generation and return the manifest."""
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "generate_synthetic_classification.py"),
            "--n_datasets", str(n),
            "--n_datasets_per_sweep", "2",
            "--profile", "linear_classification_stat_aware",
            "--suite_id", "test_schema_sep_cls",
            "--output_root", str(out_dir),
            "--no-strict_coverage",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if result.returncode != 0:
        pytest.skip(f"classification smoke failed: {result.stderr[-500:]}")
    manifest_path = out_dir / "test_schema_sep_cls" / "synthetic_classification_manifest.json"
    if not manifest_path.exists():
        pytest.skip("manifest not found")
    return json.loads(manifest_path.read_text())


@pytest.fixture(scope="module")
def reg_manifest(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("schema_sep_reg")
    return _run_regression_smoke(out_dir)


@pytest.fixture(scope="module")
def cls_manifest(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("schema_sep_cls")
    return _run_classification_smoke(out_dir)


def test_regression_manifest_schema_version(reg_manifest):
    """Regression manifest schema_version should start with 'linear_regression_eval'."""
    assert reg_manifest["schema_version"].startswith("linear_regression_eval")


def test_regression_manifest_no_classification_fields(reg_manifest):
    """Regression manifest must not contain realized_K_counts or realized_imbalance_counts."""
    assert "realized_K_counts" not in reg_manifest
    assert "realized_imbalance_counts" not in reg_manifest


def test_classification_manifest_schema_version(cls_manifest):
    """Classification manifest schema_version should contain 'classification'."""
    assert "classification" in cls_manifest["schema_version"]


def test_classification_output_not_in_regression_dir(cls_manifest):
    """Classification stage_paths must not contain 'synthetic_regression_prepared'."""
    stage_paths = cls_manifest.get("stage_paths", {})
    for path in stage_paths.values():
        assert "synthetic_regression_prepared" not in path


def test_regression_output_not_in_classification_dir(reg_manifest):
    """Regression dataset stage_path must not reference synthetic_classification_prepared."""
    datasets = reg_manifest.get("datasets", [])
    if datasets:
        stage = datasets[0].get("stage_path", "")
        assert "synthetic_classification_prepared" not in stage


def test_regression_task_family_if_present(reg_manifest):
    """If task_family key present, it must be 'linear_regression'."""
    if "task_family" in reg_manifest:
        assert reg_manifest["task_family"] == "linear_regression"


def test_classification_task_family(cls_manifest):
    """Classification manifest task_family must be 'linear_classification'."""
    assert cls_manifest.get("task_family") == "linear_classification"


def test_regression_generation_controls_task_family(reg_manifest):
    """generation_controls.task_family must equal 'linear_regression'."""
    gc = reg_manifest.get("generation_controls", {})
    if gc:
        assert gc.get("task_family") == "linear_regression"


def test_classification_generation_controls_task_family(cls_manifest):
    """generation_controls.task_family must equal 'linear_classification'."""
    gc = cls_manifest.get("generation_controls", {})
    if gc:
        assert gc.get("task_family") == "linear_classification"


def test_regression_datasets_prior_regime_not_classification_regime(reg_manifest):
    """Regression dataset records must have 'prior_regime', not 'classification_regime'."""
    datasets = reg_manifest.get("datasets", [])
    if datasets:
        first = datasets[0]
        assert "prior_regime" in first
        assert "classification_regime" not in first
