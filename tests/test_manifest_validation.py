"""Tests for validate_controlled_manifest (Section 16)."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from dgp_helpers import validate_controlled_manifest


def _make_valid_manifest(n: int = 2) -> dict:
    """Build a minimal valid controlled manifest."""
    datasets = []
    for i in range(n):
        datasets.append({
            "dataset_id": i,
            "difficulty_score": 1,
            "difficulty_tier": "core",
            "difficulty_reasons": [],
            "estimated_h_tensor_bytes": 1000,
            "memory_risk_bucket": "low",
            "exceeds_default_gpu_guard": False,
        })
    return {
        "datasets": datasets,
        "coverage_audit": {"coverage_passed": True},
        "memory_audit": {
            "memory_risk_counts": {"low": n, "medium": 0, "high": 0, "exceeds_default_guard": 0},
        },
        "difficulty_audit": {
            "difficulty_tier_counts": {"core": n, "robust": 0, "stress": 0},
        },
        "alignment_audit": {"emitted": False},
        "generation_controls": {"task_family": "linear_regression"},
    }


def test_valid_manifest_passes():
    """A valid manifest should not raise."""
    validate_controlled_manifest(_make_valid_manifest(3))


def test_missing_coverage_audit_raises():
    """Missing coverage_audit should raise ValueError."""
    manifest = _make_valid_manifest(2)
    del manifest["coverage_audit"]
    with pytest.raises(ValueError, match="coverage_audit"):
        validate_controlled_manifest(manifest)


def test_missing_memory_audit_raises():
    """Missing memory_audit should raise ValueError."""
    manifest = _make_valid_manifest(2)
    del manifest["memory_audit"]
    with pytest.raises(ValueError, match="memory_audit"):
        validate_controlled_manifest(manifest)


def test_missing_difficulty_audit_raises():
    """Missing difficulty_audit should raise ValueError."""
    manifest = _make_valid_manifest(2)
    del manifest["difficulty_audit"]
    with pytest.raises(ValueError, match="difficulty_audit"):
        validate_controlled_manifest(manifest)


def test_missing_per_dataset_field_raises():
    """Missing difficulty_score in a dataset record should raise ValueError."""
    manifest = _make_valid_manifest(2)
    del manifest["datasets"][0]["difficulty_score"]
    with pytest.raises(ValueError, match="difficulty_score"):
        validate_controlled_manifest(manifest)


def test_count_mismatch_raises():
    """Wrong difficulty_tier_counts total should raise ValueError."""
    manifest = _make_valid_manifest(2)
    # Set tier counts to wrong total (99 != 2)
    manifest["difficulty_audit"]["difficulty_tier_counts"] = {"core": 99, "robust": 0, "stress": 0}
    with pytest.raises(ValueError):
        validate_controlled_manifest(manifest)


def test_missing_memory_risk_bucket_raises():
    """Missing memory_risk_bucket in a dataset should raise ValueError."""
    manifest = _make_valid_manifest(2)
    del manifest["datasets"][1]["memory_risk_bucket"]
    with pytest.raises(ValueError, match="memory_risk_bucket"):
        validate_controlled_manifest(manifest)
