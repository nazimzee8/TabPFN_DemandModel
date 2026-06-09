"""Tests for allocate_regression_tasks and build_coverage_audit (Section 16)."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from dgp_helpers import allocate_regression_tasks, build_coverage_audit


def test_weighted_quota_covers_all_regimes():
    """weighted_quota mode should give all regimes >= min_regime_count."""
    assignments, audit = allocate_regression_tasks(
        n_datasets=120,
        profile="linear_stat_aware",
        base_seed=42,
        allocation_mode="weighted_quota",
        min_regime_count=5,
    )
    realized = audit["realized_regime_counts"]
    for regime, count in realized.items():
        assert count >= 5, f"regime {regime!r} has only {count} datasets"


def test_balanced_mode_equal_counts():
    """balanced mode should distribute datasets as evenly as possible (max-min <= 1)."""
    assignments, audit = allocate_regression_tasks(
        n_datasets=120,
        profile="linear_stat_aware",
        base_seed=42,
        allocation_mode="balanced",
    )
    realized = audit["realized_regime_counts"]
    counts = list(realized.values())
    assert max(counts) - min(counts) <= 1


def test_weighted_mode_runs_without_error():
    """weighted (stochastic) mode should run without error."""
    assignments, audit = allocate_regression_tasks(
        n_datasets=60,
        profile="linear_stat_aware",
        base_seed=99,
        allocation_mode="weighted",
    )
    assert len(assignments) == 60


def test_total_matches_n_datasets():
    """All three modes should return exactly n_datasets assignments."""
    n = 50
    for mode in ["weighted_quota", "balanced", "weighted"]:
        assignments, _ = allocate_regression_tasks(
            n_datasets=n,
            profile="linear_stat_aware",
            base_seed=42,
            allocation_mode=mode,
        )
        assert len(assignments) == n, f"mode={mode}: got {len(assignments)}, expected {n}"


def test_assignment_fields():
    """Each assignment dict must have prior_regime key."""
    assignments, _ = allocate_regression_tasks(
        n_datasets=20,
        profile="linear_stat_aware",
        base_seed=0,
        allocation_mode="balanced",
    )
    for a in assignments:
        assert "prior_regime" in a, f"Missing 'prior_regime' key in assignment: {list(a.keys())}"


def test_invalid_profile_raises():
    """A classification profile passed to allocate_regression_tasks should raise ValueError."""
    with pytest.raises(ValueError, match="not a regression profile"):
        allocate_regression_tasks(
            n_datasets=10,
            profile="linear_classification_stat_aware",
            base_seed=42,
        )


def test_strict_coverage_raises_on_undercoverage():
    """strict_coverage=True with 1 dataset and min_regime_count=5 should raise."""
    with pytest.raises((ValueError, Exception)):
        allocate_regression_tasks(
            n_datasets=1,
            profile="linear_stat_aware",
            base_seed=42,
            allocation_mode="weighted_quota",
            min_regime_count=5,
            strict_coverage=True,
        )


def test_build_coverage_audit_passes():
    """All regimes covered -> coverage_passed=True."""
    records = [
        {"prior_regime": "A_iid_dense"},
        {"prior_regime": "B_iid_sparse"},
        {"prior_regime": "C_heavy_tail_noise"},
    ]
    audit = build_coverage_audit(
        datasets=records,
        task_family="linear_regression",
        required_axes=["prior_regime"],
        min_counts={"prior_regime": 1},
        strict_coverage=False,
        allocation_mode="balanced",
    )
    assert audit["coverage_passed"] is True


def test_build_coverage_audit_fails_missing():
    """Underrepresented axis values should appear in underrepresented_axis_values."""
    records = [
        {"prior_regime": "A_iid_dense"},
        {"prior_regime": "A_iid_dense"},
    ]
    audit = build_coverage_audit(
        datasets=records,
        task_family="linear_regression",
        required_axes=["prior_regime"],
        min_counts={"prior_regime": 5},  # require 5 but only 2 for A
        strict_coverage=False,
        allocation_mode="balanced",
    )
    # A_iid_dense has 2 datasets < 5, so underrepresented
    assert len(audit["underrepresented_axis_values"]) > 0


def test_manifest_has_coverage_audit_section():
    """Integration: run allocate_regression_tasks and check audit has required keys."""
    assignments, audit = allocate_regression_tasks(
        n_datasets=30,
        profile="linear_stat_aware",
        base_seed=1,
        allocation_mode="weighted_quota",
    )
    assert "allocation_mode" in audit
    assert "profile" in audit
    assert "regime_counts" in audit
    assert "realized_regime_counts" in audit
