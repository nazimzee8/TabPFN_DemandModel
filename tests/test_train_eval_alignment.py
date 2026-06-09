"""Tests for build_train_eval_alignment_report (Section 16)."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from dgp_helpers import build_train_eval_alignment_report


def _make_manifest(
    regime_group_counts: dict,
    grid_n: list | None = None,
    grid_p: list | None = None,
    difficulty_tiers: dict | None = None,
) -> dict:
    """Helper to build a minimal manifest for alignment testing."""
    return {
        "n_datasets_by_regime_group": regime_group_counts,
        "grid_metadata": {
            "n_grid": grid_n or [64, 128, 256],
            "p_signal_grid": grid_p or [4, 8, 16],
        },
        "difficulty_audit": {
            "difficulty_tier_counts": difficulty_tiers or {"core": 5, "robust": 3, "stress": 2},
        },
    }


def test_alignment_passed_when_eval_subset_of_train():
    """eval regimes subset of train regimes -> alignment_passed=True."""
    train = _make_manifest({"A_iid_dense": 10, "B_iid_sparse": 5, "C_heavy_tail_noise": 3})
    eval_ = _make_manifest({"A_iid_dense": 5, "B_iid_sparse": 2})
    report = build_train_eval_alignment_report(train, eval_)
    assert report["alignment_passed"] is True
    assert len(report["eval_only_regimes"]) == 0


def test_alignment_failed_when_eval_has_extra_regimes():
    """eval has regime not in train -> alignment_passed=False."""
    train = _make_manifest({"A_iid_dense": 10})
    eval_ = _make_manifest({"A_iid_dense": 5, "Z_unseen_regime": 2})
    report = build_train_eval_alignment_report(train, eval_)
    assert report["alignment_passed"] is False
    assert "Z_unseen_regime" in report["eval_only_regimes"]


def test_shared_regimes():
    """Shared regimes should be correct intersection."""
    train = _make_manifest({"A_iid_dense": 10, "B_iid_sparse": 5})
    eval_ = _make_manifest({"B_iid_sparse": 3, "C_heavy_tail_noise": 2})
    report = build_train_eval_alignment_report(train, eval_)
    assert "B_iid_sparse" in report["shared_regimes"]
    assert "A_iid_dense" not in report["shared_regimes"]
    assert "C_heavy_tail_noise" not in report["shared_regimes"]


def test_train_only_regimes():
    """train_only_regimes should be correct difference."""
    train = _make_manifest({"A_iid_dense": 10, "B_iid_sparse": 5})
    eval_ = _make_manifest({"A_iid_dense": 5})
    report = build_train_eval_alignment_report(train, eval_)
    assert "B_iid_sparse" in report["train_only_regimes"]
    assert "A_iid_dense" not in report["train_only_regimes"]


def test_report_structure():
    """Report must contain all required top-level keys."""
    train = _make_manifest({"A_iid_dense": 10})
    eval_ = _make_manifest({"A_iid_dense": 5})
    report = build_train_eval_alignment_report(train, eval_)
    required_keys = {
        "train_regimes", "eval_regimes", "shared_regimes",
        "train_only_regimes", "eval_only_regimes", "alignment_passed",
        "alignment_warnings", "n_distribution_overlap",
        "p_total_distribution_overlap", "difficulty_tier_overlap",
    }
    assert required_keys.issubset(report.keys())


def test_warnings_non_empty_on_failure():
    """alignment_warnings should be non-empty when alignment fails."""
    train = _make_manifest({"A_iid_dense": 10})
    eval_ = _make_manifest({"Z_new_regime": 5})
    report = build_train_eval_alignment_report(train, eval_)
    assert not report["alignment_passed"]
    assert len(report["alignment_warnings"]) > 0


def test_emit_alignment_report_smoke():
    """Smoke test: alignment report can be built from two classification manifests."""
    train = {
        "realized_regime_counts": {"A_iid_dense_logistic": 10, "B_iid_sparse_logistic": 5},
        "grid_metadata": {"n_grid": [64, 128], "p_signal_grid": [4, 8]},
        "difficulty_audit": {"difficulty_tier_counts": {"core": 8, "robust": 5, "stress": 2}},
    }
    eval_ = {
        "realized_regime_counts": {"A_iid_dense_logistic": 3},
        "grid_metadata": {"n_grid": [64, 128], "p_signal_grid": [4, 8]},
        "difficulty_audit": {"difficulty_tier_counts": {"core": 2, "robust": 1, "stress": 0}},
    }
    report = build_train_eval_alignment_report(train, eval_)
    assert report["alignment_passed"] is True
    assert "A_iid_dense_logistic" in report["shared_regimes"]
