"""Tests for compute_difficulty_score, parse_difficulty_mix, build_difficulty_audit."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from dgp_helpers import compute_difficulty_score, parse_difficulty_mix, build_difficulty_audit


def test_regression_core_task():
    """A simple regression task should score <=1 and be 'core'."""
    meta = {
        "p_total": 8, "n_train": 200,
        "target_noise_scale": 0.25, "feature_noise_level": 0.0,
        "p_noise": 0, "covariance_type": "iid", "rho": 0.0,
        "prior_regime": "A_iid_dense", "target_noise_type": "gaussian",
    }
    result = compute_difficulty_score(meta, "linear_regression")
    assert result["difficulty_tier"] == "core"
    assert result["difficulty_score"] <= 1


def test_regression_stress_task():
    """A high-complexity regression task should score >=4 and be 'stress'."""
    meta = {
        "p_total": 256, "n_train": 50,   # high_p_total, underdetermined, low_n_over_p
        "target_noise_scale": 2.0,        # high_target_noise
        "feature_noise_level": 0.5,       # high_feature_noise
        "p_noise": 200,                    # high_noise_ratio (200/256 >= 0.5)
        "covariance_type": "block", "rho": 0.9,  # high_correlation
        "prior_regime": "A_iid_dense", "target_noise_type": "gaussian",
    }
    result = compute_difficulty_score(meta, "linear_regression")
    assert result["difficulty_tier"] == "stress"
    assert result["difficulty_score"] >= 4


def test_regression_robust_task():
    """A moderately complex regression task should be 'robust' (score 2-3)."""
    meta = {
        "p_total": 200, "n_train": 100,  # high_p_total, underdetermined, low_n_over_p = 3
        "target_noise_scale": 0.25, "feature_noise_level": 0.0,
        "p_noise": 0, "covariance_type": "iid", "rho": 0.0,
        "prior_regime": "A_iid_dense", "target_noise_type": "gaussian",
    }
    result = compute_difficulty_score(meta, "linear_regression")
    assert result["difficulty_tier"] == "robust"
    assert 2 <= result["difficulty_score"] <= 3


def test_classification_core_task():
    """Binary balanced classification with medium margin should be 'core'."""
    meta = {
        "p_total": 8, "n_train": 200,
        "label_noise_rate": 0.0, "feature_noise_level": 0.0,
        "p_noise": 0, "class_imbalance_type": "balanced",
        "margin_level": "medium", "num_classes": 2,
        "classification_regime": "A_iid_dense_logistic",
    }
    result = compute_difficulty_score(meta, "linear_classification")
    assert result["difficulty_tier"] == "core"
    assert result["difficulty_score"] <= 1


def test_classification_stress_task():
    """Classification with many stress signals should be 'stress'."""
    meta = {
        "p_total": 256, "n_train": 50,   # high_p_total, underdetermined, low_n_over_p
        "label_noise_rate": 0.20,         # high_label_noise
        "feature_noise_level": 0.5,       # high_feature_noise
        "p_noise": 200,                    # high_noise_ratio
        "class_imbalance_type": "severe", # severe_class_imbalance
        "margin_level": "low",            # low_margin
        "num_classes": 10,                # many_classes
        "classification_regime": "E_high_dim_dense_softmax",
    }
    result = compute_difficulty_score(meta, "linear_classification")
    assert result["difficulty_tier"] == "stress"
    assert result["difficulty_score"] >= 4


def test_difficulty_reasons_not_empty_for_hard_tasks():
    """Hard tasks should have non-empty difficulty_reasons."""
    meta = {
        "p_total": 300, "n_train": 50,
        "target_noise_scale": 2.0, "feature_noise_level": 0.5,
        "p_noise": 200, "covariance_type": "iid", "rho": 0.0,
        "prior_regime": "A_iid_dense", "target_noise_type": "gaussian",
    }
    result = compute_difficulty_score(meta, "linear_regression")
    assert len(result["difficulty_reasons"]) > 0


def test_unknown_task_family_raises():
    """Unknown task_family should raise ValueError."""
    with pytest.raises(ValueError, match="unknown task_family"):
        compute_difficulty_score({}, "linear_unsupervised")


def test_parse_difficulty_mix_valid():
    """Valid difficulty mix string should parse correctly."""
    result = parse_difficulty_mix("core=0.65,robust=0.30,stress=0.05")
    assert abs(result["core"] - 0.65) < 1e-6
    assert abs(result["robust"] - 0.30) < 1e-6
    assert abs(result["stress"] - 0.05) < 1e-6


def test_parse_difficulty_mix_invalid_key():
    """Unknown key in mix string should raise ValueError."""
    with pytest.raises(ValueError, match="unknown key"):
        parse_difficulty_mix("core=0.65,hard=0.35")


def test_parse_difficulty_mix_sum_not_one():
    """Sum not equal to 1 should raise ValueError."""
    with pytest.raises(ValueError, match="sum"):
        parse_difficulty_mix("core=0.50,robust=0.30,stress=0.05")


def test_build_difficulty_audit_tier_counts():
    """build_difficulty_audit should compute correct tier counts and stress fraction."""
    records = [
        {"difficulty_tier": "core", "difficulty_score": 1},
        {"difficulty_tier": "core", "difficulty_score": 0},
        {"difficulty_tier": "robust", "difficulty_score": 2},
        {"difficulty_tier": "stress", "difficulty_score": 5},
    ]
    configured_mix = {"core": 0.50, "robust": 0.25, "stress": 0.25}
    audit = build_difficulty_audit(records, "core_first", configured_mix)
    assert audit["difficulty_tier_counts"]["core"] == 2
    assert audit["difficulty_tier_counts"]["robust"] == 1
    assert audit["difficulty_tier_counts"]["stress"] == 1
    assert abs(audit["stress_task_fraction"] - 0.25) < 1e-6
