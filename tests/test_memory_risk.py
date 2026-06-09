"""Tests for estimate_memory_risk and build_memory_audit (Section 16)."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from dgp_helpers import estimate_memory_risk, build_memory_audit


def test_estimate_memory_risk_low_bucket():
    """Small p/n should land in 'low' bucket."""
    result = estimate_memory_risk(p_total=8, n_train=50, n_test=10)
    assert result["memory_risk_bucket"] == "low"
    assert result["exceeds_default_gpu_guard"] is False


def test_estimate_memory_risk_formula():
    """Verify exact formula: 5 × context × p × 64 × 4."""
    p = 16
    n = 100
    context_size = 200
    feature_cap = 128
    hidden = 64
    result = estimate_memory_risk(
        p_total=p,
        n_train=n,
        n_test=10,
        context_size=context_size,
        feature_cap=feature_cap,
        hidden_estimate=hidden,
    )
    effective_p = min(p, feature_cap)
    effective_context = min(n, context_size)
    expected_bytes = 5 * effective_context * effective_p * hidden * 4
    assert result["estimated_h_tensor_bytes"] == expected_bytes


def test_estimate_memory_risk_feature_cap_clamps():
    """p > feature_cap should use feature_cap as effective_p."""
    result = estimate_memory_risk(p_total=512, n_train=100, n_test=20, feature_cap=128)
    assert result["estimated_p_total"] == 128


def test_estimate_memory_risk_context_cap_clamps():
    """n_train > context_size should use context_size as effective_context."""
    result = estimate_memory_risk(p_total=8, n_train=1000, n_test=100, context_size=200)
    assert result["estimated_context_rows"] == 200


def test_estimate_memory_risk_medium_bucket():
    """Construct parameters that land in the medium bucket (ratio in [0.50, 0.90))."""
    guard = 268_435_456
    # bytes = 5 * context * p * 64 * 4 = 5 * 200 * 900 * 64 * 4 = 230_400_000
    # ratio = 230_400_000 / 268_435_456 ~ 0.858 -> medium
    result = estimate_memory_risk(
        p_total=900, n_train=200, n_test=50,
        context_size=200, feature_cap=900,
        gpu_guard_bytes=guard, hidden_estimate=64,
    )
    assert result["memory_risk_bucket"] == "medium"


def test_estimate_memory_risk_high_bucket():
    """Construct parameters that land in the high bucket (ratio in [0.90, 1.00])."""
    guard = 268_435_456
    # bytes = 5 * 200 * 1000 * 64 * 4 = 256_000_000, ratio ~ 0.954
    result = estimate_memory_risk(
        p_total=1000, n_train=200, n_test=50,
        context_size=200, feature_cap=1000,
        gpu_guard_bytes=guard, hidden_estimate=64,
    )
    assert result["memory_risk_bucket"] == "high"


def test_estimate_memory_risk_exceeds_guard():
    """Tiny guard forces exceeds_default_guard bucket and flag."""
    result = estimate_memory_risk(
        p_total=64, n_train=200, n_test=50,
        gpu_guard_bytes=100,  # tiny guard
    )
    assert result["memory_risk_bucket"] == "exceeds_default_guard"
    assert result["exceeds_default_gpu_guard"] is True


def test_build_memory_audit_aggregates():
    """build_memory_audit should count buckets correctly across records."""
    records = [
        {"estimated_h_tensor_bytes": 1000, "memory_risk_bucket": "low"},
        {"estimated_h_tensor_bytes": 2000, "memory_risk_bucket": "medium"},
        {"estimated_h_tensor_bytes": 3000, "memory_risk_bucket": "low"},
    ]
    audit = build_memory_audit(records, 268_435_456, 200, 128, 128)
    assert audit["memory_risk_counts"]["low"] == 2
    assert audit["memory_risk_counts"]["medium"] == 1
    assert audit["memory_risk_counts"]["exceeds_default_guard"] == 0


def test_memory_audit_memory_passed_false():
    """Any exceeds_default_guard record should set memory_passed=False."""
    records = [
        {"estimated_h_tensor_bytes": 1e9, "memory_risk_bucket": "exceeds_default_guard"},
        {"estimated_h_tensor_bytes": 1000, "memory_risk_bucket": "low"},
    ]
    audit = build_memory_audit(records, 268_435_456, 200, 128, 128)
    assert audit["memory_passed"] is False
    assert audit["exceeds_default_guard_count"] == 1


def test_recommended_caps_are_positive():
    """All three recommended cap fields must be > 0."""
    result = estimate_memory_risk(
        p_total=2000, n_train=500, n_test=100,
        feature_cap=2000, context_size=500,
        gpu_guard_bytes=100,  # tiny guard to force exceeds
    )
    assert result["recommended_feature_cap"] > 0
    assert result["recommended_context_cap"] > 0
    assert result["recommended_test_batch_size"] > 0
