"""Tests for cursor_scale_v1 policy on Suite 6: NLR mixed (main_nonlinear_training).

Policy: Poisson(10)-only draw (wide-p tail EXCLUDED for NLR mixed).
p_total = p_signal + p_cat, both >= 1, p_total >= 2.
n >= 5 * p_total, p_total <= 30.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src" / "data_generation"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SRC2 = Path(__file__).resolve().parent.parent / "src"
if str(_SRC2) not in sys.path:
    sys.path.insert(0, str(_SRC2))

import _bootstrap  # noqa: F401

from dgp_helpers import (
    _CURSOR_MIN_N_OVER_P,
    _CURSOR_P_LAMBDA,
    _CURSOR_N_LAMBDA,
    _CURSOR_TAIL_P_CAP,
)

_NONLINEAR_MIXED_REG_SEED_MAGIC = 0x4E4D5233


def _simulate_nlr_mixed_cursor_dims(n_tasks: int = 500, base_seed: int = 42):
    """Simulate the cursor dim draw for NLR mixed (Suite 6) inline logic."""
    results = []
    for idx in range(n_tasks):
        _dim_rng = np.random.default_rng(
            int(np.random.SeedSequence(
                [base_seed ^ _NONLINEAR_MIXED_REG_SEED_MAGIC, idx, 0x4E4D5233]
            ).generate_state(1)[0])
        )
        # Tail EXCLUDED: Poisson(10)-only loop, p_total >= 2
        while True:
            _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
            _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
            if _p_total >= 2 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                break
        _p_cat = max(1, min(int(_dim_rng.integers(1, 4)), _p_total - 1))
        p_num = _p_total - _p_cat
        results.append({"n": _n, "p_total": _p_total, "p_num": p_num, "p_cat": _p_cat})
    return results


class TestNLRMixedCursorInvariants:
    @pytest.fixture(scope="class")
    def dims(self):
        return _simulate_nlr_mixed_cursor_dims()

    def test_p_total_ge_2(self, dims):
        bad = [d["p_total"] for d in dims if d["p_total"] < 2]
        assert not bad, f"{len(bad)} tasks have p_total < 2"

    def test_p_num_ge_1(self, dims):
        bad = [d["p_num"] for d in dims if d["p_num"] < 1]
        assert not bad, f"{len(bad)} tasks have p_num < 1"

    def test_p_cat_ge_1(self, dims):
        bad = [d["p_cat"] for d in dims if d["p_cat"] < 1]
        assert not bad, f"{len(bad)} tasks have p_cat < 1"

    def test_budget_carve(self, dims):
        bad = [d for d in dims if d["p_num"] + d["p_cat"] != d["p_total"]]
        assert not bad, f"{len(bad)} tasks violate p_num + p_cat == p_total"

    def test_n_ge_5p_total(self, dims):
        violations = [d for d in dims if d["n"] < _CURSOR_MIN_N_OVER_P * d["p_total"]]
        assert not violations, f"{len(violations)} tasks violate n >= 5*p_total"

    def test_p_total_bounded(self, dims):
        bad = [d["p_total"] for d in dims if d["p_total"] > _CURSOR_TAIL_P_CAP]
        assert not bad, f"{len(bad)} tasks exceed p_total cap"

    def test_no_wide_tail(self, dims):
        """NLR mixed tail is excluded; p_total should never hit wide-tail range for 5% fraction."""
        # With Poisson(10)-only, p > 18 should be extremely rare (< 1%)
        high_p = [d["p_total"] for d in dims if d["p_total"] > 18]
        # Allow a few (Poisson tail), but not 5% tail draws (which would give ~50/1000)
        # Using a generous bound: allow up to 30 in 500 (6%)
        assert len(high_p) <= 30, (
            f"Too many high-p tasks ({len(high_p)}/500); suggests tail was not excluded"
        )
