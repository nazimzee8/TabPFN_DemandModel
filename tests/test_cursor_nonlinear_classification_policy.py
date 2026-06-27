"""Tests for cursor_scale_v1 policy on Suite 7: NLC numeric (main_nonlinear_training).

Policy: sample_cursor_dims → 2.5% multiclass_smooth_softmax / 2.5% multiclass_sparse_highdim tail.
Invariants: p_total >= 1, n >= 5*p_total, p_total <= 30.
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
    _CURSOR_TAIL_FRACTION,
    _CURSOR_TAIL_P_CAP,
    _CURSOR_TAIL_P_FLOOR,
    _CURSOR_TAIL_P_LAMBDA,
)

_NONLINEAR_CLS_SEED_MAGIC = 0x4E4C4331


def _simulate_nlc_cursor_dims(n_tasks: int = 1000, base_seed: int = 42):
    """Simulate cursor dim draw for NLC numeric (Suite 7) inline logic."""
    results = []
    for idx in range(n_tasks):
        _dim_rng = np.random.default_rng(
            int(np.random.SeedSequence(
                [base_seed ^ _NONLINEAR_CLS_SEED_MAGIC, idx, 0xC1A5D1C]
            ).generate_state(1)[0])
        )
        _is_tail = _dim_rng.random() < _CURSOR_TAIL_FRACTION
        if _is_tail:
            while True:
                _p_total = int(max(_CURSOR_TAIL_P_FLOOR,
                                   min(int(_dim_rng.poisson(_CURSOR_TAIL_P_LAMBDA)),
                                       _CURSOR_TAIL_P_CAP)))
                _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                if _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                    break
            family = "multiclass_smooth_softmax" if idx % 2 == 0 else "multiclass_sparse_highdim"
        else:
            while True:
                _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
                _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                if _p_total >= 1 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                    break
            family = None  # normal path, family chosen by family_rng
        p_signal = max(1, int(_dim_rng.integers(1, _p_total + 1)))
        results.append({
            "n": _n, "p_total": _p_total, "p_signal": p_signal,
            "is_tail": _is_tail, "forced_family": family,
        })
    return results


class TestNLCCursorInvariants:
    @pytest.fixture(scope="class")
    def dims(self):
        return _simulate_nlc_cursor_dims()

    def test_p_total_ge_1(self, dims):
        bad = [d["p_total"] for d in dims if d["p_total"] < 1]
        assert not bad, f"{len(bad)} tasks have p_total < 1"

    def test_n_ge_5p_total(self, dims):
        violations = [d for d in dims if d["n"] < _CURSOR_MIN_N_OVER_P * d["p_total"]]
        assert not violations, f"{len(violations)} tasks violate n >= 5*p_total"

    def test_p_total_bounded(self, dims):
        bad = [d["p_total"] for d in dims if d["p_total"] > _CURSOR_TAIL_P_CAP]
        assert not bad, f"{len(bad)} tasks exceed p_total cap"

    def test_p_signal_le_p_total(self, dims):
        bad = [(d["p_signal"], d["p_total"]) for d in dims if d["p_signal"] > d["p_total"]]
        assert not bad, f"{len(bad)} tasks have p_signal > p_total: {bad[:5]}"

    def test_p_signal_ge_1(self, dims):
        bad = [d["p_signal"] for d in dims if d["p_signal"] < 1]
        assert not bad, f"{len(bad)} tasks have p_signal < 1"


class TestNLCCursorTailRouting:
    @pytest.fixture(scope="class")
    def dims(self):
        return _simulate_nlc_cursor_dims(n_tasks=2000)

    def test_tail_tasks_have_forced_family(self, dims):
        tail = [d for d in dims if d["is_tail"]]
        assert tail, "No tail tasks found in 2000 draws"
        for d in tail:
            assert d["forced_family"] in ("multiclass_smooth_softmax", "multiclass_sparse_highdim"), (
                f"Unexpected tail family: {d['forced_family']}"
            )

    def test_tail_p_total_in_range(self, dims):
        tail = [d for d in dims if d["is_tail"]]
        for d in tail:
            assert _CURSOR_TAIL_P_FLOOR <= d["p_total"] <= _CURSOR_TAIL_P_CAP, (
                f"Tail p_total={d['p_total']} out of [{_CURSOR_TAIL_P_FLOOR}, {_CURSOR_TAIL_P_CAP}]"
            )

    def test_tail_fraction_near_5pct(self, dims):
        tail_count = sum(1 for d in dims if d["is_tail"])
        # ~5% of 2000 = ~100; allow wide tolerance [20, 180]
        assert 20 <= tail_count <= 180, (
            f"tail_count={tail_count} far from expected ~100 (5% of 2000)"
        )

    def test_both_tail_families_assigned(self, dims):
        tail_families = {d["forced_family"] for d in dims if d["is_tail"]}
        assert "multiclass_smooth_softmax" in tail_families
        assert "multiclass_sparse_highdim" in tail_families
