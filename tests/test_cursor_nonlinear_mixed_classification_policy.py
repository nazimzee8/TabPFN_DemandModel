"""Tests for cursor_scale_v1 policy on Suite 8: NLC mixed (main_nonlinear_training).

Policy: sample_cursor_dims + budget carve.
Tail (5%): full 5% → mixed_categorical_nonlinear.
Invariants: p_total >= 2, p_num + p_cat == p_total, both >= 1, n >= 5*p_total.
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

_NONLINEAR_MIXED_CLS_SEED_MAGIC = 0x4E4D4331


def _simulate_nlc_mixed_cursor_dims(n_tasks: int = 1000, base_seed: int = 42):
    """Simulate cursor dim draw for NLC mixed (Suite 8) inline logic."""
    results = []
    for idx in range(n_tasks):
        _dim_rng = np.random.default_rng(
            int(np.random.SeedSequence(
                [base_seed ^ _NONLINEAR_MIXED_CLS_SEED_MAGIC, idx, 0x4E4D4331]
            ).generate_state(1)[0])
        )
        _is_tail = _dim_rng.random() < _CURSOR_TAIL_FRACTION
        if _is_tail:
            while True:
                _p_total = int(max(_CURSOR_TAIL_P_FLOOR,
                                   min(int(_dim_rng.poisson(_CURSOR_TAIL_P_LAMBDA)),
                                       _CURSOR_TAIL_P_CAP)))
                _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                if _p_total >= 2 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                    break
            forced_family = "mixed_categorical_nonlinear"
        else:
            while True:
                _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
                _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                if _p_total >= 2 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                    break
            forced_family = None
        _p_cat = max(1, min(int(_dim_rng.integers(1, 4)), _p_total - 1))
        p_signal = _p_total - _p_cat  # p_num (numeric signal cols)
        results.append({
            "n": _n, "p_total": _p_total, "p_num": p_signal, "p_cat": _p_cat,
            "is_tail": _is_tail, "forced_family": forced_family,
        })
    return results


class TestNLCMixedCursorInvariants:
    @pytest.fixture(scope="class")
    def dims(self):
        return _simulate_nlc_mixed_cursor_dims()

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


class TestNLCMixedCursorTailRouting:
    @pytest.fixture(scope="class")
    def dims(self):
        return _simulate_nlc_mixed_cursor_dims(n_tasks=2000)

    def test_tail_tasks_route_to_mixed_categorical(self, dims):
        tail = [d for d in dims if d["is_tail"]]
        assert tail, "No tail tasks found in 2000 draws"
        for d in tail:
            assert d["forced_family"] == "mixed_categorical_nonlinear", (
                f"Expected mixed_categorical_nonlinear for tail; got {d['forced_family']}"
            )

    def test_tail_p_total_in_range(self, dims):
        tail = [d for d in dims if d["is_tail"]]
        for d in tail:
            assert _CURSOR_TAIL_P_FLOOR <= d["p_total"] <= _CURSOR_TAIL_P_CAP, (
                f"Tail p_total={d['p_total']} out of [{_CURSOR_TAIL_P_FLOOR}, {_CURSOR_TAIL_P_CAP}]"
            )

    def test_tail_fraction_near_5pct(self, dims):
        tail_count = sum(1 for d in dims if d["is_tail"])
        assert 15 <= tail_count <= 200, (
            f"tail_count={tail_count} far from expected ~100 (5% of 2000)"
        )

    def test_non_tail_p_total_ge_2(self, dims):
        non_tail = [d for d in dims if not d["is_tail"]]
        bad = [d["p_total"] for d in non_tail if d["p_total"] < 2]
        assert not bad, f"{len(bad)} non-tail tasks have p_total < 2"
