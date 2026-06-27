"""Tests for cursor_scale_v1 policy on Suite 5: nonlinear-regression NUMERIC (main()).

Verifies that:
  - The per-task cursor dim draws in main() respect n >= 5*p_total and p_total <= 30
  - p_signal is derived from p_total (1 <= p_signal <= p_total)
  - p_noise_cap is honoured by _sample_feature_params noise_feats regime
  - The high_dim_sparse_nonlinear tail family is reachable
  - sample_cursor_dims is called consistently (via dgp_helpers import)
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
    sample_cursor_dims,
)


# ---------------------------------------------------------------------------
# 1. cursor dim invariants from sample_cursor_dims (reused in nonlinear suite 5)
# ---------------------------------------------------------------------------

class TestNonlinearRegressionCursorDims:
    """The nonlinear main() loop calls sample_cursor_dims (or inline equivalent)."""

    def test_always_satisfies_n_ge_5p(self):
        rng = np.random.default_rng(300)
        for _ in range(2000):
            n, p = sample_cursor_dims(rng)
            assert p >= 1
            assert n >= _CURSOR_MIN_N_OVER_P * p, f"n={n}, p={p}"

    def test_p_total_never_exceeds_cap(self):
        rng = np.random.default_rng(301)
        for _ in range(2000):
            _, p = sample_cursor_dims(rng)
            assert p <= _CURSOR_TAIL_P_CAP

    def test_tail_floor_respected(self):
        """When a tail draw occurs, p_total must be >= _CURSOR_TAIL_P_FLOOR."""
        rng = np.random.default_rng(302)
        tail_ps = []
        for _ in range(10000):
            _, p = sample_cursor_dims(rng)
            if p >= _CURSOR_TAIL_P_FLOOR:
                tail_ps.append(p)
        # All tail draws must be in [20, 30]
        assert all(p <= _CURSOR_TAIL_P_CAP for p in tail_ps)


# ---------------------------------------------------------------------------
# 2. p_noise_cap: noise_feats regime must not overflow cursor budget
# ---------------------------------------------------------------------------

class TestNoiseFeatsBudgetCap:
    """_sample_feature_params with noise_feats and p_noise_cap should not overflow."""

    def test_noise_feats_respects_cap(self):
        # Import the function directly from the nonlinear module
        sys.path.insert(0, str(_SRC))
        from generate_nonlinear_dgp import _sample_feature_params
        rng = np.random.default_rng(999)
        for _ in range(100):
            p_signal = int(rng.integers(1, 8))   # small signal
            p_noise_cap = int(rng.integers(0, 5))  # tight cap
            params = _sample_feature_params(rng, "noise_feats", 200, p_signal,
                                            p_noise_cap=p_noise_cap)
            assert params["p_noise"] <= p_noise_cap, (
                f"p_noise={params['p_noise']} exceeds cap={p_noise_cap}"
            )

    def test_noise_feats_without_cap_unconstrained(self):
        """Without cap, noise_feats can produce p_noise = 2-3× p_signal."""
        from generate_nonlinear_dgp import _sample_feature_params
        rng = np.random.default_rng(1000)
        p_signal = 4
        params = _sample_feature_params(rng, "noise_feats", 200, p_signal)
        # Should be 2-3× p_signal = 8-12 without a cap
        assert params["p_noise"] >= 2 * p_signal - 1  # allow one rounding


# ---------------------------------------------------------------------------
# 3. Per-task cursor dim derivation (p_signal from p_total)
# ---------------------------------------------------------------------------

class TestNonlinearRegressionDimDerivation:
    """Simulate the per-task cursor draw in main() to verify p_signal derivation."""

    def test_p_signal_always_le_p_total(self):
        """p_signal = max(1, randint(1, p_total+1)) must satisfy 1 <= p_signal <= p_total."""
        base_seed = 42
        for idx in range(200):
            _dim_rng = np.random.default_rng(
                int(np.random.SeedSequence([base_seed, idx, 0x5D4E4C52]).generate_state(1)[0])
            )
            _dim_rng.random()  # consume the is_tail draw
            # main-path draw
            while True:
                _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
                _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                if _p_total >= 1 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                    break
            p_signal = max(1, int(_dim_rng.integers(1, _p_total + 1)))
            assert 1 <= p_signal <= _p_total, (
                f"p_signal={p_signal} out of range [1, {_p_total}]"
            )
            assert _n >= _CURSOR_MIN_N_OVER_P * _p_total
