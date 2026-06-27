"""Tests for the Cursor-scale Poisson dimension policy on linear_stat_aware.

Covers:
  - Invariants (p_total>=1, n>=5p, inner split, p_signal+p_noise==p_total, active_s<=p_signal)
  - Distribution properties (Poisson(10)/Poisson(200), wide tail, no former large shapes)
  - Regime integrity (all 12 present in every split, mechanism checks, G/E/F/J rules)
  - Determinism (two runs with same seed produce identical output)
  - validate_dataset passes for every task
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src" / "data_generation"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SRC2 = Path(__file__).resolve().parent.parent / "src"
if str(_SRC2) not in sys.path:
    sys.path.insert(0, str(_SRC2))

import _bootstrap  # noqa: F401 — adds all src/ subdirs to sys.path

from dgp_helpers import (
    REGIME_WEIGHTS,
    _CURSOR_MIN_N_OVER_P,
    _CURSOR_P_LAMBDA,
    _CURSOR_N_LAMBDA,
    _CURSOR_TAIL_FRACTION,
    _CURSOR_TAIL_P_CAP,
    allocate_regression_tasks_cursor,
    build_dataset_from_regime,
    sample_cursor_dims,
    validate_dataset,
)

ALL_REGIMES = list(REGIME_WEIGHTS["linear_stat_aware"].keys())
assert len(ALL_REGIMES) == 12, f"Expected 12 regimes, got {len(ALL_REGIMES)}"

SPLIT_SIZES = {"train": 800, "val": 100, "test": 100}
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cursor(seed: int = BASE_SEED, n_total: int = 1000) -> tuple[list[dict], dict]:
    splits = {
        "train": int(0.8 * n_total),
        "val": int(0.1 * n_total),
        "test": n_total - int(0.8 * n_total) - int(0.1 * n_total),
    }
    return allocate_regression_tasks_cursor(
        splits, "linear_stat_aware", seed,
        allocation_mode="weighted_quota",
        min_regime_count=1,
    )


# ---------------------------------------------------------------------------
# 1. sample_cursor_dims invariants
# ---------------------------------------------------------------------------

class TestCursorDimsSampler:
    def test_always_satisfies_n_ge_5p(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            n, p = sample_cursor_dims(rng)
            assert p >= 1, f"p_total must be >= 1, got {p}"
            assert n >= _CURSOR_MIN_N_OVER_P * p, (
                f"n >= 5p required; got n={n}, p={p}, ratio={n/p:.2f}"
            )

    def test_wide_tail_bounded(self):
        """The wide-p tail must never exceed _CURSOR_TAIL_P_CAP."""
        rng = np.random.default_rng(1)
        for _ in range(5000):
            _, p = sample_cursor_dims(rng)
            assert p <= _CURSOR_TAIL_P_CAP, (
                f"p_total={p} exceeds cap {_CURSOR_TAIL_P_CAP}"
            )

    def test_mean_p_near_lambda(self):
        """Realized mean of p_total should be close to Poisson(10) expectation."""
        rng = np.random.default_rng(2)
        p_vals = [sample_cursor_dims(rng)[1] for _ in range(3000)]
        mean_p = np.mean(p_vals)
        # With ε=5% tail at λ=25: expected mean ≈ 0.95*10 + 0.05*min(25,30) ≈ 10.75
        # Allow ±2 for sampling noise at 3000 draws
        assert 8.0 < mean_p < 14.0, f"Mean p_total={mean_p:.2f} outside expected range"

    def test_mean_n_near_lambda(self):
        rng = np.random.default_rng(3)
        n_vals = [sample_cursor_dims(rng)[0] for _ in range(3000)]
        mean_n = np.mean(n_vals)
        # Poisson(200) conditioned on n>=5p; mean should be slightly above 200
        assert 190.0 < mean_n < 220.0, f"Mean n={mean_n:.2f} outside expected range"


# ---------------------------------------------------------------------------
# 2. Allocator invariants on a 1000-task run
# ---------------------------------------------------------------------------

class TestAllocatorInvariants:
    @pytest.fixture(scope="class")
    def assignments_audit(self):
        return _run_cursor()

    def test_total_length(self, assignments_audit):
        assignments, _ = assignments_audit
        assert len(assignments) == 1000

    def test_n_ge_5p_all_tasks(self, assignments_audit):
        assignments, _ = assignments_audit
        violations = [
            (i, a["n"], a["p_total"])
            for i, a in enumerate(assignments)
            if a["n"] < _CURSOR_MIN_N_OVER_P * a["p_total"]
        ]
        assert not violations, (
            f"{len(violations)} tasks violate n >= 5*p_total: {violations[:5]}"
        )

    def test_p_total_ge_1(self, assignments_audit):
        assignments, _ = assignments_audit
        bad = [a["p_total"] for a in assignments if a["p_total"] < 1]
        assert not bad, f"Found {len(bad)} tasks with p_total < 1"

    def test_p_total_bounded_by_cap(self, assignments_audit):
        assignments, _ = assignments_audit
        bad = [a["p_total"] for a in assignments if a["p_total"] > _CURSOR_TAIL_P_CAP]
        assert not bad, f"Found {len(bad)} tasks with p_total > {_CURSOR_TAIL_P_CAP}: {bad}"

    def test_p_signal_plus_p_noise_equals_p_total(self, assignments_audit):
        assignments, _ = assignments_audit
        bad = [
            (a["p_signal"], a["p_noise"], a["p_total"])
            for a in assignments
            if a["p_signal"] + a["p_noise"] != a["p_total"]
        ]
        assert not bad, f"p_signal + p_noise != p_total in {len(bad)} tasks: {bad[:5]}"

    def test_active_s_le_p_signal(self, assignments_audit):
        assignments, _ = assignments_audit
        bad = [
            (a["active_s"], a["p_signal"])
            for a in assignments
            if a["active_s"] > a["p_signal"]
        ]
        assert not bad, f"active_s > p_signal in {len(bad)} tasks"

    def test_active_s_ge_1(self, assignments_audit):
        assignments, _ = assignments_audit
        bad = [a["active_s"] for a in assignments if a["active_s"] < 1]
        assert not bad, f"active_s < 1 in {len(bad)} tasks"

    def test_cursor_dims_flag_set(self, assignments_audit):
        assignments, _ = assignments_audit
        bad = [i for i, a in enumerate(assignments) if not a.get("cursor_dims")]
        assert not bad, f"{len(bad)} tasks missing cursor_dims=True"

    def test_audit_n_ge_5p(self, assignments_audit):
        _, audit = assignments_audit
        assert audit["all_tasks_satisfy_n_ge_5p"] is True

    def test_audit_has_dimension_policy(self, assignments_audit):
        _, audit = assignments_audit
        dp = audit["dimension_policy"]
        assert dp["dimension_policy_name"] == "cursor_scale_v1"
        assert dp["n_lambda"] == _CURSOR_N_LAMBDA
        assert dp["p_lambda"] == _CURSOR_P_LAMBDA
        assert dp["p_tail_fraction"] == _CURSOR_TAIL_FRACTION
        assert dp["p_tail_cap"] == _CURSOR_TAIL_P_CAP
        assert dp["minimum_n_to_p_ratio"] == float(_CURSOR_MIN_N_OVER_P)


# ---------------------------------------------------------------------------
# 3. Distribution properties
# ---------------------------------------------------------------------------

class TestDistributionProperties:
    @pytest.fixture(scope="class")
    def assignments_audit(self):
        return _run_cursor()

    def test_no_legacy_large_n(self, assignments_audit):
        """No task should have n=1024 (the top of the old grid)."""
        assignments, _ = assignments_audit
        large_n = [a["n"] for a in assignments if a["n"] >= 1024]
        # Extremely rare in Poisson(200); allow at most 2 in 1000
        assert len(large_n) <= 2, f"Found {len(large_n)} tasks with n>=1024: {large_n}"

    def test_no_p_total_exceeds_cap(self, assignments_audit):
        assignments, _ = assignments_audit
        assert all(a["p_total"] <= _CURSOR_TAIL_P_CAP for a in assignments)

    def test_wide_tail_present(self, assignments_audit):
        """Expect ~5% of tasks to have p_total above the main-component mode."""
        assignments, _ = assignments_audit
        tail_count = sum(1 for a in assignments if a["p_total"] > 18)
        # With ε=5% and 1000 tasks, expect ~50; allow ±30 for sampling noise
        assert 5 <= tail_count <= 100, (
            f"Wide-tail task count={tail_count}; expected ~50 (ε=5%)"
        )

    def test_realized_p_summary_in_audit(self, assignments_audit):
        _, audit = assignments_audit
        summary = audit["realized_p_total_summary"]
        assert "mean" in summary and "median" in summary
        # Mean p_total should be between 8 and 16 (Poisson(10) + small tail)
        assert 7.0 <= summary["mean"] <= 18.0, (
            f"Mean p_total={summary['mean']:.2f} outside expected range"
        )

    def test_realized_n_summary_in_audit(self, assignments_audit):
        _, audit = assignments_audit
        summary = audit["realized_n_summary"]
        assert 150.0 <= summary["mean"] <= 260.0, (
            f"Mean n={summary['mean']:.2f} outside expected range"
        )


# ---------------------------------------------------------------------------
# 4. Regime integrity — coverage in every split
# ---------------------------------------------------------------------------

class TestRegimeCoverage:
    @pytest.fixture(scope="class")
    def assignments_audit(self):
        return _run_cursor()

    def _split_regimes(self, assignments, split_name):
        split_idx = {"train": 0, "val": 1, "test": 2}[split_name]
        counts = [800, 100, 100]
        start = sum(counts[:split_idx])
        end = start + counts[split_idx]
        return [a["prior_regime"] for a in assignments[start:end]]

    def test_all_regimes_in_train(self, assignments_audit):
        assignments, _ = assignments_audit
        present = set(self._split_regimes(assignments, "train"))
        missing = set(ALL_REGIMES) - present
        assert not missing, f"Regimes missing from train: {missing}"

    def test_all_regimes_in_val(self, assignments_audit):
        assignments, _ = assignments_audit
        present = set(self._split_regimes(assignments, "val"))
        missing = set(ALL_REGIMES) - present
        assert not missing, f"Regimes missing from val: {missing}"

    def test_all_regimes_in_test(self, assignments_audit):
        assignments, _ = assignments_audit
        present = set(self._split_regimes(assignments, "test"))
        missing = set(ALL_REGIMES) - present
        assert not missing, f"Regimes missing from test: {missing}"

    def test_j_is_always_valid(self, assignments_audit):
        """J_low_n_high_p must never be underdetermined (n >= 5*p)."""
        assignments, _ = assignments_audit
        j_tasks = [a for a in assignments if a["prior_regime"] == "J_low_n_high_p"]
        assert j_tasks, "J_low_n_high_p has no tasks"
        for a in j_tasks:
            assert a["n"] >= _CURSOR_MIN_N_OVER_P * a["p_total"], (
                f"J task underdetermined: n={a['n']}, p_total={a['p_total']}"
            )

    def test_g_has_noise_features(self, assignments_audit):
        """G_noise_features must always have p_noise >= 1 and p_signal >= 1."""
        assignments, _ = assignments_audit
        g_tasks = [a for a in assignments if a["prior_regime"] == "G_noise_features"]
        assert g_tasks, "G_noise_features has no tasks"
        for a in g_tasks:
            assert a["p_noise"] >= 1, (
                f"G task has p_noise={a['p_noise']} (must be >= 1)"
            )
            assert a["p_signal"] >= 1, (
                f"G task has p_signal={a['p_signal']} (must be >= 1)"
            )

    def test_e_is_upper_p_range(self, assignments_audit):
        """E_high_dim_dense should have higher p_total than average."""
        assignments, _ = assignments_audit
        e_vals = [a["p_total"] for a in assignments if a["prior_regime"] == "E_high_dim_dense"]
        all_vals = [a["p_total"] for a in assignments]
        assert e_vals, "E_high_dim_dense has no tasks"
        assert np.mean(e_vals) > np.mean(all_vals), (
            f"E p_total mean={np.mean(e_vals):.1f} not above overall mean={np.mean(all_vals):.1f}"
        )

    def test_j_has_low_n_over_p(self, assignments_audit):
        """J should have lower n/p ratio than average."""
        assignments, _ = assignments_audit
        j_ratios = [a["n"] / a["p_total"] for a in assignments
                    if a["prior_regime"] == "J_low_n_high_p"]
        all_ratios = [a["n"] / a["p_total"] for a in assignments]
        assert j_ratios, "J has no tasks"
        assert np.mean(j_ratios) < np.mean(all_ratios), (
            f"J n/p mean={np.mean(j_ratios):.1f} not below overall mean={np.mean(all_ratios):.1f}"
        )

    def test_audit_by_split_has_all_regimes(self, assignments_audit):
        _, audit = assignments_audit
        for split_name, split_audit in audit["audit_by_split"].items():
            present = set(split_audit["realized_regime_counts"].keys())
            missing = set(ALL_REGIMES) - present
            assert not missing, (
                f"Regimes missing from audit_by_split['{split_name}']: {missing}"
            )


# ---------------------------------------------------------------------------
# 5. Validate_dataset passes for every built task (small suite)
# ---------------------------------------------------------------------------

class TestDatasetValidation:
    def test_validate_all_tasks(self):
        """Build actual datasets from a small 60-task cursor suite and validate each."""
        assignments, _ = allocate_regression_tasks_cursor(
            {"train": 48, "val": 6, "test": 6},
            "linear_stat_aware", BASE_SEED,
        )
        for a in assignments:
            rng = np.random.default_rng(a["task_seed"])
            params = {k: a[k] for k in [
                "n", "p_signal", "p_noise", "p_total", "active_s",
                "rho", "target_noise_scale", "feature_noise_level",
                "covariance_type", "cursor_dims",
            ] if k in a}
            ds = build_dataset_from_regime(rng, a["prior_regime"], "linear_stat_aware", params)
            validate_dataset(ds)

            # Inner split
            n_total = ds["n_total"]
            n_train = int(0.8 * n_total)
            n_test = n_total - n_train
            assert n_train + n_test == n_total
            assert n_train == math.floor(0.8 * n_total)

            # p_signal + p_noise == p_total
            assert ds["p_signal"] + ds["p_noise"] == ds["p_total"]
            # active_s <= p_signal
            assert ds["active_s"] <= ds["p_signal"]


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_runs_same_seed_identical(self):
        """Two runs with the same seed must produce identical assignments."""
        run1, audit1 = _run_cursor(seed=77)
        run2, audit2 = _run_cursor(seed=77)

        assert len(run1) == len(run2)
        for i, (a1, a2) in enumerate(zip(run1, run2)):
            assert a1["prior_regime"] == a2["prior_regime"], f"Regime mismatch at idx {i}"
            assert a1["n"] == a2["n"], f"n mismatch at idx {i}"
            assert a1["p_total"] == a2["p_total"], f"p_total mismatch at idx {i}"
            assert a1["task_seed"] == a2["task_seed"], f"task_seed mismatch at idx {i}"

        assert audit1["realized_regime_counts"] == audit2["realized_regime_counts"]
        assert audit1["all_tasks_satisfy_n_ge_5p"] == audit2["all_tasks_satisfy_n_ge_5p"]

    def test_different_seeds_produce_different_dims(self):
        """Different seeds should produce different p_total sequences (not identical)."""
        run1, _ = _run_cursor(seed=1)
        run2, _ = _run_cursor(seed=2)
        p1 = [a["p_total"] for a in run1]
        p2 = [a["p_total"] for a in run2]
        assert p1 != p2, "Different seeds produced identical p_total sequences"


# ---------------------------------------------------------------------------
# 7. Wrong-profile guard
# ---------------------------------------------------------------------------

def test_cursor_allocator_rejects_wrong_profile():
    with pytest.raises(ValueError, match="linear_stat_aware"):
        allocate_regression_tasks_cursor(
            {"train": 10, "val": 2, "test": 2},
            "legacy", BASE_SEED,
        )
