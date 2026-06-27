"""Tests for cursor_scale_v1 policy on Suite 2: linear-regression MIXED suite.

Covers:
  - Invariants (p_total >= 2, p_num >= 1, p_cat >= 1, n >= 5*p_total, p_total <= 30)
  - Mixed budget carve (p_num + p_cat == p_total)
  - Regime coverage and post-hoc rank-based assignment
  - Audit dict completeness and dimension_policy block
  - Determinism (two runs same seed → identical output)
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
    REGIME_WEIGHTS,
    _CURSOR_MIN_N_OVER_P,
    _CURSOR_TAIL_P_CAP,
    allocate_mixed_regression_tasks_cursor,
)

PROFILE = "linear_regression_mixed_categorical_stat_aware"
ALL_REGIMES = list(REGIME_WEIGHTS[PROFILE].keys())
BASE_SEED = 99
N_TOTAL = 500


def _run(seed: int = BASE_SEED, n_total: int = N_TOTAL) -> tuple[list[dict], dict]:
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    splits = {"train": n_train, "val": n_val, "test": n_total - n_train - n_val}
    return allocate_mixed_regression_tasks_cursor(
        splits, PROFILE, seed, allocation_mode="weighted_quota", min_regime_count=1
    )


class TestMixedRegressionCursorInvariants:
    @pytest.fixture(scope="class")
    def run_result(self):
        return _run()

    def test_total_length(self, run_result):
        assignments, _ = run_result
        assert len(assignments) == N_TOTAL

    def test_p_total_ge_2(self, run_result):
        assignments, _ = run_result
        bad = [a["p_total"] for a in assignments if a["p_total"] < 2]
        assert not bad, f"{len(bad)} tasks have p_total < 2"

    def test_p_num_ge_1(self, run_result):
        assignments, _ = run_result
        bad = [a["p_num"] for a in assignments if a["p_num"] < 1]
        assert not bad, f"{len(bad)} tasks have p_num < 1"

    def test_p_cat_ge_1(self, run_result):
        assignments, _ = run_result
        bad = [a["p_cat"] for a in assignments if a["p_cat"] < 1]
        assert not bad, f"{len(bad)} tasks have p_cat < 1"

    def test_p_num_plus_p_cat_equals_p_total(self, run_result):
        assignments, _ = run_result
        bad = [
            (a["p_num"], a["p_cat"], a["p_total"])
            for a in assignments
            if a["p_num"] + a["p_cat"] != a["p_total"]
        ]
        assert not bad, f"p_num + p_cat != p_total in {len(bad)} tasks: {bad[:5]}"

    def test_n_ge_5p_total(self, run_result):
        assignments, _ = run_result
        violations = [
            (a["n"], a["p_total"])
            for a in assignments
            if a["n"] < _CURSOR_MIN_N_OVER_P * a["p_total"]
        ]
        assert not violations, f"{len(violations)} tasks violate n >= 5*p_total"

    def test_p_total_bounded_by_cap(self, run_result):
        assignments, _ = run_result
        bad = [a["p_total"] for a in assignments if a["p_total"] > _CURSOR_TAIL_P_CAP]
        assert not bad, f"{len(bad)} tasks exceed p_total cap {_CURSOR_TAIL_P_CAP}"

    def test_cursor_dims_flag_set(self, run_result):
        assignments, _ = run_result
        bad = [i for i, a in enumerate(assignments) if not a.get("cursor_dims")]
        assert not bad, f"{len(bad)} tasks missing cursor_dims=True"

    def test_active_s_ge_1(self, run_result):
        assignments, _ = run_result
        bad = [a.get("active_s", 0) for a in assignments if a.get("active_s", 0) < 1]
        assert not bad, f"{len(bad)} tasks have active_s < 1"


class TestMixedRegressionCursorAudit:
    @pytest.fixture(scope="class")
    def run_result(self):
        return _run()

    def test_audit_n_ge_5p(self, run_result):
        _, audit = run_result
        assert audit["all_tasks_satisfy_n_ge_5p"] is True

    def test_audit_p_num_ge_1(self, run_result):
        _, audit = run_result
        assert audit["all_tasks_satisfy_p_num_ge_1"] is True

    def test_audit_p_cat_ge_1(self, run_result):
        _, audit = run_result
        assert audit["all_tasks_satisfy_p_cat_ge_1"] is True

    def test_audit_p_total_ge_2(self, run_result):
        _, audit = run_result
        assert audit["all_tasks_satisfy_p_total_ge_2"] is True

    def test_audit_has_dimension_policy(self, run_result):
        _, audit = run_result
        dp = audit["dimension_policy"]
        assert dp["dimension_policy_name"] == "cursor_scale_v1"
        assert dp["p_total_definition"] == "p_num + p_cat (categorical carved from same Poisson budget)"
        assert dp["p_total_min"] == 2

    def test_audit_has_summary_stats(self, run_result):
        _, audit = run_result
        assert "realized_p_total_summary" in audit
        assert "realized_n_summary" in audit
        assert "realized_p_num_summary" in audit
        assert "realized_p_cat_summary" in audit

    def test_audit_cursor_flag(self, run_result):
        _, audit = run_result
        assert audit["cursor_dims"] is True


class TestMixedRegressionCursorRegimeCoverage:
    @pytest.fixture(scope="class")
    def run_result(self):
        return _run(n_total=1000)

    def test_all_regimes_represented(self, run_result):
        assignments, _ = run_result
        present = {a["prior_regime"] for a in assignments}
        missing = set(ALL_REGIMES) - present
        assert not missing, f"Regimes missing: {missing}"

    def test_wide_tail_present(self, run_result):
        assignments, _ = run_result
        # E + F are assigned to the upper-quartile p_total tasks
        ef_count = sum(
            1 for a in assignments
            if a["prior_regime"] in ("E_high_dim_dense", "F_high_dim_sparse")
        )
        assert ef_count > 0, "No E/F (wide-tail) tasks found"


class TestMixedRegressionCursorDeterminism:
    def test_same_seed_same_output(self):
        a1, d1 = _run(seed=7)
        a2, d2 = _run(seed=7)
        assert len(a1) == len(a2)
        for i, (x, y) in enumerate(zip(a1, a2)):
            assert x["n"] == y["n"], f"Task {i}: n differs"
            assert x["p_total"] == y["p_total"], f"Task {i}: p_total differs"
            assert x["prior_regime"] == y["prior_regime"], f"Task {i}: regime differs"
        assert d1["all_tasks_satisfy_n_ge_5p"] == d2["all_tasks_satisfy_n_ge_5p"]

    def test_different_seeds_differ(self):
        a1, _ = _run(seed=1)
        a2, _ = _run(seed=2)
        n_vals_1 = [a["n"] for a in a1]
        n_vals_2 = [a["n"] for a in a2]
        assert n_vals_1 != n_vals_2, "Different seeds produced identical n sequences"
