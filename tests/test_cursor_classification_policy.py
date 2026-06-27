"""Tests for cursor_scale_v1 policy on Suite 3: linear-classification NUMERIC suite.

Covers:
  - Invariants (p_total >= 1, n >= 5*p_total, p_total <= 30)
  - Regime coverage: all linear classification regimes present per split
  - J regime: cursor J is never n < p (always n >= 5p)
  - E/F softmax tail: upper-quartile p tasks
  - Audit dict completeness
  - Determinism
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
    allocate_classification_tasks_cursor,
)

PROFILE = "linear_classification_stat_aware"
ALL_REGIMES = list(REGIME_WEIGHTS[PROFILE].keys())
BASE_SEED = 77
N_TOTAL = 500


def _run(seed: int = BASE_SEED, n_total: int = N_TOTAL) -> tuple[list[dict], dict]:
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    splits = {"train": n_train, "val": n_val, "test": n_total - n_train - n_val}
    return allocate_classification_tasks_cursor(
        splits, PROFILE, seed, allocation_mode="weighted_quota", min_regime_count=1
    )


class TestClassificationCursorInvariants:
    @pytest.fixture(scope="class")
    def run_result(self):
        return _run()

    def test_total_length(self, run_result):
        assignments, _ = run_result
        assert len(assignments) == N_TOTAL

    def test_p_total_ge_1(self, run_result):
        assignments, _ = run_result
        bad = [a["p_total"] for a in assignments if a["p_total"] < 1]
        assert not bad, f"{len(bad)} tasks have p_total < 1"

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
        assert not bad, f"{len(bad)} tasks exceed p_total cap"

    def test_cursor_dims_flag_set(self, run_result):
        assignments, _ = run_result
        bad = [i for i, a in enumerate(assignments) if not a.get("cursor_dims")]
        assert not bad, f"{len(bad)} tasks missing cursor_dims=True"

    def test_j_regime_never_underdetermined(self, run_result):
        """J_low_n_high_p_classification must always have n >= 5*p_total under cursor."""
        assignments, _ = run_result
        j_tasks = [a for a in assignments if a["prior_regime"] == "J_low_n_high_p_classification"]
        for a in j_tasks:
            assert a["n"] >= _CURSOR_MIN_N_OVER_P * a["p_total"], (
                f"cursor J task violates n >= 5p: n={a['n']}, p={a['p_total']}"
            )

    def test_active_s_ge_1(self, run_result):
        assignments, _ = run_result
        bad = [a.get("active_s", 0) for a in assignments if a.get("active_s", 0) < 1]
        assert not bad, f"{len(bad)} tasks have active_s < 1"

    def test_p_signal_plus_p_noise_equals_p_total(self, run_result):
        assignments, _ = run_result
        bad = [
            (a["p_signal"], a["p_noise"], a["p_total"])
            for a in assignments
            if a["p_signal"] + a["p_noise"] != a["p_total"]
        ]
        assert not bad, f"p_signal + p_noise != p_total in {len(bad)} tasks: {bad[:5]}"


class TestClassificationCursorAudit:
    @pytest.fixture(scope="class")
    def run_result(self):
        return _run()

    def test_audit_n_ge_5p(self, run_result):
        _, audit = run_result
        assert audit["all_tasks_satisfy_n_ge_5p"] is True

    def test_audit_has_dimension_policy(self, run_result):
        _, audit = run_result
        dp = audit["dimension_policy"]
        assert dp["dimension_policy_name"] == "cursor_scale_v1"
        assert "p_tail_allocation" in dp
        assert dp["p_tail_allocation"]["E_high_dim_dense_softmax"] == 0.025
        assert dp["p_tail_allocation"]["F_high_dim_sparse_softmax"] == 0.025

    def test_audit_has_summary_stats(self, run_result):
        _, audit = run_result
        assert "realized_p_total_summary" in audit
        assert "realized_n_summary" in audit

    def test_audit_cursor_flag(self, run_result):
        _, audit = run_result
        assert audit["cursor_dims"] is True


class TestClassificationCursorRegimeCoverage:
    @pytest.fixture(scope="class")
    def run_result(self):
        return _run(n_total=1000)

    def test_all_regimes_represented(self, run_result):
        assignments, _ = run_result
        present = {a["prior_regime"] for a in assignments}
        missing = set(ALL_REGIMES) - present
        assert not missing, f"Regimes missing: {missing}"

    def test_ef_softmax_present(self, run_result):
        assignments, _ = run_result
        ef = {a["prior_regime"] for a in assignments if a["prior_regime"] in (
            "E_high_dim_dense_softmax", "F_high_dim_sparse_softmax"
        )}
        assert len(ef) == 2, f"Missing E or F softmax tail: only {ef}"

    def test_j_regime_present_and_valid(self, run_result):
        assignments, _ = run_result
        j_tasks = [a for a in assignments if a["prior_regime"] == "J_low_n_high_p_classification"]
        assert j_tasks, "J_low_n_high_p_classification missing from 1000-task run"
        for a in j_tasks:
            assert a["n"] >= _CURSOR_MIN_N_OVER_P * a["p_total"], (
                f"cursor J regime violated n>=5p: n={a['n']}, p={a['p_total']}"
            )


class TestClassificationCursorDeterminism:
    def test_same_seed_same_output(self):
        a1, _ = _run(seed=13)
        a2, _ = _run(seed=13)
        for i, (x, y) in enumerate(zip(a1, a2)):
            assert x["n"] == y["n"], f"Task {i}: n differs"
            assert x["p_total"] == y["p_total"], f"Task {i}: p_total differs"
            assert x["prior_regime"] == y["prior_regime"], f"Task {i}: regime differs"

    def test_different_seeds_differ(self):
        a1, _ = _run(seed=3)
        a2, _ = _run(seed=4)
        n_vals_1 = [a["n"] for a in a1]
        n_vals_2 = [a["n"] for a in a2]
        assert n_vals_1 != n_vals_2
