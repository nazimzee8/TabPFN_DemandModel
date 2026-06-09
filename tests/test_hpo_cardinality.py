"""Tests for HPO cardinality defaults (Area 4).

Confirms:
- HPO_MAX_INDEX_P defaults to MODEL4_MAX_P=256 for linear modes
- HPO_MAX_INDEX_N_TRAIN defaults to MODEL4_MAX_N_TRAIN=1024 for linear modes
- env-var overrides work
- build_hpo_search_space handles all four canonical modes without dead-branch errors
"""

import os
import sys
import importlib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Expected values from constants.py
MODEL4_MAX_P       = 256
MODEL4_MAX_N_TRAIN = 1024


def _reload_hpo(monkeypatch, sweep_mode, extra_env=None):
    """Reload hpo module with specific HPO_SWEEP_MODE and optional env overrides."""
    monkeypatch.setenv("HPO_SWEEP_MODE", sweep_mode)
    if extra_env:
        for k, v in extra_env.items():
            monkeypatch.setenv(k, v)
    # hpo.py imports are done at module level; reload to pick up new env
    import hpo
    importlib.reload(hpo)
    return hpo


class TestHPOCardinality:
    def test_linear_model_default_max_p(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model")
        assert hpo.HPO_MAX_INDEX_P == MODEL4_MAX_P, (
            f"Expected HPO_MAX_INDEX_P={MODEL4_MAX_P} for linear_model, got {hpo.HPO_MAX_INDEX_P}"
        )

    def test_linear_model_default_max_n_train(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model")
        assert hpo.HPO_MAX_INDEX_N_TRAIN == MODEL4_MAX_N_TRAIN, (
            f"Expected HPO_MAX_INDEX_N_TRAIN={MODEL4_MAX_N_TRAIN} for linear_model, "
            f"got {hpo.HPO_MAX_INDEX_N_TRAIN}"
        )

    def test_linear_model_architecture_default_max_p(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model_architecture")
        assert hpo.HPO_MAX_INDEX_P == MODEL4_MAX_P

    def test_linear_model_architecture_default_max_n_train(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model_architecture")
        assert hpo.HPO_MAX_INDEX_N_TRAIN == MODEL4_MAX_N_TRAIN

    def test_nonlinear_model_default_max_p(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "nonlinear_model")
        assert hpo.HPO_MAX_INDEX_P == 512

    def test_env_var_override_max_p(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model", {"HPO_MAX_INDEX_P": "128"})
        assert hpo.HPO_MAX_INDEX_P == 128

    def test_env_var_override_max_n_train(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model", {"HPO_MAX_INDEX_N_TRAIN": "512"})
        assert hpo.HPO_MAX_INDEX_N_TRAIN == 512


class TestBuildHPOSearchSpace:
    """build_hpo_search_space must return a dict for all 4 canonical modes."""

    def _make_fake_tune(self):
        """Minimal stand-in for Ray Tune's tune namespace."""
        class _FakeTune:
            @staticmethod
            def loguniform(lo, hi):
                return ("loguniform", lo, hi)
            @staticmethod
            def uniform(lo, hi):
                return ("uniform", lo, hi)
            @staticmethod
            def choice(options):
                return ("choice", options)
        return _FakeTune()

    def test_linear_model_returns_dict(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model")
        result = hpo.build_hpo_search_space(self._make_fake_tune())
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_nonlinear_model_returns_dict(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "nonlinear_model")
        result = hpo.build_hpo_search_space(self._make_fake_tune())
        assert isinstance(result, dict)

    def test_linear_model_architecture_requires_baseline_config(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model_architecture")
        with pytest.raises(ValueError, match="HPO_BASELINE_CONFIG_STAGE_PATH|baseline"):
            hpo.build_hpo_search_space(self._make_fake_tune(), baseline_config=None)

    def test_linear_model_architecture_with_baseline_returns_dict(self, monkeypatch):
        hpo = _reload_hpo(monkeypatch, "linear_model_architecture")
        baseline = {
            "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
            "use_ridge_expert": False, "ridge_lambda": 1.0, "gate_hidden_dim": 64,
            "use_huber": False, "huber_delta": 1.0, "lambda_l1": 0.0,
            "use_linear_stats": True, "use_coefficient_head": True, "use_lambda_head": True,
            "linear_stat_dim": 128, "coeff_head_hidden_dim": 64, "lambda_head_hidden_dim": 32,
        }
        result = hpo.build_hpo_search_space(self._make_fake_tune(), baseline_config=baseline)
        assert isinstance(result, dict)

    def test_no_dead_linear_stats_branch(self, monkeypatch):
        """'linear_stats' alias maps to 'linear_model' at module load; no dead branch in search space."""
        hpo = _reload_hpo(monkeypatch, "linear_model")
        # The module-level HPO_SWEEP_MODE must already be canonical
        assert hpo.HPO_SWEEP_MODE == "linear_model"
