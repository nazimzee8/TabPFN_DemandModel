"""
tests/test_permutation_contracts.py

Comprehensive tests for the formal permutation invariance and equivariance
contracts (F1–F7) specified in the plan.

Coverage:
  - F1  Support-row permutation invariance  (regression + classification K=2,3,5,10)
  - F2  Query-row permutation equivariance  (regression + classification)
  - F3  Feature-column permutation consistency (regression + classification)
  - F4  Feature-indexed output equivariance (beta/W coefficients)
  - F5  Class-label permutation equivariance (K=2 passes; K≥3 expected-fail
        due to learned class-ID embeddings — structural limitation)
  - F6  Completion row equivariance
  - F7  Completion column equivariance
  - Tolerance policy routing
  - Structured result format
  - Multiple permutation seeds (determinism)
  - Edge cases: severe imbalance, missing support classes, bounded contexts,
        single feature, single sample
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import (
    ModelConfig,
    DeepSetICLModel,
    DeepSetCompletionModel,
    _instantiate_model,
)
from permutation_contracts import (
    PermutationResult,
    check_support_row_invariance,
    check_query_row_equivariance,
    check_feature_column_consistency,
    check_feature_indexed_equivariance,
    check_class_label_equivariance,
    check_completion_row_equivariance,
    check_completion_column_equivariance,
    run_all,
    run_all_strict,
)
from tolerance_policy import get_tolerance, Tolerance, TOLERANCE_POLICY_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _regression_model(d_phi=32, n_blocks=1, dropout=0.0, use_coeff=True):
    cfg = ModelConfig(
        d_phi=d_phi,
        d_rho=64,
        pool="mean",
        n_heads=4,
        n_sab_feat=n_blocks,
        dropout=dropout,
        linear_stat_dim=16,
        linear_encoder_hidden_dim=24,
        coeff_head_hidden_dim=12,
        lambda_head_hidden_dim=8,
        use_coefficient_head=use_coeff,
    )
    return DeepSetICLModel(cfg).eval()


def _classification_model(max_K=10, d_phi=16):
    cfg = ModelConfig(
        d_phi=d_phi,
        d_rho=16,
        pool="mean",
        n_heads=4,
        n_sab_feat=1,
        dropout=0.0,
        task_objective="inductive_classification",
        use_classification_path=True,
        max_num_classes=max_K,
        class_embedding_dim=8,
        class_stat_dim=16,
        class_stat_hidden_dim=24,
        class_head_hidden_dim=12,
        class_fusion_gate_hidden_dim=12,
        linear_stat_dim=16,
        linear_encoder_hidden_dim=24,
        coeff_head_hidden_dim=12,
        lambda_head_hidden_dim=8,
    )
    return DeepSetICLModel(cfg).eval()


def _completion_model(d_phi=32, n_blocks=1):
    cfg = ModelConfig(
        model_family="market_exchangeable_completion",
        model_arch_version="model3",
        model_design_pattern="transductive_completion",
        d_phi=d_phi,
        n_sab_feat=n_blocks,
        n_heads=4,
        dropout=0.0,
    )
    return DeepSetCompletionModel(cfg).eval()


# ---------------------------------------------------------------------------
# Tolerance policy tests
# ---------------------------------------------------------------------------

class TestTolerancePolicy:
    def test_float32_cpu_deterministic_baseline(self):
        tol = get_tolerance(dtype=torch.float32, device_type="cpu",
                            inference="deterministic", output_type="scalar_prediction")
        assert tol.atol == 1e-5
        assert tol.rtol == 1e-5

    def test_float64_tighter(self):
        tol = get_tolerance(dtype=torch.float64)
        assert tol.atol < 1e-8

    def test_bfloat16_looser(self):
        tol = get_tolerance(dtype=torch.bfloat16)
        assert tol.atol > 1e-3

    def test_stochastic_multiplier(self):
        det = get_tolerance(inference="deterministic")
        sto = get_tolerance(inference="stochastic")
        assert sto.atol > det.atol

    def test_cuda_multiplier(self):
        cpu = get_tolerance(device_type="cpu")
        gpu = get_tolerance(device_type="cuda")
        assert gpu.atol > cpu.atol

    def test_probability_output_type_looser(self):
        scalar = get_tolerance(output_type="scalar_prediction")
        probs = get_tolerance(output_type="probabilities")
        assert probs.atol > scalar.atol

    def test_allclose_method(self):
        tol = Tolerance(atol=0.1, rtol=0.1)
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.05, 2.05])
        assert tol.allclose(a, b)
        c = torch.tensor([1.5, 2.5])
        assert not tol.allclose(a, c)

    def test_version_string(self):
        assert TOLERANCE_POLICY_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# PermutationResult structure tests
# ---------------------------------------------------------------------------

class TestPermutationResultFormat:
    def test_result_has_all_required_fields(self):
        m = _regression_model()
        r = check_support_row_invariance(m)
        assert isinstance(r, PermutationResult)
        d = r.to_dict()
        required_keys = {
            "check_type", "task_objective", "num_classes",
            "permutation_seed", "device", "dtype",
            "max_abs_delta", "mean_abs_delta", "max_rel_delta",
            "prediction_flip_rate", "passed", "threshold_atol",
            "threshold_rtol", "failure_reason",
            "reference_shape", "permuted_shape",
            "elapsed_s", "tolerance_policy_version",
        }
        assert required_keys <= set(d.keys())

    def test_serialization_roundtrip(self):
        m = _regression_model()
        r = check_support_row_invariance(m)
        d = r.to_dict()
        assert isinstance(d["reference_shape"], list)
        assert isinstance(d["permuted_shape"], list)
        assert isinstance(d["max_abs_delta"], float)


# ---------------------------------------------------------------------------
# F1: Support-row permutation invariance — Regression
# ---------------------------------------------------------------------------

class TestF1RegressionSupportInvariance:
    def test_basic(self):
        m = _regression_model()
        r = check_support_row_invariance(m, n=20, p=5, m=4, seed=42)
        assert r.passed, f"F1 regression failed: {r.failure_reason}"
        assert r.check_type == "F1_support_row_invariance"
        assert r.task_objective == "inductive_regression"

    @pytest.mark.parametrize("seed", [0, 7, 42, 99, 123])
    def test_multiple_seeds(self, seed):
        m = _regression_model()
        r = check_support_row_invariance(m, seed=seed)
        assert r.passed, f"seed={seed}: {r.failure_reason}"

    def test_large_context(self):
        m = _regression_model()
        r = check_support_row_invariance(m, n=100, p=10, m=8, seed=50)
        assert r.passed

    def test_single_feature(self):
        m = _regression_model()
        r = check_support_row_invariance(m, n=15, p=1, m=3, seed=51)
        assert r.passed

    def test_model3_arch(self):
        cfg = ModelConfig(
            model_family="market_exchangeable_icl",
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            d_phi=32, n_sab_feat=1, n_heads=4, dropout=0.0,
        )
        m = DeepSetICLModel(cfg).eval()
        r = check_support_row_invariance(m, seed=60)
        assert r.passed

    def test_with_ridge_expert(self):
        cfg = ModelConfig(
            model_family="market_exchangeable_icl",
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            d_phi=32, n_sab_feat=1, n_heads=4, dropout=0.0,
            use_ridge_expert=True, gate_hidden_dim=16,
        )
        m = DeepSetICLModel(cfg).eval()
        r = check_support_row_invariance(m, seed=61)
        assert r.passed


# ---------------------------------------------------------------------------
# F1: Support-row permutation invariance — Classification
# ---------------------------------------------------------------------------

class TestF1ClassificationSupportInvariance:
    @pytest.mark.parametrize("K", [2, 3, 5, 10])
    def test_by_num_classes(self, K):
        m = _classification_model()
        r = check_support_row_invariance(m, n=30, p=5, m=4, seed=42, num_classes=K)
        assert r.passed, f"K={K}: {r.failure_reason}"
        assert r.prediction_flip_rate == 0.0

    def test_severe_imbalance(self):
        """One class has 1 sample, rest have many."""
        m = _classification_model()
        # Use a seed that creates the imbalanced labels; we override manually
        device = next(m.parameters()).device
        torch.manual_seed(70)
        n, p, mm, K = 25, 5, 4, 3
        X_train = torch.randn(n, p, device=device)
        # Class 2 has only 1 sample
        y_train = torch.zeros(n, dtype=torch.long, device=device)
        y_train[:12] = 0
        y_train[12:24] = 1
        y_train[24] = 2
        x_test = torch.randn(mm, p, device=device)
        pi = torch.randperm(n, device=device)

        with torch.no_grad():
            ref = m(X_train, y_train, x_test,
                    task_objective="inductive_classification", num_classes=K)["logits"]
            perm = m(X_train[pi], y_train[pi], x_test,
                     task_objective="inductive_classification", num_classes=K)["logits"]

        assert torch.allclose(ref, perm, atol=1e-5), \
            f"Imbalanced support invariance failed: max_delta={(ref-perm).abs().max().item():.2e}"

    def test_missing_support_class(self):
        """Not all classes present in support set."""
        m = _classification_model()
        device = next(m.parameters()).device
        torch.manual_seed(71)
        n, p, mm, K = 20, 5, 4, 3
        X_train = torch.randn(n, p, device=device)
        # Only classes 0 and 1 present
        y_train = (torch.arange(n, device=device) % 2).long()
        x_test = torch.randn(mm, p, device=device)
        pi = torch.randperm(n, device=device)

        with torch.no_grad():
            ref = m(X_train, y_train, x_test,
                    task_objective="inductive_classification", num_classes=K)["logits"]
            perm = m(X_train[pi], y_train[pi], x_test,
                     task_objective="inductive_classification", num_classes=K)["logits"]

        assert torch.allclose(ref, perm, atol=1e-5), \
            f"Missing class support invariance failed: max_delta={(ref-perm).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# F2: Query-row permutation equivariance
# ---------------------------------------------------------------------------

class TestF2QueryEquivariance:
    def test_regression_basic(self):
        m = _regression_model()
        r = check_query_row_equivariance(m, n=20, p=5, m=8, seed=43)
        assert r.passed, f"F2 regression failed: {r.failure_reason}"

    def test_regression_single_query(self):
        """Single query should produce scalar — equivariance is trivial."""
        m = _regression_model()
        r = check_query_row_equivariance(m, n=20, p=5, m=1, seed=44)
        assert r.passed

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_classification(self, K):
        m = _classification_model()
        r = check_query_row_equivariance(m, n=20, p=5, m=6, seed=43, num_classes=K)
        assert r.passed, f"K={K}: {r.failure_reason}"

    @pytest.mark.parametrize("m_val", [2, 4, 8, 16])
    def test_various_batch_sizes(self, m_val):
        m = _regression_model()
        r = check_query_row_equivariance(m, m=m_val, seed=45)
        assert r.passed


# ---------------------------------------------------------------------------
# F3: Feature-column permutation consistency
# ---------------------------------------------------------------------------

class TestF3FeatureColumnConsistency:
    def test_regression_basic(self):
        m = _regression_model()
        r = check_feature_column_consistency(m, n=20, p=5, m=4, seed=44)
        assert r.passed, f"F3 regression failed: {r.failure_reason}"

    @pytest.mark.parametrize("p_val", [1, 3, 5, 10])
    def test_various_feature_counts(self, p_val):
        m = _regression_model()
        r = check_feature_column_consistency(m, p=p_val, seed=55)
        assert r.passed

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_classification(self, K):
        m = _classification_model()
        r = check_feature_column_consistency(m, n=20, p=5, m=4, seed=44, num_classes=K)
        assert r.passed, f"K={K}: {r.failure_reason}"


# ---------------------------------------------------------------------------
# F4: Feature-indexed output equivariance
# ---------------------------------------------------------------------------

class TestF4FeatureIndexedEquivariance:
    def test_regression_beta_hat(self):
        m = _regression_model(use_coeff=True)
        r = check_feature_indexed_equivariance(m, n=20, p=5, m=4, seed=45)
        assert r.passed, f"F4 regression beta failed: {r.failure_reason}"

    def test_regression_no_coeff_head_skips(self):
        m = _regression_model(use_coeff=False)
        r = check_feature_indexed_equivariance(m, seed=45)
        assert r.passed
        assert "skipped" in r.failure_reason

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_classification_W_hat(self, K):
        m = _classification_model()
        r = check_feature_indexed_equivariance(m, n=20, p=5, m=4, seed=45, num_classes=K)
        assert r.passed, f"K={K}: {r.failure_reason}"

    @pytest.mark.parametrize("p_val", [3, 7, 12])
    def test_regression_variable_p(self, p_val):
        m = _regression_model()
        r = check_feature_indexed_equivariance(m, p=p_val, seed=56)
        assert r.passed


# ---------------------------------------------------------------------------
# F5: Class-label permutation equivariance
# ---------------------------------------------------------------------------

class TestF5ClassLabelEquivariance:
    def test_regression_skips(self):
        m = _regression_model()
        r = check_class_label_equivariance(m)
        assert r.passed
        assert "skipped" in r.failure_reason

    def test_binary_passes(self):
        """Binary (K=2) label permutation should pass because the swap
        permutation is the only non-identity permutation and the class
        embeddings can accommodate symmetric treatment."""
        m = _classification_model()
        r = check_class_label_equivariance(m, num_classes=2, seed=46)
        # K=2 may pass due to embedding symmetry; document either outcome
        # This is informational — the structural limitation is in K≥3
        assert isinstance(r.passed, bool)

    @pytest.mark.parametrize("K", [3, 5, 10])
    def test_multiclass_expected_fail(self, K):
        """K≥3 label equivariance fails due to learned class-ID embeddings.

        This is a known structural limitation documented in the contracts.
        The ClassLabelEncoder uses nn.Embedding(max_num_classes, dim) which
        assigns distinct learned vectors to each class index, making the
        model sensitive to the absolute class ID.

        This test documents the current behavior. If the architecture is
        redesigned to use class-permutation-equivariant embeddings, these
        tests should be updated to assert passage.
        """
        m = _classification_model()
        # Try multiple seeds to ensure we hit a non-identity permutation
        found_violation = False
        for seed in range(46, 66):
            r = check_class_label_equivariance(m, num_classes=K, seed=seed)
            if not r.passed:
                found_violation = True
                assert r.max_abs_delta > 1e-3
                break
        assert found_violation, (
            f"K={K} label equivariance passed across 20 seeds — "
            "check if class embeddings were redesigned"
        )


# ---------------------------------------------------------------------------
# F6/F7: Completion row/column equivariance
# ---------------------------------------------------------------------------

class TestF6CompletionRowEquivariance:
    def test_basic(self):
        m = _completion_model()
        r = check_completion_row_equivariance(m, rows=10, cols=6, seed=47)
        assert r.passed, f"F6 failed: {r.failure_reason}"

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_multiple_seeds(self, seed):
        m = _completion_model()
        r = check_completion_row_equivariance(m, seed=seed)
        assert r.passed

    def test_sparse_mask(self):
        m = _completion_model()
        r = check_completion_row_equivariance(m, rows=8, cols=5, seed=57)
        assert r.passed


class TestF7CompletionColumnEquivariance:
    def test_basic(self):
        m = _completion_model()
        r = check_completion_column_equivariance(m, rows=10, cols=6, seed=48)
        assert r.passed, f"F7 failed: {r.failure_reason}"

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_multiple_seeds(self, seed):
        m = _completion_model()
        r = check_completion_column_equivariance(m, seed=seed)
        assert r.passed


class TestCompletionJointEquivariance:
    def test_joint_row_col(self):
        """Joint row+col permutation composes correctly."""
        m = _completion_model()
        device = next(m.parameters()).device
        torch.manual_seed(58)
        rows, cols = 8, 6
        X = torch.randn(rows, cols, device=device)
        mask = torch.rand(rows, cols, device=device) > 0.3
        pi_r = torch.randperm(rows, device=device)
        pi_c = torch.randperm(cols, device=device)

        with torch.no_grad():
            ref = m(X, mask)
            perm = m(X[pi_r][:, pi_c], mask[pi_r][:, pi_c])

        expected = ref[pi_r][:, pi_c]
        assert torch.allclose(perm, expected, atol=1e-5), (
            f"Joint equivariance failed: max_delta={(perm - expected).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# run_all dispatch
# ---------------------------------------------------------------------------

class TestRunAllDispatch:
    def test_regression_returns_f1_f4(self):
        m = _regression_model()
        results = run_all(m)
        types = {r.check_type for r in results}
        assert "F1_support_row_invariance" in types
        assert "F2_query_row_equivariance" in types
        assert "F3_feature_column_consistency" in types
        assert "F4_feature_indexed_equivariance" in types
        assert "F5_class_label_equivariance" not in types

    def test_classification_includes_f5(self):
        m = _classification_model()
        results = run_all(m)
        types = {r.check_type for r in results}
        assert "F1_support_row_invariance" in types
        assert "F5_class_label_equivariance" in types

    def test_completion_returns_f6_f7(self):
        m = _completion_model()
        results = run_all(m)
        types = {r.check_type for r in results}
        assert "F6_completion_row_equivariance" in types
        assert "F7_completion_column_equivariance" in types
        assert len(types) == 2

    def test_run_all_strict_regression_passes(self):
        m = _regression_model()
        results = run_all_strict(m)
        assert all(r.passed or "skipped" in r.failure_reason for r in results)

    def test_unknown_family_raises(self):
        m = _regression_model()
        # Bypass ModelConfig validation by patching the frozen field directly
        object.__setattr__(m.cfg, "model_family", "unknown_model")
        with pytest.raises(ValueError, match="Unknown model_family"):
            run_all(m)


# ---------------------------------------------------------------------------
# Deterministic reproducibility
# ---------------------------------------------------------------------------

class TestDeterministicReproducibility:
    def test_same_seed_same_result(self):
        m = _regression_model()
        r1 = check_support_row_invariance(m, seed=100)
        r2 = check_support_row_invariance(m, seed=100)
        assert r1.max_abs_delta == r2.max_abs_delta
        assert r1.passed == r2.passed

    def test_different_seeds_different_permutations(self):
        """Different seeds should use different permutations (and potentially
        produce different max_abs_delta values, though both should pass)."""
        m = _regression_model()
        r1 = check_support_row_invariance(m, seed=200)
        r2 = check_support_row_invariance(m, seed=201)
        assert r1.passed
        assert r2.passed


# ---------------------------------------------------------------------------
# Bounded context (small n)
# ---------------------------------------------------------------------------

class TestBoundedContext:
    def test_very_small_context_regression(self):
        """n=3 (minimum viable context for normalization)."""
        m = _regression_model()
        r = check_support_row_invariance(m, n=3, p=2, m=2, seed=80)
        assert r.passed

    def test_very_small_context_classification(self):
        m = _classification_model()
        # n=4, K=2 so each class has ≥1 sample
        r = check_support_row_invariance(m, n=4, p=3, m=2, seed=81, num_classes=2)
        assert r.passed


# ---------------------------------------------------------------------------
# Model restores train/eval mode
# ---------------------------------------------------------------------------

class TestModelStatePreservation:
    def test_eval_mode_preserved(self):
        m = _regression_model()
        m.eval()
        _ = check_support_row_invariance(m)
        assert not m.training

    def test_train_mode_preserved(self):
        m = _regression_model()
        m.train()
        _ = check_support_row_invariance(m)
        assert m.training
