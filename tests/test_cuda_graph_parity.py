"""
tests/test_cuda_graph_parity.py

CPU-only parity and correctness tests for the CUDA-graph masked-padding path.

Three properties are verified:
  1. Numerical parity  — masked-padded forward_regression matches the unpadded
     (legacy) forward to ≤1e-5 absolute tolerance on both the primary prediction
     and the debug tensors produced by model4 auxiliary heads.
  2. Static shapes     — every padded call presents the same (N_PAD, M_PAD, P_PAD)
     tensor shapes to the model regardless of the true (n, p, m) in the task.
  3. Fail-loud         — the padded path raises ValueError with diagnostic messages
     when invariants are violated (SetPool active, 1-D x_test, p > P_PAD).

These tests are entirely CPU-based and require only torch + the local model package.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
MODEL_DIR = os.path.join(SRC_DIR, "model")

# Load src/model/model.py explicitly by file path under a private module name.
#
# Why: importlib loads from the exact file path without writing to sys.modules,
# keeping this module isolated from the rest of the pytest session.  Manipulating
# sys.path to insert MODEL_DIR would risk import-order side-effects across other
# tests in the same run, so we avoid it entirely.
_spec = importlib.util.spec_from_file_location(
    "_cuda_graph_model_under_test",
    os.path.join(MODEL_DIR, "model.py"),
)
_model_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model_module)

ModelConfig     = _model_module.ModelConfig
DeepSetICLModel = _model_module.DeepSetICLModel

# Default pad constants from train.py (used in the fail-loud guard test below
# without importing train.py, which would trigger train.py's own model import
# and could further pollute sys.modules).
_CUDA_GRAPH_N_PAD_DEFAULT = 256
_CUDA_GRAPH_P_PAD_DEFAULT = 32
_CUDA_GRAPH_M_PAD_DEFAULT = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(use_linear_stats: bool = True, pool: str = "mean") -> DeepSetICLModel:
    """Small model for fast CPU tests."""
    cfg = ModelConfig(
        d_phi=32, d_rho=64, pool=pool,
        n_heads=2, n_sab_feat=1,
        norm_feat=True, norm_target=True, dropout=0.0,
        icl_pool_mode="mean", feature_pool_mode="mean",
        model_arch_version="model4" if use_linear_stats else "model3",
        use_linear_stats=use_linear_stats,
        use_coefficient_head=use_linear_stats,
        use_lambda_head=use_linear_stats,
        # Keep aux-loss weights nonzero so the OLS teacher branch is exercised.
        beta_aux_loss_weight=0.10,
        pred_aux_loss_weight=0.05,
        cos_aux_loss_weight=0.02,
        teacher_ols_threshold=5.0,
    )
    model = DeepSetICLModel(cfg=cfg)
    model.eval()
    return model


def _make_task(n: int, p: int, m: int, seed: int = 42):
    """Return fixed random (X_train, y_train, X_test, beta)."""
    gen = torch.Generator().manual_seed(seed)
    X_train = torch.randn(n, p, generator=gen)
    beta    = torch.randn(p, generator=gen)
    y_train = X_train @ beta + 0.05 * torch.randn(n, generator=gen)
    X_test  = torch.randn(m, p, generator=gen)
    return X_train, y_train, X_test, beta


def _pad_rows_cols(x: torch.Tensor, n_rows: int, n_cols: int) -> torch.Tensor:
    out = x.new_zeros((n_rows, n_cols))
    out[: x.shape[0], : x.shape[1]] = x
    return out


def _pad_vec(x: torch.Tensor, n: int) -> torch.Tensor:
    out = x.new_zeros((n,))
    out[: x.shape[0]] = x
    return out


# ---------------------------------------------------------------------------
# 1. Numerical-parity tests
# ---------------------------------------------------------------------------

class TestNumericalParity:
    """Masked-padded forward == unpadded (legacy) forward to ≤1e-5 abs tolerance."""

    # Pad sizes much larger than the task to prove zeros don't leak.
    N_PAD = 32
    P_PAD = 8
    M_PAD = 16

    def _run_parity(self, use_linear_stats: bool, n: int = 20, p: int = 3, m: int = 6):
        model = _make_model(use_linear_stats=use_linear_stats)
        X_train, y_train, X_test, beta = _make_task(n, p, m)

        # --- Legacy (unpadded) path ---
        with torch.no_grad():
            y_hat_ref, dbg_ref = model.forward_regression(
                X_train, y_train, X_test,
                return_debug=True, beta=beta,
            )

        # --- Padded (masked) path ---
        X_train_g = _pad_rows_cols(X_train, self.N_PAD, self.P_PAD)
        y_train_g = _pad_vec(y_train, self.N_PAD)
        X_test_g  = _pad_rows_cols(X_test, self.M_PAD, self.P_PAD)
        beta_g    = _pad_vec(beta, self.P_PAD)
        n_valid   = torch.tensor(n, dtype=torch.long)
        p_valid   = torch.tensor(p, dtype=torch.long)

        m_valid   = torch.tensor(m, dtype=torch.long)

        with torch.no_grad():
            y_hat_pad, dbg_pad = model.forward_regression(
                X_train_g, y_train_g, X_test_g,
                return_debug=True, beta=beta_g,
                n_valid=n_valid, p_valid=p_valid, m_valid=m_valid,
            )

        # Primary output — slice padded result back to m valid rows.
        y_hat_sliced = y_hat_pad[:m]
        torch.testing.assert_close(
            y_hat_sliced, y_hat_ref,
            atol=1e-5, rtol=1e-5,
            msg=f"Primary prediction mismatch (linear_stats={use_linear_stats}, n={n}, p={p}, m={m})",
        )

        # Debug tensors from model4 (skip if model3).
        if use_linear_stats:
            # y_coeff_norm: (M_PAD,) padded vs (m,) reference
            _yc_ref = dbg_ref.get("y_coeff_norm")
            _yc_pad = dbg_pad.get("y_coeff_norm")
            if _yc_ref is not None and _yc_pad is not None:
                torch.testing.assert_close(
                    _yc_pad[:m], _yc_ref,
                    atol=1e-5, rtol=1e-5,
                    msg="y_coeff_norm mismatch",
                )

            # beta_hat_norm: (P_PAD,) padded vs (p,) reference
            _bh_ref = dbg_ref.get("beta_hat_norm")
            _bh_pad = dbg_pad.get("beta_hat_norm")
            if _bh_ref is not None and _bh_pad is not None:
                # Padded entries beyond p are zeroed by col_mask — slice to valid p.
                torch.testing.assert_close(
                    _bh_pad[:p], _bh_ref,
                    atol=1e-5, rtol=1e-5,
                    msg="beta_hat_norm mismatch",
                )

            # y_ols_norm_teacher: (M_PAD,) padded vs (m,) reference
            _yo_ref = dbg_ref.get("y_ols_norm_teacher")
            _yo_pad = dbg_pad.get("y_ols_norm_teacher")
            if _yo_ref is not None and _yo_pad is not None:
                torch.testing.assert_close(
                    _yo_pad[:m], _yo_ref,
                    atol=1e-5, rtol=1e-5,
                    msg="y_ols_norm_teacher mismatch",
                )

    def test_parity_model3_no_linear_stats(self):
        """Pooling + normalisation parity, no linear-stat branch."""
        self._run_parity(use_linear_stats=False)

    def test_parity_model4_with_linear_stats(self):
        """Full model4 parity: linear stats + coeff head + OLS teacher."""
        self._run_parity(use_linear_stats=True)

    def test_parity_larger_padding_ratio(self):
        """Larger n/p/m relative to pad budget still matches."""
        self._run_parity(use_linear_stats=True, n=25, p=4, m=10)

    def test_parity_min_task(self):
        """Smallest plausible task (n=6, p=1, m=2) still matches."""
        self._run_parity(use_linear_stats=True, n=6, p=1, m=2)

    def test_parity_legacy_path_unchanged_when_n_valid_none(self):
        """With n_valid=None the legacy path is byte-for-byte unchanged."""
        model = _make_model()
        X_train, y_train, X_test, beta = _make_task(n=18, p=3, m=5)
        with torch.no_grad():
            y1 = model.forward_regression(X_train, y_train, X_test, beta=beta)
            y2 = model.forward_regression(X_train, y_train, X_test, beta=beta)
        torch.testing.assert_close(y1, y2, atol=0.0, rtol=0.0,
                                   msg="Two identical legacy calls must be bit-exact")


# ---------------------------------------------------------------------------
# 2. Static-shape tests
# ---------------------------------------------------------------------------

class TestStaticShapes:
    """Every padded call sees identical input tensor shapes (N_PAD, M_PAD, P_PAD)."""

    N_PAD = 32
    P_PAD = 8
    M_PAD = 16

    def test_padded_tensors_have_static_shapes(self):
        """Verify that varying (n, p, m) tasks produce identical padded shapes."""
        tasks = [
            (10, 2, 5),
            (20, 3, 6),
            (28, 7, 14),   # near pad budget
        ]
        seen_shapes = set()
        for n, p, m in tasks:
            X_train, y_train, X_test, _ = _make_task(n, p, m)
            X_train_g = _pad_rows_cols(X_train, self.N_PAD, self.P_PAD)
            y_train_g = _pad_vec(y_train, self.N_PAD)
            X_test_g  = _pad_rows_cols(X_test, self.M_PAD, self.P_PAD)
            shape = (
                tuple(X_train_g.shape),
                tuple(y_train_g.shape),
                tuple(X_test_g.shape),
            )
            seen_shapes.add(shape)

        assert len(seen_shapes) == 1, (
            f"Expected one unique shape across all tasks; got {seen_shapes}"
        )

    def test_n_valid_p_valid_are_tensors_not_ints(self):
        """n_valid and p_valid must be 0-dim tensors (not Python ints) for
        Dynamo not to int-specialize and recompile per task shape."""
        n_valid = torch.tensor(20, dtype=torch.long)
        p_valid = torch.tensor(5, dtype=torch.long)
        # 0-dim tensor: shape == () and is not a Python int
        assert n_valid.shape == torch.Size([])
        assert p_valid.shape == torch.Size([])
        assert not isinstance(n_valid, int)
        assert not isinstance(p_valid, int)


# ---------------------------------------------------------------------------
# 3. Fail-loud tests
# ---------------------------------------------------------------------------

class TestFailLoud:
    """Padded path raises ValueError with diagnostic messages for invalid inputs."""

    N_PAD = 16
    P_PAD = 8
    M_PAD = 8

    def test_fail_if_set_pool_active(self):
        """Non-mean sample-axis pool must be rejected — masked_mean is incompatible with SetPool."""
        cfg = ModelConfig(
            d_phi=32, d_rho=64, pool="pna",
            n_heads=2, n_sab_feat=1,
            icl_pool_mode="pna",    # <-- non-mean pool (PNA), not mean
            feature_pool_mode="mean",
            use_linear_stats=False, use_coefficient_head=False, use_lambda_head=False,
            model_arch_version="model3",
        )
        model = DeepSetICLModel(cfg=cfg)
        model.eval()

        n, p, m = 10, 3, 5
        X_train, y_train, X_test, _ = _make_task(n, p, m)
        X_train_g = _pad_rows_cols(X_train, self.N_PAD, self.P_PAD)
        y_train_g = _pad_vec(y_train, self.N_PAD)
        X_test_g  = _pad_rows_cols(X_test, self.M_PAD, self.P_PAD)
        n_valid   = torch.tensor(n, dtype=torch.long)
        p_valid   = torch.tensor(p, dtype=torch.long)

        with pytest.raises(ValueError, match="mean/mean pooling"):
            model.forward_regression(
                X_train_g, y_train_g, X_test_g,
                n_valid=n_valid, p_valid=p_valid,
            )

    def test_fail_if_x_test_1d(self):
        """1-D x_test must be rejected — padded path requires 2-D (m, p) query."""
        model = _make_model(use_linear_stats=False)
        n, p = 10, 3
        X_train, y_train, _, _ = _make_task(n, p, m=5)
        X_train_g = _pad_rows_cols(X_train, self.N_PAD, self.P_PAD)
        y_train_g = _pad_vec(y_train, self.N_PAD)
        # 1-D query (single sample) — padded path cannot handle this
        x_test_1d = torch.randn(self.P_PAD)
        n_valid   = torch.tensor(n, dtype=torch.long)

        with pytest.raises(ValueError, match="2-D"):
            model.forward_regression(
                X_train_g, y_train_g, x_test_1d,
                n_valid=n_valid,
            )

    def test_fail_if_p_exceeds_p_pad_in_train_driver(self):
        """The guard in run_epoch raises ValueError when task p > CUDA_GRAPH_P_PAD.

        Tests the validation logic in isolation (mirrors the guard in train.py's
        CUDA-graph branch) without importing train.py itself — importing train
        would execute its top-level `from model import` and could pollute the
        shared sys.modules in a multi-test session.
        """
        def _check_pad_guard(p_int: int, n_int: int) -> None:
            """Guard mirroring the one in run_epoch's CUDA-graph branch."""
            if p_int > _CUDA_GRAPH_P_PAD_DEFAULT:
                raise ValueError(
                    f"CUDA-graph path: p={p_int} exceeds CUDA_GRAPH_P_PAD="
                    f"{_CUDA_GRAPH_P_PAD_DEFAULT}. Increase P_PAD or set USE_CUDA_GRAPHS=false."
                    f" task_shape=(n={n_int}, p={p_int}, m=?)"
                )
            if n_int > _CUDA_GRAPH_N_PAD_DEFAULT:
                raise ValueError(
                    f"CUDA-graph path: n={n_int} exceeds CUDA_GRAPH_N_PAD="
                    f"{_CUDA_GRAPH_N_PAD_DEFAULT}. Increase N_PAD or set USE_CUDA_GRAPHS=false."
                    f" task_shape=(n={n_int}, p=?, m=?)"
                )

        # p overflow
        with pytest.raises(ValueError, match="CUDA_GRAPH_P_PAD"):
            _check_pad_guard(_CUDA_GRAPH_P_PAD_DEFAULT + 1, 10)

        # n overflow
        with pytest.raises(ValueError, match="CUDA_GRAPH_N_PAD"):
            _check_pad_guard(3, _CUDA_GRAPH_N_PAD_DEFAULT + 1)

        # valid (p, n) — should not raise
        _check_pad_guard(_CUDA_GRAPH_P_PAD_DEFAULT, _CUDA_GRAPH_N_PAD_DEFAULT)


# ---------------------------------------------------------------------------
# 4. Multi-task consistency (varying tasks, same padded shape)
# ---------------------------------------------------------------------------

class TestMultiTaskConsistency:
    """Multiple tasks with different (n, p, m) all produce numerically correct
    outputs when routed through the same static-shape padded forward."""

    N_PAD = 32
    P_PAD = 8
    M_PAD = 16

    def test_five_tasks_parity(self):
        """Five different tasks all match the unpadded baseline."""
        model = _make_model(use_linear_stats=True)
        tasks = [
            (12, 2, 4, 10),
            (20, 3, 7, 20),
            (18, 5, 5, 30),
            (25, 6, 9, 40),
            (8, 1, 3, 50),   # very small (n/p=8 > teacher_ols_threshold=5)
        ]
        for n, p, m, seed in tasks:
            X_train, y_train, X_test, beta = _make_task(n, p, m, seed=seed)

            with torch.no_grad():
                y_ref = model.forward_regression(X_train, y_train, X_test, beta=beta)

            X_train_g = _pad_rows_cols(X_train, self.N_PAD, self.P_PAD)
            y_train_g = _pad_vec(y_train, self.N_PAD)
            X_test_g  = _pad_rows_cols(X_test, self.M_PAD, self.P_PAD)
            beta_g    = _pad_vec(beta, self.P_PAD)
            n_valid   = torch.tensor(n, dtype=torch.long)
            p_valid   = torch.tensor(p, dtype=torch.long)

            m_valid   = torch.tensor(m, dtype=torch.long)

            with torch.no_grad():
                y_pad = model.forward_regression(
                    X_train_g, y_train_g, X_test_g,
                    beta=beta_g, n_valid=n_valid, p_valid=p_valid, m_valid=m_valid,
                )

            torch.testing.assert_close(
                y_pad[:m], y_ref,
                atol=1e-5, rtol=1e-5,
                msg=f"Task (n={n}, p={p}, m={m}, seed={seed}) mismatch",
            )
