"""
tests/test_ridge_expert.py

Correctness tests for RidgeExpert: shapes, primal/dual consistency,
known solution, and device consistency.
"""

import inspect
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import RidgeExpert  # noqa: E402


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_primal_path_shape():
    """n=50 >= p=5: primal path, output shape (m,)."""
    ridge = RidgeExpert()
    torch.manual_seed(0)
    n, p, m = 50, 5, 6
    X = torch.randn(n, p)
    y = torch.randn(n)
    x_test = torch.randn(m, p)

    out = ridge.predict(X, y, x_test, lam=1.0)
    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


def test_dual_path_shape():
    """n=5 < p=50: dual path, output shape (m,)."""
    ridge = RidgeExpert()
    torch.manual_seed(1)
    n, p, m = 5, 50, 6
    X = torch.randn(n, p)
    y = torch.randn(n)
    x_test = torch.randn(m, p)

    out = ridge.predict(X, y, x_test, lam=1.0)
    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


# ---------------------------------------------------------------------------
# Primal/dual consistency
# ---------------------------------------------------------------------------

def test_primal_dual_consistency():
    """Primal and dual closed-forms agree for same problem (max_abs_delta <= 1e-4)."""
    torch.manual_seed(42)
    n, p, m = 5, 10, 6   # n < p → dual path in RidgeExpert
    lam = 1.0
    kw = dict(dtype=torch.float64)

    X = torch.randn(n, p, **kw)
    y = torch.randn(n, **kw)
    x_test = torch.randn(m, p, **kw)

    # Primal (computed explicitly, independent of RidgeExpert path selection)
    A_primal = X.T @ X + lam * torch.eye(p, **kw)
    beta_primal = torch.linalg.solve(A_primal, X.T @ y)
    pred_primal = x_test @ beta_primal

    # Dual (same problem)
    K_dual = X @ X.T + lam * torch.eye(n, **kw)
    alpha  = torch.linalg.solve(K_dual, y)
    beta_dual = X.T @ alpha
    pred_dual = x_test @ beta_dual

    max_abs_delta = (pred_primal - pred_dual).abs().max().item()
    assert max_abs_delta <= 1e-4, (
        f"Primal/dual mismatch: {max_abs_delta:.2e} (threshold 1e-4)"
    )


# ---------------------------------------------------------------------------
# No y_test in signature
# ---------------------------------------------------------------------------

def test_no_y_test_in_signature():
    """RidgeExpert.predict must not have a y_test parameter."""
    sig = inspect.signature(RidgeExpert.predict)
    param_names = list(sig.parameters.keys())
    assert "y_test" not in param_names, (
        f"RidgeExpert.predict has forbidden parameter 'y_test'. Params: {param_names}"
    )


# ---------------------------------------------------------------------------
# Known solution: identity X
# ---------------------------------------------------------------------------

def test_known_solution_identity_X():
    """X = I_n (square, n==p), lambda≈0: ridge beta ≈ y, prediction ≈ x_test @ beta."""
    torch.manual_seed(5)
    n = 8
    p = n
    m = 4
    lam = 1e-6

    X = torch.eye(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    x_test = torch.randn(m, p, dtype=torch.float64)

    ridge = RidgeExpert()
    pred = ridge.predict(X, y, x_test, lam=lam)

    # With X=I and lambda→0, beta → y, so pred ≈ x_test @ y
    expected = x_test @ y
    max_err = (pred - expected).abs().max().item()

    # With lambda=1e-6, we expect reasonable accuracy but not exact match
    # The exact solution: beta = (I + 1e-6*I)^-1 y = y / (1 + 1e-6)
    beta_exact = y / (1.0 + lam)
    expected_exact = x_test @ beta_exact
    max_err_exact = (pred - expected_exact).abs().max().item()

    assert max_err_exact <= 1e-4, (
        f"Known-solution test failed: max_err={max_err_exact:.2e} (threshold 1e-4)"
    )


# ---------------------------------------------------------------------------
# Device consistency
# ---------------------------------------------------------------------------

def test_device_consistency():
    """Output tensor is on the same device as input tensors."""
    ridge = RidgeExpert()
    torch.manual_seed(3)
    n, p, m = 20, 5, 4
    X = torch.randn(n, p)
    y = torch.randn(n)
    x_test = torch.randn(m, p)

    out = ridge.predict(X, y, x_test, lam=1.0)
    assert out.device == X.device, (
        f"Output device {out.device} != input device {X.device}"
    )
