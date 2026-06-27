"""
generate_nonlinear_dgp.py

v2 DGP library for nonlinear synthetic regression.

Exports 6 target families and 7 feature regimes for both:
  - DeepSet pretraining (write_parquet / main)
  - Evaluation suite generation (write_parquet_eval / enumerate_suite_cells)

Usage (training):
    python src/generate_nonlinear_dgp.py --n_datasets 1000 --out_dir data/nonlinear/

Usage (evaluation — via scripts/generate_nonlinear_v2.py):
    from generate_nonlinear_dgp import enumerate_suite_cells, generate_v2_dataset, write_parquet_eval
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Bootstrap sys.path so dgp_helpers and src-level modules (e.g. constants) can be
# found when run directly.  Mirrors generate_dgp.py: scan ancestors for
# _bootstrap.py, insert its directory, import it so _bootstrap.install() adds
# every src/ subdir (including src/model/ where constants.py lives), then
# re-assert the co-located data_generation/ dir so the sibling dgp_helpers wins.
_p = Path(__file__).resolve()
for _anc in _p.parents:
    if (_anc / "_bootstrap.py").exists():
        sys.path.insert(0, str(_anc))
        break
import _bootstrap  # noqa: E402,F401  — runs install(), adds src/model/ etc.
del _p, _anc

# _bootstrap inserts src/ at sys.path[0], shadowing this script's own dir and
# binding bare `dgp_helpers` to a stale src/dgp_helpers.py. Re-assert the
# co-located dir so the sibling (with the *_cursor refactor) wins.
# This is a no-op on the Snowflake flat stage (no subdirs, no bootstrap).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import cursor helpers from the co-located dgp_helpers (no fallback — we want
# loud failures here rather than silent divergence from drifting inline constants).
from dgp_helpers import (  # noqa: E402
    sample_cursor_dims as _sample_cursor_dims,
    _CURSOR_P_LAMBDA,
    _CURSOR_N_LAMBDA,
    _CURSOR_TAIL_FRACTION,
    _CURSOR_TAIL_P_LAMBDA,
    _CURSOR_TAIL_P_CAP,
    _CURSOR_TAIL_P_FLOOR,
    _CURSOR_MIN_N_OVER_P,
)
_CURSOR_AVAILABLE = True


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

V2_TARGET_FAMILIES = [
    "poly_quad",
    "sin_low",
    "hinge",
    "sparse_interact",
    "mixed_linear",
    "demand_mono",
]

V2_FEATURE_REGIMES = [
    "iid_dense",
    "iid_sparse",
    "ar1",
    "block",
    "equicorr",
    "noise_feats",
    "feat_noise",
]

V2_N_GRID = [100, 300, 1000]
V2_P_SIGNAL_GRID = [8, 16, 32, 64]
V2_TARGET_NOISE_SCALES = [0.5, 1.0, 2.0]
V2_TARGET_NOISE_TYPES = ["gaussian", "t3", "heteroscedastic", "contaminated"]
V2_SPLIT_SEEDS = [0, 1, 2]

N_CAL = 5000          # independent calibration sample rows
OLS_R2_MIN = 1e-4     # learnability gate threshold — low so purely nonlinear families pass
MAX_TEACHER_RETRIES = 20

# rho grid for correlated regimes — zero is excluded by construction
_RHO_GRID_CORR = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BASE_SEED = 20260701


# ---------------------------------------------------------------------------
# v3 public constants
# ---------------------------------------------------------------------------

V3_TARGET_FAMILIES = [
    "poly", "sparse_interact", "smooth_additive", "rbf_local",
    "piecewise_relu", "discontinuous_threshold", "exp_growth", "power",
    "sigmoid", "saturation", "gompertz", "logarithmic",
    "periodic", "heteroskedastic", "high_dim_sparse_nonlinear",
    "low_rank_compositional", "mixed_linear",
]

V3_CLASSIFICATION_FAMILIES = [
    "nonlinear_binary_logistic", "sparse_interaction_logistic",
    "radial_decision_boundary", "piecewise_margin",
    "multiclass_smooth_softmax", "multiclass_sparse_highdim",
    "correlated_nonlinear_softmax", "low_margin_overlap",
    "label_noise_nonlinear", "class_imbalance_nonlinear",
    "mixed_categorical_nonlinear",
]

V3_FEATURE_REGIMES = [
    "iid_dense", "iid_sparse", "ar1", "block", "equicorr",
    "noise_feats", "feat_noise",
    "student_t_features", "laplace_features", "uniform_bounded",
    "skewed_lognormal", "mixture_gaussian", "low_rank_factor",
    "manifold_features",
]

_FAMILY_ALIASES: dict[str, str] = {
    "poly_quad": "poly",
    "sin_low": "smooth_additive",
    "hinge": "piecewise_relu",
}

# Seed magic constants — distinct per task type to prevent overlap
_NONLINEAR_REG_SEED_MAGIC = 0x4E4C5233       # "NLR3"
_NONLINEAR_MIXED_REG_SEED_MAGIC = 0x4E4D5233  # "NMR3"
_NONLINEAR_CLS_SEED_MAGIC = 0x4E4C4331       # "NLC1"
_NONLINEAR_MIXED_CLS_SEED_MAGIC = 0x4E4D4331  # "NMC1"

# v3 parameter grids
V3_N_GRID = [32, 64, 128, 256, 512, 1024]
V3_P_SIGNAL_GRID = [1, 4, 8, 16, 32, 64]
V3_NOISE_SCALES = [0.25, 0.5, 1.0, 2.0]
V3_NOISE_TYPES = [
    "gaussian", "student_t", "laplace", "cauchy",
    "heteroscedastic", "contaminated", "gross_outlier",
]

# Classification parameter grids
CLS_NUM_CLASSES_GRID = [2, 3, 5, 10]
CLS_TEMPERATURE_GRID = [0.5, 1.0, 2.0, 5.0]
CLS_LABEL_NOISE_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]
CLS_IMBALANCE_GRID = ["balanced", "mild", "moderate", "severe"]
CLS_MARGIN_GRID = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Learnability gate error
# ---------------------------------------------------------------------------

class LearnabilityGateError(RuntimeError):
    """Raised when a teacher function fails the Ridge R² learnability gate."""


# ---------------------------------------------------------------------------
# Covariance and feature-generation helpers
# ---------------------------------------------------------------------------

def _covariance_matrix(p: int, covariance_type: str, rho: float, block_size: int = 8) -> np.ndarray:
    if covariance_type == "iid":
        return np.eye(p)
    if covariance_type == "ar1":
        idx = np.arange(p)
        sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    elif covariance_type == "block":
        sigma = np.full((p, p), 0.1)
        for start in range(0, p, block_size):
            end = min(start + block_size, p)
            sigma[start:end, start:end] = rho
        np.fill_diagonal(sigma, 1.0)
    elif covariance_type == "equicorrelation":
        sigma = np.full((p, p), rho)
        np.fill_diagonal(sigma, 1.0)
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type!r}")
    return sigma + 1e-8 * np.eye(p)


def _generate_X(
    rng: np.random.Generator,
    n: int,
    p: int,
    covariance_type: str,
    rho: float,
) -> np.ndarray:
    if covariance_type == "iid":
        return rng.standard_normal((n, p))
    if covariance_type == "ar1":
        x = np.empty((n, p))
        x[:, 0] = rng.standard_normal(n)
        scale = math.sqrt(max(0.0, 1.0 - rho ** 2))
        for col in range(1, p):
            x[:, col] = rho * x[:, col - 1] + scale * rng.standard_normal(n)
        return x
    chol = np.linalg.cholesky(_covariance_matrix(p, covariance_type, rho))
    return rng.standard_normal((n, p)) @ chol.T


def _select_active_columns(rng: np.random.Generator, p_signal: int, active_fraction: float) -> np.ndarray:
    n_active = max(1, int(round(active_fraction * p_signal)))
    n_active = min(n_active, p_signal)
    return np.sort(rng.choice(p_signal, size=n_active, replace=False))


def _generate_X_v3(
    rng: np.random.Generator,
    n: int,
    p: int,
    feature_distribution: str,
    rho: float = 0.0,
) -> np.ndarray:
    """Generate feature matrix using extended distribution families.

    Handles both legacy covariance-based regimes and new v3 distributions.
    """
    if feature_distribution in ("iid_gaussian", "iid"):
        return rng.standard_normal((n, p))
    elif feature_distribution == "correlated_ar":
        return _generate_X(rng, n, p, "ar1", rho)
    elif feature_distribution == "block_correlated":
        return _generate_X(rng, n, p, "block", rho)
    elif feature_distribution == "equicorrelated":
        return _generate_X(rng, n, p, "equicorrelation", rho)
    elif feature_distribution == "student_t_features":
        raw = rng.standard_t(df=5, size=(n, p))
        col_std = np.std(raw, axis=0, keepdims=True)
        return raw / np.maximum(col_std, 1e-8)
    elif feature_distribution == "laplace_features":
        return rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=(n, p))
    elif feature_distribution == "uniform_bounded":
        return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size=(n, p))
    elif feature_distribution == "skewed_lognormal":
        raw = np.exp(rng.standard_normal((n, p)))
        mu = raw.mean(axis=0, keepdims=True)
        std = np.maximum(raw.std(axis=0, keepdims=True), 1e-8)
        return (raw - mu) / std
    elif feature_distribution == "mixture_gaussian":
        n_components = int(rng.integers(2, 5))
        X = np.zeros((n, p))
        assignments = rng.integers(0, n_components, size=n)
        for c_idx in range(n_components):
            mask = assignments == c_idx
            nc = int(mask.sum())
            if nc == 0:
                continue
            mu_c = rng.standard_normal(p) * 2.0
            X[mask] = mu_c + rng.standard_normal((nc, p))
        return X
    elif feature_distribution == "low_rank_factor":
        r = max(1, min(p // 2, int(rng.integers(2, min(p, 10) + 1))))
        Z = rng.standard_normal((n, r))
        L = rng.standard_normal((r, p)) / math.sqrt(max(r, 1))
        noise = 0.1 * rng.standard_normal((n, p))
        return Z @ L + noise
    elif feature_distribution == "manifold_features":
        r = max(1, min(p // 2, int(rng.integers(2, min(p, 8) + 1))))
        Z = rng.standard_normal((n, r))
        L = rng.standard_normal((r, p)) / math.sqrt(max(r, 1))
        X_linear = Z @ L
        return np.tanh(X_linear) + 0.1 * rng.standard_normal((n, p))
    else:
        raise ValueError(f"Unknown feature_distribution: {feature_distribution!r}")


# ---------------------------------------------------------------------------
# Teacher functions (one per target family)
# ---------------------------------------------------------------------------

def _poly_quad_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Additive polynomial: linear + quadratic contributions."""
    k = Xa.shape[1]
    w_lin = rng.standard_normal(k)
    w_quad = rng.standard_normal(k)
    return Xa @ w_lin + (Xa ** 2) @ w_quad


def _sin_low_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Low-frequency sinusoidal — frequency capped at 0.5/sigma (learnable)."""
    k = Xa.shape[1]
    w = rng.standard_normal(k)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        w[0] = 1.0
        norm = 1.0
    w = w / norm  # unit-norm projection
    # Feature std ≈ 1 (standard normal), cap frequency to 0.5 cycles per unit std
    freq = float(rng.uniform(0.05, 0.5))
    projection = Xa @ w  # std ≈ 1 because w is unit-norm
    return np.sin(2.0 * math.pi * freq * projection) * math.sqrt(k)


def _hinge_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Hinge/threshold/saturation: piecewise linear, non-differentiable."""
    k = Xa.shape[1]
    thresholds = rng.uniform(-1.0, 1.0, size=k)
    weights = rng.standard_normal(k)
    return np.sum(weights[np.newaxis, :] * np.maximum(0.0, Xa - thresholds[np.newaxis, :]), axis=1)


def _sparse_interact_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Sparse overlapping interactions — random graph of pairs."""
    k = Xa.shape[1]
    n_pairs = max(1, k)
    # Sample pairs with overlap (not just a chain)
    i_idx = rng.integers(0, k, size=n_pairs)
    j_idx = rng.integers(0, k, size=n_pairs)
    # Avoid trivial self-interactions by shifting j if equal to i
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % k
    coeffs = rng.standard_normal(n_pairs)
    betaX = np.zeros(Xa.shape[0])
    for c, i, j in zip(coeffs, i_idx, j_idx):
        betaX += c * Xa[:, i] * Xa[:, j]
    return betaX


def _mixed_linear_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Mixed linear + nonlinear residual; alpha controls OLS ceiling."""
    k = Xa.shape[1]
    alpha = float(params.get("alpha", rng.uniform(0.1, 0.9)))
    w_lin = rng.standard_normal(k)
    w_nl = rng.standard_normal(k)
    linear_part = Xa @ w_lin
    nl_part = np.tanh(Xa @ w_nl)
    return alpha * linear_part + (1.0 - alpha) * nl_part


def _demand_mono_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Demand-oriented monotonic: negative own-price elasticity, positive cross-price.

    Always produces corr(Xa[:,0], betaX) < 0 for iid features because own_coeff < 0.
    A tanh applied to the linear index preserves direction (monotone increasing transform)
    while adding non-linearity detectable by nonlinear models.
    """
    k = Xa.shape[1]
    # Own-price coefficient always strictly negative
    own_coeff = -(float(abs(rng.standard_normal(1)[0])) + 0.5)
    betaX = own_coeff * Xa[:, 0]
    if k > 1:
        cross_coeffs = np.abs(rng.standard_normal(k - 1)) * 0.3
        betaX += Xa[:, 1:] @ cross_coeffs
    # tanh is monotonically increasing → preserves negative own-price direction
    return np.tanh(betaX)


# ---------------------------------------------------------------------------
# v3 Teacher functions — Tier 1 parametric families
# ---------------------------------------------------------------------------

def _poly_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Polynomial: degree D in {2,3,4,5} with cross-terms.

    Generalizes poly_quad to higher degrees.  Coefficients for degree d are
    scaled by 1/sqrt(d) to prevent higher-order term domination.
    """
    D = int(params.get("degree", int(rng.choice([2, 3, 4, 5]))))
    k = Xa.shape[1]
    mu = np.zeros(Xa.shape[0])
    for d in range(1, D + 1):
        w_d = rng.standard_normal(k) / math.sqrt(d)
        mu += (Xa ** d) @ w_d
    # Cross-terms: min(k, 5) random pairs
    n_cross = min(k, 5)
    if k >= 2 and n_cross > 0:
        i_idx = rng.integers(0, k, size=n_cross)
        j_idx = rng.integers(0, k, size=n_cross)
        same = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % k
        c = rng.standard_normal(n_cross)
        for ci, ii, jj in zip(c, i_idx, j_idx):
            mu += ci * Xa[:, ii] * Xa[:, jj]
    return mu


def _smooth_additive_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Smooth additive: sin + tanh + quadratic terms per feature (GAM-like)."""
    k = Xa.shape[1]
    a = rng.standard_normal(k)
    w = rng.uniform(0.05, 0.5, k)
    b = rng.standard_normal(k)
    c = rng.uniform(0.5, 2.0, k)
    d = rng.standard_normal(k) * 0.5
    mu = np.zeros(Xa.shape[0])
    for j in range(k):
        mu += a[j] * np.sin(2.0 * math.pi * w[j] * Xa[:, j])
        mu += b[j] * np.tanh(c[j] * Xa[:, j])
        mu += d[j] * Xa[:, j] ** 2
    return mu


def _rbf_local_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """RBF local bumps: sum of M Gaussian kernels centred in feature space."""
    M = int(params.get("rbf_centers", int(rng.choice([2, 4, 8, 16]))))
    k = Xa.shape[1]
    centers = rng.standard_normal((M, k))
    bandwidths = rng.uniform(0.5, 3.0, size=M)
    alphas = rng.standard_normal(M)
    mu = np.zeros(Xa.shape[0])
    for m in range(M):
        diff = Xa - centers[m]
        dist_sq = np.sum(diff ** 2, axis=1)
        mu += alphas[m] * np.exp(-dist_sq / (2.0 * bandwidths[m] ** 2))
    return mu


def _discontinuous_threshold_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Discontinuous threshold: linear base + step-function jumps."""
    k = Xa.shape[1]
    beta = rng.standard_normal(k)
    M = int(params.get("threshold_count", int(rng.choice([1, 2, 4, 8]))))
    mu = Xa @ beta
    for _m in range(M):
        j = int(rng.integers(0, k))
        tau = float(rng.uniform(-1.5, 1.5))
        alpha = float(rng.standard_normal(1)[0])
        mu = mu + alpha * (Xa[:, j] > tau).astype(np.float64)
    return mu


def _exp_growth_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Exponential growth/decay: exp(rate * clipped projection) - 1."""
    k = Xa.shape[1]
    w = rng.standard_normal(k)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        w[0] = 1.0
        norm = 1.0
    w = w / norm
    rate = float(rng.uniform(0.3, 1.5))
    projection = np.clip(Xa @ w, -5.0, 5.0)
    return np.exp(rate * projection) - 1.0


def _power_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Power / Cobb-Douglas: product of |x_j + shift|^exponent over active set."""
    k = Xa.shape[1]
    n_active = min(k, max(1, int(rng.integers(1, min(k, 6) + 1))))
    active = np.sort(rng.choice(k, size=n_active, replace=False))
    betas = rng.uniform(0.2, 1.5, size=n_active)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_active)
    shift = 2.0
    mu = np.ones(Xa.shape[0])
    for idx, j in enumerate(active):
        base = np.abs(Xa[:, j] + shift)
        base = np.maximum(base, 1e-8)
        exponent = signs[idx] * betas[idx]
        mu *= np.power(base, exponent)
    return mu


def _sigmoid_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Logistic sigmoid: L / (1 + exp(-k*(projection - x0)))."""
    k = Xa.shape[1]
    w = rng.standard_normal(k)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        w[0] = 1.0
        norm = 1.0
    w = w / norm
    L = float(rng.uniform(1.0, 5.0))
    steepness = float(rng.uniform(0.5, 3.0))
    x0 = float(rng.uniform(-1.0, 1.0))
    projection = Xa @ w
    return L / (1.0 + np.exp(-steepness * (projection - x0)))


def _saturation_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Michaelis-Menten saturation: V_max * |z| / (K_m + |z|)."""
    k = Xa.shape[1]
    w = rng.standard_normal(k)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        w[0] = 1.0
        norm = 1.0
    w = w / norm
    V_max = float(rng.uniform(2.0, 10.0))
    K_m = float(rng.uniform(0.5, 3.0))
    substrate = np.abs(Xa @ w)
    return V_max * substrate / (K_m + substrate)


def _gompertz_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Gompertz double-exponential: a * exp(-b * exp(-c * projection))."""
    k = Xa.shape[1]
    w = rng.standard_normal(k)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        w[0] = 1.0
        norm = 1.0
    w = w / norm
    a = float(rng.uniform(2.0, 8.0))
    b = float(rng.uniform(1.0, 4.0))
    c = float(rng.uniform(0.3, 2.0))
    return a * np.exp(-b * np.exp(-c * (Xa @ w)))


def _logarithmic_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Logarithmic: sum_j a_j * log(|x_j| + 1)."""
    k = Xa.shape[1]
    a = rng.standard_normal(k)
    mu = np.zeros(Xa.shape[0])
    for j in range(k):
        mu += a[j] * np.log(np.abs(Xa[:, j]) + 1.0)
    return mu


# ---------------------------------------------------------------------------
# v3 Teacher functions — Tier 2 structural families
# ---------------------------------------------------------------------------

# List of Tier 1 teacher functions for delegation by structural families
_TIER1_TEACHER_FNS = [
    _poly_teacher, _smooth_additive_teacher, _rbf_local_teacher,
    _exp_growth_teacher, _sigmoid_teacher, _saturation_teacher,
    _gompertz_teacher, _logarithmic_teacher,
]


def _periodic_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Periodic: linear base + sinusoidal additive components."""
    k = Xa.shape[1]
    beta = rng.standard_normal(k)
    mu = Xa @ beta
    n_periodic = min(k, max(1, int(rng.integers(1, min(k, 6) + 1))))
    for j in range(n_periodic):
        alpha = float(rng.standard_normal(1)[0])
        w_j = float(rng.uniform(0.1, 2.0))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        mu = mu + alpha * np.sin(w_j * Xa[:, j] + phi)
    return mu


def _heteroskedastic_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Heteroskedastic: mu from a random Tier 1 teacher (noise handled separately)."""
    fn = _TIER1_TEACHER_FNS[int(rng.integers(0, len(_TIER1_TEACHER_FNS)))]
    return fn(rng, Xa, params)


def _high_dim_sparse_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """High-dimensional sparse nonlinear: delegates to Tier 1 on sparse active columns."""
    fn = _TIER1_TEACHER_FNS[int(rng.integers(0, len(_TIER1_TEACHER_FNS)))]
    return fn(rng, Xa, params)


def _low_rank_compositional_teacher(rng: np.random.Generator, Xa: np.ndarray, params: dict) -> np.ndarray:
    """Low-rank compositional: z = X @ L_low_rank, then h(z) via Tier 1 teacher."""
    k = Xa.shape[1]
    # rng.integers(low, high) requires low < high; guard for k < 2 where min(k,8)+1 == 2
    r = 1 if k < 2 else max(1, min(k, int(rng.integers(2, min(k, 8) + 1))))
    L = rng.standard_normal((k, r)) / math.sqrt(max(k, 1))
    Z = Xa @ L
    fn = _TIER1_TEACHER_FNS[int(rng.integers(0, len(_TIER1_TEACHER_FNS)))]
    return fn(rng, Z, params)


# ---------------------------------------------------------------------------
# Teacher registry (v3 — includes all v2 entries + new families + aliases)
# ---------------------------------------------------------------------------

_TEACHER_REGISTRY = {
    # Legacy v2 names (backward compat)
    "poly_quad":      _poly_quad_teacher,
    "sin_low":        _sin_low_teacher,
    "hinge":          _hinge_teacher,
    "sparse_interact": _sparse_interact_teacher,
    "mixed_linear":   _mixed_linear_teacher,
    "demand_mono":    _demand_mono_teacher,
    # v3 Tier 1 canonical names
    "poly":                      _poly_teacher,
    "smooth_additive":           _smooth_additive_teacher,
    "rbf_local":                 _rbf_local_teacher,
    "piecewise_relu":            _hinge_teacher,  # same impl, new canonical name
    "discontinuous_threshold":   _discontinuous_threshold_teacher,
    "exp_growth":                _exp_growth_teacher,
    "power":                     _power_teacher,
    "sigmoid":                   _sigmoid_teacher,
    "saturation":                _saturation_teacher,
    "gompertz":                  _gompertz_teacher,
    "logarithmic":               _logarithmic_teacher,
    # v3 Tier 2 structural
    "periodic":                     _periodic_teacher,
    "heteroskedastic":              _heteroskedastic_teacher,
    "high_dim_sparse_nonlinear":    _high_dim_sparse_teacher,
    "low_rank_compositional":       _low_rank_compositional_teacher,
}


# ---------------------------------------------------------------------------
# Calibration with learnability gate
# ---------------------------------------------------------------------------

def _calibrate_teacher(
    teacher_rng: np.random.Generator,
    target_family: str,
    Xa_cal: np.ndarray,
    params: dict,
) -> tuple[float, float, int]:
    """Compute (cal_mean, cal_std, param_seed) from calibration sample; apply Ridge R² gate.

    Returns the param_seed used so generate_v2_dataset can reproduce the same
    teacher function parameters during evaluation (prevents cal/eval param mismatch).

    Raises LearnabilityGateError if R² < OLS_R2_MIN after MAX_TEACHER_RETRIES.
    """
    teacher_fn = _TEACHER_REGISTRY[target_family]

    for attempt in range(MAX_TEACHER_RETRIES):
        # Draw a deterministic seed for this attempt; store it so eval can reuse it
        param_seed = int(teacher_rng.integers(0, 2**63))
        sub_rng = np.random.default_rng(param_seed)
        betaX_cal_raw = teacher_fn(sub_rng, Xa_cal, params)
        betaX_cal_raw = np.asarray(betaX_cal_raw, dtype=np.float64)

        cal_mean = float(betaX_cal_raw.mean())
        cal_std = float(betaX_cal_raw.std())
        if cal_std < 1e-12:
            continue  # degenerate — retry

        # Learnability gate: Ridge R² >= OLS_R2_MIN
        betaX_cal_norm = (betaX_cal_raw - cal_mean) / cal_std
        # Ridge: closed-form (X'X + alpha*I)^{-1} X'y
        alpha_ridge = 1e-6
        XtX = Xa_cal.T @ Xa_cal + alpha_ridge * np.eye(Xa_cal.shape[1])
        Xty = Xa_cal.T @ betaX_cal_norm
        try:
            coef = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue
        y_pred = Xa_cal @ coef
        ss_res = float(np.sum((betaX_cal_norm - y_pred) ** 2))
        ss_tot = float(np.sum((betaX_cal_norm - betaX_cal_norm.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-30)
        if r2 >= OLS_R2_MIN:
            return cal_mean, cal_std, param_seed

    raise LearnabilityGateError(
        f"Target family {target_family!r} failed learnability gate "
        f"(R² < {OLS_R2_MIN}) after {MAX_TEACHER_RETRIES} attempts."
    )


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------

def _generate_noise(
    rng: np.random.Generator,
    n: int,
    noise_type: str,
    noise_scale: float,
    betaX_calibrated: np.ndarray | None = None,
) -> np.ndarray:
    if noise_type == "gaussian":
        return noise_scale * rng.standard_normal(n)
    elif noise_type == "t3":
        # t(3) scaled to unit variance: Var(t_3) = 3/(3-2) = 3; sd = sqrt(3)
        raw = rng.standard_t(df=3, size=n)
        return noise_scale * raw / math.sqrt(3.0)
    elif noise_type == "heteroscedastic":
        if betaX_calibrated is None:
            raise ValueError("heteroscedastic noise requires betaX_calibrated")
        scale_vec = noise_scale * np.abs(betaX_calibrated)
        raw = rng.standard_normal(n)
        result = scale_vec * raw
        # Clip at 3σ of global scale
        clip_val = 3.0 * noise_scale
        return np.clip(result, -clip_val, clip_val)
    elif noise_type == "contaminated":
        mask = rng.uniform(size=n) < 0.1
        clean = rng.normal(0.0, noise_scale, size=n)
        outlier = rng.normal(0.0, 5.0 * noise_scale, size=n)
        return np.where(mask, outlier, clean)
    elif noise_type == "student_t":
        # Alias for t3: t(df=3) scaled to unit variance
        raw = rng.standard_t(df=3, size=n)
        return noise_scale * raw / math.sqrt(3.0)
    elif noise_type == "laplace":
        return noise_scale * rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=n)
    elif noise_type == "cauchy":
        raw = rng.standard_cauchy(size=n)
        return noise_scale * np.clip(raw, -20.0, 20.0)
    elif noise_type == "gross_outlier":
        # 95% clean Gaussian + 5% at 10x signal scale
        mask = rng.uniform(size=n) < 0.05
        clean = rng.normal(0.0, noise_scale, size=n)
        if betaX_calibrated is not None:
            outlier_scale = 10.0 * float(np.std(betaX_calibrated).clip(1e-8))
        else:
            outlier_scale = 10.0 * noise_scale
        outlier = rng.normal(0.0, outlier_scale, size=n)
        return np.where(mask, outlier, clean)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")


# ---------------------------------------------------------------------------
# Feature-regime parameter sampling
# ---------------------------------------------------------------------------

def _sample_feature_params(
    rng: np.random.Generator,
    feature_regime: str,
    n: int,
    p_signal: int,
    *,
    stress_kwargs: dict | None = None,
    p_noise_cap: int | None = None,
) -> dict:
    """Sample feature-level parameters for a given regime.

    For ar1/block/equicorr: rho is taken from stress_kwargs['rho'] if present,
    otherwise sampled from _RHO_GRID_CORR (always > 0).
    """
    stress_kwargs = stress_kwargs or {}

    def _corr_rho() -> float:
        """Return pre-specified rho (must be > 0) or sample from corr grid."""
        preset = stress_kwargs.get("rho")
        if preset is not None and float(preset) > 0:
            return float(preset)
        return float(rng.choice(_RHO_GRID_CORR))

    if feature_regime == "iid_dense":
        return {
            "covariance_type": "iid",
            "rho": 0.0,
            "active_fraction": 1.0,
            "noise_feature_fraction": 0.0,
            "feature_noise_sigma": 0.0,
            "p_noise": 0,
        }

    elif feature_regime == "iid_sparse":
        active_fraction = float(rng.choice([0.2, 0.3, 0.5, 0.7]))
        return {
            "covariance_type": "iid",
            "rho": 0.0,
            "active_fraction": active_fraction,
            "noise_feature_fraction": 0.0,
            "feature_noise_sigma": 0.0,
            "p_noise": 0,
        }

    elif feature_regime == "ar1":
        rho = _corr_rho()
        return {
            "covariance_type": "ar1",
            "rho": rho,
            "active_fraction": 1.0,
            "noise_feature_fraction": 0.0,
            "feature_noise_sigma": 0.0,
            "p_noise": 0,
        }

    elif feature_regime == "block":
        rho = _corr_rho()
        return {
            "covariance_type": "block",
            "rho": rho,
            "active_fraction": 1.0,
            "noise_feature_fraction": 0.0,
            "feature_noise_sigma": 0.0,
            "p_noise": 0,
        }

    elif feature_regime == "equicorr":
        rho = _corr_rho()
        return {
            "covariance_type": "equicorrelation",
            "rho": rho,
            "active_fraction": 1.0,
            "noise_feature_fraction": 0.0,
            "feature_noise_sigma": 0.0,
            "p_noise": 0,
        }

    elif feature_regime == "noise_feats":
        # p_noise = 2–3 × p_signal irrelevant features, capped by cursor budget
        multiplier = float(rng.choice([2.0, 2.5, 3.0]))
        p_noise = int(round(multiplier * p_signal))
        if p_noise_cap is not None:
            p_noise = max(0, min(p_noise, p_noise_cap))
        noise_feature_fraction = float(p_noise) / float(max(1, p_signal + p_noise))
        return {
            "covariance_type": "iid",
            "rho": 0.0,
            "active_fraction": 1.0,
            "noise_feature_fraction": noise_feature_fraction,
            "feature_noise_sigma": 0.0,
            "p_noise": p_noise,
        }

    elif feature_regime == "feat_noise":
        sigma = stress_kwargs.get("feature_noise_sigma", float(rng.choice([0.1, 0.25, 0.5])))
        return {
            "covariance_type": "iid",
            "rho": 0.0,
            "active_fraction": 1.0,
            "noise_feature_fraction": 0.0,
            "feature_noise_sigma": float(sigma),
            "p_noise": 0,
        }

    # --- v3 extended feature distributions (non-covariance-based) ---

    elif feature_regime in (
        "student_t_features", "laplace_features", "uniform_bounded",
        "skewed_lognormal", "mixture_gaussian", "low_rank_factor",
        "manifold_features",
    ):
        return {
            "covariance_type": "iid",          # not used by _generate_X_v3
            "rho": 0.0,
            "active_fraction": 1.0,
            "noise_feature_fraction": 0.0,
            "feature_noise_sigma": 0.0,
            "p_noise": 0,
            "feature_distribution": feature_regime,
        }

    else:
        raise ValueError(f"Unknown feature_regime: {feature_regime!r}")


# ---------------------------------------------------------------------------
# Sample complexity bucket (unchanged)
# ---------------------------------------------------------------------------

def _sample_complexity_bucket(n: int, p_total: int) -> str:
    ratio = float(n) / float(max(1, p_total))
    if ratio < 1.0:
        return "underdetermined"
    if ratio < 3.0:
        return "low_n"
    if ratio < 10.0:
        return "moderate_n"
    return "high_n"


# ---------------------------------------------------------------------------
# Core v2 dataset generator
# ---------------------------------------------------------------------------

def generate_v2_dataset(
    target_family: str,
    feature_regime: str,
    n: int,
    p_signal: int,
    target_noise_scale: float,
    target_noise_type: str,
    suite_component: str,
    teacher_seed: int,
    sample_seed: int,
    **kwargs,
) -> dict:
    """Generate one v2 nonlinear evaluation dataset.

    Dual-RNG contract:
      teacher_rng  — controls DGP parameters + calibration sample (X_cal)
      sample_rng   — controls evaluation rows (X_eval, noise)

    Returns a dict with 12 v1-compatible columns + 13 new metadata columns.
    feature_noise_level is always a Python float.
    """
    teacher_rng = np.random.default_rng(teacher_seed)
    sample_rng = np.random.default_rng(sample_seed)

    # Sample feature parameters (teacher_rng controls this)
    # Pass rho and feature_noise_sigma from kwargs so pre-specified cell values are honoured.
    # p_noise_cap (optional) limits noise-feature columns to stay within cursor budget.
    stress_kwargs = {k: v for k, v in kwargs.items() if k in ("feature_noise_sigma", "rho")}
    _p_noise_cap = int(kwargs["p_noise_cap"]) if "p_noise_cap" in kwargs else None
    feat_params = _sample_feature_params(
        teacher_rng, feature_regime, n, p_signal, stress_kwargs=stress_kwargs,
        p_noise_cap=_p_noise_cap,
    )
    covariance_type = feat_params["covariance_type"]
    rho = feat_params["rho"]
    active_fraction = feat_params["active_fraction"]
    feature_noise_sigma = float(feat_params["feature_noise_sigma"])
    p_noise = int(feat_params["p_noise"])
    p_total = p_signal + p_noise

    # Teacher-family params (family-specific kwargs)
    family_params: dict = {}
    if target_family == "mixed_linear":
        family_params["alpha"] = float(kwargs.get("alpha", teacher_rng.uniform(0.1, 0.9)))

    # --- Calibration phase (teacher_rng controls all of this) ---
    # Draw independent calibration sample
    X_cal = _generate_X(teacher_rng, N_CAL, p_signal, covariance_type, rho)
    if active_fraction < 1.0:
        active_cols = _select_active_columns(teacher_rng, p_signal, active_fraction)
        Xa_cal = X_cal[:, active_cols]
    else:
        active_cols = np.arange(p_signal)
        Xa_cal = X_cal

    cal_mean, cal_std, teacher_param_seed = _calibrate_teacher(
        teacher_rng, target_family, Xa_cal, family_params
    )

    # --- Evaluation phase (sample_rng controls all of this) ---
    X_signal_clean = _generate_X(sample_rng, n, p_signal, covariance_type, rho)

    # Active-column mask
    if active_fraction < 1.0:
        Xa_eval = X_signal_clean[:, active_cols]
    else:
        Xa_eval = X_signal_clean

    # Teacher function call — reuse the SAME param_seed from calibration so the
    # teacher weights are identical for cal and eval (eliminates cal/eval param mismatch)
    teacher_fn = _TEACHER_REGISTRY[target_family]
    teacher_sub_rng = np.random.default_rng(teacher_param_seed)
    betaX_raw = teacher_fn(teacher_sub_rng, Xa_eval, family_params)
    betaX_raw = np.asarray(betaX_raw, dtype=np.float64)

    # Calibrate: unit-std signal, no holdout leakage
    betaX = (betaX_raw - cal_mean) / cal_std

    # Noise
    noise = _generate_noise(
        sample_rng, n, target_noise_type, target_noise_scale,
        betaX_calibrated=betaX if target_noise_type == "heteroscedastic" else None,
    )
    y = betaX + noise

    # Add noise features if regime requires it
    if p_noise > 0:
        X_noise_cols = _generate_X(sample_rng, n, p_noise, "iid", 0.0)
        X = np.concatenate([X_signal_clean, X_noise_cols], axis=1)
    else:
        X = X_signal_clean

    # Add feature measurement noise
    if feature_noise_sigma > 0.0:
        X = X + feature_noise_sigma * sample_rng.standard_normal(X.shape)

    # SNR estimation (signal variance / noise variance)
    signal_var = float(np.var(betaX))
    noise_var = float(np.var(noise))
    snr_target = signal_var / (noise_var + 1e-30) if noise_var > 1e-30 else float("inf")

    # Complexity / condition id
    condition_id = (
        f"{target_family}_{feature_regime}_n{n}_p{p_signal}"
        f"_noise{target_noise_type}_scale{target_noise_scale}"
    )

    return {
        # v1-compatible columns
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "suite_family": "primary",
        "prior_regime": target_family,
        "n_total": int(n),
        "p_signal": int(p_signal),
        "p_noise": int(p_noise),
        "p_total": int(p_total),
        "target_noise_scale": float(target_noise_scale),
        "training_size_anchor": False,
        "feature_noise_level": float(feature_noise_sigma),  # always float
        # 13 new metadata columns
        "feature_regime": feature_regime,
        "covariance_type": covariance_type,
        "rho": float(rho),
        "active_fraction": float(active_fraction),
        "noise_feature_fraction": float(feat_params["noise_feature_fraction"]),
        "feature_noise_sigma": float(feature_noise_sigma),
        "suite_component": suite_component,
        "target_noise_type": target_noise_type,
        "snr_target": float(snr_target),
        "condition_id": condition_id,
        "teacher_seed": int(teacher_seed),
        "sample_seed": int(sample_seed),
        "normalization_constant": float(cal_std),
    }


# ---------------------------------------------------------------------------
# Suite cell enumerator — 420 datasets total
# ---------------------------------------------------------------------------

def enumerate_suite_cells(base_seed: int = BASE_SEED) -> list[dict]:
    """Return a deterministic list of 420 cell specifications.

    Component A: 300 core datasets (6 families × 7 regimes × 7 datasets/cell)
    Component B: 80 robustness/stress datasets
    Component C: 40 structural OOD datasets
    """
    cells: list[dict] = []
    rng_meta = np.random.default_rng(base_seed)

    # -----------------------------------------------------------------------
    # Component A — Core (300 datasets)
    # -----------------------------------------------------------------------
    # Round-robin over (n, p_signal, noise_scale, noise_type) within each cell
    _n_cycle = V2_N_GRID
    _p_cycle = V2_P_SIGNAL_GRID
    _ns_cycle = V2_TARGET_NOISE_SCALES
    _nt_cycle = V2_TARGET_NOISE_TYPES

    # 42 cells × 7 ≈ 294; first 6 cells get 8 datasets, rest get 7 → 300 total
    target_per_cell = 300 // (len(V2_TARGET_FAMILIES) * len(V2_FEATURE_REGIMES))  # = 7
    remainder_cells = 300 % (len(V2_TARGET_FAMILIES) * len(V2_FEATURE_REGIMES))   # = 6

    cell_idx_global = 0
    for f_idx, family in enumerate(V2_TARGET_FAMILIES):
        for r_idx, regime in enumerate(V2_FEATURE_REGIMES):
            n_this_cell = target_per_cell + (1 if cell_idx_global < remainder_cells else 0)
            cell_idx_global += 1

            for ds_in_cell in range(n_this_cell):
                # Deterministic round-robin over grids
                n = _n_cycle[ds_in_cell % len(_n_cycle)]
                p_signal = _p_cycle[ds_in_cell % len(_p_cycle)]
                noise_scale = _ns_cycle[ds_in_cell % len(_ns_cycle)]
                noise_type = _nt_cycle[ds_in_cell % len(_nt_cycle)]

                # Seed derivation: unique per (family, regime, dataset_in_cell)
                teacher_seed = int(
                    np.random.default_rng([base_seed, f_idx, r_idx, ds_in_cell, 0]).integers(2**62)
                )
                sample_seed = int(
                    np.random.default_rng([base_seed, f_idx, r_idx, ds_in_cell, 1]).integers(2**62)
                )
                idx = len(cells)
                cell: dict = {
                    "target_family": family,
                    "feature_regime": regime,
                    "n": n,
                    "p_signal": p_signal,
                    "target_noise_scale": noise_scale,
                    "target_noise_type": noise_type,
                    "suite_component": "core",
                    "teacher_seed": teacher_seed,
                    "sample_seed": sample_seed,
                    "logical_dataset_key": f"nonlinear:{family}:{idx:05d}",
                    "condition_id": (
                        f"{family}_{regime}_n{n}_p{p_signal}"
                        f"_noise{noise_type}_scale{noise_scale}"
                    ),
                }
                # Pre-embed rho for correlated regimes so smoke-checks pass
                if regime in ("ar1", "block", "equicorr"):
                    rho_rng = np.random.default_rng([base_seed, f_idx, r_idx, ds_in_cell, 2])
                    cell["rho"] = float(rho_rng.choice(_RHO_GRID_CORR))
                cells.append(cell)

    # -----------------------------------------------------------------------
    # Component B — Robustness/Stress (exactly 80 datasets)
    # Flat index: i=0..79; family = V2_TARGET_FAMILIES[i % 6]
    #             stress = _STRESS_B[(i // 6) % len(_STRESS_B)]
    # -----------------------------------------------------------------------
    _STRESS_B = [
        ("contaminated_high_rho", {"target_noise_type": "contaminated", "feature_regime": "ar1"}),
        ("tiny_n_high_p",         {"n": 50, "p_signal": 64, "feature_regime": "iid_dense"}),
        ("large_n",               {"n": 3000, "feature_regime": "iid_dense"}),
        ("heavy_feat_noise",      {"feature_regime": "feat_noise", "feature_noise_sigma": 0.5}),
        ("noise_feat_heavy",      {"feature_regime": "noise_feats"}),
        ("t3_noise_only",         {"target_noise_type": "t3", "feature_regime": "iid_dense"}),
        ("high_rho_block",        {"feature_regime": "block"}),
        ("equicorr_stress",       {"feature_regime": "equicorr"}),
    ]

    for i in range(80):
        f_idx = i % len(V2_TARGET_FAMILIES)
        s_idx = (i // len(V2_TARGET_FAMILIES)) % len(_STRESS_B)
        family = V2_TARGET_FAMILIES[f_idx]
        stress_label, stress_kw = _STRESS_B[s_idx]

        n = int(stress_kw.get("n", 300))
        p_signal = int(stress_kw.get("p_signal", 16))
        noise_scale = 1.0
        noise_type = str(stress_kw.get("target_noise_type", "gaussian"))
        regime = str(stress_kw.get("feature_regime", "iid_dense"))
        extra_kw = {k: v for k, v in stress_kw.items()
                    if k not in ("n", "p_signal", "target_noise_type", "feature_regime")}

        teacher_seed = int(
            np.random.default_rng([base_seed + 1000, f_idx, s_idx, i, 0]).integers(2**62)
        )
        sample_seed = int(
            np.random.default_rng([base_seed + 1000, f_idx, s_idx, i, 1]).integers(2**62)
        )
        idx = len(cells)
        b_cell: dict = {
            "target_family": family,
            "feature_regime": regime,
            "n": n,
            "p_signal": p_signal,
            "target_noise_scale": noise_scale,
            "target_noise_type": noise_type,
            "suite_component": "robustness",
            "teacher_seed": teacher_seed,
            "sample_seed": sample_seed,
            "logical_dataset_key": f"nonlinear:{family}:{idx:05d}",
            "condition_id": (
                f"{family}_{regime}_n{n}_p{p_signal}"
                f"_noise{noise_type}_scale{noise_scale}_{stress_label}_{i}"
            ),
            **extra_kw,
        }
        # Pre-embed rho for correlated regimes
        if regime in ("ar1", "block", "equicorr"):
            rho_rng = np.random.default_rng([base_seed + 1000, f_idx, s_idx, i, 2])
            b_cell["rho"] = float(rho_rng.choice(_RHO_GRID_CORR))
        cells.append(b_cell)

    # -----------------------------------------------------------------------
    # Component C — Structural OOD (40 datasets)
    # -----------------------------------------------------------------------
    _ood_configs = [
        ("demand_high_cross_elasticity", "demand_mono",    8, {"n": 300, "p_signal": 16}),
        ("sparse_pure",                  "sparse_interact", 8, {"n": 300, "p_signal": 32}),
        ("equicorr_high_p",              "poly_quad",       8, {"n": 300, "p_signal": 64, "feature_regime": "equicorr"}),
        ("heteroscedastic_extreme",      "hinge",           8, {"n": 300, "p_signal": 16, "target_noise_type": "heteroscedastic", "target_noise_scale": 2.0}),
        ("mixed_high_linear_alpha",      "mixed_linear",    8, {"n": 300, "p_signal": 16, "alpha": 0.9}),
    ]

    for ood_label, ood_family, ood_count, ood_kw in _ood_configs:
        for rep in range(ood_count):
            n = int(ood_kw.get("n", 300))
            p_signal = int(ood_kw.get("p_signal", 16))
            noise_scale = float(ood_kw.get("target_noise_scale", 1.0))
            noise_type = str(ood_kw.get("target_noise_type", "gaussian"))
            regime = str(ood_kw.get("feature_regime", "iid_dense"))
            extra_kw = {k: v for k, v in ood_kw.items()
                        if k not in ("n", "p_signal", "target_noise_type", "target_noise_scale", "feature_regime")}

            teacher_seed = int(
                np.random.default_rng(
                    [base_seed + 2000, _ood_configs.index((ood_label, ood_family, ood_count, ood_kw)), rep, 0]
                ).integers(2**62)
            )
            sample_seed = int(
                np.random.default_rng(
                    [base_seed + 2000, _ood_configs.index((ood_label, ood_family, ood_count, ood_kw)), rep, 1]
                ).integers(2**62)
            )
            idx = len(cells)
            c_cell: dict = {
                "target_family": ood_family,
                "feature_regime": regime,
                "n": n,
                "p_signal": p_signal,
                "target_noise_scale": noise_scale,
                "target_noise_type": noise_type,
                "suite_component": "ood",
                "teacher_seed": teacher_seed,
                "sample_seed": sample_seed,
                "logical_dataset_key": f"nonlinear:{ood_family}:{idx:05d}",
                "condition_id": (
                    f"{ood_family}_{regime}_n{n}_p{p_signal}"
                    f"_noise{noise_type}_scale{noise_scale}_{ood_label}_{rep}"
                ),
                **extra_kw,
            }
            # Pre-embed rho for correlated regimes
            if regime in ("ar1", "block", "equicorr"):
                rho_rng = np.random.default_rng(
                    [base_seed + 2000, _ood_configs.index((ood_label, ood_family, ood_count, ood_kw)), rep, 2]
                )
                c_cell["rho"] = float(rho_rng.choice(_RHO_GRID_CORR))
            cells.append(c_cell)

    return cells


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------

def write_parquet(ds: dict, filepath: str) -> None:
    """Write TRAINING format parquet (X_train, y_train, X_test, betaX_test).

    Preserves the schema expected by build_meta_nonlinear_regression_dataset_index.py.
    This is the DeepSet pretraining format.
    """
    X = ds["X"]
    y = ds["y"]
    n = len(y)
    n_train = int(0.8 * n)
    if n - n_train < 1:
        n_train = n - 1

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    betaX_test = ds["betaX"][n_train:]

    p_signal = int(ds.get("p_signal", X.shape[1]))
    p_noise = int(ds.get("p_noise", 0))
    p_total = int(ds.get("p_total", X.shape[1]))

    table = pa.table({
        "X_train":    pa.array([X_train.tolist()],   type=pa.list_(pa.list_(pa.float64()))),
        "y_train":    pa.array([y_train.tolist()],   type=pa.list_(pa.float64())),
        "X_test":     pa.array([X_test.tolist()],    type=pa.list_(pa.list_(pa.float64()))),
        "betaX_test": pa.array([betaX_test.tolist()], type=pa.list_(pa.float64())),
        "n":          pa.array([n],                  type=pa.int64()),
        "p":          pa.array([p_total],            type=pa.int64()),
        "n_train":    pa.array([len(X_train)],       type=pa.int64()),
        "n_test":     pa.array([len(X_test)],        type=pa.int64()),
        "prior_regime": pa.array([ds["prior_regime"]], type=pa.utf8()),
        "profile":    pa.array(["v2"],               type=pa.utf8()),
        "feature_regime": pa.array([ds.get("feature_regime", "iid_dense")], type=pa.utf8()),
        "p_signal":   pa.array([p_signal],           type=pa.int64()),
        "p_noise":    pa.array([p_noise],            type=pa.int64()),
        "p_total":    pa.array([p_total],            type=pa.int64()),
        "active_s":   pa.array([p_signal],           type=pa.int64()),
        "sparsity_ratio": pa.array([1.0],            type=pa.float64()),
        "covariance_type": pa.array([ds.get("covariance_type", "iid")], type=pa.utf8()),
        "rho":        pa.array([float(ds.get("rho", 0.0))], type=pa.float64()),
        "target_noise_scale": pa.array([float(ds.get("target_noise_scale", 1.0))], type=pa.float64()),
        "feature_noise_level": pa.array([float(ds.get("feature_noise_level", 0.0))], type=pa.float64()),
        "sample_complexity_bucket": pa.array(
            [_sample_complexity_bucket(n, p_total)], type=pa.utf8()
        ),
        "has_noise_features": pa.array([bool(p_noise > 0)], type=pa.bool_()),
        "has_feature_noise": pa.array([bool(ds.get("feature_noise_level", 0.0) > 0.0)], type=pa.bool_()),
    })
    pq.write_table(table, filepath)


def write_parquet_eval(ds: dict, filepath: str) -> int:
    """Write EVALUATION format parquet (X, y, betaX full dataset + 13 new metadata cols).

    feature_noise_level is written as pa.float64() (never int).
    Returns the number of rows written (always 1 — one meta-row per dataset).
    """
    X = ds["X"]
    y = ds["y"]
    betaX = ds["betaX"]

    table = pa.table({
        # Core arrays
        "X":     pa.array([X.tolist()],     type=pa.list_(pa.list_(pa.float64()))),
        "y":     pa.array([y.tolist()],     type=pa.list_(pa.float64())),
        "betaX": pa.array([betaX.tolist()], type=pa.list_(pa.float64())),
        # v1-compatible scalar columns
        "suite_family":         pa.array([ds.get("suite_family", "primary")], type=pa.utf8()),
        "prior_regime":         pa.array([ds["prior_regime"]],               type=pa.utf8()),
        "n_total":              pa.array([int(ds["n_total"])],               type=pa.int64()),
        "p_signal":             pa.array([int(ds["p_signal"])],              type=pa.int64()),
        "p_noise":              pa.array([int(ds["p_noise"])],               type=pa.int64()),
        "p_total":              pa.array([int(ds["p_total"])],               type=pa.int64()),
        "target_noise_scale":   pa.array([float(ds["target_noise_scale"])],  type=pa.float64()),
        "training_size_anchor": pa.array([bool(ds.get("training_size_anchor", False))], type=pa.bool_()),
        "feature_noise_level":  pa.array([float(ds["feature_noise_level"])], type=pa.float64()),
        # 13 new metadata columns
        "feature_regime":           pa.array([ds["feature_regime"]],                   type=pa.utf8()),
        "covariance_type":          pa.array([ds["covariance_type"]],                  type=pa.utf8()),
        "rho":                      pa.array([float(ds["rho"])],                       type=pa.float64()),
        "active_fraction":          pa.array([float(ds["active_fraction"])],           type=pa.float64()),
        "noise_feature_fraction":   pa.array([float(ds["noise_feature_fraction"])],    type=pa.float64()),
        "feature_noise_sigma":      pa.array([float(ds["feature_noise_sigma"])],       type=pa.float64()),
        "suite_component":          pa.array([ds["suite_component"]],                  type=pa.utf8()),
        "target_noise_type":        pa.array([ds["target_noise_type"]],                type=pa.utf8()),
        "snr_target":               pa.array([float(ds["snr_target"])],               type=pa.float64()),
        "condition_id":             pa.array([ds["condition_id"]],                     type=pa.utf8()),
        "teacher_seed":             pa.array([int(ds["teacher_seed"])],                type=pa.int64()),
        "sample_seed":              pa.array([int(ds["sample_seed"])],                 type=pa.int64()),
        "normalization_constant":   pa.array([float(ds["normalization_constant"])],    type=pa.float64()),
    })
    pq.write_table(table, filepath)
    return 1


# ---------------------------------------------------------------------------
# v1-compat helpers (kept for backward compat with test files)
# ---------------------------------------------------------------------------

def _balanced_pair(idx: int, feature_regimes: list[str]) -> tuple[str, str]:
    """Legacy v1 balanced pair selector — kept for backward compat."""
    combos = [
        (target, feature)
        for target in ("I", "J", "K", "L")
        for feature in feature_regimes
    ]
    return combos[idx % len(combos)]


# ---------------------------------------------------------------------------
# main() — generate TRAINING datasets in v2 format
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 nonlinear DGP training datasets.")
    parser.add_argument("--n_datasets", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="data/nonlinear/")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--target_families",
        type=str,
        default=",".join(V2_TARGET_FAMILIES),
        help="Comma-separated subset of target families to generate.",
    )
    args = parser.parse_args()

    families = [f.strip() for f in args.target_families.split(",") if f.strip()]
    unknown = set(families) - set(V2_TARGET_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown target families: {unknown}")

    n_datasets = int(args.n_datasets)
    n_train_split = int(0.8 * n_datasets)
    n_val_split = int(0.1 * n_datasets)

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    test_dir = os.path.join(args.out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    base_seed = int(args.seed)
    family_rng = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    regime_rng = np.random.default_rng(base_seed ^ 0x5A5A5A5A)

    print(f"Generating {n_datasets} v2 training datasets -> {args.out_dir}")
    print(f"  target_families={families}")
    print(f"  feature_regimes={V2_FEATURE_REGIMES}")

    # Use separate attempt / n_written counters so LearnabilityGateError skips are
    # retried (up to 5× n_datasets attempts) and never create permanent file holes.
    # Mirrors the while-loop pattern used for mixed/classification paths below.
    n_written = 0
    attempt = 0
    max_attempts = n_datasets * 5

    while n_written < n_datasets and attempt < max_attempts:
        # `attempt` drives all seed derivation so every retry gets a fresh draw.
        slot = attempt
        attempt += 1

        target_family = str(family_rng.choice(families))
        feature_regime = str(regime_rng.choice(V2_FEATURE_REGIMES))

        # cursor_scale_v1 dim draw (replaces V2_N_GRID / V2_P_SIGNAL_GRID static grids)
        _dim_rng = np.random.default_rng(
            int(np.random.SeedSequence([base_seed, slot, 0x5D4E4C52]).generate_state(1)[0])
        )
        _is_tail = _dim_rng.random() < _CURSOR_TAIL_FRACTION
        if _is_tail:
            # Wide-p tail: Poisson(25) bounded [20, 30]; n ≥ 5p
            while True:
                _p_total = int(max(_CURSOR_TAIL_P_FLOOR,
                                   min(int(_dim_rng.poisson(_CURSOR_TAIL_P_LAMBDA)),
                                       _CURSOR_TAIL_P_CAP)))
                _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                if _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                    break
            # Full 5% tail → high_dim_sparse_nonlinear
            target_family = "high_dim_sparse_nonlinear"
        else:
            # Main 95%: Poisson(10) dims; n ≥ 5p
            while True:
                _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
                _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                if _p_total >= 1 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                    break
        n = _n
        p_signal = max(1, int(_dim_rng.integers(1, _p_total + 1)))
        p_noise_cap = _p_total - p_signal  # budget remaining for noise features

        noise_scale = float(np.random.default_rng(base_seed + slot + 2).choice(V2_TARGET_NOISE_SCALES))
        noise_type = str(np.random.default_rng(base_seed + slot + 3).choice(V2_TARGET_NOISE_TYPES))

        teacher_seed = int(np.random.default_rng([base_seed, slot, 0]).integers(2**62))
        sample_seed = int(np.random.default_rng([base_seed, slot, 1]).integers(2**62))

        try:
            ds = generate_v2_dataset(
                target_family=target_family,
                feature_regime=feature_regime,
                n=n,
                p_signal=p_signal,
                target_noise_scale=noise_scale,
                target_noise_type=noise_type,
                suite_component="train",
                teacher_seed=teacher_seed,
                sample_seed=sample_seed,
                p_noise_cap=p_noise_cap,
            )
        except LearnabilityGateError:
            print(f"  [slot {slot:04d}] LearnabilityGateError — skipping {target_family}/{feature_regime}")
            continue

        # File numbering is based on n_written (the success counter), not the
        # attempt slot, so no gaps appear in the output even after retries.
        if n_written < n_train_split:
            split_dir = train_dir
            filename = f"dataset_{n_written:04d}.parquet"
        elif n_written < n_train_split + n_val_split:
            split_dir = val_dir
            filename = f"dataset_{n_written - n_train_split:04d}.parquet"
        else:
            split_dir = test_dir
            filename = f"dataset_{n_written - n_train_split - n_val_split:04d}.parquet"

        write_parquet(ds, os.path.join(split_dir, filename))
        n_written += 1
        if n_written % 100 == 0:
            print(f"  [{n_written:4d}/{n_datasets}] {filename} -> {split_dir}")

    if n_written < n_datasets:
        print(
            f"WARNING: only {n_written}/{n_datasets} datasets generated after {attempt} attempts "
            "(LearnabilityGateError on some configs). Increase --n_datasets or review parameter grids."
        )
    else:
        print(f"Done. Generated {n_written} datasets -> {args.out_dir}")


# ---------------------------------------------------------------------------
# Training-format writers for nonlinear classification and mixed families
# ---------------------------------------------------------------------------

def write_parquet_training_classification(ds: dict, filepath: str) -> None:
    """Write classification TRAINING format parquet (X_train, y_train, X_test + class metadata).

    Preserves the schema expected by build_meta_nonlinear_classification_dataset_index.py.
    classification_regime is set to prior_regime (= nonlinear_family) so the builder's
    required-column check passes.
    """
    X = ds["X"]
    y = ds["y"]
    n = len(y)
    n_train = int(0.8 * n)
    if n - n_train < 1:
        n_train = n - 1

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]

    p_signal = int(ds.get("p_signal", X.shape[1]))
    p_noise = int(ds.get("p_noise", 0))
    p_total = int(ds.get("p_total", X.shape[1]))

    table = pa.table({
        "X_train":    pa.array([X_train.tolist()],   type=pa.list_(pa.list_(pa.float64()))),
        "y_train":    pa.array([y_train.tolist()],   type=pa.list_(pa.int64())),
        "X_test":     pa.array([X_test.tolist()],    type=pa.list_(pa.list_(pa.float64()))),
        "y_test":     pa.array([y[n_train:].tolist()], type=pa.list_(pa.int64())),
        "n":          pa.array([n],                  type=pa.int64()),
        "p":          pa.array([p_total],            type=pa.int64()),
        "n_train":    pa.array([len(X_train)],       type=pa.int64()),
        "n_test":     pa.array([len(X_test)],        type=pa.int64()),
        "prior_regime": pa.array([ds["prior_regime"]], type=pa.utf8()),
        # classification_regime = nonlinear_family (same as prior_regime)
        "classification_regime": pa.array([str(ds.get("nonlinear_family", ds["prior_regime"]))], type=pa.utf8()),
        "num_classes": pa.array([int(ds["num_classes"])], type=pa.int64()),
        "task_objective": pa.array([str(ds.get("task_objective", "inductive_classification"))], type=pa.utf8()),
        "p_signal":   pa.array([p_signal],           type=pa.int64()),
        "p_noise":    pa.array([p_noise],            type=pa.int64()),
        "p_total":    pa.array([p_total],            type=pa.int64()),
        "class_imbalance_type": pa.array([str(ds.get("class_imbalance_type", "balanced"))], type=pa.utf8()),
        "label_noise_rate": pa.array([float(ds.get("label_noise_rate", 0.0))], type=pa.float64()),
        "feature_noise_level": pa.array([float(ds.get("feature_noise_sigma", 0.0))], type=pa.float64()),
        "task_family": pa.array([str(ds.get("task_family", "synthetic_nonlinear_classification"))], type=pa.utf8()),
        "schema_version": pa.array([str(ds.get("schema_version", "nonlinear_classification_v1"))], type=pa.utf8()),
    })
    pq.write_table(table, filepath)


def write_parquet_training_mixed_regression(ds: dict, filepath: str) -> None:
    """Write mixed-categorical regression TRAINING format parquet.

    Preserves the schema expected by build_meta_nonlinear_mixed_regression_dataset_index.py
    and load_mixed_regression_parquet (train.py).  Column names follow the linear-mixed
    contract: X_num_train / X_num_test, y_test, and all three categorical masks.
    """
    from dgp_helpers import mark_unseen_query_categories

    X = ds["X"]
    X_cat = ds["X_cat"]
    y = ds["y"]
    cat_missing_mask = ds["cat_missing_mask"]  # (n, p_cat) bool — set by generator
    n = len(y)
    n_train = int(0.8 * n)
    if n - n_train < 1:
        n_train = n - 1

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    X_cat_train = X_cat[:n_train]
    X_cat_test = X_cat[n_train:]
    cat_missing_mask_train = cat_missing_mask[:n_train]
    cat_missing_mask_test = cat_missing_mask[n_train:]
    cat_unknown_mask_test = mark_unseen_query_categories(X_cat_train, X_cat_test)

    p_num = int(X.shape[1])
    p_cat = int(X_cat.shape[1])
    cardinalities = list(ds.get("categorical_cardinalities", []))

    table = pa.table({
        "X_num_train": pa.array([X_train.tolist()],       type=pa.list_(pa.list_(pa.float64()))),
        "y_train":     pa.array([y_train.tolist()],       type=pa.list_(pa.float64())),
        "X_num_test":  pa.array([X_test.tolist()],        type=pa.list_(pa.list_(pa.float64()))),
        "y_test":      pa.array([y[n_train:].tolist()],   type=pa.list_(pa.float64())),
        "betaX_test":  pa.array([ds["betaX"][n_train:].tolist()], type=pa.list_(pa.float64())),
        "X_cat_train": pa.array([X_cat_train.tolist()],   type=pa.list_(pa.list_(pa.int64()))),
        "X_cat_test":  pa.array([X_cat_test.tolist()],    type=pa.list_(pa.list_(pa.int64()))),
        "cat_missing_mask_train": pa.array([cat_missing_mask_train.tolist()], type=pa.list_(pa.list_(pa.bool_()))),
        "cat_missing_mask_test":  pa.array([cat_missing_mask_test.tolist()],  type=pa.list_(pa.list_(pa.bool_()))),
        "cat_unknown_mask_test":  pa.array([cat_unknown_mask_test.tolist()],  type=pa.list_(pa.list_(pa.bool_()))),
        "n":           pa.array([n],                      type=pa.int64()),
        "p_num":       pa.array([p_num],                  type=pa.int64()),
        "p_cat":       pa.array([p_cat],                  type=pa.int64()),
        "n_train":     pa.array([len(X_train)],           type=pa.int64()),
        "n_test":      pa.array([len(X_test)],            type=pa.int64()),
        "prior_regime": pa.array([ds["prior_regime"]],    type=pa.utf8()),
        "categorical_cardinalities": pa.array([cardinalities], type=pa.list_(pa.int64())),
        "task_family": pa.array([str(ds.get("task_family", "synthetic_nonlinear_regression_mixed_categorical"))], type=pa.utf8()),
        "training_data_family": pa.array([str(ds.get("training_data_family", "synthetic_nonlinear_regression_mixed_categorical"))], type=pa.utf8()),
        "task_objective": pa.array([str(ds.get("task_objective", "inductive_regression"))], type=pa.utf8()),
        "schema_version": pa.array([str(ds.get("schema_version", "nonlinear_regression_mixed_categorical_v1"))], type=pa.utf8()),
    })
    pq.write_table(table, filepath)


def write_parquet_training_mixed_classification(ds: dict, filepath: str) -> None:
    """Write mixed-categorical classification TRAINING format parquet.

    Preserves the schema expected by build_meta_nonlinear_mixed_classification_dataset_index.py
    and load_mixed_classification_parquet (train.py).  Column names follow the linear-mixed
    contract: X_num_train / X_num_test and all three categorical masks.
    """
    from dgp_helpers import mark_unseen_query_categories

    X = ds["X"]
    X_cat = ds["X_cat"]
    y = ds["y"]
    cat_missing_mask = ds["cat_missing_mask"]  # (n, p_cat) bool — set by generator
    n = len(y)
    n_train = int(0.8 * n)
    if n - n_train < 1:
        n_train = n - 1

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    X_cat_train = X_cat[:n_train]
    X_cat_test = X_cat[n_train:]
    cat_missing_mask_train = cat_missing_mask[:n_train]
    cat_missing_mask_test = cat_missing_mask[n_train:]
    cat_unknown_mask_test = mark_unseen_query_categories(X_cat_train, X_cat_test)

    p_num = int(X.shape[1])
    p_cat = int(X_cat.shape[1])
    cardinalities = list(ds.get("categorical_cardinalities", []))

    table = pa.table({
        "X_num_train": pa.array([X_train.tolist()],       type=pa.list_(pa.list_(pa.float64()))),
        "y_train":     pa.array([y_train.tolist()],       type=pa.list_(pa.int64())),
        "X_num_test":  pa.array([X_test.tolist()],        type=pa.list_(pa.list_(pa.float64()))),
        "y_test":      pa.array([y[n_train:].tolist()],   type=pa.list_(pa.int64())),
        "X_cat_train": pa.array([X_cat_train.tolist()],   type=pa.list_(pa.list_(pa.int64()))),
        "X_cat_test":  pa.array([X_cat_test.tolist()],    type=pa.list_(pa.list_(pa.int64()))),
        "cat_missing_mask_train": pa.array([cat_missing_mask_train.tolist()], type=pa.list_(pa.list_(pa.bool_()))),
        "cat_missing_mask_test":  pa.array([cat_missing_mask_test.tolist()],  type=pa.list_(pa.list_(pa.bool_()))),
        "cat_unknown_mask_test":  pa.array([cat_unknown_mask_test.tolist()],  type=pa.list_(pa.list_(pa.bool_()))),
        "n":           pa.array([n],                      type=pa.int64()),
        "p_num":       pa.array([p_num],                  type=pa.int64()),
        "p_cat":       pa.array([p_cat],                  type=pa.int64()),
        "n_train":     pa.array([len(X_train)],           type=pa.int64()),
        "n_test":      pa.array([len(X_test)],            type=pa.int64()),
        "prior_regime": pa.array([ds["prior_regime"]],    type=pa.utf8()),
        "num_classes": pa.array([int(ds["num_classes"])], type=pa.int64()),
        "task_objective": pa.array([str(ds.get("task_objective", "inductive_classification"))], type=pa.utf8()),
        "classification_regime": pa.array([str(ds.get("nonlinear_family", ds["prior_regime"]))], type=pa.utf8()),
        "categorical_cardinalities": pa.array([cardinalities], type=pa.list_(pa.int64())),
        "class_imbalance_type": pa.array([str(ds.get("class_imbalance_type", "balanced"))], type=pa.utf8()),
        "label_noise_rate": pa.array([float(ds.get("label_noise_rate", 0.0))], type=pa.float64()),
        "feature_noise_level": pa.array([float(ds.get("feature_noise_sigma", 0.0))], type=pa.float64()),
        "temperature": pa.array([float(ds.get("temperature", 1.0))], type=pa.float64()),
        "task_family": pa.array([str(ds.get("task_family", "synthetic_nonlinear_classification_mixed_categorical"))], type=pa.utf8()),
        "training_data_family": pa.array([str(ds.get("training_data_family", "synthetic_nonlinear_classification_mixed_categorical"))], type=pa.utf8()),
        "schema_version": pa.array([str(ds.get("schema_version", "nonlinear_classification_mixed_categorical_v1"))], type=pa.utf8()),
    })
    pq.write_table(table, filepath)


# ---------------------------------------------------------------------------
# Training-format generation entry point for all 4 nonlinear task families
# ---------------------------------------------------------------------------

_NONLINEAR_TRAINING_TASK_FAMILIES = (
    "synthetic_nonlinear_regression",
    "synthetic_nonlinear_classification",
    "synthetic_nonlinear_regression_mixed_categorical",
    "synthetic_nonlinear_classification_mixed_categorical",
)

# Classification parameter grids for training data generation
_CLS_NUM_CLASSES_GRID = [2, 3, 5, 10]
_CLS_TEMPERATURE_GRID = [0.5, 1.0, 2.0]
_CLS_LABEL_NOISE_GRID = [0.0, 0.05, 0.1]
_CLS_IMBALANCE_GRID   = ["balanced", "mild_imbalance"]
_CLS_MARGIN_GRID      = ["low", "medium", "high"]

# Mixed-categorical parameter grids
_MIXED_P_CAT_GRID     = [2, 3, 5]
_MIXED_CAT_EFFECT_GRID = [0.5, 1.0, 2.0]


def main_nonlinear_training() -> None:
    """Generate TRAINING datasets for one of the 4 nonlinear task families.

    Usage examples:
        python generate_nonlinear_dgp.py --task_family synthetic_nonlinear_classification \\
            --n_datasets 1000 --out_dir data/nonlinear_cls/

        python generate_nonlinear_dgp.py --task_family synthetic_nonlinear_regression_mixed_categorical \\
            --n_datasets 1000 --out_dir data/nonlinear_mixed_reg/

        python generate_nonlinear_dgp.py --task_family synthetic_nonlinear_classification_mixed_categorical \\
            --n_datasets 1000 --out_dir data/nonlinear_mixed_cls/

    The original main() handles task_family=synthetic_nonlinear_regression (v2 regression).
    """
    parser = argparse.ArgumentParser(
        description="Generate nonlinear DGP training datasets for a specific task family."
    )
    parser.add_argument(
        "--task_family",
        type=str,
        default="synthetic_nonlinear_regression",
        choices=list(_NONLINEAR_TRAINING_TASK_FAMILIES),
        help="Target task family to generate training data for.",
    )
    parser.add_argument("--n_datasets", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="data/nonlinear_training/")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()

    task_family = args.task_family
    if task_family == "synthetic_nonlinear_regression":
        # Delegate to existing main() for backward compatibility
        main()
        return

    n_datasets = int(args.n_datasets)
    n_train_split = int(0.8 * n_datasets)
    n_val_split = int(0.1 * n_datasets)

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    test_dir = os.path.join(args.out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    base_seed = int(args.seed)

    # Select seed magic and writer based on task family
    if task_family == "synthetic_nonlinear_classification":
        seed_magic = _NONLINEAR_CLS_SEED_MAGIC
        writer_fn = write_parquet_training_classification
        families = [f for f in V3_CLASSIFICATION_FAMILIES if f != "mixed_categorical_nonlinear"]
        feature_regimes = V2_FEATURE_REGIMES
    elif task_family == "synthetic_nonlinear_regression_mixed_categorical":
        seed_magic = _NONLINEAR_MIXED_REG_SEED_MAGIC
        writer_fn = write_parquet_training_mixed_regression
        families = V3_TARGET_FAMILIES
        feature_regimes = V2_FEATURE_REGIMES
    elif task_family == "synthetic_nonlinear_classification_mixed_categorical":
        seed_magic = _NONLINEAR_MIXED_CLS_SEED_MAGIC
        writer_fn = write_parquet_training_mixed_classification
        # Use all classification families including mixed_categorical_nonlinear
        families = list(V3_CLASSIFICATION_FAMILIES)
        feature_regimes = V2_FEATURE_REGIMES
    else:
        raise ValueError(f"Unknown task_family: {task_family!r}")

    family_rng = np.random.default_rng((base_seed ^ seed_magic) & 0xFFFFFFFF)
    regime_rng = np.random.default_rng((base_seed ^ seed_magic ^ 0x5A5A5A5A) & 0xFFFFFFFF)

    print(f"Generating {n_datasets} training datasets for task_family={task_family!r}")
    print(f"  -> {args.out_dir}")

    generated = 0
    attempts = 0
    max_attempts = n_datasets * 5  # allow retries for LearnabilityGateError

    while generated < n_datasets and attempts < max_attempts:
        idx = generated
        attempts += 1
        nonlinear_family = str(family_rng.choice(families))
        feature_regime = str(regime_rng.choice(feature_regimes))
        # Legacy shared V3 grid draws (kept for eval-suite compatibility; training overrides below)
        n = int(np.random.default_rng(base_seed ^ seed_magic ^ idx).choice(V3_N_GRID))
        p_signal = int(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 1).choice(V3_P_SIGNAL_GRID))
        teacher_seed = int(np.random.default_rng([base_seed ^ seed_magic, idx, 0]).integers(2**62))
        sample_seed = int(np.random.default_rng([base_seed ^ seed_magic, idx, 1]).integers(2**62))

        try:
            if task_family == "synthetic_nonlinear_classification":
                # --- cursor_scale_v1 dims (Suite 7: NLC numeric) ---
                _dim_rng = np.random.default_rng(
                    int(np.random.SeedSequence(
                        [base_seed ^ seed_magic, idx, 0xC1A5D1C]).generate_state(1)[0])
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
                    # 2.5% E (even idx) / 2.5% F (odd idx) tail allocation
                    nonlinear_family = (
                        "multiclass_smooth_softmax" if idx % 2 == 0
                        else "multiclass_sparse_highdim"
                    )
                else:
                    while True:
                        _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
                        _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                        if _p_total >= 1 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                            break
                n = _n
                p_signal = max(1, int(_dim_rng.integers(1, _p_total + 1)))
                # ---
                num_classes = int(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 2).choice(_CLS_NUM_CLASSES_GRID))
                temperature = float(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 3).choice(_CLS_TEMPERATURE_GRID))
                label_noise_rate = float(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 4).choice(_CLS_LABEL_NOISE_GRID))
                class_imbalance_type = str(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 5).choice(_CLS_IMBALANCE_GRID))
                margin_level = str(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 6).choice(_CLS_MARGIN_GRID))
                ds = generate_nonlinear_classification_dataset(
                    nonlinear_family=nonlinear_family,
                    feature_regime=feature_regime,
                    n=n,
                    p_signal=p_signal,
                    num_classes=num_classes,
                    temperature=temperature,
                    label_noise_rate=label_noise_rate,
                    class_imbalance_type=class_imbalance_type,
                    margin_level=margin_level,
                    suite_component="train",
                    teacher_seed=teacher_seed,
                    sample_seed=sample_seed,
                )
                validate_nonlinear_classification_dataset(ds, is_mixed_categorical=False)

            elif task_family == "synthetic_nonlinear_regression_mixed_categorical":
                # --- cursor_scale_v1 dims (Suite 6: NLR mixed) --- tail EXCLUDED ---
                # p_total = p_signal + p_cat; budget carved from Poisson(10)-only draw.
                _dim_rng = np.random.default_rng(
                    int(np.random.SeedSequence(
                        [base_seed ^ seed_magic, idx, 0x4E4D5233]).generate_state(1)[0])
                )
                while True:
                    _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
                    _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                    if _p_total >= 2 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                        break
                # Carve p_cat from budget (p_cat ≥ 1, p_signal ≥ 1)
                _p_cat = max(1, min(int(_dim_rng.integers(1, 4)), _p_total - 1))
                n = _n
                p_signal = _p_total - _p_cat  # numeric signal columns
                p_cat = _p_cat
                # ---
                noise_scale = float(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 2).choice(V3_NOISE_SCALES))
                noise_type = str(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 3).choice(V2_TARGET_NOISE_TYPES))
                cat_effect_scale = float(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 5).choice(_MIXED_CAT_EFFECT_GRID))
                ds = generate_v3_mixed_regression_dataset(
                    target_family=nonlinear_family,
                    feature_regime=feature_regime,
                    n=n,
                    p_signal=p_signal,
                    target_noise_scale=noise_scale,
                    target_noise_type=noise_type,
                    suite_component="train",
                    teacher_seed=teacher_seed,
                    sample_seed=sample_seed,
                    p_cat=p_cat,
                    cat_effect_scale=cat_effect_scale,
                )
                # Tag training_data_family for the builder
                ds["training_data_family"] = "synthetic_nonlinear_regression_mixed_categorical"

            else:  # synthetic_nonlinear_classification_mixed_categorical
                # --- cursor_scale_v1 dims (Suite 8: NLC mixed) ---
                # Full 5% tail → mixed_categorical_nonlinear family
                _dim_rng = np.random.default_rng(
                    int(np.random.SeedSequence(
                        [base_seed ^ seed_magic, idx, 0x4E4D4331]).generate_state(1)[0])
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
                    nonlinear_family = "mixed_categorical_nonlinear"
                else:
                    while True:
                        _p_total = int(_dim_rng.poisson(_CURSOR_P_LAMBDA))
                        _n = int(_dim_rng.poisson(_CURSOR_N_LAMBDA))
                        if _p_total >= 2 and _n >= _CURSOR_MIN_N_OVER_P * _p_total:
                            break
                # Carve p_cat from budget (p_cat ≥ 1, p_signal ≥ 1)
                _p_cat = max(1, min(int(_dim_rng.integers(1, 4)), _p_total - 1))
                n = _n
                p_signal = _p_total - _p_cat
                p_cat = _p_cat
                # ---
                num_classes = int(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 2).choice(_CLS_NUM_CLASSES_GRID))
                temperature = float(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 3).choice(_CLS_TEMPERATURE_GRID))
                label_noise_rate = float(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 4).choice(_CLS_LABEL_NOISE_GRID))
                class_imbalance_type = str(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 5).choice(_CLS_IMBALANCE_GRID))
                margin_level = str(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 6).choice(_CLS_MARGIN_GRID))
                cat_effect_scale = float(np.random.default_rng(base_seed ^ seed_magic ^ idx ^ 8).choice(_MIXED_CAT_EFFECT_GRID))
                ds = generate_nonlinear_mixed_classification_dataset(
                    nonlinear_family=nonlinear_family,
                    feature_regime=feature_regime,
                    n=n,
                    p_signal=p_signal,
                    num_classes=num_classes,
                    temperature=temperature,
                    label_noise_rate=label_noise_rate,
                    class_imbalance_type=class_imbalance_type,
                    margin_level=margin_level,
                    suite_component="train",
                    teacher_seed=teacher_seed,
                    sample_seed=sample_seed,
                    p_cat=p_cat,
                    cat_effect_scale=cat_effect_scale,
                )
                ds["training_data_family"] = "synthetic_nonlinear_classification_mixed_categorical"
                validate_nonlinear_classification_dataset(ds, is_mixed_categorical=True)

        except LearnabilityGateError:
            print(f"  [{idx:4d}] LearnabilityGateError — skipping {nonlinear_family}/{feature_regime}")
            continue
        except AssertionError as _ae:
            print(f"  [{idx:4d}] AssertionError (class coverage) — skipping {nonlinear_family}/{feature_regime}: {_ae}")
            continue

        if idx < n_train_split:
            split_dir = train_dir
            filename = f"dataset_{idx:04d}.parquet"
        elif idx < n_train_split + n_val_split:
            split_dir = val_dir
            filename = f"dataset_{idx - n_train_split:04d}.parquet"
        else:
            split_dir = test_dir
            filename = f"dataset_{idx - n_train_split - n_val_split:04d}.parquet"

        writer_fn(ds, os.path.join(split_dir, filename))
        generated += 1
        if (generated) % 100 == 0:
            print(f"  [{generated:4d}/{n_datasets}] {filename} -> {split_dir}")

    if generated < n_datasets:
        print(
            f"WARNING: only {generated}/{n_datasets} datasets generated after {attempts} attempts "
            "(LearnabilityGateError on some configs). Increase --n_datasets or review parameter grids."
        )
    else:
        print(f"Done. Generated {generated} datasets -> {args.out_dir}")


# =====================================================================
# v3 EXTENSIONS — classification helpers, generators, writers,
#                  enumerators, validation gates
# =====================================================================


# ---------------------------------------------------------------------------
# Feature-regime → distribution mapping for v3
# ---------------------------------------------------------------------------

_REGIME_TO_DISTRIBUTION: dict[str, str] = {
    "iid_dense":           "iid_gaussian",
    "iid_sparse":          "iid_gaussian",
    "ar1":                 "correlated_ar",
    "block":               "block_correlated",
    "equicorr":            "equicorrelated",
    "noise_feats":         "iid_gaussian",
    "feat_noise":          "iid_gaussian",
    "student_t_features":  "student_t_features",
    "laplace_features":    "laplace_features",
    "uniform_bounded":     "uniform_bounded",
    "skewed_lognormal":    "skewed_lognormal",
    "mixture_gaussian":    "mixture_gaussian",
    "low_rank_factor":     "low_rank_factor",
    "manifold_features":   "manifold_features",
}


def _resolve_family(name: str) -> str:
    """Resolve family aliases to canonical names."""
    return _FAMILY_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Classification helpers (self-contained — no dgp_helpers import)
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with temperature. logits: (n, K) -> probs: (n, K)."""
    scaled = logits / max(temperature, 1e-12)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def _apply_symmetric_label_noise(
    rng: np.random.Generator,
    labels: np.ndarray,
    num_classes: int,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply symmetric label noise: each selected label flipped to uniform random class.

    Returns (noisy_labels, noise_mask).
    """
    n = len(labels)
    noisy = labels.copy()
    n_flip = int(round(rate * n))
    if n_flip == 0:
        return noisy, np.zeros(n, dtype=bool)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[flip_idx] = True
    noisy[flip_idx] = rng.integers(0, num_classes, size=n_flip)
    return noisy, noise_mask


def _sample_categorical_labels(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Sample one label per row from probability matrix (n, K)."""
    n, K = probs.shape
    cumulative = np.cumsum(probs, axis=1)
    u = rng.uniform(size=(n, 1))
    return (u >= cumulative).sum(axis=1).astype(np.int64).clip(0, K - 1)


def _ensure_all_classes_present(
    y_clean: np.ndarray,
    probs: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Guarantee every class 0..K-1 appears in y_clean by reassigning DISTINCT rows.

    For each missing class k we pick the not-yet-reserved row with the highest
    probs[:,k] (most sensible placement) and overwrite its label with k.  Because
    each patched row is marked reserved before moving to the next class, no later
    patch can re-empty a class that was just fixed — avoiding the race in the
    simpler ``y[rng.integers(0, n)] = k`` pattern.

    Raises LearnabilityGateError when n < K (impossible to fill all classes
    with distinct rows), triggering the caller's existing retry loop.
    """
    n = len(y_clean)
    if n < K:
        raise LearnabilityGateError(
            f"n={n} < K={K}: cannot guarantee all classes present with distinct rows"
        )
    y = y_clean.copy()
    reserved: set = set()
    for k in range(K):
        if np.any(y == k):
            continue
        # Find the unreserved row that most strongly predicts class k
        scores = probs[:, k].copy()
        scores[list(reserved)] = -np.inf
        best_row = int(np.argmax(scores))
        y[best_row] = k
        reserved.add(best_row)
    return y


def _calibrate_classification_intercepts(
    logits: np.ndarray,
    target_prior: str,
    num_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Adjust class intercepts (logits[:, 1:]) to achieve target class balance.

    target_prior: "balanced" | "mild" | "moderate" | "severe"
    Returns adjusted logits (reference class 0 stays at 0).
    """
    K = num_classes
    if target_prior == "balanced" or K <= 1:
        return logits
    # Generate target class weights via Dirichlet
    alpha_map = {"mild": 2.0, "moderate": 1.0, "severe": 0.5}
    alpha = alpha_map.get(target_prior, 2.0)
    target_weights = rng.dirichlet(alpha * np.ones(K))
    target_weights = np.sort(target_weights)[::-1]  # descending
    # Convert to log-prior offsets relative to class 0
    log_prior = np.log(target_weights + 1e-12) - np.log(target_weights[0] + 1e-12)
    adjusted = logits.copy()
    for k in range(1, K):
        adjusted[:, k] += log_prior[k]
    return adjusted


def _normalized_margins(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Compute normalized margins: prob_top - prob_second per row."""
    probs = _softmax(logits, temperature)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def _margin_bucket(median_margin: float) -> str:
    """Classify median margin into low/medium/high."""
    if median_margin < 0.2:
        return "low"
    elif median_margin < 0.5:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Classification family → teacher mapping
# ---------------------------------------------------------------------------

_CLS_FAMILY_TEACHER_MAP: dict[str, str] = {
    "nonlinear_binary_logistic":     "smooth_additive",
    "sparse_interaction_logistic":   "sparse_interact",
    "radial_decision_boundary":      "rbf_local",
    "piecewise_margin":              "piecewise_relu",
    "multiclass_smooth_softmax":     "smooth_additive",
    "multiclass_sparse_highdim":     "high_dim_sparse_nonlinear",
    "correlated_nonlinear_softmax":  "smooth_additive",
    "low_margin_overlap":            "sigmoid",
    "label_noise_nonlinear":         "poly",
    "class_imbalance_nonlinear":     "poly",
    "mixed_categorical_nonlinear":   "smooth_additive",
}


# ---------------------------------------------------------------------------
# v3 regression dataset generator
# ---------------------------------------------------------------------------

def generate_v3_regression_dataset(
    target_family: str,
    feature_regime: str,
    n: int,
    p_signal: int,
    target_noise_scale: float,
    target_noise_type: str,
    suite_component: str,
    teacher_seed: int,
    sample_seed: int,
    **kwargs: Any,
) -> dict:
    """Generate one v3 nonlinear regression evaluation dataset.

    Uses mu_clean naming (with betaX as backward-compat alias).
    Supports extended feature distributions and noise types.
    """
    target_family = _resolve_family(target_family)
    teacher_rng = np.random.default_rng(teacher_seed)
    sample_rng = np.random.default_rng(sample_seed)

    stress_kwargs = {k: v for k, v in kwargs.items() if k in ("feature_noise_sigma", "rho")}
    feat_params = _sample_feature_params(
        teacher_rng, feature_regime, n, p_signal, stress_kwargs=stress_kwargs
    )
    covariance_type = feat_params["covariance_type"]
    rho = feat_params["rho"]
    active_fraction = feat_params["active_fraction"]
    feature_noise_sigma = float(feat_params["feature_noise_sigma"])
    p_noise = int(feat_params["p_noise"])
    p_total = p_signal + p_noise
    feature_distribution = feat_params.get("feature_distribution")

    family_params: dict[str, Any] = {}
    if target_family == "mixed_linear":
        family_params["alpha"] = float(kwargs.get("alpha", teacher_rng.uniform(0.1, 0.9)))
    if target_family == "poly":
        family_params["degree"] = int(kwargs.get("degree", int(teacher_rng.choice([2, 3, 4, 5]))))

    # --- Calibration phase ---
    if feature_distribution:
        X_cal = _generate_X_v3(teacher_rng, N_CAL, p_signal, feature_distribution, rho)
    else:
        X_cal = _generate_X(teacher_rng, N_CAL, p_signal, covariance_type, rho)

    if active_fraction < 1.0:
        active_cols = _select_active_columns(teacher_rng, p_signal, active_fraction)
        Xa_cal = X_cal[:, active_cols]
    else:
        active_cols = np.arange(p_signal)
        Xa_cal = X_cal

    cal_mean, cal_std, teacher_param_seed = _calibrate_teacher(
        teacher_rng, target_family, Xa_cal, family_params
    )

    # --- Evaluation phase ---
    if feature_distribution:
        X_signal_clean = _generate_X_v3(sample_rng, n, p_signal, feature_distribution, rho)
    else:
        X_signal_clean = _generate_X(sample_rng, n, p_signal, covariance_type, rho)

    Xa_eval = X_signal_clean[:, active_cols] if active_fraction < 1.0 else X_signal_clean

    teacher_fn = _TEACHER_REGISTRY[target_family]
    teacher_sub_rng = np.random.default_rng(teacher_param_seed)
    mu_clean_raw = teacher_fn(teacher_sub_rng, Xa_eval, family_params)
    mu_clean_raw = np.asarray(mu_clean_raw, dtype=np.float64)
    mu_clean = (mu_clean_raw - cal_mean) / cal_std

    noise = _generate_noise(
        sample_rng, n, target_noise_type, target_noise_scale,
        betaX_calibrated=mu_clean if target_noise_type in ("heteroscedastic", "gross_outlier") else None,
    )
    y = mu_clean + noise

    if p_noise > 0:
        X_noise_cols = _generate_X(sample_rng, n, p_noise, "iid", 0.0)
        X = np.concatenate([X_signal_clean, X_noise_cols], axis=1)
    else:
        X = X_signal_clean

    if feature_noise_sigma > 0.0:
        X = X + feature_noise_sigma * sample_rng.standard_normal(X.shape)

    signal_var = float(np.var(mu_clean))
    noise_var = float(np.var(noise))
    snr_target = signal_var / (noise_var + 1e-30) if noise_var > 1e-30 else float("inf")

    # Parametric model class mapping
    _PARAMETRIC_CLASS = {
        "poly": "polynomial", "smooth_additive": "additive",
        "sparse_interact": "interaction", "rbf_local": "rbf",
        "piecewise_relu": "piecewise", "discontinuous_threshold": "threshold",
        "exp_growth": "exponential", "power": "power",
        "sigmoid": "sigmoid", "saturation": "saturation",
        "gompertz": "gompertz", "logarithmic": "logarithmic",
        "periodic": "periodic", "heteroskedastic": "heteroskedastic",
        "high_dim_sparse_nonlinear": "sparse_nonlinear",
        "low_rank_compositional": "compositional", "mixed_linear": "mixed",
    }

    condition_id = (
        f"{target_family}_{feature_regime}_n{n}_p{p_signal}"
        f"_noise{target_noise_type}_scale{target_noise_scale}"
    )

    return {
        # Core arrays
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "mu_clean": mu_clean.astype(np.float64),
        "betaX": mu_clean.astype(np.float64),  # backward-compat alias
        # v1-compat
        "suite_family": "primary",
        "prior_regime": target_family,
        "n_total": int(n),
        "p_signal": int(p_signal),
        "p_noise": int(p_noise),
        "p_total": int(p_total),
        "target_noise_scale": float(target_noise_scale),
        "training_size_anchor": False,
        "feature_noise_level": float(feature_noise_sigma),
        # v3 metadata
        "feature_regime": feature_regime,
        "covariance_type": covariance_type,
        "rho": float(rho),
        "active_fraction": float(active_fraction),
        "noise_feature_fraction": float(feat_params.get("noise_feature_fraction", 0.0)),
        "feature_noise_sigma": float(feature_noise_sigma),
        "suite_component": suite_component,
        "target_noise_type": target_noise_type,
        "snr_target": float(snr_target),
        "condition_id": condition_id,
        "teacher_seed": int(teacher_seed),
        "sample_seed": int(sample_seed),
        "normalization_constant": float(cal_std),
        "nonlinear_family": target_family,
        "parametric_model_class": _PARAMETRIC_CLASS.get(target_family, "unknown"),
        "task_family": "synthetic_nonlinear_regression",
        "task_objective": "inductive_regression",
        "schema_version": "nonlinear_regression_v3",
        "teacher_available": True,
        "polynomial_degree": family_params.get("degree"),
    }


# ---------------------------------------------------------------------------
# v3 mixed-categorical regression dataset generator
# ---------------------------------------------------------------------------

def generate_v3_mixed_regression_dataset(
    target_family: str,
    feature_regime: str,
    n: int,
    p_signal: int,
    target_noise_scale: float,
    target_noise_type: str,
    suite_component: str,
    teacher_seed: int,
    sample_seed: int,
    *,
    p_cat: int = 3,
    cardinalities: list[int] | None = None,
    cat_effect_scale: float = 1.0,
    cat_missing_rate: float = 0.05,
    cat_imbalance_type: str = "balanced",
    **kwargs: Any,
) -> dict:
    """Generate one v3 nonlinear regression dataset with mixed-categorical features.

    Uses helpers from dgp_helpers.py for categorical generation.
    """
    from dgp_helpers import (
        generate_categorical_features,
        generate_categorical_effects_regression,
        apply_category_missing,
        mark_unseen_query_categories,
    )
    from constants import ENTITY_EMBED_FIRST_REAL_ID

    # Generate the numeric part first
    ds = generate_v3_regression_dataset(
        target_family=target_family,
        feature_regime=feature_regime,
        n=n,
        p_signal=p_signal,
        target_noise_scale=target_noise_scale,
        target_noise_type=target_noise_type,
        suite_component=suite_component,
        teacher_seed=teacher_seed,
        sample_seed=sample_seed,
        **kwargs,
    )

    cat_rng = np.random.default_rng(sample_seed + 7777)
    if cardinalities is None:
        cardinalities = [int(cat_rng.integers(3, 20)) for _ in range(p_cat)]

    X_cat = generate_categorical_features(
        cat_rng, n, cardinalities, imbalance_type=cat_imbalance_type
    )
    # Active mask: first half of cat features are signal, rest noise
    n_active_cat = max(1, p_cat // 2)
    cat_active_mask = [i < n_active_cat for i in range(p_cat)]
    cat_effects = generate_categorical_effects_regression(
        cat_rng, cardinalities, cat_effect_scale, cat_active_mask
    )

    # Add categorical effects to mu_clean
    mu_numeric = ds["mu_clean"].copy()
    cat_contribution = np.zeros(n)
    for j in range(p_cat):
        if cat_active_mask[j]:
            local_ids = X_cat[:, j] - ENTITY_EMBED_FIRST_REAL_ID
            cat_contribution += cat_effects[j][local_ids]

    mu_clean_mixed = mu_numeric + cat_contribution
    noise = ds["y"] - ds["mu_clean"]
    y_mixed = mu_clean_mixed + noise

    # Apply missing
    X_cat_masked, cat_missing_mask = apply_category_missing(cat_rng, X_cat, cat_missing_rate)

    ds["mu_clean"] = mu_clean_mixed.astype(np.float64)
    ds["betaX"] = mu_clean_mixed.astype(np.float64)
    ds["y"] = y_mixed.astype(np.float64)
    ds["X_cat"] = X_cat_masked.astype(np.int64)
    ds["cat_missing_mask"] = cat_missing_mask
    ds["categorical_cardinalities"] = cardinalities
    ds["p_cat"] = int(p_cat)
    ds["cat_effects"] = cat_effects
    ds["cat_support_mask"] = cat_active_mask
    ds["task_family"] = "synthetic_nonlinear_regression_mixed_categorical"
    ds["schema_version"] = "nonlinear_regression_mixed_categorical_v1"
    return ds


# ---------------------------------------------------------------------------
# Classification dataset generator
# ---------------------------------------------------------------------------

def generate_nonlinear_classification_dataset(
    nonlinear_family: str,
    feature_regime: str,
    n: int,
    p_signal: int,
    num_classes: int,
    temperature: float,
    label_noise_rate: float,
    class_imbalance_type: str,
    margin_level: str,
    suite_component: str,
    teacher_seed: int,
    sample_seed: int,
    **kwargs: Any,
) -> dict:
    """Generate one nonlinear classification evaluation dataset.

    Each class k=1..K-1 gets logits from a teacher function with class-specific params.
    Class 0 is the reference (logit_0 = 0).
    """
    teacher_rng = np.random.default_rng(teacher_seed)
    sample_rng = np.random.default_rng(sample_seed)
    K = int(num_classes)

    stress_kwargs = {k: v for k, v in kwargs.items() if k in ("feature_noise_sigma", "rho")}
    feat_params = _sample_feature_params(
        teacher_rng, feature_regime, n, p_signal, stress_kwargs=stress_kwargs
    )
    covariance_type = feat_params["covariance_type"]
    rho = feat_params["rho"]
    active_fraction = feat_params["active_fraction"]
    feature_noise_sigma = float(feat_params["feature_noise_sigma"])
    p_noise = int(feat_params["p_noise"])
    p_total = p_signal + p_noise
    feature_distribution = feat_params.get("feature_distribution")

    # Generate features
    if feature_distribution:
        X_signal = _generate_X_v3(sample_rng, n, p_signal, feature_distribution, rho)
    else:
        X_signal = _generate_X(sample_rng, n, p_signal, covariance_type, rho)

    if active_fraction < 1.0:
        active_cols = _select_active_columns(teacher_rng, p_signal, active_fraction)
        Xa = X_signal[:, active_cols]
    else:
        Xa = X_signal

    # Resolve which regression teacher to use for logit generation
    reg_teacher_name = _CLS_FAMILY_TEACHER_MAP.get(nonlinear_family, "smooth_additive")
    teacher_fn = _TEACHER_REGISTRY[reg_teacher_name]

    # Generate logits: class 0 = 0 (reference)
    logits = np.zeros((n, K), dtype=np.float64)
    for k in range(1, K):
        class_seed = int(np.random.default_rng([teacher_seed, k]).integers(2**62))
        class_rng = np.random.default_rng(class_seed)
        class_params: dict[str, Any] = {}
        raw_logit = teacher_fn(class_rng, Xa, class_params)
        raw_logit = np.asarray(raw_logit, dtype=np.float64)
        std = float(np.std(raw_logit))
        if std > 1e-12:
            raw_logit = raw_logit / std
        logits[:, k] = raw_logit

    # Scale logits by margin level
    margin_scale = {"low": 0.3, "medium": 1.0, "high": 3.0}.get(margin_level, 1.0)
    logits[:, 1:] *= margin_scale

    # Calibrate intercepts for class imbalance
    logits = _calibrate_classification_intercepts(logits, class_imbalance_type, K, teacher_rng)

    # Compute probabilities
    probs = _softmax(logits, temperature)

    # Sample clean labels
    y_clean = _sample_categorical_labels(sample_rng, probs)

    # Guarantee all K classes present before applying noise (distinct-row patch)
    y_clean = _ensure_all_classes_present(y_clean, probs, K, sample_rng)

    # Apply label noise
    y, label_noise_mask = _apply_symmetric_label_noise(sample_rng, y_clean, K, label_noise_rate)

    # Add noise features
    if p_noise > 0:
        X_noise_cols = _generate_X(sample_rng, n, p_noise, "iid", 0.0)
        X = np.concatenate([X_signal, X_noise_cols], axis=1)
    else:
        X = X_signal

    if feature_noise_sigma > 0.0:
        X = X + feature_noise_sigma * sample_rng.standard_normal(X.shape)

    # Compute realized stats
    margins = _normalized_margins(logits, temperature)
    realized_margin = _margin_bucket(float(np.median(margins)))
    class_counts = np.bincount(y_clean, minlength=K)
    class_prior = (class_counts / max(n, 1)).tolist()
    realized_label_noise = float(label_noise_mask.sum()) / max(n, 1)

    condition_id = (
        f"{nonlinear_family}_{feature_regime}_n{n}_p{p_signal}"
        f"_K{K}_T{temperature}_ln{label_noise_rate}"
    )

    return {
        "X": X.astype(np.float64),
        "y": y.astype(np.int64),
        "y_clean": y_clean.astype(np.int64),
        "label_noise_mask": label_noise_mask,
        "logits": logits.astype(np.float64),
        "probs": probs.astype(np.float64),
        "n_total": int(n),
        "p_signal": int(p_signal),
        "p_noise": int(p_noise),
        "p_total": int(p_total),
        "num_classes": int(K),
        "realized_num_classes": int(len(set(y_clean.tolist()))),
        "temperature": float(temperature),
        "label_noise_rate": float(label_noise_rate),
        "realized_label_noise_rate": float(realized_label_noise),
        "class_imbalance_type": class_imbalance_type,
        "margin_level": margin_level,
        "realized_margin_level": realized_margin,
        "class_prior": class_prior,
        "nonlinear_family": nonlinear_family,
        "feature_regime": feature_regime,
        "covariance_type": covariance_type,
        "rho": float(rho),
        "active_fraction": float(active_fraction),
        "feature_noise_sigma": float(feature_noise_sigma),
        "suite_component": suite_component,
        "condition_id": condition_id,
        "teacher_seed": int(teacher_seed),
        "sample_seed": int(sample_seed),
        "task_family": "synthetic_nonlinear_classification",
        "task_objective": "inductive_classification",
        "schema_version": "nonlinear_classification_v1",
        "teacher_available": True,
        "prior_regime": nonlinear_family,
    }


# ---------------------------------------------------------------------------
# Mixed-categorical classification dataset generator
# ---------------------------------------------------------------------------

def generate_nonlinear_mixed_classification_dataset(
    nonlinear_family: str,
    feature_regime: str,
    n: int,
    p_signal: int,
    num_classes: int,
    temperature: float,
    label_noise_rate: float,
    class_imbalance_type: str,
    margin_level: str,
    suite_component: str,
    teacher_seed: int,
    sample_seed: int,
    *,
    p_cat: int = 3,
    cardinalities: list[int] | None = None,
    cat_effect_scale: float = 1.0,
    cat_missing_rate: float = 0.05,
    cat_imbalance_type: str = "balanced",
    **kwargs: Any,
) -> dict:
    """Generate one nonlinear classification dataset with mixed-categorical features."""
    from dgp_helpers import (
        generate_categorical_features,
        generate_categorical_class_effects,
        apply_category_missing,
        mark_unseen_query_categories,
    )
    from constants import ENTITY_EMBED_FIRST_REAL_ID

    K = int(num_classes)
    cat_rng = np.random.default_rng(sample_seed + 8888)
    if cardinalities is None:
        cardinalities = [int(cat_rng.integers(3, 20)) for _ in range(p_cat)]

    # Generate base classification dataset (numeric only)
    ds = generate_nonlinear_classification_dataset(
        nonlinear_family=nonlinear_family,
        feature_regime=feature_regime,
        n=n,
        p_signal=p_signal,
        num_classes=K,
        temperature=temperature,
        label_noise_rate=label_noise_rate,
        class_imbalance_type=class_imbalance_type,
        margin_level=margin_level,
        suite_component=suite_component,
        teacher_seed=teacher_seed,
        sample_seed=sample_seed,
        **kwargs,
    )

    X_cat = generate_categorical_features(
        cat_rng, n, cardinalities, imbalance_type=cat_imbalance_type
    )
    n_active_cat = max(1, p_cat // 2)
    cat_active_mask = [i < n_active_cat for i in range(p_cat)]
    cat_class_effects = generate_categorical_class_effects(
        cat_rng, cardinalities, K, cat_effect_scale, cat_active_mask
    )

    # Add categorical effects to logits
    logits = ds["logits"].copy()
    for j in range(p_cat):
        if cat_active_mask[j]:
            local_ids = X_cat[:, j] - ENTITY_EMBED_FIRST_REAL_ID
            logits[:, :] += cat_class_effects[j][local_ids, :]

    # Recompute reference class constraint
    logits[:, 0] = 0.0

    probs = _softmax(logits, temperature)
    sample_rng = np.random.default_rng(sample_seed + 9999)
    y_clean = _sample_categorical_labels(sample_rng, probs)

    # Guarantee all K classes present after categorical effects shift probs (distinct-row patch)
    y_clean = _ensure_all_classes_present(y_clean, probs, K, sample_rng)

    y, label_noise_mask = _apply_symmetric_label_noise(
        sample_rng, y_clean, K, label_noise_rate
    )

    X_cat_masked, cat_missing_mask = apply_category_missing(cat_rng, X_cat, cat_missing_rate)

    ds["logits"] = logits.astype(np.float64)
    ds["probs"] = probs.astype(np.float64)
    ds["y"] = y.astype(np.int64)
    ds["y_clean"] = y_clean.astype(np.int64)
    ds["label_noise_mask"] = label_noise_mask
    ds["X_cat"] = X_cat_masked.astype(np.int64)
    ds["cat_missing_mask"] = cat_missing_mask
    ds["categorical_cardinalities"] = cardinalities
    ds["p_cat"] = int(p_cat)
    ds["cat_class_effects"] = cat_class_effects
    ds["cat_support_mask"] = cat_active_mask
    ds["task_family"] = "synthetic_nonlinear_classification_mixed_categorical"
    ds["schema_version"] = "nonlinear_classification_mixed_categorical_v1"

    # Update realized stats
    class_counts = np.bincount(y_clean, minlength=K)
    ds["class_prior"] = (class_counts / max(n, 1)).tolist()
    ds["realized_num_classes"] = int(len(set(y_clean.tolist())))
    ds["realized_label_noise_rate"] = float(label_noise_mask.sum()) / max(n, 1)
    margins = _normalized_margins(logits, temperature)
    ds["realized_margin_level"] = _margin_bucket(float(np.median(margins)))
    return ds


# ---------------------------------------------------------------------------
# v3 Parquet writers
# ---------------------------------------------------------------------------

def write_parquet_nonlinear_regression_eval(
    ds: dict,
    filepath: str,
    *,
    is_mixed_categorical: bool = False,
) -> int:
    """Write v3 nonlinear regression Parquet with 80/20 train/test split.

    Returns 1 (one meta-row per dataset).
    """
    X = ds["X"]
    y = ds["y"]
    mu_clean = ds["mu_clean"]
    n = len(y)
    n_train = int(0.8 * n)
    if n - n_train < 1:
        n_train = n - 1
    n_test = n - n_train

    cols: dict[str, Any] = {
        "X_train":          pa.array([X[:n_train].tolist()],          type=pa.list_(pa.list_(pa.float64()))),
        "X_test":           pa.array([X[n_train:].tolist()],          type=pa.list_(pa.list_(pa.float64()))),
        "y_train":          pa.array([y[:n_train].tolist()],          type=pa.list_(pa.float64())),
        "y_test":           pa.array([y[n_train:].tolist()],          type=pa.list_(pa.float64())),
        "mu_clean_train":   pa.array([mu_clean[:n_train].tolist()],   type=pa.list_(pa.float64())),
        "mu_clean_test":    pa.array([mu_clean[n_train:].tolist()],   type=pa.list_(pa.float64())),
        "schema_version":   pa.array([ds["schema_version"]],         type=pa.utf8()),
        "task_family":      pa.array([ds["task_family"]],            type=pa.utf8()),
        "task_objective":   pa.array([ds.get("task_objective", "inductive_regression")], type=pa.utf8()),
        "profile":          pa.array(["v3"],                          type=pa.utf8()),
        "prior_regime":     pa.array([ds["prior_regime"]],           type=pa.utf8()),
        "nonlinear_family": pa.array([ds["nonlinear_family"]],      type=pa.utf8()),
        "n_train":          pa.array([n_train],                      type=pa.int64()),
        "n_test":           pa.array([n_test],                       type=pa.int64()),
        "p_signal":         pa.array([int(ds["p_signal"])],          type=pa.int64()),
        "p_noise":          pa.array([int(ds["p_noise"])],           type=pa.int64()),
        "p_total":          pa.array([int(ds["p_total"])],           type=pa.int64()),
        "covariance_type":  pa.array([ds["covariance_type"]],       type=pa.utf8()),
        "rho":              pa.array([float(ds["rho"])],             type=pa.float64()),
        "feature_noise_level": pa.array([float(ds["feature_noise_sigma"])], type=pa.float64()),
        "target_noise_scale":  pa.array([float(ds["target_noise_scale"])],  type=pa.float64()),
        "target_noise_type":   pa.array([ds["target_noise_type"]],         type=pa.utf8()),
        "feature_regime":      pa.array([ds["feature_regime"]],            type=pa.utf8()),
        "suite_component":     pa.array([ds["suite_component"]],           type=pa.utf8()),
        "snr_target":          pa.array([float(ds["snr_target"])],         type=pa.float64()),
        "condition_id":        pa.array([ds["condition_id"]],              type=pa.utf8()),
        "teacher_seed":        pa.array([int(ds["teacher_seed"])],         type=pa.int64()),
        "sample_seed":         pa.array([int(ds["sample_seed"])],          type=pa.int64()),
        "normalization_constant": pa.array([float(ds["normalization_constant"])], type=pa.float64()),
        "parametric_model_class": pa.array([ds.get("parametric_model_class", "unknown")], type=pa.utf8()),
        "teacher_available":   pa.array([bool(ds.get("teacher_available", True))], type=pa.bool_()),
    }

    # Optional polynomial degree
    degree = ds.get("polynomial_degree")
    if degree is not None:
        cols["polynomial_degree"] = pa.array([int(degree)], type=pa.int64())

    if is_mixed_categorical:
        X_cat = ds["X_cat"]
        cat_mask = ds["cat_missing_mask"]
        cols["X_cat_train"] = pa.array([X_cat[:n_train].tolist()], type=pa.list_(pa.list_(pa.int64())))
        cols["X_cat_test"] = pa.array([X_cat[n_train:].tolist()], type=pa.list_(pa.list_(pa.int64())))
        cols["cat_missing_mask_train"] = pa.array([cat_mask[:n_train].tolist()], type=pa.list_(pa.list_(pa.bool_())))
        cols["cat_missing_mask_test"] = pa.array([cat_mask[n_train:].tolist()], type=pa.list_(pa.list_(pa.bool_())))
        cols["categorical_cardinalities"] = pa.array([ds["categorical_cardinalities"]], type=pa.list_(pa.int64()))
        cols["p_cat"] = pa.array([int(ds["p_cat"])], type=pa.int64())

    table = pa.table(cols)
    pq.write_table(table, filepath)
    return 1


def write_parquet_nonlinear_classification_eval(
    ds: dict,
    filepath: str,
    *,
    is_mixed_categorical: bool = False,
) -> int:
    """Write nonlinear classification Parquet with 80/20 train/test split.

    Returns 1 (one meta-row per dataset).
    """
    X = ds["X"]
    y = ds["y"]
    y_clean = ds["y_clean"]
    logits = ds["logits"]
    probs = ds["probs"]
    label_noise_mask = ds["label_noise_mask"]
    n = len(y)
    n_train = int(0.8 * n)
    if n - n_train < 1:
        n_train = n - 1
    n_test = n - n_train

    cols: dict[str, Any] = {
        "X_train":              pa.array([X[:n_train].tolist()],              type=pa.list_(pa.list_(pa.float64()))),
        "X_test":               pa.array([X[n_train:].tolist()],              type=pa.list_(pa.list_(pa.float64()))),
        "y_train":              pa.array([y[:n_train].tolist()],              type=pa.list_(pa.int64())),
        "y_test":               pa.array([y[n_train:].tolist()],              type=pa.list_(pa.int64())),
        "y_clean_train":        pa.array([y_clean[:n_train].tolist()],        type=pa.list_(pa.int64())),
        "y_clean_test":         pa.array([y_clean[n_train:].tolist()],        type=pa.list_(pa.int64())),
        "label_noise_mask_train": pa.array([label_noise_mask[:n_train].tolist()], type=pa.list_(pa.bool_())),
        "label_noise_mask_test":  pa.array([label_noise_mask[n_train:].tolist()], type=pa.list_(pa.bool_())),
        "logits_train":         pa.array([logits[:n_train].tolist()],         type=pa.list_(pa.list_(pa.float64()))),
        "logits_test":          pa.array([logits[n_train:].tolist()],         type=pa.list_(pa.list_(pa.float64()))),
        "probs_train":          pa.array([probs[:n_train].tolist()],          type=pa.list_(pa.list_(pa.float64()))),
        "dgp_teacher_probs_test": pa.array([probs[n_train:].tolist()],       type=pa.list_(pa.list_(pa.float64()))),
        "schema_version":       pa.array([ds["schema_version"]],             type=pa.utf8()),
        "task_family":          pa.array([ds["task_family"]],                type=pa.utf8()),
        "task_objective":       pa.array([ds.get("task_objective", "inductive_classification")], type=pa.utf8()),
        "profile":              pa.array(["v3"],                              type=pa.utf8()),
        "prior_regime":         pa.array([ds["prior_regime"]],               type=pa.utf8()),
        "nonlinear_family":     pa.array([ds["nonlinear_family"]],          type=pa.utf8()),
        "n_train":              pa.array([n_train],                          type=pa.int64()),
        "n_test":               pa.array([n_test],                           type=pa.int64()),
        "p_signal":             pa.array([int(ds["p_signal"])],              type=pa.int64()),
        "p_noise":              pa.array([int(ds["p_noise"])],               type=pa.int64()),
        "p_total":              pa.array([int(ds["p_total"])],               type=pa.int64()),
        "num_classes":          pa.array([int(ds["num_classes"])],            type=pa.int64()),
        "realized_num_classes": pa.array([int(ds["realized_num_classes"])],   type=pa.int64()),
        "temperature":          pa.array([float(ds["temperature"])],         type=pa.float64()),
        "label_noise_rate":     pa.array([float(ds["label_noise_rate"])],    type=pa.float64()),
        "realized_label_noise_rate": pa.array([float(ds["realized_label_noise_rate"])], type=pa.float64()),
        "class_imbalance_type": pa.array([ds["class_imbalance_type"]],      type=pa.utf8()),
        "margin_level":         pa.array([ds["margin_level"]],              type=pa.utf8()),
        "realized_margin_level": pa.array([ds["realized_margin_level"]],    type=pa.utf8()),
        "class_prior":          pa.array([ds["class_prior"]],               type=pa.list_(pa.float64())),
        "feature_regime":       pa.array([ds["feature_regime"]],            type=pa.utf8()),
        "covariance_type":      pa.array([ds["covariance_type"]],           type=pa.utf8()),
        "rho":                  pa.array([float(ds["rho"])],                type=pa.float64()),
        "feature_noise_level":  pa.array([float(ds["feature_noise_sigma"])], type=pa.float64()),
        "suite_component":      pa.array([ds["suite_component"]],           type=pa.utf8()),
        "condition_id":         pa.array([ds["condition_id"]],              type=pa.utf8()),
        "teacher_seed":         pa.array([int(ds["teacher_seed"])],         type=pa.int64()),
        "sample_seed":          pa.array([int(ds["sample_seed"])],          type=pa.int64()),
        "teacher_available":    pa.array([bool(ds.get("teacher_available", True))], type=pa.bool_()),
    }

    if is_mixed_categorical:
        X_cat = ds["X_cat"]
        cat_mask = ds["cat_missing_mask"]
        cols["X_cat_train"] = pa.array([X_cat[:n_train].tolist()], type=pa.list_(pa.list_(pa.int64())))
        cols["X_cat_test"] = pa.array([X_cat[n_train:].tolist()], type=pa.list_(pa.list_(pa.int64())))
        cols["cat_missing_mask_train"] = pa.array([cat_mask[:n_train].tolist()], type=pa.list_(pa.list_(pa.bool_())))
        cols["cat_missing_mask_test"] = pa.array([cat_mask[n_train:].tolist()], type=pa.list_(pa.list_(pa.bool_())))
        cols["categorical_cardinalities"] = pa.array([ds["categorical_cardinalities"]], type=pa.list_(pa.int64()))
        cols["p_cat"] = pa.array([int(ds["p_cat"])], type=pa.int64())

    table = pa.table(cols)
    pq.write_table(table, filepath)
    return 1


# ---------------------------------------------------------------------------
# Suite cell enumerators — v3
# ---------------------------------------------------------------------------

def enumerate_regression_suite_cells_v3(
    base_seed: int = BASE_SEED,
    *,
    is_mixed_categorical: bool = False,
) -> list[dict]:
    """Enumerate v3 regression suite cells.

    When is_mixed_categorical=False: 17 families × 7 regimes → ~630 datasets.
    When is_mixed_categorical=True: subset of families → ~100 datasets.
    """
    magic = _NONLINEAR_MIXED_REG_SEED_MAGIC if is_mixed_categorical else _NONLINEAR_REG_SEED_MAGIC
    families = V3_TARGET_FAMILIES
    regimes = V2_FEATURE_REGIMES  # 7 standard regimes for suite composition

    if is_mixed_categorical:
        # Smaller subset for mixed-categorical
        regimes = ["iid_dense", "ar1", "equicorr"]

    cells: list[dict] = []

    # -----------------------------------------------------------------------
    # Component A — Core
    # -----------------------------------------------------------------------
    n_cycle = V3_N_GRID
    p_cycle = V3_P_SIGNAL_GRID
    ns_cycle = V3_NOISE_SCALES
    nt_cycle = ["gaussian", "student_t", "heteroscedastic", "contaminated"]

    total_family_regime = len(families) * len(regimes)
    if is_mixed_categorical:
        target_core = 80
    else:
        target_core = 504
    per_cell = max(1, target_core // max(total_family_regime, 1))
    remainder = target_core % max(total_family_regime, 1)

    cell_idx_global = 0
    for f_idx, family in enumerate(families):
        for r_idx, regime in enumerate(regimes):
            n_this = per_cell + (1 if cell_idx_global < remainder else 0)
            cell_idx_global += 1
            for ds_in_cell in range(n_this):
                n_val = n_cycle[ds_in_cell % len(n_cycle)]
                p_signal = p_cycle[ds_in_cell % len(p_cycle)]
                noise_scale = ns_cycle[ds_in_cell % len(ns_cycle)]
                noise_type = nt_cycle[ds_in_cell % len(nt_cycle)]

                teacher_seed = int(np.random.default_rng([magic, f_idx, r_idx, ds_in_cell, 0]).integers(2**62))
                sample_seed = int(np.random.default_rng([magic, f_idx, r_idx, ds_in_cell, 1]).integers(2**62))
                idx = len(cells)

                cell: dict = {
                    "target_family": family,
                    "feature_regime": regime,
                    "n": n_val,
                    "p_signal": p_signal,
                    "target_noise_scale": noise_scale,
                    "target_noise_type": noise_type,
                    "suite_component": "core",
                    "teacher_seed": teacher_seed,
                    "sample_seed": sample_seed,
                    "logical_dataset_key": f"nonlinear_v3:{family}:{idx:05d}",
                    "condition_id": f"{family}_{regime}_n{n_val}_p{p_signal}_noise{noise_type}_scale{noise_scale}",
                    "is_mixed_categorical": is_mixed_categorical,
                }
                if regime in ("ar1", "block", "equicorr"):
                    rho_rng = np.random.default_rng([magic, f_idx, r_idx, ds_in_cell, 2])
                    cell["rho"] = float(rho_rng.choice(_RHO_GRID_CORR))
                cells.append(cell)

    # -----------------------------------------------------------------------
    # Component B — Robustness
    # -----------------------------------------------------------------------
    if is_mixed_categorical:
        target_robust = 15
    else:
        target_robust = 84

    _STRESS_B_V3 = [
        ("contaminated_heavy", {"target_noise_type": "contaminated", "feature_regime": "ar1"}),
        ("tiny_n",             {"n": 32, "p_signal": 32, "feature_regime": "iid_dense"}),
        ("large_n",            {"n": 1024, "feature_regime": "iid_dense"}),
        ("heavy_feat_noise",   {"feature_regime": "feat_noise", "feature_noise_sigma": 0.5}),
        ("cauchy_noise",       {"target_noise_type": "cauchy", "feature_regime": "iid_dense"}),
        ("gross_outlier",      {"target_noise_type": "gross_outlier", "feature_regime": "iid_dense"}),
        ("laplace_noise",      {"target_noise_type": "laplace", "feature_regime": "iid_dense"}),
    ]

    for i in range(target_robust):
        f_idx = i % len(families)
        s_idx = (i // len(families)) % len(_STRESS_B_V3)
        family = families[f_idx]
        stress_label, stress_kw = _STRESS_B_V3[s_idx]

        n_val = int(stress_kw.get("n", 256))
        p_signal = int(stress_kw.get("p_signal", 16))
        noise_scale = 1.0
        noise_type = str(stress_kw.get("target_noise_type", "gaussian"))
        regime = str(stress_kw.get("feature_regime", "iid_dense"))
        extra_kw = {k: v for k, v in stress_kw.items()
                    if k not in ("n", "p_signal", "target_noise_type", "feature_regime")}

        teacher_seed = int(np.random.default_rng([magic + 1000, f_idx, s_idx, i, 0]).integers(2**62))
        sample_seed = int(np.random.default_rng([magic + 1000, f_idx, s_idx, i, 1]).integers(2**62))
        idx = len(cells)

        b_cell: dict = {
            "target_family": family,
            "feature_regime": regime,
            "n": n_val,
            "p_signal": p_signal,
            "target_noise_scale": noise_scale,
            "target_noise_type": noise_type,
            "suite_component": "robustness",
            "teacher_seed": teacher_seed,
            "sample_seed": sample_seed,
            "logical_dataset_key": f"nonlinear_v3:{family}:{idx:05d}",
            "condition_id": f"{family}_{regime}_n{n_val}_p{p_signal}_{stress_label}_{i}",
            "is_mixed_categorical": is_mixed_categorical,
            **extra_kw,
        }
        if regime in ("ar1", "block", "equicorr"):
            rho_rng = np.random.default_rng([magic + 1000, f_idx, s_idx, i, 2])
            b_cell["rho"] = float(rho_rng.choice(_RHO_GRID_CORR))
        cells.append(b_cell)

    # -----------------------------------------------------------------------
    # Component C — OOD
    # -----------------------------------------------------------------------
    if is_mixed_categorical:
        target_ood = 5
    else:
        target_ood = 42

    _ood_families = ["discontinuous_threshold", "power", "gompertz",
                     "low_rank_compositional", "heteroskedastic", "periodic"]
    for i in range(target_ood):
        f_idx = i % len(_ood_families)
        family = _ood_families[f_idx]
        teacher_seed = int(np.random.default_rng([magic + 2000, f_idx, i, 0]).integers(2**62))
        sample_seed = int(np.random.default_rng([magic + 2000, f_idx, i, 1]).integers(2**62))
        idx = len(cells)
        c_cell: dict = {
            "target_family": family,
            "feature_regime": "iid_dense",
            "n": 256,
            "p_signal": 16,
            "target_noise_scale": 1.0,
            "target_noise_type": "gaussian",
            "suite_component": "ood",
            "teacher_seed": teacher_seed,
            "sample_seed": sample_seed,
            "logical_dataset_key": f"nonlinear_v3:{family}:{idx:05d}",
            "condition_id": f"{family}_iid_dense_n256_p16_ood_{i}",
            "is_mixed_categorical": is_mixed_categorical,
        }
        cells.append(c_cell)

    return cells


def enumerate_classification_suite_cells(
    base_seed: int = BASE_SEED,
    *,
    is_mixed_categorical: bool = False,
) -> list[dict]:
    """Enumerate nonlinear classification suite cells.

    When is_mixed_categorical=False: ~400 datasets.
    When is_mixed_categorical=True: ~100 datasets.
    """
    magic = _NONLINEAR_MIXED_CLS_SEED_MAGIC if is_mixed_categorical else _NONLINEAR_CLS_SEED_MAGIC
    families = V3_CLASSIFICATION_FAMILIES
    regimes = ["iid_dense", "ar1", "equicorr"] if is_mixed_categorical else V2_FEATURE_REGIMES

    cells: list[dict] = []

    # -----------------------------------------------------------------------
    # Component A — Core
    # -----------------------------------------------------------------------
    num_classes_cycle = CLS_NUM_CLASSES_GRID
    temp_cycle = CLS_TEMPERATURE_GRID
    ln_cycle = CLS_LABEL_NOISE_GRID
    imb_cycle = CLS_IMBALANCE_GRID
    margin_cycle = CLS_MARGIN_GRID
    n_cycle = [64, 128, 256, 512]
    p_cycle = [4, 8, 16, 32]

    if is_mixed_categorical:
        target_core = 70
    else:
        target_core = 300

    total_fr = len(families) * len(regimes)
    per_cell = max(1, target_core // max(total_fr, 1))
    remainder = target_core % max(total_fr, 1)

    cell_idx_global = 0
    for f_idx, family in enumerate(families):
        for r_idx, regime in enumerate(regimes):
            n_this = per_cell + (1 if cell_idx_global < remainder else 0)
            cell_idx_global += 1
            for ds_in_cell in range(n_this):
                n_val = n_cycle[ds_in_cell % len(n_cycle)]
                p_signal = p_cycle[ds_in_cell % len(p_cycle)]
                K = num_classes_cycle[ds_in_cell % len(num_classes_cycle)]
                temp = temp_cycle[ds_in_cell % len(temp_cycle)]
                ln_rate = ln_cycle[ds_in_cell % len(ln_cycle)]
                imbalance = imb_cycle[ds_in_cell % len(imb_cycle)]
                margin = margin_cycle[ds_in_cell % len(margin_cycle)]

                # Override defaults for family-specific families
                if family == "nonlinear_binary_logistic":
                    K = 2
                elif family == "label_noise_nonlinear" and ln_rate == 0.0:
                    ln_rate = 0.05
                elif family == "class_imbalance_nonlinear" and imbalance == "balanced":
                    imbalance = "mild"
                elif family == "low_margin_overlap":
                    margin = "low"

                teacher_seed = int(np.random.default_rng([magic, f_idx, r_idx, ds_in_cell, 0]).integers(2**62))
                sample_seed = int(np.random.default_rng([magic, f_idx, r_idx, ds_in_cell, 1]).integers(2**62))
                idx = len(cells)

                cell: dict = {
                    "nonlinear_family": family,
                    "feature_regime": regime,
                    "n": n_val,
                    "p_signal": p_signal,
                    "num_classes": K,
                    "temperature": temp,
                    "label_noise_rate": ln_rate,
                    "class_imbalance_type": imbalance,
                    "margin_level": margin,
                    "suite_component": "core",
                    "teacher_seed": teacher_seed,
                    "sample_seed": sample_seed,
                    "logical_dataset_key": f"nonlinear_cls:{family}:{idx:05d}",
                    "condition_id": f"{family}_{regime}_n{n_val}_p{p_signal}_K{K}_T{temp}_{ds_in_cell}",
                    "is_mixed_categorical": is_mixed_categorical,
                }
                if regime in ("ar1", "block", "equicorr"):
                    rho_rng = np.random.default_rng([magic, f_idx, r_idx, ds_in_cell, 2])
                    cell["rho"] = float(rho_rng.choice(_RHO_GRID_CORR))
                cells.append(cell)

    # -----------------------------------------------------------------------
    # Component B — Robustness
    # -----------------------------------------------------------------------
    target_robust = 20 if is_mixed_categorical else 70
    _CLS_STRESS = [
        ("high_k_low_n",     {"n": 64, "p_signal": 8, "num_classes": 10}),
        ("binary_noisy",     {"n": 256, "num_classes": 2, "label_noise_rate": 0.20}),
        ("severe_imbalance", {"n": 256, "num_classes": 5, "class_imbalance_type": "severe"}),
        ("low_temp",         {"n": 256, "temperature": 0.5}),
        ("high_temp",        {"n": 256, "temperature": 5.0}),
    ]
    for i in range(target_robust):
        f_idx = i % len(families)
        s_idx = (i // len(families)) % len(_CLS_STRESS)
        family = families[f_idx]
        stress_label, stress_kw = _CLS_STRESS[s_idx]

        teacher_seed = int(np.random.default_rng([magic + 1000, f_idx, s_idx, i, 0]).integers(2**62))
        sample_seed = int(np.random.default_rng([magic + 1000, f_idx, s_idx, i, 1]).integers(2**62))
        idx = len(cells)

        b_cell: dict = {
            "nonlinear_family": family,
            "feature_regime": "iid_dense",
            "n": int(stress_kw.get("n", 256)),
            "p_signal": int(stress_kw.get("p_signal", 16)),
            "num_classes": int(stress_kw.get("num_classes", 3)),
            "temperature": float(stress_kw.get("temperature", 1.0)),
            "label_noise_rate": float(stress_kw.get("label_noise_rate", 0.0)),
            "class_imbalance_type": str(stress_kw.get("class_imbalance_type", "balanced")),
            "margin_level": str(stress_kw.get("margin_level", "medium")),
            "suite_component": "robustness",
            "teacher_seed": teacher_seed,
            "sample_seed": sample_seed,
            "logical_dataset_key": f"nonlinear_cls:{family}:{idx:05d}",
            "condition_id": f"{family}_robustness_{stress_label}_{i}",
            "is_mixed_categorical": is_mixed_categorical,
        }
        cells.append(b_cell)

    # -----------------------------------------------------------------------
    # Component C — OOD
    # -----------------------------------------------------------------------
    target_ood = 10 if is_mixed_categorical else 30
    _ood_cls_families = ["radial_decision_boundary", "piecewise_margin",
                         "low_margin_overlap", "multiclass_sparse_highdim"]
    for i in range(target_ood):
        f_idx = i % len(_ood_cls_families)
        family = _ood_cls_families[f_idx]
        teacher_seed = int(np.random.default_rng([magic + 2000, f_idx, i, 0]).integers(2**62))
        sample_seed = int(np.random.default_rng([magic + 2000, f_idx, i, 1]).integers(2**62))
        idx = len(cells)
        c_cell: dict = {
            "nonlinear_family": family,
            "feature_regime": "iid_dense",
            "n": 256,
            "p_signal": 16,
            "num_classes": 3,
            "temperature": 1.0,
            "label_noise_rate": 0.0,
            "class_imbalance_type": "balanced",
            "margin_level": "medium",
            "suite_component": "ood",
            "teacher_seed": teacher_seed,
            "sample_seed": sample_seed,
            "logical_dataset_key": f"nonlinear_cls:{family}:{idx:05d}",
            "condition_id": f"{family}_ood_{i}",
            "is_mixed_categorical": is_mixed_categorical,
        }
        cells.append(c_cell)

    return cells


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def validate_nonlinear_regression_dataset(
    ds: dict,
    *,
    is_mixed_categorical: bool = False,
) -> None:
    """Validate a v3 nonlinear regression dataset. Raises AssertionError on failure."""
    X = ds["X"]
    mu_clean = ds["mu_clean"]
    y = ds["y"]

    assert np.all(np.isfinite(X)), "X contains non-finite values"
    assert np.all(np.isfinite(mu_clean)), "mu_clean contains non-finite values"
    assert np.all(np.isfinite(y)), "y contains non-finite values"

    p_signal = int(ds["p_signal"])
    p_noise = int(ds["p_noise"])
    p_total = int(ds["p_total"])
    assert p_signal + p_noise == p_total, f"p_signal({p_signal}) + p_noise({p_noise}) != p_total({p_total})"
    assert X.shape[1] == p_total, f"X.shape[1]={X.shape[1]} != p_total={p_total}"
    assert X.shape[0] == len(y), "X and y row count mismatch"

    if is_mixed_categorical:
        X_cat = ds["X_cat"]
        assert X_cat.dtype == np.int64, f"X_cat dtype must be int64, got {X_cat.dtype}"
        assert X_cat.shape[0] == len(y), "X_cat and y row count mismatch"
        cardinalities = ds["categorical_cardinalities"]
        from constants import ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_MISSING_ID
        for j in range(X_cat.shape[1]):
            vals = X_cat[:, j]
            valid_mask = (vals >= ENTITY_EMBED_FIRST_REAL_ID) | (vals == ENTITY_EMBED_MISSING_ID)
            assert np.all(valid_mask), f"X_cat column {j} has invalid IDs"
            real_mask = vals >= ENTITY_EMBED_FIRST_REAL_ID
            if real_mask.any():
                max_id = ENTITY_EMBED_FIRST_REAL_ID + cardinalities[j] - 1
                assert vals[real_mask].max() <= max_id, (
                    f"X_cat col {j}: max_id={vals[real_mask].max()} > expected {max_id}"
                )


def validate_nonlinear_classification_dataset(
    ds: dict,
    *,
    is_mixed_categorical: bool = False,
) -> None:
    """Validate a nonlinear classification dataset. Raises AssertionError on failure."""
    X = ds["X"]
    y = ds["y"]
    y_clean = ds["y_clean"]
    logits = ds["logits"]
    probs = ds["probs"]
    label_noise_mask = ds["label_noise_mask"]
    K = int(ds["num_classes"])
    temperature = float(ds["temperature"])
    label_noise_rate = float(ds["label_noise_rate"])
    n = len(y)

    assert np.all(np.isfinite(X)), "X contains non-finite values"
    assert np.all(np.isfinite(logits)), "logits contains non-finite values"

    # Reference class constraint
    assert np.allclose(logits[:, 0], 0.0), "logits[:, 0] must be 0 (reference class)"

    # Softmax consistency
    recomputed_probs = _softmax(logits, temperature)
    assert np.allclose(probs, recomputed_probs, atol=1e-10), "probs != softmax(logits/T)"

    # Probability rows sum to 1
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8), "prob rows do not sum to 1"

    # Label noise count
    expected_flips = int(round(label_noise_rate * n))
    actual_flips = int(label_noise_mask.sum())
    assert actual_flips == expected_flips, (
        f"label_noise_mask count {actual_flips} != expected {expected_flips}"
    )

    # All K classes present in y_clean
    unique_classes = set(y_clean.tolist())
    assert len(unique_classes) == K, (
        f"Expected {K} classes in y_clean, got {len(unique_classes)}: {unique_classes}"
    )

    if is_mixed_categorical:
        X_cat = ds["X_cat"]
        assert X_cat.dtype == np.int64, f"X_cat dtype must be int64, got {X_cat.dtype}"
        cardinalities = ds["categorical_cardinalities"]
        from constants import ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_MISSING_ID
        for j in range(X_cat.shape[1]):
            vals = X_cat[:, j]
            valid_mask = (vals >= ENTITY_EMBED_FIRST_REAL_ID) | (vals == ENTITY_EMBED_MISSING_ID)
            assert np.all(valid_mask), f"X_cat column {j} has invalid IDs"


if __name__ == "__main__":
    main_nonlinear_training()
