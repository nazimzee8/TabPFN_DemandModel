"""
dgp_helpers.py
==============
Shared DGP logic for generate_dgp.py and generate_synthetic_regression.py.

All linear data-generating process code lives here. Both scripts import from
this module — do not duplicate regime, covariance, beta-generation, or
teacher-target logic elsewhere.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

REGRESSION_PROFILES: list[str] = [
    "legacy",
    "linear_stat_aware",
    "linear_stress",
    "market_linear",
]
CLASSIFICATION_PROFILES: list[str] = [
    "linear_classification_stat_aware",
    "linear_classification_stress",
    "market_classification",
    "classification_legacy_debug",
]
PROFILES: list[str] = REGRESSION_PROFILES
ALL_PROFILES: list[str] = REGRESSION_PROFILES + CLASSIFICATION_PROFILES

REGRESSION_EVAL_SCHEMA_VERSION: str = "linear_regression_eval_v2"
CLASSIFICATION_EVAL_SCHEMA_VERSION: str = "linear_classification_eval_v2"
LEGACY_REGRESSION_EVAL_SCHEMA_VERSION: str = "linear_regression_eval_v1"
LEGACY_CLASSIFICATION_EVAL_SCHEMA_VERSION: str = "linear_classification_eval_v1"
SUPPORTED_REGRESSION_EVAL_SCHEMA_VERSIONS: frozenset[str] = frozenset({
    LEGACY_REGRESSION_EVAL_SCHEMA_VERSION,
    REGRESSION_EVAL_SCHEMA_VERSION,
})
SUPPORTED_CLASSIFICATION_EVAL_SCHEMA_VERSIONS: frozenset[str] = frozenset({
    LEGACY_CLASSIFICATION_EVAL_SCHEMA_VERSION,
    CLASSIFICATION_EVAL_SCHEMA_VERSION,
})

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "legacy": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
    "linear_stat_aware": {
        "A_iid_dense": 0.15,
        "B_iid_sparse": 0.15,
        "C_heavy_tail_noise": 0.10,
        "D_correlated_ar": 0.10,
        "E_high_dim_dense": 0.05,
        "F_high_dim_sparse": 0.05,
        "G_noise_features": 0.10,
        "H_block_correlated": 0.05,
        "I_equicorrelated": 0.05,
        "J_low_n_high_p": 0.05,
        "K_feature_noise": 0.05,
        "L_market_linear": 0.10,
    },
    "linear_stress": {
        "A_iid_dense": 0.10,
        "B_iid_sparse": 0.10,
        "C_heavy_tail_noise": 0.10,
        "D_correlated_ar": 0.10,
        "E_high_dim_dense": 0.10,
        "F_high_dim_sparse": 0.10,
        "G_noise_features": 0.10,
        "H_block_correlated": 0.05,
        "I_equicorrelated": 0.05,
        "J_low_n_high_p": 0.10,
        "K_feature_noise": 0.10,
    },
    "market_linear": {
        "A_iid_dense": 0.10,
        "B_iid_sparse": 0.10,
        "C_heavy_tail_noise": 0.05,
        "D_correlated_ar": 0.10,
        "G_noise_features": 0.10,
        "L_market_linear": 0.30,
        "H_block_correlated": 0.10,
        "I_equicorrelated": 0.05,
        "K_feature_noise": 0.10,
    },
    "linear_classification_stat_aware": {
        "A_iid_dense_logistic": 0.15,
        "B_iid_sparse_logistic": 0.15,
        "C_label_noise_margin": 0.10,
        "D_correlated_ar_logistic": 0.10,
        "E_high_dim_dense_softmax": 0.05,
        "F_high_dim_sparse_softmax": 0.05,
        "G_noise_features_classification": 0.10,
        "H_block_correlated_classification": 0.05,
        "I_equicorrelated_classification": 0.05,
        "J_low_n_high_p_classification": 0.05,
        "K_feature_noise_classification": 0.05,
        "L_market_sign_classification": 0.10,
    },
    "linear_classification_stress": {
        "A_iid_dense_logistic": 0.05,
        "B_iid_sparse_logistic": 0.08,
        "C_label_noise_margin": 0.15,
        "D_correlated_ar_logistic": 0.10,
        "E_high_dim_dense_softmax": 0.08,
        "F_high_dim_sparse_softmax": 0.08,
        "G_noise_features_classification": 0.10,
        "H_block_correlated_classification": 0.08,
        "I_equicorrelated_classification": 0.08,
        "J_low_n_high_p_classification": 0.08,
        "K_feature_noise_classification": 0.08,
        "L_market_sign_classification": 0.04,
    },
    "market_classification": {
        "A_iid_dense_logistic": 0.10,
        "B_iid_sparse_logistic": 0.10,
        "C_label_noise_margin": 0.05,
        "D_correlated_ar_logistic": 0.05,
        "E_high_dim_dense_softmax": 0.05,
        "F_high_dim_sparse_softmax": 0.05,
        "G_noise_features_classification": 0.10,
        "H_block_correlated_classification": 0.05,
        "I_equicorrelated_classification": 0.05,
        "J_low_n_high_p_classification": 0.05,
        "K_feature_noise_classification": 0.10,
        "L_market_sign_classification": 0.25,
    },
    "classification_legacy_debug": {
        "A_iid_dense_logistic": 0.25,
        "B_iid_sparse_logistic": 0.25,
        "C_label_noise_margin": 0.25,
        "D_correlated_ar_logistic": 0.25,
    },
}

CLASSIFICATION_REGIMES: tuple[str, ...] = tuple(
    REGIME_WEIGHTS["linear_classification_stat_aware"]
)
CLASS_COUNT_WEIGHTS: dict[int, float] = {2: 0.50, 3: 0.25, 5: 0.15, 10: 0.10}

CLASSIFICATION_EVAL_SUITE_FAMILIES: frozenset[str] = frozenset({
    "primary", "feature_noise", "label_noise", "training_size",
    "class_imbalance", "margin", "num_classes", "ood", "eval_only_unseen",
    "hidden_holdout", "stress",
})

REGRESSION_EVAL_SUITE_FAMILIES: frozenset[str] = frozenset({
    "primary", "feature_noise", "target_noise", "training_size", "sparsity",
    "correlation", "dimensionality", "ood", "eval_only_unseen",
    "hidden_holdout", "stress",
})

ALLOCATION_MODES: frozenset[str] = frozenset({
    "weighted", "weighted_quota", "balanced", "balanced_cartesian", "curriculum",
})

REGRESSION_EVAL_ONLY_REGIMES: tuple[str, ...] = (
    "U_standardized_lognormal",
    "V_three_component_gaussian_mixture",
    "W_low_rank_factor_covariance",
    "X_negative_ar1_covariance",
    "Y_exact_zero_irrelevant",
    "Z_heteroskedastic_linear",
    "AA_sparse_covariate_shift",
    "AB_high_dimensional_linear",
)
CLASSIFICATION_EVAL_ONLY_REGIMES: tuple[str, ...] = (
    "U_standardized_lognormal_classification",
    "V_three_component_gaussian_mixture_classification",
    "W_low_rank_factor_covariance_classification",
    "X_negative_ar1_covariance_classification",
    "Y_exact_zero_irrelevant_classification",
    "Z_heteroskedastic_margin_classification",
    "AA_sparse_covariate_shift_classification",
    "AB_high_dimensional_linear_classification",
)


def _regime_spec(
    *,
    family: str,
    training: bool,
    eval_only: bool = False,
    default: bool = False,
    **generator_parameters: Any,
) -> dict[str, Any]:
    return {
        "is_training_allowed": training,
        "is_eval_only": eval_only,
        "distribution_family": family,
        "is_default_family": default,
        "default_family": family if default else None,
        "generator_parameters": generator_parameters,
    }


REGRESSION_REGIME_REGISTRY: dict[str, dict[str, Any]] = {
    regime: _regime_spec(
        family="gaussian_linear",
        training=True,
        default=True,
    )
    for regime in (
        "A_iid_dense", "B_iid_sparse", "C_heavy_tail_noise",
        "D_correlated_ar", "E_high_dim_dense", "F_high_dim_sparse",
        "G_noise_features", "H_block_correlated", "I_equicorrelated",
        "J_low_n_high_p", "K_feature_noise", "L_market_linear",
    )
}
REGRESSION_REGIME_REGISTRY.update({
    "U_standardized_lognormal": _regime_spec(
        family="standardized_lognormal", training=False, eval_only=True
    ),
    "V_three_component_gaussian_mixture": _regime_spec(
        family="gaussian_mixture", training=False, eval_only=True, n_components=3
    ),
    "W_low_rank_factor_covariance": _regime_spec(
        family="low_rank_factor", training=False, eval_only=True
    ),
    "X_negative_ar1_covariance": _regime_spec(
        family="negative_ar1", training=False, eval_only=True, rho=-0.65
    ),
    "Y_exact_zero_irrelevant": _regime_spec(
        family="irrelevant_features", training=False, eval_only=True
    ),
    "Z_heteroskedastic_linear": _regime_spec(
        family="heteroskedastic_linear", training=False, eval_only=True
    ),
    "AA_sparse_covariate_shift": _regime_spec(
        family="sparse_covariate_shift", training=False, eval_only=True
    ),
    "AB_high_dimensional_linear": _regime_spec(
        family="high_dimensional_linear", training=False, eval_only=True
    ),
})

CLASSIFICATION_REGIME_REGISTRY: dict[str, dict[str, Any]] = {
    regime: _regime_spec(
        family="gaussian_linear_logits",
        training=True,
        default=True,
    )
    for regime in (
        "A_iid_dense_logistic", "B_iid_sparse_logistic",
        "C_label_noise_margin", "D_correlated_ar_logistic",
        "E_high_dim_dense_softmax", "F_high_dim_sparse_softmax",
        "G_noise_features_classification",
        "H_block_correlated_classification",
        "I_equicorrelated_classification",
        "J_low_n_high_p_classification",
        "K_feature_noise_classification",
        "L_market_sign_classification",
    )
}
CLASSIFICATION_REGIME_REGISTRY.update({
    "U_standardized_lognormal_classification": _regime_spec(
        family="standardized_lognormal", training=False, eval_only=True
    ),
    "V_three_component_gaussian_mixture_classification": _regime_spec(
        family="gaussian_mixture", training=False, eval_only=True, n_components=3
    ),
    "W_low_rank_factor_covariance_classification": _regime_spec(
        family="low_rank_factor", training=False, eval_only=True
    ),
    "X_negative_ar1_covariance_classification": _regime_spec(
        family="negative_ar1", training=False, eval_only=True, rho=-0.65
    ),
    "Y_exact_zero_irrelevant_classification": _regime_spec(
        family="irrelevant_features", training=False, eval_only=True
    ),
    "Z_heteroskedastic_margin_classification": _regime_spec(
        family="heteroskedastic_margin", training=False, eval_only=True
    ),
    "AA_sparse_covariate_shift_classification": _regime_spec(
        family="sparse_covariate_shift", training=False, eval_only=True
    ),
    "AB_high_dimensional_linear_classification": _regime_spec(
        family="high_dimensional_linear", training=False, eval_only=True
    ),
})

_REGRESSION_ONLY_FIELDS: frozenset[str] = frozenset({
    "betaX", "beta0", "target_noise_scale",
    "heteroscedastic_scale", "noise_distribution", "link_function",
})

IMBALANCE_WEIGHTS: dict[str, float] = {
    "balanced": 0.50,
    "mild": 0.25,
    "moderate": 0.20,
    "severe": 0.05,
}
IMBALANCE_STRENGTHS: dict[str, float] = {
    "balanced": 0.0,
    "mild": 0.15,
    "moderate": 0.35,
    "severe": 0.60,
}

# ---------------------------------------------------------------------------
# 2. Grid Parsing
# ---------------------------------------------------------------------------

def parse_int_grid(s: str) -> list[int]:
    """Parse a comma-separated string of integers into a list."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_grid(s: str) -> list[float]:
    """Parse a comma-separated string of floats into a list."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_string_grid(s: str) -> list[str]:
    """Parse a comma-separated string into a non-empty list of strings."""
    return [x.strip() for x in s.split(",") if x.strip()]


def largest_remainder_allocation(
    total: int,
    labels: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Allocate ``total`` deterministically while preserving the exact sum."""
    if total < 0:
        raise ValueError("total must be non-negative")
    ordered = list(dict.fromkeys(labels))
    if not ordered:
        if total:
            raise ValueError("cannot allocate a positive total to no labels")
        return {}
    raw_weights = np.asarray(
        [float(weights.get(label, 0.0)) if weights else 1.0 for label in ordered],
        dtype=float,
    )
    if np.any(raw_weights < 0) or not np.isfinite(raw_weights).all():
        raise ValueError("allocation weights must be finite and non-negative")
    if raw_weights.sum() <= 0:
        raw_weights[:] = 1.0
    quotas = total * raw_weights / raw_weights.sum()
    floors = np.floor(quotas).astype(int)
    remainder = total - int(floors.sum())
    order = sorted(
        range(len(ordered)),
        key=lambda idx: (-(quotas[idx] - floors[idx]), idx),
    )
    for idx in order[:remainder]:
        floors[idx] += 1
    return {label: int(floors[idx]) for idx, label in enumerate(ordered)}


def allocate_regimes(
    total: int,
    regimes: Sequence[str],
    *,
    mode: str,
    weights: Mapping[str, float] | None = None,
    axis_values: Mapping[str, Sequence[Any]] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return deterministic regime/axis assignments for all allocation modes."""
    if mode not in ALLOCATION_MODES:
        raise ValueError(f"unknown allocation mode {mode!r}")
    regimes = list(dict.fromkeys(regimes))
    if mode == "weighted":
        rng = np.random.default_rng(seed)
        probs = np.asarray(
            [float(weights.get(r, 0.0)) if weights else 1.0 for r in regimes],
            dtype=float,
        )
        probs = probs / probs.sum()
        return [{"regime": str(rng.choice(regimes, p=probs))} for _ in range(total)]

    if mode == "balanced_cartesian" and axis_values:
        import itertools

        axes = [(key, list(values)) for key, values in axis_values.items() if values]
        cells = [
            {"regime": regime, **dict(zip((k for k, _ in axes), values))}
            for regime in regimes
            for values in itertools.product(*(vals for _, vals in axes))
        ]
        if not cells:
            return []
        return [dict(cells[idx % len(cells)]) for idx in range(total)]

    allocation_weights = weights if mode == "weighted_quota" else None
    counts = largest_remainder_allocation(total, regimes, allocation_weights)
    assignments: list[dict[str, Any]] = []
    axis_positions = {key: 0 for key in (axis_values or {})}
    while len(assignments) < total:
        progressed = False
        for regime in regimes:
            if counts[regime] <= 0:
                continue
            assignment: dict[str, Any] = {"regime": regime}
            for key, values in (axis_values or {}).items():
                values = list(values)
                if values:
                    pos = axis_positions[key]
                    assignment[key] = values[pos % len(values)]
                    axis_positions[key] = pos + 1
            assignments.append(assignment)
            counts[regime] -= 1
            progressed = True
        if not progressed:
            break
    return assignments


def estimate_generation_memory(
    n_train: int,
    n_holdout: int,
    p_total: int,
    *,
    hidden_channels: int = 128,
    float_bytes: int = 4,
    safety_factor: float = 2.0,
) -> dict[str, Any]:
    """Estimate evaluator tensor memory using the documented bounded dimensions."""
    context = min(max(0, int(n_train)), 1024)
    query = min(max(0, int(n_holdout)), 256)
    features = min(max(0, int(p_total)), 256)
    estimated_bytes = int(
        safety_factor
        * float_bytes
        * hidden_channels
        * features
        * (context + query)
    )
    gib = estimated_bytes / (1024 ** 3)
    memory_class = "low" if gib < 2.0 else ("medium" if gib <= 8.0 else "high")
    return {
        "estimated_memory_bytes": estimated_bytes,
        "estimated_memory_gib": gib,
        "memory_class": memory_class,
        "memory_context_rows": context,
        "memory_query_rows": query,
        "memory_selected_features": features,
        "memory_hidden_channels": hidden_channels,
        "memory_float_bytes": float_bytes,
        "memory_safety_factor": safety_factor,
    }


def compute_difficulty_metadata(
    *,
    n_train: int,
    p_total: int,
    noise_scale: float = 0.0,
    feature_noise_level: float = 0.0,
    sparsity_ratio: float = 0.0,
    rho: float = 0.0,
    num_classes: int = 1,
    imbalance_strength: float = 0.0,
) -> dict[str, Any]:
    """Compute a deterministic, component-based difficulty score in ``[0, 100]``."""
    components = {
        "dimensionality": 25.0 * min(1.0, p_total / max(1.0, float(n_train))),
        "target_or_label_noise": 20.0 * min(1.0, max(0.0, noise_scale) / 2.0),
        "feature_noise": 15.0 * min(1.0, max(0.0, feature_noise_level)),
        "sparsity": 10.0 * min(1.0, max(0.0, sparsity_ratio)),
        "correlation": 10.0 * min(1.0, abs(float(rho))),
        "class_geometry": 10.0 * min(1.0, max(0, num_classes - 2) / 8.0),
        "imbalance": 10.0 * min(1.0, max(0.0, imbalance_strength)),
    }
    score = float(np.clip(sum(components.values()), 0.0, 100.0))
    tier = "easy" if score <= 33.0 else ("medium" if score <= 66.0 else "hard")
    reasons = [
        name for name, value in sorted(
            components.items(), key=lambda item: (-item[1], item[0])
        )
        if value >= 5.0
    ]
    return {
        "difficulty_score": score,
        "difficulty_tier": tier,
        "difficulty_reasons": reasons,
        "difficulty_components": components,
    }


def build_task_fingerprint(
    *,
    task_family: str,
    regime: str,
    suite_family: str,
    n_total: int,
    p_total: int,
    covariance_type: str,
    noise_type: str,
    sparsity_ratio: float,
    classification_geometry: str = "",
) -> str:
    """Build a stable coarse fingerprint used by leakage audits."""
    import hashlib

    def _bucket(value: int) -> int:
        return 0 if value <= 0 else int(2 ** math.floor(math.log2(value)))

    payload = "|".join(map(str, (
        task_family,
        regime,
        suite_family,
        _bucket(int(n_total)),
        _bucket(int(p_total)),
        covariance_type,
        noise_type,
        round(float(sparsity_ratio), 2),
        classification_geometry,
    )))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def validate_coverage(
    records: Sequence[Mapping[str, Any]],
    applicable_values: Mapping[str, Iterable[Any]],
    *,
    minimums: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate only configured/applicable axis values and return an audit."""
    minimums = minimums or {}
    missing: dict[str, list[Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    for axis, values in applicable_values.items():
        axis_counts: dict[str, int] = {}
        minimum = int(minimums.get(axis, 1))
        for value in values:
            count = sum(1 for record in records if record.get(axis) == value)
            axis_counts[str(value)] = count
            if count < minimum:
                missing.setdefault(axis, []).append(value)
        counts[axis] = axis_counts
    return {"ok": not missing, "counts": counts, "missing": missing}


# ---------------------------------------------------------------------------
# 3. Parameter Sampling
# ---------------------------------------------------------------------------

def sample_n_p(
    rng: np.random.Generator,
    profile: str,
    n_grid: list[int],
    p_signal_grid: list[int],
    p_noise_grid: list[int],
    active_s_grid: list[int],
    rho_grid: list[float],
    target_noise_grid: list[float],
    feature_noise_grid: list[float],
    allow_underdetermined: bool = False,
) -> dict:
    """Sample DGP parameters.

    Legacy branch: Poisson(200)/Poisson(10) rejection sampler, returns minimal dict.
    Non-legacy branch: Uniform draw from each grid with optional underdetermined check.
    """
    if profile == "legacy":
        while True:
            n = int(rng.poisson(200))
            p = int(rng.poisson(10))
            if p >= 1 and n >= 5 and n >= 5 * p:
                return {
                    "n": n,
                    "p_signal": p,
                    "p_noise": 0,
                    "active_s": p,
                    "rho": 0.0,
                    "target_noise_scale": 1.0,
                    "feature_noise_level": 0.0,
                    "covariance_type": "iid",
                }

    # Non-legacy: grid-based sampling
    for _ in range(10_000):
        n = int(rng.choice(n_grid))
        p_signal = int(rng.choice(p_signal_grid))
        p_noise = int(rng.choice(p_noise_grid))
        p_total = p_signal + p_noise
        active_s_raw = int(rng.choice(active_s_grid))
        active_s = int(np.clip(active_s_raw, 1, p_signal))
        rho = float(rng.choice(rho_grid))
        target_noise_scale = float(rng.choice(target_noise_grid))
        feature_noise_level = float(rng.choice(feature_noise_grid))

        if allow_underdetermined or n >= p_total:
            return {
                "n": n,
                "p_signal": p_signal,
                "p_noise": p_noise,
                "active_s": active_s,
                "rho": rho,
                "target_noise_scale": target_noise_scale,
                "feature_noise_level": feature_noise_level,
                "covariance_type": "iid",
            }

    # Fallback: always-valid parameters
    p_signal = int(rng.choice(p_signal_grid))
    return {
        "n": max(n_grid),
        "p_signal": p_signal,
        "p_noise": 0,
        "active_s": p_signal,
        "rho": 0.0,
        "target_noise_scale": 1.0,
        "feature_noise_level": 0.0,
        "covariance_type": "iid",
    }


# ---------------------------------------------------------------------------
# 4. Feature Generation
# ---------------------------------------------------------------------------

def generate_covariance_matrix(
    p: int,
    cov_type: str,
    rho: float,
    block_size: int = 8,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Build a p×p covariance matrix for the given structure."""
    if cov_type == "iid":
        return np.eye(p)

    if cov_type == "ar1":
        idx = np.arange(p)
        Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    elif cov_type == "equicorrelation":
        Sigma = np.full((p, p), rho)
        np.fill_diagonal(Sigma, 1.0)
    elif cov_type == "block":
        Sigma = np.full((p, p), 0.1)
        for start in range(0, p, block_size):
            end = min(start + block_size, p)
            Sigma[start:end, start:end] = 0.7
        np.fill_diagonal(Sigma, 1.0)
    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")

    Sigma += jitter * np.eye(p)
    return Sigma


def generate_X(
    rng: np.random.Generator,
    n: int,
    p: int,
    cov_type: str,
    rho: float,
    block_size: int = 8,
) -> np.ndarray:
    """Generate an n×p feature matrix with the requested covariance structure."""
    if cov_type == "iid":
        return rng.standard_normal((n, p))

    if cov_type == "ar1":
        # Column-wise recursion — avoids Cholesky for large p
        X = np.empty((n, p))
        X[:, 0] = rng.standard_normal(n)
        scale = math.sqrt(max(0.0, 1.0 - rho ** 2))
        for k in range(1, p):
            X[:, k] = rho * X[:, k - 1] + scale * rng.standard_normal(n)
        return X

    # Cholesky path for block / equicorrelation
    Sigma = generate_covariance_matrix(p, cov_type, rho, block_size=block_size)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, p))
    return Z @ L.T


# ---------------------------------------------------------------------------
# 5. Coefficient Generation
# ---------------------------------------------------------------------------

def _generate_market_beta(
    rng: np.random.Generator,
    p_signal: int,
    beta_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Market-structured beta: own-price (−), cross-price (±), attributes, irrelevant."""
    beta = np.zeros(p_signal)
    support = np.zeros(p_signal, dtype=bool)

    quarter = max(1, p_signal // 4)
    # Own-price: negative
    own_end = quarter
    beta[:own_end] = -np.abs(rng.normal(0, beta_scale, own_end))
    support[:own_end] = True

    # Cross-price: alternating sign
    cross_end = 2 * quarter
    raw = rng.normal(0, beta_scale, cross_end - own_end)
    signs = np.where(np.arange(cross_end - own_end) % 2 == 0, 1.0, -1.0)
    beta[own_end:cross_end] = np.abs(raw) * signs
    support[own_end:cross_end] = True

    # Attributes: unconstrained
    attr_end = 3 * quarter
    beta[cross_end:attr_end] = rng.normal(0, beta_scale, attr_end - cross_end)
    support[cross_end:attr_end] = True

    # Irrelevant: zero (support stays False)
    return beta, support


def generate_beta(
    rng: np.random.Generator,
    p_signal: int,
    beta_pattern: str,
    active_s: int,
    beta_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (beta_signal, support_mask) for the given pattern."""
    if beta_pattern == "dense_gaussian":
        beta = rng.normal(0, beta_scale, p_signal)
        support = np.ones(p_signal, dtype=bool)
        return beta, support

    if beta_pattern == "sparse":
        beta = np.zeros(p_signal)
        support = np.zeros(p_signal, dtype=bool)
        active_s = min(active_s, p_signal)
        idx = rng.choice(p_signal, size=active_s, replace=False)
        # Variance-normalise by sqrt(p_signal / active_s)
        scale = beta_scale * math.sqrt(p_signal / active_s)
        beta[idx] = rng.normal(0, scale, active_s)
        support[idx] = True
        return beta, support

    if beta_pattern == "decaying":
        beta = np.zeros(p_signal)
        support = np.ones(p_signal, dtype=bool)
        for i in range(p_signal):
            beta[i] = beta_scale * rng.standard_normal() * math.exp(-i / max(1, active_s))
        return beta, support

    if beta_pattern == "market_sign":
        return _generate_market_beta(rng, p_signal, beta_scale)

    raise ValueError(f"Unknown beta_pattern: {beta_pattern}")


# ---------------------------------------------------------------------------
# 6. Noise and Observation
# ---------------------------------------------------------------------------

def add_noise_features(
    rng: np.random.Generator,
    X_signal: np.ndarray,
    p_noise: int,
    cov_type: str,
    rho: float,
) -> np.ndarray:
    """Append p_noise irrelevant columns; returns X_total shape (n, p_signal+p_noise)."""
    if p_noise == 0:
        return X_signal
    n = X_signal.shape[0]
    X_noise = generate_X(rng, n, p_noise, cov_type, rho)
    return np.hstack([X_signal, X_noise])


def add_feature_noise(
    rng: np.random.Generator,
    X_clean: np.ndarray,
    feature_noise_level: float,
) -> np.ndarray:
    """Add measurement noise: X_observed = X_clean + level * N(0,1)."""
    if feature_noise_level == 0.0:
        return X_clean
    return X_clean + feature_noise_level * rng.standard_normal(X_clean.shape)


def generate_target(
    rng: np.random.Generator,
    betaX: np.ndarray,
    target_noise_scale: float,
    noise_type: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (y, eps) given the noiseless signal betaX."""
    n = betaX.shape[0]
    if noise_type == "student_t":
        eps = rng.standard_t(df=3, size=n) * target_noise_scale
    else:
        eps = rng.standard_normal(n) * target_noise_scale
    return betaX + eps, eps


# ---------------------------------------------------------------------------
# 7. Legacy Dataset Builder
# ---------------------------------------------------------------------------

def _build_legacy_dataset(
    rng: np.random.Generator,
    regime: str,
    params: dict,
) -> dict:
    """Bit-identical to the original _generate_dataset in both scripts."""
    n = params["n"]
    p = params["p_signal"]  # p_noise=0 in legacy

    # Features
    if regime in ("A", "B", "C"):
        X = rng.standard_normal((n, p))
    else:  # D
        X = np.empty((n, p))
        X[:, 0] = rng.standard_normal(n)
        for k in range(1, p):
            X[:, k] = 0.6 * X[:, k - 1] + math.sqrt(0.64) * rng.standard_normal(n)

    # Coefficients
    if regime == "B":
        beta = rng.normal(0, 2, size=p)
        mask = rng.random(p) < 0.70
        beta[mask] = 0.0
        support = ~mask
    else:
        beta = rng.standard_normal(p)
        support = np.ones(p, dtype=bool)

    # Noise
    if regime == "C":
        eps = rng.standard_t(df=3, size=n)
    else:
        eps = rng.standard_normal(n)

    betaX = X @ beta
    y = betaX + eps

    # Map short name to long name for consistency
    _regime_map = {"A": "A_iid_dense", "B": "B_iid_sparse", "C": "C_heavy_tail_noise", "D": "D_correlated_ar"}
    prior_regime = _regime_map.get(regime, regime)

    sparsity_ratio = float(np.sum(support == False)) / p if p > 0 else 0.0

    return {
        "X": X.astype(np.float64),
        "X_clean": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "beta": beta.astype(np.float64),
        "support_mask": support,
        "prior_regime": prior_regime,
        "regime_group": regime,
        "n_total": n,
        "p_signal": p,
        "p_noise": 0,
        "p_total": p,
        "active_s": int(np.sum(support)),
        "sparsity_ratio": sparsity_ratio,
        "covariance_type": "ar1" if regime == "D" else "iid",
        "rho": 0.6 if regime == "D" else 0.0,
        "target_noise_scale": 1.0,
        "feature_noise_level": 0.0,
        "target_noise_type": "student_t" if regime == "C" else "gaussian",
        "has_noise_features": False,
        "has_feature_noise": False,
    }


# ---------------------------------------------------------------------------
# 8. New Regime Builders
# ---------------------------------------------------------------------------

def _base_dataset(
    *,
    X_clean: np.ndarray,
    y: np.ndarray,
    betaX: np.ndarray,
    beta_signal: np.ndarray,
    support: np.ndarray,
    p_noise: int,
    regime: str,
    cov_type: str,
    rho: float,
    target_noise_scale: float,
    feature_noise_level: float,
    target_noise_type: str,
    X_noisy: np.ndarray | None = None,
) -> dict:
    """Assemble the standard return dict shared by all regime builders."""
    n, p_total = X_clean.shape
    p_signal = p_total - p_noise

    # Full beta including zeros for noise columns
    beta_full = np.zeros(p_total)
    beta_full[:p_signal] = beta_signal

    support_full = np.zeros(p_total, dtype=bool)
    support_full[:p_signal] = support

    X_obs = X_noisy if X_noisy is not None else X_clean
    active_s = int(np.sum(support))
    sparsity_ratio = 1.0 - active_s / max(1, p_signal)

    return {
        "X": X_obs.astype(np.float64),
        "X_clean": X_clean.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "beta": beta_full.astype(np.float64),
        "support_mask": support_full,
        "prior_regime": regime,
        "regime_group": regime,
        "n_total": n,
        "p_signal": p_signal,
        "p_noise": p_noise,
        "p_total": p_total,
        "active_s": active_s,
        "sparsity_ratio": sparsity_ratio,
        "covariance_type": cov_type,
        "rho": rho,
        "target_noise_scale": target_noise_scale,
        "feature_noise_level": feature_noise_level,
        "target_noise_type": target_noise_type,
        "has_noise_features": p_noise > 0,
        "has_feature_noise": feature_noise_level > 0.0,
    }


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, keepdims=True)
    return (X - mean) / np.where(scale > 1e-12, scale, 1.0)


def generate_eval_only_features(
    rng: np.random.Generator,
    regime: str,
    n: int,
    p: int,
) -> tuple[np.ndarray, str, float]:
    """Generate unseen feature distributions without changing the linear law."""
    base = regime.replace("_classification", "")
    if base == "U_standardized_lognormal":
        X = _standardize_columns(rng.lognormal(mean=0.0, sigma=0.8, size=(n, p)))
        return X, "standardized_lognormal", 0.0
    if base == "V_three_component_gaussian_mixture":
        components = rng.choice(3, size=n, p=[0.35, 0.40, 0.25])
        means = np.asarray([-1.5, 0.0, 1.75])
        scales = np.asarray([0.65, 1.0, 0.8])
        X = rng.standard_normal((n, p)) * scales[components, None]
        X += means[components, None]
        return _standardize_columns(X), "gaussian_mixture_3", 0.0
    if base == "W_low_rank_factor_covariance":
        rank = max(1, min(8, p // 4))
        factors = rng.standard_normal((n, rank))
        loadings = rng.standard_normal((rank, p)) / math.sqrt(rank)
        X = factors @ loadings + 0.2 * rng.standard_normal((n, p))
        return _standardize_columns(X), "low_rank_factor", 0.0
    if base == "X_negative_ar1_covariance":
        rho = -0.65
        return generate_X(rng, n, p, "ar1", rho), "ar1", rho
    if base == "AA_sparse_covariate_shift":
        X = rng.standard_normal((n, p))
        shift_cols = max(1, p // 5)
        X[:, :shift_cols] += rng.choice([-2.0, 2.0], size=(n, 1))
        return X, "sparse_covariate_shift", 0.0
    return rng.standard_normal((n, p)), "iid", 0.0


def _build_eval_only_regression(
    rng: np.random.Generator,
    regime: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    p_signal = max(4, int(params.get("p_signal", 16)))
    p_noise = max(0, int(params.get("p_noise", 0)))
    if regime == "Y_exact_zero_irrelevant":
        p_noise = max(8, p_noise)
    if regime == "AB_high_dimensional_linear":
        p_signal = max(64, p_signal)
    n = max(16, int(params.get("n", 128)))
    if regime != "AB_high_dimensional_linear":
        n = max(n, p_signal + p_noise)
    target_noise_scale = float(params.get("target_noise_scale", 1.0))
    X_signal, covariance_type, rho = generate_eval_only_features(
        rng, regime, n, p_signal
    )
    active_s = min(
        p_signal,
        max(1, int(params.get("active_s", p_signal))),
    )
    beta_pattern = (
        "sparse"
        if regime in {"AA_sparse_covariate_shift", "AB_high_dimensional_linear"}
        else "dense_gaussian"
    )
    beta_signal, support = generate_beta(
        rng, p_signal, beta_pattern, active_s
    )
    X_clean = add_noise_features(rng, X_signal, p_noise, "iid", 0.0)
    betaX = X_signal @ beta_signal
    if regime == "Z_heteroskedastic_linear":
        standardized_abs_score = np.abs(
            (betaX - betaX.mean()) / max(betaX.std(), 1e-12)
        )
        residual_scale = target_noise_scale * np.clip(
            0.5 + standardized_abs_score, 0.5, 3.0
        )
        y = betaX + rng.standard_normal(n) * residual_scale
        target_noise_type = "heteroskedastic_gaussian"
    else:
        y, _ = generate_target(rng, betaX, target_noise_scale, "gaussian")
        residual_scale = np.full(n, target_noise_scale)
        target_noise_type = "gaussian"
    ds = _base_dataset(
        X_clean=X_clean,
        y=y,
        betaX=betaX,
        beta_signal=beta_signal,
        support=support,
        p_noise=p_noise,
        regime=regime,
        cov_type=covariance_type,
        rho=rho,
        target_noise_scale=target_noise_scale,
        feature_noise_level=0.0,
        target_noise_type=target_noise_type,
    )
    ds["heteroskedastic_scale"] = residual_scale.astype(np.float64)
    ds["distribution_family"] = REGRESSION_REGIME_REGISTRY[regime][
        "distribution_family"
    ]
    return ds


def _build_E_high_dim_dense(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(32, min(128, params["p_signal"]))
    rho = params.get("rho", 0.0)
    tns = params.get("target_noise_scale", 1.0)
    n = max(p_signal * 2, params["n"])

    X_clean = generate_X(rng, n, p_signal, "iid", 0.0)
    beta_sig, support = generate_beta(rng, p_signal, "dense_gaussian", p_signal)
    betaX = X_clean @ beta_sig
    y, _ = generate_target(rng, betaX, tns, "gaussian")

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=0, regime="E_high_dim_dense", cov_type="iid", rho=0.0,
        target_noise_scale=tns, feature_noise_level=0.0, target_noise_type="gaussian",
    )


def _build_F_high_dim_sparse(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(32, min(128, params["p_signal"]))
    active_s = max(1, min(params.get("active_s", max(1, p_signal // 8)), p_signal // 4))
    tns = params.get("target_noise_scale", 1.0)
    n = max(p_signal * 2, params["n"])

    X_clean = generate_X(rng, n, p_signal, "iid", 0.0)
    beta_sig, support = generate_beta(rng, p_signal, "sparse", active_s)
    betaX = X_clean @ beta_sig
    y, _ = generate_target(rng, betaX, tns, "gaussian")

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=0, regime="F_high_dim_sparse", cov_type="iid", rho=0.0,
        target_noise_scale=tns, feature_noise_level=0.0, target_noise_type="gaussian",
    )


def _build_G_noise_features(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(4, min(32, params["p_signal"]))
    p_noise = max(8, min(120, params.get("p_noise", 24)))
    rho = params.get("rho", 0.0)
    cov_type = "ar1" if rho > 0.0 else "iid"
    tns = params.get("target_noise_scale", 1.0)
    active_s = params.get("active_s", p_signal)
    n = max(p_signal + p_noise, params["n"])

    X_signal = generate_X(rng, n, p_signal, cov_type, rho)
    beta_sig, support = generate_beta(rng, p_signal, "dense_gaussian", active_s)
    betaX = X_signal @ beta_sig
    y, _ = generate_target(rng, betaX, tns, "gaussian")

    X_clean = add_noise_features(rng, X_signal, p_noise, cov_type, rho)
    # betaX uses only signal features; beta_full[p_signal:] = 0

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=p_noise, regime="G_noise_features", cov_type=cov_type, rho=rho,
        target_noise_scale=tns, feature_noise_level=0.0, target_noise_type="gaussian",
    )


def _build_H_block_correlated(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(8, min(64, params["p_signal"]))
    block_size = int(rng.choice([4, 8, 16]))
    tns = params.get("target_noise_scale", 1.0)
    n = max(p_signal * 2, params["n"])

    X_clean = generate_X(rng, n, p_signal, "block", 0.7, block_size=block_size)
    beta_sig, support = generate_beta(rng, p_signal, "dense_gaussian", p_signal)
    betaX = X_clean @ beta_sig
    y, _ = generate_target(rng, betaX, tns, "gaussian")

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=0, regime="H_block_correlated", cov_type="block", rho=0.7,
        target_noise_scale=tns, feature_noise_level=0.0, target_noise_type="gaussian",
    )


def _build_I_equicorrelated(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(8, min(64, params["p_signal"]))
    rho_choices = [0.1, 0.3, 0.5, 0.7, 0.9]
    rho = float(rng.choice(rho_choices))
    tns = params.get("target_noise_scale", 1.0)
    n = max(p_signal * 2, params["n"])

    X_clean = generate_X(rng, n, p_signal, "equicorrelation", rho)
    beta_sig, support = generate_beta(rng, p_signal, "dense_gaussian", p_signal)
    betaX = X_clean @ beta_sig
    y, _ = generate_target(rng, betaX, tns, "gaussian")

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=0, regime="I_equicorrelated", cov_type="equicorrelation", rho=rho,
        target_noise_scale=tns, feature_noise_level=0.0, target_noise_type="gaussian",
    )


def _build_J_low_n_high_p(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(8, min(64, params["p_signal"]))
    # Force n <= p_total (underdetermined)
    n = max(2, min(p_signal - 1, params["n"]))
    active_s = max(1, min(params.get("active_s", p_signal // 4), p_signal))
    tns = params.get("target_noise_scale", 1.0)

    X_clean = generate_X(rng, n, p_signal, "iid", 0.0)
    beta_sig, support = generate_beta(rng, p_signal, "sparse", active_s)
    betaX = X_clean @ beta_sig
    y, _ = generate_target(rng, betaX, tns, "gaussian")

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=0, regime="J_low_n_high_p", cov_type="iid", rho=0.0,
        target_noise_scale=tns, feature_noise_level=0.0, target_noise_type="gaussian",
    )


def _build_K_feature_noise(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(4, min(64, params["p_signal"]))
    feature_noise_level = params.get("feature_noise_level", 0.1)
    if feature_noise_level == 0.0:
        feature_noise_level = 0.1  # K always has feature noise
    tns = params.get("target_noise_scale", 1.0)
    n = max(p_signal * 2, params["n"])

    X_clean = generate_X(rng, n, p_signal, "iid", 0.0)
    beta_sig, support = generate_beta(rng, p_signal, "dense_gaussian", p_signal)
    betaX = X_clean @ beta_sig  # always computed on clean X
    y, _ = generate_target(rng, betaX, tns, "gaussian")
    X_noisy = add_feature_noise(rng, X_clean, feature_noise_level)

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=0, regime="K_feature_noise", cov_type="iid", rho=0.0,
        target_noise_scale=tns, feature_noise_level=feature_noise_level,
        target_noise_type="gaussian", X_noisy=X_noisy,
    )


def _build_L_market_linear(rng: np.random.Generator, params: dict) -> dict:
    p_signal = max(8, min(64, params["p_signal"]))
    p_noise = max(0, min(32, params.get("p_noise", 0)))
    tns = params.get("target_noise_scale", 1.0)
    use_block = rng.random() < 0.5
    cov_type = "block" if use_block else "iid"
    rho = 0.7 if use_block else 0.0
    n = max(p_signal + p_noise + 1, params["n"])

    X_signal = generate_X(rng, n, p_signal, cov_type, rho)
    beta_sig, support = generate_beta(rng, p_signal, "market_sign", p_signal)
    betaX = X_signal @ beta_sig
    y, _ = generate_target(rng, betaX, tns, "gaussian")

    X_clean = add_noise_features(rng, X_signal, p_noise, cov_type, rho)

    return _base_dataset(
        X_clean=X_clean, y=y, betaX=betaX, beta_signal=beta_sig, support=support,
        p_noise=p_noise, regime="L_market_linear", cov_type=cov_type, rho=rho,
        target_noise_scale=tns, feature_noise_level=0.0, target_noise_type="gaussian",
    )


# ---------------------------------------------------------------------------
# 9. Top-Level Factory
# ---------------------------------------------------------------------------

_LEGACY_SHORT_NAMES = {"A", "B", "C", "D"}
_LEGACY_LONG_TO_SHORT = {
    "A_iid_dense": "A",
    "B_iid_sparse": "B",
    "C_heavy_tail_noise": "C",
    "D_correlated_ar": "D",
}
_REGIME_BUILDERS = {
    "E_high_dim_dense": _build_E_high_dim_dense,
    "F_high_dim_sparse": _build_F_high_dim_sparse,
    "G_noise_features": _build_G_noise_features,
    "H_block_correlated": _build_H_block_correlated,
    "I_equicorrelated": _build_I_equicorrelated,
    "J_low_n_high_p": _build_J_low_n_high_p,
    "K_feature_noise": _build_K_feature_noise,
    "L_market_linear": _build_L_market_linear,
}


def build_dataset_from_regime(
    rng: np.random.Generator,
    regime: str,
    profile: str,
    params: dict,
) -> dict:
    """Top-level factory: dispatch to legacy or new regime builder."""
    # Normalise long legacy names to short for the legacy builder
    short = _LEGACY_LONG_TO_SHORT.get(regime, regime)
    is_legacy_regime = short in _LEGACY_SHORT_NAMES

    if profile == "legacy" or is_legacy_regime:
        # Ensure params has n and p_signal via legacy sampler if needed
        legacy_params = dict(params)
        if "n" not in legacy_params:
            legacy_params["n"] = params.get("n", 200)
        if "p_signal" not in legacy_params:
            legacy_params["p_signal"] = params.get("p", params.get("p_signal", 10))
        return _build_legacy_dataset(rng, short, legacy_params)

    if regime in REGRESSION_EVAL_ONLY_REGIMES:
        return _build_eval_only_regression(rng, regime, params)
    builder = _REGIME_BUILDERS.get(regime)
    if builder is None:
        raise ValueError(f"Unknown regime: {regime!r}")
    return builder(rng, params)


# ---------------------------------------------------------------------------
# 10. Validation
# ---------------------------------------------------------------------------

def validate_dataset(ds: dict) -> None:
    """Raise AssertionError with descriptive message for any schema violation."""
    n = ds["n_total"]
    p_total = ds["p_total"]
    p_signal = ds["p_signal"]
    p_noise = ds["p_noise"]

    X = ds["X"]
    y = ds["y"]
    betaX = ds["betaX"]
    beta = ds["beta"]
    X_clean = ds["X_clean"]
    support_mask = ds["support_mask"]

    assert X.shape == (n, p_total), (
        f"X.shape {X.shape} != expected ({n}, {p_total})"
    )
    assert beta.shape == (p_total,), (
        f"beta.shape {beta.shape} != ({p_total},)"
    )
    if p_noise > 0:
        assert np.all(beta[p_signal:] == 0), (
            f"beta[p_signal:] not all zero (p_signal={p_signal}, p_noise={p_noise})"
        )
    assert y.shape == (n,), f"y.shape {y.shape} != ({n},)"
    assert betaX.shape == (n,), f"betaX.shape {betaX.shape} != ({n},)"
    assert support_mask.shape == (p_total,), (
        f"support_mask.shape {support_mask.shape} != ({p_total},)"
    )
    assert np.all(np.isfinite(X)), "X contains non-finite values"
    assert np.all(np.isfinite(y)), "y contains non-finite values"
    assert np.all(np.isfinite(betaX)), "betaX contains non-finite values"
    assert np.allclose(X_clean @ beta, betaX, atol=1e-9), (
        "betaX != X_clean @ beta (max diff: "
        f"{np.max(np.abs(X_clean @ beta - betaX)):.3e})"
    )


# ---------------------------------------------------------------------------
# 11. Teacher Targets
# ---------------------------------------------------------------------------

def _ridge_beta(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression coefficient: (X'X + lam*I)^{-1} X'y."""
    p = X.shape[1]
    A = X.T @ X + lam * np.eye(p)
    b = X.T @ y
    return np.linalg.solve(A, b)


def compute_linear_teacher_targets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambdas: list[float] | None = None,
) -> dict:
    """Fit OLS/min-norm and ridge estimators; return predictions and diagnostics.

    WARNING — ORACLE DIAGNOSTIC: the best ridge lambda is selected by minimising
    MSE on y_test (query labels).  The result is stored as ``oracle_ridge_lambda``
    and ``ridge_oracle_mse``.  These values are leakage-free only when used as
    DGP diagnostic baselines.
    MUST NOT enter model training, HPO, or checkpoint selection.
    """
    if lambdas is None:
        lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]

    results: dict[str, Any] = {}

    # OLS / min-norm
    beta_ols, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    pred_ols = X_test @ beta_ols
    mse_ols = float(np.mean((pred_ols - y_test) ** 2))
    results["ols_or_min_norm_beta"] = beta_ols
    results["ols_or_min_norm_mse"] = mse_ols

    # Ridge for each lambda
    mse_by_lambda: dict[float, float] = {}
    best_lam = lambdas[0]
    best_mse = float("inf")

    for lam in lambdas:
        if lam == 0.0:
            beta_r = beta_ols
        else:
            beta_r = _ridge_beta(X_train, y_train, lam)
        pred_r = X_test @ beta_r
        mse_r = float(np.mean((pred_r - y_test) ** 2))
        results[f"ridge_beta_lambda_{lam}"] = beta_r
        results[f"ridge_pred_lambda_{lam}_test"] = pred_r
        mse_by_lambda[lam] = mse_r
        if mse_r < best_mse:
            best_mse = mse_r
            best_lam = lam

    results["oracle_ridge_lambda"] = float(best_lam)
    results["best_ridge_mse"] = best_mse
    results["ridge_oracle_mse"] = best_mse
    results["ridge_mse_by_lambda"] = mse_by_lambda

    return results


# ---------------------------------------------------------------------------
# 12. Diagnostics
# ---------------------------------------------------------------------------

def compute_linear_diagnostics(
    X: np.ndarray,
    beta: np.ndarray,
    p_signal: int,
) -> dict:
    """Compute matrix and signal diagnostics for a dataset."""
    n, p = X.shape
    try:
        sv = np.linalg.svd(X, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        rank = int(np.sum(sv > sv[0] * max(n, p) * np.finfo(float).eps))
        # Effective rank via Roy-Vetterli entropy
        sv_norm = sv / (sv.sum() + 1e-30)
        eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-30))))
    except np.linalg.LinAlgError:
        cond, rank, eff_rank = float("inf"), 0, 0.0

    betaX = X[:, :p_signal] @ beta[:p_signal]
    snr = float(np.var(betaX))

    return {
        "condition_number": cond,
        "matrix_rank": rank,
        "effective_rank": eff_rank,
        "snr": snr,
        "p_over_n": p / max(1, n),
        "n_over_p": n / max(1, p),
        "sparsity_ratio": float(np.sum(beta[:p_signal] == 0)) / max(1, p_signal),
    }


# ---------------------------------------------------------------------------
# 13. Utilities
# ---------------------------------------------------------------------------

def _compute_sample_complexity_bucket(n: int, p_total: int) -> str:
    ratio = n / max(1, p_total)
    if ratio < 1.0:
        return "underdetermined"
    if ratio < 2.0:
        return "critically_determined"
    if ratio < 5.0:
        return "low_n_high_p"
    if ratio < 20.0:
        return "moderate"
    return "high"


def _add_scalar(
    columns: dict,
    name: str,
    value: Any,
    arrow_type: pa.DataType,
) -> None:
    """Safe insert into PyArrow column dict; no-op if value is None."""
    if value is None:
        return
    columns[name] = pa.array([value], type=arrow_type)


# ---------------------------------------------------------------------------
# 14. Parquet Writers
# ---------------------------------------------------------------------------

def write_parquet_eval(
    ds: dict,
    filepath: str,
    *,
    profile: str,
    store_beta: bool = False,
    store_teacher_preds: bool = False,
    store_linear_moments: bool = False,
    teacher_results: dict | None = None,
    diagnostics: dict | None = None,
    sample_complexity_bucket: str | None = None,
    suite_id: str = "",
    suite_family: str = "primary",
    global_idx: int = -1,
    dataset_seed: int = -1,
    schema_version: str = REGRESSION_EVAL_SCHEMA_VERSION,
    extra_metadata: Mapping[str, Any] | None = None,
) -> int:
    """Write evaluation-suite parquet (generate_synthetic_regression.py format).

    Always writes the 12 locked columns first, then additive metadata.
    Returns file size in bytes.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    teacher_results = teacher_results or {}
    diagnostics = diagnostics or {}

    feature_noise_float = float(ds.get("feature_noise_level", 0.0))

    # --- 12 locked columns (must come first, exact types) ---
    columns: dict[str, pa.Array] = {
        "X":                    pa.array([ds["X"].tolist()],     type=pa.list_(pa.list_(pa.float64()))),
        "y":                    pa.array([ds["y"].tolist()],     type=pa.list_(pa.float64())),
        "betaX":                pa.array([ds["betaX"].tolist()], type=pa.list_(pa.float64())),
        "suite_family":         pa.array([suite_family],          type=pa.utf8()),
        "prior_regime":         pa.array([ds["prior_regime"]],   type=pa.utf8()),
        "n_total":              pa.array([ds["n_total"]],        type=pa.int64()),
        "p_signal":             pa.array([ds["p_signal"]],       type=pa.int64()),
        "p_noise":              pa.array([ds["p_noise"]],        type=pa.int64()),
        "p_total":              pa.array([ds["p_total"]],        type=pa.int64()),
        "target_noise_scale":   pa.array([float(ds["target_noise_scale"])], type=pa.float64()),
        "training_size_anchor": pa.array([False],                type=pa.bool_()),
        "feature_noise_level":  pa.array([feature_noise_float], type=pa.float64()),
    }

    # --- Always-present additive metadata ---
    _add_scalar(columns, "feature_noise_level_float", feature_noise_float, pa.float64())
    _add_scalar(columns, "schema_version", schema_version, pa.utf8())
    _add_scalar(columns, "task_family", "linear_regression", pa.utf8())
    _add_scalar(columns, "suite_id", suite_id, pa.utf8())
    _add_scalar(columns, "global_idx", global_idx, pa.int64())
    _add_scalar(columns, "dataset_seed", dataset_seed, pa.int64())
    _add_scalar(columns, "profile", profile, pa.utf8())
    _add_scalar(columns, "regime_group", ds.get("regime_group", ds["prior_regime"]), pa.utf8())
    _add_scalar(columns, "active_s", ds.get("active_s"), pa.int64())
    _add_scalar(columns, "sparsity_ratio", ds.get("sparsity_ratio"), pa.float64())
    _add_scalar(columns, "covariance_type", ds.get("covariance_type", "iid"), pa.utf8())
    _add_scalar(columns, "rho", ds.get("rho", 0.0), pa.float64())
    _add_scalar(columns, "target_noise_type", ds.get("target_noise_type", "gaussian"), pa.utf8())
    _add_scalar(columns, "has_noise_features", ds.get("has_noise_features", False), pa.bool_())
    _add_scalar(columns, "has_feature_noise", ds.get("has_feature_noise", False), pa.bool_())
    _add_scalar(columns, "sample_complexity_bucket", sample_complexity_bucket, pa.utf8())
    _add_scalar(
        columns,
        "distribution_family",
        ds.get("distribution_family", "gaussian_linear"),
        pa.utf8(),
    )
    if extra_metadata:
        for name, value in extra_metadata.items():
            if isinstance(value, bool):
                _add_scalar(columns, name, value, pa.bool_())
            elif isinstance(value, int):
                _add_scalar(columns, name, value, pa.int64())
            elif isinstance(value, float):
                _add_scalar(columns, name, value, pa.float64())
            elif isinstance(value, str):
                _add_scalar(columns, name, value, pa.utf8())
            elif isinstance(value, (list, tuple)):
                columns[name] = pa.array(
                    [json.dumps(list(value), sort_keys=True)], type=pa.utf8()
                )

    # Diagnostics
    _add_scalar(columns, "condition_number", diagnostics.get("condition_number"), pa.float64())
    _add_scalar(columns, "matrix_rank", diagnostics.get("matrix_rank"), pa.int64())
    _add_scalar(columns, "effective_rank", diagnostics.get("effective_rank"), pa.float64())
    _add_scalar(columns, "snr", diagnostics.get("snr"), pa.float64())
    _add_scalar(columns, "p_over_n", diagnostics.get("p_over_n"), pa.float64())
    _add_scalar(columns, "n_over_p", diagnostics.get("n_over_p"), pa.float64())

    # Teacher targets (always write scalar summary)
    # oracle_ridge_lambda: selected via test labels — diagnostic only
    _add_scalar(columns, "oracle_ridge_lambda", teacher_results.get("oracle_ridge_lambda"), pa.float64())
    _add_scalar(columns, "ridge_oracle_mse", teacher_results.get("ridge_oracle_mse"), pa.float64())
    _add_scalar(columns, "ols_or_min_norm_mse", teacher_results.get("ols_or_min_norm_mse"), pa.float64())
    teacher_available = len(teacher_results) > 0
    _add_scalar(columns, "teacher_available", teacher_available, pa.bool_())

    # --- Conditional: beta arrays ---
    if store_beta:
        beta = ds.get("beta")
        if beta is not None:
            columns["beta"] = pa.array([beta.tolist()], type=pa.list_(pa.float64()))
        support = ds.get("support_mask")
        if support is not None:
            columns["support_mask"] = pa.array([support.tolist()], type=pa.list_(pa.bool_()))
        # Split into signal / noise parts for convenience
        p_sig = ds["p_signal"]
        if beta is not None:
            columns["beta_signal"] = pa.array([beta[:p_sig].tolist()], type=pa.list_(pa.float64()))
            columns["beta_noise"] = pa.array([beta[p_sig:].tolist()], type=pa.list_(pa.float64()))

    # --- Conditional: ridge predictions ---
    if store_teacher_preds and teacher_results:
        lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
        for lam in lambdas:
            key = f"ridge_pred_lambda_{lam}_test"
            pred = teacher_results.get(key)
            if pred is not None:
                columns[key] = pa.array([pred.tolist()], type=pa.list_(pa.float64()))

    # --- Conditional: linear sufficient statistics ---
    if store_linear_moments:
        X = ds["X"]
        y = ds["y"]
        Sxx = (X.T @ X).flatten()
        Sxy = X.T @ y
        columns["Sxx"] = pa.array([Sxx.tolist()], type=pa.list_(pa.float64()))
        columns["Sxy"] = pa.array([Sxy.tolist()], type=pa.list_(pa.float64()))

    table = pa.table(columns)
    pq.write_table(table, filepath)
    return int(Path(filepath).stat().st_size)


def write_parquet_dgp(
    ds_split: dict,
    filepath: str,
    *,
    store_beta: bool = False,
    store_teacher_preds: bool = False,
    teacher_results: dict | None = None,
    diagnostics: dict | None = None,
) -> None:
    """Write training-pipeline parquet (generate_dgp.py format).

    Always writes 9 locked DGP columns + additive metadata.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    teacher_results = teacher_results or {}
    diagnostics = diagnostics or {}

    feature_noise_float = float(ds_split.get("feature_noise_level", 0.0))

    # --- 9 locked DGP columns ---
    columns: dict[str, pa.Array] = {
        "X_train":      pa.array([ds_split["X_train"].tolist()],    type=pa.list_(pa.list_(pa.float64()))),
        "y_train":      pa.array([ds_split["y_train"].tolist()],    type=pa.list_(pa.float64())),
        "X_test":       pa.array([ds_split["X_test"].tolist()],     type=pa.list_(pa.list_(pa.float64()))),
        "betaX_test":   pa.array([ds_split["betaX_test"].tolist()], type=pa.list_(pa.float64())),
        "n":            pa.array([int(ds_split["n"])],              type=pa.int64()),
        "p":            pa.array([int(ds_split["p"])],              type=pa.int64()),
        "n_train":      pa.array([int(ds_split["n_train"])],        type=pa.int64()),
        "n_test":       pa.array([int(ds_split["n_test"])],         type=pa.int64()),
        "prior_regime": pa.array([ds_split["prior_regime"]],        type=pa.utf8()),
    }

    # --- Always-present additive columns ---
    y_test = ds_split.get("y_test")
    if y_test is not None:
        columns["y_test"] = pa.array([y_test.tolist()], type=pa.list_(pa.float64()))
    betaX_train = ds_split.get("betaX_train")
    if betaX_train is not None:
        columns["betaX_train"] = pa.array([betaX_train.tolist()], type=pa.list_(pa.float64()))

    _add_scalar(columns, "p_signal", ds_split.get("p_signal"), pa.int64())
    _add_scalar(columns, "p_noise", ds_split.get("p_noise"), pa.int64())
    _add_scalar(columns, "p_total", ds_split.get("p_total"), pa.int64())
    _add_scalar(columns, "active_s", ds_split.get("active_s"), pa.int64())
    _add_scalar(columns, "sparsity_ratio", ds_split.get("sparsity_ratio"), pa.float64())
    _add_scalar(columns, "covariance_type", ds_split.get("covariance_type", "iid"), pa.utf8())
    _add_scalar(columns, "rho", ds_split.get("rho", 0.0), pa.float64())
    _add_scalar(columns, "target_noise_scale", ds_split.get("target_noise_scale", 1.0), pa.float64())
    _add_scalar(columns, "feature_noise_level", feature_noise_float, pa.float64())
    _add_scalar(columns, "feature_noise_level_float", feature_noise_float, pa.float64())

    # Minimum teacher signal — always written when available
    teacher_available = len(teacher_results) > 0
    _add_scalar(columns, "teacher_available", teacher_available, pa.bool_())
    # oracle_ridge_lambda: selected via test labels — diagnostic only
    _add_scalar(columns, "oracle_ridge_lambda", teacher_results.get("oracle_ridge_lambda"), pa.float64())

    ridge_pred_1 = teacher_results.get("ridge_pred_lambda_1.0_test")
    if ridge_pred_1 is not None:
        columns["ridge_pred_lambda_1.0_test"] = pa.array(
            [ridge_pred_1.tolist()], type=pa.list_(pa.float64())
        )

    # Diagnostics
    _add_scalar(columns, "condition_number", diagnostics.get("condition_number"), pa.float64())
    _add_scalar(columns, "matrix_rank", diagnostics.get("matrix_rank"), pa.int64())
    _add_scalar(columns, "effective_rank", diagnostics.get("effective_rank"), pa.float64())
    _add_scalar(columns, "snr", diagnostics.get("snr"), pa.float64())

    # --- Conditional: beta arrays ---
    if store_beta:
        beta = ds_split.get("beta")
        if beta is not None:
            columns["beta"] = pa.array([beta.tolist()], type=pa.list_(pa.float64()))
        support = ds_split.get("support_mask")
        if support is not None:
            columns["support_mask"] = pa.array([support.tolist()], type=pa.list_(pa.bool_()))

    # --- Conditional: all ridge predictions ---
    if store_teacher_preds and teacher_results:
        lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
        for lam in lambdas:
            key = f"ridge_pred_lambda_{lam}_test"
            pred = teacher_results.get(key)
            if pred is not None and key not in columns:
                columns[key] = pa.array([pred.tolist()], type=pa.list_(pa.float64()))

    table = pa.table(columns)
    pq.write_table(table, filepath)


# ---------------------------------------------------------------------------
# 15. Linear Classification
# ---------------------------------------------------------------------------

def largest_remainder_counts(
    weights: dict[Any, float],
    total: int,
) -> dict[Any, int]:
    """Convert non-negative weights to exact integer counts deterministically."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("weights must have positive total mass")
    keys = list(weights)
    normalized = np.asarray([float(weights[k]) for k in keys], dtype=float)
    normalized /= normalized.sum()
    exact = normalized * total
    counts = np.floor(exact).astype(int)
    remainder = total - int(counts.sum())
    order = sorted(range(len(keys)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    return {key: int(counts[i]) for i, key in enumerate(keys)}


def _move_quota(
    counts: dict[int, int],
    destination: int,
    amount: int,
    sources: list[int],
) -> None:
    for source in sorted(sources, key=lambda k: (-counts.get(k, 0), k)):
        moved = min(amount, counts.get(source, 0))
        counts[source] -= moved
        counts[destination] = counts.get(destination, 0) + moved
        amount -= moved
        if amount == 0:
            return
    if amount:
        raise ValueError("classification class-count constraints are infeasible")


def allocate_classification_tasks(
    n_datasets: int,
    profile: str,
    base_seed: int,
    *,
    allow_underdetermined: bool = False,
    num_classes: list[int] | None = None,
    imbalance_levels: list[str] | None = None,
    margin_levels: list[str] | None = None,
    label_noise_values: list[float] | None = None,
    allocation_mode: str = "weighted_quota",
) -> tuple[list[dict[str, Any]], dict[str, dict[Any, int]]]:
    """Allocate exact regime, class-count, and imbalance quotas."""
    if profile not in CLASSIFICATION_PROFILES:
        raise ValueError(f"Not a classification profile: {profile!r}")
    weights = dict(REGIME_WEIGHTS[profile])
    if not allow_underdetermined:
        weights.pop("J_low_n_high_p_classification", None)
    if allocation_mode not in ALLOCATION_MODES:
        raise ValueError(f"unknown allocation mode {allocation_mode!r}")
    if allocation_mode in {"balanced", "balanced_cartesian"}:
        weights = {regime: 1.0 for regime in weights}
    if allocation_mode == "weighted":
        allocation_rng = np.random.default_rng(base_seed)
        regime_names = list(weights)
        probabilities = np.asarray([weights[name] for name in regime_names], dtype=float)
        probabilities /= probabilities.sum()
        regime_counts = dict(
            zip(
                regime_names,
                np.bincount(
                    [
                        regime_names.index(
                            str(allocation_rng.choice(regime_names, p=probabilities))
                        )
                        for _ in range(n_datasets)
                    ],
                    minlength=len(regime_names),
                ).tolist(),
            )
        )
    else:
        regime_counts = largest_remainder_counts(weights, n_datasets)
    regimes = [
        regime
        for regime, count in regime_counts.items()
        for _ in range(count)
    ]

    num_classes = num_classes or list(CLASS_COUNT_WEIGHTS)
    unknown_classes = set(num_classes) - set(CLASS_COUNT_WEIGHTS)
    if unknown_classes:
        raise ValueError(f"unsupported class counts: {sorted(unknown_classes)}")
    class_count_weights = {
        k: CLASS_COUNT_WEIGHTS[k] for k in num_classes
    }
    k_counts = largest_remainder_counts(class_count_weights, n_datasets)
    nonbinary_values = [k for k in num_classes if k > 2]
    forced_binary = sum(
        regime_counts.get(regime, 0)
        for regime in (
            "A_iid_dense_logistic",
            "B_iid_sparse_logistic",
            "D_correlated_ar_logistic",
        )
    )
    forced_multiclass = sum(
        regime_counts.get(regime, 0)
        for regime in ("E_high_dim_dense_softmax", "F_high_dim_sparse_softmax")
    )
    if forced_binary and 2 not in k_counts:
        raise ValueError("A, B, and D regimes require K=2 in --num_classes_grid")
    if forced_multiclass and not nonbinary_values:
        raise ValueError("E and F regimes require a multiclass K value")
    if k_counts.get(2, 0) < forced_binary:
        _move_quota(
            k_counts, 2, forced_binary - k_counts.get(2, 0), nonbinary_values
        )
    nonbinary = n_datasets - k_counts.get(2, 0)
    if nonbinary < forced_multiclass:
        _move_quota(
            k_counts,
            nonbinary_values[0],
            forced_multiclass - nonbinary,
            [2],
        )

    remaining_k = dict(k_counts)
    class_counts: list[int] = []
    forced_binary_regimes = {
        "A_iid_dense_logistic",
        "B_iid_sparse_logistic",
        "D_correlated_ar_logistic",
    }
    forced_multiclass_regimes = {
        "E_high_dim_dense_softmax",
        "F_high_dim_sparse_softmax",
    }
    for regime_index, regime in enumerate(regimes):
        future_regimes = regimes[regime_index + 1:]
        future_binary = sum(r in forced_binary_regimes for r in future_regimes)
        future_multiclass = sum(r in forced_multiclass_regimes for r in future_regimes)
        if regime in {
            "A_iid_dense_logistic",
            "B_iid_sparse_logistic",
            "D_correlated_ar_logistic",
        }:
            k = 2
        elif regime in {
            "E_high_dim_dense_softmax",
            "F_high_dim_sparse_softmax",
        }:
            candidates = list(nonbinary_values)
            k = max(candidates, key=lambda value: (remaining_k[value], -value))
        else:
            candidates = list(num_classes)
            candidates = [
                value for value in candidates
                if (
                    value != 2
                    or remaining_k[value] > future_binary
                )
                and (
                    value == 2
                    or sum(remaining_k[x] for x in nonbinary_values)
                    > future_multiclass
                )
            ]
            k = max(candidates, key=lambda value: (remaining_k[value], -value))
        if remaining_k[k] <= 0:
            candidates = [
                value for value in candidates if remaining_k[value] > 0
            ]
            if not candidates:
                raise ValueError("classification K allocation exhausted early")
            k = max(candidates, key=lambda value: (remaining_k[value], -value))
        remaining_k[k] -= 1
        class_counts.append(k)
    if any(remaining_k.values()):
        raise AssertionError(f"unconsumed class-count quotas: {remaining_k}")

    imbalance_levels = imbalance_levels or list(IMBALANCE_WEIGHTS)
    imbalance_weights = {
        level: IMBALANCE_WEIGHTS[level] for level in imbalance_levels
    }
    imbalance_counts = largest_remainder_counts(imbalance_weights, n_datasets)
    remaining_imbalance = dict(imbalance_counts)
    assigned_imbalance: list[str] = []
    for regime in regimes:
        allowed = list(imbalance_weights)
        if regime in {"A_iid_dense_logistic", "C_label_noise_margin"}:
            allowed = [
                level for level in ("balanced", "mild")
                if level in imbalance_weights
            ]
        candidates = [name for name in allowed if remaining_imbalance[name] > 0]
        if not candidates:
            candidates = [
                name for name in imbalance_weights
                if remaining_imbalance[name] > 0
            ]
        level = max(
            candidates,
            key=lambda name: (
                remaining_imbalance[name],
                -list(imbalance_weights).index(name),
            ),
        )
        remaining_imbalance[level] -= 1
        assigned_imbalance.append(level)
    if any(remaining_imbalance.values()):
        raise AssertionError(
            f"unconsumed imbalance quotas: {remaining_imbalance}"
        )

    margin_levels = margin_levels or ["low", "medium", "high"]
    label_noise_values = label_noise_values or [0.0, 0.02, 0.05, 0.10]
    assignments: list[dict[str, Any]] = []
    d_count = 0
    for i, (regime, k, imbalance) in enumerate(
        zip(regimes, class_counts, assigned_imbalance)
    ):
        margin = margin_levels[i % len(margin_levels)]
        if regime == "C_label_noise_margin":
            margin = "low"
        elif imbalance in {"moderate", "severe"} and margin == "low":
            margin = "medium"
        label_noise = float(label_noise_values[i % len(label_noise_values)])
        if regime in {"A_iid_dense_logistic", "B_iid_sparse_logistic"}:
            label_noise = 0.0
        elif regime == "C_label_noise_margin":
            allowed_noise = [
                value for value in label_noise_values
                if math.isclose(value, 0.05) or math.isclose(value, 0.10)
            ]
            label_noise = float(allowed_noise[i % len(allowed_noise)] if allowed_noise else 0.05)
        coefficient_regime = None
        if regime == "D_correlated_ar_logistic":
            coefficient_regime = "dense" if d_count % 2 == 0 else "sparse"
            d_count += 1
        assignments.append({
            "classification_regime": regime,
            "num_classes": k,
            "class_imbalance_type": imbalance,
            "margin_level": margin,
            "label_noise_rate": label_noise,
            "coefficient_regime": coefficient_regime,
        })

    rng = np.random.default_rng(base_seed ^ 0xC1A551F1)
    order = rng.permutation(n_datasets)
    shuffled = [assignments[int(i)] for i in order]
    realized_k = {
        k: sum(a["num_classes"] == k for a in shuffled)
        for k in class_count_weights
    }
    realized_imbalance = {
        level: sum(a["class_imbalance_type"] == level for a in shuffled)
        for level in imbalance_weights
    }
    return shuffled, {
        "regime_counts": regime_counts,
        "class_count_targets": k_counts,
        "realized_class_counts": realized_k,
        "imbalance_targets": imbalance_counts,
        "realized_imbalance_counts": realized_imbalance,
    }


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _sample_categorical(
    rng: np.random.Generator,
    probs: np.ndarray,
) -> np.ndarray:
    draws = rng.random(probs.shape[0])
    return np.sum(draws[:, None] > np.cumsum(probs, axis=1), axis=1).astype(np.int64)


def apply_symmetric_label_noise(
    rng: np.random.Generator,
    labels: np.ndarray,
    num_classes: int,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt exactly round(rate * n) labels, always to a different class."""
    observed = np.asarray(labels, dtype=np.int64).copy()
    mask = np.zeros(observed.shape[0], dtype=bool)
    n_corrupt = int(round(float(rate) * observed.shape[0]))
    if n_corrupt:
        indices = rng.choice(observed.shape[0], size=n_corrupt, replace=False)
        offsets = rng.integers(1, num_classes, size=n_corrupt)
        observed[indices] = (observed[indices] + offsets) % num_classes
        mask[indices] = True
    return observed, mask


def _target_class_prior(
    rng: np.random.Generator,
    num_classes: int,
    imbalance_type: str,
) -> tuple[np.ndarray, int]:
    strength = IMBALANCE_STRENGTHS[imbalance_type]
    majority = int(rng.integers(0, num_classes))
    prior = np.full(num_classes, (1.0 - strength) / num_classes)
    prior[majority] += strength
    return prior, majority


def _calibrate_intercepts(
    base_logits: np.ndarray,
    target_prior: np.ndarray,
    temperature: float,
) -> np.ndarray:
    num_classes = base_logits.shape[1]
    if num_classes == 2:
        z = base_logits[:, 1]
        target = float(target_prior[1])
        lo, hi = -100.0 * temperature, 100.0 * temperature
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            mean = float(np.mean(1.0 / (1.0 + np.exp(-(z + mid) / temperature))))
            if mean < target:
                lo = mid
            else:
                hi = mid
        return np.asarray([0.0, 0.5 * (lo + hi)], dtype=np.float64)

    intercepts = np.zeros(num_classes, dtype=np.float64)
    for _ in range(250):
        probs = _softmax(base_logits + intercepts, temperature)
        mean_prior = probs.mean(axis=0)
        error = float(np.max(np.abs(mean_prior - target_prior)))
        if error <= 1e-8:
            break
        intercepts += temperature * np.log(
            np.clip(target_prior, 1e-12, None)
            / np.clip(mean_prior, 1e-12, None)
        )
        intercepts -= intercepts[0]
    return intercepts


def _normalized_margins(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    top_two = np.partition(logits, -2, axis=1)[:, -2:]
    return (top_two[:, 1] - top_two[:, 0]) / float(temperature)


def _margin_bucket(value: float) -> str:
    if value < 0.75:
        return "low"
    if value <= 2.0:
        return "medium"
    return "high"


def _classification_coefficients(
    rng: np.random.Generator,
    p_signal: int,
    p_total: int,
    num_classes: int,
    pattern: str,
    active_s: int,
    coefficient_scale: float,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.zeros((p_total, num_classes), dtype=np.float64)
    support = np.zeros((p_total, num_classes), dtype=bool)
    for class_id in range(1, num_classes):
        if pattern == "dense":
            indices = np.arange(p_signal)
            values = rng.normal(size=p_signal) / math.sqrt(max(1, p_signal))
        elif pattern == "sparse":
            count = max(1, min(active_s, p_signal))
            indices = rng.choice(p_signal, size=count, replace=False)
            values = rng.normal(size=count) / math.sqrt(count)
        elif pattern == "decaying":
            indices = np.arange(p_signal)
            values = rng.normal(size=p_signal)
            values *= np.exp(-np.arange(p_signal) / max(1, active_s))
            values /= np.linalg.norm(values) + 1e-12
        elif pattern == "group_sparse":
            n_blocks = max(1, math.ceil(p_signal / block_size))
            chosen = rng.choice(n_blocks, size=max(1, min(2, n_blocks)), replace=False)
            indices = np.concatenate([
                np.arange(start * block_size, min((start + 1) * block_size, p_signal))
                for start in np.sort(chosen)
            ])
            values = rng.normal(size=indices.size) / math.sqrt(indices.size)
        elif pattern == "market_sign":
            quarter = max(1, p_signal // 4)
            indices = np.arange(min(p_signal, 3 * quarter))
            values = rng.normal(size=indices.size) / math.sqrt(max(1, indices.size))
            values[:quarter] = -np.abs(values[:quarter])
            cross_end = min(indices.size, 2 * quarter)
            cross_signs = np.where(
                np.arange(cross_end - quarter) % 2 == 0, 1.0, -1.0
            )
            values[quarter:cross_end] = (
                np.abs(values[quarter:cross_end]) * cross_signs
            )
        else:
            raise ValueError(f"Unknown classification coefficient pattern: {pattern}")
        W[indices, class_id] = coefficient_scale * values
        support[indices, class_id] = True
    return W, support


def _classification_shape_params(
    rng: np.random.Generator,
    regime: str,
    assignment: dict[str, Any],
    grids: dict[str, list[Any]],
) -> dict[str, Any]:
    num_classes = int(assignment["num_classes"])
    n = int(rng.choice(grids["n_grid"]))
    p_signal = int(rng.choice(grids["p_signal_grid"]))
    p_noise = 0
    active_s = int(np.clip(
        int(rng.choice(grids["active_s_grid"])), 1, p_signal
    ))
    rho = 0.0
    block_size = 0
    cov_type = "iid"
    feature_noise = 0.0
    pattern = assignment.get("coefficient_regime") or "dense"

    if regime in CLASSIFICATION_EVAL_ONLY_REGIMES:
        if regime == "Y_exact_zero_irrelevant_classification":
            p_noise = max(8, int(rng.choice(grids["p_noise_grid"] or [8])))
        if regime in {
            "AA_sparse_covariate_shift_classification",
            "AB_high_dimensional_linear_classification",
        }:
            pattern = "sparse"
        if regime == "AB_high_dimensional_linear_classification":
            p_signal = max(64, p_signal)
            n = max(n, 20 * num_classes)
        if regime == "X_negative_ar1_covariance_classification":
            rho = -0.65
            cov_type = "ar1"

    if regime == "B_iid_sparse_logistic":
        pattern = "sparse"
    elif regime == "C_label_noise_margin":
        pattern = str(rng.choice(["dense", "sparse"]))
    elif regime == "D_correlated_ar_logistic":
        rho_choices = [x for x in grids["rho_grid"] if float(x) != 0.0]
        rho = float(rng.choice(rho_choices or [0.6]))
        cov_type = "ar1"
    elif regime == "E_high_dim_dense_softmax":
        p_signal = max(32, p_signal)
        pattern = "dense"
        n = max(n, 2 * p_signal)
    elif regime == "F_high_dim_sparse_softmax":
        p_signal = max(32, p_signal)
        active_s = min(active_s, max(1, p_signal // 4))
        pattern = "sparse"
        n = max(n, 2 * p_signal)
    elif regime == "G_noise_features_classification":
        p_noise_choices = [
            int(x) for x in grids["p_noise_grid"] if 8 <= int(x) <= 120
        ]
        p_noise = int(rng.choice(p_noise_choices or [8]))
        pattern = str(rng.choice(["dense", "sparse"]))
    elif regime == "H_block_correlated_classification":
        block_size = int(rng.choice([4, 8, 16]))
        cov_type = "block"
        rho = 0.7
        pattern = "group_sparse"
    elif regime == "I_equicorrelated_classification":
        rho_choices = [x for x in grids["rho_grid"] if float(x) != 0.0]
        rho = float(rng.choice(rho_choices or [0.3]))
        cov_type = "equicorrelation"
        pattern = str(rng.choice(["dense", "sparse"]))
    elif regime == "J_low_n_high_p_classification":
        pattern = "sparse"
        p_noise = max(0, int(rng.choice(grids["p_noise_grid"])))
        n = max(2 * num_classes, min(n, 64))
        if p_signal + p_noise <= n:
            p_noise = n - p_signal + 1
    elif regime == "K_feature_noise_classification":
        nonzero = [
            float(x) for x in grids["feature_noise_grid"] if float(x) > 0.0
        ]
        feature_noise = float(rng.choice(nonzero or [0.1]))
        pattern = str(rng.choice(["dense", "sparse"]))
    elif regime == "L_market_sign_classification":
        pattern = "market_sign"
        p_noise = max(0, min(120, int(rng.choice(grids["p_noise_grid"]))))

    p_total = p_signal + p_noise
    if regime != "J_low_n_high_p_classification":
        n = max(n, p_total, 20 * num_classes)
    return {
        "n": n,
        "p_signal": p_signal,
        "p_noise": p_noise,
        "p_total": p_total,
        "active_s": min(active_s, p_signal),
        "rho": rho,
        "block_size": block_size,
        "covariance_type": cov_type,
        "feature_noise_level": feature_noise,
        "coefficient_regime": pattern,
    }


def generate_classification_dataset(
    rng: np.random.Generator,
    regime: str,
    assignment: dict[str, Any],
    grids: dict[str, list[Any]],
    *,
    task_seed: int,
    max_attempts: int = 64,
) -> dict[str, Any]:
    """Generate one validated linear classification task."""
    if regime not in CLASSIFICATION_REGIMES and regime not in CLASSIFICATION_EVAL_ONLY_REGIMES:
        raise ValueError(f"Unknown classification regime: {regime!r}")
    sampled: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        sampled = _classification_shape_params(rng, regime, assignment, grids)
        num_classes = int(assignment["num_classes"])
        n = sampled["n"]
        p_signal = sampled["p_signal"]
        p_noise = sampled["p_noise"]
        p_total = sampled["p_total"]
        block_size = sampled["block_size"]
        temperature_choices = list(grids["temperature_grid"])
        if regime == "C_label_noise_margin":
            temperature_choices = [
                x for x in temperature_choices if float(x) in {2.0, 4.0}
            ] or [2.0, 4.0]
        if regime == "Z_heteroskedastic_margin_classification":
            # Temperature is constant within a task, so logits remain explicitly
            # linear while task-level margin/noise geometry varies.
            temperature_choices = [0.5, 1.0, 2.0, 4.0]
        temperature = float(rng.choice(temperature_choices))
        coefficient_scale = float(rng.choice(grids["coefficient_scale_grid"]))
        intercept_scale = float(rng.choice(grids["intercept_scale_grid"]))

        if regime in CLASSIFICATION_EVAL_ONLY_REGIMES:
            X_signal, generated_covariance, generated_rho = generate_eval_only_features(
                rng, regime, n, p_signal
            )
            sampled["covariance_type"] = generated_covariance
            sampled["rho"] = generated_rho
        else:
            X_signal = generate_X(
                rng,
                n,
                p_signal,
                sampled["covariance_type"],
                sampled["rho"],
                block_size=max(1, block_size),
            )
        X_clean = add_noise_features(rng, X_signal, p_noise, "iid", 0.0)
        W, class_support = _classification_coefficients(
            rng,
            p_signal,
            p_total,
            num_classes,
            sampled["coefficient_regime"],
            sampled["active_s"],
            coefficient_scale,
            max(1, block_size),
        )

        target_prior, majority_class = _target_class_prior(
            rng, num_classes, assignment["class_imbalance_type"]
        )
        requested_margin = assignment["margin_level"]
        target_margin = {"low": 0.45, "medium": 1.25, "high": 3.0}[requested_margin]
        initial_intercepts = np.zeros(num_classes, dtype=np.float64)
        if intercept_scale:
            initial_intercepts = rng.normal(
                0.0, intercept_scale, num_classes
            )
            initial_intercepts -= initial_intercepts[0]
        intercepts = np.zeros(num_classes)
        for _ in range(8):
            raw_logits = X_clean @ W + initial_intercepts
            correction = _calibrate_intercepts(
                raw_logits, target_prior, temperature
            )
            intercepts = initial_intercepts + correction
            logits = X_clean @ W + intercepts
            median_margin = float(np.median(
                _normalized_margins(logits, temperature)
            ))
            if median_margin <= 1e-10:
                W *= 2.0
            else:
                adjustment = float(np.clip(target_margin / median_margin, 0.25, 4.0))
                W *= adjustment
            if _margin_bucket(median_margin) == requested_margin:
                break
        raw_logits = X_clean @ W + initial_intercepts
        correction = _calibrate_intercepts(raw_logits, target_prior, temperature)
        intercepts = initial_intercepts + correction
        logits = X_clean @ W + intercepts
        logits[:, 0] = 0.0
        intercepts -= intercepts[0]
        probs = _softmax(logits, temperature)
        normalized_margins = _normalized_margins(logits, temperature)
        realized_margin = _margin_bucket(float(np.median(normalized_margins)))
        prior_error = float(np.max(np.abs(probs.mean(axis=0) - target_prior)))
        if realized_margin != requested_margin or prior_error > 0.03:
            continue

        y_clean = _sample_categorical(rng, probs)
        if np.unique(y_clean).size != num_classes:
            continue
        y, label_noise_mask = apply_symmetric_label_noise(
            rng,
            y_clean,
            num_classes,
            float(assignment["label_noise_rate"]),
        )
        n_train = int(0.8 * n)
        if np.unique(y[:n_train]).size < 2:
            continue
        X = add_feature_noise(rng, X_clean, sampled["feature_noise_level"])
        active_support = np.any(class_support[:, 1:], axis=1)
        class_active_counts = class_support[:, 1:].sum(axis=0)
        active_s = int(np.max(class_active_counts)) if class_active_counts.size else 0
        class_sparsity_ratio = float(
            1.0 - class_support[:p_signal, 1:].sum()
            / max(1, p_signal * (num_classes - 1))
        )
        observed_counts = np.bincount(y, minlength=num_classes).astype(np.int64)
        clean_counts = np.bincount(y_clean, minlength=num_classes).astype(np.int64)
        return {
            "X": X.astype(np.float64),
            "X_clean": X_clean.astype(np.float64),
            "y": y.astype(np.int64),
            "y_clean": y_clean.astype(np.int64),
            "label_noise_mask": label_noise_mask,
            "logits": logits.astype(np.float64),
            "probs": probs.astype(np.float64),
            "W_true": W.copy(),           # always (p, K); column 0 = reference class (zeros)
            "b_true": intercepts.copy(),  # always (K,); b[0] = 0.0 after normalization
            # Legacy alias kept for backward compat with old readers (deprecated):
            "w_true": W[:, 1].copy(),     # deprecated; equals W_true[:, 1] for K=2
            "active_support": active_support,
            "class_active_support": class_support,
            "n_total": n,
            "p_signal": p_signal,
            "p_noise": p_noise,
            "p_total": p_total,
            "num_classes": num_classes,
            "realized_num_classes": int(np.unique(y).size),
            "classification_regime": regime,
            "prior_regime": regime,
            "coefficient_regime": sampled["coefficient_regime"],
            "active_s": active_s,
            "sparsity_ratio": float(1.0 - active_support[:p_signal].mean()),
            "class_sparsity_ratio": class_sparsity_ratio,
            "covariance_type": sampled["covariance_type"],
            "rho": float(sampled["rho"]),
            "block_size": int(block_size),
            "temperature": temperature,
            "label_noise_rate": float(assignment["label_noise_rate"]),
            "realized_label_noise_rate": float(label_noise_mask.mean()),
            "class_imbalance_type": assignment["class_imbalance_type"],
            "majority_class": majority_class,
            "class_prior": target_prior.astype(np.float64),
            "realized_class_prior": (observed_counts / n).astype(np.float64),
            "clean_class_prior": (clean_counts / n).astype(np.float64),
            "class_prior_error": prior_error,
            "margin_level": requested_margin,
            "realized_margin_level": realized_margin,
            "coefficient_scale": coefficient_scale,
            "intercept_scale": intercept_scale,
            "feature_noise_level": float(sampled["feature_noise_level"]),
            "distribution_family": CLASSIFICATION_REGIME_REGISTRY.get(
                regime, {}
            ).get("distribution_family", "gaussian_linear_logits"),
            "attempt": attempt,
            "task_seed": int(task_seed),
        }
    raise RuntimeError(
        "classification generation failed after "
        f"{max_attempts} attempts: regime={regime}, seed={task_seed}, "
        f"sampled={sampled}, assignment={assignment}"
    )


def split_classification_dataset(ds: dict[str, Any]) -> dict[str, Any]:
    """Create the deterministic 80/20 context/query split."""
    n = int(ds["n_total"])
    n_train = int(0.8 * n)
    split = dict(ds)
    split.update({
        "X_train": ds["X"][:n_train],
        "y_train": ds["y"][:n_train],
        "X_test": ds["X"][n_train:],
        "y_test": ds["y"][n_train:],
        "y_clean_train": ds["y_clean"][:n_train],
        "y_clean_test": ds["y_clean"][n_train:],
        "label_noise_mask_train": ds["label_noise_mask"][:n_train],
        "label_noise_mask_test": ds["label_noise_mask"][n_train:],
        "logits_train": ds["logits"][:n_train],
        "logits_test": ds["logits"][n_train:],
        "probs_train": ds["probs"][:n_train],
        "probs_test": ds["probs"][n_train:],
        "n": n,
        "p": int(ds["p_total"]),
        "n_train": n_train,
        "n_test": n - n_train,
        "train_class_counts": np.bincount(
            ds["y"][:n_train], minlength=ds["num_classes"]
        ).astype(np.int64),
        "test_class_counts": np.bincount(
            ds["y"][n_train:], minlength=ds["num_classes"]
        ).astype(np.int64),
    })
    return split


def compute_classification_diagnostics(ds: dict[str, Any]) -> dict[str, Any]:
    X = ds["X"]
    n, p = X.shape
    singular_values = np.linalg.svd(X, compute_uv=False)
    condition_number = float(
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > 0 else np.finfo(float).max
    )
    threshold = singular_values[0] * max(n, p) * np.finfo(float).eps
    matrix_rank = int(np.sum(singular_values > threshold))
    weights = singular_values / (singular_values.sum() + 1e-30)
    effective_rank = float(np.exp(-np.sum(weights * np.log(weights + 1e-30))))
    raw_margins = _normalized_margins(ds["logits"], 1.0)
    normalized = _normalized_margins(ds["logits"], ds["temperature"])
    counts = np.bincount(ds["y"], minlength=ds["num_classes"])
    fractions = counts / counts.sum()
    entropy = float(-np.sum(
        fractions[fractions > 0] * np.log(fractions[fractions > 0])
    ))
    return {
        "condition_number": condition_number,
        "matrix_rank": matrix_rank,
        "effective_rank": effective_rank,
        "mean_margin": float(raw_margins.mean()),
        "median_margin": float(np.median(raw_margins)),
        "min_margin": float(raw_margins.min()),
        "mean_normalized_margin": float(normalized.mean()),
        "median_normalized_margin": float(np.median(normalized)),
        "min_normalized_margin": float(normalized.min()),
        "class_entropy": entropy,
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "minority_class_fraction": float(fractions.min()),
        "majority_class_fraction": float(fractions.max()),
        "bayes_error_proxy": float(np.mean(1.0 - ds["probs"].max(axis=1))),
    }


def validate_classification_dataset(ds: dict[str, Any]) -> None:
    """Validate the in-memory classification contract and regime invariants."""
    n, p, k = ds["n_total"], ds["p_total"], ds["num_classes"]
    assert ds["X"].shape == (n, p)
    assert ds["X_clean"].shape == (n, p)
    assert ds["y"].shape == (n,) and ds["y"].dtype == np.int64
    assert ds["y_clean"].shape == (n,) and ds["y_clean"].dtype == np.int64
    assert ds["label_noise_mask"].shape == (n,)
    assert ds["label_noise_mask"].dtype == bool
    assert ds["active_support"].shape == (p,)
    assert ds["class_active_support"].shape == (p, k)
    assert not ds["class_active_support"][:, 0].any()
    assert ds["logits"].shape == (n, k)
    assert ds["probs"].shape == (n, k)
    assert ds["p_signal"] + ds["p_noise"] == p
    assert np.all((ds["y"] >= 0) & (ds["y"] < k))
    assert np.all(np.isfinite(ds["X"]))
    assert np.all(np.isfinite(ds["logits"]))
    assert np.all(np.isfinite(ds["probs"]))
    assert np.all((ds["probs"] >= 0.0) & (ds["probs"] <= 1.0))
    np.testing.assert_allclose(ds["probs"].sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        ds["probs"], _softmax(ds["logits"], ds["temperature"]), atol=1e-10
    )
    assert np.all(ds["logits"][:, 0] == 0.0)
    # Unified canonical check: W_true always (p, K) for all K
    assert ds["W_true"].shape == (p, k)
    assert ds["b_true"].shape == (k,)
    assert np.all(ds["W_true"][:, 0] == 0.0)
    assert ds["b_true"][0] == 0.0
    np.testing.assert_allclose(
        ds["logits"], ds["X_clean"] @ ds["W_true"] + ds["b_true"], atol=1e-9
    )
    # Deprecated w_true alias must equal W_true[:, 1]
    np.testing.assert_array_equal(ds["w_true"], ds["W_true"][:, 1])
    coefficients = ds["W_true"][:, 1:]
    coefficient_support = ds["class_active_support"][:, 1:]
    assert np.array_equal(coefficient_support, coefficients != 0.0)
    assert np.array_equal(
        ds["active_support"], np.any(coefficient_support, axis=1)
    )
    if ds["p_noise"]:
        assert np.all(coefficients[ds["p_signal"]:] == 0.0)
        assert not coefficient_support[ds["p_signal"]:].any()
    changed = ds["y"] != ds["y_clean"]
    assert np.array_equal(changed, ds["label_noise_mask"])
    assert int(changed.sum()) == int(round(ds["label_noise_rate"] * n))
    assert math.isclose(ds["realized_label_noise_rate"], changed.mean())
    assert np.unique(ds["y_clean"]).size == k
    np.testing.assert_allclose(
        ds["realized_class_prior"],
        np.bincount(ds["y"], minlength=k) / n,
        atol=0.0,
    )
    assert _margin_bucket(
        float(np.median(_normalized_margins(ds["logits"], ds["temperature"])))
    ) == ds["margin_level"]
    assert ds["class_prior_error"] <= 0.03
    if ds["classification_regime"] == "J_low_n_high_p_classification":
        assert n < p
    if ds["classification_regime"] == "K_feature_noise_classification":
        assert ds["feature_noise_level"] > 0.0
        assert not np.array_equal(ds["X"], ds["X_clean"])


def compute_classification_teacher(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    *,
    requested: bool,
) -> dict[str, Any]:
    """Fit an optional L2 logistic teacher with stable failure reasons."""
    base = {
        "teacher_available": False,
        "teacher_type": "l2_logistic" if requested else "",
        "teacher_failure_reason": "not_requested" if not requested else "",
        "teacher_regularization": 1.0,
    }
    if not requested:
        return base
    if np.unique(y_train).size != num_classes:
        base["teacher_failure_reason"] = "missing_context_class"
        return base
    try:
        from sklearn.linear_model import LogisticRegression
    except (ImportError, ModuleNotFoundError):
        base["teacher_failure_reason"] = "dependency_missing"
        return base
    try:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)
        train_scores = model.decision_function(X_train)
        test_scores = model.decision_function(X_test)
        if num_classes == 2:
            train_logits = np.column_stack([np.zeros(X_train.shape[0]), train_scores])
            test_logits = np.column_stack([np.zeros(X_test.shape[0]), test_scores])
            teacher_W = np.column_stack([
                np.zeros(X_train.shape[1]), model.coef_[0]
            ])
            teacher_b = np.asarray([0.0, model.intercept_[0]])
        else:
            train_logits = np.asarray(train_scores, dtype=float)
            test_logits = np.asarray(test_scores, dtype=float)
            train_logits -= train_logits[:, [0]]
            test_logits -= test_logits[:, [0]]
            teacher_W = model.coef_.T.copy()
            teacher_W -= teacher_W[:, [0]]
            teacher_b = model.intercept_.copy()
            teacher_b -= teacher_b[0]
        base.update({
            "teacher_available": True,
            "teacher_failure_reason": "",
            "teacher_logits_train": train_logits.astype(np.float64),
            "teacher_logits_test": test_logits.astype(np.float64),
            "teacher_probs_train": _softmax(train_logits, 1.0),
            "teacher_probs_test": _softmax(test_logits, 1.0),
            "teacher_W": teacher_W.astype(np.float64),
            "teacher_b": teacher_b.astype(np.float64),
        })
    except Exception:
        base["teacher_failure_reason"] = "fit_failed"
    return base


def write_classification_parquet(
    ds: dict[str, Any],
    filepath: str,
    *,
    store_class_params: bool = True,
    store_teacher_preds: bool = False,
    teacher_results: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> int:
    """Write one row using the linear_classification_v1 contract."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    teacher = teacher_results or compute_classification_teacher(
        ds["X_train"], ds["y_train"], ds["X_test"], ds["num_classes"],
        requested=False,
    )
    diagnostics = diagnostics or {}
    list_f64 = pa.list_(pa.float64())
    matrix_f64 = pa.list_(list_f64)
    list_i64 = pa.list_(pa.int64())
    columns: dict[str, pa.Array] = {
        "X_train": pa.array([ds["X_train"].tolist()], type=matrix_f64),
        "y_train": pa.array([ds["y_train"].tolist()], type=list_i64),
        "X_test": pa.array([ds["X_test"].tolist()], type=matrix_f64),
        "y_test": pa.array([ds["y_test"].tolist()], type=list_i64),
        "y_clean_train": pa.array([ds["y_clean_train"].tolist()], type=list_i64),
        "y_clean_test": pa.array([ds["y_clean_test"].tolist()], type=list_i64),
        "label_noise_mask_train": pa.array(
            [ds["label_noise_mask_train"].tolist()], type=pa.list_(pa.bool_())
        ),
        "label_noise_mask_test": pa.array(
            [ds["label_noise_mask_test"].tolist()], type=pa.list_(pa.bool_())
        ),
        "logits_train": pa.array([ds["logits_train"].tolist()], type=matrix_f64),
        "logits_test": pa.array([ds["logits_test"].tolist()], type=matrix_f64),
        "probs_train": pa.array([ds["probs_train"].tolist()], type=matrix_f64),
        "dgp_teacher_probs_test": pa.array([ds["probs_test"].tolist()], type=matrix_f64),
        "n": pa.array([ds["n"]], type=pa.int64()),
        "p": pa.array([ds["p"]], type=pa.int64()),
        "n_train": pa.array([ds["n_train"]], type=pa.int64()),
        "n_test": pa.array([ds["n_test"]], type=pa.int64()),
        "p_signal": pa.array([ds["p_signal"]], type=pa.int64()),
        "p_noise": pa.array([ds["p_noise"]], type=pa.int64()),
        "p_total": pa.array([ds["p_total"]], type=pa.int64()),
        "num_classes": pa.array([ds["num_classes"]], type=pa.int64()),
        "realized_num_classes": pa.array(
            [ds["realized_num_classes"]], type=pa.int64()
        ),
        "schema_version": pa.array(["linear_classification_v1"], type=pa.utf8()),
        "task_family": pa.array(["linear_classification"], type=pa.utf8()),
        "task_objective": pa.array(["inductive_classification"], type=pa.utf8()),
        "default_target_mode": pa.array(["observed_label"], type=pa.utf8()),
        "prior_regime": pa.array([ds["prior_regime"]], type=pa.utf8()),
        "classification_regime": pa.array(
            [ds["classification_regime"]], type=pa.utf8()
        ),
        "coefficient_regime": pa.array([ds["coefficient_regime"]], type=pa.utf8()),
        "class_imbalance_type": pa.array(
            [ds["class_imbalance_type"]], type=pa.utf8()
        ),
        "margin_level": pa.array([ds["margin_level"]], type=pa.utf8()),
        "realized_margin_level": pa.array(
            [ds["realized_margin_level"]], type=pa.utf8()
        ),
        "class_prior": pa.array([ds["class_prior"].tolist()], type=list_f64),
        "realized_class_prior": pa.array(
            [ds["realized_class_prior"].tolist()], type=list_f64
        ),
        "train_class_counts": pa.array(
            [ds["train_class_counts"].tolist()], type=list_i64
        ),
        "test_class_counts": pa.array(
            [ds["test_class_counts"].tolist()], type=list_i64
        ),
    }
    scalar_types = {
        "active_s": pa.int64(),
        "sparsity_ratio": pa.float64(),
        "class_sparsity_ratio": pa.float64(),
        "covariance_type": pa.utf8(),
        "rho": pa.float64(),
        "block_size": pa.int64(),
        "temperature": pa.float64(),
        "label_noise_rate": pa.float64(),
        "realized_label_noise_rate": pa.float64(),
        "coefficient_scale": pa.float64(),
        "intercept_scale": pa.float64(),
        "feature_noise_level": pa.float64(),          # canonical FLOAT
        "feature_noise_level_percent": pa.int64(),     # legacy int (percent * 100)
        "feature_noise_level_float": pa.float64(),     # kept for old readers
        "teacher_available": pa.bool_(),
        "teacher_type": pa.utf8(),
        "teacher_failure_reason": pa.utf8(),
        "teacher_regularization": pa.float64(),
    }
    scalar_values = {
        **ds,
        "feature_noise_level": float(ds["feature_noise_level"]),          # canonical FLOAT
        "feature_noise_level_percent": int(round(ds["feature_noise_level"] * 100)),  # legacy int
        "feature_noise_level_float": float(ds["feature_noise_level"]),    # kept for old readers
        **teacher,
    }
    for name, arrow_type in scalar_types.items():
        _add_scalar(columns, name, scalar_values.get(name), arrow_type)
    for name in (
        "condition_number",
        "effective_rank",
        "mean_margin",
        "median_margin",
        "min_margin",
        "mean_normalized_margin",
        "median_normalized_margin",
        "min_normalized_margin",
        "class_entropy",
        "minority_class_fraction",
        "majority_class_fraction",
        "bayes_error_proxy",
    ):
        _add_scalar(columns, name, diagnostics.get(name), pa.float64())
    for name in ("matrix_rank", "min_class_count", "max_class_count"):
        _add_scalar(columns, name, diagnostics.get(name), pa.int64())

    if store_class_params:
        columns["active_support"] = pa.array(
            [ds["active_support"].tolist()], type=pa.list_(pa.bool_())
        )
        # Canonical for all K: W_true always (p, K), b_true always (K,)
        columns["W_true"] = pa.array([ds["W_true"].tolist()], type=matrix_f64)
        columns["b_true"] = pa.array([ds["b_true"].tolist()], type=list_f64)
        # Deprecated w_true column kept for old readers: equals W_true[:, 1]
        columns["w_true"] = pa.array([ds["W_true"][:, 1].tolist()], type=list_f64)
        columns["class_active_support"] = pa.array(
            [ds["class_active_support"].tolist()],
            type=pa.list_(pa.list_(pa.bool_())),
        )

    if store_teacher_preds and teacher.get("teacher_available"):
        for name in (
            "teacher_logits_train",
            "teacher_logits_test",
            "teacher_probs_train",
            "teacher_probs_test",
            "teacher_W",
        ):
            columns[name] = pa.array([teacher[name].tolist()], type=matrix_f64)
        columns["teacher_b"] = pa.array(
            [teacher["teacher_b"].tolist()], type=list_f64
        )
    table = pa.table(columns, metadata={"coeff_schema_version": "canonical_v1"})
    pq.write_table(table, filepath)
    return int(Path(filepath).stat().st_size)


# ---------------------------------------------------------------------------
# 16. Classification Eval Parquet Writer
# ---------------------------------------------------------------------------

def write_parquet_classification_eval(
    ds: dict[str, Any],
    filepath: str,
    *,
    suite_id: str,
    suite_family: str,
    dataset_idx: int,
    global_idx: int,
    profile: str,
    regime: str,
    task_seed: int | None = None,
    store_class_params: bool = True,
    store_teacher_preds: bool = False,
    teacher_results: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    is_tabpfn_anchor: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    """Write evaluation-suite parquet in the linear_classification_eval_v1 format.

    Produces one row per sample (context + query split), not one row per dataset.
    Returns n_train + n_test (total number of rows written).
    """
    if suite_family not in CLASSIFICATION_EVAL_SUITE_FAMILIES:
        raise ValueError(
            f"suite_family {suite_family!r} not in CLASSIFICATION_EVAL_SUITE_FAMILIES; "
            f"valid: {sorted(CLASSIFICATION_EVAL_SUITE_FAMILIES)}"
        )
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_test = ds["X_test"]
    y_test = ds["y_test"]
    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    n_features = int(ds["p_total"])
    num_classes = int(ds["num_classes"])
    n_rows = n_train + n_test

    # Build per-row arrays
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    split_col = ["context"] * n_train + ["query"] * n_test

    list_f64 = pa.list_(pa.float64())
    matrix_f64 = pa.list_(list_f64)

    columns: dict[str, pa.Array] = {
        "feature_vector": pa.array(X_all.tolist(), type=list_f64),
        "label": pa.array(y_all.tolist(), type=pa.int64()),
        "split": pa.array(split_col, type=pa.utf8()),
        "schema_version": pa.array(
            [CLASSIFICATION_EVAL_SCHEMA_VERSION] * n_rows, type=pa.utf8()
        ),
        "suite_id": pa.array([suite_id] * n_rows, type=pa.utf8()),
        "suite_family": pa.array([suite_family] * n_rows, type=pa.utf8()),
        "dataset_idx": pa.array([dataset_idx] * n_rows, type=pa.int64()),
        "global_idx": pa.array([global_idx] * n_rows, type=pa.int64()),
        "profile": pa.array([profile] * n_rows, type=pa.utf8()),
        "regime": pa.array([regime] * n_rows, type=pa.utf8()),
        "task_objective": pa.array(["inductive_classification"] * n_rows, type=pa.utf8()),
        "task_family": pa.array(["linear_classification"] * n_rows, type=pa.utf8()),
        "n_train": pa.array([n_train] * n_rows, type=pa.int64()),
        "n_test": pa.array([n_test] * n_rows, type=pa.int64()),
        "n_features": pa.array([n_features] * n_rows, type=pa.int64()),
        "num_classes": pa.array([num_classes] * n_rows, type=pa.int64()),
        "imbalance_type": pa.array(
            [ds.get("class_imbalance_type", "")] * n_rows, type=pa.utf8()
        ),
        "margin_level": pa.array(
            [ds.get("margin_level", "")] * n_rows, type=pa.utf8()
        ),
        "label_noise_rate": pa.array(
            [float(ds.get("label_noise_rate", 0.0))] * n_rows, type=pa.float64()
        ),
        "feature_noise_level": pa.array(
            [float(ds.get("feature_noise_level", 0.0))] * n_rows,
            type=pa.float64(),
        ),
        "task_seed": pa.array(
            [int(task_seed) if task_seed is not None else -1] * n_rows,
            type=pa.int64(),
        ),
        "is_tabpfn_anchor": pa.array([is_tabpfn_anchor] * n_rows, type=pa.bool_()),
        "distribution_family": pa.array(
            [ds.get("distribution_family", "gaussian_linear_logits")] * n_rows,
            type=pa.utf8(),
        ),
        "covariance_type": pa.array(
            [ds.get("covariance_type", "iid")] * n_rows, type=pa.utf8()
        ),
        "rho": pa.array([float(ds.get("rho", 0.0))] * n_rows, type=pa.float64()),
        "temperature": pa.array(
            [float(ds.get("temperature", 1.0))] * n_rows, type=pa.float64()
        ),
    }
    for name, value in (extra_metadata or {}).items():
        if isinstance(value, bool):
            columns[name] = pa.array([value] * n_rows, type=pa.bool_())
        elif isinstance(value, int):
            columns[name] = pa.array([value] * n_rows, type=pa.int64())
        elif isinstance(value, float):
            columns[name] = pa.array([value] * n_rows, type=pa.float64())
        elif isinstance(value, str):
            columns[name] = pa.array([value] * n_rows, type=pa.utf8())
        elif isinstance(value, (list, tuple)):
            columns[name] = pa.array(
                [json.dumps(list(value), sort_keys=True)] * n_rows, type=pa.utf8()
            )

    # Phase 7: Add y_clean and label_noise_mask per-row columns if available
    y_clean_train = ds.get("y_clean_train")
    y_clean_test = ds.get("y_clean_test")
    if y_clean_train is not None and y_clean_test is not None:
        y_clean_all = np.concatenate([
            np.asarray(y_clean_train, dtype=np.int64),
            np.asarray(y_clean_test, dtype=np.int64),
        ])
        columns["y_clean"] = pa.array(y_clean_all.tolist(), type=pa.int64())
    else:
        # Try flat y_clean if split not done
        y_clean = ds.get("y_clean")
        if y_clean is not None:
            yc = np.asarray(y_clean, dtype=np.int64)
            if len(yc) == n_rows:
                columns["y_clean"] = pa.array(yc.tolist(), type=pa.int64())

    lnm_train = ds.get("label_noise_mask_train")
    lnm_test = ds.get("label_noise_mask_test")
    if lnm_train is not None and lnm_test is not None:
        lnm_all = np.concatenate([
            np.asarray(lnm_train, dtype=bool),
            np.asarray(lnm_test, dtype=bool),
        ])
        columns["label_noise_mask"] = pa.array(lnm_all.tolist(), type=pa.bool_())
    else:
        lnm = ds.get("label_noise_mask")
        if lnm is not None:
            lnm_arr = np.asarray(lnm, dtype=bool)
            if len(lnm_arr) == n_rows:
                columns["label_noise_mask"] = pa.array(lnm_arr.tolist(), type=pa.bool_())

    if store_class_params:
        # Canonical for all K: W_true always (p, K), b_true always (K,)
        W = ds.get("W_true")
        b = ds.get("b_true")
        if W is not None:
            columns["class_weight_matrix"] = pa.array(
                [W.tolist()] * n_rows, type=matrix_f64
            )
        if b is not None:
            b_arr = np.asarray(b)
            if b_arr.ndim == 0:
                # scalar → treat as b[1] with b[0]=0 for K=2
                columns["class_bias_vector"] = pa.array(
                    [[0.0, float(b_arr)]] * n_rows, type=list_f64
                )
            else:
                columns["class_bias_vector"] = pa.array(
                    [list(map(float, b))] * n_rows, type=list_f64
                )

    if store_teacher_preds and teacher_results is not None and teacher_results.get("teacher_available"):
        teacher = teacher_results
        tp_train = teacher.get("teacher_probs_train")
        tp_test = teacher.get("teacher_probs_test")
        if tp_train is not None and tp_test is not None:
            tp_all = np.vstack([tp_train, tp_test])
            for k in range(num_classes):
                columns[f"teacher_prob_{k}"] = pa.array(
                    tp_all[:, k].tolist(), type=pa.float64()
                )
        for col, arrow_type in [
            ("teacher_converged", pa.bool_()),
            ("teacher_n_iter", pa.int64()),
            ("teacher_C", pa.float64()),
        ]:
            val = teacher_results.get(col)
            if val is not None:
                columns[col] = pa.array([val] * n_rows, type=arrow_type)

    # Assert no regression-only fields leaked into schema
    for col in _REGRESSION_ONLY_FIELDS:
        if col in columns:
            raise AssertionError(
                f"Regression-only field {col!r} must not appear in classification eval parquet"
            )

    # Embed diagnostics as Parquet schema metadata (not a column)
    metadata: dict[bytes, bytes] = {}
    if diagnostics:
        import json as _json
        metadata[b"diagnostics"] = _json.dumps(diagnostics).encode()
    if extra_metadata:
        import json as _json
        for k, v in extra_metadata.items():
            metadata[k.encode()] = _json.dumps(v).encode()

    table = pa.table(columns)
    existing = table.schema.metadata or {}
    schema_meta = {
        b"coeff_schema_version": b"canonical_v1",
        **existing,
        **{k.encode() if isinstance(k, str) else k: (v if isinstance(v, bytes) else v) for k, v in metadata.items()},
    }
    table = table.replace_schema_metadata(schema_meta)
    pq.write_table(table, filepath)
    return n_train + n_test


# ---------------------------------------------------------------------------
# 17. Classification Eval Validator
# ---------------------------------------------------------------------------

def validate_classification_eval_dataset(
    ds_or_table: Any,
    *,
    strict: bool = True,
    check_teacher: bool = False,
) -> list[str]:
    """Validate a loaded Parquet table in the linear_classification_eval_v1 format.

    Returns list of error strings. If strict=True, raises ValueError on first error.
    """
    errors: list[str] = []

    def _err(msg: str) -> None:
        errors.append(msg)
        if strict:
            raise ValueError(msg)

    # Must be a PyArrow Table
    if not hasattr(ds_or_table, "column_names"):
        _err("Input must be a PyArrow Table")
        return errors

    table = ds_or_table

    def _col_values(name: str) -> list:
        return table.column(name).to_pylist()

    def _has_col(name: str) -> bool:
        return name in table.column_names

    col_types = {f.name: f.type for f in table.schema}

    # 1. schema_version
    if not _has_col("schema_version"):
        _err("Missing column: schema_version")
    else:
        versions = set(_col_values("schema_version"))
        if not versions.issubset(SUPPORTED_CLASSIFICATION_EVAL_SCHEMA_VERSIONS):
            _err(
                "schema_version must be one of "
                f"{sorted(SUPPORTED_CLASSIFICATION_EVAL_SCHEMA_VERSIONS)!r}; "
                f"found {versions}"
            )

    # 2. suite_family
    if not _has_col("suite_family"):
        _err("Missing column: suite_family")
    else:
        families = set(_col_values("suite_family"))
        unknown = families - CLASSIFICATION_EVAL_SUITE_FAMILIES
        if unknown:
            _err(f"Unknown suite_family values: {unknown}")

    # 3. task_objective
    if not _has_col("task_objective"):
        _err("Missing column: task_objective")
    else:
        objs = set(_col_values("task_objective"))
        if objs != {"inductive_classification"}:
            _err(f"task_objective must be 'inductive_classification'; found {objs}")

    # 4. split column
    if not _has_col("split"):
        _err("Missing column: split")
    else:
        split_vals = set(_col_values("split"))
        if not split_vals.issubset({"context", "query"}):
            _err(f"split column has unexpected values: {split_vals - {'context', 'query'}}")

    # 5. context/query row counts vs n_train/n_test
    if _has_col("split") and _has_col("n_train") and _has_col("n_test"):
        split_vals_list = _col_values("split")
        n_context = sum(1 for s in split_vals_list if s == "context")
        n_query = sum(1 for s in split_vals_list if s == "query")
        n_train_val = _col_values("n_train")[0]
        n_test_val = _col_values("n_test")[0]
        if n_context != n_train_val:
            _err(f"Context row count {n_context} != n_train {n_train_val}")
        if n_query != n_test_val:
            _err(f"Query row count {n_query} != n_test {n_test_val}")

    # 6. No regression-only fields
    for field in _REGRESSION_ONLY_FIELDS:
        if _has_col(field):
            _err(f"Regression-only field {field!r} must not appear in eval table")

    # 7. label column integer dtype with values in [0, K-1]
    if not _has_col("label"):
        _err("Missing column: label")
    elif not _has_col("num_classes"):
        _err("Missing column: num_classes (needed to validate label range)")
    else:
        label_type = col_types.get("label")
        if label_type not in (pa.int64(), pa.int32(), pa.int16(), pa.int8()):
            _err(f"label column must be integer dtype; found {label_type}")
        num_classes_val = _col_values("num_classes")[0]
        labels = _col_values("label")
        if any(v < 0 or v >= num_classes_val for v in labels):
            _err(f"label values out of range [0, {num_classes_val - 1}]")

    # 8. Feature vector is list<float64>
    if _has_col("feature_vector"):
        fv_type = col_types.get("feature_vector")
        if not (pa.types.is_list(fv_type) and fv_type.value_type == pa.float64()):
            _err(f"feature_vector must be list<float64>; found {fv_type}")

    # 9. dataset_idx and global_idx are non-negative integers
    for idx_col in ("dataset_idx", "global_idx"):
        if not _has_col(idx_col):
            _err(f"Missing column: {idx_col}")
        else:
            vals = _col_values(idx_col)
            if any(v < 0 for v in vals):
                _err(f"{idx_col} contains negative values")

    # 10. is_tabpfn_anchor is boolean
    if not _has_col("is_tabpfn_anchor"):
        _err("Missing column: is_tabpfn_anchor")
    else:
        anchor_type = col_types.get("is_tabpfn_anchor")
        if anchor_type != pa.bool_():
            _err(f"is_tabpfn_anchor must be bool; found {anchor_type}")

    # Optional teacher check
    if check_teacher:
        if not _has_col("teacher_prob_0"):
            _err("check_teacher=True but teacher_prob_0 column missing")

    return errors


# ---------------------------------------------------------------------------
# 18. Classification Eval Task Loader
# ---------------------------------------------------------------------------

def load_classification_eval_task(table: Any) -> dict[str, Any]:
    """Load a PyArrow Table in linear_classification_eval_v1/v2 format into a task dict.

    Converts per-row parquet (context/query split) into the canonical task object:
      X_train, y_train  — from rows where split == "context"
      X_test,  y_test   — from rows where split == "query"

    Also extracts all scalar metadata fields and, if present, optional columns:
      y_clean, label_noise_mask  → y_clean_test, label_noise_mask_test (query rows)
      teacher_prob_0..K-1        → dgp_teacher_probs_test (query rows, shape [n_test, K])

    Parameters
    ----------
    table : pyarrow.Table
        Table produced by write_parquet_classification_eval().

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test  (numpy arrays)
        num_classes, suite_id, suite_family, n_train, n_test, n_features,
        label_noise_rate, feature_noise_level, margin_level, task_seed,
        regime, profile, dataset_idx, global_idx, is_tabpfn_anchor,
        task_family, task_objective, distribution_family, covariance_type,
        rho, temperature, imbalance_type
        (optional) y_clean_test, label_noise_mask_test (if columns present)
        (optional) dgp_teacher_probs_test               (if teacher_prob_0 column present)

    Raises
    ------
    ValueError on:
        - Wrong schema version
        - Missing context or query rows (empty arrays)
        - Labels out of range [0, K-1]
        - Inconsistent feature widths between context and query
        - Inconsistent scalar metadata across rows
    """
    import numpy as np

    if not hasattr(table, "column_names"):
        raise ValueError("load_classification_eval_task: input must be a PyArrow Table")

    # Fail-fast validation
    validate_classification_eval_dataset(table, strict=True)

    col_names = set(table.column_names)

    def _col(name: str):
        return table.column(name).to_pylist()

    def _col_np(name: str):
        return table.column(name).to_pylist()

    split_col = _col("split")
    context_mask = [s == "context" for s in split_col]
    query_mask = [s == "query" for s in split_col]

    n_context = sum(context_mask)
    n_query = sum(query_mask)

    if n_context == 0:
        raise ValueError("load_classification_eval_task: no context rows found (n_train=0)")
    if n_query == 0:
        raise ValueError("load_classification_eval_task: no query rows found (n_test=0)")

    # Extract feature vectors
    fv_col = _col("feature_vector")
    fv_context = [fv_col[i] for i, m in enumerate(context_mask) if m]
    fv_query = [fv_col[i] for i, m in enumerate(query_mask) if m]

    X_train = np.array(fv_context, dtype=np.float64)
    X_test = np.array(fv_query, dtype=np.float64)

    # Validate consistent feature widths
    if X_train.ndim == 2 and X_test.ndim == 2 and X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"load_classification_eval_task: feature width mismatch — "
            f"context has {X_train.shape[1]} features, query has {X_test.shape[1]}"
        )

    # Extract labels
    label_col = _col("label")
    y_train = np.array(
        [label_col[i] for i, m in enumerate(context_mask) if m], dtype=np.int64
    )
    y_test = np.array(
        [label_col[i] for i, m in enumerate(query_mask) if m], dtype=np.int64
    )

    # Validate label range
    num_classes_col = _col("num_classes")
    # Scalar metadata must be consistent across rows
    def _scalar(col_name: str, cast=None):
        if col_name not in col_names:
            return None
        vals = _col(col_name)
        unique = set(vals)
        if len(unique) > 1:
            raise ValueError(
                f"load_classification_eval_task: column {col_name!r} has inconsistent "
                f"scalar values across rows: {unique}"
            )
        v = vals[0]
        return cast(v) if cast is not None and v is not None else v

    num_classes = _scalar("num_classes", int)
    if num_classes is not None:
        if np.any(y_train < 0) or np.any(y_train >= num_classes):
            raise ValueError(
                f"load_classification_eval_task: y_train labels out of range [0, {num_classes - 1}]"
            )
        if np.any(y_test < 0) or np.any(y_test >= num_classes):
            raise ValueError(
                f"load_classification_eval_task: y_test labels out of range [0, {num_classes - 1}]"
            )

    task = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "num_classes": num_classes,
        "n_train": n_context,
        "n_test": n_query,
        "n_features": X_train.shape[1] if X_train.ndim == 2 else 0,
        "suite_id": _scalar("suite_id"),
        "suite_family": _scalar("suite_family"),
        "label_noise_rate": _scalar("label_noise_rate", float),
        "feature_noise_level": _scalar("feature_noise_level", float),
        "margin_level": _scalar("margin_level"),
        "task_seed": _scalar("task_seed", int),
        "regime": _scalar("regime"),
        "profile": _scalar("profile"),
        "dataset_idx": _scalar("dataset_idx", int),
        "global_idx": _scalar("global_idx", int),
        "is_tabpfn_anchor": _scalar("is_tabpfn_anchor", bool),
        "task_family": _scalar("task_family"),
        "task_objective": _scalar("task_objective"),
        "distribution_family": _scalar("distribution_family"),
        "covariance_type": _scalar("covariance_type"),
        "rho": _scalar("rho", float),
        "temperature": _scalar("temperature", float),
        "imbalance_type": _scalar("imbalance_type"),
    }

    # Optional: y_clean and label_noise_mask (Phase 7 compatibility)
    if "y_clean" in col_names:
        yc_col = _col("y_clean")
        task["y_clean_test"] = np.array(
            [yc_col[i] for i, m in enumerate(query_mask) if m], dtype=np.int64
        )
    else:
        task["y_clean_test"] = None

    if "label_noise_mask" in col_names:
        lnm_col = _col("label_noise_mask")
        task["label_noise_mask_test"] = np.array(
            [lnm_col[i] for i, m in enumerate(query_mask) if m], dtype=bool
        )
    else:
        task["label_noise_mask_test"] = None

    # Optional: teacher probability columns teacher_prob_0 .. teacher_prob_{K-1}
    if num_classes is not None and "teacher_prob_0" in col_names:
        try:
            tp_cols = []
            for k in range(num_classes):
                col_k = f"teacher_prob_{k}"
                if col_k not in col_names:
                    break
                col_vals = _col(col_k)
                tp_cols.append(
                    [col_vals[i] for i, m in enumerate(query_mask) if m]
                )
            if len(tp_cols) == num_classes:
                task["dgp_teacher_probs_test"] = np.array(tp_cols, dtype=np.float64).T
            else:
                task["dgp_teacher_probs_test"] = None
        except Exception:
            task["dgp_teacher_probs_test"] = None
    else:
        task["dgp_teacher_probs_test"] = None

    return task


# ---------------------------------------------------------------------------
# 15b. Mixed-categorical eval task loaders
# ---------------------------------------------------------------------------

def load_mixed_regression_eval_task(table: Any) -> dict[str, Any]:
    """Load a mixed-categorical regression eval task from a PyArrow Table.

    The parquet format stores one row per task with nested array columns
    (produced by write_parquet_mixed_regression_dgp).

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test  (numpy arrays, numeric features only)
        X_cat_train, X_cat_test            (numpy int64 arrays, entity-embed-encoded)
        categorical_cardinalities          (numpy int64 array)
        p_num, p_cat, max_cardinality, n_train, n_test, n_features
        prior_regime, schema_version, task_family, training_data_family,
        task_objective
    """
    import numpy as np

    if not hasattr(table, "column_names"):
        raise ValueError("load_mixed_regression_eval_task: input must be a PyArrow Table")

    if table.num_rows != 1:
        raise ValueError(
            f"load_mixed_regression_eval_task: expected 1 row per task, got {table.num_rows}"
        )

    d = table.to_pydict()

    X_train = np.array(d["X_num_train"][0], dtype=np.float64)
    X_test = np.array(d["X_num_test"][0], dtype=np.float64)
    y_train = np.array(d["y_train"][0], dtype=np.float64)
    y_test = np.array(d["y_test"][0], dtype=np.float64)
    X_cat_train = np.array(d["X_cat_train"][0], dtype=np.int64)
    X_cat_test = np.array(d["X_cat_test"][0], dtype=np.int64)
    categorical_cardinalities = np.array(d["categorical_cardinalities"][0], dtype=np.int64)

    p_num = int(d["p_num"][0])
    p_cat = int(d["p_cat"][0])

    task = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_cat_train": X_cat_train,
        "X_cat_test": X_cat_test,
        "categorical_cardinalities": categorical_cardinalities,
        "p_num": p_num,
        "p_cat": p_cat,
        "max_cardinality": int(categorical_cardinalities.max()) if len(categorical_cardinalities) > 0 else 0,
        "n_train": int(d["n_train"][0]),
        "n_test": int(d["n_test"][0]),
        "n_features": p_num + p_cat,
        "prior_regime": str(d["prior_regime"][0]) if "prior_regime" in d else None,
        "schema_version": str(d["schema_version"][0]) if "schema_version" in d else None,
        "task_family": str(d.get("task_family", ["linear_regression"])[0]),
        "training_data_family": str(d["training_data_family"][0]) if "training_data_family" in d else None,
        "task_objective": str(d.get("task_objective", ["inductive_regression"])[0]),
    }

    # Optional mask columns
    if "cat_missing_mask_train" in d:
        task["cat_missing_mask_train"] = np.array(d["cat_missing_mask_train"][0], dtype=bool)
    if "cat_missing_mask_test" in d:
        task["cat_missing_mask_test"] = np.array(d["cat_missing_mask_test"][0], dtype=bool)
    if "cat_unknown_mask_test" in d:
        task["cat_unknown_mask_test"] = np.array(d["cat_unknown_mask_test"][0], dtype=bool)

    return task


def load_mixed_classification_eval_task(table: Any) -> dict[str, Any]:
    """Load a mixed-categorical classification eval task from a PyArrow Table.

    The parquet format stores one row per task with nested array columns
    (produced by write_parquet_mixed_classification_dgp).

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test  (numpy arrays, numeric features only)
        X_cat_train, X_cat_test            (numpy int64 arrays, entity-embed-encoded)
        categorical_cardinalities          (numpy int64 array)
        num_classes, p_num, p_cat, max_cardinality, n_train, n_test, n_features
        prior_regime, schema_version, task_family, training_data_family,
        task_objective, temperature, label_noise_rate, class_imbalance_type
    """
    import numpy as np

    if not hasattr(table, "column_names"):
        raise ValueError("load_mixed_classification_eval_task: input must be a PyArrow Table")

    if table.num_rows != 1:
        raise ValueError(
            f"load_mixed_classification_eval_task: expected 1 row per task, got {table.num_rows}"
        )

    d = table.to_pydict()

    X_train = np.array(d["X_num_train"][0], dtype=np.float64)
    X_test = np.array(d["X_num_test"][0], dtype=np.float64)
    y_train = np.array(d["y_train"][0], dtype=np.int64)
    y_test = np.array(d["y_test"][0], dtype=np.int64)
    X_cat_train = np.array(d["X_cat_train"][0], dtype=np.int64)
    X_cat_test = np.array(d["X_cat_test"][0], dtype=np.int64)
    categorical_cardinalities = np.array(d["categorical_cardinalities"][0], dtype=np.int64)

    p_num = int(d["p_num"][0])
    p_cat = int(d["p_cat"][0])
    num_classes = int(d["num_classes"][0])

    # Validate label range
    if np.any(y_train < 0) or np.any(y_train >= num_classes):
        raise ValueError(
            f"load_mixed_classification_eval_task: y_train labels out of range [0, {num_classes - 1}]"
        )
    if np.any(y_test < 0) or np.any(y_test >= num_classes):
        raise ValueError(
            f"load_mixed_classification_eval_task: y_test labels out of range [0, {num_classes - 1}]"
        )

    task = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_cat_train": X_cat_train,
        "X_cat_test": X_cat_test,
        "categorical_cardinalities": categorical_cardinalities,
        "num_classes": num_classes,
        "p_num": p_num,
        "p_cat": p_cat,
        "max_cardinality": int(categorical_cardinalities.max()) if len(categorical_cardinalities) > 0 else 0,
        "n_train": int(d["n_train"][0]),
        "n_test": int(d["n_test"][0]),
        "n_features": p_num + p_cat,
        "prior_regime": str(d["prior_regime"][0]) if "prior_regime" in d else None,
        "schema_version": str(d["schema_version"][0]) if "schema_version" in d else None,
        "task_family": str(d.get("task_family", ["linear_classification"])[0]),
        "training_data_family": str(d["training_data_family"][0]) if "training_data_family" in d else None,
        "task_objective": str(d.get("task_objective", ["inductive_classification"])[0]),
        "temperature": float(d.get("temperature", [1.0])[0]),
        "label_noise_rate": float(d.get("label_noise_rate", [0.0])[0]),
        "class_imbalance_type": str(d.get("class_imbalance_type", ["balanced"])[0]),
    }

    # Optional mask columns
    if "cat_missing_mask_train" in d:
        task["cat_missing_mask_train"] = np.array(d["cat_missing_mask_train"][0], dtype=bool)
    if "cat_missing_mask_test" in d:
        task["cat_missing_mask_test"] = np.array(d["cat_missing_mask_test"][0], dtype=bool)
    if "cat_unknown_mask_test" in d:
        task["cat_unknown_mask_test"] = np.array(d["cat_unknown_mask_test"][0], dtype=bool)

    return task


# ---------------------------------------------------------------------------
# 16. Controlled Suite Design Utilities
# ---------------------------------------------------------------------------

def parse_difficulty_mix(s: str) -> dict[str, float]:
    """Parse a difficulty-mix string like 'core=0.65,robust=0.30,stress=0.05'.

    Returns a dict mapping tier name to float fraction.
    Raises ValueError on unknown keys or if fractions don't sum to ~1.
    """
    valid_keys = {"core", "robust", "stress"}
    result: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"parse_difficulty_mix: expected 'key=value', got {part!r}")
        key, val = part.split("=", 1)
        key = key.strip()
        if key not in valid_keys:
            raise ValueError(
                f"parse_difficulty_mix: unknown key {key!r}; valid keys are {sorted(valid_keys)}"
            )
        result[key] = float(val.strip())
    total = sum(result.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"parse_difficulty_mix: fractions must sum to ~1.0; got {total:.4f}"
        )
    return result


def compute_difficulty_score(task_metadata: dict, task_family: str) -> dict:
    """Compute difficulty score, tier, and reasons for a task.

    Parameters
    ----------
    task_metadata : dict
        Metadata dict for the task (uses .get() throughout for safety).
    task_family : str
        Either 'linear_regression' or 'linear_classification'.

    Returns
    -------
    dict with keys: difficulty_score (int), difficulty_tier (str), difficulty_reasons (list[str])
    """
    if task_family not in {"linear_regression", "linear_classification"}:
        raise ValueError(
            f"compute_difficulty_score: unknown task_family {task_family!r}; "
            "must be 'linear_regression' or 'linear_classification'"
        )

    score = 0
    reasons: list[str] = []
    p_total = task_metadata.get("p_total", 0)
    n_train = task_metadata.get("n_train", 0)

    if task_family == "linear_regression":
        if p_total > 128:
            score += 1; reasons.append("high_p_total")
        if n_train < p_total:
            score += 1; reasons.append("underdetermined")
        if n_train / max(1, p_total) < 2:
            score += 1; reasons.append("low_n_over_p")
        target_noise_scale = task_metadata.get("target_noise_scale", 0.0)
        if target_noise_scale >= 1.0:
            score += 1; reasons.append("high_target_noise")
        feature_noise_level = task_metadata.get("feature_noise_level", 0.0)
        if feature_noise_level >= 0.25:
            score += 1; reasons.append("high_feature_noise")
        p_noise = task_metadata.get("p_noise", 0)
        if p_noise / max(1, p_total) >= 0.50:
            score += 1; reasons.append("high_noise_ratio")
        covariance_type = task_metadata.get("covariance_type", "iid")
        rho = task_metadata.get("rho", 0.0)
        if covariance_type in {"block", "equicorrelation"} and rho >= 0.6:
            score += 1; reasons.append("high_correlation")
        prior_regime = task_metadata.get("prior_regime", "")
        if prior_regime == "J_low_n_high_p":
            score += 1; reasons.append("j_low_n_high_p_regime")
        target_noise_type = task_metadata.get("target_noise_type", "gaussian")
        if target_noise_type == "student_t":
            score += 1; reasons.append("heavy_tail_noise")
    else:  # linear_classification
        if p_total > 128:
            score += 1; reasons.append("high_p_total")
        if n_train < p_total:
            score += 1; reasons.append("underdetermined")
        if n_train / max(1, p_total) < 2:
            score += 1; reasons.append("low_n_over_p")
        label_noise_rate = task_metadata.get("label_noise_rate", 0.0)
        if label_noise_rate >= 0.10:
            score += 1; reasons.append("high_label_noise")
        feature_noise_level = task_metadata.get("feature_noise_level", 0.0)
        if feature_noise_level >= 0.25:
            score += 1; reasons.append("high_feature_noise")
        p_noise = task_metadata.get("p_noise", 0)
        if p_noise / max(1, p_total) >= 0.50:
            score += 1; reasons.append("high_noise_ratio")
        class_imbalance_type = task_metadata.get("class_imbalance_type", "balanced")
        if class_imbalance_type == "severe":
            score += 1; reasons.append("severe_class_imbalance")
        margin_level = task_metadata.get("margin_level", "medium")
        if margin_level == "low":
            score += 1; reasons.append("low_margin")
        num_classes = task_metadata.get("num_classes", 2)
        if num_classes >= 5:
            score += 1; reasons.append("many_classes")
        classification_regime = task_metadata.get("classification_regime", "")
        if classification_regime == "J_low_n_high_p_classification":
            score += 1; reasons.append("j_low_n_high_p_regime")

    # Tier thresholds: 0-1 -> core, 2-3 -> robust, >=4 -> stress
    if score <= 1:
        tier = "core"
    elif score <= 3:
        tier = "robust"
    else:
        tier = "stress"

    return {
        "difficulty_score": score,
        "difficulty_tier": tier,
        "difficulty_reasons": reasons,
    }


def estimate_memory_risk(
    p_total: int,
    n_train: int,
    n_test: int,
    context_size: int = 200,
    test_batch_size: int = 128,
    feature_cap: int = 128,
    gpu_guard_bytes: int = 268_435_456,
    hidden_estimate: int = 64,
) -> dict:
    """Estimate GPU memory risk for a DeepSet model on this task.

    Formula:
        effective_p       = min(p_total, feature_cap)
        effective_context = min(n_train, context_size)
        h_tensor_bytes    = 5 × effective_context × effective_p × hidden_estimate × 4
        ratio             = h_tensor_bytes / gpu_guard_bytes

    Returns a dict with memory risk metadata.
    """
    effective_p = min(p_total, feature_cap)
    effective_context = min(n_train, context_size)
    h_tensor_bytes = 5 * effective_context * effective_p * hidden_estimate * 4
    ratio = h_tensor_bytes / max(1, gpu_guard_bytes)

    if ratio < 0.50:
        bucket = "low"
    elif ratio < 0.90:
        bucket = "medium"
    elif ratio <= 1.00:
        bucket = "high"
    else:
        bucket = "exceeds_default_guard"

    exceeds = bucket == "exceeds_default_guard"

    # Recommended caps: binary-search descending from limits until bytes < guard (min 32)
    def _find_cap(fixed_ctx: int, fixed_feat: int, vary: str) -> int:
        lo, hi = 32, (context_size if vary == "context" else feature_cap)
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if vary == "context":
                b = 5 * mid * fixed_feat * hidden_estimate * 4
            else:
                b = 5 * fixed_ctx * mid * hidden_estimate * 4
            if b < gpu_guard_bytes:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return max(32, best)

    rec_feature_cap = _find_cap(effective_context, effective_p, "feature") if exceeds else feature_cap
    rec_context_cap = _find_cap(effective_context, effective_p, "context") if exceeds else context_size
    rec_test_batch_size = max(32, test_batch_size // 2) if exceeds else test_batch_size

    return {
        "estimated_context_rows": effective_context,
        "estimated_query_batch_size": min(n_test, test_batch_size),
        "estimated_p_total": effective_p,
        "estimated_h_tensor_bytes": h_tensor_bytes,
        "memory_risk_bucket": bucket,
        "exceeds_default_gpu_guard": exceeds,
        "recommended_feature_cap": rec_feature_cap,
        "recommended_context_cap": rec_context_cap,
        "recommended_test_batch_size": rec_test_batch_size,
    }


def allocate_regression_tasks(
    n_datasets: int,
    profile: str,
    base_seed: int,
    *,
    allow_underdetermined: bool = False,
    n_grid: "list[int] | None" = None,
    p_signal_grid: "list[int] | None" = None,
    p_noise_grid: "list[int] | None" = None,
    active_s_grid: "list[int] | None" = None,
    rho_grid: "list[float] | None" = None,
    target_noise_grid: "list[float] | None" = None,
    feature_noise_grid: "list[float] | None" = None,
    allocation_mode: str = "weighted_quota",
    min_regime_count: int = 10,
    strict_coverage: bool = False,
) -> "tuple[list[dict], dict]":
    """Allocate regression task assignments with controlled regime distribution.

    Returns (assignments, allocation_audit) where assignments is a list of dicts
    each containing prior_regime and sampled parameters.
    """
    if profile not in REGRESSION_PROFILES:
        raise ValueError(
            f"allocate_regression_tasks: {profile!r} is not a regression profile; "
            f"valid profiles: {REGRESSION_PROFILES}"
        )

    # Default grids
    n_grid = n_grid or [32, 64, 128, 256, 512, 1024]
    p_signal_grid = p_signal_grid or [4, 8, 16, 32, 64]
    p_noise_grid = p_noise_grid or [0, 8, 24, 56, 120]
    active_s_grid = active_s_grid or [2, 4, 8, 16, 32]
    rho_grid = rho_grid or [0.0, 0.3, 0.6, 0.9]
    target_noise_grid = target_noise_grid or [0.25, 0.5, 1.0, 2.0]
    feature_noise_grid = feature_noise_grid or [0.0, 0.05, 0.10, 0.25]

    # Legacy profile: fall through to sample_n_p only (Poisson)
    if profile == "legacy":
        rng = np.random.default_rng(seed=base_seed)
        regime_rng = np.random.default_rng(seed=base_seed ^ 0xDEADBEEF)
        weights_dict = dict(REGIME_WEIGHTS["legacy"])
        regimes = list(weights_dict.keys())
        probs = np.array([weights_dict[r] for r in regimes], dtype=float)
        probs /= probs.sum()
        assignments = []
        for _ in range(n_datasets):
            regime = str(regime_rng.choice(regimes, p=probs))
            params = sample_n_p(
                rng, profile, n_grid, p_signal_grid, p_noise_grid,
                active_s_grid, rho_grid, target_noise_grid, feature_noise_grid,
                allow_underdetermined=allow_underdetermined,
            )
            assignments.append({"prior_regime": regime, **params})
        audit: dict = {
            "allocation_mode": "legacy",
            "profile": profile,
            "regime_counts": dict(REGIME_WEIGHTS["legacy"]),
            "realized_regime_counts": {},
            "min_regime_count": min_regime_count,
            "allow_underdetermined": allow_underdetermined,
        }
        return assignments, audit

    weights = dict(REGIME_WEIGHTS[profile])
    if not allow_underdetermined:
        weights.pop("J_low_n_high_p", None)
    regimes = list(weights.keys())
    n_regimes = len(regimes)

    if allocation_mode == "weighted_quota":
        regime_counts = largest_remainder_counts(weights, n_datasets)
        # Enforce min_regime_count via _move_quota using a convergence loop so that
        # regimes drained as sources in one pass can be fixed in the next pass.
        max_iters = 3 * len(regimes) + 1
        for _iter in range(max_iters):
            any_deficit = False
            for regime_name in regimes:
                if regime_counts.get(regime_name, 0) < min_regime_count:
                    any_deficit = True
                    deficit = min_regime_count - regime_counts.get(regime_name, 0)
                    sources = [r for r in regimes if regime_counts.get(r, 0) > min_regime_count]
                    if sources and deficit > 0:
                        try:
                            _move_quota(regime_counts, regime_name, deficit, sources)
                        except ValueError:
                            if strict_coverage:
                                raise ValueError(
                                    f"allocate_regression_tasks: cannot satisfy min_regime_count={min_regime_count} "
                                    f"for regime {regime_name!r} with n_datasets={n_datasets}"
                                )
            if not any_deficit:
                break
    elif allocation_mode == "balanced":
        base_count = n_datasets // n_regimes
        regime_counts = {r: base_count for r in regimes}
        remainder = n_datasets - base_count * n_regimes
        for i, r in enumerate(regimes):
            if i < remainder:
                regime_counts[r] += 1
    elif allocation_mode in {"weighted", "curriculum"}:
        # For weighted/curriculum: will use rng.choice per slot
        probs = np.array([weights[r] for r in regimes], dtype=float)
        probs /= probs.sum()
        rng_alloc = np.random.default_rng(base_seed ^ 0xDEADBEEF)
        regime_list_alloc = [str(rng_alloc.choice(regimes, p=probs)) for _ in range(n_datasets)]
        from collections import Counter as _Counter
        regime_counts = dict(_Counter(regime_list_alloc))
        for r in regimes:
            if r not in regime_counts:
                regime_counts[r] = 0
    elif allocation_mode == "balanced_cartesian":
        # Same as balanced for regression
        base_count = n_datasets // n_regimes
        regime_counts = {r: base_count for r in regimes}
        remainder = n_datasets - base_count * n_regimes
        for i, r in enumerate(regimes):
            if i < remainder:
                regime_counts[r] += 1
    else:
        raise ValueError(f"allocate_regression_tasks: unknown allocation_mode {allocation_mode!r}")

    # Build regime_list from counts
    if allocation_mode in {"weighted", "curriculum"}:
        # Already built above as regime_list_alloc
        regime_list = regime_list_alloc
    else:
        regime_list = [
            regime for regime, count in regime_counts.items()
            for _ in range(count)
        ]

    # Sample parameters for each slot
    rng_params = np.random.default_rng(base_seed)
    assignments = []
    for regime in regime_list:
        params = sample_n_p(
            rng_params, profile, n_grid, p_signal_grid, p_noise_grid,
            active_s_grid, rho_grid, target_noise_grid, feature_noise_grid,
            allow_underdetermined=allow_underdetermined,
        )
        assignments.append({"prior_regime": regime, **params})

    # Shuffle assignments
    perm_rng = np.random.default_rng(base_seed ^ 0xDEADC0DE)
    perm = perm_rng.permutation(len(assignments))
    assignments = [assignments[i] for i in perm]

    # Compute realized regime counts
    from collections import Counter as _Counter2
    realized = dict(_Counter2(a["prior_regime"] for a in assignments))

    audit = {
        "allocation_mode": allocation_mode,
        "profile": profile,
        "regime_counts": regime_counts,
        "realized_regime_counts": realized,
        "min_regime_count": min_regime_count,
        "allow_underdetermined": allow_underdetermined,
    }

    if strict_coverage:
        for r in regimes:
            if realized.get(r, 0) < min_regime_count:
                raise ValueError(
                    f"allocate_regression_tasks: strict_coverage=True but regime {r!r} "
                    f"has only {realized.get(r, 0)} datasets (< min_regime_count={min_regime_count})"
                )

    return assignments, audit


def build_coverage_audit(
    datasets: "list[dict]",
    task_family: str,
    required_axes: "list[str]",
    min_counts: "dict[str, int]",
    strict_coverage: bool,
    allocation_mode: str,
    configured_profile_weights: "dict | None" = None,
    effective_profile_weights: "dict | None" = None,
) -> dict:
    """Build a coverage audit for a list of dataset records.

    Counts occurrences of each value per axis; reports missing and under-
    represented axis values.
    """
    realized_counts_by_axis: dict[str, dict] = {}
    for axis in required_axes:
        counts: dict = {}
        for ds in datasets:
            val = str(ds.get(axis, ""))
            counts[val] = counts.get(val, 0) + 1
        realized_counts_by_axis[axis] = counts

    missing_axis_values: dict[str, list] = {}
    underrepresented_axis_values: dict[str, list] = {}
    coverage_warnings: list[str] = []

    for axis in required_axes:
        axis_counts = realized_counts_by_axis.get(axis, {})
        min_val = min_counts.get(axis, 1)
        missing = [v for v, c in axis_counts.items() if c == 0]
        under = [v for v, c in axis_counts.items() if 0 < c < min_val]
        if missing:
            missing_axis_values[axis] = missing
        if under:
            underrepresented_axis_values[axis] = under
            coverage_warnings.append(
                f"axis={axis!r}: {len(under)} values underrepresented (count < {min_val})"
            )

    coverage_passed = len(missing_axis_values) == 0

    if strict_coverage and not coverage_passed:
        first_axis = next(iter(missing_axis_values))
        missing_vals = missing_axis_values[first_axis]
        raise ValueError(
            f"build_coverage_audit: strict_coverage=True but axis {first_axis!r} "
            f"has {len(missing_vals)} missing values: {missing_vals[:5]}"
        )

    return {
        "allocation_mode": allocation_mode,
        "strict_coverage": strict_coverage,
        "required_axes": required_axes,
        "required_min_counts": min_counts,
        "configured_profile_weights": configured_profile_weights or {},
        "effective_profile_weights": effective_profile_weights or {},
        "realized_counts_by_axis": realized_counts_by_axis,
        "missing_axis_values": missing_axis_values,
        "underrepresented_axis_values": underrepresented_axis_values,
        "coverage_passed": coverage_passed,
        "coverage_warnings": coverage_warnings,
    }


def build_memory_audit(
    datasets: "list[dict]",
    memory_guard_bytes: int,
    default_context_size: int,
    default_test_batch_size: int,
    default_feature_cap: int,
) -> dict:
    """Build a memory usage audit across all dataset records."""
    from constants import MEMORY_RISK_BUCKETS
    h_bytes_list = [
        float(ds.get("estimated_h_tensor_bytes", 0)) for ds in datasets
    ]
    bucket_list = [str(ds.get("memory_risk_bucket", "low")) for ds in datasets]

    memory_risk_counts = {bucket: 0 for bucket in MEMORY_RISK_BUCKETS}
    for b in bucket_list:
        if b in memory_risk_counts:
            memory_risk_counts[b] += 1

    exceeds_count = memory_risk_counts.get("exceeds_default_guard", 0)
    memory_passed = exceeds_count == 0

    max_bytes = max(h_bytes_list) if h_bytes_list else 0.0
    median_bytes = float(np.median(h_bytes_list)) if h_bytes_list else 0.0

    memory_warnings: list[str] = []
    if not memory_passed:
        memory_warnings.append(
            f"{exceeds_count} dataset(s) exceed the default GPU guard of {memory_guard_bytes} bytes"
        )

    return {
        "memory_guard_bytes": memory_guard_bytes,
        "default_context_size": default_context_size,
        "default_test_batch_size": default_test_batch_size,
        "default_feature_cap": default_feature_cap,
        "memory_risk_counts": memory_risk_counts,
        "exceeds_default_guard_count": exceeds_count,
        "max_estimated_h_tensor_bytes": max_bytes,
        "median_estimated_h_tensor_bytes": median_bytes,
        "memory_passed": memory_passed,
        "memory_warnings": memory_warnings,
    }


def build_difficulty_audit(
    datasets: "list[dict]",
    curriculum_policy: str,
    configured_mix: "dict[str, float]",
) -> dict:
    """Build a difficulty distribution audit across all dataset records."""
    tier_counts: dict[str, int] = {"core": 0, "robust": 0, "stress": 0}
    scores: list[float] = []

    for ds in datasets:
        tier = str(ds.get("difficulty_tier", "core"))
        if tier in tier_counts:
            tier_counts[tier] += 1
        score = ds.get("difficulty_score", 0)
        scores.append(float(score))

    total = len(datasets)
    realized_mix: dict[str, float] = {
        t: (tier_counts[t] / total if total > 0 else 0.0)
        for t in tier_counts
    }

    # Check if within ±10 pp of configured mix
    difficulty_passed = True
    difficulty_warnings: list[str] = []
    for tier, configured_frac in configured_mix.items():
        realized_frac = realized_mix.get(tier, 0.0)
        if abs(realized_frac - configured_frac) > 0.10:
            difficulty_passed = False
            difficulty_warnings.append(
                f"tier={tier!r}: configured={configured_frac:.2f}, realized={realized_frac:.2f} "
                f"(delta={abs(realized_frac - configured_frac):.2f} > 0.10)"
            )

    if scores:
        score_arr = np.array(scores)
        score_summary = {
            "mean": float(np.mean(score_arr)),
            "p25": float(np.percentile(score_arr, 25)),
            "p75": float(np.percentile(score_arr, 75)),
            "max": float(np.max(score_arr)),
        }
    else:
        score_summary = {"mean": 0.0, "p25": 0.0, "p75": 0.0, "max": 0.0}

    stress_count = tier_counts.get("stress", 0)
    stress_fraction = stress_count / total if total > 0 else 0.0

    return {
        "curriculum_policy": curriculum_policy,
        "configured_difficulty_mix": configured_mix,
        "realized_difficulty_mix": realized_mix,
        "difficulty_tier_counts": tier_counts,
        "difficulty_score_summary": score_summary,
        "stress_task_count": stress_count,
        "stress_task_fraction": stress_fraction,
        "difficulty_passed": difficulty_passed,
        "difficulty_warnings": difficulty_warnings,
    }


def build_train_eval_alignment_report(
    train_manifest: dict,
    eval_manifest: dict,
) -> dict:
    """Build an alignment report comparing train and eval manifests.

    Extracts regime sets from n_datasets_by_regime_group (regression) or
    realized_regime_counts (classification) keys.
    """
    def _extract_regimes(manifest: dict) -> set:
        # Regression: n_datasets_by_regime_group
        if "n_datasets_by_regime_group" in manifest:
            return set(manifest["n_datasets_by_regime_group"].keys())
        # Classification: realized_regime_counts
        if "realized_regime_counts" in manifest:
            return set(manifest["realized_regime_counts"].keys())
        # Fallback: scan datasets
        return {
            str(ds.get("prior_regime", ds.get("classification_regime", "")))
            for ds in manifest.get("datasets", [])
        }

    train_regimes = _extract_regimes(train_manifest)
    eval_regimes = _extract_regimes(eval_manifest)

    shared_regimes = train_regimes & eval_regimes
    train_only_regimes = train_regimes - eval_regimes
    eval_only_regimes = eval_regimes - train_regimes

    alignment_passed = len(eval_only_regimes) == 0
    alignment_warnings: list[str] = []
    if not alignment_passed:
        alignment_warnings.append(
            f"eval_only_regimes not in train: {sorted(eval_only_regimes)}"
        )

    # Compute n_distribution_overlap and p_total_distribution_overlap from grid_metadata
    train_grid = train_manifest.get("grid_metadata", {})
    eval_grid = eval_manifest.get("grid_metadata", {})
    train_n = set(train_grid.get("n_grid", []))
    eval_n = set(eval_grid.get("n_grid", []))
    train_p = set(train_grid.get("p_signal_grid", []))
    eval_p = set(eval_grid.get("p_signal_grid", []))

    n_distribution_overlap = sorted(train_n & eval_n)
    p_total_distribution_overlap = sorted(train_p & eval_p)

    # Difficulty tier overlap from difficulty_audit sections
    train_diff = train_manifest.get("difficulty_audit", {})
    eval_diff = eval_manifest.get("difficulty_audit", {})
    train_tiers = set(k for k, v in train_diff.get("difficulty_tier_counts", {}).items() if v > 0)
    eval_tiers = set(k for k, v in eval_diff.get("difficulty_tier_counts", {}).items() if v > 0)
    difficulty_tier_overlap = sorted(train_tiers & eval_tiers)

    return {
        "train_regimes": sorted(train_regimes),
        "eval_regimes": sorted(eval_regimes),
        "shared_regimes": sorted(shared_regimes),
        "train_only_regimes": sorted(train_only_regimes),
        "eval_only_regimes": sorted(eval_only_regimes),
        "alignment_passed": alignment_passed,
        "alignment_warnings": alignment_warnings,
        "n_distribution_overlap": n_distribution_overlap,
        "p_total_distribution_overlap": p_total_distribution_overlap,
        "difficulty_tier_overlap": difficulty_tier_overlap,
    }


def validate_controlled_manifest(manifest: dict) -> None:
    """Validate that a manifest has all required controlled suite sections.

    Raises ValueError with a descriptive message if any check fails.
    Never silently passes on error.
    """
    required_top_level = {
        "coverage_audit", "memory_audit", "difficulty_audit",
        "alignment_audit", "generation_controls",
    }
    for section in required_top_level:
        if section not in manifest:
            raise ValueError(
                f"validate_controlled_manifest: missing required top-level section "
                f"{section!r}; found keys: {sorted(manifest.keys())}"
            )

    required_per_dataset_fields = {
        "difficulty_score", "difficulty_tier", "difficulty_reasons",
        "estimated_h_tensor_bytes", "memory_risk_bucket", "exceeds_default_gpu_guard",
    }
    for ds in manifest.get("datasets", []):
        ds_id = ds.get("dataset_id", ds.get("global_idx", "?"))
        for field in required_per_dataset_fields:
            if field not in ds:
                raise ValueError(
                    f"validate_controlled_manifest: dataset_id={ds_id!r} missing "
                    f"required field {field!r}"
                )

    # difficulty_audit tier counts must sum to len(datasets)
    n_datasets = len(manifest.get("datasets", []))
    diff_audit = manifest.get("difficulty_audit", {})
    tier_counts = diff_audit.get("difficulty_tier_counts", {})
    tier_total = sum(tier_counts.values())
    if tier_total != n_datasets:
        raise ValueError(
            f"validate_controlled_manifest: difficulty_audit.difficulty_tier_counts "
            f"sum={tier_total} != len(datasets)={n_datasets}"
        )

    # memory_audit risk counts must sum to len(datasets)
    mem_audit = manifest.get("memory_audit", {})
    risk_counts = mem_audit.get("memory_risk_counts", {})
    risk_total = sum(risk_counts.values())
    if risk_total != n_datasets:
        raise ValueError(
            f"validate_controlled_manifest: memory_audit.memory_risk_counts "
            f"sum={risk_total} != len(datasets)={n_datasets}"
        )


# ---------------------------------------------------------------------------
# 17. Per-split K coverage validation (F1)
# ---------------------------------------------------------------------------

def validate_per_split_k_coverage(
    datasets: list[dict],
    required_k: list[int],
    strict_coverage: bool = False,
) -> dict:
    """Validate that every required K value appears at least once across datasets.

    Parameters
    ----------
    datasets : list of per-dataset dicts, each must have a ``num_classes`` field.
    required_k : list of K values that must be present (e.g., [2, 3, 5, 10]).
    strict_coverage : if True, raise ValueError when any required K is absent.

    Returns
    -------
    dict with keys:
        realized_k_counts : dict[int, int] — actual counts per K value
        required_k        : list[int]
        missing_k         : list[int] of K values with count == 0
        coverage_passed   : bool — True when all required_k are present
    """
    realized: dict[int, int] = {}
    for ds in datasets:
        k = int(ds.get("num_classes", 0))
        realized[k] = realized.get(k, 0) + 1

    missing_k = sorted(k for k in required_k if realized.get(k, 0) == 0)
    coverage_passed = len(missing_k) == 0

    if strict_coverage and not coverage_passed:
        raise ValueError(
            f"validate_per_split_k_coverage: missing required K values {missing_k}. "
            f"Realized K counts: {realized}. "
            "Add datasets covering the missing K values or pass strict_coverage=False."
        )

    return {
        "realized_k_counts": {int(k): int(v) for k, v in sorted(realized.items())},
        "required_k":        sorted(required_k),
        "missing_k":         missing_k,
        "coverage_passed":   coverage_passed,
    }


# ---------------------------------------------------------------------------
# 18. Paired robustness task builder (F5)
# ---------------------------------------------------------------------------

def build_paired_robustness_tasks(
    base_params: dict,
    sweep_param: str,
    sweep_values: list,
    base_seed: int,
) -> list[dict]:
    """Build a list of paired robustness task descriptors sharing a common seed.

    All tasks in the returned list share the same ``pair_id`` UUID so downstream
    analysis can match them across sweep points. The first entry is flagged as
    the reference (``is_reference=True``).

    Parameters
    ----------
    base_params : dict of DGP parameters shared across all sweep points.
    sweep_param : one of ``"feature_noise_level"``, ``"label_noise_rate"``,
        ``"p_noise"``, ``"p_signal"``, ``"n_train"``.
    sweep_values : list of values to sweep; the first value is treated as the
        reference condition.
    base_seed : integer seed shared across all tasks (same latent task).

    Returns
    -------
    list[dict] with one entry per sweep value, each containing all ``base_params``
    keys plus ``sweep_param``, ``sweep_value``, ``pair_id``, ``is_reference``,
    and ``base_seed``.
    """
    import uuid

    _ALLOWED = {
        "feature_noise_level", "label_noise_rate",
        "p_noise", "p_signal", "n_train",
    }
    if sweep_param not in _ALLOWED:
        raise ValueError(
            f"build_paired_robustness_tasks: sweep_param={sweep_param!r} is not "
            f"in the allowed set {sorted(_ALLOWED)}."
        )
    if not sweep_values:
        raise ValueError("build_paired_robustness_tasks: sweep_values must be non-empty.")

    pair_id = str(uuid.uuid4())
    tasks = []
    for i, val in enumerate(sweep_values):
        task = dict(base_params)
        task[sweep_param] = val
        task["sweep_param"] = sweep_param
        task["sweep_value"] = val
        task["pair_id"] = pair_id
        task["is_reference"] = (i == 0)
        task["base_seed"] = int(base_seed)
        tasks.append(task)
    return tasks

# ---------------------------------------------------------------------------
# Mixed-Categorical DGP (Step 3 additions)
# ---------------------------------------------------------------------------

# Add new profiles to the existing lists
REGRESSION_PROFILES.append("linear_regression_mixed_categorical_stat_aware")
CLASSIFICATION_PROFILES.append("linear_classification_mixed_categorical_stat_aware")
ALL_PROFILES.append("linear_regression_mixed_categorical_stat_aware")
ALL_PROFILES.append("linear_classification_mixed_categorical_stat_aware")

# Add regime weights for new profiles (same regime weights as pure numeric equivalents)
REGIME_WEIGHTS["linear_regression_mixed_categorical_stat_aware"] = dict(
    REGIME_WEIGHTS["linear_stat_aware"]
)
REGIME_WEIGHTS["linear_classification_mixed_categorical_stat_aware"] = dict(
    REGIME_WEIGHTS["linear_classification_stat_aware"]
)


# ---------------------------------------------------------------------------
# 3b. Categorical feature generation helpers
# ---------------------------------------------------------------------------

def generate_categorical_features(
    rng: np.random.Generator,
    n: int,
    cardinalities: list[int],
    imbalance_type: str = "balanced",
) -> np.ndarray:
    """Generate integer category IDs for mixed-categorical DGP.

    Returns X_cat (n, p_cat) dtype int64.
    IDs are in [ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + C - 1].
    Never returns IDs 0 (PAD), 1 (MISSING), or 2 (UNKNOWN).
    """
    from constants import ENTITY_EMBED_FIRST_REAL_ID
    p_cat = len(cardinalities)
    X_cat = np.zeros((n, p_cat), dtype=np.int64)
    _IMBALANCE_ALPHA = {
        "balanced": None,
        "mild": 2.0,
        "moderate": 1.0,
        "severe": 0.5,
    }
    alpha = _IMBALANCE_ALPHA.get(imbalance_type, None)
    for j, cardinality in enumerate(cardinalities):
        if cardinality < 1:
            raise ValueError(f"cardinality must be >= 1, got {cardinality}")
        if alpha is None:
            probs = np.ones(cardinality, dtype=float) / cardinality
        else:
            raw = rng.dirichlet(alpha * np.ones(cardinality))
            probs = raw / raw.sum()
        raw_ids = rng.choice(cardinality, size=n, p=probs)  # 0-indexed
        X_cat[:, j] = raw_ids + ENTITY_EMBED_FIRST_REAL_ID
    return X_cat


def generate_categorical_effects_regression(
    rng: np.random.Generator,
    cardinalities: list[int],
    effect_scale: float,
    active_mask: list[bool],
) -> list[np.ndarray]:
    """Generate scalar effect vectors per categorical feature (regression).

    Signal features: effects ~ N(0, effect_scale**2), mean-centered.
    Noise features: effects = 0.
    Returns cat_effects[j] shape (cardinality_j,) for j in 0..p_cat-1.
    """
    cat_effects = []
    for j, cardinality in enumerate(cardinalities):
        if active_mask[j]:
            effects = rng.normal(0.0, effect_scale, size=cardinality)
            effects = effects - effects.mean()
        else:
            effects = np.zeros(cardinality, dtype=float)
        cat_effects.append(effects.astype(np.float64))
    return cat_effects


def generate_categorical_class_effects(
    rng: np.random.Generator,
    cardinalities: list[int],
    num_classes: int,
    effect_scale: float,
    active_mask: list[bool],
) -> list[np.ndarray]:
    """Generate per-class effect matrices per categorical feature (classification).

    Returns cat_class_effects[j] shape (cardinality_j, K).
    Column 0 = reference class = 0 (identifiability constraint).
    Inactive features: all-zero.
    """
    cat_class_effects = []
    K = int(num_classes)
    for j, cardinality in enumerate(cardinalities):
        if active_mask[j]:
            effects = rng.normal(0.0, effect_scale, size=(cardinality, K))
            effects[:, 0] = 0.0  # reference class = 0
        else:
            effects = np.zeros((cardinality, K), dtype=float)
        cat_class_effects.append(effects.astype(np.float64))
    return cat_class_effects


def apply_category_missing(
    rng: np.random.Generator,
    X_cat: np.ndarray,
    missing_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace sampled cells with ENTITY_EMBED_MISSING_ID (1).

    Returns (X_cat_masked, missing_mask) where missing_mask dtype bool.
    """
    from constants import ENTITY_EMBED_MISSING_ID
    if missing_rate <= 0.0:
        return X_cat.copy(), np.zeros(X_cat.shape, dtype=bool)
    missing_mask = rng.random(X_cat.shape) < missing_rate
    X_cat_masked = X_cat.copy()
    X_cat_masked[missing_mask] = ENTITY_EMBED_MISSING_ID
    return X_cat_masked, missing_mask


def mark_unseen_query_categories(
    X_cat_train: np.ndarray,
    X_cat_test: np.ndarray,
) -> np.ndarray:
    """Return unknown_mask (n_test, p_cat) bool.

    True where a query category does not appear in the corresponding training column.
    Does NOT modify X_cat_test in place.
    """
    from constants import ENTITY_EMBED_MISSING_ID
    n_test, p_cat = X_cat_test.shape
    unknown_mask = np.zeros((n_test, p_cat), dtype=bool)
    for j in range(p_cat):
        train_vocab = set(X_cat_train[:, j].tolist())
        # MISSING_ID is never "unknown" -- it has its own special embedding
        train_vocab.discard(ENTITY_EMBED_MISSING_ID)
        for i in range(n_test):
            cid = int(X_cat_test[i, j])
            if cid != ENTITY_EMBED_MISSING_ID and cid not in train_vocab:
                unknown_mask[i, j] = True
    return unknown_mask


# ---------------------------------------------------------------------------
# 3b. Dataset builders
# ---------------------------------------------------------------------------

def build_mixed_regression_dataset(
    rng: np.random.Generator,
    regime: str,
    params: dict,
) -> dict:
    """Build one mixed-categorical regression task.

    Formula:
      y = X_num_clean @ beta_num
        + sum_j cat_effects[j][X_cat_clean[:, j] - FIRST_REAL_ID]
        + epsilon

    Returns dict with all schema fields for MIXED_REG_DGP_SCHEMA_VERSION.
    """
    from constants import (
        ENTITY_EMBED_FIRST_REAL_ID,
        ENTITY_EMBED_MISSING_ID,
        MIXED_CAT_REGRESSION_TRAINING_FAMILY,
        MIXED_REG_DGP_SCHEMA_VERSION,
    )

    n = int(params["n"])
    p_num_signal = int(params.get("p_num_signal", 4))
    p_num_noise = int(params.get("p_num_noise", 0))
    p_num = p_num_signal + p_num_noise
    p_cat_signal = int(params.get("p_cat_signal", 2))
    p_cat_noise = int(params.get("p_cat_noise", 0))
    p_cat = p_cat_signal + p_cat_noise
    cardinalities = list(params.get("cardinalities", [5] * p_cat))
    if len(cardinalities) != p_cat:
        raise ValueError(f"len(cardinalities)={len(cardinalities)} != p_cat={p_cat}")
    cat_effect_scale = float(params.get("cat_effect_scale", 1.0))
    missing_rate = float(params.get("missing_rate", 0.0))
    target_noise_scale = float(params.get("target_noise_scale", 1.0))
    rho = float(params.get("rho", 0.0))
    imbalance_type = str(params.get("imbalance_type", "balanced"))

    # Generate numeric features (iid Gaussian for simplicity;
    # more complex covariance can be added via regime-level dispatch)
    if abs(rho) > 1e-8:
        cov = (1 - rho) * np.eye(p_num) + rho * np.ones((p_num, p_num))
        X_num = rng.multivariate_normal(np.zeros(p_num), cov, size=n)
    else:
        X_num = rng.standard_normal((n, p_num))

    # Generate numeric beta
    beta_num = np.zeros(p_num, dtype=float)
    if p_num_signal > 0:
        beta_num[:p_num_signal] = rng.standard_normal(p_num_signal)

    # Generate categorical features
    X_cat_clean = generate_categorical_features(rng, n, cardinalities, imbalance_type)

    # Generate categorical effects
    active_mask_cat = [j < p_cat_signal for j in range(p_cat)]
    cat_effects = generate_categorical_effects_regression(
        rng, cardinalities, cat_effect_scale, active_mask_cat
    )

    # Compute target (using clean X_cat)
    y_num = X_num @ beta_num
    y_cat = np.zeros(n, dtype=float)
    for j in range(p_cat):
        for i in range(n):
            cid = int(X_cat_clean[i, j]) - ENTITY_EMBED_FIRST_REAL_ID
            if 0 <= cid < len(cat_effects[j]):
                y_cat[i] += cat_effects[j][cid]
    epsilon = rng.standard_normal(n) * target_noise_scale
    y = y_num + y_cat + epsilon

    # Apply missing values to X_cat
    X_cat_masked, missing_mask = apply_category_missing(rng, X_cat_clean, missing_rate)

    return {
        "X_num": X_num.astype(np.float64),
        "X_cat": X_cat_masked.astype(np.int64),
        "X_cat_clean": X_cat_clean.astype(np.int64),
        "missing_mask": missing_mask,
        "y": y.astype(np.float64),
        "beta_num": beta_num.astype(np.float64),
        "cat_effects": cat_effects,
        "n": n,
        "p_num": p_num,
        "p_cat": p_cat,
        "p_num_signal": p_num_signal,
        "p_num_noise": p_num_noise,
        "p_cat_signal": p_cat_signal,
        "p_cat_noise": p_cat_noise,
        "cardinalities": cardinalities,
        "cat_effect_scale": cat_effect_scale,
        "missing_rate": missing_rate,
        "target_noise_scale": target_noise_scale,
        "rho": rho,
        "prior_regime": regime,
        "imbalance_type": imbalance_type,
        "schema_version": MIXED_REG_DGP_SCHEMA_VERSION,
        "training_data_family": MIXED_CAT_REGRESSION_TRAINING_FAMILY,
        "task_objective": "inductive_regression",
        "task_family": "linear_regression",
        "numeric_support_mask": (beta_num != 0).tolist(),
        "cat_support_mask": active_mask_cat,
    }


def build_mixed_classification_dataset(
    rng: np.random.Generator,
    regime: str,
    params: dict,
) -> dict:
    """Build one mixed-categorical classification task.

    Formula:
      logits = X_num_clean @ W_num + b
             + sum_j cat_class_effects[j][X_cat_clean[:, j] - FIRST_REAL_ID, :]
      probs = softmax(logits / temperature)
      y ~ Categorical(probs)
    """
    from constants import (
        ENTITY_EMBED_FIRST_REAL_ID,
        MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
        MIXED_CLS_DGP_SCHEMA_VERSION,
    )

    n = int(params["n"])
    p_num_signal = int(params.get("p_num_signal", 4))
    p_num_noise = int(params.get("p_num_noise", 0))
    p_num = p_num_signal + p_num_noise
    p_cat_signal = int(params.get("p_cat_signal", 2))
    p_cat_noise = int(params.get("p_cat_noise", 0))
    p_cat = p_cat_signal + p_cat_noise
    num_classes = int(params.get("num_classes", 2))
    cardinalities = list(params.get("cardinalities", [5] * p_cat))
    if len(cardinalities) != p_cat:
        raise ValueError(f"len(cardinalities)={len(cardinalities)} != p_cat={p_cat}")
    cat_effect_scale = float(params.get("cat_effect_scale", 1.0))
    missing_rate = float(params.get("missing_rate", 0.0))
    temperature = float(params.get("temperature", 1.0))
    rho = float(params.get("rho", 0.0))
    imbalance_type = str(params.get("imbalance_type", "balanced"))
    label_noise_rate = float(params.get("label_noise_rate", 0.0))
    coefficient_scale = float(params.get("coefficient_scale", 1.0))

    K = num_classes

    # Generate numeric features
    if abs(rho) > 1e-8:
        cov = (1 - rho) * np.eye(p_num) + rho * np.ones((p_num, p_num))
        X_num = rng.multivariate_normal(np.zeros(p_num), cov, size=n)
    else:
        X_num = rng.standard_normal((n, p_num))

    # Generate weight matrix W_num (p_num, K); first column = 0 (reference)
    W_num = np.zeros((p_num, K), dtype=float)
    if p_num_signal > 0 and K > 1:
        W_num[:p_num_signal, 1:] = rng.standard_normal((p_num_signal, K - 1)) * coefficient_scale
    # b (K,); b[0] = 0
    b = np.zeros(K, dtype=float)

    # Generate categorical features
    X_cat_clean = generate_categorical_features(rng, n, cardinalities, imbalance_type)

    # Generate categorical class effects
    active_mask_cat = [j < p_cat_signal for j in range(p_cat)]
    cat_class_effects = generate_categorical_class_effects(
        rng, cardinalities, K, cat_effect_scale, active_mask_cat
    )

    # Compute logits using clean X_cat
    logits = X_num @ W_num + b  # (n, K)
    for j in range(p_cat):
        for i in range(n):
            cid = int(X_cat_clean[i, j]) - ENTITY_EMBED_FIRST_REAL_ID
            if 0 <= cid < len(cat_class_effects[j]):
                logits[i] += cat_class_effects[j][cid]

    # Softmax with temperature
    logits_scaled = logits / max(temperature, 1e-8)
    logits_shifted = logits_scaled - logits_scaled.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Sample labels
    y = np.array([rng.choice(K, p=probs[i]) for i in range(n)], dtype=np.int64)

    # Apply label noise
    if label_noise_rate > 0.0:
        noise_mask = rng.random(n) < label_noise_rate
        y[noise_mask] = rng.integers(0, K, size=noise_mask.sum())

    # Apply missing values to X_cat
    X_cat_masked, missing_mask = apply_category_missing(rng, X_cat_clean, missing_rate)

    # Ensure all classes present in y (at least 1 per class)
    for k in range(K):
        if not np.any(y == k):
            y[rng.integers(0, n)] = k

    return {
        "X_num": X_num.astype(np.float64),
        "X_cat": X_cat_masked.astype(np.int64),
        "X_cat_clean": X_cat_clean.astype(np.int64),
        "missing_mask": missing_mask,
        "y": y.astype(np.int64),
        "W_num": W_num.astype(np.float64),
        "b": b.astype(np.float64),
        "cat_class_effects": cat_class_effects,
        "logits": logits.astype(np.float64),
        "probs": probs.astype(np.float64),
        "num_classes": K,
        "n": n,
        "p_num": p_num,
        "p_cat": p_cat,
        "p_num_signal": p_num_signal,
        "p_num_noise": p_num_noise,
        "p_cat_signal": p_cat_signal,
        "p_cat_noise": p_cat_noise,
        "cardinalities": cardinalities,
        "cat_effect_scale": cat_effect_scale,
        "missing_rate": missing_rate,
        "temperature": temperature,
        "rho": rho,
        "prior_regime": regime,
        "imbalance_type": imbalance_type,
        "label_noise_rate": label_noise_rate,
        "coefficient_scale": coefficient_scale,
        "schema_version": MIXED_CLS_DGP_SCHEMA_VERSION,
        "training_data_family": MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
        "task_objective": "inductive_classification",
        "task_family": "linear_classification",
        "cat_support_mask": active_mask_cat,
        "numeric_support_mask": (W_num.any(axis=1)).tolist(),
    }


# ---------------------------------------------------------------------------
# 3b. Validation helpers
# ---------------------------------------------------------------------------

def validate_mixed_regression_dataset(ds: dict) -> None:
    """Assert schema contract for MIXED_REG_DGP_SCHEMA_VERSION."""
    from constants import (
        ENTITY_EMBED_FIRST_REAL_ID,
        ENTITY_EMBED_MISSING_ID,
        ENTITY_EMBED_PAD_ID,
        ENTITY_EMBED_UNKNOWN_ID,
        MIXED_CAT_REGRESSION_TRAINING_FAMILY,
        MIXED_REG_DGP_SCHEMA_VERSION,
    )
    _sv = ds.get("schema_version")
    assert _sv == MIXED_REG_DGP_SCHEMA_VERSION, (
        f"Expected schema_version={MIXED_REG_DGP_SCHEMA_VERSION!r}, got {_sv!r}"
    )
    assert ds.get("training_data_family") == MIXED_CAT_REGRESSION_TRAINING_FAMILY
    X_num = ds["X_num"]
    X_cat = ds["X_cat"]
    n = int(ds["n"])
    p_num = int(ds["p_num"])
    p_cat = int(ds["p_cat"])
    assert X_num.shape == (n, p_num), f"X_num shape mismatch: {X_num.shape} != ({n}, {p_num})"
    assert X_cat.shape == (n, p_cat), f"X_cat shape mismatch: {X_cat.shape} != ({n}, {p_cat})"
    assert X_cat.dtype == np.int64, f"X_cat must be int64, got {X_cat.dtype}"
    # PAD (0) and UNKNOWN (2) must never appear
    for forbidden in (ENTITY_EMBED_PAD_ID, ENTITY_EMBED_UNKNOWN_ID):
        assert not np.any(X_cat == forbidden), (
            f"Forbidden token {forbidden} found in X_cat"
        )
    # All IDs must be MISSING_ID or >= FIRST_REAL_ID
    cardinalities = ds["cardinalities"]
    for j, cardinality in enumerate(cardinalities):
        col = X_cat[:, j]
        valid_real = (col >= ENTITY_EMBED_FIRST_REAL_ID) & (
            col < ENTITY_EMBED_FIRST_REAL_ID + cardinality
        )
        valid_missing = col == ENTITY_EMBED_MISSING_ID
        assert np.all(valid_real | valid_missing), (
            f"Column {j} has out-of-range IDs"
        )
    # Noise numeric features must have zero beta
    beta_num = np.array(ds["beta_num"])
    p_num_signal = int(ds["p_num_signal"])
    assert np.all(beta_num[p_num_signal:] == 0), "Noise numeric features must have beta=0"
    # Noise categorical features must have zero effects
    cat_effects = ds["cat_effects"]
    cat_support = ds["cat_support_mask"]
    for j, (eff, active) in enumerate(zip(cat_effects, cat_support)):
        if not active:
            assert np.all(np.array(eff) == 0), f"Noise cat feature {j} has non-zero effects"


def validate_mixed_classification_dataset(ds: dict) -> None:
    """Assert schema contract for MIXED_CLS_DGP_SCHEMA_VERSION."""
    from constants import (
        ENTITY_EMBED_FIRST_REAL_ID,
        ENTITY_EMBED_MISSING_ID,
        ENTITY_EMBED_PAD_ID,
        ENTITY_EMBED_UNKNOWN_ID,
        MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
        MIXED_CLS_DGP_SCHEMA_VERSION,
    )
    assert ds.get("schema_version") == MIXED_CLS_DGP_SCHEMA_VERSION
    assert ds.get("training_data_family") == MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY
    X_num = ds["X_num"]
    X_cat = ds["X_cat"]
    n = int(ds["n"])
    p_num = int(ds["p_num"])
    p_cat = int(ds["p_cat"])
    K = int(ds["num_classes"])
    assert X_num.shape == (n, p_num)
    assert X_cat.shape == (n, p_cat)
    assert X_cat.dtype == np.int64
    # PAD and UNKNOWN forbidden
    for forbidden in (ENTITY_EMBED_PAD_ID, ENTITY_EMBED_UNKNOWN_ID):
        assert not np.any(X_cat == forbidden)
    # Reference class identifiability
    W_num = np.array(ds["W_num"])
    b = np.array(ds["b"])
    assert W_num.shape == (p_num, K)
    assert b.shape == (K,)
    assert b[0] == 0, "b[0] must be 0 (reference class)"
    assert np.all(W_num[:, 0] == 0), "W_num[:, 0] must be 0 (reference class)"
    # Cat class effects reference class
    cat_class_effects = ds["cat_class_effects"]
    for j, eff in enumerate(cat_class_effects):
        eff_arr = np.array(eff)
        assert np.all(eff_arr[:, 0] == 0), f"cat_class_effects[{j}][:, 0] must be 0"
    # All K classes present in y
    y = np.array(ds["y"])
    for k in range(K):
        assert np.any(y == k), f"Class {k} not present in y"


# ---------------------------------------------------------------------------
# 3b. Parquet writers
# ---------------------------------------------------------------------------

def write_parquet_mixed_regression_dgp(
    ds_split: dict,
    filepath: str,
    *,
    store_cat_params: bool = True,
) -> None:
    """Write one mixed-categorical regression meta-task to Parquet."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def _arr1d_f64(v):
        return pa.array([np.asarray(v, dtype=np.float64).tolist()],
                        type=pa.list_(pa.float64()))

    def _arr2d_f64(v):
        return pa.array([np.asarray(v, dtype=np.float64).tolist()],
                        type=pa.list_(pa.list_(pa.float64())))

    def _arr1d_i64(v):
        return pa.array([np.asarray(v, dtype=np.int64).tolist()],
                        type=pa.list_(pa.int64()))

    def _arr2d_i64(v):
        return pa.array([np.asarray(v, dtype=np.int64).tolist()],
                        type=pa.list_(pa.list_(pa.int64())))

    def _arr2d_bool(v):
        return pa.array([np.asarray(v, dtype=bool).tolist()],
                        type=pa.list_(pa.list_(pa.bool_())))

    columns = {
        # Core numeric arrays
        "X_num_train": _arr2d_f64(ds_split["X_num_train"]),
        "X_num_test":  _arr2d_f64(ds_split["X_num_test"]),
        "y_train":     _arr1d_f64(ds_split["y_train"]),
        "y_test":      _arr1d_f64(ds_split["y_test"]),
        # Core categorical arrays
        "X_cat_train":   _arr2d_i64(ds_split["X_cat_train"]),
        "X_cat_test":    _arr2d_i64(ds_split["X_cat_test"]),
        # Masks
        "cat_missing_mask_train": _arr2d_bool(ds_split["cat_missing_mask_train"]),
        "cat_missing_mask_test":  _arr2d_bool(ds_split["cat_missing_mask_test"]),
        "cat_unknown_mask_test":  _arr2d_bool(ds_split["cat_unknown_mask_test"]),
        # Feature type metadata
        "categorical_cardinalities": _arr1d_i64(ds_split["categorical_cardinalities"]),
        # Scalar metadata
        "n":                pa.array([int(ds_split["n"])],          type=pa.int64()),
        "p_num":            pa.array([int(ds_split["p_num"])],       type=pa.int64()),
        "p_cat":            pa.array([int(ds_split["p_cat"])],       type=pa.int64()),
        "n_train":          pa.array([int(ds_split["n_train"])],     type=pa.int64()),
        "n_test":           pa.array([int(ds_split["n_test"])],      type=pa.int64()),
        "prior_regime":     pa.array([str(ds_split["prior_regime"])], type=pa.utf8()),
        "schema_version":   pa.array([str(ds_split["schema_version"])], type=pa.utf8()),
        "task_family":      pa.array([str(ds_split.get("task_family", "linear_regression"))], type=pa.utf8()),
        "training_data_family": pa.array([str(ds_split["training_data_family"])], type=pa.utf8()),
        "task_objective":   pa.array([str(ds_split.get("task_objective", "inductive_regression"))], type=pa.utf8()),
    }

    if store_cat_params:
        columns["beta_num"] = _arr1d_f64(ds_split["beta_num"])
        columns["numeric_support_mask"] = pa.array(
            [list(map(bool, ds_split["numeric_support_mask"]))],
            type=pa.list_(pa.bool_())
        )
        columns["cat_support_mask"] = pa.array(
            [list(map(bool, ds_split["cat_support_mask"]))],
            type=pa.list_(pa.bool_())
        )
        # cat_effects: list of arrays, store as variable-length list of lists
        cat_effects_serialized = [
            np.asarray(eff, dtype=np.float64).tolist()
            for eff in ds_split["cat_effects"]
        ]
        columns["cat_effects"] = pa.array(
            [cat_effects_serialized],
            type=pa.list_(pa.list_(pa.float64()))
        )

    table = pa.table(columns)
    pq.write_table(table, filepath)


def write_parquet_mixed_classification_dgp(
    ds_split: dict,
    filepath: str,
    *,
    store_class_cat_params: bool = True,
) -> None:
    """Write one mixed-categorical classification meta-task to Parquet."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def _arr1d_f64(v):
        return pa.array([np.asarray(v, dtype=np.float64).tolist()],
                        type=pa.list_(pa.float64()))

    def _arr2d_f64(v):
        return pa.array([np.asarray(v, dtype=np.float64).tolist()],
                        type=pa.list_(pa.list_(pa.float64())))

    def _arr1d_i64(v):
        return pa.array([np.asarray(v, dtype=np.int64).tolist()],
                        type=pa.list_(pa.int64()))

    def _arr2d_i64(v):
        return pa.array([np.asarray(v, dtype=np.int64).tolist()],
                        type=pa.list_(pa.list_(pa.int64())))

    def _arr2d_bool(v):
        return pa.array([np.asarray(v, dtype=bool).tolist()],
                        type=pa.list_(pa.list_(pa.bool_())))

    columns = {
        # Core arrays
        "X_num_train": _arr2d_f64(ds_split["X_num_train"]),
        "X_num_test":  _arr2d_f64(ds_split["X_num_test"]),
        "y_train":     _arr1d_i64(ds_split["y_train"]),
        "y_test":      _arr1d_i64(ds_split["y_test"]),
        "X_cat_train": _arr2d_i64(ds_split["X_cat_train"]),
        "X_cat_test":  _arr2d_i64(ds_split["X_cat_test"]),
        # Masks
        "cat_missing_mask_train": _arr2d_bool(ds_split["cat_missing_mask_train"]),
        "cat_missing_mask_test":  _arr2d_bool(ds_split["cat_missing_mask_test"]),
        "cat_unknown_mask_test":  _arr2d_bool(ds_split["cat_unknown_mask_test"]),
        # Feature metadata
        "categorical_cardinalities": _arr1d_i64(ds_split["categorical_cardinalities"]),
        # Scalar metadata
        "n":             pa.array([int(ds_split["n"])],          type=pa.int64()),
        "p_num":         pa.array([int(ds_split["p_num"])],       type=pa.int64()),
        "p_cat":         pa.array([int(ds_split["p_cat"])],       type=pa.int64()),
        "n_train":       pa.array([int(ds_split["n_train"])],     type=pa.int64()),
        "n_test":        pa.array([int(ds_split["n_test"])],      type=pa.int64()),
        "num_classes":   pa.array([int(ds_split["num_classes"])], type=pa.int64()),
        "prior_regime":  pa.array([str(ds_split["prior_regime"])], type=pa.utf8()),
        "schema_version": pa.array([str(ds_split["schema_version"])], type=pa.utf8()),
        "task_family":   pa.array([str(ds_split.get("task_family", "linear_classification"))], type=pa.utf8()),
        "training_data_family": pa.array([str(ds_split["training_data_family"])], type=pa.utf8()),
        "task_objective": pa.array([str(ds_split.get("task_objective", "inductive_classification"))], type=pa.utf8()),
        "temperature":   pa.array([float(ds_split.get("temperature", 1.0))], type=pa.float64()),
        "label_noise_rate": pa.array([float(ds_split.get("label_noise_rate", 0.0))], type=pa.float64()),
        "class_imbalance_type": pa.array([str(ds_split.get("imbalance_type", "balanced"))], type=pa.utf8()),
    }

    if store_class_cat_params:
        columns["W_num"] = _arr2d_f64(ds_split["W_num"])
        columns["b"] = _arr1d_f64(ds_split["b"])
        columns["cat_support_mask"] = pa.array(
            [list(map(bool, ds_split["cat_support_mask"]))],
            type=pa.list_(pa.bool_())
        )
        columns["numeric_support_mask"] = pa.array(
            [list(map(bool, ds_split["numeric_support_mask"]))],
            type=pa.list_(pa.bool_())
        )
        # cat_class_effects: list[array(C, K)] -> list[list[list[float]]]
        cat_cls_ser = [
            np.asarray(eff, dtype=np.float64).tolist()
            for eff in ds_split["cat_class_effects"]
        ]
        columns["cat_class_effects"] = pa.array(
            [cat_cls_ser],
            type=pa.list_(pa.list_(pa.list_(pa.float64())))
        )

    table = pa.table(columns)
    pq.write_table(table, filepath)


# ---------------------------------------------------------------------------
# 3c. Allocation helpers
# ---------------------------------------------------------------------------

def allocate_mixed_regression_tasks(
    n_datasets: int,
    profile: str,
    base_seed: int,
    *,
    n_grid: "list[int] | None" = None,
    p_num_signal_grid: "list[int] | None" = None,
    p_num_noise_grid: "list[int] | None" = None,
    p_cat_signal_grid: "list[int] | None" = None,
    p_cat_noise_grid: "list[int] | None" = None,
    cat_cardinality_grid: "list[int] | None" = None,
    cat_effect_scale_grid: "list[float] | None" = None,
    cat_missing_rate_grid: "list[float] | None" = None,
    cat_imbalance_grid: "list[str] | None" = None,
    target_noise_grid: "list[float] | None" = None,
    rho_grid: "list[float] | None" = None,
    allocation_mode: str = "weighted_quota",
    min_regime_count: int = 10,
    allow_underdetermined: bool = False,
) -> "tuple[list[dict], dict]":
    """Allocate mixed-regression task assignments.

    Returns (assignments, allocation_audit).
    """
    if profile not in REGRESSION_PROFILES:
        raise ValueError(
            f"allocate_mixed_regression_tasks: {profile!r} is not a regression profile"
        )
    n_grid = n_grid or [32, 64, 128, 256, 512, 1024]
    p_num_signal_grid = p_num_signal_grid or [2, 4, 8, 16, 32]
    p_num_noise_grid = p_num_noise_grid or [0, 8, 24, 56]
    p_cat_signal_grid = p_cat_signal_grid or [1, 2, 4, 8]
    p_cat_noise_grid = p_cat_noise_grid or [0, 2, 4, 8]
    cat_cardinality_grid = cat_cardinality_grid or [2, 3, 5, 10, 20, 50]
    cat_effect_scale_grid = cat_effect_scale_grid or [0.25, 0.5, 1.0, 2.0]
    cat_missing_rate_grid = cat_missing_rate_grid or [0.0, 0.01, 0.05, 0.10]
    cat_imbalance_grid = cat_imbalance_grid or ["balanced", "mild", "moderate", "severe"]
    target_noise_grid = target_noise_grid or [0.25, 0.5, 1.0, 2.0]
    rho_grid = rho_grid or [0.0, 0.3, 0.6, 0.9]

    weights = dict(REGIME_WEIGHTS[profile])
    if not allow_underdetermined:
        weights.pop("J_low_n_high_p", None)
    regimes = list(weights.keys())

    regime_counts = largest_remainder_counts(weights, n_datasets)

    rng = np.random.default_rng(seed=base_seed)
    assignments = []
    for regime, count in regime_counts.items():
        for _ in range(count):
            n = int(rng.choice(n_grid))
            p_num_signal = int(rng.choice(p_num_signal_grid))
            p_num_noise = int(rng.choice(p_num_noise_grid))
            p_cat_signal = int(rng.choice(p_cat_signal_grid))
            p_cat_noise = int(rng.choice(p_cat_noise_grid))
            p_cat = p_cat_signal + p_cat_noise
            cardinalities = [int(rng.choice(cat_cardinality_grid)) for _ in range(p_cat)]
            cat_effect_scale = float(rng.choice(cat_effect_scale_grid))
            missing_rate = float(rng.choice(cat_missing_rate_grid))
            imbalance_type = str(rng.choice(cat_imbalance_grid))
            target_noise_scale = float(rng.choice(target_noise_grid))
            rho = float(rng.choice(rho_grid))
            assignments.append({
                "prior_regime": regime,
                "n": n,
                "p_num_signal": p_num_signal,
                "p_num_noise": p_num_noise,
                "p_cat_signal": p_cat_signal,
                "p_cat_noise": p_cat_noise,
                "cardinalities": cardinalities,
                "cat_effect_scale": cat_effect_scale,
                "missing_rate": missing_rate,
                "imbalance_type": imbalance_type,
                "target_noise_scale": target_noise_scale,
                "rho": rho,
            })

    rng.shuffle(assignments)
    audit = {
        "allocation_mode": allocation_mode,
        "profile": profile,
        "regime_counts": {r: regime_counts.get(r, 0) for r in regimes},
        "n_datasets": n_datasets,
        "allow_underdetermined": allow_underdetermined,
    }
    return assignments, audit


def allocate_mixed_classification_tasks(
    n_datasets: int,
    profile: str,
    base_seed: int,
    *,
    n_grid: "list[int] | None" = None,
    p_num_signal_grid: "list[int] | None" = None,
    p_num_noise_grid: "list[int] | None" = None,
    p_cat_signal_grid: "list[int] | None" = None,
    p_cat_noise_grid: "list[int] | None" = None,
    cat_cardinality_grid: "list[int] | None" = None,
    cat_effect_scale_grid: "list[float] | None" = None,
    cat_missing_rate_grid: "list[float] | None" = None,
    cat_imbalance_grid: "list[str] | None" = None,
    num_classes_grid: "list[int] | None" = None,
    temperature_grid: "list[float] | None" = None,
    label_noise_grid: "list[float] | None" = None,
    rho_grid: "list[float] | None" = None,
    allocation_mode: str = "weighted_quota",
    min_regime_count: int = 10,
    allow_underdetermined: bool = False,
) -> "tuple[list[dict], dict]":
    """Allocate mixed-classification task assignments."""
    if profile not in CLASSIFICATION_PROFILES:
        raise ValueError(
            f"allocate_mixed_classification_tasks: {profile!r} is not a classification profile"
        )
    n_grid = n_grid or [32, 64, 128, 256, 512, 1024]
    p_num_signal_grid = p_num_signal_grid or [2, 4, 8, 16, 32]
    p_num_noise_grid = p_num_noise_grid or [0, 8, 24, 56]
    p_cat_signal_grid = p_cat_signal_grid or [1, 2, 4, 8]
    p_cat_noise_grid = p_cat_noise_grid or [0, 2, 4, 8]
    cat_cardinality_grid = cat_cardinality_grid or [2, 3, 5, 10, 20, 50]
    cat_effect_scale_grid = cat_effect_scale_grid or [0.25, 0.5, 1.0, 2.0]
    cat_missing_rate_grid = cat_missing_rate_grid or [0.0, 0.01, 0.05, 0.10]
    cat_imbalance_grid = cat_imbalance_grid or ["balanced", "mild", "moderate", "severe"]
    num_classes_grid = num_classes_grid or [2, 3, 5, 10]
    temperature_grid = temperature_grid or [0.5, 1.0, 2.0, 4.0]
    label_noise_grid = label_noise_grid or [0.0, 0.02, 0.05, 0.10]
    rho_grid = rho_grid or [0.0, 0.3, 0.6, 0.9]

    weights = dict(REGIME_WEIGHTS[profile])
    if not allow_underdetermined:
        weights.pop("J_low_n_high_p_classification", None)
    regimes = list(weights.keys())
    regime_counts = largest_remainder_counts(weights, n_datasets)

    rng = np.random.default_rng(seed=base_seed)
    assignments = []
    for regime, count in regime_counts.items():
        for _ in range(count):
            n = int(rng.choice(n_grid))
            p_num_signal = int(rng.choice(p_num_signal_grid))
            p_num_noise = int(rng.choice(p_num_noise_grid))
            p_cat_signal = int(rng.choice(p_cat_signal_grid))
            p_cat_noise = int(rng.choice(p_cat_noise_grid))
            p_cat = p_cat_signal + p_cat_noise
            cardinalities = [int(rng.choice(cat_cardinality_grid)) for _ in range(p_cat)]
            cat_effect_scale = float(rng.choice(cat_effect_scale_grid))
            missing_rate = float(rng.choice(cat_missing_rate_grid))
            imbalance_type = str(rng.choice(cat_imbalance_grid))
            num_classes = int(rng.choice(num_classes_grid))
            temperature = float(rng.choice(temperature_grid))
            label_noise_rate = float(rng.choice(label_noise_grid))
            rho = float(rng.choice(rho_grid))
            assignments.append({
                "prior_regime": regime,
                "n": n,
                "p_num_signal": p_num_signal,
                "p_num_noise": p_num_noise,
                "p_cat_signal": p_cat_signal,
                "p_cat_noise": p_cat_noise,
                "cardinalities": cardinalities,
                "cat_effect_scale": cat_effect_scale,
                "missing_rate": missing_rate,
                "imbalance_type": imbalance_type,
                "num_classes": num_classes,
                "temperature": temperature,
                "label_noise_rate": label_noise_rate,
                "rho": rho,
            })

    rng.shuffle(assignments)
    audit = {
        "allocation_mode": allocation_mode,
        "profile": profile,
        "regime_counts": {r: regime_counts.get(r, 0) for r in regimes},
        "n_datasets": n_datasets,
        "allow_underdetermined": allow_underdetermined,
    }
    return assignments, audit
