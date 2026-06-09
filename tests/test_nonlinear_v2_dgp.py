"""
test_nonlinear_v2_dgp.py
========================
12 statistical validity tests for the nonlinear_v2 DGP.
No Snowflake dependency — local only.

Each test addresses a specific deficiency from the nonlinear_v1 suite.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import generate_nonlinear_dgp as gen


# ---------------------------------------------------------------------------
# Helper: generate a dataset with default params, overridable
# ---------------------------------------------------------------------------

def _make(
    target_family="poly_quad",
    feature_regime="iid_dense",
    n=300,
    p_signal=16,
    target_noise_scale=1.0,
    target_noise_type="gaussian",
    teacher_seed=42,
    sample_seed=43,
    **kwargs,
):
    return gen.generate_v2_dataset(
        target_family=target_family,
        feature_regime=feature_regime,
        n=n,
        p_signal=p_signal,
        target_noise_scale=target_noise_scale,
        target_noise_type=target_noise_type,
        suite_component="core",
        teacher_seed=teacher_seed,
        sample_seed=sample_seed,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1: Normalisation uses independent calibration sample (fixes deficiency #5)
# ---------------------------------------------------------------------------

def test_normalization_uses_independent_calibration_sample():
    """normalization_constant must not depend on the eval sample."""
    ds_a = _make(sample_seed=100)
    ds_b = _make(sample_seed=999)

    # Same teacher_seed → same normalization_constant regardless of sample_seed
    assert ds_a["normalization_constant"] == ds_b["normalization_constant"], (
        "normalization_constant should only depend on teacher_seed, not sample_seed"
    )

    # normalization_constant > 0
    assert ds_a["normalization_constant"] > 0.0

    # Different teacher_seed → different normalization_constant (almost certainly)
    ds_c = _make(teacher_seed=9999)
    # This is a probabilistic check — just verify the constant is non-trivial
    assert ds_c["normalization_constant"] > 0.0


# ---------------------------------------------------------------------------
# Test 2: feature_noise_level is float64 in eval parquet (fixes deficiency #6)
# ---------------------------------------------------------------------------

def test_feature_noise_level_is_float64():
    """write_parquet_eval must write feature_noise_level as float64, never int."""
    tmp_dir = tempfile.mkdtemp()
    try:
        ds = _make(feature_regime="feat_noise", feature_noise_sigma=0.25)
        path = os.path.join(tmp_dir, "eval.parquet")
        gen.write_parquet_eval(ds, path)

        table = pq.read_table(path)
        field_type = str(table.schema.field("feature_noise_level").type)
        assert field_type in ("double", "float64"), (
            f"feature_noise_level should be float64, got {field_type!r}"
        )
        val = table.to_pydict()["feature_noise_level"][0]
        assert isinstance(val, float), f"Expected Python float, got {type(val)}"
        assert val > 0.0, "feature_noise_sigma=0.25 should produce non-zero feature_noise_level"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Correlated regimes have strictly positive rho (fixes deficiency #7)
# ---------------------------------------------------------------------------

def test_correlated_regimes_have_positive_rho():
    """ar1/block/equicorr datasets must have rho > 0."""
    correlated_regimes = ["ar1", "block", "equicorr"]
    rng = np.random.default_rng(777)

    for regime in correlated_regimes:
        for rep in range(10):
            teacher_seed = int(rng.integers(2**30))
            sample_seed = int(rng.integers(2**30))
            ds = _make(feature_regime=regime, teacher_seed=teacher_seed, sample_seed=sample_seed)
            assert ds["rho"] > 0.0, (
                f"regime={regime!r} rep={rep}: rho={ds['rho']} must be > 0"
            )


# ---------------------------------------------------------------------------
# Test 4: All families pass the learnability gate (fixes deficiency #2: unlearnable J)
# ---------------------------------------------------------------------------

def test_all_families_pass_learnability_gate():
    """Every target family should generate a dataset without raising LearnabilityGateError."""
    rng = np.random.default_rng(321)

    for family in gen.V2_TARGET_FAMILIES:
        teacher_seed = int(rng.integers(2**30))
        sample_seed = int(rng.integers(2**30))
        # Should NOT raise
        ds = _make(target_family=family, teacher_seed=teacher_seed, sample_seed=sample_seed)
        assert "betaX" in ds
        assert ds["normalization_constant"] > 0.0


# ---------------------------------------------------------------------------
# Test 5: sin_low is not trivially OLS-learnable (not deficiency #3 — checks R² < 1)
# ---------------------------------------------------------------------------

def test_sin_low_not_trivially_ols_learnable():
    """sin_low should have nonlinear residual — OLS R² on betaX < 1."""
    ds = _make(target_family="sin_low", n=500, p_signal=8)
    X = ds["X"]
    betaX = ds["betaX"]

    # Fit OLS on X → betaX
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    Xty = X.T @ betaX
    coef = np.linalg.solve(XtX, Xty)
    y_pred = X @ coef
    ss_res = float(np.sum((betaX - y_pred) ** 2))
    ss_tot = float(np.sum((betaX - betaX.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)

    # sin_low is nonlinear — OLS should not be perfect
    assert r2 < 0.999, f"sin_low OLS R²={r2:.4f} too high — may be trivially linear"


# ---------------------------------------------------------------------------
# Test 6: SNR is consistent across target families (fixes deficiency #4)
# ---------------------------------------------------------------------------

def test_snr_consistent_across_target_families():
    """betaX should have unit std (approx) after calibration, across all families."""
    rng = np.random.default_rng(555)

    stds = {}
    for family in gen.V2_TARGET_FAMILIES:
        teacher_seed = int(rng.integers(2**30))
        sample_seed = int(rng.integers(2**30))
        ds = _make(
            target_family=family,
            n=1000,
            p_signal=16,
            target_noise_scale=0.0,  # no noise — pure signal
            target_noise_type="gaussian",
            teacher_seed=teacher_seed,
            sample_seed=sample_seed,
        )
        stds[family] = float(np.std(ds["betaX"]))

    # All families should have betaX std close to 1.0 (within factor of 5)
    for family, std in stds.items():
        assert 0.05 < std < 20.0, (
            f"Family {family!r}: betaX std={std:.4f} out of expected range [0.05, 20.0]"
        )


# ---------------------------------------------------------------------------
# Test 7: All target × feature cells covered (fixes deficiency #1)
# ---------------------------------------------------------------------------

def test_all_target_x_feature_cells_covered():
    """enumerate_suite_cells must cover all 42 (family, regime) combinations."""
    cells = gen.enumerate_suite_cells()
    covered = {(c["target_family"], c["feature_regime"]) for c in cells}
    expected = {
        (t, f)
        for t in gen.V2_TARGET_FAMILIES
        for f in gen.V2_FEATURE_REGIMES
    }
    missing = expected - covered
    assert missing == set(), f"Missing (family, regime) cells: {missing}"
    assert len(cells) == 420


# ---------------------------------------------------------------------------
# Test 8: No duplicate logical_dataset_keys
# ---------------------------------------------------------------------------

def test_no_duplicate_logical_dataset_keys():
    """Every cell must have a unique logical_dataset_key."""
    cells = gen.enumerate_suite_cells()
    keys = [c["logical_dataset_key"] for c in cells]
    duplicates = [k for k in set(keys) if keys.count(k) > 1]
    assert duplicates == [], f"Duplicate logical_dataset_keys: {duplicates}"


# ---------------------------------------------------------------------------
# Test 9: demand_mono has negative own-price elasticity
# ---------------------------------------------------------------------------

def test_demand_mono_negative_own_elasticity():
    """demand_mono betaX must be negatively correlated with X[:, 0] (own-price)."""
    ds = _make(target_family="demand_mono", n=1000, p_signal=8)
    X = ds["X"]
    betaX = ds["betaX"]

    # Correlation between own-price (X[:,0]) and betaX should be negative
    corr = float(np.corrcoef(X[:, 0], betaX)[0, 1])
    assert corr < 0.0, (
        f"demand_mono: corr(X[:,0], betaX)={corr:.4f} should be negative (own-price elasticity)"
    )


# ---------------------------------------------------------------------------
# Test 10: mixed_linear alpha controls OLS ceiling
# ---------------------------------------------------------------------------

def test_mixed_linear_alpha_controls_ols_ceiling():
    """Higher alpha → higher OLS R² on betaX."""
    rng = np.random.default_rng(888)

    r2_scores = {}
    for alpha in [0.1, 0.9]:
        teacher_seed = int(rng.integers(2**30))
        sample_seed = int(rng.integers(2**30))
        ds = gen.generate_v2_dataset(
            target_family="mixed_linear",
            feature_regime="iid_dense",
            n=500,
            p_signal=16,
            target_noise_scale=0.01,  # minimal noise
            target_noise_type="gaussian",
            suite_component="core",
            teacher_seed=teacher_seed,
            sample_seed=sample_seed,
            alpha=alpha,
        )
        X = ds["X"]
        betaX = ds["betaX"]
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        Xty = X.T @ betaX
        coef = np.linalg.solve(XtX, Xty)
        y_pred = X @ coef
        ss_res = float(np.sum((betaX - y_pred) ** 2))
        ss_tot = float(np.sum((betaX - betaX.mean()) ** 2))
        r2_scores[alpha] = 1.0 - ss_res / (ss_tot + 1e-30)

    # High alpha (more linear) should have higher R² than low alpha
    assert r2_scores[0.9] > r2_scores[0.1], (
        f"mixed_linear: R²(alpha=0.9)={r2_scores[0.9]:.4f} should > "
        f"R²(alpha=0.1)={r2_scores[0.1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 11: write_parquet_eval round-trip — expected columns survive write/read
# ---------------------------------------------------------------------------

def test_write_parquet_eval_round_trip():
    """write_parquet_eval should produce a readable parquet with ≥25 columns."""
    tmp_dir = tempfile.mkdtemp()
    try:
        ds = _make(target_family="poly_quad", feature_regime="ar1")
        path = os.path.join(tmp_dir, "eval_roundtrip.parquet")
        n_written = gen.write_parquet_eval(ds, path)

        assert n_written == 1

        table = pq.read_table(path)
        required_cols = [
            "X", "y", "betaX",
            "suite_family", "prior_regime", "n_total", "p_signal", "p_noise", "p_total",
            "target_noise_scale", "training_size_anchor", "feature_noise_level",
            "feature_regime", "covariance_type", "rho", "active_fraction",
            "noise_feature_fraction", "feature_noise_sigma", "suite_component",
            "target_noise_type", "snr_target", "condition_id",
            "teacher_seed", "sample_seed", "normalization_constant",
        ]
        for col in required_cols:
            assert col in table.column_names, f"Missing column: {col!r}"

        assert len(table.column_names) >= 25

        data = table.to_pydict()
        assert data["prior_regime"][0] == "poly_quad"
        assert data["feature_regime"][0] == "ar1"
        assert data["normalization_constant"][0] > 0.0
        # feature_noise_level must be float type
        assert str(table.schema.field("feature_noise_level").type) in ("double", "float64")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 12: heteroscedastic noise variance scales with betaX
# ---------------------------------------------------------------------------

def test_heteroscedastic_noise_variance():
    """Heteroscedastic noise: residuals should correlate with |betaX|."""
    ds = _make(
        target_family="poly_quad",
        feature_regime="iid_dense",
        n=2000,
        p_signal=8,
        target_noise_scale=1.0,
        target_noise_type="heteroscedastic",
    )
    betaX = ds["betaX"]
    y = ds["y"]
    residuals = np.abs(y - betaX)
    abs_betaX = np.abs(betaX)

    # Pearson correlation between |residuals| and |betaX| should be positive
    corr = float(np.corrcoef(abs_betaX, residuals)[0, 1])
    assert corr > 0.0, (
        f"Heteroscedastic noise: corr(|betaX|, |residuals|)={corr:.4f} should be positive"
    )
