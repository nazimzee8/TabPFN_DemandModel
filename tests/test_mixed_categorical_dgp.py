"""Tests for mixed-categorical DGP generation (Step 9 — 8 tests)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from constants import (
    ENTITY_EMBED_FIRST_REAL_ID,
    ENTITY_EMBED_MISSING_ID,
    ENTITY_EMBED_PAD_ID,
    ENTITY_EMBED_UNKNOWN_ID,
)
from dgp_helpers import (
    apply_category_missing,
    build_mixed_classification_dataset,
    build_mixed_regression_dataset,
    generate_categorical_features,
    mark_unseen_query_categories,
    validate_mixed_classification_dataset,
    validate_mixed_regression_dataset,
)


# Test 1: Category IDs in valid range
def test_generate_categorical_features_id_range():
    rng = np.random.default_rng(42)
    cardinalities = [3, 5, 10, 20]
    X_cat = generate_categorical_features(rng, 100, cardinalities)
    assert X_cat.shape == (100, 4)
    assert X_cat.dtype == np.int64
    # No PAD, MISSING, or UNKNOWN tokens
    assert not np.any(X_cat == ENTITY_EMBED_PAD_ID)
    assert not np.any(X_cat == ENTITY_EMBED_MISSING_ID)
    assert not np.any(X_cat == ENTITY_EMBED_UNKNOWN_ID)
    # All IDs in valid range per feature
    for j, card in enumerate(cardinalities):
        col = X_cat[:, j]
        assert np.all(col >= ENTITY_EMBED_FIRST_REAL_ID)
        assert np.all(col < ENTITY_EMBED_FIRST_REAL_ID + card)


# Test 2: Missing values applied correctly
def test_apply_category_missing_sets_missing_id():
    rng = np.random.default_rng(42)
    X_cat = generate_categorical_features(rng, 200, [5, 10])
    X_masked, mask = apply_category_missing(rng, X_cat, missing_rate=0.2)
    assert mask.dtype == bool
    assert X_masked.shape == X_cat.shape
    # Where mask is True, value should be MISSING_ID
    assert np.all(X_masked[mask] == ENTITY_EMBED_MISSING_ID)
    # Where mask is False, value should be unchanged
    assert np.all(X_masked[~mask] == X_cat[~mask])
    # Roughly 20% missing
    assert 0.1 < mask.mean() < 0.35


# Test 3: Unseen query categories detected
def test_mark_unseen_query_returns_correct_mask():
    # Train has categories [3, 4], test has [3, 5] → 5 is unseen
    X_train = np.array([[3, 3], [4, 4]], dtype=np.int64)
    X_test = np.array([[3, 5], [5, 3]], dtype=np.int64)
    mask = mark_unseen_query_categories(X_train, X_test)
    assert mask.shape == (2, 2)
    # Row 0, Col 0: 3 is in train → False
    assert not mask[0, 0]
    # Row 0, Col 1: 5 is NOT in train col 1 → True
    assert mask[0, 1]
    # Row 1, Col 0: 5 is NOT in train col 0 → True
    assert mask[1, 0]
    # Row 1, Col 1: 3 is in train col 1 → False
    assert not mask[1, 1]


# Test 4: Deterministic regression dataset generation
def test_build_mixed_regression_dataset_deterministic():
    params = {
        "n": 100, "p_num_signal": 4, "p_num_noise": 2,
        "p_cat_signal": 2, "p_cat_noise": 1,
        "cardinalities": [3, 5, 10],
        "cat_effect_scale": 1.0, "missing_rate": 0.05,
    }
    ds1 = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
    ds2 = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
    np.testing.assert_array_equal(ds1["X_num"], ds2["X_num"])
    np.testing.assert_array_equal(ds1["X_cat"], ds2["X_cat"])
    np.testing.assert_array_equal(ds1["y"], ds2["y"])


# Test 5: Classification all K classes present
def test_build_mixed_classification_dataset_K_constraint():
    params = {
        "n": 200, "p_num_signal": 4, "p_num_noise": 0,
        "p_cat_signal": 2, "p_cat_noise": 0,
        "cardinalities": [5, 10], "num_classes": 5,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(np.random.default_rng(42), "A_iid_dense_logistic", params)
    y = ds["y"]
    for k in range(5):
        assert np.any(y == k), f"Class {k} not present in y"


# Test 6: Validation passes for valid regression dataset
def test_validate_mixed_regression_dataset_passes():
    params = {
        "n": 50, "p_num_signal": 3, "p_num_noise": 1,
        "p_cat_signal": 2, "p_cat_noise": 1,
        "cardinalities": [3, 5, 10],
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
    validate_mixed_regression_dataset(ds)  # Should not raise


# Test 7: Validation rejects if PAD_ID in X_cat
def test_validate_mixed_regression_dataset_rejects_pad_id():
    params = {
        "n": 50, "p_num_signal": 3, "p_num_noise": 0,
        "p_cat_signal": 1, "p_cat_noise": 0,
        "cardinalities": [5],
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
    # Inject a PAD_ID
    ds["X_cat"][0, 0] = ENTITY_EMBED_PAD_ID
    with pytest.raises(AssertionError, match="Forbidden token"):
        validate_mixed_regression_dataset(ds)


# Test 8: Noise categorical features have zero effects
def test_cat_effects_inactive_features_zero():
    params = {
        "n": 50, "p_num_signal": 2, "p_num_noise": 0,
        "p_cat_signal": 1, "p_cat_noise": 2,
        "cardinalities": [5, 5, 5],
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
    cat_effects = ds["cat_effects"]
    cat_support = ds["cat_support_mask"]
    # Feature 0 is signal → effects should be non-zero (with high probability)
    assert cat_support[0] is True
    # Features 1, 2 are noise → effects should be exactly zero
    assert cat_support[1] is False
    assert cat_support[2] is False
    np.testing.assert_array_equal(cat_effects[1], 0.0)
    np.testing.assert_array_equal(cat_effects[2], 0.0)
