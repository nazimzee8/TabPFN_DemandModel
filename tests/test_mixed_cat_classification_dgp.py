"""Tests for mixed-categorical classification DGP generation (Step 12b)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import tempfile

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from constants import (
    ENTITY_EMBED_FIRST_REAL_ID,
    MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
    MIXED_CLS_DGP_SCHEMA_VERSION,
)
from dgp_helpers import (
    generate_categorical_class_effects,
    build_mixed_classification_dataset,
    validate_mixed_classification_dataset,
    write_parquet_mixed_classification_dgp,
    mark_unseen_query_categories,
    allocate_mixed_classification_tasks,
)


# ---------------------------------------------------------------------------
# test_balanced_labels_exact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K,n", [
    (2, 100), (3, 99), (5, 200), (10, 100),
])
def test_balanced_labels_present(K, n):
    """All K classes should be present in the generated labels."""
    params = {
        "n": n, "p_num_signal": 4, "p_num_noise": 0,
        "p_cat_signal": 2, "p_cat_noise": 0,
        "cardinalities": [5, 10], "num_classes": K,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(
        np.random.default_rng(42), "A_iid_dense_logistic", params,
    )
    y = ds["y"]
    for k in range(K):
        assert np.any(y == k), f"Class {k} not present in y"


# ---------------------------------------------------------------------------
# test_balanced_split_preserves_balance
# ---------------------------------------------------------------------------
def test_balanced_split_preserves_class_presence():
    """After 80/20 split, both halves should have all classes."""
    K = 5
    params = {
        "n": 200, "p_num_signal": 4, "p_num_noise": 0,
        "p_cat_signal": 2, "p_cat_noise": 0,
        "cardinalities": [5, 10], "num_classes": K,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(
        np.random.default_rng(42), "A_iid_dense_logistic", params,
    )
    n = ds["n"]
    n_train = int(0.8 * n)
    y_train = ds["y"][:n_train]
    y_test = ds["y"][n_train:]
    for k in range(K):
        assert np.any(y_train == k), f"Class {k} missing from train"
    # With n=200, test has 40 samples — should have most classes
    present_test = len(np.unique(y_test))
    assert present_test >= K - 1, (
        f"Expected at least {K-1} classes in test, got {present_test}"
    )


# ---------------------------------------------------------------------------
# test_reference_class_effects_zero
# ---------------------------------------------------------------------------
def test_reference_class_effects_zero():
    rng = np.random.default_rng(42)
    cardinalities = [3, 5, 10]
    K = 4
    active_mask = [True, True, True]
    effects = generate_categorical_class_effects(
        rng, cardinalities, K, effect_scale=1.0, active_mask=active_mask,
    )
    for j, eff in enumerate(effects):
        assert eff.shape == (cardinalities[j], K)
        np.testing.assert_array_equal(
            eff[:, 0], 0.0,
            err_msg=f"Feature {j}: reference class column must be zero",
        )


# ---------------------------------------------------------------------------
# test_generate_mixed_cat_classification_all_regimes (parametrized)
# ---------------------------------------------------------------------------
_CLASSIFICATION_REGIMES = [
    "A_iid_dense_logistic", "B_iid_sparse_logistic",
    "D_ar1_dense_logistic", "E_high_dim_dense_logistic",
]


@pytest.mark.parametrize("regime", _CLASSIFICATION_REGIMES)
def test_generate_mixed_cat_classification_regime(regime):
    params = {
        "n": 100, "p_num_signal": 4, "p_num_noise": 0,
        "p_cat_signal": 2, "p_cat_noise": 0,
        "cardinalities": [5, 10], "num_classes": 3,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(
        np.random.default_rng(42), regime, params,
    )
    assert ds["n"] == 100
    assert ds["num_classes"] == 3
    assert ds["training_data_family"] == MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY
    validate_mixed_classification_dataset(ds)


# ---------------------------------------------------------------------------
# test_validate_balanced_catches_imbalanced
# ---------------------------------------------------------------------------
def test_validate_catches_reference_class_violation():
    params = {
        "n": 60, "p_num_signal": 3, "p_num_noise": 0,
        "p_cat_signal": 1, "p_cat_noise": 0,
        "cardinalities": [5], "num_classes": 3,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(
        np.random.default_rng(42), "A_iid_dense_logistic", params,
    )
    # Corrupt: set W_num reference class to non-zero
    ds["W_num"][0, 0] = 1.0
    with pytest.raises(AssertionError, match="W_num.*0.*must be 0"):
        validate_mixed_classification_dataset(ds)


# ---------------------------------------------------------------------------
# test_allocate_balanced_only
# ---------------------------------------------------------------------------
def test_allocate_classification_tasks_structure():
    assignments, audit = allocate_mixed_classification_tasks(
        n_datasets=30,
        profile="linear_classification_mixed_categorical_stat_aware",
        base_seed=42,
    )
    assert len(assignments) == 30
    for a in assignments:
        assert "prior_regime" in a
        assert "num_classes" in a
        assert "p_cat_signal" in a
        assert "cardinalities" in a
        assert len(a["cardinalities"]) == a["p_cat_signal"] + a["p_cat_noise"]
    assert audit["n_datasets"] == 30


# ---------------------------------------------------------------------------
# test_label_noise_off_by_default
# ---------------------------------------------------------------------------
def test_label_noise_off_by_default():
    params = {
        "n": 60, "p_num_signal": 3, "p_num_noise": 0,
        "p_cat_signal": 1, "p_cat_noise": 0,
        "cardinalities": [5], "num_classes": 3,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
        # label_noise_rate not specified → should default to 0.0
    }
    ds = build_mixed_classification_dataset(
        np.random.default_rng(42), "A_iid_dense_logistic", params,
    )
    assert ds["label_noise_rate"] == 0.0


# ---------------------------------------------------------------------------
# test_infeasible_n_k_rejected
# ---------------------------------------------------------------------------
def test_infeasible_n_k_still_generates():
    """With very small n and large K, the builder should still produce valid output.

    The injection fallback may not guarantee all classes when n < 2*K because
    later injections can overwrite earlier ones. We just verify the builder
    succeeds and produces a valid array.
    """
    params = {
        "n": 20, "p_num_signal": 2, "p_num_noise": 0,
        "p_cat_signal": 1, "p_cat_noise": 0,
        "cardinalities": [3], "num_classes": 5,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(
        np.random.default_rng(42), "A_iid_dense_logistic", params,
    )
    y = ds["y"]
    assert y.shape == (20,)
    assert y.dtype == np.int64
    assert np.all((y >= 0) & (y < 5))
    # With n=20 and K=5, all classes should be present after injection
    for k in range(5):
        assert np.any(y == k), f"Class {k} not present"


# ---------------------------------------------------------------------------
# test_write_load_roundtrip_classification
# ---------------------------------------------------------------------------
def test_write_load_roundtrip_classification():
    import pyarrow.parquet as pq
    params = {
        "n": 100, "p_num_signal": 3, "p_num_noise": 1,
        "p_cat_signal": 2, "p_cat_noise": 0,
        "cardinalities": [3, 5], "num_classes": 3,
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_classification_dataset(
        np.random.default_rng(42), "A_iid_dense_logistic", params,
    )
    n = ds["n"]
    n_train = int(0.8 * n)
    n_test = n - n_train
    unknown_mask = mark_unseen_query_categories(
        ds["X_cat"][:n_train], ds["X_cat"][n_train:]
    )
    ds_split = {
        "X_num_train": ds["X_num"][:n_train],
        "X_num_test": ds["X_num"][n_train:],
        "y_train": ds["y"][:n_train],
        "y_test": ds["y"][n_train:],
        "X_cat_train": ds["X_cat"][:n_train],
        "X_cat_test": ds["X_cat"][n_train:],
        "cat_missing_mask_train": np.zeros((n_train, 2), dtype=bool),
        "cat_missing_mask_test": np.zeros((n_test, 2), dtype=bool),
        "cat_unknown_mask_test": unknown_mask,
        "categorical_cardinalities": np.array(ds["cardinalities"], dtype=np.int64),
        "n": n, "p_num": ds["p_num"], "p_cat": ds["p_cat"],
        "n_train": n_train, "n_test": n_test,
        "num_classes": 3,
        "prior_regime": "A_iid_dense_logistic",
        "schema_version": MIXED_CLS_DGP_SCHEMA_VERSION,
        "task_family": "linear_classification",
        "training_data_family": MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
        "task_objective": "inductive_classification",
        "W_num": ds["W_num"],
        "b": ds["b"],
        "numeric_support_mask": ds["numeric_support_mask"],
        "cat_support_mask": ds["cat_support_mask"],
        "cat_class_effects": ds["cat_class_effects"],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = str(Path(tmpdir) / "task_cls_rt.parquet")
        write_parquet_mixed_classification_dgp(
            ds_split, fp, store_class_cat_params=True,
        )
        table = pq.read_table(fp)
        d = table.to_pydict()
        assert d["schema_version"][0] == MIXED_CLS_DGP_SCHEMA_VERSION
        assert d["training_data_family"][0] == MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY
        assert d["num_classes"][0] == 3
