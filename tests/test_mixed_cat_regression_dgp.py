"""Tests for mixed-categorical regression DGP generation (Step 12a)."""

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
    MIXED_CAT_REGRESSION_TRAINING_FAMILY,
    MIXED_REG_DGP_SCHEMA_VERSION,
)
from dgp_helpers import (
    generate_categorical_features,
    generate_categorical_effects_regression,
    build_mixed_regression_dataset,
    validate_mixed_regression_dataset,
    write_parquet_mixed_regression_dgp,
    mark_unseen_query_categories,
    allocate_mixed_regression_tasks,
)


# ---------------------------------------------------------------------------
# test_generate_categorical_features_shape_and_range
# ---------------------------------------------------------------------------
def test_generate_categorical_features_shape_and_range():
    rng = np.random.default_rng(42)
    cardinalities = [2, 5, 10, 32]
    X_cat = generate_categorical_features(rng, 200, cardinalities)
    assert X_cat.shape == (200, 4)
    assert X_cat.dtype == np.int64
    for j, card in enumerate(cardinalities):
        col = X_cat[:, j]
        assert np.all(col >= ENTITY_EMBED_FIRST_REAL_ID)
        assert np.all(col < ENTITY_EMBED_FIRST_REAL_ID + card)
        # All categories should be represented with high probability for large n
        unique = set(col.tolist())
        assert len(unique) >= min(card, 2)  # at least 2 unique values


# ---------------------------------------------------------------------------
# test_categorical_effects_centered
# ---------------------------------------------------------------------------
def test_categorical_effects_centered():
    rng = np.random.default_rng(123)
    cardinalities = [3, 5, 10, 20]
    active_mask = [True, True, False, True]
    effects = generate_categorical_effects_regression(
        rng, cardinalities, effect_scale=1.0, active_mask=active_mask,
    )
    assert len(effects) == 4
    for j, (eff, active) in enumerate(zip(effects, active_mask)):
        assert eff.shape == (cardinalities[j],)
        if active:
            assert abs(eff.mean()) < 1e-10, f"Feature {j} effects not centered"
        else:
            np.testing.assert_array_equal(eff, 0.0)


# ---------------------------------------------------------------------------
# test_apply_categorical_effects_regression_shape
# ---------------------------------------------------------------------------
def test_apply_categorical_effects_regression_shape():
    rng = np.random.default_rng(99)
    n, p_cat = 50, 3
    cardinalities = [5, 10, 3]
    X_cat = generate_categorical_features(rng, n, cardinalities)
    active_mask = [True, True, True]
    effects = generate_categorical_effects_regression(
        rng, cardinalities, effect_scale=1.0, active_mask=active_mask,
    )
    # Manually compute the categorical contribution
    y_cat = np.zeros(n, dtype=float)
    for j in range(p_cat):
        for i in range(n):
            cid = int(X_cat[i, j]) - ENTITY_EMBED_FIRST_REAL_ID
            y_cat[i] += effects[j][cid]
    assert y_cat.shape == (n,)
    assert np.all(np.isfinite(y_cat))
    # Non-trivial contribution
    assert np.std(y_cat) > 0.01


# ---------------------------------------------------------------------------
# test_generate_mixed_cat_regression_dataset_all_regimes (parametrized)
# ---------------------------------------------------------------------------
_REGRESSION_REGIMES = [
    "A_iid_dense", "B_iid_sparse", "C_label_noise",
    "D_ar1_dense", "E_high_dim_dense", "F_high_dim_sparse",
]


@pytest.mark.parametrize("regime", _REGRESSION_REGIMES)
def test_generate_mixed_cat_regression_dataset_regime(regime):
    params = {
        "n": 100, "p_num_signal": 4, "p_num_noise": 2,
        "p_cat_signal": 2, "p_cat_noise": 1,
        "cardinalities": [3, 5, 10],
        "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_regression_dataset(
        np.random.default_rng(42), regime, params,
    )
    assert ds["n"] == 100
    assert ds["p_num"] == 6
    assert ds["p_cat"] == 3
    assert ds["y"].shape == (100,)
    assert ds["X_num"].shape == (100, 6)
    assert ds["X_cat"].shape == (100, 3)
    assert ds["training_data_family"] == MIXED_CAT_REGRESSION_TRAINING_FAMILY
    assert ds["schema_version"] == MIXED_REG_DGP_SCHEMA_VERSION
    validate_mixed_regression_dataset(ds)


# ---------------------------------------------------------------------------
# test_validate_mixed_cat_regression_catches_invalid
# ---------------------------------------------------------------------------
def test_validate_mixed_cat_regression_catches_invalid_schema_version():
    params = {
        "n": 50, "p_num_signal": 3, "p_num_noise": 0,
        "p_cat_signal": 1, "p_cat_noise": 0,
        "cardinalities": [5], "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
    ds["schema_version"] = "wrong_version"
    with pytest.raises(AssertionError, match="schema_version"):
        validate_mixed_regression_dataset(ds)


def test_validate_mixed_cat_regression_catches_bad_noise_beta():
    params = {
        "n": 50, "p_num_signal": 2, "p_num_noise": 2,
        "p_cat_signal": 1, "p_cat_noise": 0,
        "cardinalities": [5], "cat_effect_scale": 1.0, "missing_rate": 0.0,
    }
    ds = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
    # Corrupt: set noise beta to non-zero
    ds["beta_num"][3] = 1.0
    with pytest.raises(AssertionError, match="Noise numeric"):
        validate_mixed_regression_dataset(ds)


# ---------------------------------------------------------------------------
# test_write_load_roundtrip_regression
# ---------------------------------------------------------------------------
def test_write_load_roundtrip_regression():
    import pyarrow.parquet as pq
    params = {
        "n": 60, "p_num_signal": 3, "p_num_noise": 1,
        "p_cat_signal": 2, "p_cat_noise": 0,
        "cardinalities": [3, 5],
        "cat_effect_scale": 1.0, "missing_rate": 0.05,
    }
    ds = build_mixed_regression_dataset(np.random.default_rng(42), "A_iid_dense", params)
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
        "cat_missing_mask_train": ds["missing_mask"][:n_train],
        "cat_missing_mask_test": ds["missing_mask"][n_train:],
        "cat_unknown_mask_test": unknown_mask,
        "categorical_cardinalities": np.array(ds["cardinalities"], dtype=np.int64),
        "n": n, "p_num": ds["p_num"], "p_cat": ds["p_cat"],
        "n_train": n_train, "n_test": n_test,
        "prior_regime": "A_iid_dense",
        "schema_version": MIXED_REG_DGP_SCHEMA_VERSION,
        "task_family": "linear_regression",
        "training_data_family": MIXED_CAT_REGRESSION_TRAINING_FAMILY,
        "task_objective": "inductive_regression",
        "beta_num": ds["beta_num"],
        "numeric_support_mask": ds["numeric_support_mask"],
        "cat_support_mask": ds["cat_support_mask"],
        "cat_effects": ds["cat_effects"],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = str(Path(tmpdir) / "task_rt.parquet")
        write_parquet_mixed_regression_dgp(ds_split, fp, store_cat_params=True)
        table = pq.read_table(fp)
        d = table.to_pydict()
        assert d["schema_version"][0] == MIXED_REG_DGP_SCHEMA_VERSION
        assert d["training_data_family"][0] == MIXED_CAT_REGRESSION_TRAINING_FAMILY
        np.testing.assert_array_equal(
            np.array(d["X_cat_train"][0]), ds_split["X_cat_train"]
        )
        np.testing.assert_array_equal(
            np.array(d["X_cat_test"][0]), ds_split["X_cat_test"]
        )


# ---------------------------------------------------------------------------
# test_allocate_mixed_cat_regression_tasks_quota
# ---------------------------------------------------------------------------
def test_allocate_mixed_cat_regression_tasks_quota():
    assignments, audit = allocate_mixed_regression_tasks(
        n_datasets=50,
        profile="linear_regression_mixed_categorical_stat_aware",
        base_seed=42,
    )
    assert len(assignments) == 50
    # Every assignment has required keys
    for a in assignments:
        assert "prior_regime" in a
        assert "p_cat_signal" in a
        assert "p_cat_noise" in a
        assert "cardinalities" in a
        assert len(a["cardinalities"]) == a["p_cat_signal"] + a["p_cat_noise"]
    # Verify audit structure
    assert audit["n_datasets"] == 50
    assert audit["profile"] == "linear_regression_mixed_categorical_stat_aware"
    # Regime counts should sum to n_datasets
    total = sum(audit["regime_counts"].values())
    assert total == 50
