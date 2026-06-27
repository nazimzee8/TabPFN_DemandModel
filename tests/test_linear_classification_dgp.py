from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_DATA = SRC / "data_generation"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Ensure the canonical data_generation subdir is ahead of src/ so that
# bare `import generate_dgp` / `from dgp_helpers import` resolve to the
# co-located copy after the stale src/*.py duplicates were removed.
if str(SRC_DATA) not in sys.path:
    sys.path.insert(0, str(SRC_DATA))

import generate_dgp as generator  # noqa: E402
from dgp_helpers import (  # noqa: E402
    CLASSIFICATION_REGIMES,
    allocate_classification_tasks,
    apply_symmetric_label_noise,
    compute_classification_diagnostics,
    compute_classification_teacher,
    generate_classification_dataset,
    split_classification_dataset,
    validate_classification_dataset,
    write_classification_parquet,
)


def _grids() -> dict[str, list]:
    return {
        "n_grid": [64],
        "p_signal_grid": [32],
        "p_noise_grid": [8, 56],
        "active_s_grid": [4],
        "rho_grid": [0.0, 0.3, 0.6],
        "feature_noise_grid": [0.0, 0.1],
        "num_classes_grid": [2, 3, 5, 10],
        "temperature_grid": [0.5, 1.0, 2.0, 4.0],
        "label_noise_grid": [0.0, 0.02, 0.05, 0.1],
        "class_imbalance_grid": ["balanced", "mild", "moderate", "severe"],
        "margin_grid": ["low", "medium", "high"],
        "coefficient_scale_grid": [0.5, 1.0, 2.0],
        "intercept_scale_grid": [0.0, 0.5],
    }


REGIME_K = {
    "A_iid_dense_logistic": 2,
    "B_iid_sparse_logistic": 2,
    "C_label_noise_margin": 3,
    "D_correlated_ar_logistic": 2,
    "E_high_dim_dense_softmax": 3,
    "F_high_dim_sparse_softmax": 5,
    "G_noise_features_classification": 3,
    "H_block_correlated_classification": 5,
    "I_equicorrelated_classification": 2,
    "J_low_n_high_p_classification": 3,
    "K_feature_noise_classification": 2,
    "L_market_sign_classification": 3,
}


def _dataset(regime: str, seed: int = 123) -> dict:
    assignment = {
        "classification_regime": regime,
        "num_classes": REGIME_K[regime],
        "class_imbalance_type": "balanced",
        "margin_level": "low" if regime == "C_label_noise_margin" else "medium",
        "label_noise_rate": 0.05 if regime == "C_label_noise_margin" else 0.0,
        "coefficient_regime": (
            "sparse" if regime == "D_correlated_ar_logistic" else None
        ),
    }
    rng = np.random.default_rng(seed)
    return generate_classification_dataset(
        rng, regime, assignment, _grids(), task_seed=seed
    )


@pytest.mark.parametrize("regime", CLASSIFICATION_REGIMES)
def test_all_classification_regimes_generate_and_validate(regime):
    ds = _dataset(regime)
    validate_classification_dataset(ds)
    k = ds["num_classes"]

    assert ds["logits"].shape == (ds["n_total"], k)
    assert ds["probs"].shape == (ds["n_total"], k)
    np.testing.assert_allclose(ds["probs"].sum(axis=1), 1.0)
    assert np.all(ds["logits"][:, 0] == 0.0)
    # Canonical schema: W_true always (p, K) and b_true always (K,) for all K
    assert ds["W_true"].shape == (ds["p_total"], k)
    assert np.all(ds["W_true"][:, 0] == 0.0)
    assert ds["b_true"].shape == (k,)
    assert ds["b_true"][0] == 0.0
    # w_true deprecated alias: present for all K, equals W_true[:, 1]
    assert ds["w_true"].shape == (ds["p_total"],)
    np.testing.assert_array_equal(ds["w_true"], ds["W_true"][:, 1])


def test_regime_specific_invariants():
    dense = _dataset("A_iid_dense_logistic")
    sparse = _dataset("B_iid_sparse_logistic")
    noisy_labels = _dataset("C_label_noise_margin")
    correlated = _dataset("D_correlated_ar_logistic")
    high_dense = _dataset("E_high_dim_dense_softmax")
    per_class_sparse = _dataset("F_high_dim_sparse_softmax")
    irrelevant = _dataset("G_noise_features_classification")
    grouped = _dataset("H_block_correlated_classification")
    equicorrelated = _dataset("I_equicorrelated_classification")
    underdetermined = _dataset("J_low_n_high_p_classification")
    feature_noise = _dataset("K_feature_noise_classification")
    market = _dataset("L_market_sign_classification")

    assert dense["active_support"][:dense["p_signal"]].all()
    assert sparse["active_support"][:sparse["p_signal"]].sum() < sparse["p_signal"]
    assert noisy_labels["temperature"] in {2.0, 4.0}
    assert noisy_labels["label_noise_mask"].sum() == round(
        noisy_labels["label_noise_rate"] * noisy_labels["n_total"]
    )
    assert correlated["covariance_type"] == "ar1" and correlated["rho"] != 0.0
    assert high_dense["p_signal"] >= 32 and high_dense["n_total"] >= 2 * high_dense["p_total"]
    assert per_class_sparse["p_signal"] >= 32
    assert not np.array_equal(
        per_class_sparse["class_active_support"][:, 1],
        per_class_sparse["class_active_support"][:, 2],
    )
    assert 8 <= irrelevant["p_noise"] <= 120
    assert np.all(irrelevant["W_true"][irrelevant["p_signal"]:] == 0.0)
    assert grouped["coefficient_regime"] == "group_sparse"
    assert grouped["block_size"] in {4, 8, 16}
    assert equicorrelated["rho"] != 0.0
    assert underdetermined["n_total"] < underdetermined["p_total"]
    assert feature_noise["feature_noise_level"] > 0.0
    assert not np.array_equal(feature_noise["X"], feature_noise["X_clean"])
    np.testing.assert_allclose(
        feature_noise["logits"][:, 1],
        # b_true is now always (K,); use b_true[1] for class-1 intercept
        feature_noise["X_clean"] @ feature_noise["w_true"] + feature_noise["b_true"][1],
    )
    quarter = max(1, market["p_signal"] // 4)
    assert np.all(market["W_true"][:quarter, 1:] <= 0.0)
    cross = market["W_true"][quarter:2 * quarter, 1]
    assert np.all(np.sign(cross) == np.where(np.arange(cross.size) % 2 == 0, 1, -1))


def test_exact_symmetric_label_noise():
    labels = np.arange(100, dtype=np.int64) % 5
    observed, mask = apply_symmetric_label_noise(
        np.random.default_rng(9), labels, 5, 0.1
    )
    assert mask.dtype == bool
    assert mask.sum() == 10
    assert np.array_equal(mask, observed != labels)
    assert np.all(observed[mask] != labels[mask])


def test_allocation_exact_quotas_and_forced_regimes():
    assignments, audit = allocate_classification_tasks(
        100,
        "linear_classification_stat_aware",
        42,
        allow_underdetermined=False,
    )
    assert len(assignments) == 100
    assert audit["realized_class_counts"] == {2: 50, 3: 25, 5: 15, 10: 10}
    assert audit["realized_imbalance_counts"] == {
        "balanced": 50,
        "mild": 25,
        "moderate": 20,
        "severe": 5,
    }
    assert "J_low_n_high_p_classification" not in audit["regime_counts"]
    for assignment in assignments:
        regime = assignment["classification_regime"]
        if regime.startswith(("A_", "B_", "D_")):
            assert assignment["num_classes"] == 2
        if regime.startswith(("E_", "F_")):
            assert assignment["num_classes"] > 2


def test_allocation_is_seed_deterministic_and_j_is_opt_in():
    first, _ = allocate_classification_tasks(
        100, "linear_classification_stat_aware", 7
    )
    second, _ = allocate_classification_tasks(
        100, "linear_classification_stat_aware", 7
    )
    with_j, audit = allocate_classification_tasks(
        100,
        "linear_classification_stat_aware",
        7,
        allow_underdetermined=True,
    )
    assert first == second
    assert "J_low_n_high_p_classification" in audit["regime_counts"]
    assert any(
        a["classification_regime"] == "J_low_n_high_p_classification"
        for a in with_j
    )


def test_classification_parquet_contract(tmp_path):
    ds = _dataset("G_noise_features_classification")
    split = split_classification_dataset(ds)
    path = tmp_path / "task.parquet"
    write_classification_parquet(
        split,
        str(path),
        diagnostics=compute_classification_diagnostics(ds),
    )
    table = pq.read_table(path)
    names = set(table.column_names)
    required = {
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "y_clean_train",
        "label_noise_mask_train",
        "logits_train",
        "probs_train",
        "schema_version",
        "task_family",
        "W_true",
        "b_true",
        "active_support",
        "class_active_support",
    }
    assert required <= names
    assert not names.intersection(
        {"betaX_train", "betaX_test", "target_noise_scale", "best_ridge_lambda"}
    )
    assert table.schema.field("y_train").type == pa.list_(pa.int64())
    assert table.schema.field("label_noise_mask_train").type == pa.list_(pa.bool_())
    assert table.schema.field("X_train").type == pa.list_(pa.list_(pa.float64()))


def test_no_store_class_params_omits_parameters_and_supports(tmp_path):
    ds = _dataset("A_iid_dense_logistic")
    path = tmp_path / "no_params.parquet"
    write_classification_parquet(
        split_classification_dataset(ds),
        str(path),
        store_class_params=False,
        diagnostics=compute_classification_diagnostics(ds),
    )
    names = set(pq.read_table(path).column_names)
    assert not names.intersection(
        {"w_true", "W_true", "b_true", "active_support", "class_active_support"}
    )


def test_teacher_success_and_stable_failure_states():
    ds = split_classification_dataset(_dataset("A_iid_dense_logistic"))
    not_requested = compute_classification_teacher(
        ds["X_train"], ds["y_train"], ds["X_test"], 2, requested=False
    )
    assert not_requested["teacher_failure_reason"] == "not_requested"
    success = compute_classification_teacher(
        ds["X_train"], ds["y_train"], ds["X_test"], 2, requested=True
    )
    assert success["teacher_available"] is True
    assert success["teacher_logits_test"].shape == (ds["n_test"], 2)
    assert success["teacher_probs_test"].shape == (ds["n_test"], 2)

    missing = compute_classification_teacher(
        ds["X_train"], np.zeros(ds["n_train"], dtype=np.int64),
        ds["X_test"], 2, requested=True,
    )
    assert missing["teacher_failure_reason"] == "missing_context_class"


def test_teacher_dependency_missing_and_fit_failed(monkeypatch):
    ds = split_classification_dataset(_dataset("A_iid_dense_logistic"))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "sklearn.linear_model", None)
        missing = compute_classification_teacher(
            ds["X_train"], ds["y_train"], ds["X_test"], 2, requested=True
        )
    assert missing["teacher_failure_reason"] == "dependency_missing"

    import sklearn.linear_model

    class BrokenLogisticRegression:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(
        sklearn.linear_model, "LogisticRegression", BrokenLogisticRegression
    )
    failed = compute_classification_teacher(
        ds["X_train"], ds["y_train"], ds["X_test"], 2, requested=True
    )
    assert failed["teacher_failure_reason"] == "fit_failed"


def test_required_teacher_failure_aborts_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        generator,
        "compute_classification_teacher",
        lambda *args, **kwargs: {
            "teacher_available": False,
            "teacher_type": "l2_logistic",
            "teacher_failure_reason": "fit_failed",
            "teacher_regularization": 1.0,
        },
    )
    with pytest.raises(RuntimeError, match="required classification teacher failed"):
        generator.main([
            "--task_family", "linear_classification",
            "--profile", "classification_legacy_debug",
            "--n_datasets", "1",
            "--out_dir", str(tmp_path / "required_teacher"),
            "--base_seed", "2",
            "--n_grid", "64",
            "--p_signal_grid", "8",
            "--require_class_teachers",
        ])


def _run_small_suite(out_dir: Path, *, explicit_task_family: bool = True) -> dict:
    args = []
    if explicit_task_family:
        args += ["--task_family", "linear_classification"]
    args += [
        "--profile", "classification_legacy_debug",
        "--n_datasets", "4",
        "--out_dir", str(out_dir),
        "--base_seed", "19",
        "--n_grid", "64",
        "--p_signal_grid", "8",
        "--p_noise_grid", "8",
        "--active_s_grid", "2,4",
    ]
    generator.main(args)
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_checksums_determinism_and_output_collision(tmp_path):
    first = _run_small_suite(tmp_path / "first")
    second = _run_small_suite(tmp_path / "second")
    assert first["outer_split_counts"] == {"train": 3, "val": 0, "test": 1}
    assert first["realized_regime_counts"] == second["realized_regime_counts"]
    assert first["realized_K_counts"] == second["realized_K_counts"]
    assert first["output_checksum"] == second["output_checksum"]
    assert len(first["file_checksums"]) == 4
    for relative, expected in first["file_checksums"].items():
        payload = (tmp_path / "first" / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
    with pytest.raises(ValueError, match="empty --out_dir"):
        _run_small_suite(tmp_path / "first")


def test_profile_family_rejection(tmp_path):
    with pytest.raises(SystemExit):
        generator.main([
            "--task_family", "linear_classification",
            "--profile", "linear_stat_aware",
            "--out_dir", str(tmp_path / "bad"),
            "--n_datasets", "1",
        ])


def test_regression_omitted_and_explicit_task_family_are_equivalent(tmp_path):
    common = [
        "--profile", "legacy",
        "--n_datasets", "2",
        "--base_seed", "31",
    ]
    omitted = tmp_path / "omitted"
    explicit = tmp_path / "explicit"
    generator.main([*common, "--out_dir", str(omitted)])
    generator.main([
        "--task_family", "linear_regression",
        *common,
        "--out_dir", str(explicit),
    ])
    omitted_files = sorted(omitted.rglob("*.parquet"))
    explicit_files = sorted(explicit.rglob("*.parquet"))
    assert [p.relative_to(omitted) for p in omitted_files] == [
        p.relative_to(explicit) for p in explicit_files
    ]
    for left, right in zip(omitted_files, explicit_files):
        assert pq.read_table(left).to_pydict() == pq.read_table(right).to_pydict()
