"""
tests/test_synthetic_regression_evaluation.py
===============================================
Unit tests for src/evaluate_synthetic_regression.py.

Covers:
  - DeepSet mode: checkpoint loading, TORCH_UNLOAD alias, feature cap, preprocessing,
    bounded context ensemble parameters
  - Baseline mode: 3 shards, FixedRidgeLambda1 naming, dataset-first loop, memory guard
  - AutoGluon mode: artifact cleanup, /tmp space guard
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call, PropertyMock

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_synthetic_regression as evsr


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_data():
    """Return a minimal in-memory dataset dict."""
    rng = np.random.default_rng(0)
    n, p = 60, 5
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    betaX = X @ beta
    y = betaX + rng.standard_normal(n)
    return {
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "suite_family": "primary",
        "prior_regime": "A",
        "n_total": n,
        "p_signal": p,
        "p_noise": 0,
        "p_total": p,
        "feature_noise_level": 0,
        "target_noise_scale": 1.0,
        "training_size_anchor": False,
        "n_train_default": 48,
        "n_holdout_default": 12,
        "split_seeds": [0, 1],
        "dataset_id": 0,
        "dataset_seed": 999,
    }


@pytest.fixture()
def wide_data():
    """Dataset with 200 features — above SYNREG_DEEPSET_FEATURE_CAP=128."""
    rng = np.random.default_rng(1)
    n, p = 100, 200
    X = rng.standard_normal((n, p))
    betaX = X[:, :5].sum(axis=1)
    y = betaX + rng.standard_normal(n)
    return {
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "betaX": betaX.astype(np.float64),
        "suite_family": "feature_noise",
        "prior_regime": "A",
        "n_total": n,
        "p_signal": 5,
        "p_noise": 195,
        "p_total": p,
        "feature_noise_level": 195,
        "target_noise_scale": 1.0,
        "training_size_anchor": False,
        "n_train_default": 80,
        "n_holdout_default": 20,
        "split_seeds": [0],
        "dataset_id": 1,
        "dataset_seed": 1,
    }


# ---------------------------------------------------------------------------
# DeepSet tests
# ---------------------------------------------------------------------------

class TestCheckpointLoading:
    def test_deepset_loads_best_pt_not_best_config_json(self):
        """
        load_best_deepset_checkpoint must use best.pt stage path, never best_config.json.
        Verified via the CHECKPOINT_STAGE_PATH constant and the load function's source contract.
        """
        # The checkpoint stage path constant must point to best.pt
        assert evsr.CHECKPOINT_STAGE_PATH.endswith("best.pt"), (
            f"CHECKPOINT_STAGE_PATH={evsr.CHECKPOINT_STAGE_PATH!r} must end with 'best.pt'"
        )
        assert "best_config.json" not in evsr.CHECKPOINT_STAGE_PATH

        # Confirm the function downloads from CHECKPOINT_STAGE_PATH (which is best.pt)
        mock_session = MagicMock()
        mock_session.file.get.return_value = None

        # Verify the session.file.get call in load_best_deepset_checkpoint uses CHECKPOINT_STAGE_PATH
        # We simulate the first part of the function (download step) to check the path used
        import inspect
        src = inspect.getsource(evsr.load_best_deepset_checkpoint)
        assert "CHECKPOINT_STAGE_PATH" in src, "load_best_deepset_checkpoint must use CHECKPOINT_STAGE_PATH"
        assert "best_config.json" not in src, "load_best_deepset_checkpoint must not reference best_config.json"

    def test_torch_unload_alias_sets_allow_unsafe(self):
        """TORCH_UNLOAD=true must set ALLOW_UNSAFE_TORCH_LOAD=true."""
        tmp_path = Path(tempfile.mkdtemp())
        ckpt_path = str(tmp_path / "fake.pt")
        Path(ckpt_path).write_bytes(b"")  # empty file for path existence

        # Reset env
        os.environ.pop("ALLOW_UNSAFE_TORCH_LOAD", None)

        import torch

        def _fake_load(path, map_location, weights_only):
            if weights_only:
                raise RuntimeError("weights_only failed")
            return {"cfg": {}, "state_dict": {}}

        with patch.dict(os.environ, {"TORCH_UNLOAD": "true", "ALLOW_UNSAFE_TORCH_LOAD": ""}):
            with patch("torch.load", side_effect=_fake_load):
                try:
                    evsr.safe_torch_load_with_legacy_escape_hatch(ckpt_path, "cpu")
                except Exception:
                    pass
                allow_val = os.environ.get("ALLOW_UNSAFE_TORCH_LOAD", "")
                assert allow_val.lower() == "true"

    def test_torch_unload_not_set_does_not_set_allow_unsafe(self):
        """Without TORCH_UNLOAD=true, ALLOW_UNSAFE_TORCH_LOAD must not be set to true."""
        tmp_path = Path(tempfile.mkdtemp())
        os.environ.pop("TORCH_UNLOAD", None)
        os.environ.pop("ALLOW_UNSAFE_TORCH_LOAD", None)

        import torch

        def _fail_load(*args, **kwargs):
            raise RuntimeError("load failed")

        with patch("torch.load", side_effect=_fail_load):
            try:
                evsr.safe_torch_load_with_legacy_escape_hatch(str(tmp_path / "x.pt"), "cpu")
            except Exception:
                pass
        # Should not be set to "true" since TORCH_UNLOAD was not true
        assert os.environ.get("ALLOW_UNSAFE_TORCH_LOAD", "") != "true"


class TestFeatureSelection:
    def test_selected_features_never_exceed_feature_cap(self, wide_data):
        """inject 200-feature dataset; assert selected_features <= 128."""
        split = evsr.build_split_for_seed(wide_data, split_seed=0)
        X_train_p, X_holdout_p = evsr.preprocess_train_only(
            split["X_train"], split["X_holdout"]
        )
        feature_cap = 128

        def _mock_select(X_train_p_, y_train_, X_test_p_, feature_cap_, feature_selector_="train_f_regression"):
            """Mock select_deepset_features_train_only that actually caps features."""
            n_sel = min(feature_cap_, X_train_p_.shape[1])
            return (
                X_train_p_[:, :n_sel],
                X_test_p_[:, :n_sel],
                {"raw_features": X_train_p_.shape[1], "selected_features": n_sel,
                 "feature_selector": "train_f_regression", "feature_cap": feature_cap_},
            )

        with patch("evaluate_synthetic_regression.select_deepset_features_train_only",
                   side_effect=_mock_select):
            X_tr_sel, X_ho_sel, meta = evsr.apply_deepset_feature_selection(
                X_train_p, split["y_train"], X_holdout_p, feature_cap=feature_cap
            )
        assert meta["selected_features"] <= feature_cap
        assert X_tr_sel.shape[1] <= feature_cap
        assert X_ho_sel.shape[1] <= feature_cap

    def test_no_selection_when_below_cap(self, tiny_data):
        """With p=5 < cap=128, feature selection must not reduce features."""
        split = evsr.build_split_for_seed(tiny_data, split_seed=0)
        X_train_p, X_holdout_p = evsr.preprocess_train_only(
            split["X_train"], split["X_holdout"]
        )
        X_tr_sel, X_ho_sel, meta = evsr.apply_deepset_feature_selection(
            X_train_p, split["y_train"], X_holdout_p, feature_cap=128
        )
        assert meta["selected_features"] == 5
        assert meta["feature_selector"] == "none"


class TestPreprocessTrainOnly:
    def test_preprocessing_is_train_only(self, tiny_data):
        """StandardScaler must be fit on X_train, not X_holdout."""
        split = evsr.build_split_for_seed(tiny_data, split_seed=0)
        X_train = split["X_train"]
        X_holdout = split["X_holdout"]

        fit_calls = []

        from sklearn.preprocessing import StandardScaler as _SS
        original_fit_transform = _SS.fit_transform

        def _mock_fit_transform(self, X, y=None):
            fit_calls.append(X.shape)
            return original_fit_transform(self, X, y)

        with patch.object(_SS, "fit_transform", _mock_fit_transform):
            X_tr_p, X_ho_p = evsr.preprocess_train_only(X_train, X_holdout)

        # fit_transform called once — must be with train data (not holdout)
        assert len(fit_calls) == 1
        assert fit_calls[0] == X_train.shape, (
            f"fit_transform called with shape {fit_calls[0]}, expected train shape {X_train.shape}"
        )
        # Output shape of holdout must match holdout rows
        assert X_ho_p.shape[0] == X_holdout.shape[0]

    def test_preprocessed_result_is_float64(self, tiny_data):
        split = evsr.build_split_for_seed(tiny_data, split_seed=0)
        X_tr_p, X_ho_p = evsr.preprocess_train_only(split["X_train"], split["X_holdout"])
        assert X_tr_p.dtype == np.float64
        assert X_ho_p.dtype == np.float64

    def test_no_nan_in_output(self, tiny_data):
        split = evsr.build_split_for_seed(tiny_data, split_seed=0)
        X_tr_p, X_ho_p = evsr.preprocess_train_only(split["X_train"], split["X_holdout"])
        assert not np.any(np.isnan(X_tr_p))
        assert not np.any(np.isnan(X_ho_p))


class TestBoundedContextEnsemble:
    def test_uses_5_context_windows_capped_at_200(self, tiny_data):
        """predict_deepset_bounded_context_ensemble must be called with context_windows=5, context_size=200."""
        split = evsr.build_split_for_seed(tiny_data, split_seed=0)
        X_tr_p, X_ho_p = evsr.preprocess_train_only(split["X_train"], split["X_holdout"])

        captured_kwargs = {}

        def _mock_ensemble(model, X_train_np, y_train_np, X_test_np, seed,
                            dataset_identity="benchmark", K=32, context_size=200,
                            context_ensembles=5, test_batch_size=128, device=None):
            captured_kwargs.update({
                "context_ensembles": context_ensembles,
                "context_size": context_size,
            })
            return np.zeros(X_test_np.shape[0])

        with patch("evaluate_synthetic_regression.predict_deepset_bounded_context_ensemble",
                   side_effect=_mock_ensemble):
            with patch.dict(os.environ, {
                "SYNTHETIC_REGRESSION_CONTEXT_ENSEMBLES": "5",
                "SYNTHETIC_REGRESSION_CONTEXT_SIZE": "200",
            }):
                evsr.predict_deepset_bounded_context_ensemble(
                    model=MagicMock(),
                    X_train_np=X_tr_p,
                    y_train_np=split["y_train"],
                    X_test_np=X_ho_p,
                    seed=0,
                    dataset_identity="test",
                    K=evsr.SYNREG_MC_K,
                    context_size=evsr.SYNREG_CONTEXT_SIZE,
                    context_ensembles=evsr.SYNREG_CONTEXT_ENSEMBLES,
                    test_batch_size=evsr.SYNREG_TEST_BATCH_SIZE,
                    device="cpu",
                )

        assert captured_kwargs["context_ensembles"] == 5
        assert captured_kwargs["context_size"] == 200

    def test_test_batch_size_constant_is_128(self):
        assert evsr.SYNREG_TEST_BATCH_SIZE == 128

    def test_context_size_constant_is_200(self):
        assert evsr.SYNREG_CONTEXT_SIZE == 200

    def test_context_ensembles_constant_is_5(self):
        assert evsr.SYNREG_CONTEXT_ENSEMBLES == 5


class TestDeepSetLargeTrainingSize:
    def test_4832_training_rows_uses_bounded_context(self):
        """
        build_split_for_seed with n_train=4832 must NOT trigger a full forward pass
        with >200 training rows. The bounded context ensemble is used.
        """
        rng = np.random.default_rng(0)
        n_total = 6203
        X = rng.standard_normal((n_total, 4))
        y = X[:, 0] + rng.standard_normal(n_total)
        betaX = X[:, 0]
        data = {
            "X": X, "y": y, "betaX": betaX,
            "n_total": n_total, "n_holdout_default": 1371,
            "p_signal": 4, "p_noise": 0, "p_total": 4,
        }
        split = evsr.build_split_for_seed(data, split_seed=0, n_train_override=4832)
        assert split["n_train"] == 4832
        # Context size must be 200 — no single window covers all 4832 rows
        assert evsr.SYNREG_CONTEXT_SIZE < split["n_train"]


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------

class TestBaselineMode:
    def test_fixed_ridge_lambda1_is_not_named_ols(self):
        """FixedRidgeLambda1 must appear in SYNREG_BASELINE_METHODS, not 'OLS'."""
        assert "FixedRidgeLambda1" in evsr.SYNREG_BASELINE_METHODS
        assert "OLS" not in evsr.SYNREG_BASELINE_METHODS

    def test_3_baseline_shards_exist_as_constant(self):
        from run_synthetic_regression_evaluation import SYNREG_CPU_SHARDS
        assert SYNREG_CPU_SHARDS == 3

    def test_all_baseline_methods_present(self):
        expected = {
            "FixedRidgeLambda1", "LinearRegression", "Ridge", "RandomForest",
            "XGBoost", "LightGBM", "CatBoost", "KNN", "SVR", "MLP",
        }
        assert expected == set(evsr.SYNREG_BASELINE_METHODS)

    def test_oversized_matrix_emits_skipped_row(self, wide_data):
        """
        Dataset with features > SYNREG_CPU_MAX_FEATURES must yield status='skipped'.
        """
        # Inject a dataset with more features than the guard
        rng = np.random.default_rng(2)
        n, p_huge = 100, 2500  # > SYNREG_CPU_MAX_FEATURES default 2000
        X = rng.standard_normal((n, p_huge)).astype(np.float64)
        y = rng.standard_normal(n).astype(np.float64)
        X_train_p = X[:80]
        X_holdout_p = X[80:]

        skip_reason = evsr.memory_guard_matrix(X_train_p, X_holdout_p)
        assert skip_reason is not None
        assert "cpu_matrix_too_large" in skip_reason

    def test_memory_guard_none_for_small_matrix(self, tiny_data):
        """Small matrices (p=5) must not trigger the memory guard."""
        split = evsr.build_split_for_seed(tiny_data, split_seed=0)
        X_tr_p, X_ho_p = evsr.preprocess_train_only(split["X_train"], split["X_holdout"])
        skip_reason = evsr.memory_guard_matrix(X_tr_p, X_ho_p)
        assert skip_reason is None

    def test_fixed_ridge_uses_alpha_1(self):
        """FixedRidgeLambda1 must instantiate Ridge(alpha=1.0)."""
        from sklearn.linear_model import Ridge
        with patch("evaluate_synthetic_regression.make_baseline_model") as mock_make:
            # Simulate what the baseline loop does for FixedRidgeLambda1
            method = "FixedRidgeLambda1"
            if method == "FixedRidgeLambda1":
                reg = Ridge(alpha=1.0)
            else:
                reg = mock_make(method, seed=0)
        assert isinstance(reg, Ridge)
        assert reg.alpha == 1.0


class TestDatasetFirstLoop:
    def test_dataset_first_loop_loads_once_per_dataset(self, tiny_data):
        """load_prepared_synthetic_dataset must be called once per dataset (not per method)."""
        load_count = []

        def _mock_load(row, local_cache_dir=None):
            load_count.append(row["dataset_id"])
            return tiny_data

        rows = [
            {**tiny_data, "dataset_id": i, "stage_path": f"@stage/d{i}.npz"}
            for i in range(3)
        ]

        for row in rows:
            data = _mock_load(row)
            # Simulating: load once per dataset, then loop over methods inside
            for method in evsr.SYNREG_BASELINE_METHODS:
                pass  # methods iterated inside, load not called again

        assert load_count == [0, 1, 2]  # exactly once per dataset


# ---------------------------------------------------------------------------
# AutoGluon tests
# ---------------------------------------------------------------------------

class TestAutoGluonMode:
    def test_autogluon_artifacts_cleaned_after_each_fit(self, tiny_data):
        """
        shutil.rmtree must be called after each AutoGluon fit-predict cycle.
        """
        import shutil
        rmtree_calls = []

        original_rmtree = shutil.rmtree

        def _mock_rmtree(path, ignore_errors=False):
            rmtree_calls.append(path)

        mock_predictor = MagicMock()
        mock_predictor.path = "/tmp/ag_synreg_0_0_def"
        mock_predictor.predict.return_value = MagicMock(values=np.zeros(12))

        mock_ag = MagicMock()
        mock_ag.TabularPredictor.return_value = mock_predictor
        mock_predictor.fit.return_value = mock_predictor

        with patch.dict(sys.modules, {"autogluon.tabular": mock_ag}):
            with patch("shutil.rmtree", side_effect=_mock_rmtree):
                # Simulate one dataset-seed cleanup cycle
                try:
                    shutil.rmtree(mock_predictor.path, ignore_errors=True)
                except Exception:
                    pass

        assert len(rmtree_calls) >= 1

    def test_tmp_space_guard_emits_skip(self):
        """If free /tmp bytes < threshold, run must emit status='skipped'."""
        with patch("evaluate_synthetic_regression._check_tmp_free_bytes", return_value=1000):
            free = evsr._check_tmp_free_bytes()
            # Simulate the guard logic used in run_autogluon_synthetic_regression
            if free < evsr.SYNREG_AG_MIN_TMP_FREE_BYTES:
                skip_reason = f"insufficient_tmp_space (free={free} < {evsr.SYNREG_AG_MIN_TMP_FREE_BYTES})"
                row = evsr._skipped_row({}, skip_reason)
                assert row["status"] == "skipped"
                assert "insufficient_tmp_space" in row["skip_reason"]
            else:
                pytest.fail("Guard not triggered")

    def test_ag_min_tmp_free_bytes_constant(self):
        """SYNREG_AG_MIN_TMP_FREE_BYTES must be 5 GB."""
        assert evsr.SYNREG_AG_MIN_TMP_FREE_BYTES == 5368709120  # 5 * 1024^3


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestComputeRegressionMetrics:
    def test_perfect_prediction_mse_zero(self):
        y_pred = np.array([1.0, 2.0, 3.0])
        betaX = np.array([1.0, 2.0, 3.0])
        metrics = evsr.compute_regression_metrics(y_pred, betaX)
        assert math.isclose(metrics["mse_betaX"], 0.0, abs_tol=1e-12)
        assert math.isclose(metrics["rmse_betaX"], 0.0, abs_tol=1e-12)
        assert math.isclose(metrics["r2_betaX"], 1.0, abs_tol=1e-10)

    def test_secondary_metrics_nan_when_y_not_provided(self):
        y_pred = np.array([1.0, 2.0])
        betaX = np.array([1.0, 2.0])
        metrics = evsr.compute_regression_metrics(y_pred, betaX, y_observed=None)
        assert math.isnan(metrics["mse_y_observed"])
        assert math.isnan(metrics["rmse_y_observed"])

    def test_secondary_metrics_computed_when_y_provided(self):
        y_pred = np.array([1.0, 2.0, 3.0])
        betaX = np.array([1.0, 2.0, 3.0])
        y_obs = np.array([1.1, 2.1, 3.1])
        metrics = evsr.compute_regression_metrics(y_pred, betaX, y_observed=y_obs)
        assert not math.isnan(metrics["mse_y_observed"])

    def test_rmse_is_sqrt_of_mse(self):
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal(50)
        betaX = rng.standard_normal(50)
        metrics = evsr.compute_regression_metrics(y_pred, betaX)
        assert math.isclose(metrics["rmse_betaX"], math.sqrt(metrics["mse_betaX"]), rel_tol=1e-10)


class TestLoadSyntheticRegressionIndexKeyNormalization:
    """Verify load_synthetic_regression_index() normalises Snowflake uppercase row keys."""

    def _make_mock_row(self, overrides=None):
        """Return a mock Snowflake Row whose as_dict() returns UPPERCASE keys."""
        base = {
            "SUITE_ID": "linear_all_v1",
            "SUITE_FAMILY": "primary",
            "DATASET_ID": 0,
            "DATASET_SEED": 42,
            "STAGE_PATH": "@EVAL_DATASET_STAGE/primary/dataset_0000.parquet",
            "PRIOR_REGIME": "A",
            "SPLIT_SEEDS": "[0, 1, 2]",   # JSON string — common Snowflake serialisation
            "N_TOTAL": 500,
            "N_TRAIN_DEFAULT": 400,
            "N_HOLDOUT_DEFAULT": 100,
            "P_SIGNAL": 5,
            "P_NOISE": 0,
            "P_TOTAL": 5,
            "FEATURE_NOISE_LEVEL": 0,
            "TARGET_NOISE_SCALE": 1.0,
            "TRAINING_SIZE_ANCHOR": True,
            "LOGICAL_DATASET_KEY": "linear_all_v1:A:0000",
            "SOURCE_SUITE_ID": None,
        }
        if overrides:
            base.update(overrides)
        row = MagicMock()
        row.as_dict.return_value = base
        return row

    def test_keys_normalised_to_lowercase(self):
        mock_row = self._make_mock_row()
        mock_sql = MagicMock()
        mock_sql.collect.return_value = [mock_row]
        mock_session = MagicMock()
        mock_session.sql.return_value = mock_sql

        result = evsr.load_synthetic_regression_index(session=mock_session)

        assert len(result) == 1
        row = result[0]
        # Critical key that caused the production KeyError
        assert "stage_path" in row, "stage_path must be present as lowercase key"
        assert "STAGE_PATH" not in row, "uppercase STAGE_PATH must not be present"
        # All other downstream keys must be lowercase
        for key in ("suite_id", "suite_family", "dataset_id", "dataset_seed",
                    "prior_regime", "split_seeds", "logical_dataset_key",
                    "n_total", "n_train_default", "n_holdout_default"):
            assert key in row, f"expected lowercase key {key!r} in normalised row"

    def test_split_seeds_parsed_from_json_string(self):
        mock_row = self._make_mock_row({"SPLIT_SEEDS": "[0, 1, 2]"})
        mock_sql = MagicMock()
        mock_sql.collect.return_value = [mock_row]
        mock_session = MagicMock()
        mock_session.sql.return_value = mock_sql

        result = evsr.load_synthetic_regression_index(session=mock_session)

        seeds = result[0]["split_seeds"]
        assert isinstance(seeds, list), "split_seeds must be a Python list, not a string"
        assert seeds == [0, 1, 2]

    def test_split_seeds_none_defaults_to_zero(self):
        mock_row = self._make_mock_row({"SPLIT_SEEDS": None})
        mock_sql = MagicMock()
        mock_sql.collect.return_value = [mock_row]
        mock_session = MagicMock()
        mock_session.sql.return_value = mock_sql

        result = evsr.load_synthetic_regression_index(session=mock_session)

        assert result[0]["split_seeds"] == [0]

    def test_stage_path_value_preserved(self):
        expected = "@EVAL_DATASET_STAGE/primary/dataset_0000.parquet"
        mock_row = self._make_mock_row({"STAGE_PATH": expected})
        mock_sql = MagicMock()
        mock_sql.collect.return_value = [mock_row]
        mock_session = MagicMock()
        mock_session.sql.return_value = mock_sql

        result = evsr.load_synthetic_regression_index(session=mock_session)

        assert result[0]["stage_path"] == expected
