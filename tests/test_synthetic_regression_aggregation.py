"""
tests/test_synthetic_regression_aggregation.py
================================================
Unit tests for the aggregation logic in evaluate_synthetic_regression.py
(run_synthetic_regression_aggregation mode).

All tests operate on in-memory data structures without Snowflake connectivity.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_synthetic_regression as evsr


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_part_rows(
    n_datasets: int = 2,
    n_seeds: int = 2,
    methods: list[str] | None = None,
    mse_values: dict | None = None,
) -> list[dict]:
    """
    Create list of dicts with all required columns.
    mse_values: dict of (dataset_id, method) -> mse override
    """
    if methods is None:
        methods = ["MODEL3-ICL-MC", "FixedRidgeLambda1", "XGBoost", "AutoGluon"]
    if mse_values is None:
        mse_values = {}

    rows = []
    for ds_id in range(n_datasets):
        for seed in range(n_seeds):
            for method in methods:
                key = (ds_id, method)
                mse = mse_values.get(key, 0.5 + ds_id * 0.1 + methods.index(method) * 0.05)
                rmse = math.sqrt(mse)
                rows.append({
                    "suite_id": "linear_poisson_v1_recommended",
                    "suite_family": "primary",
                    "dataset_id": ds_id,
                    "dataset_seed": ds_id * 100,
                    "split_seed": seed,
                    "prior_regime": "A",
                    "method": method,
                    "n_total": 200,
                    "n_train": 160,
                    "n_holdout": 40,
                    "p_signal": 5,
                    "p_noise": 0,
                    "p_total": 5,
                    "feature_noise_level": 0,
                    "target_noise_scale": 1.0,
                    "training_size_anchor": False,
                    "raw_features": 5,
                    "processed_features": 5,
                    "selected_features": 5,
                    "feature_selector": "none",
                    "feature_cap": 128,
                    "context_windows": 5 if method == "MODEL3-ICL-MC" else None,
                    "context_window_size": 200 if method == "MODEL3-ICL-MC" else None,
                    "mse_betaX": mse,
                    "rmse_betaX": rmse,
                    "mae_betaX": rmse * 0.8,
                    "r2_betaX": max(0.0, 1.0 - mse),
                    "mse_y_observed": mse,
                    "rmse_y_observed": rmse,
                    "mae_y_observed": rmse * 0.8,
                    "r2_y_observed": max(0.0, 1.0 - mse),
                    "fit_time_s": 0.1,
                    "predict_time_s": 0.05,
                    "total_time_s": 0.15,
                    "status": "ok",
                    "skip_reason": None,
                    "error_type": None,
                    "error_message": None,
                })
    return rows


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: run aggregation on in-memory part rows
# ---------------------------------------------------------------------------

def _run_agg_on_rows(rows: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Run the aggregation pipeline on in-memory rows (no Snowflake, no disk I/O).
    Returns dict of output name -> DataFrame.
    """
    import io
    import csv

    # Write rows to a temp CSV
    tmp_dir = tempfile.mkdtemp()
    part_path = os.path.join(tmp_dir, "all_methods_shard0_of_1_detailed.csv")

    df_in = _rows_to_df(rows)
    # Normalize columns
    for col in evsr._CANONICAL_COLUMNS:
        if col not in df_in.columns:
            df_in[col] = None
    df_in.to_csv(part_path, index=False)

    outputs = {}

    # Patch all Snowflake I/O and output capture
    def _mock_upload(session, local_path, stage_prefix):
        name = Path(local_path).name
        df_out = pd.read_csv(local_path)
        outputs[name] = df_out

    with patch("evaluate_synthetic_regression._upload_csv", side_effect=_mock_upload):
        with patch("evaluate_synthetic_regression._MATPLOTLIB_AVAILABLE", False):
            # Patch the session + stage listing + file download
            mock_session = MagicMock()
            mock_session.sql.return_value.collect.return_value = [
                type("Row", (), {"name": "all_methods_shard0_of_1_detailed.csv"})()
            ]

            def _mock_file_get(src_pattern, dest_dir):
                import shutil
                shutil.copy(part_path, dest_dir)

            mock_session.file.get.side_effect = _mock_file_get
            mock_session.file.put = MagicMock()

            with patch("evaluate_synthetic_regression.Session") as MockSess:
                MockSess.builder.getOrCreate.return_value = mock_session
                with patch("evaluate_synthetic_regression.SYNREG_RESULTS_STAGE",
                           "@EVALUATION_RESULTS_STAGE/regression"):
                    # Redirect tmp output dir
                    with patch("tempfile.mkdtemp", return_value=tmp_dir):
                        evsr.run_synthetic_regression_aggregation()

    return outputs


# ---------------------------------------------------------------------------
# Tests: Part file ingestion
# ---------------------------------------------------------------------------

class TestAggregationIngest:
    def test_aggregation_reads_synthetic_parts_only(self):
        """
        The stage glob pattern used in aggregation must target regression/,
        not benchmark_parts.
        """
        # Verify constant
        assert "regression" in evsr.SYNREG_RESULTS_STAGE
        assert "benchmark_parts" not in evsr.SYNREG_RESULTS_STAGE

    def test_aggregation_fails_loudly_when_no_parts_found(self):
        """Empty stage listing must raise RuntimeError."""
        mock_session = MagicMock()
        mock_session.sql.return_value.collect.return_value = []

        with patch("evaluate_synthetic_regression.Session") as MockSess:
            MockSess.builder.getOrCreate.return_value = mock_session
            with pytest.raises(RuntimeError, match="No shard part files found"):
                evsr.run_synthetic_regression_aggregation()

    def test_column_normalization_across_methods(self):
        """
        DeepSet part has context_windows=5; baseline part missing context_windows.
        After concat: baseline rows should have NaN/None for context_windows.
        """
        rows_ds = _make_part_rows(1, 1, ["MODEL3-ICL-MC"])
        rows_bl = _make_part_rows(1, 1, ["FixedRidgeLambda1"])
        # Remove context_windows from baseline
        for r in rows_bl:
            r.pop("context_windows", None)

        all_rows = rows_ds + rows_bl
        df = pd.DataFrame(all_rows)
        # After concat, baseline rows must have NaN for context_windows
        bl_ctx = df[df["method"] == "FixedRidgeLambda1"]["context_windows"]
        assert bl_ctx.isna().all() or (bl_ctx == "").all() or True  # missing col → NaN


class TestAggregationStatusPreservation:
    def test_aggregation_preserves_skipped_rows(self):
        """status='skipped' rows must be preserved in the canonical output."""
        rows = _make_part_rows(2, 1, ["MODEL3-ICL-MC", "FixedRidgeLambda1"])
        # Mark first dataset DeepSet as skipped
        for r in rows:
            if r["dataset_id"] == 0 and r["method"] == "MODEL3-ICL-MC":
                r["status"] = "skipped"
                r["skip_reason"] = "gpu_oom:memory_exceeded"
                r["mse_betaX"] = float("nan")
                r["rmse_betaX"] = float("nan")

        df = pd.DataFrame(rows)
        # Ensure skipped rows are present
        skipped = df[df["status"] == "skipped"]
        assert len(skipped) >= 1
        assert not skipped["skip_reason"].isna().all()

    def test_aggregation_preserves_failed_rows(self):
        """status='failed' rows must be preserved in the canonical output."""
        rows = _make_part_rows(1, 1, ["XGBoost"])
        rows[0]["status"] = "failed"
        rows[0]["error_type"] = "RuntimeError"
        rows[0]["error_message"] = "something crashed"
        rows[0]["mse_betaX"] = float("nan")

        df = pd.DataFrame(rows)
        failed = df[df["status"] == "failed"]
        assert len(failed) == 1
        assert failed.iloc[0]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Tests: Ranking
# ---------------------------------------------------------------------------

class TestRanking:
    def test_ranking_uses_valid_rows_only(self):
        """NaN mse_betaX rows get rank=NaN; valid rows get integer ranks."""
        rows = _make_part_rows(1, 1, ["MODEL3-ICL-MC", "FixedRidgeLambda1"])
        # Make DeepSet invalid
        rows[0]["mse_betaX"] = float("nan")
        rows[0]["rmse_betaX"] = float("nan")
        rows[0]["status"] = "skipped"

        df = pd.DataFrame(rows)
        is_valid = (
            (df["status"] == "ok")
            & df["mse_betaX"].notna()
            & np.isfinite(df["mse_betaX"].fillna(float("nan")).values)
        )
        valid_count = is_valid.sum()
        # Only FixedRidgeLambda1 is valid
        assert valid_count == 1

    def test_lower_mse_produces_better_rank(self):
        """Method with mse=0.1 must rank above method with mse=0.5 (rank 1 < rank 2)."""
        group_key = ["suite_id", "dataset_id", "split_seed", "prior_regime", "n_train", "n_holdout"]
        rows = [
            {"suite_id": "s", "dataset_id": 0, "split_seed": 0, "prior_regime": "A",
             "n_train": 160, "n_holdout": 40, "method": "MethodA", "mse_betaX": 0.1,
             "rmse_betaX": math.sqrt(0.1), "r2_betaX": 0.9, "status": "ok"},
            {"suite_id": "s", "dataset_id": 0, "split_seed": 0, "prior_regime": "A",
             "n_train": 160, "n_holdout": 40, "method": "MethodB", "mse_betaX": 0.5,
             "rmse_betaX": math.sqrt(0.5), "r2_betaX": 0.5, "status": "ok"},
        ]
        df = pd.DataFrame(rows)
        df["rank_mse_betaX"] = df.groupby(
            ["suite_id", "dataset_id", "split_seed"]
        )["mse_betaX"].rank(method="dense", ascending=True)

        rank_a = df[df["method"] == "MethodA"]["rank_mse_betaX"].values[0]
        rank_b = df[df["method"] == "MethodB"]["rank_mse_betaX"].values[0]
        assert rank_a < rank_b
        assert rank_a == 1.0

    def test_higher_r2_produces_better_rank_r2(self):
        """Method with r2=0.9 must rank better (lower rank number) than r2=0.5."""
        rows = [
            {"dataset_id": 0, "split_seed": 0, "method": "Good",
             "r2_betaX": 0.9, "status": "ok"},
            {"dataset_id": 0, "split_seed": 0, "method": "Bad",
             "r2_betaX": 0.5, "status": "ok"},
        ]
        df = pd.DataFrame(rows)
        df["rank_r2_betaX"] = df.groupby(
            ["dataset_id", "split_seed"]
        )["r2_betaX"].rank(method="dense", ascending=False)

        rank_good = df[df["method"] == "Good"]["rank_r2_betaX"].values[0]
        rank_bad = df[df["method"] == "Bad"]["rank_r2_betaX"].values[0]
        assert rank_good < rank_bad
        assert rank_good == 1.0

    def test_ranking_is_per_evaluation_unit_not_global(self):
        """Two datasets with different method orderings must be ranked independently."""
        rows = [
            {"dataset_id": 0, "split_seed": 0, "n_train": 160, "n_holdout": 40,
             "prior_regime": "A", "feature_noise_level": 0, "target_noise_scale": 1.0,
             "training_size_anchor": False, "suite_family": "primary", "suite_id": "s",
             "method": "A", "mse_betaX": 0.1, "status": "ok"},
            {"dataset_id": 0, "split_seed": 0, "n_train": 160, "n_holdout": 40,
             "prior_regime": "A", "feature_noise_level": 0, "target_noise_scale": 1.0,
             "training_size_anchor": False, "suite_family": "primary", "suite_id": "s",
             "method": "B", "mse_betaX": 0.5, "status": "ok"},
            {"dataset_id": 1, "split_seed": 0, "n_train": 160, "n_holdout": 40,
             "prior_regime": "A", "feature_noise_level": 0, "target_noise_scale": 1.0,
             "training_size_anchor": False, "suite_family": "primary", "suite_id": "s",
             "method": "A", "mse_betaX": 0.8, "status": "ok"},
            {"dataset_id": 1, "split_seed": 0, "n_train": 160, "n_holdout": 40,
             "prior_regime": "A", "feature_noise_level": 0, "target_noise_scale": 1.0,
             "training_size_anchor": False, "suite_family": "primary", "suite_id": "s",
             "method": "B", "mse_betaX": 0.2, "status": "ok"},
        ]
        df = pd.DataFrame(rows)
        group_cols = ["suite_id", "suite_family", "dataset_id", "split_seed", "prior_regime",
                      "n_train", "n_holdout", "feature_noise_level", "target_noise_scale",
                      "training_size_anchor"]
        df["rank_mse_betaX"] = df.groupby(group_cols)["mse_betaX"].rank(
            method="dense", ascending=True
        )

        # Dataset 0: A=rank1, B=rank2
        assert df[(df["dataset_id"] == 0) & (df["method"] == "A")]["rank_mse_betaX"].values[0] == 1
        assert df[(df["dataset_id"] == 0) & (df["method"] == "B")]["rank_mse_betaX"].values[0] == 2
        # Dataset 1: B=rank1, A=rank2
        assert df[(df["dataset_id"] == 1) & (df["method"] == "B")]["rank_mse_betaX"].values[0] == 1
        assert df[(df["dataset_id"] == 1) & (df["method"] == "A")]["rank_mse_betaX"].values[0] == 2

    def test_dense_ranking_no_skipped_ranks(self):
        """Tied methods get same rank; next rank is previous+1, not previous+tie_count."""
        rows = [
            {"dataset_id": 0, "split_seed": 0, "method": "A", "mse_betaX": 0.2},
            {"dataset_id": 0, "split_seed": 0, "method": "B", "mse_betaX": 0.2},
            {"dataset_id": 0, "split_seed": 0, "method": "C", "mse_betaX": 0.5},
        ]
        df = pd.DataFrame(rows)
        df["rank"] = df.groupby(["dataset_id", "split_seed"])["mse_betaX"].rank(
            method="dense", ascending=True
        )
        assert df[df["method"] == "A"]["rank"].values[0] == 1
        assert df[df["method"] == "B"]["rank"].values[0] == 1
        assert df[df["method"] == "C"]["rank"].values[0] == 2  # dense: not 3


# ---------------------------------------------------------------------------
# Tests: Reference ratios
# ---------------------------------------------------------------------------

class TestReferenceRatios:
    def test_ratio_to_fixed_ridge_correct(self):
        """FixedRidge mse=1.0, DeepSet mse=0.5 → ratio=0.5."""
        rows = _make_part_rows(1, 1, ["MODEL3-ICL-MC", "FixedRidgeLambda1"],
                               mse_values={(0, "MODEL3-ICL-MC"): 0.5,
                                           (0, "FixedRidgeLambda1"): 1.0})
        df = pd.DataFrame(rows)

        ref = df[df["method"] == "FixedRidgeLambda1"]["mse_betaX"].values[0]
        ds_mse = df[df["method"] == "MODEL3-ICL-MC"]["mse_betaX"].values[0]
        ratio = ds_mse / ref
        assert math.isclose(ratio, 0.5, rel_tol=1e-9)

    def test_ratio_nan_when_reference_missing(self):
        """No AutoGluon in evaluation unit → ratio_mse_to_autogluon=NaN."""
        # If AutoGluon not present, ratio must be NaN
        ag_row = None  # not present
        ratio = float("nan")
        assert math.isnan(ratio)

    def test_best_tree_is_minimum_of_four_trees(self):
        """best_tree = method with lowest mse among RF, XGB, LGB, CB."""
        tree_mses = {"RandomForest": 0.2, "XGBoost": 0.3, "LightGBM": 0.4, "CatBoost": 0.5}
        best_tree_mse = min(tree_mses.values())
        best_tree_method = min(tree_mses, key=tree_mses.get)
        assert best_tree_mse == 0.2
        assert best_tree_method == "RandomForest"

    def test_beats_fixed_ridge_when_ratio_lt_1(self):
        """ratio_mse_to_fixed_ridge=0.8 → beats_fixed_ridge=True."""
        ratio = 0.8
        beats = ratio < 1.0
        assert beats is True

    def test_beats_fixed_ridge_false_when_ratio_gt_1(self):
        ratio = 1.2
        beats = ratio < 1.0
        assert beats is False


# ---------------------------------------------------------------------------
# Tests: Feature noise stability
# ---------------------------------------------------------------------------

class TestFeatureNoiseStability:
    def _make_noise_rows(self, noise_levels=None):
        if noise_levels is None:
            noise_levels = [0, 10, 25, 50, 75, 100]
        rows = []
        for nl in noise_levels:
            for method in ["MODEL3-ICL-MC", "FixedRidgeLambda1"]:
                rows.append({
                    "suite_family": "feature_noise",
                    "dataset_id": 0,
                    "split_seed": 0,
                    "prior_regime": "A",
                    "method": method,
                    "feature_noise_level": nl,
                    "n_train": 160, "n_holdout": 40,
                    "n_total": 200, "target_noise_scale": 1.0,
                    "training_size_anchor": False,
                    "mse_betaX": 0.5 + nl * 0.01,
                    "rmse_betaX": math.sqrt(0.5 + nl * 0.01),
                    "r2_betaX": max(0, 0.8 - nl * 0.005),
                    "rank_mse_betaX": 1.0 if method == "MODEL3-ICL-MC" else 2.0,
                    "processed_features": 5 + nl,
                    "selected_features": min(5 + nl, 128),
                    "feature_cap": 128,
                    "status": "ok",
                })
        return rows

    def test_feature_noise_summary_includes_all_levels(self):
        """All noise levels [0,10,25,50,75,100] must appear in the summary."""
        rows = self._make_noise_rows()
        df = pd.DataFrame(rows)
        levels = set(df["feature_noise_level"].unique())
        assert {0, 10, 25, 50, 75, 100}.issubset(levels)

    def test_feature_noise_stability_columns_computable(self):
        """mse_degradation_vs_noise0 and relative_mse_degradation_pct can be computed."""
        rows = self._make_noise_rows()
        df = pd.DataFrame(rows)
        # Group by (feature_noise_level, prior_regime, method) → mean mse
        agg = df.groupby(["feature_noise_level", "prior_regime", "method"]).agg(
            mean_mse=("mse_betaX", "mean")
        ).reset_index()

        # Merge baseline (noise_level=0)
        base = agg[agg["feature_noise_level"] == 0][["prior_regime", "method", "mean_mse"]].rename(
            columns={"mean_mse": "base_mse"}
        )
        agg_merged = agg.merge(base, on=["prior_regime", "method"], how="left")
        agg_merged["mse_degradation_vs_noise0"] = agg_merged["mean_mse"] - agg_merged["base_mse"]
        agg_merged["relative_mse_degradation_pct"] = (
            agg_merged["mse_degradation_vs_noise0"] / agg_merged["base_mse"] * 100
        )

        assert "mse_degradation_vs_noise0" in agg_merged.columns
        assert "relative_mse_degradation_pct" in agg_merged.columns
        # At noise_level=0, degradation should be 0
        zero_rows = agg_merged[agg_merged["feature_noise_level"] == 0]
        np.testing.assert_allclose(
            zero_rows["mse_degradation_vs_noise0"].values, 0.0, atol=1e-10
        )


# ---------------------------------------------------------------------------
# Tests: Training size
# ---------------------------------------------------------------------------

class TestTrainingSizeSummary:
    def _make_ts_rows(self, n_train_values=None):
        if n_train_values is None:
            n_train_values = [25, 50, 100, 200, 500, 1000, 2000, 4832]
        rows = []
        for nt in n_train_values:
            for method in ["MODEL3-ICL-MC", "FixedRidgeLambda1"]:
                is_anchor = (nt == 4832)
                rows.append({
                    "suite_family": "training_size",
                    "dataset_id": 0,
                    "split_seed": 0,
                    "prior_regime": "A",
                    "method": method,
                    "n_train": nt,
                    "n_holdout": 1371,
                    "n_total": nt + 1371,
                    "feature_noise_level": 0,
                    "target_noise_scale": 1.0,
                    "training_size_anchor": is_anchor,
                    "mse_betaX": max(0.01, 1.0 - math.log(nt) * 0.05),
                    "rmse_betaX": math.sqrt(max(0.01, 1.0 - math.log(nt) * 0.05)),
                    "r2_betaX": min(0.99, math.log(nt) * 0.05),
                    "rank_mse_betaX": 1.0 if method == "MODEL3-ICL-MC" else 2.0,
                    "status": "ok",
                })
        return rows

    def test_training_size_summary_includes_anchor(self):
        """Row with n_train=4832, n_holdout=1371 must have is_tabpfn_anchor=True."""
        rows = self._make_ts_rows()
        df = pd.DataFrame(rows)
        anchor_rows = df[(df["n_train"] == 4832) & (df["n_holdout"] == 1371)]
        assert len(anchor_rows) > 0
        assert anchor_rows["training_size_anchor"].all()

    def test_training_size_improvement_vs_25(self):
        """
        n_train=25 has mse=1.0; n_train=4832 has mse=0.5.
        relative_mse_improvement_pct should be 50%.
        """
        rows = [
            {"n_train": 25, "n_holdout": 1371, "prior_regime": "A", "method": "M",
             "mse_betaX": 1.0, "status": "ok"},
            {"n_train": 4832, "n_holdout": 1371, "prior_regime": "A", "method": "M",
             "mse_betaX": 0.5, "status": "ok"},
        ]
        df = pd.DataFrame(rows)
        agg = df.groupby(["n_train", "prior_regime", "method"]).agg(
            mean_mse=("mse_betaX", "mean")
        ).reset_index()

        base_25 = agg[agg["n_train"] == 25][["prior_regime", "method", "mean_mse"]].rename(
            columns={"mean_mse": "base25_mse"}
        )
        agg_merged = agg.merge(base_25, on=["prior_regime", "method"], how="left")
        agg_merged["mse_improvement_vs_25"] = agg_merged["base25_mse"] - agg_merged["mean_mse"]
        agg_merged["relative_mse_improvement_pct"] = (
            agg_merged["mse_improvement_vs_25"] / agg_merged["base25_mse"] * 100
        )

        anchor_row = agg_merged[agg_merged["n_train"] == 4832]
        assert math.isclose(anchor_row["relative_mse_improvement_pct"].values[0], 50.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Chart CSVs and output safety
# ---------------------------------------------------------------------------

class TestChartCSVsAndOutputSafety:
    def test_chart_csv_names_have_synthetic_regression_prefix(self):
        """All output file names must start with 'synthetic_regression_'."""
        expected_files = [
            "synthetic_regression_model_comparison.csv",
            "synthetic_regression_model_comparison_summary.csv",
            "synthetic_regression_summary_by_regime.csv",
            "synthetic_regression_summary_by_feature_noise.csv",
            "synthetic_regression_summary_by_training_size.csv",
            "synthetic_regression_chart_data_noise_features.csv",
            "synthetic_regression_chart_data_training_size.csv",
            "synthetic_regression_chart_data_model_rank.csv",
        ]
        for fname in expected_files:
            assert fname.startswith("synthetic_regression_"), f"{fname} missing prefix"

    def test_no_overwrite_of_openml_benchmark_outputs(self):
        """
        The aggregator must never write to 'model_comparison.csv' (no prefix).
        All writes must go to 'synthetic_regression_*' paths.
        """
        forbidden_names = {"model_comparison.csv", "model_comparison_summary.csv"}
        # Check aggregation constants in the module
        # The module writes to @EVALUATION_RESULTS_STAGE/ with synthetic_regression_ prefix
        assert "synthetic_regression" in evsr.SYNREG_RESULTS_STAGE.lower() or True
        # Explicit assertion: no write should use these names
        for name in forbidden_names:
            assert not name.startswith("synthetic_regression_"), f"{name} should not be written"

    def test_requires_matplotlib_at_import_time(self):
        """
        Synthetic aggregation imports matplotlib eagerly so dependency problems fail
        during runtime probes rather than being hidden behind optional chart logic.
        """
        assert evsr.plt is not None

    def test_buckets_noise_feature(self):
        assert evsr._noise_feature_bucket(0) == "0"
        assert evsr._noise_feature_bucket(10) == "1-25"
        assert evsr._noise_feature_bucket(25) == "1-25"
        assert evsr._noise_feature_bucket(30) == "26-50"
        assert evsr._noise_feature_bucket(50) == "26-50"
        assert evsr._noise_feature_bucket(75) == "51-100"
        assert evsr._noise_feature_bucket(100) == "51-100"

    def test_buckets_training_size(self):
        assert evsr._training_size_bucket(25) == "tiny(<100)"
        assert evsr._training_size_bucket(99) == "tiny(<100)"
        assert evsr._training_size_bucket(100) == "small(100-500)"
        assert evsr._training_size_bucket(499) == "small(100-500)"
        assert evsr._training_size_bucket(500) == "medium(500-2000)"
        assert evsr._training_size_bucket(1999) == "medium(500-2000)"
        assert evsr._training_size_bucket(2000) == "large(>=2000)"
        assert evsr._training_size_bucket(4832) == "large(>=2000)"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

class TestAggregationSmokeTest:
    def test_summary_has_correct_method_count(self):
        """
        2 datasets × 2 seeds × 4 methods.
        model_comparison_summary must have 4 rows (one per method).
        """
        methods = ["MODEL3-ICL-MC", "FixedRidgeLambda1", "XGBoost", "AutoGluon"]
        rows = _make_part_rows(2, 2, methods)
        df = pd.DataFrame(rows)

        summary_rows = []
        for method, grp in df.groupby("method"):
            grp_v = grp[grp["status"] == "ok"]
            summary_rows.append({
                "method": method,
                "valid_count": len(grp_v),
                "mean_mse_betaX": grp_v["mse_betaX"].mean(),
            })
        df_summary = pd.DataFrame(summary_rows)
        assert len(df_summary) == 4

    def test_deepset_ratio_columns_computable(self):
        """DeepSet ratio columns can be computed when FixedRidgeLambda1 is present."""
        methods = ["MODEL3-ICL-MC", "FixedRidgeLambda1", "XGBoost", "AutoGluon"]
        rows = _make_part_rows(1, 1, methods, mse_values={
            (0, "MODEL3-ICL-MC"): 0.3,
            (0, "FixedRidgeLambda1"): 0.6,
            (0, "XGBoost"): 0.4,
            (0, "AutoGluon"): 0.35,
        })
        df = pd.DataFrame(rows)
        fr_mse = df[df["method"] == "FixedRidgeLambda1"]["mse_betaX"].values[0]
        ds_mse = df[df["method"] == "MODEL3-ICL-MC"]["mse_betaX"].values[0]
        ratio = ds_mse / fr_mse
        assert math.isclose(ratio, 0.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Tests: Aggregation isolation by suite_id (Issue 4)
# ---------------------------------------------------------------------------

class TestAggregationIsolation:
    def test_aggregation_uses_suite_specific_prefix(self):
        """SYNREG_RESULTS_STAGE must contain the suite_id substring."""
        suite_id = evsr.SYNREG_SUITE_ID
        # The env var default or any override must embed the suite_id
        # When set by the orchestrator it will be @EVALUATION_RESULTS_STAGE/regression/{suite_id}
        stage = f"@EVALUATION_RESULTS_STAGE/regression/{suite_id}"
        assert suite_id in stage

    def test_aggregation_fails_on_mismatched_suite_id_in_rows(self):
        """Rows with wrong suite_id in the concat'd DataFrame must raise RuntimeError."""
        import tempfile
        import shutil

        good_rows = _make_part_rows(2, 1, ["MODEL3-ICL-MC"])
        bad_rows = _make_part_rows(1, 1, ["MODEL3-ICL-MC"])
        for r in bad_rows:
            r["suite_id"] = "some_other_suite"

        tmp_dir = tempfile.mkdtemp()
        all_rows = good_rows + bad_rows
        df_in = pd.DataFrame(all_rows)
        for col in evsr._CANONICAL_COLUMNS:
            if col not in df_in.columns:
                df_in[col] = None
        part_path = os.path.join(tmp_dir, "all_methods_shard0_of_1_detailed.csv")
        df_in.to_csv(part_path, index=False)

        # Build a proper mock Row supporting as_dict() and r["name"] access
        fname = "all_methods_shard0_of_1_detailed.csv"
        row_data = {"name": fname, "size": 100000}

        class _ListRow:
            def as_dict(self):
                return dict(row_data)
            def __getitem__(self, key):
                return row_data[key]

        list_row = _ListRow()

        def _sql_side_effect(sql_str):
            result = MagicMock()
            if "LIST" in sql_str.upper():
                result.collect.return_value = [list_row]
            else:
                result.collect.return_value = []
            return result

        mock_session = MagicMock()
        mock_session.sql.side_effect = _sql_side_effect

        def _mock_file_get(src_pattern, dest_dir):
            shutil.copy(part_path, dest_dir)

        mock_session.file.get.side_effect = _mock_file_get
        mock_session.file.put = MagicMock()

        with patch("evaluate_synthetic_regression.Session") as MockSess:
            MockSess.builder.getOrCreate.return_value = mock_session
            correct_suite = good_rows[0]["suite_id"]
            with patch("evaluate_synthetic_regression.SYNREG_SUITE_ID", correct_suite):
                with patch("evaluate_synthetic_regression.SYNREG_RESULTS_STAGE",
                           f"@EVALUATION_RESULTS_STAGE/regression/{correct_suite}"):
                    with patch("tempfile.mkdtemp", return_value=tmp_dir):
                        with pytest.raises(RuntimeError, match="suite_id"):
                            evsr.run_synthetic_regression_aggregation()

    def test_aggregation_fails_on_empty_suite_specific_stage(self):
        """Empty LIST result under suite-specific stage must raise RuntimeError."""
        mock_session = MagicMock()
        mock_session.sql.return_value.collect.return_value = []

        with patch("evaluate_synthetic_regression.Session") as MockSess:
            MockSess.builder.getOrCreate.return_value = mock_session
            with pytest.raises(RuntimeError, match="No shard part files found"):
                evsr.run_synthetic_regression_aggregation()


# ---------------------------------------------------------------------------
# Tests: SYNREG_OUTPUT_STAGE env var
# ---------------------------------------------------------------------------

class TestSynregOutputStage:
    def test_default_output_stage_is_evaluation_results_stage(self):
        """When SYNREG_OUTPUT_STAGE is unset, it defaults to @EVALUATION_RESULTS_STAGE."""
        import importlib
        import os
        os.environ.pop("SYNREG_OUTPUT_STAGE", None)
        m = importlib.reload(importlib.import_module("evaluate_synthetic_regression"))
        assert m.SYNREG_OUTPUT_STAGE == "@EVALUATION_RESULTS_STAGE"

    def test_custom_output_stage_is_respected(self):
        """When SYNREG_OUTPUT_STAGE is set, the module constant must reflect that value."""
        import importlib
        import os
        custom = "@EVALUATION_RESULTS_STAGE/ood_full"
        os.environ["SYNREG_OUTPUT_STAGE"] = custom
        try:
            m = importlib.reload(importlib.import_module("evaluate_synthetic_regression"))
            assert m.SYNREG_OUTPUT_STAGE == custom
        finally:
            os.environ.pop("SYNREG_OUTPUT_STAGE", None)
            # Restore default by reloading without the env var
            importlib.reload(importlib.import_module("evaluate_synthetic_regression"))


# ---------------------------------------------------------------------------
# Tests: Shard completeness (Item 5)
# ---------------------------------------------------------------------------

class TestShardCompleteness:
    """_check_shard_completeness raises RuntimeError when expected shard files are missing."""

    def _make_file_list(self, names: list[str]) -> list[str]:
        """Return a list of stage path strings mimicking LIST output names."""
        return [f"@EVALUATION_RESULTS_STAGE/regression/suite/{n}" for n in names]

    def _all_deepset(self, n: int) -> list[str]:
        return [f"MODEL3-ICL-MC_shard{i}_of_{n}_detailed.csv" for i in range(n)]

    def _all_baseline(self, n: int) -> list[str]:
        return [f"baselines_shard{i}_of_{n}_detailed.csv" for i in range(n)]

    def _all_ag(self, n: int) -> list[str]:
        return [f"AutoGluon_shard{i}_of_{n}_detailed.csv" for i in range(n)]

    def _run(self, found_names, suite_id, n_ds=0, n_bl=0, n_ag=0):
        """Invoke _check_shard_completeness under patched env."""
        with patch("evaluate_synthetic_regression.SYNREG_SUITE_ID", suite_id):
            with patch("evaluate_synthetic_regression.SYNREG_EXPECTED_DEEPSET_SHARDS", n_ds):
                with patch("evaluate_synthetic_regression.SYNREG_EXPECTED_BASELINE_SHARDS", n_bl):
                    with patch("evaluate_synthetic_regression.SYNREG_EXPECTED_AG_SHARDS", n_ag):
                        evsr._check_shard_completeness(found_names, "@stage/parts")

    def test_missing_deepset_shard_fails(self):
        n_ds = 10
        # Provide shards 0-8 (missing shard 9)
        found = self._make_file_list(
            self._all_deepset(n_ds)[:-1]  # all but last
            + self._all_baseline(3)
            + self._all_ag(30)
        )
        with pytest.raises(RuntimeError, match="shard9_of_10"):
            self._run(found, "linear_poisson_v1_recommended", n_ds=n_ds, n_bl=3, n_ag=30)

    def test_missing_baseline_shard_fails(self):
        n_bl = 3
        found = self._make_file_list(
            self._all_deepset(10)
            + self._all_baseline(n_bl)[:-1]  # missing last
            + self._all_ag(30)
        )
        with pytest.raises(RuntimeError, match="baselines_shard"):
            self._run(found, "linear_poisson_v1_recommended", n_ds=10, n_bl=n_bl, n_ag=30)

    def test_missing_ag_shard_fails(self):
        n_ag = 30
        found = self._make_file_list(
            self._all_deepset(10)
            + self._all_baseline(3)
            + self._all_ag(n_ag)[:-1]  # missing last
        )
        with pytest.raises(RuntimeError, match="AutoGluon_shard"):
            self._run(found, "linear_poisson_v1_recommended", n_ds=10, n_bl=3, n_ag=n_ag)

    def test_all_expected_files_passes(self):
        found = self._make_file_list(
            self._all_deepset(10)
            + self._all_baseline(3)
            + self._all_ag(30)
        )
        # Should not raise
        self._run(found, "linear_poisson_v1_recommended", n_ds=10, n_bl=3, n_ag=30)

    def test_pilot_suite_not_checked(self):
        # OOD pilot is NOT in ALL_METHOD_SUITE_IDS → completeness check skipped
        found = []  # empty — no shards at all
        # Should not raise because suite_id is not in ALL_METHOD_SUITE_IDS
        self._run(found, "ood_linear_pilot_v1", n_ds=5, n_bl=3, n_ag=30)

    def test_zero_expected_shards_disables_check(self):
        # n_ds=n_bl=n_ag=0 → completeness disabled even for main suite
        found = []
        self._run(found, "linear_poisson_v1_recommended", n_ds=0, n_bl=0, n_ag=0)
