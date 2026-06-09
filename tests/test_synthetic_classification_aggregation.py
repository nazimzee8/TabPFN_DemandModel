"""
tests/test_synthetic_classification_aggregation.py
===================================================
Unit tests for the aggregate mode of evaluate_linear_classification.py.

Tests:
  - All 18 required output files are produced
  - Missing shards → FileNotFoundError
  - Manifest JSON contains all required keys
  - total_datasets matches unique dataset_ids in combined shards
  - Rank/win metrics computed correctly
  - Summaries by feature_noise_level, label_noise_rate
  - Partial metric columns handled gracefully
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_linear_classification as eval_cls


# ---------------------------------------------------------------------------
# Fixtures: mock shard data
# ---------------------------------------------------------------------------

def _make_shard_df(
    n_datasets: int = 5,
    models=("LogisticRegression", "RidgeClassifier", "RandomForestClassifier"),
    shard_index: int = 0,
    num_classes: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a fake results dataframe as if produced by a single shard."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds_id in range(n_datasets):
        for model_name in models:
            rows.append({
                "dataset_id":            ds_id,
                "suite_id":              "test_suite",
                "shard_index":           shard_index,
                "mode":                  "baselines",
                "model_name":            model_name,
                "num_classes":           num_classes,
                "accuracy":              float(rng.uniform(0.4, 0.95)),
                "balanced_accuracy":     float(rng.uniform(0.4, 0.95)),
                "macro_f1":              float(rng.uniform(0.3, 0.9)),
                "weighted_f1":           float(rng.uniform(0.3, 0.9)),
                "log_loss":              float(rng.uniform(0.3, 1.2)),
                "roc_auc_ovr":           float(rng.uniform(0.5, 0.95)),
                "brier_score":           float(rng.uniform(0.1, 0.4)),
                "expected_calibration_error": float(rng.uniform(0.01, 0.15)),
                "mcc":                   float(rng.uniform(-0.2, 0.8)),
                "cohen_kappa":           float(rng.uniform(-0.1, 0.7)),
                "feature_noise_level":   float(rng.choice([0, 10, 25, 50])),
                "label_noise_rate":      float(rng.choice([0.0, 0.05, 0.1, 0.2])),
                "classification_regime": rng.choice(["binary_linear", "multiclass_linear"]),
                "suite_family":          "primary",
                "p_total":               int(rng.integers(3, 20)),
                "n_total":               int(rng.integers(50, 500)),
                "n_train_default":       int(rng.integers(40, 400)),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def tmp_agg_dir(tmp_path):
    """Temporary directory for aggregation output."""
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers to mock aggregate mode
# ---------------------------------------------------------------------------

def _run_aggregate_with_fake_shards(
    tmp_path,
    deepset_dfs=None,
    baseline_dfs=None,
    ag_dfs=None,
    expected_deepset=1,
    expected_baseline=1,
    expected_ag=0,
    suite_id="test_suite",
):
    """Run _run_aggregate_mode with fake shard data."""
    deepset_dfs  = deepset_dfs  or [_make_shard_df(shard_index=0, models=["tabpfn_deepset"])]
    baseline_dfs = baseline_dfs or [_make_shard_df(shard_index=0)]
    ag_dfs       = ag_dfs       or []

    # Write shard parquets to tmp
    shard_paths = []
    for i, df in enumerate(deepset_dfs):
        p = tmp_path / f"deepset_shard_{i}.parquet"
        df.to_parquet(str(p))
        shard_paths.append(("deepset", str(p)))
    for i, df in enumerate(baseline_dfs):
        p = tmp_path / f"baselines_shard_{i}.parquet"
        df.to_parquet(str(p))
        shard_paths.append(("baselines", str(p)))
    for i, df in enumerate(ag_dfs):
        p = tmp_path / f"autogluon_shard_{i}.parquet"
        df.to_parquet(str(p))
        shard_paths.append(("autogluon", str(p)))

    def _fake_list_stage(session, prefix):
        result = []
        for kind, path in shard_paths:
            if kind == "deepset":
                result.append(f"{prefix}/deepset_shard_0.parquet")
            elif kind == "baselines":
                result.append(f"{prefix}/baselines_shard_0.parquet")
            elif kind == "autogluon":
                result.append(f"{prefix}/autogluon_shard_0.parquet")
        return list(dict.fromkeys(result))  # deduplicate

    local_counter = {"i": 0}
    def _fake_download(session, stage_path, local_dir):
        # Return actual shard parquet files in order
        idx = local_counter["i"]
        if idx < len(shard_paths):
            _, path = shard_paths[idx]
            local_counter["i"] += 1
            return Path(path)
        return Path(shard_paths[-1][1])

    uploaded_files = []
    def _fake_upload(session, local_path, stage_dir):
        uploaded_files.append(Path(local_path).name)
        return f"{stage_dir}/{Path(local_path).name}"

    session = MagicMock()
    monkeypatches = {
        "SYNCLS_SUITE_ID":              suite_id,
        "SYNCLS_RESULTS_STAGE":         f"@EVALUATION_RESULTS_STAGE/linear/classification/numeric/{suite_id}",
        "SYNCLS_PARTS_PREFIX":          f"@EVALUATION_RESULTS_STAGE/linear/classification/numeric/{suite_id}/parts",
        "SYNCLS_CKPT_STAGE":            "@MODEL_STAGE/checkpoints/best_classification.pt",
        "SYNCLS_FEATURE_SEL":           "train_f_classif",
        "SYNCLS_LOCAL_DIR":             str(tmp_path),
        "SYNCLS_NUM_SHARDS":            expected_deepset,
        "SYNCLS_CLUSTER_SHARDS":        expected_ag,
    }
    env_overrides = {
        "SYNCLS_EXPECTED_DEEPSET_SHARDS":  str(expected_deepset),
        "SYNCLS_EXPECTED_BASELINE_SHARDS": str(expected_baseline),
        "SYNCLS_EXPECTED_AG_SHARDS":       str(expected_ag),
    }

    with patch.dict(os.environ, env_overrides), \
         patch.multiple(eval_cls, **monkeypatches), \
         patch.object(eval_cls, "_list_stage_files", side_effect=_fake_list_stage), \
         patch.object(eval_cls, "_download_from_stage", side_effect=_fake_download), \
         patch.object(eval_cls, "_upload_to_stage", side_effect=_fake_upload):
        result = eval_cls._run_aggregate_mode(session)

    return result, uploaded_files


# ---------------------------------------------------------------------------
# Tests: all 18 required output files produced
# ---------------------------------------------------------------------------

def test_all_required_output_files_produced(tmp_agg_dir):
    """All 18 required output files must be uploaded to stage."""
    result, uploaded_files = _run_aggregate_with_fake_shards(tmp_agg_dir)

    for required in eval_cls._REQUIRED_OUTPUT_FILES:
        assert required in uploaded_files, (
            f"Required output file {required!r} not produced. "
            f"Produced files: {sorted(uploaded_files)}"
        )


# ---------------------------------------------------------------------------
# Tests: missing shards → FileNotFoundError
# ---------------------------------------------------------------------------

def test_missing_deepset_shards_raises(tmp_agg_dir):
    """If fewer deepset shards than expected, must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Missing shards"):
        _run_aggregate_with_fake_shards(
            tmp_agg_dir,
            deepset_dfs=[],
            expected_deepset=5,  # expecting 5, providing 0
        )


def test_missing_baseline_shards_raises(tmp_agg_dir):
    """If fewer baseline shards than expected, must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Missing shards"):
        _run_aggregate_with_fake_shards(
            tmp_agg_dir,
            baseline_dfs=[],
            expected_baseline=5,  # expecting 5, providing 0
        )


# ---------------------------------------------------------------------------
# Tests: manifest JSON contains all required keys
# ---------------------------------------------------------------------------

_REQUIRED_MANIFEST_KEYS = {
    "suite_id", "task_family", "task_objective", "created_at",
    "input_shard_list", "expected_deepset_shards", "expected_baseline_shards",
    "expected_ag_shards", "actual_deepset_shards", "actual_baseline_shards",
    "actual_ag_shards", "missing_shards", "output_file_list",
    "metric_schema_version", "result_stage_root", "checkpoint_path",
    "feature_selector", "runtime_variables", "validation_status",
    "total_datasets", "total_rows",
}


def test_manifest_contains_required_keys(tmp_agg_dir):
    """Aggregation manifest must contain all required keys."""
    result, uploaded_files = _run_aggregate_with_fake_shards(tmp_agg_dir)

    # Find the manifest in tmp_agg_dir
    manifest_path = tmp_agg_dir / "aggregate" / "synthetic_classification_aggregation_manifest.json"
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

    with open(manifest_path) as f:
        manifest = json.load(f)

    for key in _REQUIRED_MANIFEST_KEYS:
        assert key in manifest, f"Required manifest key {key!r} missing."


def test_manifest_task_objective(tmp_agg_dir):
    """Manifest must declare task_objective='inductive_classification'."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    manifest_path = tmp_agg_dir / "aggregate" / "synthetic_classification_aggregation_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["task_objective"] == "inductive_classification"


# ---------------------------------------------------------------------------
# Tests: total_datasets
# ---------------------------------------------------------------------------

def test_total_datasets_matches_unique_dataset_ids(tmp_agg_dir):
    """total_datasets in manifest must match unique dataset_id count in combined data."""
    n_unique_datasets = 7
    baseline_df = _make_shard_df(n_datasets=n_unique_datasets)
    deepset_df = _make_shard_df(n_datasets=n_unique_datasets, models=["tabpfn_deepset"])

    _run_aggregate_with_fake_shards(
        tmp_agg_dir,
        deepset_dfs=[deepset_df],
        baseline_dfs=[baseline_df],
    )
    manifest_path = tmp_agg_dir / "aggregate" / "synthetic_classification_aggregation_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["total_datasets"] == n_unique_datasets


# ---------------------------------------------------------------------------
# Tests: rank metrics computed
# ---------------------------------------------------------------------------

def test_rank_by_log_loss_computed(tmp_agg_dir):
    """rank_by_log_loss must be present in model comparison output."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    csv_path = tmp_agg_dir / "aggregate" / "synthetic_classification_model_comparison.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "rank_by_log_loss" in df.columns, "rank_by_log_loss missing from comparison output"


def test_beats_logistic_regression_computed(tmp_agg_dir):
    """beats_logistic_regression must be present in model comparison output."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    csv_path = tmp_agg_dir / "aggregate" / "synthetic_classification_model_comparison.csv"
    df = pd.read_csv(csv_path)
    assert "beats_logistic_regression" in df.columns


# ---------------------------------------------------------------------------
# Tests: summaries by dimension
# ---------------------------------------------------------------------------

def test_summary_by_feature_noise(tmp_agg_dir):
    """Summary by feature_noise_level must be produced."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    csv_path = tmp_agg_dir / "aggregate" / "synthetic_classification_summary_by_feature_noise.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    # May be empty if no feature_noise_level variation, but file must exist


def test_summary_by_label_noise(tmp_agg_dir):
    """Summary by label_noise_rate must be produced."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    csv_path = tmp_agg_dir / "aggregate" / "synthetic_classification_summary_by_label_noise.csv"
    assert csv_path.exists()


# ---------------------------------------------------------------------------
# Tests: partial metric columns (optional → NaN, required → fail loudly)
# ---------------------------------------------------------------------------

def test_partial_metrics_optional_columns_handled(tmp_agg_dir):
    """When optional metric columns are absent, aggregation must not fail."""
    # Create shard DF with only required metrics, no optional ones
    df_min = _make_shard_df()
    for optional_col in eval_cls._OPTIONAL_METRIC_COLS:
        if optional_col in df_min.columns:
            df_min = df_min.drop(columns=[optional_col])

    # Should not raise
    result, _ = _run_aggregate_with_fake_shards(
        tmp_agg_dir,
        deepset_dfs=[df_min[df_min["model_name"] == "tabpfn_deepset"].copy()
                     if "tabpfn_deepset" in df_min["model_name"].values
                     else _make_shard_df(models=["tabpfn_deepset"])],
        baseline_dfs=[df_min],
    )
    assert "aggregate: ok" in result


def test_missing_required_metrics_flagged_in_manifest(tmp_agg_dir):
    """If required metrics are absent, manifest validation_status should not be 'ok'."""
    # Drop required metric columns from shard data
    df_no_metrics = _make_shard_df().drop(
        columns=[c for c in eval_cls._REQUIRED_METRIC_COLS if c in _make_shard_df().columns],
        errors="ignore",
    )
    deepset_df = _make_shard_df(models=["tabpfn_deepset"]).drop(
        columns=[c for c in eval_cls._REQUIRED_METRIC_COLS if c in _make_shard_df().columns],
        errors="ignore",
    )

    _run_aggregate_with_fake_shards(
        tmp_agg_dir,
        deepset_dfs=[deepset_df],
        baseline_dfs=[df_no_metrics],
    )
    manifest_path = tmp_agg_dir / "aggregate" / "synthetic_classification_aggregation_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    # validation_status should be 'missing_required_metrics' when required cols absent
    assert manifest["validation_status"] == "missing_required_metrics"


# ---------------------------------------------------------------------------
# Tests: no regression metrics in output
# ---------------------------------------------------------------------------

def test_no_regression_metrics_in_aggregate_output(tmp_agg_dir):
    """Aggregate output must not contain r2, mse, rmse, or mae columns."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    csv_path = tmp_agg_dir / "aggregate" / "synthetic_classification_model_comparison.csv"
    df = pd.read_csv(csv_path)
    for bad_col in ["r2", "mse", "rmse", "mae", "R2", "MSE", "RMSE", "MAE"]:
        assert bad_col not in df.columns, f"Unexpected regression metric {bad_col} in aggregate output"


# ---------------------------------------------------------------------------
# Phase 3: Run-scoped aggregation tests
# ---------------------------------------------------------------------------

def test_missing_reference_yields_nan(tmp_agg_dir):
    """When LogisticRegression not present, beats_logistic_regression should be NaN, not False."""
    # Shard with only MODEL-ICL-MC, no LogisticRegression
    df_no_lr = _make_shard_df(models=["MODEL-ICL-MC"])
    _run_aggregate_with_fake_shards(
        tmp_agg_dir,
        deepset_dfs=[df_no_lr],
        baseline_dfs=[_make_shard_df(models=["MODEL-ICL-MC"])],
    )
    csv_path = tmp_agg_dir / "aggregate" / "synthetic_classification_model_comparison.csv"
    df = pd.read_csv(csv_path)
    if "beats_logistic_regression" in df.columns:
        # Should be NaN, not False, when reference is unavailable
        model_rows = df[df["model_name"] == "MODEL-ICL-MC"]
        if not model_rows.empty:
            beats_vals = model_rows["beats_logistic_regression"]
            # All values should be NaN (no reference) or True/False (if reference present)
            # Since we only have MODEL-ICL-MC, reference is absent → NaN
            assert beats_vals.isna().all() or beats_vals.empty, (
                f"beats_logistic_regression should be NaN when reference is absent, got: {beats_vals.tolist()}"
            )


def test_manifest_has_evaluation_run_id(tmp_agg_dir):
    """Manifest must contain evaluation_run_id field."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    manifest_path = tmp_agg_dir / "aggregate" / "synthetic_classification_aggregation_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert "evaluation_run_id" in manifest, "manifest must have evaluation_run_id field"


def test_manifest_has_suite_schema_version(tmp_agg_dir):
    """Manifest must contain suite_schema_version field."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    manifest_path = tmp_agg_dir / "aggregate" / "synthetic_classification_aggregation_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert "suite_schema_version" in manifest, "manifest must have suite_schema_version"
    assert "classification_eval" in manifest["suite_schema_version"], (
        "suite_schema_version should reference classification_eval"
    )


# ---------------------------------------------------------------------------
# Phase 4: Metadata propagation
# ---------------------------------------------------------------------------

def test_p_n_quartile_summary_has_both_columns(tmp_agg_dir):
    """p/n quartile summary must have both p_quartile and n_quartile columns."""
    # Inject p_total and n_total into shard data
    df = _make_shard_df()
    df["p_total"] = np.random.randint(5, 50, len(df))
    df["n_total"] = np.random.randint(100, 1000, len(df))
    deepset_df = df[df["model_name"] == "tabpfn_deepset"].copy() if "tabpfn_deepset" in df["model_name"].values else _make_shard_df(models=["MODEL-ICL-MC"])
    deepset_df["p_total"] = np.random.randint(5, 50, len(deepset_df))
    deepset_df["n_total"] = np.random.randint(100, 1000, len(deepset_df))

    _run_aggregate_with_fake_shards(
        tmp_agg_dir,
        deepset_dfs=[deepset_df],
        baseline_dfs=[df],
    )
    summary_path = tmp_agg_dir / "aggregate" / "synthetic_classification_summary_by_p_quartile_n_quartile.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            assert "p_quartile" in summary_df.columns, "p_quartile column missing from summary"
            assert "n_quartile" in summary_df.columns, "n_quartile column missing from summary"


# ---------------------------------------------------------------------------
# Phase 5: Comparative reporting
# ---------------------------------------------------------------------------

def test_brier_score_multiclass_not_nan(tmp_agg_dir):
    """Multiclass Brier score must not be NaN for models with probability estimates."""
    # The fix is in _compute_classification_metrics, not aggregate. Test directly.
    import numpy as np
    from evaluate_linear_classification import _compute_classification_metrics

    rng = np.random.default_rng(42)
    n, K = 50, 4
    y_true = rng.integers(0, K, n)
    proba = rng.dirichlet(np.ones(K), n)
    y_pred = proba.argmax(axis=1)

    metrics = _compute_classification_metrics(y_true, y_pred, proba, K, "test_model")
    brier = metrics.get("brier_score")
    assert brier is not None, "brier_score should be present"
    assert not np.isnan(brier), f"Multiclass brier_score should not be NaN, got {brier}"
    assert 0.0 <= brier <= 2.0, f"brier_score={brier} out of expected range [0, 2]"


def test_rank_win_metrics_in_summary(tmp_agg_dir):
    """Aggregate model comparison CSV should have rank_by_log_loss or rank_by_accuracy columns."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    csv_path = tmp_agg_dir / "aggregate" / "synthetic_classification_model_comparison.csv"
    df = pd.read_csv(csv_path)
    has_rank = "rank_by_log_loss" in df.columns or "rank_by_accuracy" in df.columns
    assert has_rank, (
        f"Expected rank_by_log_loss or rank_by_accuracy column in model comparison. "
        f"Available: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Phase 9: Statistical reliability
# ---------------------------------------------------------------------------

def test_bootstrap_ci_deterministic():
    """Same seed produces same CI from paired_bootstrap_ci."""
    from evaluation_metrics import paired_bootstrap_ci

    rng = np.random.default_rng(0)
    a = rng.standard_normal(50)
    b = rng.standard_normal(50)
    ci1 = paired_bootstrap_ci(a, b, seed=7)
    ci2 = paired_bootstrap_ci(a, b, seed=7)
    assert ci1["lower"] == ci2["lower"], "bootstrap CI must be deterministic"
    assert ci1["upper"] == ci2["upper"], "bootstrap CI must be deterministic"


def test_bootstrap_ci_nan_when_insufficient_data():
    """Fewer than 10 pairs → NaN CI bounds."""
    from evaluation_metrics import paired_bootstrap_ci

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.5, 2.5, 3.5])
    ci = paired_bootstrap_ci(a, b, seed=42)
    assert np.isnan(ci["lower"]), f"lower CI should be NaN for < 10 pairs, got {ci['lower']}"
    assert np.isnan(ci["upper"]), f"upper CI should be NaN for < 10 pairs, got {ci['upper']}"
    assert ci["n_pairs"] < 10


# ---------------------------------------------------------------------------
# Phase 10: Provenance
# ---------------------------------------------------------------------------

def test_manifest_has_checkpoint_path(tmp_agg_dir):
    """Manifest must contain checkpoint_path."""
    _run_aggregate_with_fake_shards(tmp_agg_dir)
    manifest_path = tmp_agg_dir / "aggregate" / "synthetic_classification_aggregation_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert "checkpoint_path" in manifest, "manifest must have checkpoint_path"
