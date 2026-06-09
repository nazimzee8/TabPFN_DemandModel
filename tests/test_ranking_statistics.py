"""Tests for F7: build_ranking_summary in evaluation_metrics."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from evaluation_metrics import build_ranking_summary


def _make_df(n_tasks=20, n_models=3, seed=0):
    rng = np.random.default_rng(seed)
    tasks  = [f"t{i}" for i in range(n_tasks)]
    models = [f"m{j}" for j in range(n_models)]
    rows = []
    for t in tasks:
        for m in models:
            rows.append({"task": t, "model": m, "log_loss": float(rng.uniform(0.1, 2.0))})
    return pd.DataFrame(rows)


def test_returns_dataframe_with_required_columns():
    df = _make_df()
    summary = build_ranking_summary(df, metric_col="log_loss", task_col="task", model_col="model")
    required = {
        "model", "mean_rank", "median_rank", "strict_win_rate",
        "tie_aware_win_rate", "top_3_rate", "task_count",
        "valid_metric_task_count", "completion_rate",
    }
    assert required <= set(summary.columns)


def test_one_row_per_model():
    df = _make_df(n_models=4)
    summary = build_ranking_summary(df, metric_col="log_loss", task_col="task", model_col="model")
    assert len(summary) == 4


def test_win_rates_in_unit_interval():
    df = _make_df()
    summary = build_ranking_summary(df, metric_col="log_loss", task_col="task", model_col="model")
    for col in ["strict_win_rate", "tie_aware_win_rate", "top_3_rate"]:
        vals = summary[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"


def test_win_rates_sum_to_one_approx_for_non_tie_ranking():
    """When all metric values are distinct, strict_win_rate should sum to 1/n_tasks."""
    rng = np.random.default_rng(99)
    n_tasks, n_models = 10, 3
    tasks  = [f"t{i}" for i in range(n_tasks)]
    models = [f"m{j}" for j in range(n_models)]
    # Make values distinct within each task to avoid ties.
    rows = []
    for t in tasks:
        vals = rng.choice(np.arange(n_models * 10, dtype=float), n_models, replace=False)
        for m, v in zip(models, vals):
            rows.append({"task": t, "model": m, "acc": v})
    df = pd.DataFrame(rows)
    summary = build_ranking_summary(df, metric_col="acc", task_col="task",
                                    model_col="model", higher_is_better=True)
    # Exactly one model wins each task → sum of strict_win_rate == 1.0 total
    total_wins = summary["strict_win_rate"].sum() * n_tasks
    assert abs(total_wins - n_tasks) < 1e-9


def test_missing_column_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="missing columns"):
        build_ranking_summary(df, metric_col="nonexistent", task_col="task", model_col="model")


def test_completion_rate_with_nan_metrics():
    """Models with NaN for some tasks should have completion_rate < 1."""
    df = _make_df(n_tasks=10, n_models=2)
    # Introduce NaN for model m0 on some tasks
    mask = (df["model"] == "m0") & (df["task"].isin(["t0", "t1", "t2"]))
    df.loc[mask, "log_loss"] = float("nan")
    summary = build_ranking_summary(df, metric_col="log_loss", task_col="task", model_col="model")
    m0_row = summary[summary["model"] == "m0"].iloc[0]
    assert m0_row["completion_rate"] < 1.0
