"""Shared evaluation metric and ranking helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats


def ci95(vals):
    """Return (mean, lo, hi) for a 1-D array; NaN CI if fewer than two values."""
    vals = np.asarray(vals, dtype=float)
    valid = vals[~np.isnan(vals)]
    if len(valid) < 2:
        mu = float(np.nanmean(vals)) if len(valid) == 1 else float("nan")
        return mu, float("nan"), float("nan")
    mu = float(np.mean(valid))
    sem = float(stats.sem(valid))
    if sem == 0.0:
        return mu, mu, mu
    lo, hi = stats.t.interval(0.95, df=len(valid) - 1, loc=mu, scale=sem)
    return mu, float(lo), float(hi)


def rank_methods(metrics_matrix, higher_is_better=False):
    """
    NaN-aware per-dataset ranking.

    Args:
        metrics_matrix: np.ndarray shape (n_methods, n_datasets), may contain NaN.
        higher_is_better: if True, rank 1 = highest value.

    Returns:
        rank_matrix: same shape; NaN where method had no result for that dataset.
    """
    metrics_matrix = np.asarray(metrics_matrix, dtype=float)
    n_methods, n_datasets = metrics_matrix.shape
    rank_matrix = np.full_like(metrics_matrix, np.nan)

    for j in range(n_datasets):
        col = metrics_matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        vals = col[valid]
        if higher_is_better:
            vals = -vals
        ranks = stats.rankdata(vals, method="average")
        rank_matrix[valid, j] = ranks

    return rank_matrix


def paired_bootstrap_ci(a, b, *, n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Compute paired bootstrap confidence interval for the difference (a - b).

    Parameters
    ----------
    a, b : array-like of paired scores (same dataset matched by key)
    n_bootstrap : number of bootstrap replicates
    alpha : significance level for CI (default 0.05 → 95% CI)
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys: estimate, lower, upper, n_pairs, seed, reason (if CI unavailable)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    valid = np.isfinite(diffs)
    n_pairs = int(valid.sum())

    if n_pairs < 10:
        return {
            "estimate": float(np.nanmean(diffs)),
            "lower": float("nan"),
            "upper": float("nan"),
            "n_pairs": n_pairs,
            "seed": seed,
            "reason": f"insufficient paired samples: {n_pairs} < 10",
        }

    rng = np.random.default_rng(seed)
    valid_diffs = diffs[valid]
    estimate = float(np.mean(valid_diffs))
    boot_means = np.array([
        rng.choice(valid_diffs, size=n_pairs, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "n_pairs": n_pairs,
        "seed": seed,
    }


def build_ranking_summary(df, metric_col: str, task_col: str, model_col: str,
                          higher_is_better: bool = False) -> "pd.DataFrame":
    """Build per-model ranking statistics across tasks.

    Parameters
    ----------
    df : pandas.DataFrame with columns [model_col, task_col, metric_col]
    metric_col : name of the metric column to rank on
    task_col : name of the column identifying individual tasks/datasets
    model_col : name of the column identifying models/methods
    higher_is_better : if True, rank 1 = highest metric value

    Returns
    -------
    pandas.DataFrame with one row per model and columns:
        mean_rank, median_rank, strict_win_rate, tie_aware_win_rate,
        top_3_rate, task_count, valid_metric_task_count, completion_rate
    """
    import pandas as pd

    required = {model_col, task_col, metric_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_ranking_summary: missing columns {missing}")

    models = sorted(df[model_col].dropna().unique())
    tasks  = sorted(df[task_col].dropna().unique())

    # Build matrix: rows = models, cols = tasks
    pivot = df.pivot_table(
        index=model_col, columns=task_col, values=metric_col, aggfunc="first"
    ).reindex(index=models, columns=tasks)

    metrics_matrix = pivot.values  # shape (n_models, n_tasks)
    rank_matrix = rank_methods(metrics_matrix, higher_is_better=higher_is_better)

    records = []
    for i, model in enumerate(models):
        ranks = rank_matrix[i]
        valid_ranks = ranks[~np.isnan(ranks)]
        n_total = len(tasks)
        n_valid = int(len(valid_ranks))
        mean_rank   = float(np.mean(valid_ranks)) if n_valid > 0 else float("nan")
        median_rank = float(np.median(valid_ranks)) if n_valid > 0 else float("nan")
        # strict win: ranked exactly 1 (no tie handling)
        strict_win  = float(np.sum(valid_ranks == 1.0)) / n_valid if n_valid > 0 else float("nan")
        # tie-aware win: rank < 1.5 (includes shared rank-1 ties with average rank 1.0)
        tie_win     = float(np.sum(valid_ranks <= 1.5)) / n_valid if n_valid > 0 else float("nan")
        top3        = float(np.sum(valid_ranks <= 3.0)) / n_valid if n_valid > 0 else float("nan")
        records.append({
            model_col:                model,
            "mean_rank":              mean_rank,
            "median_rank":            median_rank,
            "strict_win_rate":        strict_win,
            "tie_aware_win_rate":     tie_win,
            "top_3_rate":             top3,
            "task_count":             n_total,
            "valid_metric_task_count": n_valid,
            "completion_rate":        n_valid / n_total if n_total > 0 else float("nan"),
        })
    return pd.DataFrame(records)


def robust_summary(values) -> dict:
    """Compute mean, median, IQR, p90, p95, and n_valid for an array.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    dict with keys: mean, median, iqr, p90, p95, n_valid
    """
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    n_valid = int(len(valid))
    if n_valid == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "n_valid": 0,
        }
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "n_valid": n_valid,
    }
