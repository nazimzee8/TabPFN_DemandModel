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
