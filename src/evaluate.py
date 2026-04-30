"""
evaluate.py

Evaluate a saved DeepSetModel checkpoint:
  1. Permutation-invariance tests (7 synthetic)
  2. Synthetic DGP evaluation: DeepSetModel vs OLS, stratified by regime (A/B/C/D)
  3. MC dropout noise assessment on synthetic DGP
  4. OpenML + Kaggle regression benchmark: DeepSetModel-MC vs 9 baselines
     (28 OpenML + 2 Kaggle regression datasets, 10 reps, 90/10 split, ≤10k samples, ≤500 features, 95% CI)
     Datasets: AutoML Benchmark (study ~271) + OpenML-CTR23 (study 353) + Kaggle TPS-S3 (s3e5, s3e9)
     Methods: DeepSetModel-MC, XGBoost, LightGBM, CatBoost, RandomForest,
              KNN, LinearRegression, Ridge, SVR, MLP, AutoGluon

Usage (inside container or locally):
    python evaluate.py --model_path best.pt --data_dir /tmp/data --results_dir results/
    python evaluate.py --model_path best.pt --no_openml   # Kaggle only (fast smoke test)
    python evaluate.py --model_path best.pt --no_kaggle   # OpenML only (28 datasets)

Outputs (results_dir/):
    synthetic/test_report.csv              — synthetic DGP per-regime MSE
    synthetic/mc_report.csv                — MC dropout vs point prediction per regime
    benchmark_parts/<method>_detailed.csv  — per-method benchmark metrics
    model_comparison.csv                   — canonical benchmark comparison
    model_comparison_summary.csv           — mean ranks and metrics per method

Note: Metrics are NOT directly comparable to TabPFN v2 Nature 2025 paper — the paper
uses a 64/16/20 split with 15 seeds; this benchmark uses 90/10 with 10 seeds.
"""

import argparse
import csv
import os
import glob
import shutil
import tempfile
import warnings
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch
import pandas as pd
from scipy import stats

from model import DeepSetModel, ModelConfig, POOL_SCALE
from snowflake_io import materialize_meta_dataset_stage

# SPCS home guard — redirect ~ to writable path before any Snowflake imports
os.environ.setdefault("HOME", "/tmp")

# ---------------------------------------------------------------------------
# Optional benchmark dependencies
# ---------------------------------------------------------------------------

try:
    import openml
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    BENCHMARK_DEPS_AVAILABLE = True
except ImportError:
    BENCHMARK_DEPS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_REPS            = 10
SEEDS             = list(range(10))          # [0, 1, ..., 9]
TRAIN_FRAC        = 0.9                      # 90/10 train/test split
MAX_SAMPLES       = 10_000
MAX_FEATURES      = 500
N_MC_DROPOUT      = 32
OPENML_N_DATASETS = 28                       # AutoML Benchmark + OpenML-CTR23
EVAL_RESULTS_STAGE = os.environ.get("EVAL_RESULTS_STAGE", "@EVALUATION_RESULTS_STAGE")
CHECKPOINT_STAGE = os.environ.get("CHECKPOINT_STAGE", "@MODEL_STAGE/checkpoints/")
ALL_BENCHMARK_METHODS = [
    "DeepSetModel-MC",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "KNN",
    "LinearRegression",
    "Ridge",
    "SVR",
    "MLP",
    "AutoGluon",
]
AUTOGLUON_TIME_LIMIT = int(os.environ.get("AUTOGLUON_TIME_LIMIT", "300"))


# ---------------------------------------------------------------------------
# Parquet loader (mirrors train.py)
# ---------------------------------------------------------------------------

def load_parquet(path):
    table = pq.read_table(path)
    d = table.to_pydict()
    X_train      = torch.tensor(np.array(d["X_train"][0]),    dtype=torch.float32)
    y_train      = torch.tensor(np.array(d["y_train"][0]),    dtype=torch.float32)
    X_test       = torch.tensor(np.array(d["X_test"][0]),     dtype=torch.float32)
    betaX_test   = torch.tensor(np.array(d["betaX_test"][0]), dtype=torch.float32)
    prior_regime = d["prior_regime"][0]
    n            = int(d["n"][0])
    p            = int(d["p"][0])
    return X_train, y_train, X_test, betaX_test, prior_regime, n, p


# ---------------------------------------------------------------------------
# Dispatch helpers for equivariant layers
# ---------------------------------------------------------------------------

def apply_feat_equiv(model, h):
    """h: (n, p, d) → (n, p, d). Dispatches to SAB or linear equivariance."""
    if model.cfg.n_sab_feat > 0:
        return model.sab_feat(h)
    mean_i = h.mean(dim=1, keepdim=True)
    return model.lambda_feat * h + model.gamma_feat * mean_i


def apply_samp_equiv(model, r):
    """r: (n, d) → (n, d). Dispatches to SAB or linear equivariance."""
    if model.cfg.n_sab_samp > 0:
        return model.sab_samp(r.unsqueeze(0)).squeeze(0)
    mean_j = r.mean(dim=0, keepdim=True)
    return model.lambda_samp * r + model.gamma_samp * mean_j


# ---------------------------------------------------------------------------
# Section A — Load model
# ---------------------------------------------------------------------------

def load_model(model_path):
    if not os.path.exists(model_path):
        download_stage_prefix(CHECKPOINT_STAGE, os.path.dirname(model_path) or ".")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint {model_path!r} does not exist after attempting Snowflake stage fetch "
            f"from {CHECKPOINT_STAGE}."
        )
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg, state_dict = ckpt["cfg"], ckpt["state_dict"]
    else:
        # Legacy bare state_dict
        cfg = ModelConfig(d_phi=128, d_rho=256, pool="pna", dropout=0.1,
                          n_sab_feat=0, n_sab_samp=0,
                          norm_feat=False, norm_target=False)
        state_dict = ckpt
    model = DeepSetModel(cfg=cfg)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Section B — Evaluate on held-out synthetic DGP test split
# ---------------------------------------------------------------------------

def evaluate_synthetic_dgp(model, test_dir):
    """
    Predict and compute OLS baseline for every test Parquet file.
    Returns list of per-dataset dicts: model_mse, ols_mse, prior_regime, n, p.
    """
    files = sorted(
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {test_dir}")

    records = []
    model.eval()
    with torch.no_grad():
        for path in files:
            X_train, y_train, X_test, betaX_test, prior_regime, n, p = load_parquet(path)

            y_hat      = model(X_train, y_train, X_test)
            preds_np   = y_hat.cpu().numpy()
            betaX_np   = betaX_test.numpy()
            model_mse  = float(np.mean((preds_np - betaX_np) ** 2))

            # OLS baseline: β̂ = (X'X + I)^{-1} X'y
            X_train_np = X_train.numpy()
            y_train_np = y_train.numpy()
            X_test_np  = X_test.numpy()
            A          = X_train_np.T @ X_train_np + np.eye(p)
            beta_hat   = np.linalg.solve(A, X_train_np.T @ y_train_np)
            ols_preds  = X_test_np @ beta_hat
            ols_mse    = float(np.mean((ols_preds - betaX_np) ** 2))

            records.append({
                "prior_regime": prior_regime,
                "n":            n,
                "p":            p,
                "model_mse":    model_mse,
                "ols_mse":      ols_mse,
            })

    return records


# ---------------------------------------------------------------------------
# Section C — Stratified report
# ---------------------------------------------------------------------------

def build_report(records):
    """Compute per-regime and overall summary rows."""
    groups = defaultdict(list)
    for r in records:
        groups[r["prior_regime"]].append(r)

    rows = []
    for regime in sorted(groups.keys()):
        grp   = groups[regime]
        m_mse = float(np.mean([r["model_mse"] for r in grp]))
        o_mse = float(np.mean([r["ols_mse"]   for r in grp]))
        rows.append({
            "prior_regime":    regime,
            "mean_model_mse":  m_mse,
            "mean_ols_mse":    o_mse,
            "ratio_model_ols": m_mse / o_mse if o_mse > 0 else float("nan"),
            "count":           len(grp),
        })

    # ALL row
    m_mse = float(np.mean([r["model_mse"] for r in records]))
    o_mse = float(np.mean([r["ols_mse"]   for r in records]))
    rows.append({
        "prior_regime":    "ALL",
        "mean_model_mse":  m_mse,
        "mean_ols_mse":    o_mse,
        "ratio_model_ols": m_mse / o_mse if o_mse > 0 else float("nan"),
        "count":           len(records),
    })
    return rows


def print_report(rows):
    col_w = [14, 16, 14, 12, 7]
    header = (
        f"{'prior_regime':>{col_w[0]}}"
        f"  {'mean_model_mse':>{col_w[1]}}"
        f"  {'mean_ols_mse':>{col_w[2]}}"
        f"  {'ratio(m/o)':>{col_w[3]}}"
        f"  {'count':>{col_w[4]}}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['prior_regime']:>{col_w[0]}}"
            f"  {row['mean_model_mse']:>{col_w[1]}.6f}"
            f"  {row['mean_ols_mse']:>{col_w[2]}.6f}"
            f"  {row['ratio_model_ols']:>{col_w[3]}.4f}"
            f"  {row['count']:>{col_w[4]}}"
        )


def save_report_csv(rows, path):
    fieldnames = ["prior_regime", "mean_model_mse", "mean_ols_mse", "ratio_model_ols", "count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Section C2 — MC Dropout Noise Assessment
# ---------------------------------------------------------------------------

def mc_dropout_inference(model, X_train, y_train, X_test, K=32):
    """
    K stochastic forward passes with dropout active.
    Returns (mean_preds, std_preds) as numpy arrays shape (m,).

    Note: model.train() activates nn.Dropout; torch.no_grad() is orthogonal
    (prevents gradient tracking). try/finally always restores model.eval().
    """
    model.train()   # activates dropout
    preds = []
    try:
        with torch.no_grad():
            for _ in range(K):
                p = model(X_train, y_train, X_test)
                preds.append(p.detach().cpu().numpy())
    finally:
        model.eval()   # always restore, even on exception
    stack = np.stack(preds)   # (K, m)
    return stack.mean(0), stack.std(0)


def evaluate_synthetic_dgp_mc(model, test_dir, K=32):
    """Like evaluate_synthetic_dgp but adds mc_mse and mc_std_mean columns."""
    files = sorted(
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {test_dir}")

    records = []
    for path in files:
        X_train, y_train, X_test, betaX_test, prior_regime, n, p = load_parquet(path)
        betaX_np = betaX_test.numpy()

        # Point prediction
        model.eval()
        with torch.no_grad():
            preds_np = model(X_train, y_train, X_test).detach().cpu().numpy()
        model_mse = float(np.mean((preds_np - betaX_np) ** 2))

        # MC dropout
        mc_mean, mc_std = mc_dropout_inference(model, X_train, y_train, X_test, K=K)
        mc_mse      = float(np.mean((mc_mean - betaX_np) ** 2))
        mc_std_mean = float(mc_std.mean())

        # OLS
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()
        X_test_np  = X_test.numpy()
        A          = X_train_np.T @ X_train_np + np.eye(p)
        beta_hat   = np.linalg.solve(A, X_train_np.T @ y_train_np)
        ols_preds  = X_test_np @ beta_hat
        ols_mse    = float(np.mean((ols_preds - betaX_np) ** 2))

        records.append({
            "prior_regime": prior_regime,
            "n":            n,
            "p":            p,
            "model_mse":    model_mse,
            "mc_mse":       mc_mse,
            "mc_std_mean":  mc_std_mean,
            "ols_mse":      ols_mse,
        })

    return records


def build_mc_report(records):
    """
    Stratify by regime. ratio_mc_ols < ratio_model_ols quantifies noise reduction
    from MC averaging vs point prediction.
    """
    groups = defaultdict(list)
    for r in records:
        groups[r["prior_regime"]].append(r)

    rows = []
    for regime in sorted(groups.keys()):
        grp    = groups[regime]
        m_mse  = float(np.mean([r["model_mse"]   for r in grp]))
        mc_mse = float(np.mean([r["mc_mse"]      for r in grp]))
        mc_std = float(np.mean([r["mc_std_mean"] for r in grp]))
        o_mse  = float(np.mean([r["ols_mse"]     for r in grp]))
        rows.append({
            "prior_regime":    regime,
            "mean_model_mse":  m_mse,
            "mean_mc_mse":     mc_mse,
            "mean_mc_std":     mc_std,
            "mean_ols_mse":    o_mse,
            "ratio_model_ols": m_mse  / o_mse if o_mse > 0 else float("nan"),
            "ratio_mc_ols":    mc_mse / o_mse if o_mse > 0 else float("nan"),
            "count":           len(grp),
        })

    # ALL row
    m_mse  = float(np.mean([r["model_mse"]   for r in records]))
    mc_mse = float(np.mean([r["mc_mse"]      for r in records]))
    mc_std = float(np.mean([r["mc_std_mean"] for r in records]))
    o_mse  = float(np.mean([r["ols_mse"]     for r in records]))
    rows.append({
        "prior_regime":    "ALL",
        "mean_model_mse":  m_mse,
        "mean_mc_mse":     mc_mse,
        "mean_mc_std":     mc_std,
        "mean_ols_mse":    o_mse,
        "ratio_model_ols": m_mse  / o_mse if o_mse > 0 else float("nan"),
        "ratio_mc_ols":    mc_mse / o_mse if o_mse > 0 else float("nan"),
        "count":           len(records),
    })
    return rows


def print_mc_report(rows):
    print(
        f"\n{'regime':>12}  {'model_mse':>10}  {'mc_mse':>10}  {'mc_std':>8}  "
        f"{'ols_mse':>10}  {'m/o':>6}  {'mc/o':>6}  {'n':>5}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{row['prior_regime']:>12}"
            f"  {row['mean_model_mse']:>10.6f}"
            f"  {row['mean_mc_mse']:>10.6f}"
            f"  {row['mean_mc_std']:>8.6f}"
            f"  {row['mean_ols_mse']:>10.6f}"
            f"  {row['ratio_model_ols']:>6.4f}"
            f"  {row['ratio_mc_ols']:>6.4f}"
            f"  {row['count']:>5}"
        )


def save_mc_report_csv(rows, path):
    fieldnames = [
        "prior_regime", "mean_model_mse", "mean_mc_mse", "mean_mc_std",
        "mean_ols_mse", "ratio_model_ols", "ratio_mc_ols", "count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Section D — Permutation invariance tests (Tests 1–7)
# ---------------------------------------------------------------------------

def run_permutation_tests(model):
    torch.manual_seed(0)
    n, p   = 20, 5
    cfg    = model.cfg
    D_PHI  = cfg.d_phi
    D_RHO  = cfg.d_rho

    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(p)
    model.eval()
    results = {}

    with torch.no_grad():
        # Test 1 — row permutation invariance (end-to-end)
        pi = torch.randperm(n)
        results["Test 1 (row permutation invariance)"] = torch.allclose(
            model(X_train, y_train, x_test),
            model(X_train[pi], y_train[pi], x_test), atol=1e-5)

        # Test 2 — column permutation invariance (end-to-end)
        pi_col = torch.randperm(p)
        results["Test 2 (column permutation invariance)"] = torch.allclose(
            model(X_train, y_train, x_test),
            model(X_train[:, pi_col], y_train, x_test[pi_col]), atol=1e-5)

        # Test 3 — sample equivariance: apply_samp_equiv(r[π]) == apply_samp_equiv(r)[π]
        r, pi = torch.randn(n, D_RHO), torch.randperm(n)
        results["Test 3 (sample equiv equivariance)"] = torch.allclose(
            apply_samp_equiv(model, r[pi]),
            apply_samp_equiv(model, r)[pi], atol=1e-5)

        # Test 4 — feature equivariance: apply_feat_equiv(h[:,π,:]) == apply_feat_equiv(h)[:,π,:]
        h, pi_feat = torch.randn(n, p, D_PHI), torch.randperm(p)
        results["Test 4 (feature equiv equivariance)"] = torch.allclose(
            apply_feat_equiv(model, h[:, pi_feat, :]),
            apply_feat_equiv(model, h)[:, pi_feat, :], atol=1e-5)

        # Test 5 — mean-pool after sample equiv is permutation invariant
        r, pi = torch.randn(n, D_RHO), torch.randperm(n)
        results["Test 5 (sample invariance after pool)"] = torch.allclose(
            apply_samp_equiv(model, r).mean(dim=0),
            apply_samp_equiv(model, r[pi]).mean(dim=0), atol=1e-5)

        if cfg.n_sab_samp == 0:
            # Test 6 (linear mode) — Θ = λI + γ/n·11ᵀ matrix form
            r   = torch.randn(n, D_RHO)
            lam = model.lambda_samp.item()
            gam = model.gamma_samp.item()
            theta = lam * torch.eye(n) + (gam / n) * torch.ones(n, n)
            results["Test 6 (Theta matrix form)"] = torch.allclose(
                apply_samp_equiv(model, r), theta @ r, atol=1e-5)

            # Test 7 (linear mode) — mean after permuted equiv == mean after equiv
            r, pi = torch.randn(n, D_RHO), torch.randperm(n)
            results["Test 7 (mean after permuted equiv)"] = torch.allclose(
                apply_samp_equiv(model, r[pi]).mean(dim=0),
                apply_samp_equiv(model, r).mean(dim=0), atol=1e-5)
        else:
            # Test 6 (SAB mode) — SAB_samp equivariance via raw module
            r, pi = torch.randn(n, D_RHO), torch.randperm(n)
            rb    = r.unsqueeze(0)                  # (1, n, d_rho)
            results["Test 6 (SAB sample equivariance)"] = torch.allclose(
                model.sab_samp(rb[:, pi, :]).squeeze(0),
                model.sab_samp(rb).squeeze(0)[pi], atol=1e-4)

            # Test 7 (SAB mode) — SAB_feat equivariance via raw module
            h, pi_feat = torch.randn(n, p, D_PHI), torch.randperm(p)
            results["Test 7 (SAB feature equivariance)"] = torch.allclose(
                model.sab_feat(h[:, pi_feat, :]),
                model.sab_feat(h)[:, pi_feat, :], atol=1e-4)

    print("\nPermutation Invariance Tests:")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Section D2 — Baselines and shared metric helpers
# ---------------------------------------------------------------------------

def get_baselines(seed):
    """Return 9 sklearn-compatible regressors (tree, linear, SVM, neural)."""
    if not BENCHMARK_DEPS_AVAILABLE:
        return {}
    return {
        "XGBoost":          xgb.XGBRegressor(n_estimators=200, random_state=seed,
                                verbosity=0, n_jobs=1),
        "LightGBM":         lgb.LGBMRegressor(n_estimators=200, random_state=seed,
                                verbose=-1, n_jobs=1),
        "CatBoost":         CatBoostRegressor(iterations=200, random_state=seed,
                                verbose=0, allow_writing_files=False),
        "RandomForest":     RandomForestRegressor(n_estimators=200, random_state=seed,
                                n_jobs=1),
        "KNN":              KNeighborsRegressor(n_neighbors=5, n_jobs=1),
        "LinearRegression": LinearRegression(n_jobs=1),
        "Ridge":            Ridge(alpha=1.0),
        # SVR and MLP need feature scaling — wrap in Pipeline with StandardScaler
        "SVR":              Pipeline([("scaler", StandardScaler()), ("svr", SVR())]),
        "MLP":              Pipeline([("scaler", StandardScaler()),
                                      ("mlp",   MLPRegressor(hidden_layer_sizes=(256, 128),
                                                             max_iter=300,
                                                             random_state=seed))]),
    }


def predict_deepset_mc(model, X_train_np, y_train_np, X_test_np, K=32):
    """Convert numpy float64 → float32 tensors, then run MC dropout inference."""
    Xtr = torch.tensor(X_train_np, dtype=torch.float32)
    ytr = torch.tensor(y_train_np, dtype=torch.float32)
    Xte = torch.tensor(X_test_np,  dtype=torch.float32)
    mean_preds, _ = mc_dropout_inference(model, Xtr, ytr, Xte, K=K)
    return mean_preds   # np.ndarray (m,)


def compute_metrics(y_true, y_pred):
    """Returns dict with mse, rmse, r2."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse":  float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Section E — OpenML Benchmark
# ---------------------------------------------------------------------------

def preprocess_split(X_train, X_test, categorical_indicator):
    """
    Fit encoder + imputer on X_train; apply to both splits.

    Encoding strategy (fit on training split only):
      - Categorical, ≤ 10 unique values → OneHotEncoder
      - Categorical, > 10 unique values → OrdinalEncoder
      - Numerical                       → SimpleImputer(strategy='mean')

    Missing value strategy:
      - Training split: mean-imputed (numerical) / most-frequent-imputed (categorical)
      - Test split: residual NaN filled with 0 via np.nan_to_num

    Returns (X_train_proc, X_test_proc) as float64 np.ndarray.
    """
    if categorical_indicator is None:
        categorical_indicator = [False] * X_train.shape[1]

    cat_idx = [i for i, c in enumerate(categorical_indicator) if c]
    num_idx = [i for i, c in enumerate(categorical_indicator) if not c]

    transformers = []
    for i in cat_idx:
        # Cardinality check on training data (ignoring NaN)
        try:
            col_f   = X_train[:, i].astype(float)
            valid   = col_f[~np.isnan(col_f)]
            n_unique = len(np.unique(valid))
        except (ValueError, TypeError):
            n_unique = len(np.unique(X_train[:, i]))

        enc = (OneHotEncoder(sparse_output=False, handle_unknown='ignore')
               if n_unique <= 10 else
               OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        # Pipeline: impute first (handles NaN in categoricals), then encode
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", enc),
        ])
        transformers.append((f"cat_{i}", pipe, [i]))

    if num_idx:
        transformers.append(("num", SimpleImputer(strategy="mean"), num_idx))

    ct = ColumnTransformer(transformers, remainder="drop")
    X_train_proc = ct.fit_transform(X_train).astype(np.float64)
    X_test_proc  = ct.transform(X_test).astype(np.float64)
    X_test_proc  = np.nan_to_num(X_test_proc, nan=0.0)   # fill residual test NaN with 0
    return X_train_proc, X_test_proc


def fetch_tabpfn_datasets(max_samples=MAX_SAMPLES, max_features=MAX_FEATURES,
                          target_n=OPENML_N_DATASETS,
                          max_cat_fraction: float = 0.3):
    """
    Replicate TabPFN v2 Nature 2025 regression benchmark dataset selection:
      AutoML Benchmark regression (OpenML study ~271) + OpenML-CTR23 (study 353),
      filtered to ≤max_samples rows and ≤max_features columns; deduped; capped at target_n.
    """
    openml.config.cache_directory = "/tmp/openml_cache"
    os.makedirs("/tmp/openml_cache", exist_ok=True)

    task_ids = set()

    # 1. AutoML Benchmark regression study (try known IDs; fall back to tag search)
    for study_id in [271, 269, 218]:
        try:
            suite = openml.study.get_suite(study_id)
            if suite.tasks:
                task_ids.update(suite.tasks)
                print(f"  AutoML Benchmark study {study_id}: {len(suite.tasks)} tasks")
                break
        except Exception:
            continue

    # 2. OpenML-CTR23 regression study (study 353 — confirmed)
    try:
        ctr23 = openml.study.get_suite(353)
        task_ids.update(ctr23.tasks)
        print(f"  OpenML-CTR23 study 353: {len(ctr23.tasks)} tasks")
    except Exception as exc:
        print(f"  [WARN] Could not load CTR23 study 353: {exc}")

    datasets = []
    seen_did = set()
    for task_id in sorted(task_ids):
        if len(datasets) >= target_n:
            break
        try:
            task = openml.tasks.get_task(task_id)
            # Only supervised regression tasks
            if "Regression" not in str(task.task_type):
                continue
            dataset = task.get_dataset()
            X, y, categorical_indicator, _ = dataset.get_data(
                dataset_format="array", target=task.target_name
            )
            if X is None or y is None:
                continue
            # Filter: samples and features
            if X.shape[0] > max_samples or X.shape[1] > max_features:
                continue
            # Filter: categorical fraction
            if categorical_indicator is not None:
                cat_frac = sum(categorical_indicator) / max(len(categorical_indicator), 1)
                if cat_frac > max_cat_fraction:
                    continue
            did = dataset.dataset_id
            if did in seen_did:
                continue
            seen_did.add(did)
            if categorical_indicator is None:
                categorical_indicator = [False] * X.shape[1]
            datasets.append({
                "task_id":               task_id,
                "name":                  dataset.name,
                "X":                     X,
                "y":                     y.astype(np.float64),
                "categorical_indicator": categorical_indicator,
            })
            print(f"  Fetched: {dataset.name} (tid={task_id}, "
                  f"n={X.shape[0]}, p={X.shape[1]})")
        except Exception as exc:
            print(f"  [SKIP] task {task_id}: {exc}")

    print(f"Fetched {len(datasets)}/{target_n} TabPFN benchmark datasets.")
    return datasets


def fetch_staged_kaggle_datasets(stage_path="@META_DATASET_STAGE/kaggle/",
                                 local_dir="/tmp/kaggle_cache",
                                 task_type_filter="regression",
                                 require_numeric_only: bool = True):
    """
    Download .npz files from the Kaggle stage path and return them in the same
    format as fetch_tabpfn_datasets(). Only returns datasets matching task_type_filter.

    Falls back gracefully if no active Snowpark session (e.g., local dev).
    Returns list of dicts: {task_id, name, X, y, categorical_indicator, source}.

    DeepSetModel fitness:
      s3e5 Wine Quality     — 11 numerical, ordinal target 3–9; moderate fit.
      s3e9 Concrete Strength — 8 numerical, ~1030 rows; best fit (small n is model's strength).
      Both are genuine OOD generalization tests — unseen by the model during training.
    """
    os.makedirs(local_dir, exist_ok=True)

    # Download from stage if session available
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        rows = session.sql(f"LIST {stage_path}").collect()
        for row in rows:
            # row["name"] = "meta_dataset_stage/kaggle/playground-series-s3e9.npz"
            fname      = os.path.basename(row["name"])
            local_path = os.path.join(local_dir, fname)
            if not os.path.exists(local_path):   # skip if already cached
                session.file.get(f"{stage_path}{fname}", local_dir)
        print(f"  Downloaded {len(rows)} files from {stage_path}")
    except Exception as exc:
        print(f"  [WARN] Snowflake stage unavailable ({exc}); using cached files in {local_dir}")

    # Load .npz files
    datasets = []
    for fname in sorted(os.listdir(local_dir)):
        if not fname.endswith(".npz"):
            continue
        try:
            data      = np.load(os.path.join(local_dir, fname), allow_pickle=True)
            task_type = str(data["task_type"][0])
            if task_type_filter and task_type != task_type_filter:
                continue   # skip classification datasets in regression benchmark
            # Guard: skip datasets with categorical features if require_numeric_only
            cat_mask = data.get("categorical_indicator", np.zeros(data["X"].shape[1], dtype=bool))
            if require_numeric_only and np.asarray(cat_mask).any():
                slug = str(data.get("slug", [fname])[0])
                print(f"[Kaggle] Skipping {slug}: {np.asarray(cat_mask).sum()} categorical "
                      f"feature(s) detected — model trained on numeric DGPs only")
                continue
            X = data["X"].astype(np.float64)
            datasets.append({
                "task_id":               str(data["slug"][0]),
                "name":                  str(data["dataset_name"][0]),
                "X":                     X,
                "y":                     data["y"].astype(np.float64),
                "categorical_indicator": data["categorical_indicator"].tolist(),
                "source":                "kaggle",
            })
            print(f"  Loaded Kaggle ({task_type}): {fname}  n={X.shape[0]}, p={X.shape[1]}")
        except Exception as exc:
            print(f"  [SKIP] {fname}: {exc}")

    print(f"Loaded {len(datasets)} Kaggle regression datasets.")
    return datasets


def rank_methods(metrics_matrix, higher_is_better=False):
    """
    NaN-aware per-dataset ranking.

    Args:
        metrics_matrix: np.ndarray shape (n_methods, n_datasets), may contain NaN.
        higher_is_better: if True, rank 1 = highest value (used for R²).

    Returns:
        rank_matrix: same shape; NaN where method had no result for that dataset.
    """
    n_methods, n_datasets = metrics_matrix.shape
    rank_matrix = np.full_like(metrics_matrix, np.nan)

    for j in range(n_datasets):
        col   = metrics_matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        vals = col[valid]
        if higher_is_better:
            vals = -vals   # invert so rank 1 = best
        ranks = stats.rankdata(vals, method="average")
        rank_matrix[valid, j] = ranks

    return rank_matrix


def add_rank_columns(detailed_df):
    """Add per-suite/task/rep rank columns for mse, rmse, and r2."""
    if detailed_df is None or detailed_df.empty:
        return detailed_df

    out = detailed_df.copy()
    group_cols = ["source", "task_id", "rep"]
    rank_specs = {
        "mse": True,
        "rmse": True,
        "r2": False,
    }

    for metric, ascending in rank_specs.items():
        out[f"rank_{metric}"] = (
            out.groupby(group_cols)[metric]
            .rank(method="average", ascending=ascending, na_option="bottom")
        )
        out.loc[out[metric].isna(), f"rank_{metric}"] = np.nan
    return out


def predict_autogluon(X_train_np, y_train_np, X_test_np):
    """Fit an AutoGluon stacked ensemble in /tmp and remove artifacts afterward."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGluon is not available. Install autogluon.tabular[all]==1.0.0."
        ) from exc

    feature_cols = [f"f{i}" for i in range(X_train_np.shape[1])]
    train_df = pd.DataFrame(X_train_np, columns=feature_cols)
    train_df["target"] = y_train_np
    test_df = pd.DataFrame(X_test_np, columns=feature_cols)
    model_dir = tempfile.mkdtemp(prefix="autogluon_", dir="/tmp")
    try:
        predictor = TabularPredictor(
            label="target",
            path=model_dir,
            problem_type="regression",
            verbosity=0,
        )
        predictor.fit(
            train_df,
            presets="best_quality",
            time_limit=AUTOGLUON_TIME_LIMIT,
            num_cpus=1,
            num_gpus=0,
            verbosity=0,
        )
        return predictor.predict(test_df).to_numpy()
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


def normalize_benchmark_columns(detailed_df):
    """Ensure the canonical comparison CSV has explicit suite and dataset fields."""
    if detailed_df is None or detailed_df.empty:
        return detailed_df

    out = detailed_df.copy()
    if "benchmark_suite" not in out.columns:
        out["benchmark_suite"] = out.get("source", "benchmark")
    if "dataset_name" not in out.columns:
        out["dataset_name"] = out.get("name", "")
    if "seed" not in out.columns:
        out["seed"] = out.get("rep", np.nan)
    return add_rank_columns(out)


def benchmark_aggregation_smoke_test():
    """Tiny guardrail for benchmark normalization, ranking, and aggregation."""
    rows = []
    for rep in [0, 1]:
        for task_id in ["d1", "d2"]:
            rows.append({
                "source": "smoke",
                "task_id": task_id,
                "name": task_id,
                "rep": rep,
                "method": "method_a",
                "mse": 1.0 + rep,
                "rmse": 1.0 + rep,
                "r2": 0.8 - rep * 0.1,
            })
            rows.append({
                "source": "smoke",
                "task_id": task_id,
                "name": task_id,
                "rep": rep,
                "method": "method_b",
                "mse": 2.0 + rep,
                "rmse": 2.0 + rep,
                "r2": 0.4 - rep * 0.1,
            })

    normalized = normalize_benchmark_columns(pd.DataFrame(rows))
    required = {"rank_mse", "rank_rmse", "rank_r2", "benchmark_suite", "dataset_name", "seed"}
    missing = required - set(normalized.columns)
    if missing:
        raise AssertionError(f"Benchmark normalization missing columns: {sorted(missing)}")
    if normalized[["rank_mse", "rank_rmse", "rank_r2"]].isna().any().any():
        raise AssertionError("Benchmark smoke ranks contain unexpected NaN values")

    rank_summary_df, metric_summary_df = aggregate_benchmark_results(normalized)
    if len(rank_summary_df) != 2 or len(metric_summary_df) != 2:
        raise AssertionError("Benchmark smoke aggregation did not preserve both methods")
    return normalized, rank_summary_df, metric_summary_df


def aggregate_benchmark_results(detailed_df):
    """
    Compute mean ± 95% CI (t-distribution, df=4) for ranks and raw metrics.

    Rank CI:
      Per rep → rank methods across datasets → mean rank per method.
      Across 5 reps → mean ± 95% CI of the 5 rep-level mean ranks.

    Metric CI:
      Per rep → mean metric over datasets.
      Across 5 reps → mean ± 95% CI.

    Returns (rank_summary_df, metric_summary_df).
    """
    detailed_df = normalize_benchmark_columns(detailed_df)
    methods = list(detailed_df["method"].unique())
    reps = sorted(detailed_df["rep"].dropna().unique())
    metrics = ["mse", "rmse", "r2"]
    hib     = {"mse": False, "rmse": False, "r2": True}

    def ci95(vals):
        """Returns (mean, lo, hi) for a 1-D array; NaN if < 2 valid values."""
        valid = vals[~np.isnan(vals)]
        if len(valid) < 2:
            mu = float(np.nanmean(vals)) if len(valid) == 1 else float("nan")
            return mu, float("nan"), float("nan")
        mu  = float(np.mean(valid))
        sem = float(stats.sem(valid))
        if sem == 0.0:
            return mu, mu, mu
        lo, hi = stats.t.interval(0.95, df=len(valid) - 1, loc=mu, scale=sem)
        return mu, float(lo), float(hi)

    # --- Metric summary ---
    metric_rows = []
    for method in methods:
        mdf = detailed_df[detailed_df["method"] == method]
        row = {"method": method}
        for metric in metrics:
            rep_means = (
                mdf.groupby("rep")[metric]
                .mean()
                .reindex(reps)
                .values.astype(float)
            )
            mu, lo, hi = ci95(rep_means)
            row[f"{metric}_mean"]  = mu
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        metric_rows.append(row)
    metric_summary_df = pd.DataFrame(metric_rows)

    # --- Rank summary (per rep, rank per dataset column, then mean across datasets) ---
    rep_rank_means = {metric: {m: [] for m in methods} for metric in metrics}

    for rep in reps:
        rep_df      = detailed_df[detailed_df["rep"] == rep]
        dataset_ids = list(rep_df["task_id"].unique())

        for metric in metrics:
            mat = np.full((len(methods), len(dataset_ids)), np.nan)
            for j, did in enumerate(dataset_ids):
                sub = rep_df[rep_df["task_id"] == did]
                for i, method in enumerate(methods):
                    vals = sub[sub["method"] == method][metric].values
                    if len(vals) > 0 and not np.isnan(vals[0]):
                        mat[i, j] = float(vals[0])

            rank_mat = rank_methods(mat, higher_is_better=hib[metric])
            for i, method in enumerate(methods):
                rep_rank_means[metric][method].append(
                    float(np.nanmean(rank_mat[i, :]))
                )

    rank_rows = []
    for method in methods:
        row = {"method": method}
        for metric in metrics:
            vals = np.array(rep_rank_means[metric][method])
            mu, lo, hi = ci95(vals)
            row[f"rank_{metric}_mean"]  = mu
            row[f"rank_{metric}_ci_lo"] = lo
            row[f"rank_{metric}_ci_hi"] = hi
        rank_rows.append(row)
    rank_summary_df = pd.DataFrame(rank_rows)

    return rank_summary_df, metric_summary_df


def run_openml_benchmark(model, n_datasets=OPENML_N_DATASETS, seeds=SEEDS,
                         mc_K=N_MC_DROPOUT, results_dir="results/",
                         include_openml=True, include_kaggle=True,
                         methods=None, dataset_limit=None):
    """
    Run TabPFN v2 Nature 2025 aligned regression benchmark (OpenML + optional Kaggle).

    Outer loop: seed in SEEDS (10 reps, seeds 0–9)
      Inner loop: dataset in datasets (up to 28 OpenML + 2 Kaggle regression datasets)
        - Subsample to MAX_SAMPLES if needed
        - 90/10 train/test split (same split for ALL methods)
        - preprocess_split: encode categoricals, impute train NaN, fill test NaN with 0
        - DeepSetModel-MC + 9 baselines; store NaN on per-method failure

    Returns (detailed_df, rank_summary_df, metric_summary_df).
    Note: Metrics not directly comparable to the TabPFN paper (different split ratio).
    The 'source' column in detailed_df identifies "openml" vs "kaggle" datasets,
    enabling stratified analysis of DeepSet OOD generalization across dataset types.
    """
    openml_datasets = []
    if include_openml:
        print("\nFetching TabPFN benchmark datasets (AutoML Benchmark + OpenML-CTR23)...")
        openml_datasets = fetch_tabpfn_datasets(target_n=n_datasets)

    kaggle_datasets = []
    if include_kaggle:
        print("\nFetching Kaggle benchmark datasets from stage...")
        kaggle_datasets = fetch_staged_kaggle_datasets()

    datasets = openml_datasets + kaggle_datasets
    if dataset_limit is not None:
        datasets = datasets[:int(dataset_limit)]
    if not datasets:
        print("[WARNING] No datasets fetched. Skipping benchmark.")
        return None, None, None

    selected_methods = methods or ALL_BENCHMARK_METHODS
    unknown_methods = sorted(set(selected_methods) - set(ALL_BENCHMARK_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown benchmark method(s): {unknown_methods}")
    if "DeepSetModel-MC" in selected_methods and model is None:
        raise ValueError("DeepSetModel-MC benchmark requires a loaded model.")

    all_rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        print(f"\n--- Benchmark rep seed={seed} ---")

        for ds in datasets:
            X, y    = ds["X"], ds["y"]
            cat_ind = ds["categorical_indicator"]
            name    = ds["name"]
            task_id = ds["task_id"]

            # Subsample
            if X.shape[0] > MAX_SAMPLES:
                idx = rng.choice(X.shape[0], MAX_SAMPLES, replace=False)
                X, y = X[idx], y[idx]

            # 50/50 split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1.0 - TRAIN_FRAC, random_state=seed
            )

            # Preprocess (same data for all methods)
            try:
                X_train_p, X_test_p = preprocess_split(X_train, X_test, cat_ind)
            except Exception as exc:
                print(f"  [SKIP preprocess] {name}: {exc}")
                continue

            source  = ds.get("source", "openml")
            nan_row = {"task_id": task_id, "name": name, "rep": seed, "source": source,
                       "mse": float("nan"), "rmse": float("nan"), "r2": float("nan")}

            if "DeepSetModel-MC" in selected_methods:
                try:
                    preds = predict_deepset_mc(model, X_train_p, y_train, X_test_p, K=mc_K)
                    m     = compute_metrics(y_test, preds)
                    all_rows.append({"task_id": task_id, "name": name, "rep": seed,
                                      "source": source, "method": "DeepSetModel-MC", **m})
                except Exception as exc:
                    print(f"  [FAIL DeepSetModel-MC] {name}: {exc}")
                    all_rows.append({**nan_row, "method": "DeepSetModel-MC"})

            # Baselines
            for bl_name, bl_model in get_baselines(seed).items():
                if bl_name not in selected_methods:
                    continue
                try:
                    bl_model.fit(X_train_p, y_train)
                    preds_bl = bl_model.predict(X_test_p)
                    m = compute_metrics(y_test, preds_bl)
                    all_rows.append({"task_id": task_id, "name": name, "rep": seed,
                                     "source": source, "method": bl_name, **m})
                except Exception as exc:
                    print(f"  [FAIL {bl_name}] {name}: {exc}")
                    all_rows.append({**nan_row, "method": bl_name})

            if "AutoGluon" in selected_methods:
                try:
                    preds_ag = predict_autogluon(X_train_p, y_train, X_test_p)
                    m = compute_metrics(y_test, preds_ag)
                    all_rows.append({"task_id": task_id, "name": name, "rep": seed,
                                     "source": source, "method": "AutoGluon", **m})
                except Exception as exc:
                    print(f"  [FAIL AutoGluon] {name}: {exc}")
                    all_rows.append({**nan_row, "method": "AutoGluon"})

    if not all_rows:
        return None, None, None

    detailed_df = normalize_benchmark_columns(pd.DataFrame(all_rows))
    rank_summary_df, metric_summary_df = aggregate_benchmark_results(detailed_df)
    return detailed_df, rank_summary_df, metric_summary_df


def print_benchmark_table(rank_summary_df, metric_summary_df):
    """Print M.Rank and Mean Metric tables to console (TabPFN-style)."""

    def fmt(mu, lo, hi):
        if np.isnan(mu):
            return "N/A"
        if np.isnan(lo):
            return f"{mu:.2f}"
        return f"{mu:.2f} [{lo:.2f},{hi:.2f}]"

    print("\n=== Mean Ranks (lower = better; R2: lower rank = higher R2) ===")
    header = (f"{'Method':<22}  {'M.Rank MSE':>24}  {'M.Rank R2':>24}  {'M.Rank RMSE':>24}")
    print(header)
    print("-" * len(header))
    for _, row in rank_summary_df.sort_values("rank_mse_mean").iterrows():
        print(
            f"{row['method']:<22}"
            f"  {fmt(row['rank_mse_mean'],  row['rank_mse_ci_lo'],  row['rank_mse_ci_hi']):>24}"
            f"  {fmt(row['rank_r2_mean'],   row['rank_r2_ci_lo'],   row['rank_r2_ci_hi']):>24}"
            f"  {fmt(row['rank_rmse_mean'], row['rank_rmse_ci_lo'], row['rank_rmse_ci_hi']):>24}"
        )

    print("\n=== Mean Metrics (10 reps × all datasets, 95% CI) ===")
    header2 = (f"{'Method':<22}  {'Mean MSE':>26}  {'Mean R2':>26}  {'Mean RMSE':>26}")
    print(header2)
    print("-" * len(header2))
    for _, row in metric_summary_df.sort_values("mse_mean").iterrows():
        print(
            f"{row['method']:<22}"
            f"  {fmt(row['mse_mean'],  row['mse_ci_lo'],  row['mse_ci_hi']):>26}"
            f"  {fmt(row['r2_mean'],   row['r2_ci_lo'],   row['r2_ci_hi']):>26}"
            f"  {fmt(row['rmse_mean'], row['rmse_ci_lo'], row['rmse_ci_hi']):>26}"
        )


def save_benchmark_csvs(detailed_df, rank_summary_df, metric_summary_df, results_dir):
    """Save canonical model comparison CSVs to results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    detailed_path = os.path.join(results_dir, "model_comparison.csv")
    summary_path  = os.path.join(results_dir, "model_comparison_summary.csv")

    detailed_df = normalize_benchmark_columns(detailed_df)
    preferred = [
        "benchmark_suite", "source", "task_id", "dataset_name", "name",
        "rep", "seed", "method", "mse", "rmse", "r2",
        "rank_mse", "rank_rmse", "rank_r2",
    ]
    cols = [c for c in preferred if c in detailed_df.columns]
    cols.extend([c for c in detailed_df.columns if c not in cols])
    detailed_df[cols].to_csv(detailed_path, index=False)

    summary_df = rank_summary_df.merge(metric_summary_df, on="method")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved {detailed_path}  ({len(detailed_df)} rows)")
    print(f"Saved {summary_path}")
    return detailed_path, summary_path


def safe_method_name(method):
    return method.replace("/", "_").replace(" ", "_")


def save_benchmark_part_csv(detailed_df, method, results_dir):
    """Save a single-method benchmark detail file under benchmark_parts/."""
    part_dir = os.path.join(results_dir, "benchmark_parts")
    os.makedirs(part_dir, exist_ok=True)
    part_path = os.path.join(part_dir, f"{safe_method_name(method)}_detailed.csv")
    normalize_benchmark_columns(detailed_df).to_csv(part_path, index=False)
    print(f"Saved {part_path} ({len(detailed_df)} rows)")
    return part_path


def download_stage_prefix(stage_path, local_dir):
    """Download a stage prefix into local_dir; no-op outside Snowflake."""
    os.makedirs(local_dir, exist_ok=True)
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        session.file.get(stage_path, local_dir)
        print(f"Downloaded {stage_path} to {local_dir}")
    except Exception as exc:
        print(f"[WARNING] Snowpark download failed (skipping): {exc}")


def aggregate_benchmark_parts(results_dir, eval_stage=EVAL_RESULTS_STAGE):
    """Create model_comparison.csv from staged per-method benchmark part files."""
    local_parts = os.path.join(results_dir, "benchmark_parts")
    download_stage_prefix(f"{eval_stage}/benchmark_parts/", local_parts)

    part_files = sorted(glob.glob(os.path.join(local_parts, "*_detailed.csv*")))
    if not part_files:
        raise FileNotFoundError(f"No benchmark part CSVs found in {local_parts}")

    frames = [pd.read_csv(path) for path in part_files]
    detailed_df = normalize_benchmark_columns(pd.concat(frames, ignore_index=True))
    rank_summary_df, metric_summary_df = aggregate_benchmark_results(detailed_df)
    return save_benchmark_csvs(detailed_df, rank_summary_df, metric_summary_df, results_dir)


# ---------------------------------------------------------------------------
# Section F — Upload results to Snowflake (fixed: get_active_session)
# ---------------------------------------------------------------------------

def upload_to_snowflake(local_path: str, stage_path: str):
    """
    Upload a local file to a Snowflake internal stage via Snowpark.
    Uses get_active_session() — the only correct pattern inside SPCS containers.
    Degrades gracefully when running locally (no active session).
    """
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        session.file.put(local_path, stage_path, overwrite=True, auto_compress=False)
        print(f"Uploaded {local_path} to {stage_path}")
    except Exception as exc:
        print(f"[WARNING] Snowpark upload failed (skipping): {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSetModel checkpoint.")
    parser.add_argument("--model_path",  default=os.environ.get("MODEL_PATH", "best.pt"),
                        help="Path to model checkpoint.")
    parser.add_argument("--data_dir",    default=os.environ.get("DATA_DIR", "/tmp/data"),
                        help="Root data directory (contains test/ subdir).")
    parser.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "results/"),
                        help="Directory for output files.")
    parser.add_argument("--no_openml",   action="store_true",
                        help="Skip OpenML datasets; run Kaggle benchmark only.")
    parser.add_argument("--no_kaggle",   action="store_true",
                        help="Skip Kaggle staged datasets (use if @META_DATASET_STAGE/kaggle/ "
                             "not yet populated).")
    parser.add_argument("--mc_K",        type=int, default=N_MC_DROPOUT,
                        help="Number of MC dropout forward passes (default: 32).")
    parser.add_argument("--mode",        default=os.environ.get("EVAL_MODE", "full"),
                        choices=["full", "synthetic", "benchmark", "aggregate"],
                        help="Evaluation mode.")
    parser.add_argument("--benchmark_method",
                        default=os.environ.get("BENCHMARK_METHOD"),
                        help="Run only one benchmark method.")
    parser.add_argument("--eval_results_stage",
                        default=os.environ.get("EVAL_RESULTS_STAGE", EVAL_RESULTS_STAGE),
                        help="Snowflake stage for evaluation CSV outputs.")
    parser.add_argument("--benchmark_dataset_limit", type=int,
                        default=os.environ.get("BENCHMARK_DATASET_LIMIT"),
                        help="Limit benchmark datasets for one-dataset smoke runs.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode == "aggregate":
        det_path, sum_path = aggregate_benchmark_parts(args.results_dir, args.eval_results_stage)
        upload_to_snowflake(det_path, args.eval_results_stage)
        upload_to_snowflake(sum_path, args.eval_results_stage)
        return

    needs_model = (
        args.mode in ("full", "synthetic")
        or args.benchmark_method in (None, "DeepSetModel-MC")
    )
    model = load_model(args.model_path) if needs_model else None

    if args.mode in ("full", "synthetic"):
        materialize_meta_dataset_stage(args.data_dir, splits=("test",))
        all_pass = run_permutation_tests(model)
        if not all_pass:
            print("[WARNING] One or more permutation tests FAILED.")

        test_dir = os.path.join(args.data_dir, "test")
        print(f"\nEvaluating on synthetic DGP test files in {test_dir} ...")
        try:
            records = evaluate_synthetic_dgp(model, test_dir)
            print(f"Evaluated {len(records)} datasets.")

            report_rows = build_report(records)
            print("\nTest Split Results (Point Prediction):")
            print_report(report_rows)

            synthetic_dir = os.path.join(args.results_dir, "synthetic")
            os.makedirs(synthetic_dir, exist_ok=True)
            csv_path = os.path.join(synthetic_dir, "test_report.csv")
            save_report_csv(report_rows, csv_path)
            print(f"Saved {csv_path}")
            upload_to_snowflake(csv_path, f"{args.eval_results_stage}/synthetic/")

            print(f"\nRunning MC dropout noise assessment (K={args.mc_K}) ...")
            mc_records = evaluate_synthetic_dgp_mc(model, test_dir, K=args.mc_K)
            mc_rows = build_mc_report(mc_records)
            print("\nMC Dropout Noise Assessment (ratio_mc_ols <= ratio_model_ols = noise reduction):")
            print_mc_report(mc_rows)

            mc_csv_path = os.path.join(synthetic_dir, "mc_report.csv")
            save_mc_report_csv(mc_rows, mc_csv_path)
            print(f"Saved {mc_csv_path}")
            upload_to_snowflake(mc_csv_path, f"{args.eval_results_stage}/synthetic/")

        except FileNotFoundError as exc:
            print(f"[WARNING] {exc} - skipping synthetic DGP evaluation.")

    if args.mode == "synthetic":
        return

    if args.no_openml and args.no_kaggle:
        print("\n--no_openml and --no_kaggle both set: skipping benchmark.")
        return

    if not BENCHMARK_DEPS_AVAILABLE:
        raise RuntimeError(
            "Benchmark dependencies are not available. Install: "
            "openml scikit-learn xgboost lightgbm catboost pandas scipy"
        )
        return

    methods = [args.benchmark_method] if args.benchmark_method else ALL_BENCHMARK_METHODS
    detailed_df, rank_summary_df, metric_summary_df = run_openml_benchmark(
        model, results_dir=args.results_dir, mc_K=args.mc_K,
        include_openml=not args.no_openml,
        include_kaggle=not args.no_kaggle,
        methods=methods,
        dataset_limit=args.benchmark_dataset_limit,
    )

    if detailed_df is not None:
        print_benchmark_table(rank_summary_df, metric_summary_df)
        if args.benchmark_method:
            part_path = save_benchmark_part_csv(detailed_df, args.benchmark_method, args.results_dir)
            upload_to_snowflake(part_path, f"{args.eval_results_stage}/benchmark_parts/")
        else:
            det_path, sum_path = save_benchmark_csvs(
                detailed_df, rank_summary_df, metric_summary_df, args.results_dir
            )
            upload_to_snowflake(det_path, args.eval_results_stage)
            upload_to_snowflake(sum_path, args.eval_results_stage)

if __name__ == "__main__":
    main()
