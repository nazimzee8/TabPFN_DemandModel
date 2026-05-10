"""
evaluate.py

Evaluate a saved DeepSetModel checkpoint:
  1. Permutation-invariance tests (7 synthetic)
  2. Synthetic DGP evaluation: DeepSetModel vs OLS, stratified by regime (A/B/C/D)
  3. MC dropout noise assessment on synthetic DGP
  4. Prepared OpenML + Kaggle regression benchmark: DeepSetModel-MC vs 9 baselines
     (prepared manifest datasets, 10 reps, 90/10 split, <=10k samples, <=500 features, 95% CI)
     Datasets are prepared by prepare_benchmark_datasets.py before benchmark shards run.
     Methods: DeepSetModel-MC, XGBoost, LightGBM, CatBoost, RandomForest,
              KNN, LinearRegression, Ridge, SVR, MLP, AutoGluon

Usage (inside container or locally):
    python evaluate.py --model_path best.pt --data_dir /tmp/data --results_dir results/

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
import gc
import os
import glob
import shutil
import tempfile
import warnings
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
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
BENCHMARK_DEEPSET_CONTEXT_SIZE = int(os.environ.get("BENCHMARK_DEEPSET_CONTEXT_SIZE", "200"))
BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES = int(os.environ.get("BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES", "5"))
BENCHMARK_DEEPSET_TEST_BATCH_SIZE = int(os.environ.get("BENCHMARK_DEEPSET_TEST_BATCH_SIZE", "128"))
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


def parse_benchmark_methods(benchmark_method=None, benchmark_methods=None):
    """Resolve benchmark method env/CLI selection into an explicit method list."""
    if benchmark_method and benchmark_methods:
        raise ValueError(
            "Set only one of BENCHMARK_METHOD or BENCHMARK_METHODS; "
            "BENCHMARK_METHODS is a comma-separated list."
        )
    if benchmark_methods:
        methods = [m.strip() for m in benchmark_methods.split(",") if m.strip()]
        if not methods:
            raise ValueError("BENCHMARK_METHODS was set but did not contain any method names.")
        return methods
    if benchmark_method:
        return [benchmark_method]
    return list(ALL_BENCHMARK_METHODS)


AUTOGLUON_TIME_LIMIT = int(os.environ.get("AUTOGLUON_TIME_LIMIT", "300"))
BENCHMARK_NUM_CPUS = int(os.environ.get("BENCHMARK_NUM_CPUS", "1"))
BENCHMARK_PREPARED_STAGE = os.environ.get(
    "BENCHMARK_PREPARED_STAGE", "@META_DATASET_STAGE/benchmark_prepared/"
)
BENCHMARK_MANIFEST_STAGE_PATH = os.environ.get(
    "BENCHMARK_MANIFEST_PATH",
    "@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json",
)
BENCHMARK_DATASET_INDEX_TABLE = os.environ.get(
    "BENCHMARK_DATASET_INDEX_TABLE", "BENCHMARK_DATASET_INDEX"
)
BENCHMARK_SHARD_STRATEGY = os.environ.get("BENCHMARK_SHARD_STRATEGY", "modulo").lower()


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

def model_config_to_dict(cfg):
    """
    Convert ModelConfig into a plain dict safe for torch.load(weights_only=True).
    Called when saving new checkpoints. ModelConfig is a dataclass.
    """
    import dataclasses
    if isinstance(cfg, dict):
        return dict(cfg)
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def model_config_from_payload(payload):
    """
    Convert a checkpoint cfg payload into a ModelConfig instance.
    Accepts ModelConfig (legacy trusted load) or plain dict (new format).
    """
    if isinstance(payload, ModelConfig):
        return payload
    if isinstance(payload, dict):
        import dataclasses
        allowed = {f.name for f in dataclasses.fields(ModelConfig)}
        unknown = set(payload.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown ModelConfig field(s) in checkpoint: {sorted(unknown)}"
            )
        return ModelConfig(**payload)
    raise TypeError(
        f"Unsupported checkpoint cfg payload type: {type(payload)!r}"
    )


def default_model_config():
    """
    Canonical default config for bare state-dict checkpoints (legacy format 0).
    Matches the defaults used during the original training run.
    """
    return ModelConfig(
        d_phi=128, d_rho=256, pool="pna", dropout=0.1,
        n_sab_feat=0, n_sab_samp=0,
        norm_feat=False, norm_target=False,
    )


def load_checkpoint_compat(model_path):
    """
    Load a DeepSet checkpoint in a PyTorch 2.6+ compatible way.

    Strategy (in order):
    1. torch.load(..., weights_only=True) — preferred; works for new format-v2 checkpoints.
    2. safe_globals([ModelConfig]) + weights_only=True — for legacy checkpoints with pickled
       ModelConfig. Still safe: only allowlists our own known class.
    3. weights_only=False — only if ALLOW_UNSAFE_TORCH_LOAD=true env var is set.
       Security warning is printed. Intended only for trusted internally generated artifacts.

    Note: weights_only=False can execute arbitrary code from the checkpoint file.
    Never use it with checkpoints from untrusted sources.
    """
    # Attempt 1: clean weights_only=True
    try:
        return torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        # PyTorch version does not support weights_only parameter at all
        if "weights_only" not in str(exc):
            raise
        print(
            "[WARNING] PyTorch version does not support weights_only; "
            "falling back to legacy torch.load. Use only trusted checkpoints.",
            flush=True,
        )
        return torch.load(model_path, map_location="cpu")
    except Exception as exc:
        msg = str(exc)
        is_modelconfig_issue = (
            "ModelConfig" in msg
            or "Weights only load failed" in msg
            or "Unsupported global" in msg
            or "UnpicklingError" in msg
        )
        if not is_modelconfig_issue:
            raise

    # Attempt 2: allowlist ModelConfig via safe_globals (still weights_only=True)
    print(
        "[WARNING] Safe checkpoint load failed — legacy pickled ModelConfig detected. "
        "Retrying with torch.serialization.safe_globals([ModelConfig]). "
        "This is safe for internally generated checkpoints.",
        flush=True,
    )
    try:
        from torch.serialization import safe_globals
        with safe_globals([ModelConfig]):
            return torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as safe_exc:
        pass

    # Attempt 3: weights_only=False — requires opt-in env var
    allow_unsafe = (
        os.environ.get("ALLOW_UNSAFE_TORCH_LOAD", "false").lower() == "true"
    )
    if not allow_unsafe:
        raise RuntimeError(
            f"Failed to load checkpoint {model_path!r} with weights_only=True even after "
            "allowlisting ModelConfig. If this is a trusted internally generated checkpoint, "
            "set ALLOW_UNSAFE_TORCH_LOAD=true as a temporary escape hatch. "
            "Long-term fix: resave with cfg as a plain dict (checkpoint_format_version=2)."
        )
    print(
        "[SECURITY WARNING] ALLOW_UNSAFE_TORCH_LOAD=true — loading checkpoint with "
        "weights_only=False. Only do this for trusted internally generated checkpoints. "
        "weights_only=False can execute arbitrary code embedded in the checkpoint file.",
        flush=True,
    )
    return torch.load(model_path, map_location="cpu", weights_only=False)


def load_model(model_path):
    if not os.path.exists(model_path):
        download_stage_prefix(CHECKPOINT_STAGE, os.path.dirname(model_path) or ".")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint {model_path!r} does not exist after attempting Snowflake stage "
            f"fetch from {CHECKPOINT_STAGE}."
        )

    print(f"Loading checkpoint from {model_path}", flush=True)
    ckpt = load_checkpoint_compat(model_path)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg_payload = ckpt.get("cfg")
        if cfg_payload is None:
            cfg = default_model_config()
            fmt = "dict_with_state_dict_no_cfg"
        else:
            cfg = model_config_from_payload(cfg_payload)
            fmt = (
                f"format_v{ckpt['checkpoint_format_version']}"
                if "checkpoint_format_version" in ckpt
                else "legacy_dict_with_cfg"
            )
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # Bare state dict — no "state_dict" key, just weight tensors
        cfg = default_model_config()
        state_dict = ckpt
        fmt = "bare_state_dict"
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(ckpt)!r}")

    print(f"Checkpoint format: {fmt}", flush=True)
    print(f"ModelConfig: {cfg}", flush=True)
    print(f"State dict tensors: {len(state_dict)}", flush=True)

    model = DeepSetModel(cfg=cfg)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {model_path}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Section B — Evaluate on held-out synthetic DGP test split
# ---------------------------------------------------------------------------

def evaluate_synthetic_dgp(model, test_dir, rank=0, world_size=1,
                            using_torch_distributed=False):
    """
    Predict and compute OLS baseline for every test Parquet file.
    Returns list of per-dataset dicts: model_mse, ols_mse, prior_regime, n, p.
    In distributed mode, each rank processes a round-robin shard; rank 0 gathers
    and returns all records; non-rank-0 workers return None.
    """
    files = sorted(
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {test_dir}")

    if world_size > 1:
        files = [f for i, f in enumerate(files) if i % world_size == rank]

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

    if using_torch_distributed and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        if rank != 0:
            return None
        records = [r for chunk in gathered for r in chunk]
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


def evaluate_synthetic_dgp_mc(model, test_dir, K=32, rank=0, world_size=1,
                               using_torch_distributed=False):
    """Like evaluate_synthetic_dgp but adds mc_mse and mc_std_mean columns.
    In distributed mode, each rank processes a round-robin shard; rank 0 gathers
    and returns all records; non-rank-0 workers return None.
    """
    files = sorted(
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {test_dir}")

    if world_size > 1:
        files = [f for i, f in enumerate(files) if i % world_size == rank]

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

    if using_torch_distributed and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        if rank != 0:
            return None
        records = [r for chunk in gathered for r in chunk]
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


def select_deepset_context_indices(n_train, context_size, seed, context_index):
    """
    Deterministically select one bounded DeepSet context from processed train rows.
    Rows are sampled without replacement from the already-split training set only.
    """
    n_train = int(n_train)
    context_size = int(context_size)
    if n_train <= 0:
        raise ValueError("DeepSet context selection requires at least one training row.")
    if context_size <= 0:
        raise ValueError("BENCHMARK_DEEPSET_CONTEXT_SIZE must be positive.")
    if n_train <= context_size:
        return np.arange(n_train, dtype=np.int64)

    rng_seed = (int(seed) + 1) * 1_000_003 + int(context_index) * 97_003
    rng = np.random.default_rng(rng_seed)
    return np.sort(rng.choice(n_train, size=context_size, replace=False)).astype(np.int64)


def deepset_inference_device():
    """Select and log the torch device used only by DeepSet benchmark inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(device)
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(
            f"DeepSet benchmark inference device: cuda ({name}, {total_gb:.1f} GiB)",
            flush=True,
        )
        return device

    print("DeepSet benchmark inference device: cpu (CUDA unavailable)", flush=True)
    return torch.device("cpu")


def predict_deepset_mc_streamed(
    model,
    X_train_np,
    y_train_np,
    X_test_np,
    K=32,
    test_batch_size=128,
    device=None,
):
    """
    MC dropout mean for one bounded context, streamed over test rows.
    Chunking is memory-only: output order and length match X_test_np exactly.
    """
    if K <= 0:
        raise ValueError("MC dropout K must be positive.")
    if test_batch_size <= 0:
        raise ValueError("BENCHMARK_DEEPSET_TEST_BATCH_SIZE must be positive.")

    device = device or deepset_inference_device()
    model.to(device)
    Xtr = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)
    n_test = int(X_test_np.shape[0])
    out = np.empty(n_test, dtype=np.float64)

    model.train()
    try:
        with torch.no_grad():
            for start in range(0, n_test, test_batch_size):
                end = min(start + test_batch_size, n_test)
                Xte = torch.as_tensor(X_test_np[start:end], dtype=torch.float32, device=device)
                pred_sum = None
                for _ in range(K):
                    pred = model(Xtr, ytr, Xte).detach()
                    pred_sum = pred if pred_sum is None else pred_sum + pred
                out[start:end] = (pred_sum / float(K)).cpu().numpy()
    finally:
        model.eval()
    return out


def predict_deepset_bounded_context_ensemble(
    model,
    X_train_np,
    y_train_np,
    X_test_np,
    seed,
    K=32,
    context_size=BENCHMARK_DEEPSET_CONTEXT_SIZE,
    context_ensembles=BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES,
    test_batch_size=BENCHMARK_DEEPSET_TEST_BATCH_SIZE,
    device=None,
):
    """
    DeepSetModel-MC bounded-context benchmark prediction.

    Deterministic train-only contexts each predict the same full processed test
    split; predictions are averaged once before metrics are computed.
    """
    if context_ensembles <= 0:
        raise ValueError("BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES must be positive.")

    device = device or deepset_inference_device()
    context_preds = []
    for context_index in range(int(context_ensembles)):
        idx = select_deepset_context_indices(
            X_train_np.shape[0], context_size, seed, context_index
        )
        if len(idx) > context_size:
            raise AssertionError("DeepSet context exceeded configured context size.")
        preds = predict_deepset_mc_streamed(
            model,
            X_train_np[idx],
            y_train_np[idx],
            X_test_np,
            K=K,
            test_batch_size=test_batch_size,
            device=device,
        )
        if preds.shape[0] != X_test_np.shape[0]:
            raise AssertionError("DeepSet prediction length did not match test split length.")
        context_preds.append(preds)

    return np.mean(np.stack(context_preds, axis=0), axis=0)


def compute_metrics(y_true, y_pred):
    """Returns dict with mse, rmse, r2."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse":  float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Section E - Prepared Benchmark
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
            num_cpus=BENCHMARK_NUM_CPUS,
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


def _download_stage_file_to_dir(stage_path, local_dir):
    """Download a single file from a Snowflake stage; returns local path."""
    os.makedirs(local_dir, exist_ok=True)
    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()
    session.file.get(stage_path, local_dir)
    fname = os.path.basename(stage_path.rstrip("/"))
    return os.path.join(local_dir, fname)


def load_prepared_benchmark_manifest(stage_path=None):
    """Download and parse benchmark_manifest.json from stage."""
    import json
    stage_path = stage_path or BENCHMARK_MANIFEST_STAGE_PATH
    local_dir = "/tmp/benchmark_manifest"
    local_path = _download_stage_file_to_dir(stage_path, local_dir)
    with open(local_path) as f:
        manifest = json.load(f)
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest):
    """Raise ValueError if manifest is missing required fields."""
    if "datasets" not in manifest or not manifest["datasets"]:
        raise ValueError("Benchmark manifest has no datasets.")
    seen = set()
    for ds in manifest["datasets"]:
        for field in ("dataset_index", "source", "task_id", "name",
                      "stage_path", "n_samples", "n_features", "categorical_indicator"):
            if field not in ds:
                raise ValueError(f"Manifest dataset missing field {field!r}: {ds}")
        dataset_index = int(ds["dataset_index"])
        if dataset_index in seen:
            raise ValueError(
                f"Benchmark manifest has duplicate dataset_index={dataset_index}. "
                "Re-run CALL prepare_benchmark_datasets(); to rebuild the manifest "
                "and BENCHMARK_DATASET_INDEX."
            )
        seen.add(dataset_index)


def _manifest_rows_by_index(datasets_meta):
    """Return manifest rows keyed by their stable dataset_index value."""
    rows_by_index = {}
    for ds in datasets_meta:
        dataset_index = int(ds["dataset_index"])
        if dataset_index in rows_by_index:
            raise ValueError(
                f"Benchmark manifest has duplicate dataset_index={dataset_index}. "
                "Re-run CALL prepare_benchmark_datasets(); to rebuild the manifest "
                "and BENCHMARK_DATASET_INDEX."
            )
        rows_by_index[dataset_index] = ds
    return rows_by_index


def _query_benchmark_index_rows(index_table=BENCHMARK_DATASET_INDEX_TABLE):
    """
    Read benchmark metadata rows from Snowflake.

    The index is used only for assignment metadata. Prepared .npz files remain
    opaque staged payloads and are downloaded exactly by stage_path later.
    """
    try:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
        rows = session.sql(
            f"""
            SELECT
                dataset_index,
                source,
                task_id,
                dataset_id,
                name,
                stage_path,
                n_samples,
                n_features,
                benchmark_weight
            FROM {index_table}
            ORDER BY dataset_index
            """
        ).collect()
        return [
            {
                "dataset_index": int(row[0]),
                "source": None if row[1] is None else str(row[1]),
                "task_id": None if row[2] is None else str(row[2]),
                "dataset_id": None if row[3] is None else str(row[3]),
                "name": None if row[4] is None else str(row[4]),
                "stage_path": None if row[5] is None else str(row[5]),
                "n_samples": None if row[6] is None else int(row[6]),
                "n_features": None if row[7] is None else int(row[7]),
                "benchmark_weight": None if row[8] is None else float(row[8]),
            }
            for row in rows
        ]
    except Exception as exc:
        raise RuntimeError(
            f"Balanced benchmark sharding requires a readable {index_table} table, "
            f"but it could not be queried: {exc}. "
            "Run CALL prepare_benchmark_datasets(); to refresh the manifest and "
            "BENCHMARK_DATASET_INDEX, or set BENCHMARK_SHARD_STRATEGY=modulo "
            "for modulo sharding."
        )


def _identity_value(row, field):
    value = row.get(field)
    if value is None:
        return None
    if field in ("n_samples", "n_features"):
        return int(value)
    return str(value)


def _validate_benchmark_index_rows(
    index_rows,
    datasets_meta,
    index_table=BENCHMARK_DATASET_INDEX_TABLE,
    allow_extra_index_rows=False,
):
    """Return index rows matching datasets_meta after strict freshness checks."""
    manifest_rows = _manifest_rows_by_index(datasets_meta)
    manifest_indices = set(manifest_rows)

    index_counts = Counter(int(row["dataset_index"]) for row in index_rows)
    duplicate_indices = sorted(i for i, count in index_counts.items() if count > 1)
    if duplicate_indices:
        raise RuntimeError(
            f"{index_table} has duplicate dataset_index values: {duplicate_indices[:10]}. "
            "Run CALL prepare_benchmark_datasets(); to rebuild the benchmark index."
        )

    index_rows_by_index = {int(row["dataset_index"]): row for row in index_rows}
    index_indices = set(index_rows_by_index)
    missing = sorted(manifest_indices - index_indices)
    extra = sorted(index_indices - manifest_indices)
    if missing:
        raise RuntimeError(
            f"{index_table} is missing manifest dataset_index values: {missing[:10]}. "
            "Run CALL prepare_benchmark_datasets(); to refresh the benchmark index, "
            "or set BENCHMARK_SHARD_STRATEGY=modulo for modulo sharding."
        )
    if extra and not allow_extra_index_rows:
        raise RuntimeError(
            f"{index_table} contains dataset_index values not present in the manifest: "
            f"{extra[:10]}. Run CALL prepare_benchmark_datasets(); to refresh the "
            "benchmark manifest and index together."
        )

    identity_fields = (
        "source", "task_id", "dataset_id", "name", "stage_path",
        "n_samples", "n_features",
    )
    selected_rows = []
    for dataset_index in sorted(manifest_indices):
        manifest_row = manifest_rows[dataset_index]
        index_row = index_rows_by_index[dataset_index]
        for field in identity_fields:
            if field == "dataset_id" and field not in manifest_row:
                continue
            if _identity_value(manifest_row, field) != _identity_value(index_row, field):
                raise RuntimeError(
                    f"{index_table} is stale for dataset_index={dataset_index}: "
                    f"{field} is {index_row.get(field)!r}, manifest has "
                    f"{manifest_row.get(field)!r}. Run CALL prepare_benchmark_datasets(); "
                    "to refresh the benchmark manifest and index together."
                )
        if index_row.get("benchmark_weight") is None:
            raise RuntimeError(
                f"{index_table} row dataset_index={dataset_index} has NULL "
                "benchmark_weight. Run CALL prepare_benchmark_datasets(); to backfill "
                "benchmark metadata."
            )
        selected_rows.append(index_row)

    return selected_rows


def _balanced_dataset_indices(index_rows, num_shards, shard_index):
    """
    Greedily assign datasets by descending benchmark_weight to balance shards.

    Ties are stable by dataset_index, so the result is deterministic.
    """
    shard_loads = [0.0] * num_shards
    shard_indices = [[] for _ in range(num_shards)]
    weighted_rows = []
    for row in index_rows:
        dataset_index = int(row["dataset_index"])
        weight = row.get("benchmark_weight")
        if weight is None:
            weight = row.get("n_samples", 0) * row.get("n_features", 0)
        weighted_rows.append((dataset_index, float(weight or 0.0)))

    for dataset_index, weight in sorted(weighted_rows, key=lambda item: (-item[1], item[0])):
        target_shard = min(range(num_shards), key=lambda i: (shard_loads[i], i))
        shard_indices[target_shard].append(dataset_index)
        shard_loads[target_shard] += weight

    return sorted(shard_indices[shard_index])


def select_benchmark_dataset_indices(
    datasets_meta,
    shard_index=0,
    num_shards=1,
    strategy=None,
    allow_extra_index_rows=False,
):
    """Return stable dataset_index values assigned to this shard."""
    strategy = (strategy or BENCHMARK_SHARD_STRATEGY or "modulo").lower()
    manifest_indices = [int(ds["dataset_index"]) for ds in datasets_meta]
    if num_shards < 1:
        raise ValueError(f"Invalid num_shards={num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"Invalid shard_index={shard_index} for num_shards={num_shards}"
        )

    if strategy in ("modulo", "dataset_modulo"):
        if num_shards <= 1:
            return manifest_indices
        return [i for i in manifest_indices if i % num_shards == shard_index]

    if strategy == "balanced":
        index_rows = _query_benchmark_index_rows()
        index_rows = _validate_benchmark_index_rows(
            index_rows,
            datasets_meta,
            allow_extra_index_rows=allow_extra_index_rows,
        )
        assigned = _balanced_dataset_indices(index_rows, num_shards, shard_index)
        return assigned

    raise ValueError(
        f"Unsupported BENCHMARK_SHARD_STRATEGY={strategy!r}. "
        "Expected 'modulo' or 'balanced'."
    )


def _default_benchmark_cache_dir():
    method = os.environ.get("BENCHMARK_METHOD") or os.environ.get("BENCHMARK_METHODS", "all")
    method = method.replace(os.sep, "_").replace(",", "_")
    shard = os.environ.get("BENCHMARK_SHARD_INDEX", "0")
    num_shards = os.environ.get("BENCHMARK_NUM_SHARDS", "1")
    return os.path.join(
        "/tmp",
        "benchmark_datasets",
        f"pid-{os.getpid()}",
        f"method-{method}",
        f"shard-{shard}-of-{num_shards}",
    )


def load_prepared_dataset(ds_meta, local_cache_dir=None):
    """Download and load a single prepared .npz dataset file from stage."""
    stage_path = ds_meta["stage_path"]
    local_cache_dir = local_cache_dir or _default_benchmark_cache_dir()
    fname = os.path.basename(stage_path.rstrip("/"))
    local_path = os.path.join(local_cache_dir, fname)
    if os.path.exists(local_path):
        os.remove(local_path)
    _download_stage_file_to_dir(stage_path, local_cache_dir)
    data = np.load(local_path, allow_pickle=False)
    return {
        "task_id":               str(ds_meta["task_id"]),
        "name":                  str(ds_meta["name"]),
        "X":                     data["X"].astype(np.float64),
        "y":                     data["y"].astype(np.float64),
        "categorical_indicator": data["categorical_indicator"].tolist(),
        "source":                str(ds_meta["source"]),
    }


def run_prepared_benchmark(
    model,
    manifest_stage_path=None,
    seeds=SEEDS,
    mc_K=N_MC_DROPOUT,
    results_dir="results/",
    methods=None,
    dataset_limit=None,
    rank=0,
    world_size=1,
    using_torch_distributed=False,
):
    """
    Run the regression benchmark using pre-staged dataset files from
    benchmark_manifest.json. Sharding is dataset-first: a shard owns dataset
    indices, downloads one owned dataset at a time, evaluates every seed, then
    releases the arrays before continuing.

    Returns (detailed_df, rank_summary_df, metric_summary_df).
    """
    manifest = load_prepared_benchmark_manifest(manifest_stage_path)
    datasets_meta = manifest["datasets"]
    manifest_dataset_count = len(datasets_meta)
    if dataset_limit:
        datasets_meta = datasets_meta[:int(dataset_limit)]

    if not datasets_meta:
        print("[WARNING] Benchmark manifest has no datasets. Skipping benchmark.")
        return None, None, None

    rows_by_index = _manifest_rows_by_index(datasets_meta)
    assigned_indices = select_benchmark_dataset_indices(
        datasets_meta,
        shard_index=rank,
        num_shards=world_size,
        allow_extra_index_rows=dataset_limit is not None,
    )
    missing_assigned = [i for i in assigned_indices if i not in rows_by_index]
    if missing_assigned:
        raise RuntimeError(
            f"Shard assignment produced dataset_index values not present in the "
            f"manifest subset: {missing_assigned[:10]}"
        )
    assigned_datasets = [rows_by_index[i] for i in assigned_indices]
    assigned_weight = sum(
        int(ds.get("benchmark_weight") or int(ds["n_samples"]) * int(ds["n_features"]))
        for ds in assigned_datasets
    )
    total_work_units = len(datasets_meta) * len(seeds)
    assigned_work_units = len(assigned_datasets) * len(seeds)

    print(
        f"Benchmark manifest: {len(datasets_meta)} datasets\n"
        f"Benchmark manifest total before limit: {manifest_dataset_count} datasets\n"
        f"Shard: rank={rank} world_size={world_size}\n"
        f"Shard strategy: {BENCHMARK_SHARD_STRATEGY}\n"
        f"Assigned dataset indices: {assigned_indices}\n"
        f"Assigned benchmark weight: {assigned_weight}\n"
        f"Assigned datasets: {len(assigned_datasets)}/{len(datasets_meta)}\n"
        f"Assigned work units: {assigned_work_units}/{total_work_units} "
        f"(each dataset x {len(seeds)} seeds)",
        flush=True,
    )

    selected_methods = methods or ALL_BENCHMARK_METHODS
    unknown_methods = sorted(set(selected_methods) - set(ALL_BENCHMARK_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown benchmark method(s): {unknown_methods}")
    if "DeepSetModel-MC" in selected_methods and model is None:
        raise ValueError("DeepSetModel-MC benchmark requires a loaded model.")
    deepset_device = (
        deepset_inference_device()
        if "DeepSetModel-MC" in selected_methods
        else None
    )

    all_rows = []
    for ds_meta in assigned_datasets:
        i_ds = int(ds_meta["dataset_index"])
        print(f"\n--- Benchmark dataset {i_ds}: {ds_meta['name']} ---", flush=True)
        ds = load_prepared_dataset(ds_meta)
        try:
            X, y    = ds["X"], ds["y"]
            cat_ind = ds["categorical_indicator"]
            name    = ds["name"]
            task_id = ds["task_id"]
            source  = ds.get("source", "openml")

            for seed in seeds:
                rng = np.random.default_rng(seed)
                print(f"  seed={seed}", flush=True)
                X_seed, y_seed = X, y

                # Subsample
                if X_seed.shape[0] > MAX_SAMPLES:
                    idx = rng.choice(X_seed.shape[0], MAX_SAMPLES, replace=False)
                    X_seed, y_seed = X_seed[idx], y_seed[idx]

                # 90/10 split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_seed, y_seed, test_size=1.0 - TRAIN_FRAC, random_state=seed
                )

                # Preprocess (same data for all methods)
                try:
                    X_train_p, X_test_p = preprocess_split(X_train, X_test, cat_ind)
                except Exception as exc:
                    print(f"  [SKIP preprocess] {name}: {exc}")
                    continue

                nan_row = {"task_id": task_id, "name": name, "rep": seed, "source": source,
                           "mse": float("nan"), "rmse": float("nan"), "r2": float("nan")}

                if "DeepSetModel-MC" in selected_methods:
                    try:
                        preds = predict_deepset_bounded_context_ensemble(
                            model,
                            X_train_p,
                            y_train,
                            X_test_p,
                            seed=seed,
                            K=mc_K,
                            context_size=BENCHMARK_DEEPSET_CONTEXT_SIZE,
                            context_ensembles=BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES,
                            test_batch_size=BENCHMARK_DEEPSET_TEST_BATCH_SIZE,
                            device=deepset_device,
                        )
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
        finally:
            del ds
            gc.collect()

    # Gather results from all distributed workers.
    if using_torch_distributed and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, all_rows)
        if rank != 0:
            return None, None, None
        all_rows = [row for chunk in gathered for row in chunk]

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

    shard_index = os.environ.get("BENCHMARK_SHARD_INDEX")
    num_shards = os.environ.get("BENCHMARK_NUM_SHARDS")

    if shard_index is not None and num_shards is not None and int(num_shards) > 1:
        filename = (
            f"{safe_method_name(method)}_"
            f"shard{int(shard_index)}_of_{int(num_shards)}_detailed.csv"
        )
    else:
        filename = f"{safe_method_name(method)}_detailed.csv"

    part_path = os.path.join(part_dir, filename)
    normalize_benchmark_columns(detailed_df).to_csv(part_path, index=False)
    print(f"Saved {part_path} ({len(detailed_df)} rows)", flush=True)
    return part_path


def save_benchmark_part_csvs(detailed_df, methods, results_dir):
    """Save one benchmark part CSV per method present in detailed_df."""
    detailed_df = normalize_benchmark_columns(detailed_df)
    part_paths = []
    for method in methods:
        method_df = detailed_df[detailed_df["method"] == method]
        if method_df.empty:
            print(f"[WARNING] No benchmark rows for {method}; no part CSV written.", flush=True)
            continue
        part_paths.append(save_benchmark_part_csv(method_df, method, results_dir))
    return part_paths


def download_stage_prefix(stage_path, local_dir):
    """Download a stage prefix into local_dir; no-op outside Snowflake."""
    os.makedirs(local_dir, exist_ok=True)
    try:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
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

    print(f"Found {len(part_files)} benchmark part files in {local_parts}:", flush=True)
    for pf in part_files:
        print(f"  {pf}", flush=True)

    frames = [pd.read_csv(path) for path in part_files]
    detailed_df = normalize_benchmark_columns(pd.concat(frames, ignore_index=True))

    methods_found = sorted(detailed_df["method"].unique().tolist())
    print(f"Methods discovered: {methods_found}", flush=True)
    print(f"Total rows: {len(detailed_df)}", flush=True)
    rank_summary_df, metric_summary_df = aggregate_benchmark_results(detailed_df)
    return save_benchmark_csvs(detailed_df, rank_summary_df, metric_summary_df, results_dir)


def resolve_parallel_context():
    """
    Resolve rank/world_size/using_torch_distributed for evaluation.

    Priority:
    1. Explicit shard mode via BENCHMARK_NUM_SHARDS + BENCHMARK_SHARD_INDEX.
    2. Real PyTorch distributed (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT all set).
    3. Single-process fallback.

    Returns (rank, world_size, using_torch_distributed).
    """
    num_shards = int(os.environ.get("BENCHMARK_NUM_SHARDS", "1"))
    shard_index = int(os.environ.get("BENCHMARK_SHARD_INDEX", "0"))

    if num_shards > 1:
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(
                f"Invalid BENCHMARK_SHARD_INDEX={shard_index} "
                f"for BENCHMARK_NUM_SHARDS={num_shards}"
            )
        print(
            f"Using independent benchmark shard mode: "
            f"shard {shard_index + 1}/{num_shards}",
            flush=True,
        )
        return shard_index, num_shards, False

    req_nodes = int(os.environ.get("EVAL_NUM_NODES", "1"))
    req_workers = int(os.environ.get("EVAL_WORKERS_PER_NODE", "1"))
    req_world_size = req_nodes * req_workers

    dist_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    has_dist_env = all(os.environ.get(v) for v in dist_vars)

    if req_world_size > 1:
        if has_dist_env:
            dist.init_process_group(backend="gloo")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(
                f"Using PyTorch distributed evaluation: "
                f"rank={rank}, world_size={world_size}, "
                f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}",
                flush=True,
            )
            return rank, world_size, True
        raise RuntimeError(
            f"Distributed evaluation requested (EVAL_NUM_NODES={req_nodes}, "
            f"EVAL_WORKERS_PER_NODE={req_workers}, world_size={req_world_size}) "
            "but PyTorch distributed env vars are missing "
            f"({', '.join(v for v in dist_vars if not os.environ.get(v))}). "
            "submit_from_stage(target_instances=N) alone does not create a PyTorch "
            "process group. For CPU baselines and AutoGluon, use "
            "BENCHMARK_NUM_SHARDS + BENCHMARK_SHARD_INDEX instead."
        )

    print("Using single-process evaluation mode.", flush=True)
    return 0, 1, False


# ---------------------------------------------------------------------------
# Section F — Upload results to Snowflake
# ---------------------------------------------------------------------------

def upload_to_snowflake(local_path: str, stage_path: str):
    """
    Upload a local file to a Snowflake internal stage via Snowpark.
    Uses Session.builder.getOrCreate() so uploads work from both the main process
    and distributed worker processes (rank 0 only). Degrades gracefully locally.
    """
    try:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
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
    parser.add_argument("--mc_K",        type=int, default=int(os.environ.get("MC_K", N_MC_DROPOUT)),
                        help="Number of MC dropout forward passes (default: 32).")
    parser.add_argument("--mode",        default=os.environ.get("EVAL_MODE", "full"),
                        choices=["full", "synthetic", "benchmark", "aggregate"],
                        help="Evaluation mode.")
    parser.add_argument("--benchmark_method",
                        default=os.environ.get("BENCHMARK_METHOD"),
                        help="Run only one benchmark method.")
    parser.add_argument("--benchmark_methods",
                        default=os.environ.get("BENCHMARK_METHODS"),
                        help="Comma-separated benchmark methods to run in one shard.")
    parser.add_argument("--eval_results_stage",
                        default=os.environ.get("EVAL_RESULTS_STAGE", EVAL_RESULTS_STAGE),
                        help="Snowflake stage for evaluation CSV outputs.")
    parser.add_argument("--benchmark_dataset_limit", type=int,
                        default=os.environ.get("BENCHMARK_DATASET_LIMIT"),
                        help="Limit benchmark datasets for one-dataset smoke runs.")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    rank, world_size, using_torch_distributed = resolve_parallel_context()

    try:
        if args.mode == "aggregate":
            det_path, sum_path = aggregate_benchmark_parts(args.results_dir, args.eval_results_stage)
            upload_to_snowflake(det_path, args.eval_results_stage)
            upload_to_snowflake(sum_path, args.eval_results_stage)
            return

        selected_methods = parse_benchmark_methods(args.benchmark_method, args.benchmark_methods)
        explicit_method_selection = bool(args.benchmark_method or args.benchmark_methods)

        needs_model = (
            args.mode in ("full", "synthetic")
            or "DeepSetModel-MC" in selected_methods
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
                records = evaluate_synthetic_dgp(model, test_dir, rank=rank, world_size=world_size,
                                                 using_torch_distributed=using_torch_distributed)
                if records is not None:
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
                mc_records = evaluate_synthetic_dgp_mc(model, test_dir, K=args.mc_K,
                                                       rank=rank, world_size=world_size,
                                                       using_torch_distributed=using_torch_distributed)
                if mc_records is not None:
                    mc_rows = build_mc_report(mc_records)
                    print("\nMC Dropout Noise Assessment (ratio_mc_ols <= ratio_model_ols = noise reduction):")
                    print_mc_report(mc_rows)

                    synthetic_dir = os.path.join(args.results_dir, "synthetic")
                    os.makedirs(synthetic_dir, exist_ok=True)
                    mc_csv_path = os.path.join(synthetic_dir, "mc_report.csv")
                    save_mc_report_csv(mc_rows, mc_csv_path)
                    print(f"Saved {mc_csv_path}")
                    upload_to_snowflake(mc_csv_path, f"{args.eval_results_stage}/synthetic/")

            except FileNotFoundError as exc:
                print(f"[WARNING] {exc} - skipping synthetic DGP evaluation.")

        if args.mode == "synthetic":
            return

        if not BENCHMARK_DEPS_AVAILABLE:
            raise RuntimeError(
                "Benchmark dependencies are not available. Install: "
                "scikit-learn xgboost lightgbm catboost pandas scipy"
            )

        detailed_df, rank_summary_df, metric_summary_df = run_prepared_benchmark(
            model,
            manifest_stage_path=BENCHMARK_MANIFEST_STAGE_PATH,
            seeds=SEEDS,
            mc_K=args.mc_K,
            results_dir=args.results_dir,
            methods=selected_methods,
            dataset_limit=args.benchmark_dataset_limit,
            rank=rank, world_size=world_size,
            using_torch_distributed=using_torch_distributed,
        )

        if detailed_df is not None:
            print_benchmark_table(rank_summary_df, metric_summary_df)
            if explicit_method_selection:
                for part_path in save_benchmark_part_csvs(
                    detailed_df, selected_methods, args.results_dir
                ):
                    upload_to_snowflake(part_path, f"{args.eval_results_stage}/benchmark_parts/")
            else:
                det_path, sum_path = save_benchmark_csvs(
                    detailed_df, rank_summary_df, metric_summary_df, args.results_dir
                )
                upload_to_snowflake(det_path, args.eval_results_stage)
                upload_to_snowflake(sum_path, args.eval_results_stage)

    except Exception as exc:
        import traceback
        import json as _json_eval
        shard_index = os.environ.get("BENCHMARK_SHARD_INDEX", "")
        num_shards = os.environ.get("BENCHMARK_NUM_SHARDS", "")
        method = os.environ.get("BENCHMARK_METHOD") or os.environ.get("BENCHMARK_METHODS", "unknown")
        mode = os.environ.get("EVAL_MODE", "unknown")
        suffix = (f"_shard{shard_index}_of_{num_shards}"
                  if shard_index and num_shards and int(num_shards) > 1 else "")
        failure_payload = {
            "method": method, "mode": mode,
            "shard_index": shard_index, "num_shards": num_shards,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_local = f"/tmp/{method}{suffix}_failure.json"
        with open(failure_local, "w") as _f:
            _json_eval.dump(failure_payload, _f, indent=2)
        print(f"[EVAL FAILURE JSON]\n{_json_eval.dumps(failure_payload, indent=2)}", flush=True)
        upload_to_snowflake(
            failure_local,
            f"{args.eval_results_stage}/failures/",
        )
        raise
    finally:
        if using_torch_distributed and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
