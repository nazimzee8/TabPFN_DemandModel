"""
evaluate.py

Evaluate a saved DeepSetModel checkpoint against held-out test Parquet files
and run permutation-invariance unit tests.

Usage (inside container or locally):
    python evaluate.py --model_path best.pt --data_dir /data --results_dir results/
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch

from model import DeepSetModel, ModelConfig, POOL_SCALE


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
        return model.sab_feat(h)                    # SAB: (batch=n, set=p, d)
    mean_i = h.mean(dim=1, keepdim=True)
    return model.lambda_feat * h + model.gamma_feat * mean_i


def apply_samp_equiv(model, r):
    """r: (n, d) → (n, d). Dispatches to SAB or linear equivariance."""
    if model.cfg.n_sab_samp > 0:
        return model.sab_samp(r.unsqueeze(0)).squeeze(0)   # (1,n,d) → (n,d)
    mean_j = r.mean(dim=0, keepdim=True)
    return model.lambda_samp * r + model.gamma_samp * mean_j


# ---------------------------------------------------------------------------
# Section A — Load model
# ---------------------------------------------------------------------------

def load_model(model_path):
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
# Section B — Evaluate on held-out test split
# ---------------------------------------------------------------------------

def evaluate_test_split(model, test_dir):
    """
    Predict and compute OLS baseline for every test Parquet file.
    Returns list of per-dataset dicts with model_mse, ols_mse, prior_regime, n, p.
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

            # Model predictions (one test row at a time, same as train.py)
            preds = []
            for k in range(X_test.shape[0]):
                y_hat = model(X_train, y_train, X_test[k])
                preds.append(y_hat.item())
            preds_np   = np.array(preds)
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
                model.sab_samp(rb).squeeze(0)[pi], atol=1e-4)  # 1e-4: float32 reorder

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
# Section E — Upload results to Snowflake (mirrors train.py)
# ---------------------------------------------------------------------------

def upload_to_snowflake(local_path: str, stage_path: str):
    try:
        from snowflake.snowpark import Session
        connection_params = {
            "host":          os.environ["SNOWFLAKE_HOST"],
            "account":       os.environ.get("SNOWFLAKE_ACCOUNT_OVERRIDE", ""),
            "authenticator": "oauth",
            "token":         open("/snowflake/session/token").read(),
            "warehouse":     os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
            "database":      "TABPFN_DB",
            "schema":        "TABPFN_SCHEMA",
        }
        session = Session.builder.configs(connection_params).create()
        session.file.put(local_path, stage_path, overwrite=True)
        print(f"Uploaded {local_path} to {stage_path}")
    except Exception as exc:
        print(f"[WARNING] Snowpark upload failed (skipping): {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSetModel checkpoint.")
    parser.add_argument("--model_path",  default="best.pt",  help="Path to model checkpoint.")
    parser.add_argument("--data_dir",    default="/data",    help="Root data directory (contains test/ subdir).")
    parser.add_argument("--results_dir", default="results/", help="Directory for output files.")
    args = parser.parse_args()

    # A. Load model
    model = load_model(args.model_path)

    # D. Permutation invariance tests (synthetic data, no file I/O)
    all_pass = run_permutation_tests(model)
    if not all_pass:
        print("[WARNING] One or more permutation tests FAILED.")

    # B. Evaluate on test split
    test_dir = os.path.join(args.data_dir, "test")
    print(f"\nEvaluating on test files in {test_dir} ...")
    records = evaluate_test_split(model, test_dir)
    print(f"Evaluated {len(records)} datasets.")

    # C. Stratified report
    report_rows = build_report(records)
    print("\nTest Split Results:")
    print_report(report_rows)

    # Save CSV
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "test_report.csv")
    save_report_csv(report_rows, csv_path)
    print(f"\nSaved report to {csv_path}")

    # E. Upload to Snowflake (no-op if not in SPCS)
    upload_to_snowflake(csv_path, "@MODEL_STAGE/results/")


if __name__ == "__main__":
    main()
