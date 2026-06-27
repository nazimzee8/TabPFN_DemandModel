"""
download_kaggle_benchmark.py

Downloads the five Kaggle Tabular Playground Series Season 3 competitions
used in the TabPFN v2 Nature 2025 benchmark and converts each to a .npz file.

Usage:
    python download_kaggle_benchmark.py [--out_dir ./kaggle_datasets] [--max_samples 10000]
    python download_kaggle_benchmark.py --local_creds   # use ~/.kaggle/kaggle.json instead

Output: one .npz per competition in out_dir/

.npz schema (identical for all files):
    X                    : (n, p) float64  — feature matrix (NaN-free)
    y                    : (n,) float64    — target (regression: raw; classification: label-encoded int)
    categorical_indicator: (p,) bool       — True = originally categorical column
    feature_names        : (p,) object     — column names from train.csv
    dataset_name         : (1,) object     — human-readable name
    slug                 : (1,) object     — Kaggle competition slug
    source               : (1,) object     — always "kaggle"
    task_type            : (1,) object     — "regression" or "classification"

MODEL3-ICL fitness notes (for regression benchmark):
    s3e5 Wine Quality     — 11 numerical features, ordinal target 3–9, ~130k rows → subsampled.
                            Smooth chemistry features; ordinal-as-continuous is a standard approximation.
    s3e9 Concrete Strength — 8 numerical features, ~1030 rows, continuous target.
                            Small dataset is MODEL3-ICL's strength; non-linear cement/water/age
                            interactions may challenge a linear-prior-trained model.
    s3e3/s3e22/s3e26      — Classification: staged for future benchmark; excluded from regression run.
"""
import argparse
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

MAX_SAMPLES = 10_000

# ---------------------------------------------------------------------------
# Competition manifest
# ---------------------------------------------------------------------------

COMPETITIONS = [
    {
        "slug":      "playground-series-s3e3",
        "name":      "Employee Attrition",
        "target":    "Attrition",
        "drop":      ["id"],
        "task_type": "classification",
    },
    {
        "slug":      "playground-series-s3e5",
        "name":      "Wine Quality",
        "target":    "quality",
        "drop":      ["id"],
        "task_type": "regression",    # ordinal target treated as continuous
    },
    {
        "slug":      "playground-series-s3e9",
        "name":      "Concrete Strength",
        "target":    "Strength",
        "drop":      ["id"],
        "task_type": "regression",
    },
    {
        "slug":      "playground-series-s3e22",
        "name":      "Horse Health Outcomes",
        "target":    "outcome",
        "drop":      ["id"],
        "task_type": "classification",
    },
    {
        "slug":      "playground-series-s3e26",
        "name":      "Cirrhosis Outcomes",
        "target":    "Status",
        "drop":      ["id"],
        "task_type": "classification",
    },
]


def convert_competition(api, comp, out_dir, max_samples=MAX_SAMPLES, seed=42):
    slug   = comp["slug"]
    target = comp["target"]
    print(f"\n--- {slug} ({comp['name']}) ---")

    # 1. Download
    with tempfile.TemporaryDirectory() as tmp:
        api.competition_download_files(slug, path=tmp, quiet=False)
        # Unzip — competition downloads as a single .zip
        zip_path = os.path.join(tmp, f"{slug}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp)
        # Find train.csv (may be nested inside a subfolder)
        train_csv = os.path.join(tmp, "train.csv")
        if not os.path.exists(train_csv):
            for root, _, files in os.walk(tmp):
                if "train.csv" in files:
                    train_csv = os.path.join(root, "train.csv")
                    break
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"train.csv not found for {slug}")

        df = pd.read_csv(train_csv)

    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")

    # 2. Drop non-feature columns (id, etc.)
    drop_cols = [c for c in comp.get("drop", []) if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Verify target exists
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available: {list(df.columns)}"
        )

    # 3. Separate X / y
    y_series      = df[target]
    X_df          = df.drop(columns=[target])
    feature_names = np.array(X_df.columns.tolist())

    # 4. Infer categorical columns: object/category dtype → categorical
    cat_mask = np.array([
        X_df[c].dtype.kind in ("O", "U", "S") or str(X_df[c].dtype) == "category"
        for c in X_df.columns
    ], dtype=bool)

    # 5. Encode target
    if comp["task_type"] == "regression":
        y = y_series.values.astype(np.float64)
    else:
        # Label-encode classification target to integers
        y       = pd.Categorical(y_series).codes.astype(np.float64)
        classes = pd.Categorical(y_series).categories.tolist()
        print(f"  Classification target classes: {classes}")

    # 6. Ordinal-encode categorical features (global codes on full dataset)
    #    Per-split one-hot/ordinal encoding is still done by evaluate.py's preprocess_split().
    X_df = X_df.copy()
    for col in X_df.columns[cat_mask]:
        X_df[col] = pd.Categorical(X_df[col]).codes.astype(float)
        X_df[col] = X_df[col].replace(-1, np.nan)   # -1 = NaN from codes

    X = X_df.values.astype(np.float64)

    # 7. Mean-impute remaining NaN; drop rows with NaN target
    nan_target = np.isnan(y)
    if nan_target.any():
        print(f"  Dropping {nan_target.sum()} rows with NaN target")
        X, y = X[~nan_target], y[~nan_target]

    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)   # fallback for all-NaN cols
    nan_idx   = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_means, nan_idx[1])

    # 8. Subsample to max_samples if needed
    n_orig = X.shape[0]
    if n_orig > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_orig, max_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"  Subsampled {n_orig} → {max_samples} rows")

    print(f"  Final: n={X.shape[0]}, p={X.shape[1]}, task={comp['task_type']}")

    # 9. Save .npz
    out_path = os.path.join(out_dir, f"{slug}.npz")
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        categorical_indicator=cat_mask,
        feature_names=feature_names,
        dataset_name=np.array([comp["name"]]),
        slug=np.array([slug]),
        source=np.array(["kaggle"]),
        task_type=np.array([comp["task_type"]]),
    )
    print(f"  Saved → {out_path}")
    return out_path


def get_kaggle_credentials_from_snowflake(
    secret_name="tabpfn_db.tabpfn_schema.kaggle_api_token",
):
    """
    Retrieve Kaggle credentials from a Snowflake Generic Secret and inject them as
    KAGGLE_USERNAME / KAGGLE_KEY environment variables. The Kaggle Python library
    reads these env vars automatically, so no ~/.kaggle/kaggle.json is needed.

    Authenticates via SSO/MFA browser popup (no plaintext password in code).
    """
    import json
    import snowflake.connector

    conn = snowflake.connector.connect(
        account="YOUR_ACCOUNT",           # e.g. "abc12345.us-east-1"
        user="nazerrouki@gmail.com",
        authenticator="externalbrowser",  # SSO / MFA login — no password in code
        database="TABPFN_DB",
        schema="TABPFN_SCHEMA",
    )
    cur = conn.cursor()
    cur.execute(f"SELECT SYSTEM$GET_SECRET('{secret_name}')")
    secret_json = json.loads(cur.fetchone()[0])
    conn.close()

    os.environ["KAGGLE_USERNAME"] = secret_json["username"]
    os.environ["KAGGLE_KEY"]      = secret_json["key"]
    print(f"  Kaggle credentials loaded from Snowflake secret '{secret_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Kaggle TPS-S3 benchmark competitions to .npz."
    )
    parser.add_argument("--out_dir",     default="./kaggle_datasets",
                        help="Output directory for .npz files.")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES,
                        help="Maximum rows per dataset (default: 10000).")
    parser.add_argument("--slugs",       nargs="*", default=None,
                        help="Subset of slugs to download (default: all 5). "
                             "Example: --slugs playground-series-s3e9")
    parser.add_argument("--local_creds", action="store_true",
                        help="Use ~/.kaggle/kaggle.json instead of Snowflake secret "
                             "(fallback for offline dev).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.local_creds:
        get_kaggle_credentials_from_snowflake()
    # else: Kaggle API falls back to ~/.kaggle/kaggle.json automatically

    api = KaggleApi()
    api.authenticate()

    competitions = COMPETITIONS
    if args.slugs:
        competitions = [c for c in COMPETITIONS if c["slug"] in args.slugs]

    for comp in competitions:
        try:
            convert_competition(api, comp, args.out_dir, max_samples=args.max_samples)
        except Exception as exc:
            print(f"  [FAIL] {comp['slug']}: {exc}")

    print(f"\nDone. Files in {args.out_dir}:")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith(".npz"):
            size = os.path.getsize(os.path.join(args.out_dir, f)) // 1024
            print(f"  {f}  ({size} KB)")


if __name__ == "__main__":
    main()
