"""Smoke tests: Cursor-scale Parquet write → read → loader → index → stage-path normalization.

Uses a small 30-task suite written to a temp directory to validate the full local
data-contract pipeline without Snowflake.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_SRC  = _REPO / "src"
for _sub in [
    _SRC / "data_generation",
    _SRC / "snowflake_io",
    _SRC / "dataset_index",
    _SRC,
]:
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))

import _bootstrap  # noqa: F401

from dgp_helpers import (
    allocate_regression_tasks_cursor,
    build_dataset_from_regime,
    compute_linear_diagnostics,
    compute_linear_teacher_targets,
    validate_dataset,
    write_parquet_dgp,
)

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from snowflake_io import _normalize_stage_path
    HAS_SNOWFLAKE_IO = True
except ImportError:
    HAS_SNOWFLAKE_IO = False


# ---------------------------------------------------------------------------
# Fixture: 30-task cursor suite written to tmp dir
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cursor_suite(tmp_path_factory):
    """Generate a 30-task cursor suite in a temp dir and return (numeric_dir, assignments)."""
    tmp = tmp_path_factory.mktemp("cursor_smoke")
    numeric_dir = tmp / "numeric"
    for split in ("train", "val", "test"):
        (numeric_dir / split).mkdir(parents=True)

    split_sizes = {"train": 24, "val": 3, "test": 3}
    assignments, audit = allocate_regression_tasks_cursor(
        split_sizes, "linear_stat_aware", base_seed=123,
        allocation_mode="weighted_quota", min_regime_count=1,
    )

    counts = list(split_sizes.values())
    n_train, n_val = counts[0], counts[1]

    for idx, a in enumerate(assignments):
        rng = np.random.default_rng(a["task_seed"])
        params = {k: a[k] for k in [
            "n", "p_signal", "p_noise", "p_total", "active_s",
            "rho", "target_noise_scale", "feature_noise_level",
            "covariance_type", "cursor_dims",
        ] if k in a}
        ds = build_dataset_from_regime(rng, a["prior_regime"], "linear_stat_aware", params)
        validate_dataset(ds)

        n_total = ds["n_total"]
        n_tr = int(0.8 * n_total)
        n_te = n_total - n_tr

        ds_split = {
            "X_train": ds["X"][:n_tr],
            "y_train": ds["y"][:n_tr],
            "X_test":  ds["X"][n_tr:],
            "y_test":  ds["y"][n_tr:],
            "betaX_test":  ds["betaX"][n_tr:],
            "betaX_train": ds["betaX"][:n_tr],
            "n": n_total, "p": ds["p_total"],
            "n_train": n_tr, "n_test": n_te,
            "prior_regime": ds["prior_regime"],
            "p_signal": ds["p_signal"], "p_noise": ds["p_noise"],
            "p_total": ds["p_total"], "active_s": ds["active_s"],
            "sparsity_ratio": ds["sparsity_ratio"],
            "covariance_type": ds["covariance_type"], "rho": ds["rho"],
            "target_noise_scale": ds["target_noise_scale"],
            "feature_noise_level": ds["feature_noise_level"],
            "beta": ds["beta"], "support_mask": ds["support_mask"],
        }
        teacher = compute_linear_teacher_targets(
            ds_split["X_train"], ds_split["y_train"],
            ds_split["X_test"],  ds_split["y_test"],
        )
        diag = compute_linear_diagnostics(ds["X"], ds["beta"], ds["p_signal"])

        if idx < n_train:
            split_dir, fname = numeric_dir / "train", f"dataset_{idx:04d}.parquet"
        elif idx < n_train + n_val:
            split_dir = numeric_dir / "val"
            fname = f"dataset_{idx - n_train:04d}.parquet"
        else:
            split_dir = numeric_dir / "test"
            fname = f"dataset_{idx - n_train - n_val:04d}.parquet"

        write_parquet_dgp(
            ds_split, str(split_dir / fname),
            store_beta=True, store_teacher_preds=True,
            teacher_results=teacher, diagnostics=diag,
        )

    return numeric_dir, assignments, audit


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PARQUET, reason="pyarrow not available")
class TestParquetContract:
    def test_parquet_files_exist(self, cursor_suite):
        numeric_dir, assignments, _ = cursor_suite
        files = list(numeric_dir.rglob("*.parquet"))
        assert len(files) == len(assignments), (
            f"Expected {len(assignments)} parquet files, found {len(files)}"
        )

    def test_parquet_schema_has_required_columns(self, cursor_suite):
        numeric_dir, _, _ = cursor_suite
        required = {"X_train", "y_train", "X_test", "betaX_test",
                    "n", "p", "n_train", "n_test", "prior_regime"}
        for f in numeric_dir.rglob("*.parquet"):
            schema = pq.read_schema(str(f))
            col_names = set(schema.names)
            missing = required - col_names
            assert not missing, f"{f.name}: missing columns {missing}"
            break  # check one representative file

    def test_parquet_roundtrip_dimensions(self, cursor_suite):
        """Read back parquet and verify n/p consistency with inner split."""
        numeric_dir, assignments, _ = cursor_suite
        for f in sorted((numeric_dir / "train").glob("*.parquet"))[:5]:
            tbl = pq.read_table(str(f))
            row = {col: tbl[col][0].as_py() for col in tbl.schema.names}
            n = row["n"]
            n_train = row["n_train"]
            n_test = row["n_test"]
            assert n_train + n_test == n, (
                f"{f.name}: n_train+n_test={n_train+n_test} != n={n}"
            )
            assert n_train == math.floor(0.8 * n), (
                f"{f.name}: n_train={n_train} != floor(0.8*{n})={math.floor(0.8*n)}"
            )
            p = row["p"]
            X_train_shape = (n_train, p)
            actual_X_train = tbl["X_train"][0].as_py()
            assert len(actual_X_train) == n_train, (
                f"X_train rows: expected {n_train}, got {len(actual_X_train)}"
            )
            assert len(actual_X_train[0]) == p, (
                f"X_train cols: expected {p}, got {len(actual_X_train[0])}"
            )

    def test_n_ge_5p_in_parquet(self, cursor_suite):
        """Every written task must satisfy n >= 5*p when read back from parquet."""
        numeric_dir, _, _ = cursor_suite
        for f in numeric_dir.rglob("*.parquet"):
            tbl = pq.read_table(str(f), columns=["n", "p"])
            n = tbl["n"][0].as_py()
            p = tbl["p"][0].as_py()
            assert n >= 5 * p, f"{f.name}: n={n} < 5*p={5*p}"


@pytest.mark.skipif(not HAS_SNOWFLAKE_IO, reason="snowflake_io not importable")
class TestStagePathNormalization:
    def test_normalize_numeric_path(self):
        path = "numeric/train/dataset_0001.parquet"
        result = _normalize_stage_path(path)
        assert result == "numeric/train/dataset_0001.parquet"

    def test_normalize_with_stage_prefix(self):
        path = "@META_DATASET_STAGE/linear/regression/numeric/val/dataset_0002.parquet"
        result = _normalize_stage_path(path)
        assert result == "linear/regression/numeric/val/dataset_0002.parquet"

    def test_normalize_split_injection(self):
        path = "dataset_0005.parquet"
        result = _normalize_stage_path(path, split="test")
        assert "test" in result
        assert "dataset_0005.parquet" in result


class TestManifestContent:
    def test_audit_all_tasks_satisfy_n_ge_5p(self, cursor_suite):
        _, _, audit = cursor_suite
        assert audit["all_tasks_satisfy_n_ge_5p"] is True

    def test_audit_has_dimension_policy(self, cursor_suite):
        _, _, audit = cursor_suite
        assert "dimension_policy" in audit
        dp = audit["dimension_policy"]
        assert dp["dimension_policy_name"] == "cursor_scale_v1"
        assert "E_high_dim_dense" in dp["regime_specific_dimension_conditions"]
        assert "J_low_n_high_p" in dp["regime_specific_dimension_conditions"]

    def test_audit_by_split_present(self, cursor_suite):
        _, _, audit = cursor_suite
        assert "audit_by_split" in audit
        for split_name in ("train", "val", "test"):
            assert split_name in audit["audit_by_split"], (
                f"audit_by_split missing split {split_name!r}"
            )
