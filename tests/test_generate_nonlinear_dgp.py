from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
SRC_DATA = os.path.join(SRC, "data_generation")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# Ensure the canonical data_generation subdir is ahead of src/ so that
# bare `import generate_nonlinear_dgp` resolves to the co-located copy.
if SRC_DATA not in sys.path:
    sys.path.insert(0, SRC_DATA)

import generate_nonlinear_dgp as gen  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: feature_noise_level is preserved as float in generate_v2_dataset
# ---------------------------------------------------------------------------

def test_generate_v2_dataset_preserves_float_feature_noise():
    ds = gen.generate_v2_dataset(
        target_family="poly_quad",
        feature_regime="feat_noise",
        n=100,
        p_signal=8,
        target_noise_scale=1.0,
        target_noise_type="gaussian",
        suite_component="core",
        teacher_seed=123,
        sample_seed=456,
        feature_noise_sigma=0.10,
    )

    assert ds["prior_regime"] == "poly_quad"
    assert ds["feature_regime"] == "feat_noise"
    assert isinstance(ds["feature_noise_level"], float)
    assert ds["feature_noise_level"] == pytest_approx_or_close(0.10, ds["feature_noise_level"])
    assert ds["feature_noise_sigma"] > 0.0
    assert ds["n_total"] == 100
    assert ds["p_signal"] == 8


def pytest_approx_or_close(expected, actual, tol=1e-9):
    """Inline tolerance check that returns actual (truthy) if close."""
    return actual if abs(actual - expected) <= tol else expected


# ---------------------------------------------------------------------------
# Test 2: noise_feats regime expands p_total
# ---------------------------------------------------------------------------

def test_noise_feats_regime_expands_p_total():
    ds = gen.generate_v2_dataset(
        target_family="hinge",
        feature_regime="noise_feats",
        n=300,
        p_signal=8,
        target_noise_scale=1.0,
        target_noise_type="gaussian",
        suite_component="core",
        teacher_seed=789,
        sample_seed=101,
    )

    assert ds["feature_regime"] == "noise_feats"
    assert ds["p_noise"] > 0, "noise_feats regime should add noise features"
    assert ds["p_total"] > ds["p_signal"]
    assert ds["X"].shape[1] == ds["p_total"]
    assert ds["p_noise"] >= 2 * ds["p_signal"], "p_noise should be >= 2x p_signal"


# ---------------------------------------------------------------------------
# Test 3: enumerate_suite_cells covers all 42 (family, regime) cells
# ---------------------------------------------------------------------------

def test_enumerate_suite_cells_covers_all_combinations():
    cells = gen.enumerate_suite_cells()

    covered = {(c["target_family"], c["feature_regime"]) for c in cells}
    expected = {
        (t, f)
        for t in gen.V2_TARGET_FAMILIES
        for f in gen.V2_FEATURE_REGIMES
    }

    missing = expected - covered
    assert missing == set(), f"Missing (family, regime) combinations: {missing}"
    assert len(cells) == 420, f"Expected 420 cells, got {len(cells)}"


# ---------------------------------------------------------------------------
# Test 4: write_parquet emits v2 training metadata columns
# ---------------------------------------------------------------------------

def test_write_parquet_emits_v2_training_metadata():
    tmp_dir = tempfile.mkdtemp()
    try:
        ds = gen.generate_v2_dataset(
            target_family="sin_low",
            feature_regime="ar1",
            n=100,
            p_signal=8,
            target_noise_scale=1.0,
            target_noise_type="gaussian",
            suite_component="core",
            teacher_seed=999,
            sample_seed=888,
        )
        path = os.path.join(tmp_dir, "sample.parquet")
        gen.write_parquet(ds, path)

        table = pq.read_table(path)
        for column in (
            "prior_regime",
            "profile",
            "feature_regime",
            "p_signal",
            "p_noise",
            "p_total",
            "covariance_type",
            "feature_noise_level",
            "sample_complexity_bucket",
            "X_train",
            "y_train",
            "X_test",
            "betaX_test",
        ):
            assert column in table.column_names, f"Missing column: {column!r}"
        data = table.to_pydict()
        assert data["prior_regime"][0] == "sin_low"
        assert data["feature_regime"][0] == "ar1"
        assert data["profile"][0] == "v2"
        # feature_noise_level must be float type in the schema
        assert str(table.schema.field("feature_noise_level").type) in ("double", "float64")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: backward compat — build_meta_nonlinear_dataset_index handles legacy parquet
# ---------------------------------------------------------------------------

def test_nonlinear_index_metadata_reader_handles_legacy_parquet():
    import pytest

    pytest.importorskip("snowflake.snowpark")
    import build_meta_nonlinear_dataset_index as indexer

    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "legacy.parquet")
        pq.write_table(
            pa.table({
                "n": pa.array([40], type=pa.int64()),
                "p": pa.array([5], type=pa.int64()),
                "n_train": pa.array([32], type=pa.int64()),
                "n_test": pa.array([8], type=pa.int64()),
                "prior_regime": pa.array(["I"], type=pa.utf8()),
            }),
            path,
        )

        metadata = indexer._read_metadata(path)

        assert metadata["profile"] == "legacy"
        assert metadata["feature_regime"] == "legacy_gaussian"
        assert metadata["p_signal"] == 5
        assert metadata["p_noise"] == 0
        assert metadata["p_total"] == 5
        assert metadata["feature_noise_level"] == 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
