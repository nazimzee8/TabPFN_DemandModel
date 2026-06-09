"""Tests for beta field loading in train.py (Area 3).

Four cases:
1. Parquet with 'beta' field → tensor returned, canonical path.
2. Parquet without 'beta' or 'beta_true' → None returned.
3. Parquet with 'beta_true' only (legacy) → fallback works, tensor returned.
4. Nonlinear parquet (no beta) → None returned, no error.
"""

import io
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch or pyarrow not installed")


def _make_parquet(tmp_dir, include_beta=None, include_beta_true=None, n_train=5, p=3):
    """Create a minimal meta-dataset parquet with optional beta/beta_true columns."""
    rng = np.random.default_rng(42)
    n_test = 2
    n_total = n_train + n_test
    X = rng.normal(size=(n_total, p)).astype(np.float32)
    y = rng.normal(size=n_total).astype(np.float32)
    beta_vec = rng.normal(size=p).astype(np.float32)

    payload = {
        "X_train":    [X[:n_train].tolist()],
        "y_train":    [y[:n_train].tolist()],
        "X_test":     [X[n_train:].tolist()],
        "betaX_test": [y[n_train:].tolist()],
        "n":          [n_total],
        "p":          [p],
        "n_train":    [n_train],
        "n_test":     [n_test],
        "prior_regime": ["A_iid_dense"],
    }
    if include_beta is not None:
        payload["beta"] = [include_beta.tolist()]
    if include_beta_true is not None:
        payload["beta_true"] = [include_beta_true.tolist()]

    table = pa.table({k: pa.array(v) for k, v in payload.items()})
    path = os.path.join(tmp_dir, "test_dataset.parquet")
    pq.write_table(table, path)
    return path, beta_vec


class TestLoadParquet:
    def test_canonical_beta_field_returned(self, tmp_path):
        """Case 1: canonical 'beta' field read correctly."""
        from train import load_parquet
        rng = np.random.default_rng(0)
        beta_vec = rng.normal(size=3).astype(np.float32)
        path, _ = _make_parquet(str(tmp_path), include_beta=beta_vec)
        _, _, _, _, beta = load_parquet(path)
        assert beta is not None
        assert isinstance(beta, torch.Tensor)
        np.testing.assert_allclose(beta.numpy(), beta_vec, atol=1e-6)

    def test_no_beta_returns_none(self, tmp_path):
        """Case 2: parquet without beta or beta_true → None."""
        from train import load_parquet
        path, _ = _make_parquet(str(tmp_path), include_beta=None, include_beta_true=None)
        _, _, _, _, beta = load_parquet(path)
        assert beta is None

    def test_legacy_beta_true_fallback(self, tmp_path):
        """Case 3: parquet with 'beta_true' only (legacy) → fallback tensor returned."""
        from train import load_parquet
        rng = np.random.default_rng(1)
        beta_vec = rng.normal(size=3).astype(np.float32)
        path, _ = _make_parquet(str(tmp_path), include_beta=None, include_beta_true=beta_vec)
        _, _, _, _, beta = load_parquet(path)
        assert beta is not None
        assert isinstance(beta, torch.Tensor)
        np.testing.assert_allclose(beta.numpy(), beta_vec, atol=1e-6)

    def test_canonical_beats_legacy_when_both_present(self, tmp_path):
        """'beta' field takes priority over 'beta_true' when both exist."""
        from train import load_parquet
        rng = np.random.default_rng(2)
        canonical_beta = rng.normal(size=3).astype(np.float32)
        legacy_beta    = rng.normal(size=3).astype(np.float32) + 100.0  # clearly different
        path, _ = _make_parquet(
            str(tmp_path),
            include_beta=canonical_beta,
            include_beta_true=legacy_beta,
        )
        _, _, _, _, beta = load_parquet(path)
        assert beta is not None
        np.testing.assert_allclose(beta.numpy(), canonical_beta, atol=1e-6)

    def test_nonlinear_parquet_no_error(self, tmp_path):
        """Case 4: nonlinear parquet (no beta) → None, no error."""
        from train import load_parquet
        path, _ = _make_parquet(str(tmp_path), include_beta=None, include_beta_true=None)
        result = load_parquet(path)
        assert len(result) == 5
        assert result[4] is None

    def test_return_is_5_tuple(self, tmp_path):
        from train import load_parquet
        path, _ = _make_parquet(str(tmp_path))
        result = load_parquet(path)
        assert len(result) == 5
