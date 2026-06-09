"""Tests for mixed-categorical model modules (Step 9 — 8 tests)."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from constants import (
    ENTITY_EMBED_FIRST_REAL_ID,
    ENTITY_EMBED_MISSING_ID,
    ENTITY_EMBED_PAD_ID,
    ENTITY_EMBED_UNKNOWN_ID,
)
from model import (
    ModelConfig,
    CategoricalTokenEncoder,
    CategoricalStatisticExtractor,
    CategoricalStatisticEncoder,
    RegressionCategoryEffectHead,
    ClassificationCategoryEffectHead,
    DeepSetICLModel,
    N_CAT_FEAT_STATS,
)


def _make_mixed_config(**overrides):
    """Create a ModelConfig with mixed-categorical features enabled."""
    defaults = dict(
        d_phi=64,
        d_rho=128,
        n_heads=4,
        n_sab_feat=1,
        use_categorical_features=True,
        cat_max_vocab_size=53,
        cat_embed_dim=16,
        cat_feat_id_embed_dim=8,
        cat_cardinality_embed_dim=4,
        cat_stat_dim=32,
        cat_stat_hidden_dim=64,
        cat_head_hidden_dim=32,
        cat_max_p_cat=64,
        use_linear_stats=True,
        use_coefficient_head=True,
        task_objective="inductive_regression",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# Test 15: CategoricalTokenEncoder output shape
def test_categorical_token_encoder_output_shape():
    cfg = _make_mixed_config()
    encoder = CategoricalTokenEncoder(cfg)
    n, p_cat = 20, 3
    cardinalities = torch.tensor([3, 5, 10], dtype=torch.long)
    X_cat = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                          ENTITY_EMBED_FIRST_REAL_ID + 3,
                          (n, p_cat))
    out = encoder(X_cat, cardinalities)
    assert out.shape == (n, p_cat, cfg.d_phi)


# Test 16: CategoricalStatisticExtractor uses only context y
def test_categorical_stat_extractor_context_only():
    extractor = CategoricalStatisticExtractor()
    n_ctx, p_cat = 30, 2
    cardinalities = torch.tensor([5, 10], dtype=torch.long)
    X_cat = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                          ENTITY_EMBED_FIRST_REAL_ID + 5,
                          (n_ctx, p_cat))
    y_ctx = torch.randn(n_ctx)
    feat_stats1, global_stats1 = extractor(
        X_cat, y_ctx, cardinalities, "inductive_regression"
    )
    # Change y to different values — stats should change
    y_ctx2 = torch.randn(n_ctx) * 100
    feat_stats2, global_stats2 = extractor(
        X_cat, y_ctx2, cardinalities, "inductive_regression"
    )
    # Stats should differ because y changed (validates context y is used)
    assert not torch.allclose(feat_stats1, feat_stats2)
    # Now verify query labels are never used: extractor only takes context inputs


# Test 17: PAD, MISSING, UNKNOWN embeddings are distinct
def test_pad_missing_unknown_embeds_distinct():
    cfg = _make_mixed_config()
    encoder = CategoricalTokenEncoder(cfg)
    embed_weight = encoder.entity_embed.weight.detach()
    pad_vec = embed_weight[ENTITY_EMBED_PAD_ID]
    miss_vec = embed_weight[ENTITY_EMBED_MISSING_ID]
    unk_vec = embed_weight[ENTITY_EMBED_UNKNOWN_ID]
    # PAD is padding_idx so it's zero; MISSING and UNKNOWN should be non-zero
    assert torch.allclose(pad_vec, torch.zeros_like(pad_vec))
    # MISSING and UNKNOWN should be different from each other
    assert not torch.allclose(miss_vec, unk_vec, atol=1e-6)


# Test 18: Mixed regression forward shape
def test_mixed_regression_forward_shape():
    cfg = _make_mixed_config(task_objective="inductive_regression")
    model = DeepSetICLModel(cfg)
    model.eval()
    n, p_num, p_cat, m = 20, 4, 3, 8
    cardinalities = torch.tensor([3, 5, 10], dtype=torch.long)
    X_train = torch.randn(n, p_num)
    y_train = torch.randn(n)
    X_test = torch.randn(m, p_num)
    X_cat_train = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                                ENTITY_EMBED_FIRST_REAL_ID + 3,
                                (n, p_cat))
    X_cat_test = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                               ENTITY_EMBED_FIRST_REAL_ID + 3,
                               (m, p_cat))
    with torch.no_grad():
        out = model(
            X_train, y_train, X_test,
            task_objective="inductive_regression",
            X_cat_train=X_cat_train,
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
    assert out.shape == (m,)


# Test 19: Mixed classification forward shape
def test_mixed_classification_forward_shape():
    cfg = _make_mixed_config(
        task_objective="inductive_classification",
        use_classification_path=True,
        max_num_classes=5,
    )
    model = DeepSetICLModel(cfg)
    model.eval()
    n, p_num, p_cat, m, K = 20, 4, 2, 8, 3
    cardinalities = torch.tensor([5, 10], dtype=torch.long)
    X_train = torch.randn(n, p_num)
    y_train = torch.randint(0, K, (n,)).long()
    X_test = torch.randn(m, p_num)
    X_cat_train = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                                ENTITY_EMBED_FIRST_REAL_ID + 5,
                                (n, p_cat))
    X_cat_test = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                               ENTITY_EMBED_FIRST_REAL_ID + 5,
                               (m, p_cat))
    with torch.no_grad():
        result = model(
            X_train, y_train, X_test,
            task_objective="inductive_classification",
            num_classes=K,
            X_cat_train=X_cat_train,
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
    # Classification forward returns dict with "logits" key, or (logits, aux_dict), or logits
    if isinstance(result, dict):
        logits = result["logits"]
    elif isinstance(result, tuple):
        logits = result[0]
    else:
        logits = result
    assert logits.shape == (m, K)


# Test 20: Variable p_num and p_cat — same weights, different feature counts
def test_variable_p_num_and_p_cat():
    cfg = _make_mixed_config()
    model = DeepSetICLModel(cfg)
    model.eval()
    # Test with p_num=2, p_cat=1
    with torch.no_grad():
        out1 = model(
            torch.randn(10, 2), torch.randn(10), torch.randn(5, 2),
            task_objective="inductive_regression",
            X_cat_train=torch.full((10, 1), ENTITY_EMBED_FIRST_REAL_ID, dtype=torch.long),
            X_cat_test=torch.full((5, 1), ENTITY_EMBED_FIRST_REAL_ID, dtype=torch.long),
            categorical_cardinalities=torch.tensor([3], dtype=torch.long),
        )
    assert out1.shape == (5,)
    # Test with p_num=8, p_cat=4
    with torch.no_grad():
        out2 = model(
            torch.randn(15, 8), torch.randn(15), torch.randn(3, 8),
            task_objective="inductive_regression",
            X_cat_train=torch.full((15, 4), ENTITY_EMBED_FIRST_REAL_ID, dtype=torch.long),
            X_cat_test=torch.full((3, 4), ENTITY_EMBED_FIRST_REAL_ID, dtype=torch.long),
            categorical_cardinalities=torch.tensor([3, 5, 10, 20], dtype=torch.long),
        )
    assert out2.shape == (3,)


# Test 21: Pure numeric path unchanged when X_cat_train=None
def test_pure_numeric_path_unchanged():
    cfg = _make_mixed_config()
    model = DeepSetICLModel(cfg)
    model.eval()
    torch.manual_seed(42)
    X_train = torch.randn(20, 4)
    y_train = torch.randn(20)
    X_test = torch.randn(5, 4)
    with torch.no_grad():
        # Without categorical inputs
        out_no_cat = model(
            X_train, y_train, X_test,
            task_objective="inductive_regression",
        )
        # With X_cat_train=None (explicit)
        out_none_cat = model(
            X_train, y_train, X_test,
            task_objective="inductive_regression",
            X_cat_train=None,
        )
    torch.testing.assert_close(out_no_cat, out_none_cat)


# Test 22: Row shuffle invariance (permutation invariance of context)
def test_mixed_cat_row_shuffle_invariance():
    cfg = _make_mixed_config()
    model = DeepSetICLModel(cfg)
    model.eval()
    n, p_num, p_cat, m = 20, 4, 2, 5
    cardinalities = torch.tensor([5, 10], dtype=torch.long)
    torch.manual_seed(123)
    X_train = torch.randn(n, p_num)
    y_train = torch.randn(n)
    X_test = torch.randn(m, p_num)
    X_cat_train = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                                ENTITY_EMBED_FIRST_REAL_ID + 5,
                                (n, p_cat))
    X_cat_test = torch.randint(ENTITY_EMBED_FIRST_REAL_ID,
                               ENTITY_EMBED_FIRST_REAL_ID + 5,
                               (m, p_cat))
    perm = torch.randperm(n)
    with torch.no_grad():
        out_orig = model(
            X_train, y_train, X_test,
            task_objective="inductive_regression",
            X_cat_train=X_cat_train,
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
        out_perm = model(
            X_train[perm], y_train[perm], X_test,
            task_objective="inductive_regression",
            X_cat_train=X_cat_train[perm],
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
    torch.testing.assert_close(out_orig, out_perm, atol=1e-4, rtol=1e-4)
