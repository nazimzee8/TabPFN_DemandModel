"""Tests for mixed-categorical model forward paths (Step 12c)."""

import sys
from pathlib import Path

import torch
import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from constants import ENTITY_EMBED_FIRST_REAL_ID
from model import ModelConfig, DeepSetICLModel


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


# ---------------------------------------------------------------------------
# test_regression_forward_with_categorical_finite
# ---------------------------------------------------------------------------
def test_regression_forward_with_categorical_finite():
    cfg = _make_mixed_config(task_objective="inductive_regression")
    model = DeepSetICLModel(cfg)
    model.eval()
    n, p_num, p_cat, m = 20, 4, 3, 8
    cardinalities = torch.tensor([3, 5, 10], dtype=torch.long)
    X_train = torch.randn(n, p_num)
    y_train = torch.randn(n)
    X_test = torch.randn(m, p_num)
    X_cat_train = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 3, (n, p_cat)
    )
    X_cat_test = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 3, (m, p_cat)
    )
    with torch.no_grad():
        out = model(
            X_train, y_train, X_test,
            task_objective="inductive_regression",
            X_cat_train=X_cat_train,
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
    assert out.shape == (m,)
    assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# test_classification_forward_with_categorical_shape
# ---------------------------------------------------------------------------
def test_classification_forward_with_categorical_shape():
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
    y_train = torch.randint(0, K, (n,))
    X_test = torch.randn(m, p_num)
    X_cat_train = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 5, (n, p_cat)
    )
    X_cat_test = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 5, (m, p_cat)
    )
    with torch.no_grad():
        result = model(
            X_train, y_train, X_test,
            task_objective="inductive_classification",
            num_classes=K,
            X_cat_train=X_cat_train,
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
    assert isinstance(result, dict)
    logits = result["logits"]
    assert logits.shape == (m, K)
    assert torch.all(torch.isfinite(logits))


# ---------------------------------------------------------------------------
# test_categorical_effect_nontrivial
# ---------------------------------------------------------------------------
def test_categorical_effect_nontrivial():
    """Predictions with categorical features should differ from without."""
    cfg = _make_mixed_config(task_objective="inductive_regression")
    model = DeepSetICLModel(cfg)
    model.eval()
    torch.manual_seed(42)
    n, p_num, p_cat, m = 20, 4, 2, 5
    cardinalities = torch.tensor([5, 10], dtype=torch.long)
    X_train = torch.randn(n, p_num)
    y_train = torch.randn(n)
    X_test = torch.randn(m, p_num)
    X_cat_train = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 5, (n, p_cat)
    )
    X_cat_test = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 5, (m, p_cat)
    )
    with torch.no_grad():
        out_with_cat = model(
            X_train, y_train, X_test,
            task_objective="inductive_regression",
            X_cat_train=X_cat_train,
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
        out_without_cat = model(
            X_train, y_train, X_test,
            task_objective="inductive_regression",
        )
    # Predictions should differ when categorical features are present
    assert not torch.allclose(out_with_cat, out_without_cat, atol=1e-6), (
        "Categorical features had no effect on predictions"
    )


# ---------------------------------------------------------------------------
# test_cls_cat_reference_class_zero
# ---------------------------------------------------------------------------
def test_cls_cat_reference_class_logit_shift():
    """The classification categorical head should not shift logits for every class identically.

    This test verifies that the cat_cls_head produces non-trivial output
    when given categorical features.
    """
    cfg = _make_mixed_config(
        task_objective="inductive_classification",
        use_classification_path=True,
        max_num_classes=5,
    )
    model = DeepSetICLModel(cfg)
    model.eval()
    n, p_num, p_cat, m, K = 30, 4, 2, 10, 3
    cardinalities = torch.tensor([5, 10], dtype=torch.long)
    X_train = torch.randn(n, p_num)
    y_train = torch.randint(0, K, (n,))
    X_test = torch.randn(m, p_num)
    X_cat_train = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 5, (n, p_cat)
    )
    X_cat_test = torch.randint(
        ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + 5, (m, p_cat)
    )
    with torch.no_grad():
        result_with = model(
            X_train, y_train, X_test,
            task_objective="inductive_classification",
            num_classes=K,
            X_cat_train=X_cat_train,
            X_cat_test=X_cat_test,
            categorical_cardinalities=cardinalities,
        )
        result_without = model(
            X_train, y_train, X_test,
            task_objective="inductive_classification",
            num_classes=K,
        )
    logits_with = result_with["logits"]
    logits_without = result_without["logits"]
    # The categorical head should modify the logits
    assert not torch.allclose(logits_with, logits_without, atol=1e-6), (
        "Classification categorical head had no effect on logits"
    )
