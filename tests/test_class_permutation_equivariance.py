"""Tests for F2: class-label permutation equivariance in permute_class_labels."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from classification import permute_class_labels


def _make_data(n_train=40, n_test=20, num_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y_train = torch.tensor(rng.integers(0, num_classes, n_train), dtype=torch.long)
    y_test  = torch.tensor(rng.integers(0, num_classes, n_test),  dtype=torch.long)
    teacher_probs  = torch.softmax(torch.randn(n_test,  num_classes), dim=-1)
    teacher_logits = torch.randn(n_test, num_classes)
    W = torch.randn(5, num_classes)
    b = torch.randn(num_classes)
    teacher_dict = {
        "teacher_probs": teacher_probs,
        "teacher_logits": teacher_logits,
        "W_teacher_norm": W,
        "b_teacher": b,
    }
    return y_train, y_test, teacher_dict, num_classes


@pytest.mark.parametrize("num_classes", [2, 3, 5, 10])
def test_inverse_permutation_recovers_original(num_classes):
    """Applying permutation then its inverse must recover the original labels."""
    rng_np = np.random.default_rng(42)
    y_train, y_test, td, K = _make_data(num_classes=num_classes)

    perm = rng_np.permutation(K)
    perm_tensor = torch.as_tensor(perm, dtype=torch.long)

    rng_perm = np.random.default_rng(42)
    y_tr_p, y_te_p, _ = permute_class_labels(y_train, y_test, None, rng_perm, K)

    # Build inverse permutation
    inv_perm = torch.zeros(K, dtype=torch.long)
    inv_perm[perm_tensor] = torch.arange(K)
    y_tr_rec = inv_perm[y_tr_p]
    y_te_rec = inv_perm[y_te_p]

    assert torch.equal(y_tr_rec, y_train), "y_train not recovered by inverse permutation"
    assert torch.equal(y_te_rec, y_test),  "y_test not recovered by inverse permutation"


def test_identity_permutation_unchanged():
    """With K=1 effective permutation (single class repeated), output equals input.

    We test by seeding the rng such that the chosen permutation is identity,
    or by using K=1 (degenerate). Instead we check the teacher probs column
    sums remain normalised.
    """
    K = 3
    rng = np.random.default_rng(0)
    y_train, y_test, td, _ = _make_data(num_classes=K, seed=1)

    # Force identity permutation by using a custom rng that returns [0,1,...,K-1].
    class _IdentityRng:
        def permutation(self, n):
            return np.arange(n)

    y_tr_p, y_te_p, td_p = permute_class_labels(y_train, y_test, td, _IdentityRng(), K)

    assert torch.equal(y_tr_p, y_train)
    assert torch.equal(y_te_p, y_test)
    assert torch.allclose(td_p["teacher_probs"], td["teacher_probs"])


def test_teacher_probs_consistent_with_label_perm():
    """Teacher probability columns must follow the same permutation as labels."""
    K = 4
    rng_np = np.random.default_rng(7)
    y_train, y_test, td, _ = _make_data(num_classes=K, seed=3)

    perm = np.array([2, 0, 3, 1])  # fixed permutation for reproducibility

    class _FixedRng:
        def permutation(self, n):
            return perm[:n]

    y_tr_p, y_te_p, td_p = permute_class_labels(y_train, y_test, td, _FixedRng(), K)

    perm_tensor = torch.as_tensor(perm, dtype=torch.long)
    expected_probs = td["teacher_probs"][:, perm_tensor]
    assert torch.allclose(td_p["teacher_probs"], expected_probs)


def test_none_teacher_dict_returned_as_none():
    """When teacher_dict is None the function must return None for teacher output."""
    K = 3
    y_train, y_test, _, _ = _make_data(num_classes=K)
    rng = np.random.default_rng(0)
    _, _, td_out = permute_class_labels(y_train, y_test, None, rng, K)
    assert td_out is None
