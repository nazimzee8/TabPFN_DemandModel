"""Tests for run_permutation_tests device handling in deepset_inference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import DeepSetModel, ModelConfig
from deepset_inference import run_permutation_tests


def _make_tiny_model():
    cfg = ModelConfig(d_phi=8, d_rho=8, n_sab_feat=0, n_sab_samp=0)
    return DeepSetModel(cfg=cfg)


class TestRunPermutationTestsDevice:
    def test_run_permutation_tests_cpu(self):
        model = _make_tiny_model()
        assert run_permutation_tests(model) is True

    def test_run_permutation_tests_cuda_no_device_mismatch(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = _make_tiny_model().to("cuda")
        assert run_permutation_tests(model) is True
