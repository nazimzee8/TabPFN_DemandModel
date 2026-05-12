import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import hpo  # noqa: E402
from model import ModelConfig  # noqa: E402


def test_checkpoint_architecture_mismatches_accepts_dict_cfg():
    saved_cfg = {
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_heads": 4,
        "n_sab_feat": 1,
        "n_sab_samp": 1,
        "norm_feat": True,
        "norm_target": True,
        "dropout": 0.1,
    }
    current_cfg = ModelConfig(**saved_cfg)

    assert hpo.checkpoint_architecture_mismatches(saved_cfg, current_cfg) == {}


def test_checkpoint_architecture_mismatches_ignores_dropout():
    saved_cfg = {
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_heads": 4,
        "n_sab_feat": 1,
        "n_sab_samp": 1,
        "norm_feat": True,
        "norm_target": True,
        "dropout": 0.1,
    }
    current_cfg = ModelConfig(**{**saved_cfg, "dropout": 0.3})

    assert hpo.checkpoint_architecture_mismatches(saved_cfg, current_cfg) == {}


def test_checkpoint_architecture_mismatches_rejects_missing_cfg():
    with pytest.raises(ValueError, match="missing required cfg"):
        hpo.checkpoint_architecture_mismatches(None, ModelConfig())
