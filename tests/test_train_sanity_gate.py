"""
tests/test_train_sanity_gate.py

Tests for Phase 4: train-time model sanity gate.
Mocks sanity_checks module; does not run real checks.
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import importlib
import train


def _make_mock_model():
    """Create a minimal mock model."""
    import torch.nn as nn
    model = nn.Linear(2, 1)
    return model


class TestSanityGateConstants:
    def test_train_run_sanity_checks_constant_exists(self):
        assert hasattr(train, "TRAIN_RUN_SANITY_CHECKS")

    def test_train_sanity_check_strict_constant_exists(self):
        assert hasattr(train, "TRAIN_SANITY_CHECK_STRICT")

    def test_train_sanity_out_dir_constant_exists(self):
        assert hasattr(train, "TRAIN_SANITY_OUT_DIR")

    def test_train_sanity_write_all_ranks_constant_exists(self):
        assert hasattr(train, "TRAIN_SANITY_WRITE_ALL_RANKS")


class TestRunTrainSanityGate:
    def test_gate_skipped_when_disabled(self, monkeypatch):
        """When TRAIN_RUN_SANITY_CHECKS=false, gate returns immediately."""
        monkeypatch.setattr(train, "TRAIN_RUN_SANITY_CHECKS", False)
        model = _make_mock_model()
        # Should return without calling any sanity_checks functions
        with patch("sanity_checks.run_all_checks") as mock_checks:
            train._run_train_sanity_gate(model, "cpu", rank=0, is_main=True)
            mock_checks.assert_not_called()

    def test_gate_called_when_enabled(self, monkeypatch):
        """When TRAIN_RUN_SANITY_CHECKS=true, run_all_checks is invoked."""
        tmp_path = tempfile.mkdtemp()
        monkeypatch.setattr(train, "TRAIN_RUN_SANITY_CHECKS", True)
        monkeypatch.setattr(train, "TRAIN_SANITY_CHECK_STRICT", False)
        monkeypatch.setattr(train, "TRAIN_SANITY_OUT_DIR", tmp_path)
        monkeypatch.setattr(train, "TRAIN_SANITY_WRITE_ALL_RANKS", False)
        model = _make_mock_model()
        mock_results = {"all_passed": True}
        mock_save = MagicMock()
        with patch("sanity_checks.run_all_checks", return_value=mock_results) as mock_checks, \
             patch("sanity_checks.save_results", mock_save):
            train._run_train_sanity_gate(model, "cpu", rank=0, is_main=True)
            mock_checks.assert_called_once()

    def test_strict_failure_raises(self, monkeypatch):
        """When all_passed=False and STRICT=true, raises RuntimeError."""
        tmp_path = tempfile.mkdtemp()
        monkeypatch.setattr(train, "TRAIN_RUN_SANITY_CHECKS", True)
        monkeypatch.setattr(train, "TRAIN_SANITY_CHECK_STRICT", True)
        monkeypatch.setattr(train, "TRAIN_SANITY_OUT_DIR", tmp_path)
        monkeypatch.setattr(train, "TRAIN_SANITY_WRITE_ALL_RANKS", False)
        model = _make_mock_model()
        mock_results = {
            "all_passed": False,
            "check_permutation_invariance": {"passed": False, "max_abs_delta": 0.5},
        }
        with patch("sanity_checks.run_all_checks", return_value=mock_results), \
             patch("sanity_checks.save_results"):
            with pytest.raises(RuntimeError, match="sanity checks FAILED"):
                train._run_train_sanity_gate(model, "cpu", rank=0, is_main=True)

    def test_strict_false_logs_and_continues(self, monkeypatch, capsys):
        """When all_passed=False and STRICT=false, no exception, but warning printed."""
        tmp_path = tempfile.mkdtemp()
        monkeypatch.setattr(train, "TRAIN_RUN_SANITY_CHECKS", True)
        monkeypatch.setattr(train, "TRAIN_SANITY_CHECK_STRICT", False)
        monkeypatch.setattr(train, "TRAIN_SANITY_OUT_DIR", tmp_path)
        monkeypatch.setattr(train, "TRAIN_SANITY_WRITE_ALL_RANKS", False)
        model = _make_mock_model()
        mock_results = {
            "all_passed": False,
            "check_permutation_invariance": {"passed": False, "max_abs_delta": 0.5},
        }
        with patch("sanity_checks.run_all_checks", return_value=mock_results), \
             patch("sanity_checks.save_results"):
            # Should not raise
            train._run_train_sanity_gate(model, "cpu", rank=0, is_main=True)
        captured = capsys.readouterr()
        assert "FAILED" in captured.out

    def test_rank_specific_output_dir_when_write_all_ranks(self, monkeypatch):
        """When WRITE_ALL_RANKS=true, out_dir contains rank suffix."""
        tmp_path = tempfile.mkdtemp()
        monkeypatch.setattr(train, "TRAIN_RUN_SANITY_CHECKS", True)
        monkeypatch.setattr(train, "TRAIN_SANITY_CHECK_STRICT", False)
        monkeypatch.setattr(train, "TRAIN_SANITY_OUT_DIR", tmp_path)
        monkeypatch.setattr(train, "TRAIN_SANITY_WRITE_ALL_RANKS", True)
        model = _make_mock_model()
        mock_results = {"all_passed": True}
        captured_dirs = []
        def mock_save(results, out_dir):
            captured_dirs.append(out_dir)
        with patch("sanity_checks.run_all_checks", return_value=mock_results), \
             patch("sanity_checks.save_results", side_effect=mock_save):
            train._run_train_sanity_gate(model, "cpu", rank=2, is_main=False)
        assert len(captured_dirs) == 1
        assert "rank2" in captured_dirs[0]

    def test_rank0_shared_dir_when_write_all_ranks_false(self, monkeypatch):
        """When WRITE_ALL_RANKS=false and is_main=True, out_dir is TRAIN_SANITY_OUT_DIR directly."""
        base_dir = tempfile.mkdtemp()
        monkeypatch.setattr(train, "TRAIN_RUN_SANITY_CHECKS", True)
        monkeypatch.setattr(train, "TRAIN_SANITY_CHECK_STRICT", False)
        monkeypatch.setattr(train, "TRAIN_SANITY_OUT_DIR", base_dir)
        monkeypatch.setattr(train, "TRAIN_SANITY_WRITE_ALL_RANKS", False)
        model = _make_mock_model()
        mock_results = {"all_passed": True}
        captured_dirs = []
        def mock_save(results, out_dir):
            captured_dirs.append(out_dir)
        with patch("sanity_checks.run_all_checks", return_value=mock_results), \
             patch("sanity_checks.save_results", side_effect=mock_save):
            train._run_train_sanity_gate(model, "cpu", rank=0, is_main=True)
        assert len(captured_dirs) == 1
        assert captured_dirs[0] == base_dir
