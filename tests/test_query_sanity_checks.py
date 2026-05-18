"""
tests/test_query_sanity_checks.py

Tests that the sanity_checks module produces correct outputs (JSON/CSV)
and that individual checks pass on a fresh model.
"""

import json
import os
import shutil
import sys
import tempfile

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import sanity_checks  # noqa: E402


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------

def test_check_permutation_invariance_passes():
    """check_permutation_invariance passes on a fresh MarketAwareDeepSetModel."""
    result = sanity_checks.check_permutation_invariance(model=None)
    assert result["passed"], (
        f"check_permutation_invariance failed: {result}"
    )


def test_check_ridge_primal_dual_passes():
    """check_ridge_primal_dual_consistency passes."""
    result = sanity_checks.check_ridge_primal_dual_consistency()
    assert result["passed"], (
        f"check_ridge_primal_dual_consistency failed: {result}"
    )


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

def test_run_all_checks_returns_six_keys():
    """run_all_checks returns exactly 6 check keys (plus all_passed and device_info)."""
    results = sanity_checks.run_all_checks(model=None)
    check_keys = [k for k in results if k not in ("all_passed", "device_info")]
    assert len(check_keys) == 6, (
        f"Expected 6 check keys, got {len(check_keys)}: {check_keys}"
    )
    assert "all_passed" in results
    assert "device_info" in results


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

def test_save_results_writes_json():
    """save_results writes sanity_checks.json to the given directory."""
    out_dir = tempfile.mkdtemp(prefix="sanity_test_json_")
    try:
        results = sanity_checks.run_all_checks(model=None)
        sanity_checks.save_results(results, out_dir)

        json_path = os.path.join(out_dir, "sanity_checks.json")
        assert os.path.exists(json_path), f"Expected {json_path} to exist"

        with open(json_path) as f:
            loaded = json.load(f)
        assert "all_passed" in loaded
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_save_results_writes_csv():
    """save_results writes sanity_checks.csv with correct header."""
    out_dir = tempfile.mkdtemp(prefix="sanity_test_csv_")
    try:
        results = sanity_checks.run_all_checks(model=None)
        sanity_checks.save_results(results, out_dir)

        csv_path = os.path.join(out_dir, "sanity_checks.csv")
        assert os.path.exists(csv_path), f"Expected {csv_path} to exist"

        with open(csv_path) as f:
            header_line = f.readline().strip()

        expected_header = "check_name,passed,metric_name,metric_value"
        assert header_line == expected_header, (
            f"CSV header mismatch.\nExpected: {expected_header!r}\nGot:      {header_line!r}"
        )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
