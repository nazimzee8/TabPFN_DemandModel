"""Tests for F1: validate_per_split_k_coverage in dgp_helpers."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from dgp_helpers import validate_per_split_k_coverage


def _make_datasets(*k_values):
    return [{"num_classes": k} for k in k_values]


def test_all_required_k_present():
    datasets = _make_datasets(2, 3, 5, 10, 2, 3)
    result = validate_per_split_k_coverage(datasets, required_k=[2, 3, 5, 10])
    assert result["coverage_passed"] is True
    assert result["missing_k"] == []


def test_missing_k_detected():
    datasets = _make_datasets(2, 2, 3, 3)
    result = validate_per_split_k_coverage(datasets, required_k=[2, 3, 5, 10])
    assert result["coverage_passed"] is False
    assert 5 in result["missing_k"]
    assert 10 in result["missing_k"]


def test_strict_coverage_raises_on_missing():
    datasets = _make_datasets(2, 2)
    with pytest.raises(ValueError, match="missing required K"):
        validate_per_split_k_coverage(datasets, required_k=[2, 3], strict_coverage=True)


def test_strict_coverage_passes_when_all_present():
    datasets = _make_datasets(2, 3, 5, 10)
    result = validate_per_split_k_coverage(datasets, required_k=[2, 3, 5, 10], strict_coverage=True)
    assert result["coverage_passed"] is True


def test_realized_k_counts_correct():
    datasets = _make_datasets(2, 2, 3, 5, 5, 5, 10)
    result = validate_per_split_k_coverage(datasets, required_k=[2, 3, 5, 10])
    assert result["realized_k_counts"][2] == 2
    assert result["realized_k_counts"][3] == 1
    assert result["realized_k_counts"][5] == 3
    assert result["realized_k_counts"][10] == 1


def test_empty_datasets():
    result = validate_per_split_k_coverage([], required_k=[2, 3, 5, 10])
    assert result["coverage_passed"] is False
    assert result["missing_k"] == [2, 3, 5, 10]
