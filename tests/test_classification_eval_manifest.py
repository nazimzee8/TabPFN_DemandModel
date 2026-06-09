"""
test_classification_eval_manifest.py
======================================
Tests for manifest JSON correctness produced by
scripts/generate_synthetic_classification.py.

Uses subprocess to run the script with a small dataset count so tests
remain fast.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_synthetic_classification.py"

# Ensure src/ is importable for dgp_helpers constants.
import sys as _sys
_src = str(ROOT / "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest_data(tmp_path_factory):
    """Run the script with tiny counts and return parsed manifest + suite_dir."""
    tmp = tmp_path_factory.mktemp("cls_manifest")
    suite_id = "manifest_test_v1"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--suite_id", suite_id,
            "--base_seed", "7777",
            "--n_datasets", "4",
            "--n_datasets_per_sweep", "2",
            "--output_root", str(tmp),
            "--profile", "linear_classification_stat_aware",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"Script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    suite_dir = tmp / suite_id
    manifest_path = suite_dir / "synthetic_classification_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, suite_dir, suite_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_suite_id_present_and_matches(manifest_data):
    """1. suite_id present and matches arg."""
    manifest, _, suite_id = manifest_data
    assert "suite_id" in manifest
    assert manifest["suite_id"] == suite_id


def test_task_family(manifest_data):
    """2. task_family == 'linear_classification'."""
    manifest, _, _ = manifest_data
    assert manifest["task_family"] == "linear_classification"


def test_task_objective(manifest_data):
    """3. task_objective == 'inductive_classification'."""
    manifest, _, _ = manifest_data
    assert manifest["task_objective"] == "inductive_classification"


def test_schema_version(manifest_data):
    """4. schema_version matches CLASSIFICATION_EVAL_SCHEMA_VERSION constant."""
    from dgp_helpers import CLASSIFICATION_EVAL_SCHEMA_VERSION
    manifest, _, _ = manifest_data
    assert manifest["schema_version"] == CLASSIFICATION_EVAL_SCHEMA_VERSION


def test_base_seed_present_and_int(manifest_data):
    """5. base_seed present and int."""
    manifest, _, _ = manifest_data
    assert "base_seed" in manifest
    assert isinstance(manifest["base_seed"], int)


def test_n_datasets_matches_datasets_length(manifest_data):
    """6. n_datasets matches len(datasets)."""
    manifest, _, _ = manifest_data
    assert manifest["n_datasets"] == len(manifest["datasets"])


def test_configured_profile_weights_present(manifest_data):
    """7. configured_profile_weights present and non-empty dict."""
    manifest, _, _ = manifest_data
    assert "configured_profile_weights" in manifest
    assert isinstance(manifest["configured_profile_weights"], dict)
    assert len(manifest["configured_profile_weights"]) > 0


def test_effective_profile_weights_present(manifest_data):
    """8. effective_profile_weights present."""
    manifest, _, _ = manifest_data
    assert "effective_profile_weights" in manifest
    assert isinstance(manifest["effective_profile_weights"], dict)


def test_realized_regime_counts_present(manifest_data):
    """9. realized_regime_counts present and dict."""
    manifest, _, _ = manifest_data
    assert "realized_regime_counts" in manifest
    assert isinstance(manifest["realized_regime_counts"], dict)


def test_realized_suite_family_counts_present(manifest_data):
    """10. realized_suite_family_counts present."""
    manifest, _, _ = manifest_data
    assert "realized_suite_family_counts" in manifest
    assert isinstance(manifest["realized_suite_family_counts"], dict)


def test_realized_k_counts_present(manifest_data):
    """11. realized_K_counts present."""
    manifest, _, _ = manifest_data
    assert "realized_K_counts" in manifest


def test_realized_imbalance_counts_present(manifest_data):
    """12. realized_imbalance_counts present."""
    manifest, _, _ = manifest_data
    assert "realized_imbalance_counts" in manifest


def test_realized_margin_counts_present(manifest_data):
    """13. realized_margin_counts present."""
    manifest, _, _ = manifest_data
    assert "realized_margin_counts" in manifest


def test_realized_label_noise_counts_present(manifest_data):
    """14. realized_label_noise_counts present."""
    manifest, _, _ = manifest_data
    assert "realized_label_noise_counts" in manifest


def test_realized_feature_noise_counts_present(manifest_data):
    """15. realized_feature_noise_counts present."""
    manifest, _, _ = manifest_data
    assert "realized_feature_noise_counts" in manifest


def test_datasets_list_present(manifest_data):
    """16. datasets list present."""
    manifest, _, _ = manifest_data
    assert "datasets" in manifest
    assert isinstance(manifest["datasets"], list)
    assert len(manifest["datasets"]) > 0


def test_file_checksums_64char_hex(manifest_data):
    """17. file_checksums present with 64-char hex values."""
    manifest, _, _ = manifest_data
    assert "file_checksums" in manifest
    checksums = manifest["file_checksums"]
    assert isinstance(checksums, dict)
    for filepath, checksum in checksums.items():
        assert len(checksum) == 64, f"Checksum for {filepath} is not 64 chars"
        int(checksum, 16)  # raises if not valid hex


def test_output_checksum_64char_hex(manifest_data):
    """18. output_checksum present (64-char hex)."""
    manifest, _, _ = manifest_data
    assert "output_checksum" in manifest
    oc = manifest["output_checksum"]
    assert len(oc) == 64
    int(oc, 16)  # raises if not valid hex


def test_stage_paths_use_correct_prefix(manifest_data):
    """19. stage_paths use @EVALUATION_DATASET_STAGE and synthetic_classification_prepared."""
    manifest, _, suite_id = manifest_data
    assert "stage_paths" in manifest
    for fam, path in manifest["stage_paths"].items():
        assert "@EVALUATION_DATASET_STAGE" in path, f"Bad stage path for {fam}: {path}"
        assert "synthetic_classification_prepared" in path, f"Bad stage path for {fam}: {path}"
        assert suite_id in path


def test_grid_metadata_contains_required_keys(manifest_data):
    """20. grid_metadata contains n_grid, num_classes_grid, and label_noise_grid."""
    manifest, _, _ = manifest_data
    assert "grid_metadata" in manifest
    gm = manifest["grid_metadata"]
    assert "n_grid" in gm
    assert "num_classes_grid" in gm
    assert "label_noise_grid" in gm


def test_created_at_is_parseable_iso8601(manifest_data):
    """21. created_at parseable ISO-8601."""
    manifest, _, _ = manifest_data
    assert "created_at" in manifest
    # datetime.fromisoformat raises if not valid ISO-8601
    dt = datetime.fromisoformat(manifest["created_at"])
    assert dt.tzinfo is not None


def test_git_revision_present(manifest_data):
    """22. git_revision present."""
    manifest, _, _ = manifest_data
    assert "git_revision" in manifest
    assert isinstance(manifest["git_revision"], str)


def test_generation_elapsed_seconds_positive(manifest_data):
    """23. generation_elapsed_seconds > 0."""
    manifest, _, _ = manifest_data
    assert "generation_elapsed_seconds" in manifest
    assert manifest["generation_elapsed_seconds"] > 0


def test_per_dataset_records_have_required_fields(manifest_data):
    """24. Per-dataset records contain all required fields."""
    manifest, _, _ = manifest_data
    required = {
        "filepath", "dataset_idx", "global_idx", "suite_family",
        "regime", "K", "imbalance_type", "margin_level",
        "label_noise_rate", "feature_noise_level",
        "n_train", "n_test", "n_features", "n_rows",
        "is_tabpfn_anchor", "file_checksum_sha256", "extra",
    }
    for rec in manifest["datasets"]:
        missing = required - set(rec.keys())
        assert not missing, f"Record missing fields: {missing}\nRecord: {rec}"


def test_global_idx_unique_across_records(manifest_data):
    """25. global_idx values unique across all records."""
    manifest, _, _ = manifest_data
    ids = [r["global_idx"] for r in manifest["datasets"]]
    assert len(ids) == len(set(ids)), "global_idx values are not unique"


def test_enabled_suite_families_matches_realized_counts_keys(manifest_data):
    """26. enabled_suite_families matches realized_suite_family_counts keys."""
    manifest, _, _ = manifest_data
    enabled = set(manifest["enabled_suite_families"])
    realized = set(manifest["realized_suite_family_counts"].keys())
    assert enabled == realized


def test_generation_flags_block_present(manifest_data):
    """27. generation_flags block present with all 4 flags."""
    manifest, _, _ = manifest_data
    assert "generation_flags" in manifest
    flags = manifest["generation_flags"]
    required_flags = {
        "store_class_params",
        "store_class_teacher_preds",
        "require_class_teachers",
        "allow_underdetermined",
    }
    missing = required_flags - set(flags.keys())
    assert not missing, f"generation_flags missing: {missing}"
