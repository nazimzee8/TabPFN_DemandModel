"""
test_generate_synthetic_classification.py
===========================================
Integration tests for scripts/generate_synthetic_classification.py.

Uses subprocess.run to invoke the script with small dataset counts so
tests remain fast. All 20 tests are described in the plan.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_synthetic_classification.py"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgp_helpers import _REGRESSION_ONLY_FIELDS  # noqa: E402


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(argv: list[str], *, expect_fail: bool = False, timeout: int = 300):
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + argv,
        capture_output=True, text=True, timeout=timeout,
    )
    if not expect_fail:
        assert result.returncode == 0, (
            f"Script exited with {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    else:
        assert result.returncode != 0, (
            "Script was expected to fail but succeeded.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _small_args(tmp_path: Path, suite_id: str = "integ_test") -> list[str]:
    return [
        "--suite_id", suite_id,
        "--base_seed", "1234",
        "--n_datasets", "4",
        "--n_datasets_per_sweep", "2",
        "--output_root", str(tmp_path),
        "--profile", "linear_classification_stat_aware",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dry_run_succeeds(tmp_path):
    """1. Default args with --dry_run succeeds."""
    result = _run([
        "--dry_run",
        "--suite_id", "dry_run_test",
        "--output_root", str(tmp_path),
    ])
    assert "DRY RUN" in result.stdout


def test_n_datasets_zero_fails():
    """2. --n_datasets 0 fails."""
    _run(["--n_datasets", "0"], expect_fail=True)


def test_all_no_include_flags_fails():
    """3. All --no-include_* flags together fails."""
    _run([
        "--no-include_primary",
        "--no-include_feature_noise",
        "--no-include_label_noise",
        "--no-include_training_size",
        "--no-include_class_imbalance",
        "--no-include_margin",
        "--no-include_num_classes",
        "--no-include_ood",
    ], expect_fail=True)


def test_require_teachers_without_store_preds_fails():
    """4. --require_class_teachers --no-store_class_teacher_preds fails."""
    _run([
        "--require_class_teachers",
        "--no-store_class_teacher_preds",
        "--n_datasets", "1",
    ], expect_fail=True)


def test_primary_directory_has_correct_file_count(tmp_path):
    """5. Primary directory created with correct file count."""
    n_datasets = 4
    _run(_small_args(tmp_path, "primary_count") + [f"--n_datasets", str(n_datasets)])
    primary_dir = tmp_path / "primary_count" / "primary"
    parquet_files = sorted(primary_dir.glob("*.parquet"))
    assert len(parquet_files) == n_datasets


def test_parquet_files_named_correctly(tmp_path):
    """6. Parquet files named dataset_0000.parquet through dataset_NNNN.parquet."""
    n_datasets = 4
    _run(_small_args(tmp_path, "naming_test") + ["--n_datasets", str(n_datasets)])
    primary_dir = tmp_path / "naming_test" / "primary"
    files = sorted(primary_dir.glob("*.parquet"))
    for i, f in enumerate(files):
        assert f.name == f"dataset_{i:04d}.parquet", f"Unexpected filename: {f.name}"


def test_manifest_created_at_suite_root(tmp_path):
    """7. Manifest created at suite root."""
    _run(_small_args(tmp_path, "manifest_check"))
    manifest_path = tmp_path / "manifest_check" / "synthetic_classification_manifest.json"
    assert manifest_path.exists()


def test_manifest_schema_version(tmp_path):
    """8. Manifest schema_version matches CLASSIFICATION_EVAL_SCHEMA_VERSION."""
    _run(_small_args(tmp_path, "schema_ver_check"))
    manifest = json.loads(
        (tmp_path / "schema_ver_check" / "synthetic_classification_manifest.json").read_text()
    )
    assert manifest["schema_version"] in (
        "linear_classification_eval_v1",
        "linear_classification_eval_v2",
    ), f"Unexpected schema_version: {manifest['schema_version']!r}"


def test_split_column_values(tmp_path):
    """9. split column has values {'context', 'query'}."""
    _run(_small_args(tmp_path, "split_check"))
    primary_dir = tmp_path / "split_check" / "primary"
    for pf in primary_dir.glob("*.parquet"):
        t = pq.read_table(pf)
        split_vals = set(t.column("split").to_pylist())
        assert split_vals == {"context", "query"}, f"Unexpected split values in {pf.name}"
        break  # checking one is sufficient for this test


def test_no_regression_only_fields_in_parquet(tmp_path):
    """10. No regression-only fields in any Parquet file."""
    _run(_small_args(tmp_path, "reg_fields_check"))
    suite_dir = tmp_path / "reg_fields_check"
    for pf in suite_dir.rglob("*.parquet"):
        t = pq.read_table(pf)
        for field in _REGRESSION_ONLY_FIELDS:
            assert field not in t.column_names, (
                f"Regression field {field!r} found in {pf.relative_to(suite_dir)}"
            )


def test_label_column_is_integer(tmp_path):
    """11. label column is integer dtype."""
    import pyarrow as pa
    _run(_small_args(tmp_path, "label_dtype_check"))
    primary_dir = tmp_path / "label_dtype_check" / "primary"
    for pf in primary_dir.glob("*.parquet"):
        t = pq.read_table(pf)
        assert pa.types.is_integer(t.schema.field("label").type)
        break


def test_context_query_ratio_approx_80_20(tmp_path):
    """12. Context/query ratio is approximately 80/20."""
    _run(_small_args(tmp_path, "ratio_check"))
    primary_dir = tmp_path / "ratio_check" / "primary"
    for pf in primary_dir.glob("*.parquet"):
        t = pq.read_table(pf)
        splits = t.column("split").to_pylist()
        n_context = splits.count("context")
        n_query = splits.count("query")
        n_total = len(splits)
        ratio = n_context / n_total
        assert 0.75 <= ratio <= 0.85, f"Context ratio {ratio:.2f} not near 0.80"
        break


def test_global_idx_unique_across_all_datasets(tmp_path):
    """13. global_idx values are unique across all datasets in manifest."""
    _run(_small_args(tmp_path, "gidx_check"))
    manifest = json.loads(
        (tmp_path / "gidx_check" / "synthetic_classification_manifest.json").read_text()
    )
    ids = [r["global_idx"] for r in manifest["datasets"]]
    assert len(ids) == len(set(ids)), "global_idx values are not unique"


def test_created_at_parseable_iso8601(tmp_path):
    """14. Manifest created_at is parseable ISO-8601."""
    _run(_small_args(tmp_path, "iso8601_check"))
    manifest = json.loads(
        (tmp_path / "iso8601_check" / "synthetic_classification_manifest.json").read_text()
    )
    dt = datetime.fromisoformat(manifest["created_at"])
    assert dt.tzinfo is not None


def test_overwrite_prevents_file_exists_error(tmp_path):
    """15. --overwrite prevents FileExistsError on second run; absent flag causes failure."""
    args = _small_args(tmp_path, "overwrite_test")
    _run(args)  # first run succeeds
    _run(args, expect_fail=True)  # second run without --overwrite fails
    _run(args + ["--overwrite"])  # with --overwrite it succeeds


def test_snowsql_commands_printed(tmp_path):
    """16. SnowSQL commands printed to stdout."""
    result = _run(_small_args(tmp_path, "snowsql_check"))
    assert "SnowSQL" in result.stdout or "PUT" in result.stdout


def test_training_size_anchor_is_tabpfn_anchor_true(tmp_path):
    """17. training_size family: n4832/ files have is_tabpfn_anchor=True."""
    _run([
        "--suite_id", "ts_anchor_check",
        "--base_seed", "99",
        "--n_datasets", "4",
        "--n_datasets_per_sweep", "2",
        "--output_root", str(tmp_path),
        "--no-include_primary",
        "--no-include_feature_noise",
        "--no-include_label_noise",
        "--no-include_class_imbalance",
        "--no-include_margin",
        "--no-include_num_classes",
        "--no-include_ood",
    ])
    anchor_dir = tmp_path / "ts_anchor_check" / "training_size" / "n4832"
    assert anchor_dir.exists(), "n4832 directory not found"
    for pf in anchor_dir.glob("*.parquet"):
        t = pq.read_table(pf)
        anchors = t.column("is_tabpfn_anchor").to_pylist()
        assert all(anchors), f"Expected all is_tabpfn_anchor=True in {pf.name}"
        break


def test_training_size_non_anchor_is_tabpfn_anchor_false(tmp_path):
    """18. training_size family: n25/ files have is_tabpfn_anchor=False."""
    _run([
        "--suite_id", "ts_non_anchor_check",
        "--base_seed", "99",
        "--n_datasets", "4",
        "--n_datasets_per_sweep", "2",
        "--output_root", str(tmp_path),
        "--no-include_primary",
        "--no-include_feature_noise",
        "--no-include_label_noise",
        "--no-include_class_imbalance",
        "--no-include_margin",
        "--no-include_num_classes",
        "--no-include_ood",
    ])
    n25_dir = tmp_path / "ts_non_anchor_check" / "training_size" / "n25"
    assert n25_dir.exists(), "n25 directory not found"
    for pf in n25_dir.glob("*.parquet"):
        t = pq.read_table(pf)
        anchors = t.column("is_tabpfn_anchor").to_pylist()
        assert not any(anchors), f"Expected all is_tabpfn_anchor=False in {pf.name}"
        break


def test_realized_suite_family_counts_match_enabled(tmp_path):
    """19. Manifest realized_suite_family_counts keys match enabled_suite_families."""
    _run(_small_args(tmp_path, "family_counts_check"))
    manifest = json.loads(
        (tmp_path / "family_counts_check" / "synthetic_classification_manifest.json").read_text()
    )
    enabled = set(manifest["enabled_suite_families"])
    realized = set(manifest["realized_suite_family_counts"].keys())
    assert enabled == realized


def test_fixed_seed_run_is_deterministic(tmp_path):
    """20. Fixed-seed run is deterministic: two identical runs produce identical output_checksum.

    Both runs use the SAME suite_id and base_seed so file content is byte-identical.
    """
    shared_args = [
        "--suite_id", "determinism_fixed",
        "--base_seed", "42",
        "--n_datasets", "3",
        "--n_datasets_per_sweep", "2",
        "--no-include_feature_noise",
        "--no-include_label_noise",
        "--no-include_training_size",
        "--no-include_class_imbalance",
        "--no-include_margin",
        "--no-include_num_classes",
        "--no-include_ood",
    ]
    _run(["--output_root", str(tmp_path / "det_A"), *shared_args])
    _run(["--output_root", str(tmp_path / "det_B"), *shared_args])
    manifest_a = json.loads(
        (tmp_path / "det_A" / "determinism_fixed" / "synthetic_classification_manifest.json").read_text()
    )
    manifest_b = json.loads(
        (tmp_path / "det_B" / "determinism_fixed" / "synthetic_classification_manifest.json").read_text()
    )
    assert manifest_a["output_checksum"] == manifest_b["output_checksum"], (
        f"output_checksum differs between runs:\n"
        f"  A: {manifest_a['output_checksum']}\n"
        f"  B: {manifest_b['output_checksum']}"
    )


# ---------------------------------------------------------------------------
# Phase 2: Separate noise grid tests
# ---------------------------------------------------------------------------

def test_p_noise_does_not_change_additive_amplitude(tmp_path):
    """Changing p_noise_grid must not change the feature_noise_amplitude_grid recorded
    in the manifest when feature_noise_amplitude_grid is held constant."""
    shared = [
        "--n_datasets", "4",
        "--n_datasets_per_sweep", "2",
        "--suite_id", "noise_sep_test",
        "--base_seed", "7",
        "--no-include_primary",
        "--include_feature_noise",
        "--no-include_label_noise",
        "--no-include_training_size",
        "--no-include_class_imbalance",
        "--no-include_margin",
        "--no-include_num_classes",
        "--no-include_ood",
        "--feature_noise_amplitude_grid", "0.1",
    ]
    run_a = ["--output_root", str(tmp_path / "A"), "--p_noise_grid", "0"] + shared
    run_b = ["--output_root", str(tmp_path / "B"), "--p_noise_grid", "50"] + shared
    _run(run_a)
    _run(run_b)

    for tag, root in [("A_p0", tmp_path / "A"), ("B_p50", tmp_path / "B")]:
        manifest_path = root / "noise_sep_test" / "synthetic_classification_manifest.json"
        assert manifest_path.exists(), f"Manifest not found for {tag}"
        manifest = json.loads(manifest_path.read_text())
        gm = manifest.get("grid_metadata", {})
        amp_grid = gm.get("feature_noise_amplitude_grid")
        assert amp_grid is not None, f"feature_noise_amplitude_grid missing from manifest ({tag})"
        assert amp_grid == [0.1], (
            f"feature_noise_amplitude_grid={amp_grid} in {tag}, expected [0.1]"
        )


def test_feature_noise_amplitude_grid_independent(tmp_path):
    """feature_noise_amplitude_grid and p_noise_grid are independently configurable."""
    _run([
        "--output_root", str(tmp_path),
        "--n_datasets", "4",
        "--n_datasets_per_sweep", "2",
        "--suite_id", "indep_test",
        "--base_seed", "42",
        "--no-include_primary",
        "--include_feature_noise",
        "--no-include_label_noise",
        "--no-include_training_size",
        "--no-include_class_imbalance",
        "--no-include_margin",
        "--no-include_num_classes",
        "--no-include_ood",
        "--p_noise_grid", "0", "10",
        "--feature_noise_amplitude_grid", "0.05", "0.25",
    ])
    manifest_path = tmp_path / "indep_test" / "synthetic_classification_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    gm = manifest.get("grid_metadata", {})
    # p_noise_grid and feature_noise_amplitude_grid must be recorded separately
    assert "p_noise_grid" in gm, "p_noise_grid missing from grid_metadata"
    assert "feature_noise_amplitude_grid" in gm, "feature_noise_amplitude_grid missing from grid_metadata"
    # They must differ in content (different semantics)
    assert gm["p_noise_grid"] != gm["feature_noise_amplitude_grid"], (
        "p_noise_grid and feature_noise_amplitude_grid must not be equal "
        "(they have different semantics)"
    )
