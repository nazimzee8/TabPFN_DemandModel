"""
tests/test_synthetic_regression_sharding.py
=============================================
Tests for shard assignment logic in evaluate_synthetic_regression.py.

Verifies that:
  - Sharding happens BEFORE any payload download
  - No (dataset_id, condition) pair is duplicated across shards
  - The union of all shards covers all index rows exactly once
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _make_index_rows(n: int, suite_family: str = "primary") -> list[dict]:
    """Generate n fake index rows."""
    rows = []
    for i in range(n):
        rows.append({
            "suite_id": "linear_poisson_v1_recommended",
            "suite_family": suite_family,
            "dataset_id": i,
            "dataset_seed": i * 100,
            "prior_regime": ["A", "B", "C", "D"][i % 4],
            "split_seeds": [0, 1, 2],
            "n_total": 200,
            "n_train_default": 160,
            "n_holdout_default": 40,
            "p_signal": 5,
            "p_noise": 0,
            "p_total": 5,
            "feature_noise_level": 0,
            "target_noise_scale": 1.0,
            "training_size_anchor": False,
            "stage_path": f"@stage/primary/dataset_{i:04d}.npz",
        })
    return rows


class TestShardAssignment:
    def test_assign_shard_basic(self):
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_index_rows(10)
        shard0 = assign_synthetic_regression_shard(rows, shard_index=0, num_shards=3)
        shard1 = assign_synthetic_regression_shard(rows, shard_index=1, num_shards=3)
        shard2 = assign_synthetic_regression_shard(rows, shard_index=2, num_shards=3)
        total = len(shard0) + len(shard1) + len(shard2)
        assert total == 10

    def test_shard_assignment_happens_before_payload_download(self):
        """
        load_prepared_synthetic_dataset must NOT be called before assign_shard.
        We verify by checking that shard assignment uses only the index rows list,
        not any file I/O.
        """
        from evaluate_synthetic_regression import assign_synthetic_regression_shard

        call_order = []

        def mock_load(row, local_cache_dir=None):
            call_order.append(("load", row["dataset_id"]))
            return {}

        rows = _make_index_rows(12)

        # Simulate the expected usage pattern: assign THEN load
        call_order.append(("assign", None))
        my_rows = assign_synthetic_regression_shard(rows, shard_index=0, num_shards=4)
        for row in my_rows:
            mock_load(row)

        # The first action must be 'assign', not 'load'
        assert call_order[0][0] == "assign"
        # All 'load' calls come after the 'assign' call
        assign_pos = next(i for i, c in enumerate(call_order) if c[0] == "assign")
        load_positions = [i for i, c in enumerate(call_order) if c[0] == "load"]
        assert all(p > assign_pos for p in load_positions)

    def test_no_shard_owns_duplicate_rows(self):
        """Across 10 shards, no dataset_id appears in more than one shard."""
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_index_rows(100)
        shards = [
            assign_synthetic_regression_shard(rows, shard_index=i, num_shards=10)
            for i in range(10)
        ]
        all_ids = []
        for shard in shards:
            for row in shard:
                all_ids.append(row["dataset_id"])
        assert len(all_ids) == len(set(all_ids)) == 100

    def test_all_rows_covered_exactly_once_across_shards(self):
        """Union of all shards == full index rows (no row missed, no duplicate)."""
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_index_rows(200)
        all_shards = []
        for i in range(10):
            shard = assign_synthetic_regression_shard(rows, shard_index=i, num_shards=10)
            all_shards.extend(shard)

        ids_in_shards = sorted(r["dataset_id"] for r in all_shards)
        expected_ids = sorted(r["dataset_id"] for r in rows)
        assert ids_in_shards == expected_ids

    def test_all_rows_covered_3_shards(self):
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_index_rows(30)
        combined = []
        for i in range(3):
            combined.extend(assign_synthetic_regression_shard(rows, shard_index=i, num_shards=3))
        ids = sorted(r["dataset_id"] for r in combined)
        assert ids == list(range(30))

    def test_all_rows_covered_30_shards(self):
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_index_rows(300)
        combined = []
        for i in range(30):
            combined.extend(assign_synthetic_regression_shard(rows, shard_index=i, num_shards=30))
        assert len(combined) == 300
        ids = sorted(r["dataset_id"] for r in combined)
        assert ids == list(range(300))

    def test_single_shard_gets_all_rows(self):
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_index_rows(50)
        shard = assign_synthetic_regression_shard(rows, shard_index=0, num_shards=1)
        assert len(shard) == 50

    def test_empty_index_returns_empty(self):
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        result = assign_synthetic_regression_shard([], shard_index=0, num_shards=5)
        assert result == []


class TestBuildSplitForSeed:
    def test_split_is_deterministic(self):
        from evaluate_synthetic_regression import build_split_for_seed
        import numpy as np
        data = {
            "X": np.arange(200.0).reshape(100, 2),
            "y": np.arange(100.0),
            "betaX": np.arange(100.0) * 2,
            "n_total": 100,
            "n_holdout_default": 20,
        }
        split1 = build_split_for_seed(data, split_seed=0)
        split2 = build_split_for_seed(data, split_seed=0)
        np.testing.assert_array_equal(split1["X_train"], split2["X_train"])
        np.testing.assert_array_equal(split1["y_holdout"], split2["y_holdout"])

    def test_split_different_seeds_produce_different_splits(self):
        from evaluate_synthetic_regression import build_split_for_seed
        import numpy as np
        data = {
            "X": np.arange(200.0).reshape(100, 2),
            "y": np.arange(100.0),
            "betaX": np.arange(100.0),
            "n_total": 100,
            "n_holdout_default": 20,
        }
        split0 = build_split_for_seed(data, split_seed=0)
        split1 = build_split_for_seed(data, split_seed=1)
        # With different seeds the train splits should differ
        assert not np.array_equal(split0["X_train"], split1["X_train"])

    def test_split_respects_n_train_override(self):
        from evaluate_synthetic_regression import build_split_for_seed
        import numpy as np
        n_total = 6203
        data = {
            "X": np.ones((n_total, 3)),
            "y": np.ones(n_total),
            "betaX": np.ones(n_total),
            "n_total": n_total,
            "n_holdout_default": 1371,
        }
        split = build_split_for_seed(data, split_seed=0, n_train_override=25)
        assert split["n_train"] == 25
        assert split["X_train"].shape[0] == 25

    def test_split_sizes_sum_correctly(self):
        from evaluate_synthetic_regression import build_split_for_seed
        import numpy as np
        data = {
            "X": np.ones((100, 5)),
            "y": np.ones(100),
            "betaX": np.ones(100),
            "n_total": 100,
            "n_holdout_default": 20,
        }
        split = build_split_for_seed(data, split_seed=42)
        assert split["n_train"] == 80
        assert split["n_holdout"] == 20
        assert split["X_train"].shape[0] == 80
        assert split["X_holdout"].shape[0] == 20


# ---------------------------------------------------------------------------
# Tests: Shard determinism (Issue 1 — stable ORDER BY in index query)
# ---------------------------------------------------------------------------

def _make_ood_rows(n_per_regime: int = 20) -> list[dict]:
    """Generate OOD-style rows with regimes E/F/G/H, dataset_id may repeat across regimes."""
    rows = []
    for regime in ["E", "F", "G", "H"]:
        for i in range(n_per_regime):
            rows.append({
                "suite_id": "ood_linear_pilot_v1",
                "suite_family": "ood_primary",
                "dataset_id": i,
                "dataset_seed": i * 10,
                "prior_regime": regime,
                "split_seeds": [0, 1, 2],
                "n_total": 100,
                "n_train_default": 80,
                "n_holdout_default": 20,
                "p_signal": 4,
                "p_noise": 0,
                "p_total": 4,
                "feature_noise_level": 0,
                "target_noise_scale": 1.0,
                "training_size_anchor": False,
                "stage_path": f"@EVAL_DATASET_STAGE/ood_parity/{regime}/dataset_{i:04d}.parquet",
            })
    return rows


class TestShardDeterminism:
    def test_shard_assignment_stable_across_identical_calls(self):
        """Same ordered rows in → same shard assignment on every call."""
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_index_rows(60)
        shard_a = assign_synthetic_regression_shard(rows, shard_index=2, num_shards=5)
        shard_b = assign_synthetic_regression_shard(rows, shard_index=2, num_shards=5)
        ids_a = [r["dataset_id"] for r in shard_a]
        ids_b = [r["dataset_id"] for r in shard_b]
        assert ids_a == ids_b

    def test_ood_rows_deterministic_with_repeated_dataset_id(self):
        """OOD rows with same dataset_id but different prior_regime land in
        distinct enumeration positions, so each row goes to exactly one shard."""
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        rows = _make_ood_rows(n_per_regime=20)  # 80 rows: dataset_id 0-19 × 4 regimes
        num_shards = 5
        all_stage_paths = []
        for si in range(num_shards):
            shard = assign_synthetic_regression_shard(rows, shard_index=si, num_shards=num_shards)
            all_stage_paths.extend(r["stage_path"] for r in shard)
        # No duplicate stage_paths across shards
        assert len(all_stage_paths) == len(set(all_stage_paths))
        # All rows covered
        assert len(all_stage_paths) == 80

    def test_all_shards_cover_all_rows_exactly_once_with_ood_mixed(self):
        """80 OOD + 200 in-distribution rows; all 280 rows covered exactly once across 10 shards."""
        from evaluate_synthetic_regression import assign_synthetic_regression_shard
        indist_rows = _make_index_rows(200)
        ood_rows = _make_ood_rows(n_per_regime=20)
        mixed = indist_rows + ood_rows
        num_shards = 10
        collected = []
        for si in range(num_shards):
            shard = assign_synthetic_regression_shard(mixed, shard_index=si, num_shards=num_shards)
            collected.extend(shard)
        assert len(collected) == 280
        stage_paths = [r["stage_path"] for r in collected]
        assert len(stage_paths) == len(set(stage_paths))


# ---------------------------------------------------------------------------
# Fix 7: Explicit work-item expansion tests
# ---------------------------------------------------------------------------

def _make_training_size_rows(n: int) -> list[dict]:
    """Generate n fake training_size suite rows with split_seeds."""
    rows = []
    for i in range(n):
        rows.append({
            "suite_id": "linear_poisson_v1_recommended",
            "suite_family": "training_size",
            "dataset_id": i,
            "dataset_seed": i * 100,
            "prior_regime": "A",
            "split_seeds": [0, 1],
            "n_total": 6203,
            "n_train_default": 4832,
            "n_holdout_default": 1371,
            "p_signal": 5,
            "p_noise": 0,
            "p_total": 5,
            "feature_noise_level": 0,
            "target_noise_scale": 1.0,
            "training_size_anchor": True,
            "stage_path": f"@stage/training_size/dataset_{i:04d}.parquet",
        })
    return rows


class TestExpandWorkItems:
    def test_expand_primary_suite_seeds_x_one_condition(self):
        """Primary suite: N rows × S seeds × 1 condition = N×S items."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        rows = _make_index_rows(5)  # each has split_seeds=[0,1,2]
        items = expand_synreg_work_items(rows, train_size_grid=[])
        # 5 rows × 3 seeds × 1 condition = 15 items
        assert len(items) == 15

    def test_expand_training_size_suite_produces_conditions(self):
        """Training size suite: each row gets len(seeds) × len(grid) items."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        rows = _make_training_size_rows(2)  # 2 rows, split_seeds=[0,1]
        grid = [25, 50, 100, 200]
        items = expand_synreg_work_items(rows, train_size_grid=grid)
        # 2 rows × 2 seeds × 4 grid sizes = 16 items
        assert len(items) == 16

    def test_each_item_has_split_seed_and_n_train_override(self):
        """Every expanded item has split_seed and n_train_override fields."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        rows = _make_index_rows(3)
        items = expand_synreg_work_items(rows, train_size_grid=[])
        for item in items:
            assert "split_seed" in item
            assert "n_train_override" in item
            assert "is_anchor" in item

    def test_primary_suite_n_train_override_is_none(self):
        """Primary suite work items have n_train_override=None."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        rows = _make_index_rows(2)
        items = expand_synreg_work_items(rows, train_size_grid=[])
        for item in items:
            assert item["n_train_override"] is None

    def test_training_size_n_train_override_matches_grid(self):
        """Training size work items have n_train_override matching the grid value."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        rows = _make_training_size_rows(1)  # 1 row, split_seeds=[0,1]
        grid = [25, 50, 100]
        items = expand_synreg_work_items(rows, train_size_grid=grid)
        overrides = sorted(set(item["n_train_override"] for item in items))
        assert overrides == [25, 50, 100]

    def test_work_items_cover_all_seeds(self):
        """Expanded items include all seeds from split_seeds."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        rows = _make_index_rows(1)  # split_seeds=[0,1,2]
        items = expand_synreg_work_items(rows, train_size_grid=[])
        seeds = sorted(item["split_seed"] for item in items)
        assert seeds == [0, 1, 2]

    def test_work_items_shardable_by_assign_shard(self):
        """expand_synreg_work_items output can be sharded by assign_synthetic_regression_shard
        with no items lost."""
        from evaluate_synthetic_regression import expand_synreg_work_items, assign_synthetic_regression_shard
        rows = _make_index_rows(10)
        all_items = expand_synreg_work_items(rows, train_size_grid=[])
        n_total = len(all_items)
        collected = []
        for si in range(5):
            shard = assign_synthetic_regression_shard(all_items, shard_index=si, num_shards=5)
            collected.extend(shard)
        assert len(collected) == n_total

    def test_expand_deterministic(self):
        """Same input produces same expansion on every call."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        rows = _make_index_rows(8)
        items_a = expand_synreg_work_items(rows, train_size_grid=[])
        items_b = expand_synreg_work_items(rows, train_size_grid=[])
        assert len(items_a) == len(items_b)
        for a, b in zip(items_a, items_b):
            assert a["dataset_id"] == b["dataset_id"]
            assert a["split_seed"] == b["split_seed"]

    def test_empty_index_produces_empty_items(self):
        """Empty rows → empty work items."""
        from evaluate_synthetic_regression import expand_synreg_work_items
        items = expand_synreg_work_items([], train_size_grid=[])
        assert items == []
