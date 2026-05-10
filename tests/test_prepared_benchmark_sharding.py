import os
import sys
import types
import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import evaluate  # noqa: E402
import prepare_benchmark_datasets as prepare  # noqa: E402

EVALUATE_PATH = os.path.join(SRC_DIR, "evaluate.py")


class _LoadedDataset(dict):
    def __init__(self, payload, tracker):
        super().__init__(payload)
        self._tracker = tracker

    def __del__(self):
        self._tracker["current"] -= 1


class _MeanRegressor:
    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._mean)


def _manifest(n_datasets):
    return {
        "datasets": [
            {
                "dataset_index": i,
                "source": "unit",
                "task_id": f"task-{i}",
                "dataset_id": f"dataset-{i}",
                "name": f"dataset-{i}",
                "stage_path": f"@stage/dataset-{i}.npz",
                "n_samples": 20,
                "n_features": 3,
                "estimated_bytes": 1000,
                "benchmark_weight": 60,
                "categorical_indicator": [False, False, False],
            }
            for i in range(n_datasets)
        ]
    }


def _index_rows_from_manifest(manifest):
    rows = []
    for ds in manifest["datasets"]:
        rows.append(
            {
                "dataset_index": ds["dataset_index"],
                "source": ds["source"],
                "task_id": ds["task_id"],
                "dataset_id": ds["dataset_id"],
                "name": ds["name"],
                "stage_path": ds["stage_path"],
                "n_samples": ds["n_samples"],
                "n_features": ds["n_features"],
                "benchmark_weight": ds["benchmark_weight"],
            }
        )
    return rows


def _patch_fast_benchmark(monkeypatch, n_datasets):
    tracker = {"current": 0, "peak": 0, "loaded": [], "seeds": []}

    def fake_load_manifest(stage_path=None):
        return _manifest(n_datasets)

    def fake_load_dataset(ds_meta):
        tracker["current"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["current"])
        tracker["loaded"].append(int(ds_meta["dataset_index"]))
        base = int(ds_meta["dataset_index"])
        X = np.arange(60, dtype=float).reshape(20, 3) + base
        y = np.arange(20, dtype=float) + base
        return _LoadedDataset(
            {
                "task_id": str(ds_meta["task_id"]),
                "name": str(ds_meta["name"]),
                "X": X,
                "y": y,
                "categorical_indicator": [False, False, False],
                "source": "unit",
            },
            tracker,
        )

    def fake_split(X, y, test_size, random_state):
        tracker["seeds"].append(int(random_state))
        return X[:16], X[16:], y[:16], y[16:]

    monkeypatch.setattr(evaluate, "load_prepared_benchmark_manifest", fake_load_manifest)
    monkeypatch.setattr(evaluate, "load_prepared_dataset", fake_load_dataset)
    monkeypatch.setattr(evaluate, "train_test_split", fake_split)
    monkeypatch.setattr(evaluate, "preprocess_split", lambda X_train, X_test, cat: (X_train, X_test))
    monkeypatch.setattr(evaluate, "get_baselines", lambda seed: {"Ridge": _MeanRegressor()})
    monkeypatch.setattr(
        evaluate,
        "compute_metrics",
        lambda y_true, y_pred: {"mse": 1.0, "rmse": 1.0, "r2": 0.0},
    )
    return tracker


def test_dataset_first_shard_loads_one_dataset_at_a_time(monkeypatch):
    tracker = _patch_fast_benchmark(monkeypatch, n_datasets=6)

    detailed_df, _, _ = evaluate.run_prepared_benchmark(
        model=None,
        seeds=[0, 1, 2],
        methods=["Ridge"],
        rank=0,
        world_size=2,
    )

    assert tracker["loaded"] == [0, 2, 4]
    assert tracker["peak"] == 1
    assert tracker["seeds"] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert len(detailed_df) == 3 * 3 * 1


def test_benchmark_methods_parses_comma_separated_list():
    assert evaluate.parse_benchmark_methods(
        benchmark_methods="Ridge, LinearRegression,SVR"
    ) == ["Ridge", "LinearRegression", "SVR"]


def test_benchmark_method_single_fallback():
    assert evaluate.parse_benchmark_methods(benchmark_method="Ridge") == ["Ridge"]


def test_benchmark_method_and_methods_conflict_fails_fast():
    with pytest.raises(ValueError, match="Set only one of BENCHMARK_METHOD or BENCHMARK_METHODS"):
        evaluate.parse_benchmark_methods(
            benchmark_method="Ridge",
            benchmark_methods="Ridge,SVR",
        )


def test_multi_method_shard_writes_one_part_csv_per_method(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_SHARD_INDEX", "1")
    monkeypatch.setenv("BENCHMARK_NUM_SHARDS", "3")
    detailed_df = pd.DataFrame(
        [
            {"source": "unit", "task_id": "task-0", "name": "dataset-0", "rep": 0, "method": "Ridge", "mse": 1.0, "rmse": 1.0, "r2": 0.0},
            {"source": "unit", "task_id": "task-0", "name": "dataset-0", "rep": 0, "method": "SVR", "mse": 2.0, "rmse": 1.4, "r2": -1.0},
        ]
    )

    paths = evaluate.save_benchmark_part_csvs(detailed_df, ["Ridge", "SVR"], str(tmp_path))

    assert [os.path.basename(path) for path in paths] == [
        "Ridge_shard1_of_3_detailed.csv",
        "SVR_shard1_of_3_detailed.csv",
    ]
    assert pd.read_csv(paths[0])["method"].tolist() == ["Ridge"]
    assert pd.read_csv(paths[1])["method"].tolist() == ["SVR"]


def test_dataset_shards_are_disjoint(monkeypatch):
    _patch_fast_benchmark(monkeypatch, n_datasets=5)

    shard_0 = evaluate.select_benchmark_dataset_indices(
        _manifest(5)["datasets"], shard_index=0, num_shards=2, strategy="modulo"
    )
    shard_1 = evaluate.select_benchmark_dataset_indices(
        _manifest(5)["datasets"], shard_index=1, num_shards=2, strategy="modulo"
    )

    assert set(shard_0).isdisjoint(shard_1)
    assert sorted(shard_0 + shard_1) == [0, 1, 2, 3, 4]


def test_large_assigned_dataset_count_keeps_peak_loaded_at_one(monkeypatch):
    tracker = _patch_fast_benchmark(monkeypatch, n_datasets=25)

    detailed_df, _, _ = evaluate.run_prepared_benchmark(
        model=None,
        seeds=[7],
        methods=["Ridge"],
        rank=0,
        world_size=1,
    )

    assert tracker["loaded"] == list(range(25))
    assert tracker["peak"] == 1
    assert len(detailed_df) == 25


def test_balanced_assignment_uses_benchmark_weight(monkeypatch):
    manifest = _manifest(4)
    rows = _index_rows_from_manifest(manifest)
    for row, weight in zip(rows, [100, 90, 10, 10]):
        row["benchmark_weight"] = weight
    monkeypatch.setattr(evaluate, "_query_benchmark_index_rows", lambda: rows)

    datasets_meta = manifest["datasets"]
    shard_0 = evaluate.select_benchmark_dataset_indices(
        datasets_meta, shard_index=0, num_shards=2, strategy="balanced"
    )
    shard_1 = evaluate.select_benchmark_dataset_indices(
        datasets_meta, shard_index=1, num_shards=2, strategy="balanced"
    )

    assert set(shard_0).isdisjoint(shard_1)
    assert sorted(shard_0 + shard_1) == [0, 1, 2, 3]


def test_balanced_assignment_covers_all_manifest_datasets_once(monkeypatch):
    manifest = _manifest(7)
    rows = _index_rows_from_manifest(manifest)
    for row, weight in zip(rows, [90, 80, 70, 60, 50, 40, 30]):
        row["benchmark_weight"] = weight
    monkeypatch.setattr(evaluate, "_query_benchmark_index_rows", lambda: rows)

    shards = [
        evaluate.select_benchmark_dataset_indices(
            manifest["datasets"], shard_index=i, num_shards=3, strategy="balanced"
        )
        for i in range(3)
    ]

    flattened = [idx for shard in shards for idx in shard]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(7))


def test_balanced_assignment_fails_when_index_missing_manifest_dataset(monkeypatch):
    manifest = _manifest(3)
    rows = _index_rows_from_manifest(manifest)[:2]
    monkeypatch.setattr(evaluate, "_query_benchmark_index_rows", lambda: rows)

    with pytest.raises(RuntimeError, match="missing manifest dataset_index"):
        evaluate.select_benchmark_dataset_indices(
            manifest["datasets"], shard_index=0, num_shards=2, strategy="balanced"
        )


def test_balanced_assignment_fails_on_duplicate_index_rows(monkeypatch):
    manifest = _manifest(2)
    rows = _index_rows_from_manifest(manifest)
    rows.append(dict(rows[0]))
    monkeypatch.setattr(evaluate, "_query_benchmark_index_rows", lambda: rows)

    with pytest.raises(RuntimeError, match="duplicate dataset_index"):
        evaluate.select_benchmark_dataset_indices(
            manifest["datasets"], shard_index=0, num_shards=2, strategy="balanced"
        )


def test_modulo_assignment_does_not_query_benchmark_index(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "_query_benchmark_index_rows",
        lambda: pytest.fail("modulo sharding should not query BENCHMARK_DATASET_INDEX"),
    )

    assert evaluate.select_benchmark_dataset_indices(
        _manifest(4)["datasets"], shard_index=1, num_shards=2, strategy="modulo"
    ) == [1, 3]


def test_old_manifest_is_backfilled_without_rebuilding_datasets():
    manifest = {
        "datasets": [
            {
                "dataset_index": 0,
                "source": "unit",
                "task_id": "task-0",
                "dataset_id": "dataset-0",
                "name": "dataset-0",
                "stage_path": "@stage/dataset-0.npz",
                "n_samples": 20,
                "n_features": 3,
                "categorical_indicator": [False, False, False],
            }
        ]
    }

    updated, changed = prepare._validate_and_backfill_existing_manifest(manifest)

    assert changed is True
    assert updated["created_at_utc"]
    assert updated["datasets"][0]["estimated_bytes"] is None
    assert updated["datasets"][0]["benchmark_weight"] == 60
    assert updated["datasets"][0]["created_at_utc"] == updated["created_at_utc"]


def test_invalid_manifest_missing_payload_field_triggers_rebuild_path():
    manifest = {
        "created_at_utc": "2026-01-01T00:00:00Z",
        "datasets": [
            {
                "dataset_index": 0,
                "source": "unit",
                "task_id": "task-0",
                "name": "dataset-0",
                "n_samples": 20,
                "n_features": 3,
                "categorical_indicator": [False, False, False],
            }
        ],
    }

    with pytest.raises(ValueError, match="stage_path"):
        prepare._validate_and_backfill_existing_manifest(manifest)


def test_benchmark_num_cpus_defaults_to_one(monkeypatch):
    monkeypatch.delenv("BENCHMARK_NUM_CPUS", raising=False)
    spec = importlib.util.spec_from_file_location("evaluate_default_cpu_under_test", EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.BENCHMARK_NUM_CPUS == 1


def test_autogluon_fit_receives_configured_benchmark_num_cpus(tmp_path, monkeypatch):
    fit_kwargs = {}

    class _FakePredictions:
        def to_numpy(self):
            return np.array([1.0, 2.0])

    class _FakePredictor:
        def __init__(self, **kwargs):
            pass

        def fit(self, train_df, **kwargs):
            fit_kwargs.update(kwargs)
            return self

        def predict(self, test_df):
            return _FakePredictions()

    autogluon = types.ModuleType("autogluon")
    tabular = types.ModuleType("autogluon.tabular")
    tabular.TabularPredictor = _FakePredictor
    monkeypatch.setitem(sys.modules, "autogluon", autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)
    monkeypatch.setattr(evaluate, "BENCHMARK_NUM_CPUS", 6)
    monkeypatch.setattr(evaluate.tempfile, "mkdtemp", lambda prefix, dir: str(tmp_path / "ag"))

    preds = evaluate.predict_autogluon(
        np.ones((4, 2)),
        np.arange(4, dtype=float),
        np.ones((2, 2)),
    )

    assert preds.tolist() == [1.0, 2.0]
    assert fit_kwargs["num_cpus"] == 6


def test_deepset_benchmark_uses_train_only_contexts_and_averages_predictions(monkeypatch):
    manifest = _manifest(1)
    calls = {"preprocess": [], "contexts": [], "metrics": []}
    stream_outputs = [1.0, 2.0, 4.0, 8.0, 16.0]
    context_indices = [
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([4, 5]),
        np.array([6, 7]),
        np.array([8, 9]),
    ]

    def fake_load_dataset(ds_meta):
        X = np.arange(60, dtype=float).reshape(20, 3)
        y = np.arange(20, dtype=float)
        return {
            "task_id": "task-0",
            "name": "dataset-0",
            "X": X,
            "y": y,
            "categorical_indicator": [False, False, False],
            "source": "unit",
        }

    def fake_split(X, y, test_size, random_state):
        assert random_state == 7
        return X[:18], X[18:], y[:18], y[18:]

    def fake_preprocess(X_train, X_test, categorical_indicator):
        calls["preprocess"].append((X_train.copy(), X_test.copy()))
        return X_train + 1000.0, X_test + 2000.0

    def fake_select(n_train, context_size, seed, context_index):
        assert n_train == 18
        assert context_size == 200
        assert seed == 7
        return context_indices[context_index]

    def fake_stream(model, X_train_np, y_train_np, X_test_np, K, test_batch_size, device):
        call_index = len(calls["contexts"])
        expected_idx = context_indices[call_index]
        np.testing.assert_array_equal(X_train_np, np.arange(60, dtype=float).reshape(20, 3)[:18][expected_idx] + 1000.0)
        np.testing.assert_array_equal(y_train_np, np.arange(20, dtype=float)[:18][expected_idx])
        np.testing.assert_array_equal(X_test_np, np.arange(60, dtype=float).reshape(20, 3)[18:] + 2000.0)
        assert K == 3
        assert test_batch_size == 128
        assert device == "cpu"
        calls["contexts"].append(X_train_np.shape[0])
        return np.full(X_test_np.shape[0], stream_outputs[call_index])

    def fake_metrics(y_true, y_pred):
        calls["metrics"].append((y_true.copy(), y_pred.copy()))
        np.testing.assert_array_equal(y_true, np.array([18.0, 19.0]))
        np.testing.assert_allclose(y_pred, np.full(2, np.mean(stream_outputs)))
        return {"mse": 1.0, "rmse": 1.0, "r2": 0.0}

    monkeypatch.setattr(evaluate, "load_prepared_benchmark_manifest", lambda stage_path=None: manifest)
    monkeypatch.setattr(evaluate, "load_prepared_dataset", fake_load_dataset)
    monkeypatch.setattr(evaluate, "train_test_split", fake_split)
    monkeypatch.setattr(evaluate, "preprocess_split", fake_preprocess)
    monkeypatch.setattr(evaluate, "select_deepset_context_indices", fake_select)
    monkeypatch.setattr(evaluate, "predict_deepset_mc_streamed", fake_stream)
    monkeypatch.setattr(evaluate, "deepset_inference_device", lambda: "cpu")
    monkeypatch.setattr(evaluate, "compute_metrics", fake_metrics)

    detailed_df, _, _ = evaluate.run_prepared_benchmark(
        model=object(),
        seeds=[7],
        methods=["DeepSetModel-MC"],
        rank=0,
        world_size=1,
        mc_K=3,
    )

    assert calls["contexts"] == [2, 2, 2, 2, 2]
    assert len(calls["preprocess"]) == 1
    assert len(calls["metrics"]) == 1
    assert detailed_df["method"].tolist() == ["DeepSetModel-MC"]


def test_deepset_context_selection_is_deterministic_and_bounded_to_train_rows():
    first = evaluate.select_deepset_context_indices(
        n_train=300, context_size=200, seed=11, context_index=2
    )
    second = evaluate.select_deepset_context_indices(
        n_train=300, context_size=200, seed=11, context_index=2
    )
    other = evaluate.select_deepset_context_indices(
        n_train=300, context_size=200, seed=11, context_index=3
    )

    assert len(first) == 200
    assert first.min() >= 0
    assert first.max() < 300
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_deepset_streamed_mc_chunking_preserves_test_order_and_length():
    class EchoModel(torch.nn.Module):
        def forward(self, X_train, y_train, X_test):
            return X_test[:, 0]

    X_train = np.ones((4, 2), dtype=float)
    y_train = np.arange(4, dtype=float)
    X_test = np.column_stack([np.arange(7, dtype=float), np.ones(7)])

    preds = evaluate.predict_deepset_mc_streamed(
        EchoModel(),
        X_train,
        y_train,
        X_test,
        K=3,
        test_batch_size=2,
        device=torch.device("cpu"),
    )

    np.testing.assert_array_equal(preds, np.arange(7, dtype=float))


def test_baselines_still_receive_full_processed_training_split(monkeypatch):
    tracker = _patch_fast_benchmark(monkeypatch, n_datasets=1)
    fit_shapes = []

    class ShapeRegressor:
        def fit(self, X, y):
            fit_shapes.append((X.shape, y.shape))
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

    monkeypatch.setattr(evaluate, "get_baselines", lambda seed: {"Ridge": ShapeRegressor()})

    evaluate.run_prepared_benchmark(
        model=None,
        seeds=[5],
        methods=["Ridge"],
        rank=0,
        world_size=1,
    )

    assert tracker["seeds"] == [5]
    assert fit_shapes == [((16, 3), (16,))]


def test_autogluon_still_receives_full_processed_training_split(monkeypatch):
    tracker = _patch_fast_benchmark(monkeypatch, n_datasets=1)
    ag_shapes = []

    def fake_autogluon(X_train_np, y_train_np, X_test_np):
        ag_shapes.append((X_train_np.shape, y_train_np.shape, X_test_np.shape))
        return np.zeros(X_test_np.shape[0])

    monkeypatch.setattr(evaluate, "predict_autogluon", fake_autogluon)

    evaluate.run_prepared_benchmark(
        model=None,
        seeds=[6],
        methods=["AutoGluon"],
        rank=0,
        world_size=1,
    )

    assert tracker["seeds"] == [6]
    assert ag_shapes == [((16, 3), (16,), (4, 3))]
