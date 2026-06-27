from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classification import compute_classification_losses  # noqa: E402
from model import DeepSetICLModel, ModelConfig  # noqa: E402
from task_routing import get_training_data_spec  # noqa: E402
from train import load_classification_parquet  # noqa: E402


def _classification_model(max_num_classes: int = 10) -> DeepSetICLModel:
    cfg = ModelConfig(
        d_phi=16,
        d_rho=16,
        pool="mean",
        n_heads=4,
        n_sab_feat=1,
        dropout=0.0,
        task_objective="inductive_classification",
        use_classification_path=True,
        max_num_classes=max_num_classes,
        class_embedding_dim=8,
        class_stat_dim=16,
        class_stat_hidden_dim=24,
        class_head_hidden_dim=12,
        class_fusion_gate_hidden_dim=12,
        linear_stat_dim=16,
        linear_encoder_hidden_dim=24,
        coeff_head_hidden_dim=12,
        lambda_head_hidden_dim=8,
    )
    return DeepSetICLModel(cfg).eval()


@pytest.mark.parametrize("num_classes", [2, 3, 5, 10])
@pytest.mark.parametrize("num_features", [3, 7])
def test_classification_shapes_are_multiclass_compatible(num_classes, num_features):
    torch.manual_seed(3)
    model = _classification_model()
    X_train = torch.randn(20, num_features)
    y_train = torch.arange(20) % num_classes
    X_test = torch.randn(4, num_features)

    output = model(
        X_train,
        y_train,
        X_test,
        task_objective="inductive_classification",
        num_classes=num_classes,
    )

    assert output["logits"].shape == (4, num_classes)
    assert output["probs"].shape == (4, num_classes)
    assert output["pred"].shape == (4,)
    assert output["W_hat_norm"].shape == (num_features, num_classes)
    assert output["b_hat"].shape == (num_classes,)
    assert torch.isfinite(output["logits"]).all()
    torch.testing.assert_close(
        output["probs"].sum(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6
    )


def test_classification_row_invariance_and_column_consistency():
    torch.manual_seed(7)
    model = _classification_model()
    X_train = torch.randn(18, 7)
    y_train = torch.arange(18) % 3
    X_test = torch.randn(5, 7)
    reference = model.forward_classification(X_train, y_train, X_test, 3)["logits"]

    row_order = torch.randperm(X_train.shape[0])
    row_output = model.forward_classification(
        X_train[row_order], y_train[row_order], X_test, 3
    )["logits"]
    column_order = torch.randperm(X_train.shape[1])
    column_output = model.forward_classification(
        X_train[:, column_order], y_train, X_test[:, column_order], 3
    )["logits"]

    torch.testing.assert_close(reference, row_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reference, column_output, atol=1e-5, rtol=1e-5)


def test_missing_support_class_is_masked_and_finite():
    model = _classification_model()
    X_train = torch.randn(12, 5)
    y_train = torch.arange(12) % 2
    output = model.forward_classification(
        X_train, y_train, torch.randn(3, 5), num_classes=3
    )

    assert output["class_present_mask"].tolist() == [True, True, False]
    assert torch.isfinite(output["class_feat_stats"]).all()
    assert torch.isfinite(output["logits"]).all()


def test_classification_stats_remain_float32_under_cpu_autocast():
    model = _classification_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forward_classification(
            torch.randn(12, 4),
            torch.arange(12) % 3,
            torch.randn(2, 4),
            num_classes=3,
            return_aux=True,
        )
    assert output["class_feat_stats"].dtype == torch.float32
    assert torch.isfinite(output["logits"]).all()


def test_classification_loss_is_finite_without_teacher():
    model = _classification_model()
    y_train = torch.arange(15) % 3
    output = model.forward_classification(
        torch.randn(15, 6), y_train, torch.randn(5, 6), num_classes=3
    )
    losses = compute_classification_losses(
        output,
        torch.arange(5) % 3,
        y_train=y_train,
        cfg=model.cfg,
    )
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["ce"])


def test_regression_config_has_no_classification_checkpoint_parameters():
    regression = DeepSetICLModel(
        ModelConfig(
            d_phi=16,
            d_rho=16,
            pool="mean",
            n_heads=4,
            n_sab_feat=1,
            dropout=0.0,
        )
    )
    assert not any(
        key.startswith(
            (
                "class_label_encoder.",
                "classification_stat_encoder.",
                "class_fusion.",
                "class_coefficient_head.",
                "class_bias_head.",
            )
        )
        for key in regression.state_dict()
    )


def test_classification_checkpoint_state_round_trip_is_strict():
    source = _classification_model()
    restored = DeepSetICLModel(ModelConfig(**vars(source.cfg)))
    restored.load_state_dict(source.state_dict(), strict=True)
    assert restored.cfg.model_arch_version == "model4"
    assert restored.cfg.task_objective == "inductive_classification"


def test_training_data_family_routes_classification_explicitly():
    spec = get_training_data_spec("synthetic_linear_classification")
    assert spec.task_objective == "inductive_classification"
    assert spec.index_table == "META_CLASSIFICATION_DATASET_INDEX"
    assert spec.stage == "@META_DATASET_STAGE"
    assert spec.hpo_metric == "val_cross_entropy"


def test_linear_classification_parquet_loads_with_integer_labels(tmp_path):
    matrix = pa.list_(pa.list_(pa.float64()))
    labels = pa.list_(pa.int64())
    path = tmp_path / "classification.parquet"
    table = pa.table({
        "X_train": pa.array(
            [[[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]], type=matrix
        ),
        "y_train": pa.array([[0, 1, 1]], type=labels),
        "X_test": pa.array([[[0.5, 0.5], [1.5, 0.5]]], type=matrix),
        "y_test": pa.array([[0, 1]], type=labels),
        "num_classes": pa.array([2], type=pa.int64()),
        "w_true": pa.array([[1.0, -1.0]], type=pa.list_(pa.float64())),
        "b_true": pa.array([0.25], type=pa.float64()),
    })
    pq.write_table(table, path)

    batch = load_classification_parquet(str(path))

    assert batch["task_objective"] == "inductive_classification"
    assert batch["X_train"].shape == (3, 2)
    assert batch["y_train"].dtype == torch.long
    assert batch["num_classes"] == 2
    assert batch["W_true"].shape == (2,)
    assert batch["b_true"].ndim == 0
