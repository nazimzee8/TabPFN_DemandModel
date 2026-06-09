"""Canonical TRAINING_DATA_FAMILY routing for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from constants import (
    LINEAR_REGRESSION_TRAINING_FAMILY,
    MIXED_CAT_REGRESSION_TRAINING_FAMILY,
    MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
    NONLINEAR_CLASSIFICATION_TRAINING_FAMILY,
    NONLINEAR_MIXED_REGRESSION_TRAINING_FAMILY,
    NONLINEAR_MIXED_CLASSIFICATION_TRAINING_FAMILY,
)


REGRESSION_OBJECTIVE = "inductive_regression"
CLASSIFICATION_OBJECTIVE = "inductive_classification"
CLASSIFICATION_TRAINING_FAMILY = "synthetic_linear_classification"
NONLINEAR_TRAINING_FAMILY = "synthetic_regression_nonlinear"

# All four nonlinear task families — used by is_nonlinear property
_NONLINEAR_FAMILIES: frozenset[str] = frozenset({
    NONLINEAR_TRAINING_FAMILY,
    NONLINEAR_CLASSIFICATION_TRAINING_FAMILY,
    NONLINEAR_MIXED_REGRESSION_TRAINING_FAMILY,
    NONLINEAR_MIXED_CLASSIFICATION_TRAINING_FAMILY,
})


@dataclass(frozen=True)
class TrainingDataSpec:
    family: str
    task_objective: str
    index_table: str
    stage: str
    hpo_metric: str
    hpo_mode: str
    expected_total_env: str
    index_builder: str

    @property
    def is_classification(self) -> bool:
        return self.task_objective == CLASSIFICATION_OBJECTIVE

    @property
    def is_nonlinear(self) -> bool:
        return self.family in _NONLINEAR_FAMILIES


_LINEAR_REGRESSION_SPEC = TrainingDataSpec(
    family=LINEAR_REGRESSION_TRAINING_FAMILY,
    task_objective=REGRESSION_OBJECTIVE,
    index_table="META_REGRESSION_DATASET_INDEX",
    stage="@META_REGRESSION_DATASET_STAGE",
    hpo_metric="val_mse",
    hpo_mode="min",
    expected_total_env="META_REGRESSION_DATASET_EXPECTED_TOTAL",
    index_builder="build_meta_dataset_index",
)

_FAMILY_SPECS = {
    LINEAR_REGRESSION_TRAINING_FAMILY: _LINEAR_REGRESSION_SPEC,
    "synthetic_regression_primary": _LINEAR_REGRESSION_SPEC,
    "synthetic_regression_ood": _LINEAR_REGRESSION_SPEC,
    "synthetic_regression_combined": _LINEAR_REGRESSION_SPEC,  # backward-compat alias
    "market_mental_model": _LINEAR_REGRESSION_SPEC,
    "unknown": _LINEAR_REGRESSION_SPEC,
    NONLINEAR_TRAINING_FAMILY: TrainingDataSpec(
        family=NONLINEAR_TRAINING_FAMILY,
        task_objective=REGRESSION_OBJECTIVE,
        index_table="META_NONLINEAR_REGRESSION_DATASET_INDEX",
        stage="@META_NONLINEAR_REGRESSION_DATASET_STAGE",
        hpo_metric="val_mse",
        hpo_mode="min",
        expected_total_env="META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL",
        index_builder="build_meta_nonlinear_dataset_index",
    ),
    CLASSIFICATION_TRAINING_FAMILY: TrainingDataSpec(
        family=CLASSIFICATION_TRAINING_FAMILY,
        task_objective=CLASSIFICATION_OBJECTIVE,
        index_table="META_CLASSIFICATION_DATASET_INDEX",
        stage="@META_CLASSIFICATION_DATASET_STAGE",
        hpo_metric="val_cross_entropy",
        hpo_mode="min",
        expected_total_env="META_CLASSIFICATION_DATASET_EXPECTED_TOTAL",
        index_builder="build_meta_classification_dataset_index",
    ),
    MIXED_CAT_REGRESSION_TRAINING_FAMILY: TrainingDataSpec(
        family=MIXED_CAT_REGRESSION_TRAINING_FAMILY,
        task_objective=REGRESSION_OBJECTIVE,
        index_table="META_MIXED_REGRESSION_DATASET_INDEX",
        stage="@META_REGRESSION_DATASET_STAGE",
        hpo_metric="val_mse",
        hpo_mode="min",
        expected_total_env="META_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL",
        index_builder="build_meta_mixed_regression_dataset_index",
    ),
    MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY: TrainingDataSpec(
        family=MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
        task_objective=CLASSIFICATION_OBJECTIVE,
        index_table="META_MIXED_CATEGORICAL_DATASET_INDEX",
        stage="@META_CLASSIFICATION_DATASET_STAGE",
        hpo_metric="val_cross_entropy",
        hpo_mode="min",
        expected_total_env="META_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL",
        index_builder="build_meta_mixed_classification_dataset_index",
    ),
    # ---- Nonlinear classification ----
    NONLINEAR_CLASSIFICATION_TRAINING_FAMILY: TrainingDataSpec(
        family=NONLINEAR_CLASSIFICATION_TRAINING_FAMILY,
        task_objective=CLASSIFICATION_OBJECTIVE,
        index_table="META_NONLINEAR_CLASSIFICATION_DATASET_INDEX",
        stage="@META_NONLINEAR_CLASSIFICATION_DATASET_STAGE",
        hpo_metric="val_cross_entropy",
        hpo_mode="min",
        expected_total_env="META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL",
        index_builder="build_meta_nonlinear_classification_dataset_index",
    ),
    # ---- Nonlinear mixed-categorical regression ----
    NONLINEAR_MIXED_REGRESSION_TRAINING_FAMILY: TrainingDataSpec(
        family=NONLINEAR_MIXED_REGRESSION_TRAINING_FAMILY,
        task_objective=REGRESSION_OBJECTIVE,
        index_table="META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX",
        stage="@META_NONLINEAR_REGRESSION_DATASET_STAGE",
        hpo_metric="val_mse",
        hpo_mode="min",
        expected_total_env="META_NONLINEAR_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL",
        index_builder="build_meta_nonlinear_mixed_regression_dataset_index",
    ),
    # ---- Nonlinear mixed-categorical classification ----
    NONLINEAR_MIXED_CLASSIFICATION_TRAINING_FAMILY: TrainingDataSpec(
        family=NONLINEAR_MIXED_CLASSIFICATION_TRAINING_FAMILY,
        task_objective=CLASSIFICATION_OBJECTIVE,
        index_table="META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX",
        stage="@META_NONLINEAR_CLASSIFICATION_DATASET_STAGE",
        hpo_metric="val_cross_entropy",
        hpo_mode="min",
        expected_total_env="META_NONLINEAR_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL",
        index_builder="build_meta_nonlinear_mixed_classification_dataset_index",
    ),
}


def get_training_data_spec(training_data_family: str) -> TrainingDataSpec:
    family = str(training_data_family or "").strip()
    try:
        spec = _FAMILY_SPECS[family]
    except KeyError as exc:
        raise ValueError(
            f"Unknown TRAINING_DATA_FAMILY={family!r}. "
            f"Allowed values: {sorted(_FAMILY_SPECS)}"
        ) from exc
    if spec.family == family:
        return spec
    return TrainingDataSpec(
        family=family,
        task_objective=spec.task_objective,
        index_table=spec.index_table,
        stage=spec.stage,
        hpo_metric=spec.hpo_metric,
        hpo_mode=spec.hpo_mode,
        expected_total_env=spec.expected_total_env,
        index_builder=spec.index_builder,
    )


def task_objective_for_family(training_data_family: str) -> str:
    return get_training_data_spec(training_data_family).task_objective


def is_classification_family(training_data_family: str) -> bool:
    return get_training_data_spec(training_data_family).is_classification


def allowed_training_data_families() -> frozenset[str]:
    return frozenset(_FAMILY_SPECS)
