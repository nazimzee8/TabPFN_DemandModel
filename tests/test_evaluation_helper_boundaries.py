import builtins
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_synthetic_regression_entrypoint_does_not_import_evaluate():
    src = (SRC_DIR / "evaluate_synthetic_regression.py").read_text()

    assert "from evaluate import" not in src
    assert "import evaluate" not in src


def test_synthetic_regression_imports_when_evaluate_is_blocked(monkeypatch):
    original_synreg = sys.modules.get("evaluate_synthetic_regression")
    original_evaluate = sys.modules.get("evaluate")
    sys.modules.pop("evaluate_synthetic_regression", None)
    sys.modules.pop("evaluate", None)
    real_import = builtins.__import__

    def block_evaluate(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "evaluate":
            raise ImportError("blocked evaluate import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_evaluate)
    try:
        module = importlib.import_module("evaluate_synthetic_regression")

        assert module.DeepSetModel is not None
        assert module.ModelConfig is not None
    finally:
        sys.modules.pop("evaluate_synthetic_regression", None)
        sys.modules.pop("evaluate", None)
        if original_synreg is not None:
            sys.modules["evaluate_synthetic_regression"] = original_synreg
        if original_evaluate is not None:
            sys.modules["evaluate"] = original_evaluate


def test_autogluon_helper_missing_import_message(monkeypatch):
    import autogluon_models

    real_autogluon = sys.modules.pop("autogluon", None)
    real_tabular = sys.modules.pop("autogluon.tabular", None)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", None)
    try:
        with pytest.raises(RuntimeError, match="AUTOGLUON_RUNTIME_ENVIRONMENT"):
            autogluon_models.predict_autogluon(
                np.ones((4, 2)),
                np.arange(4, dtype=float),
                np.ones((2, 2)),
            )
    finally:
        sys.modules.pop("autogluon.tabular", None)
        if real_autogluon is not None:
            sys.modules["autogluon"] = real_autogluon
        if real_tabular is not None:
            sys.modules["autogluon.tabular"] = real_tabular


def test_autogluon_helper_fit_configuration_and_cleanup(tmp_path, monkeypatch):
    import autogluon_models

    fit_kwargs = {}
    model_dir = tmp_path / "ag_model"

    class _Predictions:
        def to_numpy(self):
            return np.array([1.0, 2.0])

    class _FakePredictor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, train_df, **kwargs):
            fit_kwargs.update(kwargs)
            return self

        def predict(self, test_df):
            return _Predictions()

    preds = autogluon_models.predict_autogluon(
        np.ones((4, 2)),
        np.arange(4, dtype=float),
        np.ones((2, 2)),
        time_limit=17,
        presets="medium_quality",
        num_cpus=6,
        num_gpus=0,
        model_dir=str(model_dir),
        predictor_cls=_FakePredictor,
    )

    assert preds.tolist() == [1.0, 2.0]
    assert fit_kwargs["time_limit"] == 17
    assert fit_kwargs["presets"] == "medium_quality"
    assert fit_kwargs["num_cpus"] == 6
    assert fit_kwargs["num_gpus"] == 0
    assert not model_dir.exists()


def test_autogluon_helper_cleans_up_after_fit_failure(tmp_path):
    import autogluon_models

    model_dir = tmp_path / "ag_model_failure"

    class _FailingPredictor:
        def __init__(self, **kwargs):
            pass

        def fit(self, train_df, **kwargs):
            raise RuntimeError("fit failed")

    with pytest.raises(RuntimeError, match="fit failed"):
        autogluon_models.predict_autogluon(
            np.ones((4, 2)),
            np.arange(4, dtype=float),
            np.ones((2, 2)),
            model_dir=str(model_dir),
            predictor_cls=_FailingPredictor,
        )

    assert not model_dir.exists()
