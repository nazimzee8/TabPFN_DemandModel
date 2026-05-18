"""Shared CPU baseline model construction for evaluation entrypoints."""

from __future__ import annotations

BENCHMARK_DEP_IMPORT_ERRORS = {}

try:
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    BENCHMARK_DEP_IMPORT_ERRORS["sklearn"] = exc

try:
    import xgboost as xgb
except ImportError as exc:
    BENCHMARK_DEP_IMPORT_ERRORS["xgboost"] = exc

try:
    import lightgbm as lgb
except ImportError as exc:
    BENCHMARK_DEP_IMPORT_ERRORS["lightgbm"] = exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    BENCHMARK_DEP_IMPORT_ERRORS["catboost"] = exc


BENCHMARK_DEP_PACKAGE_NAMES = {
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}

CPU_BASELINE_METHODS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "KNN",
    "LinearRegression",
    "Ridge",
    "SVR",
    "MLP",
]

BASELINE_METHOD_DEPS = {
    "XGBoost": ("sklearn", "xgboost"),
    "LightGBM": ("sklearn", "lightgbm"),
    "CatBoost": ("sklearn", "catboost"),
    "RandomForest": ("sklearn",),
    "KNN": ("sklearn",),
    "LinearRegression": ("sklearn",),
    "Ridge": ("sklearn",),
    "SVR": ("sklearn",),
    "MLP": ("sklearn",),
}


def _dep_label(dep_name, import_errors=None):
    import_errors = BENCHMARK_DEP_IMPORT_ERRORS if import_errors is None else import_errors
    exc = import_errors.get(dep_name)
    package_name = BENCHMARK_DEP_PACKAGE_NAMES.get(dep_name, dep_name)
    if exc is None:
        return package_name
    return f"{package_name} ({getattr(exc, 'name', None) or exc})"


def benchmark_dependency_error_message(
    missing_deps=None,
    selected_methods=None,
    import_errors=None,
):
    import_errors = BENCHMARK_DEP_IMPORT_ERRORS if import_errors is None else import_errors
    if missing_deps is None:
        missing_deps = list(import_errors)
    missing = ", ".join(_dep_label(dep, import_errors) for dep in missing_deps) or "unknown"
    method_text = ""
    if selected_methods:
        method_text = f" Selected method(s): {', '.join(selected_methods)}."
    return (
        "Selected benchmark dependencies are not available in the runtime image. "
        "Check BENCHMARK_RUNTIME_ENVIRONMENT/AUTOGLUON_RUNTIME_ENVIRONMENT. "
        f"Missing module(s): {missing}.{method_text} Install/validate only the "
        "packages required by the selected method(s): scikit-learn for shared "
        "benchmark preprocessing/metrics; xgboost, lightgbm, or catboost only "
        "when those exact baseline methods are selected."
    )


def make_baseline_model(
    method,
    seed,
    import_errors=None,
    constructors=None,
    dependency_error_builder=None,
):
    """Construct one selected sklearn-compatible baseline."""
    import_errors = BENCHMARK_DEP_IMPORT_ERRORS if import_errors is None else import_errors
    dependency_error_builder = (
        benchmark_dependency_error_message
        if dependency_error_builder is None
        else dependency_error_builder
    )
    constructors = {} if constructors is None else constructors

    if method not in BASELINE_METHOD_DEPS:
        raise ValueError(f"Unknown CPU baseline method: {method}")
    missing_deps = [
        dep for dep in BASELINE_METHOD_DEPS[method]
        if dep in import_errors
    ]
    if missing_deps:
        raise RuntimeError(
            dependency_error_builder(
                missing_deps=missing_deps,
                selected_methods=[method],
            )
        )

    ridge_cls = constructors.get("Ridge", Ridge)
    if method == "XGBoost":
        xgb_mod = constructors.get("xgb", xgb)
        return xgb_mod.XGBRegressor(n_estimators=200, random_state=seed, verbosity=0, n_jobs=1)
    if method == "LightGBM":
        lgb_mod = constructors.get("lgb", lgb)
        return lgb_mod.LGBMRegressor(n_estimators=200, random_state=seed, verbose=-1, n_jobs=1)
    if method == "CatBoost":
        catboost_cls = constructors.get("CatBoostRegressor", CatBoostRegressor)
        return catboost_cls(iterations=200, random_state=seed, verbose=0, allow_writing_files=False)
    if method == "RandomForest":
        return constructors.get("RandomForestRegressor", RandomForestRegressor)(
            n_estimators=200, random_state=seed, n_jobs=1
        )
    if method == "KNN":
        return constructors.get("KNeighborsRegressor", KNeighborsRegressor)(n_neighbors=5, n_jobs=1)
    if method == "LinearRegression":
        return constructors.get("LinearRegression", LinearRegression)(n_jobs=1)
    if method == "Ridge":
        return ridge_cls(alpha=1.0)
    if method == "SVR":
        return constructors.get("Pipeline", Pipeline)([
            ("scaler", constructors.get("StandardScaler", StandardScaler)()),
            ("svr", constructors.get("SVR", SVR)()),
        ])
    if method == "MLP":
        return constructors.get("Pipeline", Pipeline)([
            ("scaler", constructors.get("StandardScaler", StandardScaler)()),
            ("mlp", constructors.get("MLPRegressor", MLPRegressor)(
                hidden_layer_sizes=(256, 128),
                max_iter=300,
                random_state=seed,
            )),
        ])
    raise AssertionError(f"Unhandled CPU baseline method: {method}")


def validate_baseline_dependencies(
    methods=None,
    *,
    import_errors=None,
    dependency_error_builder=None,
):
    methods = CPU_BASELINE_METHODS if methods is None else list(methods)
    import_errors = BENCHMARK_DEP_IMPORT_ERRORS if import_errors is None else import_errors
    dependency_error_builder = (
        benchmark_dependency_error_message
        if dependency_error_builder is None
        else dependency_error_builder
    )

    unknown_methods = sorted(set(methods) - set(BASELINE_METHOD_DEPS))
    if unknown_methods:
        raise ValueError(f"Unknown CPU baseline method(s): {unknown_methods}")

    required_deps = []
    for method in methods:
        required_deps.extend(BASELINE_METHOD_DEPS[method])
    missing_deps = sorted(
        {dep for dep in required_deps if dep in import_errors},
        key=required_deps.index,
    )
    if missing_deps:
        raise RuntimeError(
            dependency_error_builder(
                missing_deps=missing_deps,
                selected_methods=methods,
            )
        )


def get_baselines(seed, methods=None, **kwargs):
    """Return selected sklearn-compatible baseline regressors."""
    methods = CPU_BASELINE_METHODS if methods is None else list(methods)
    return {
        method: make_baseline_model(method, seed, **kwargs)
        for method in methods
    }
