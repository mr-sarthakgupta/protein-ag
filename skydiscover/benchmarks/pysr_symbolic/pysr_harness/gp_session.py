"""Run a bounded PySR genetic-programming session inside one evaluator call."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pysr.export_numpy import CallableEquation, sympy2numpy
from pysr.export_sympy import create_sympy_symbols, pysr2sympy
from pysr.sr import calculate_scores, idx_model_selection

from pysr_harness.backend import build_pysr_regressor, default_gp_config
from pysr_harness.metrics import combined_score_from_nmse, nmse


def _predict_equation(
    equation: str,
    X: NDArray,
    *,
    feature_names: list[str],
) -> NDArray:
    """Evaluate an equation string on feature matrix X."""
    symbols = create_sympy_symbols(feature_names)
    expr = pysr2sympy(equation, feature_names_in=feature_names)
    fn = sympy2numpy(expr, symbols)
    if isinstance(fn, CallableEquation):
        return np.asarray(fn(X), dtype=float).ravel()
    return np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=float).ravel()


def run_gp_session(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run symbolic regression with PySR Julia primitives and return best candidate metrics.

    Uses PySRRegressor internally (equation_search with bounded niterations), which
    delegates tree GP, mutation/crossover, constant optimization, and simplification
    to SymbolicRegression.jl.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).ravel()

    merged = default_gp_config()
    if config:
        merged.update(config)

    feature_names = merged.get("feature_names")
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]

    import warnings

    model = build_pysr_regressor(merged)
    with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train, y_train, variable_names=feature_names)

    equations = model.equations_
    if equations is None or len(equations) == 0:
        return {
            "equation": "",
            "loss": float("inf"),
            "complexity": 0.0,
            "nmse_train": float("inf"),
            "nmse_val": float("inf"),
            "combined_score": 0.0,
            "selection_strategy": merged.get("selection_strategy", "best"),
        }

    if merged.get("calculate_scores", True) and "loss" in equations.columns:
        score_df = calculate_scores(equations)
        equations = equations.assign(score=score_df["score"])

    strategy = merged.get("selection_strategy", merged.get("model_selection", "best"))
    idx = idx_model_selection(equations, model_selection=strategy)
    row = equations.iloc[idx]
    equation = str(row["equation"])
    complexity = float(row["complexity"])
    loss = float(row["loss"])

    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            y_pred_train = _predict_equation(
                equation, X_train, feature_names=list(feature_names)
            )
            y_pred_val = _predict_equation(
                equation, X_val, feature_names=list(feature_names)
            )
    except Exception:
        y_pred_train = np.full_like(y_train, np.nan)
        y_pred_val = np.full_like(y_val, np.nan)

    nmse_train = nmse(y_train, y_pred_train)
    nmse_val = nmse(y_val, y_pred_val)

    return {
        "equation": equation,
        "loss": loss,
        "complexity": complexity,
        "nmse_train": nmse_train,
        "nmse_val": nmse_val,
        "combined_score": combined_score_from_nmse(nmse_val),
        "selection_strategy": strategy,
        "niterations": merged.get("niterations"),
        "populations": merged.get("populations"),
        "population_size": merged.get("population_size"),
    }
