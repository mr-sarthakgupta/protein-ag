"""Evaluate SkyDiscover-proposed symbolic expressions with PySR export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from pysr.export_numpy import CallableEquation, sympy2numpy
from pysr.export_sympy import create_sympy_symbols, pysr2sympy, sympy_mappings
from scipy.optimize import least_squares

from pysr_harness.metrics import combined_score_from_nmse, nmse
from pysr_harness.operators import (
    GENERAL_BINARY_OPERATORS,
    GENERAL_TERNARY_OPERATORS,
    GENERAL_UNARY_OPERATORS,
)


@dataclass(frozen=True)
class FittedExpression:
    """A fitted symbolic expression and its validation predictions."""

    expression_template: sp.Expr
    expression: sp.Expr
    constants: dict[str, float]
    y_pred_train: NDArray
    y_pred_val: NDArray


def feature_symbols(n_features: int) -> list[sp.Symbol]:
    """Create PySR-compatible feature symbols x0, x1, ..."""
    return create_sympy_symbols([f"x{i}" for i in range(n_features)])


def constant_symbols(n_constants: int) -> list[sp.Symbol]:
    """Create tunable constant symbols c0, c1, ..."""
    return [sp.Symbol(f"c{i}", real=True) for i in range(n_constants)]


def pysr_operator_names() -> dict[str, list[str]]:
    """Return the PySR operator vocabulary exposed to evolved equation builders."""
    return {
        "unary": GENERAL_UNARY_OPERATORS.copy(),
        "binary": GENERAL_BINARY_OPERATORS.copy(),
        "ternary": GENERAL_TERNARY_OPERATORS.copy(),
    }


def pysr_operator_namespace() -> dict[str, Any]:
    """
    Return PySR-to-SymPy operator callables usable in expression templates.

    Operators like ``+``, ``-``, ``*``, and ``/`` should be written with normal
    Python syntax. Named PySR operators such as ``sin``, ``cos``, ``sqrt``,
    ``log1p``, ``max``, ``min``, ``muladd``, and ``clamp`` are exposed here.
    """
    return {
        **sympy_mappings,
        "pi": sp.pi,
        "E": sp.E,
    }


def _as_sympy_expr(expression: sp.Expr | str | float | int, feature_names: Sequence[str]) -> sp.Expr:
    if isinstance(expression, sp.Expr):
        return expression
    return pysr2sympy(expression, feature_names_in=list(feature_names))


def _complexity(expression: sp.Expr) -> float:
    return float(sum(1 for _ in sp.preorder_traversal(expression)))


def _predict(expression: sp.Expr, X: NDArray, feature_names: Sequence[str]) -> NDArray:
    symbols = create_sympy_symbols(feature_names)
    fn = sympy2numpy(expression, symbols)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if isinstance(fn, CallableEquation):
            y_pred = fn(np.asarray(X, dtype=float))
        else:
            y_pred = fn(*[X[:, i] for i in range(X.shape[1])])
    return np.asarray(y_pred, dtype=float).reshape(-1)


def fit_expression_constants(
    expression: sp.Expr | str | float | int,
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    *,
    constants: Sequence[sp.Symbol] | None = None,
    initial_values: Sequence[float] | None = None,
    max_nfev: int = 300,
) -> FittedExpression:
    """
    Fit free constants in a SkyDiscover-proposed expression template.

    The expression structure is supplied by the evolved program. This helper only
    performs the continuous inner fit and PySR-style export/evaluation.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_val = np.asarray(X_val, dtype=float)
    feature_names = [f"x{i}" for i in range(X_train.shape[1])]
    template = _as_sympy_expr(expression, feature_names)

    if constants is None:
        feature_set = set(feature_symbols(X_train.shape[1]))
        constants = sorted(template.free_symbols - feature_set, key=lambda s: s.name)
    constants = list(constants)

    if not constants:
        fitted_expr = sp.simplify(template)
        return FittedExpression(
            expression_template=template,
            expression=fitted_expr,
            constants={},
            y_pred_train=_predict(fitted_expr, X_train, feature_names),
            y_pred_val=_predict(fitted_expr, X_val, feature_names),
        )

    if initial_values is None:
        x0 = np.ones(len(constants), dtype=float)
    else:
        x0 = np.asarray(initial_values, dtype=float).reshape(-1)
        if x0.size != len(constants):
            raise ValueError("initial_values must match the number of constants")

    def residual(theta: NDArray) -> NDArray:
        substitutions = {symbol: float(value) for symbol, value in zip(constants, theta)}
        candidate = template.subs(substitutions)
        try:
            prediction = _predict(candidate, X_train, feature_names)
        except Exception:
            return np.full_like(y_train, 1e12, dtype=float)
        if prediction.shape != y_train.shape or not np.all(np.isfinite(prediction)):
            return np.full_like(y_train, 1e12, dtype=float)
        return prediction - y_train

    import warnings

    with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = least_squares(
            residual,
            x0,
            loss="soft_l1",
            max_nfev=max_nfev,
        )
    fitted_constants = {
        str(symbol): float(value) for symbol, value in zip(constants, result.x)
    }
    substitutions = dict(zip(constants, result.x))
    fitted_expr = sp.simplify(template.subs(substitutions))

    return FittedExpression(
        expression_template=template,
        expression=fitted_expr,
        constants=fitted_constants,
        y_pred_train=_predict(fitted_expr, X_train, feature_names),
        y_pred_val=_predict(fitted_expr, X_val, feature_names),
    )


def evaluate_expression(
    expression: sp.Expr | str | float | int,
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
    *,
    constants: Sequence[sp.Symbol] | None = None,
    initial_values: Sequence[float] | None = None,
    max_nfev: int = 300,
) -> dict[str, Any]:
    """Score a proposed equation template after fitting its constants."""
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    fitted = fit_expression_constants(
        expression,
        X_train,
        y_train,
        X_val,
        constants=constants,
        initial_values=initial_values,
        max_nfev=max_nfev,
    )
    nmse_train = nmse(y_train, fitted.y_pred_train)
    nmse_val = nmse(y_val, fitted.y_pred_val)
    loss = float(np.mean((fitted.y_pred_train - y_train) ** 2))

    return {
        "equation_template": str(fitted.expression_template),
        "equation": str(fitted.expression),
        "constants": fitted.constants,
        "loss": loss,
        "complexity": _complexity(fitted.expression_template),
        "nmse_train": nmse_train,
        "nmse_val": nmse_val,
        "combined_score": combined_score_from_nmse(nmse_val),
    }
