"""Evaluate SkyDiscover-proposed symbolic expressions with PySR export helpers."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
from typing import Any, Sequence

import numpy as np
import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
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


class SingleEquationViolation(RuntimeError):
    """Raised when a candidate tries to score more than one equation template."""


MAX_EQUATION_COMPLEXITY = int(os.environ.get("SKYDISCOVER_MAX_EQUATION_COMPLEXITY", "1600"))
MAX_EQUATION_CONSTANTS = int(os.environ.get("SKYDISCOVER_MAX_EQUATION_CONSTANTS", "15"))
_BANNED_EXPR_TYPES = (
    sp.Piecewise,
    Relational,
    BooleanFunction,
    sp.Heaviside,
    sp.DiracDelta,
)
_BANNED_EXPR_FUNCTIONS = {
    "Max",
    "Min",
    "sign",
    "floor",
    "ceiling",
    "Mod",
}


@dataclass(frozen=True)
class FittedExpression:
    """A fitted symbolic expression and its validation predictions."""

    expression_template: sp.Expr
    expression: sp.Expr
    constants: dict[str, float]
    y_pred_train: NDArray
    y_pred_val: NDArray


@dataclass
class _SingleEquationState:
    calls: int = 0
    template_fingerprint: str | None = None
    result_snapshot: dict[str, Any] | None = None


_SINGLE_EQUATION_STATE = threading.local()


@contextmanager
def single_equation_evaluation():
    """
    Enforce the benchmark contract for one evaluate_symbolic_candidate() call.

    A candidate may build helper subexpressions, but it may submit exactly one
    symbolic template to the scorer. This prevents reward hacking where a
    program tries many equations and returns whichever receives the best score.
    """
    previous = getattr(_SINGLE_EQUATION_STATE, "state", None)
    state = _SingleEquationState()
    _SINGLE_EQUATION_STATE.state = state
    try:
        yield
        if state.calls != 1:
            raise SingleEquationViolation(
                "Single-equation violation: evaluate_symbolic_candidate() must call "
                "evaluate_expression() exactly once with one symbolic equation template. "
                f"It submitted {state.calls} templates. Build one base equation and let "
                "the harness fit that equation's constants per dataset."
            )
    finally:
        _SINGLE_EQUATION_STATE.state = previous


def _record_scored_template(template: sp.Expr) -> None:
    state = getattr(_SINGLE_EQUATION_STATE, "state", None)
    if state is None:
        return

    state.calls += 1
    fingerprint = sp.srepr(template)
    if state.template_fingerprint is None:
        state.template_fingerprint = fingerprint
        return

    if state.template_fingerprint != fingerprint:
        raise SingleEquationViolation(
            "Single-equation violation: one evaluate_symbolic_candidate() call "
            "submitted multiple different symbolic equation templates. The base "
            "equation must be fixed; only fitted constants may change."
        )

    raise SingleEquationViolation(
        "Single-equation violation: evaluate_symbolic_candidate() called "
        "evaluate_expression()/fit_expression_constants() more than once. Do not "
        "try multiple equation templates or multiple scorer calls and pick the "
        "best result; submit exactly one symbolic expression."
    )


def _record_scorer_result(result: dict[str, Any]) -> None:
    state = getattr(_SINGLE_EQUATION_STATE, "state", None)
    if state is not None:
        state.result_snapshot = copy.deepcopy(result)


def validate_single_equation_result(result: Any) -> None:
    """
    Ensure the candidate returns the actual scorer output, unchanged.

    This closes the loophole where a program calls evaluate_expression() once
    to satisfy the call-count guard, then fabricates or edits the metrics it
    returns to the evaluator.
    """
    state = getattr(_SINGLE_EQUATION_STATE, "state", None)
    if state is None:
        return

    if state.result_snapshot is None:
        raise SingleEquationViolation(
            "Single-equation violation: evaluate_symbolic_candidate() must return "
            "the dictionary produced by evaluate_expression()."
        )

    if result != state.result_snapshot:
        raise SingleEquationViolation(
            "Single-equation violation: evaluate_symbolic_candidate() must return "
            "the exact, unmodified result from its single evaluate_expression() "
            "call. Do not fabricate, overwrite, wrap, or post-process metrics."
        )


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


def _validate_expression_template(template: sp.Expr, constants: Sequence[sp.Symbol]) -> None:
    complexity = int(_complexity(template))
    if complexity > MAX_EQUATION_COMPLEXITY:
        raise SingleEquationViolation(
            "Single-equation violation: symbolic template is too complex "
            f"({complexity} nodes > limit {MAX_EQUATION_COMPLEXITY}). Do not hide "
            "many candidate equations inside one oversized expression."
        )

    if len(constants) > MAX_EQUATION_CONSTANTS:
        raise SingleEquationViolation(
            "Single-equation violation: too many fitted constants "
            f"({len(constants)} > limit {MAX_EQUATION_CONSTANTS}). Use one compact "
            "universal equation, not an over-parameterized surrogate."
        )

    for node in sp.preorder_traversal(template):
        if isinstance(node, _BANNED_EXPR_TYPES) or node.func.__name__ in _BANNED_EXPR_FUNCTIONS:
            raise SingleEquationViolation(
                "Single-equation violation: conditional, piecewise, discontinuous, "
                f"or gated symbolic construct '{node.func.__name__}' is not allowed. "
                "Use one smooth global equation template."
            )


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
    _record_scored_template(template)

    if constants is None:
        feature_set = set(feature_symbols(X_train.shape[1]))
        constants = sorted(template.free_symbols - feature_set, key=lambda s: s.name)
    constants = list(constants)
    _validate_expression_template(template, constants)

    if not constants:
        return FittedExpression(
            expression_template=template,
            expression=template,
            constants={},
            y_pred_train=_predict(template, X_train, feature_names),
            y_pred_val=_predict(template, X_val, feature_names),
        )

    if initial_values is None:
        x0 = np.ones(len(constants), dtype=float)
    else:
        x0 = np.asarray(initial_values, dtype=float).reshape(-1)
        if x0.size != len(constants):
            raise ValueError("initial_values must match the number of constants")

    feature_syms = create_sympy_symbols(feature_names)
    X_cols = [X_train[:, i] for i in range(X_train.shape[1])]

    # Compile expression once upfront so the residual avoids repeated
    # SymPy subs() + sympy2numpy() calls (the main bottleneck).
    try:
        _compiled = sp.lambdify(feature_syms + constants, template, modules=["numpy"])
        _test_args = X_cols + [float(v) for v in x0]
        _test_out = np.asarray(_compiled(*_test_args), dtype=float).reshape(-1)
        if _test_out.shape != y_train.shape:
            raise ValueError("shape mismatch")
        _use_compiled = True
    except Exception:
        _use_compiled = False

    if _use_compiled:
        def residual(theta: NDArray) -> NDArray:
            try:
                args = X_cols + [float(v) for v in theta]
                prediction = np.asarray(_compiled(*args), dtype=float).reshape(-1)
            except Exception:
                return np.full_like(y_train, 1e12, dtype=float)
            if prediction.shape != y_train.shape or not np.all(np.isfinite(prediction)):
                return np.full_like(y_train, 1e12, dtype=float)
            return prediction - y_train
    else:
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
    fitted_expr = template.subs(substitutions)

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

    result = {
        "equation_template": str(fitted.expression_template),
        "equation": str(fitted.expression),
        "constants": fitted.constants,
        "loss": loss,
        "complexity": _complexity(fitted.expression_template),
        "nmse_train": nmse_train,
        "nmse_val": nmse_val,
        "combined_score": combined_score_from_nmse(nmse_val),
    }
    _record_scorer_result(result)
    return result
