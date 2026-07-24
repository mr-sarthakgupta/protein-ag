"""Evaluate SkyDiscover-proposed symbolic expressions with PySR export helpers."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
import threading
from typing import Any, Literal, Sequence

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
EXTRA_EQUATION_PENALTY = float(os.environ.get("SKYDISCOVER_EXTRA_EQUATION_PENALTY", "0.01"))
if EXTRA_EQUATION_PENALTY < 0.0:
    raise ValueError("SKYDISCOVER_EXTRA_EQUATION_PENALTY must be nonnegative")
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


@dataclass(frozen=True)
class CandidateEquation:
    """One equation in an ordered observed-variable DAE candidate."""

    kind: Literal["ode", "algebraic"]
    expression: sp.Expr | str | float | int
    target: sp.Symbol | str = "x4"


def ode_equation(expression: sp.Expr | str | float | int) -> CandidateEquation:
    """Declare the sole ODE for the current observed concentration, x4."""
    return CandidateEquation("ode", expression, "x4")


def algebraic_equation(
    target: sp.Symbol | str,
    expression: sp.Expr | str | float | int,
) -> CandidateEquation:
    """Declare a derived symbolic alias (not a dynamic or latent state)."""
    return CandidateEquation("algebraic", expression, target)


@dataclass(frozen=True)
class ResolvedEquationSystem:
    """Validated system metadata plus its scalar, alias-free ODE RHS."""

    equations: tuple[CandidateEquation, ...]
    equation_templates: tuple[str, ...]
    resolved_ode: sp.Expr
    resolved_ode_template: str
    system_fingerprint: str
    total_tree_complexity: float
    structural_penalty: float


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


def _record_scored_fingerprint(fingerprint: str) -> None:
    state = getattr(_SINGLE_EQUATION_STATE, "state", None)
    if state is None:
        return

    state.calls += 1
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


def _record_scored_template(template: sp.Expr) -> None:
    """Backward-compatible scalar fingerprint recorder."""
    _record_scored_fingerprint(sp.srepr(template))


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


def _as_sympy_expr(
    expression: sp.Expr | str | float | int, feature_names: Sequence[str]
) -> sp.Expr:
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


def _target_symbol(target: sp.Symbol | str) -> sp.Symbol:
    if isinstance(target, sp.Symbol):
        return target
    if not isinstance(target, str) or not target.isidentifier():
        raise SingleEquationViolation(
            f"Single-equation violation: invalid equation target {target!r}."
        )
    return sp.Symbol(target)


def resolve_equation_system(
    equations: Sequence[CandidateEquation],
    *,
    constants: Sequence[sp.Symbol] | None = None,
    n_features: int = 5,
    extra_equation_penalty: float | None = None,
) -> ResolvedEquationSystem:
    """Validate and collapse an ordered 1–5 equation DAE candidate to one ODE."""
    equations = tuple(equations)
    if not 1 <= len(equations) <= 5:
        raise SingleEquationViolation(
            "Single-equation violation: an equation system must contain 1–5 equations."
        )
    if n_features != 5:
        raise SingleEquationViolation(
            "Single-equation violation: the inhibited observed-variable contract "
            "requires exactly x0, x1, x2, x3, and x4."
        )

    constants = tuple(constants or ())
    if len(set(constants)) != len(constants) or not all(
        isinstance(symbol, sp.Symbol) for symbol in constants
    ):
        raise SingleEquationViolation(
            "Single-equation violation: fitted constants must be unique SymPy symbols."
        )
    features = tuple(feature_symbols(n_features))
    feature_set = set(features)
    constant_set = set(constants)
    if feature_set & constant_set:
        raise SingleEquationViolation(
            "Single-equation violation: fitted constants may not shadow x0–x4."
        )

    normalized: list[CandidateEquation] = []
    ode_expressions: list[sp.Expr] = []
    definitions: dict[sp.Symbol, sp.Expr] = {}
    ordered_targets: list[sp.Symbol] = []
    feature_names = [str(symbol) for symbol in features]
    for equation in equations:
        if not isinstance(equation, CandidateEquation):
            raise SingleEquationViolation(
                "Single-equation violation: systems must contain CandidateEquation "
                "objects made by ode_equation()/algebraic_equation()."
            )
        target = _target_symbol(equation.target)
        expression = _as_sympy_expr(equation.expression, feature_names)
        if equation.kind == "ode":
            if target != features[4]:
                raise SingleEquationViolation(
                    "Single-equation violation: the ODE target must be current "
                    "observed concentration x4."
                )
            ode_expressions.append(expression)
        elif equation.kind == "algebraic":
            if target in feature_set or target in constant_set:
                raise SingleEquationViolation(
                    f"Single-equation violation: algebraic target '{target}' shadows "
                    "a feature or fitted constant."
                )
            if target in definitions:
                raise SingleEquationViolation(
                    f"Single-equation violation: duplicate algebraic target '{target}'."
                )
            definitions[target] = expression
            ordered_targets.append(target)
        else:
            raise SingleEquationViolation(
                f"Single-equation violation: unsupported equation kind {equation.kind!r}."
            )
        normalized.append(CandidateEquation(equation.kind, expression, target))

    if len(ode_expressions) != 1:
        raise SingleEquationViolation(
            "Single-equation violation: a system must contain exactly one ODE."
        )

    aliases = set(definitions)
    allowed = feature_set | constant_set | aliases
    for expression in [*definitions.values(), ode_expressions[0]]:
        undefined = expression.free_symbols - allowed
        if undefined:
            raise SingleEquationViolation(
                "Single-equation violation: undefined/free/latent symbols: "
                + ", ".join(sorted(str(symbol) for symbol in undefined))
            )

    resolved_defs: dict[sp.Symbol, sp.Expr] = {}
    visiting: set[sp.Symbol] = set()

    def resolve_alias(symbol: sp.Symbol) -> sp.Expr:
        if symbol in resolved_defs:
            return resolved_defs[symbol]
        if symbol in visiting:
            raise SingleEquationViolation(
                "Single-equation violation: cycle in algebraic definitions."
            )
        visiting.add(symbol)
        expression = definitions[symbol]
        dependencies = expression.free_symbols & aliases
        substitutions = {dependency: resolve_alias(dependency) for dependency in dependencies}
        visiting.remove(symbol)
        resolved = expression.xreplace(substitutions)
        resolved_defs[symbol] = resolved
        return resolved

    for alias in ordered_targets:
        resolve_alias(alias)

    direct_aliases = ode_expressions[0].free_symbols & aliases
    used_aliases: set[sp.Symbol] = set()

    def mark_used(symbol: sp.Symbol) -> None:
        if symbol in used_aliases:
            return
        used_aliases.add(symbol)
        for dependency in definitions[symbol].free_symbols & aliases:
            mark_used(dependency)

    for alias in direct_aliases:
        mark_used(alias)
    if used_aliases != aliases:
        dead = aliases - used_aliases
        raise SingleEquationViolation(
            "Single-equation violation: algebraic definitions do not contribute "
            "to the ODE: " + ", ".join(sorted(str(symbol) for symbol in dead))
        )

    resolved_ode = ode_expressions[0].xreplace(
        {symbol: resolved_defs[symbol] for symbol in direct_aliases}
    )
    if resolved_ode.free_symbols - (feature_set | constant_set):
        raise SingleEquationViolation(
            "Single-equation violation: resolved ODE contains latent symbols."
        )
    total_complexity = float(sum(_complexity(equation.expression) for equation in normalized))
    _validate_expression_template(resolved_ode, constants)
    if total_complexity > MAX_EQUATION_COMPLEXITY:
        raise SingleEquationViolation(
            "Single-equation violation: total equation-system complexity is too "
            f"large ({int(total_complexity)} > limit {MAX_EQUATION_COMPLEXITY})."
        )

    templates = tuple(
        (
            f"d(x4)/dt = {equation.expression}"
            if equation.kind == "ode"
            else f"{equation.target} = {equation.expression}"
        )
        for equation in normalized
    )
    canonical = "|".join(
        f"{equation.kind}:{equation.target}:{sp.srepr(equation.expression)}"
        for equation in normalized
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    penalty_rate = (
        EXTRA_EQUATION_PENALTY if extra_equation_penalty is None else float(extra_equation_penalty)
    )
    if penalty_rate < 0.0:
        raise ValueError("extra_equation_penalty must be nonnegative")
    return ResolvedEquationSystem(
        equations=tuple(normalized),
        equation_templates=templates,
        resolved_ode=resolved_ode,
        resolved_ode_template=str(resolved_ode),
        system_fingerprint=fingerprint,
        total_tree_complexity=total_complexity,
        structural_penalty=float((len(equations) - 1) * penalty_rate),
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
    fitted_constants = {str(symbol): float(value) for symbol, value in zip(constants, result.x)}
    substitutions = dict(zip(constants, result.x))
    fitted_expr = template.subs(substitutions)

    return FittedExpression(
        expression_template=template,
        expression=fitted_expr,
        constants=fitted_constants,
        y_pred_train=_predict(fitted_expr, X_train, feature_names),
        y_pred_val=_predict(fitted_expr, X_val, feature_names),
    )


def _evaluate_scalar_expression(
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
    """Internal scalar regression scorer used by the system API."""
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
    return result


_expression_scorer = _evaluate_scalar_expression


def evaluate_equation_system(
    equations: Sequence[CandidateEquation],
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
    *,
    constants: Sequence[sp.Symbol] | None = None,
    initial_values: Sequence[float] | None = None,
    max_nfev: int = 300,
    extra_equation_penalty: float | None = None,
) -> dict[str, Any]:
    """Resolve and score exactly one ordered observed-variable DAE system."""
    n_features = int(np.asarray(X_train).shape[1] or np.asarray(X_val).shape[1])
    system = resolve_equation_system(
        equations,
        constants=constants,
        n_features=n_features,
        extra_equation_penalty=extra_equation_penalty,
    )
    _record_scored_fingerprint(system.system_fingerprint)
    result = _expression_scorer(
        system.resolved_ode,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=constants,
        initial_values=initial_values,
        max_nfev=max_nfev,
    )
    raw_combined = float(result.get("combined_score", 0.0))
    result.update(
        {
            "equation_count": len(system.equations),
            "equation_templates": list(system.equation_templates),
            "resolved_ode_template": system.resolved_ode_template,
            "system_fingerprint": system.system_fingerprint,
            "total_tree_complexity": system.total_tree_complexity,
            "complexity": system.total_tree_complexity,
            "structural_penalty": system.structural_penalty,
            "raw_combined_score": raw_combined,
            "combined_score": raw_combined * max(0.0, 1.0 - system.structural_penalty),
        }
    )
    _record_scorer_result(result)
    return result


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
    """Backward-compatible one-equation wrapper around the system scorer."""
    if constants is None:
        n_features = int(np.asarray(X_train).shape[1] or np.asarray(X_val).shape[1])
        feature_names = [f"x{i}" for i in range(n_features)]
        template = _as_sympy_expr(expression, feature_names)
        constants = sorted(
            template.free_symbols - set(feature_symbols(n_features)),
            key=lambda symbol: symbol.name,
        )
    else:
        n_features = int(np.asarray(X_train).shape[1] or np.asarray(X_val).shape[1])
    if n_features != 5:
        template = _as_sympy_expr(expression, [f"x{i}" for i in range(n_features)])
        fingerprint = hashlib.sha256(
            f"ode:x{n_features - 1}:{sp.srepr(template)}".encode("utf-8")
        ).hexdigest()
        _record_scored_fingerprint(fingerprint)
        result = _expression_scorer(
            template,
            X_train,
            y_train,
            X_val,
            y_val,
            constants=constants,
            initial_values=initial_values,
            max_nfev=max_nfev,
        )
        raw_combined = float(result.get("combined_score", 0.0))
        result.update(
            {
                "equation_count": 1,
                "equation_templates": [str(template)],
                "resolved_ode_template": str(template),
                "system_fingerprint": fingerprint,
                "total_tree_complexity": _complexity(template),
                "structural_penalty": 0.0,
                "raw_combined_score": raw_combined,
            }
        )
        _record_scorer_result(result)
        return result
    return evaluate_equation_system(
        [ode_equation(expression)],
        X_train,
        y_train,
        X_val,
        y_val,
        constants=constants,
        initial_values=initial_values,
        max_nfev=max_nfev,
    )
