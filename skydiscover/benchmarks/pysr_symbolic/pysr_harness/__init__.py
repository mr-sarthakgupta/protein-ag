"""Shared PySR-backed symbolic regression utilities for SkyDiscover benchmarks."""

from pysr_harness.equation_session import (
    CandidateEquation,
    algebraic_equation,
    constant_symbols,
    evaluate_equation_system,
    evaluate_expression,
    feature_symbols,
    fit_expression_constants,
    ode_equation,
    pysr_operator_names,
    pysr_operator_namespace,
    resolve_equation_system,
)
from pysr_harness.metrics import combined_score_from_nmse, nmse
from pysr_harness.operators import (
    default_constraints,
    default_operators,
    general_operators,
    operator_config,
)

__all__ = [
    "feature_symbols",
    "constant_symbols",
    "CandidateEquation",
    "ode_equation",
    "algebraic_equation",
    "resolve_equation_system",
    "evaluate_equation_system",
    "fit_expression_constants",
    "evaluate_expression",
    "pysr_operator_names",
    "pysr_operator_namespace",
    "nmse",
    "combined_score_from_nmse",
    "default_operators",
    "general_operators",
    "default_constraints",
    "operator_config",
    "run_gp_session",
    "build_pysr_regressor",
    "default_gp_config",
]


def __getattr__(name):
    """Keep legacy imports available without eagerly importing PySRRegressor."""
    if name in {"build_pysr_regressor", "default_gp_config"}:
        from pysr_harness.backend import build_pysr_regressor, default_gp_config

        return {
            "build_pysr_regressor": build_pysr_regressor,
            "default_gp_config": default_gp_config,
        }[name]
    if name == "run_gp_session":
        from pysr_harness.gp_session import run_gp_session

        return run_gp_session
    raise AttributeError(name)
