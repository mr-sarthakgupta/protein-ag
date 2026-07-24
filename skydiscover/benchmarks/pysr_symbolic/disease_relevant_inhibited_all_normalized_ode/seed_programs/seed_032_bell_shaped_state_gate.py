# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: bell_shaped_state_gate."""

from __future__ import annotations

from typing import Any

import sympy as sp
from numpy.typing import NDArray

from pysr_harness.equation_session import (
    algebraic_equation,
    constant_symbols,
    evaluate_equation_system,
    feature_symbols,
    ode_equation,
)


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Explore the bell shaped state gate kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    center = c[0]
    width = sp.Symbol("width")
    bell = sp.Symbol("bell")
    source = sp.Symbol("source")
    growth = sp.Symbol("growth")
    expression = source + growth - (c[8] ** 2 + c[9] ** 2 * inhibitor) * concentration
    equations = [
        algebraic_equation(
            width,
            c[1] ** 2 + c[2] ** 2 * inhibitor + 0.01,
        ),
        algebraic_equation(
            bell,
            sp.exp(-((concentration - center) ** 2) / width),
        ),
        algebraic_equation(
            source,
            c[3] ** 2 * monomer + c[4] ** 2 * seed + c[5] ** 2,
        ),
        algebraic_equation(
            growth,
            c[6] ** 2 * bell * (1 + c[7] ** 2 * concentration),
        ),
        ode_equation(expression),
    ]

    return evaluate_equation_system(
        equations,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.7, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0],
    )


def run_discovery(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Entry point used by the evaluator subprocess."""
    return evaluate_symbolic_candidate(X_train, y_train, X_val, y_val)


# EVOLVE-BLOCK-END
