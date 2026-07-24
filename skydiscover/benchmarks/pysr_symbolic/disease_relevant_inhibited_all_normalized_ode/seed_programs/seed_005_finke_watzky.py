# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: finke_watzky."""

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
    """Explore the finke watzky kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    free_monomer = sp.Symbol("free_monomer")
    nucleation = c[1] ** 2 * free_monomer + c[2] ** 2 * seed
    autocatalysis = c[3] ** 2 * free_monomer * concentration
    available = c[4] / (1 + c[5] ** 2 * inhibitor) - concentration
    surface_block = 1 + c[6] ** 2 * inhibitor * concentration + c[7] ** 2 * concentration**2
    expression = (
        available * (nucleation + autocatalysis / surface_block)
        - c[8] ** 2 * inhibitor * concentration
    )
    equations = [
        algebraic_equation(
            free_monomer,
            monomer / (1 + c[0] ** 2 * inhibitor),
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
        initial_values=[0.7, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0],
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
