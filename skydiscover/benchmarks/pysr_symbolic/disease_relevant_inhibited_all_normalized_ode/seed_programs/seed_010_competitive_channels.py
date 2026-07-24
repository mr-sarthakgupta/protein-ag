# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: competitive_channels."""

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
    """Explore the competitive channels kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    primary = sp.Symbol("primary")
    seeded = c[3] ** 2 * seed / (1 + c[4] ** 2 * inhibitor * seed)
    secondary = sp.Symbol("secondary")
    fragmentation = c[7] ** 2 * concentration**2 / (1 + c[8] ** 2 * inhibitor)
    expression = (c[9] - concentration) * (primary + seeded + secondary + fragmentation) - c[
        10
    ] ** 2 * inhibitor * concentration
    equations = [
        algebraic_equation(
            primary,
            (c[0] ** 2 * monomer + c[1] ** 2) / (1 + c[2] ** 2 * inhibitor),
        ),
        algebraic_equation(
            secondary,
            c[5] ** 2 * monomer * concentration / (1 + c[6] ** 2 * inhibitor * concentration),
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
        initial_values=[0.7, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5],
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
