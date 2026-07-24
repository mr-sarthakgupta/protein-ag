# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: transient_overshoot."""

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
    """Explore the transient overshoot kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    rise = sp.Symbol("rise")
    delayed_removal = sp.Symbol("delayed_removal")
    source = c[4] ** 2 * monomer + c[5] ** 2 * seed
    inhibitor_gate = sp.Symbol("inhibitor_gate")
    expression = (
        source / inhibitor_gate
        + c[8] ** 2 * rise
        - (c[9] ** 2 + c[10] ** 2 * delayed_removal) * concentration
    )
    equations = [
        algebraic_equation(
            rise,
            c[0] ** 2 * time * sp.exp(-c[1] ** 2 * time),
        ),
        algebraic_equation(
            delayed_removal,
            c[2] ** 2 * time**2 / (1 + c[3] ** 2 * time**2),
        ),
        algebraic_equation(
            inhibitor_gate,
            1 + c[6] ** 2 * inhibitor + c[7] ** 2 * inhibitor * time,
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
