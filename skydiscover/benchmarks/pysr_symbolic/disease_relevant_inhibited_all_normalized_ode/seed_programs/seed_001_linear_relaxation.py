# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: linear_relaxation."""

from __future__ import annotations

from typing import Any

import sympy as sp
from numpy.typing import NDArray

from pysr_harness.equation_session import (
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
    """Explore the linear relaxation kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    equilibrium = c[0] / (1 + c[1] ** 2 * inhibitor)
    rate = c[2] ** 2 + c[3] ** 2 * monomer + c[4] ** 2 * seed
    removal = c[5] ** 2 * inhibitor * concentration / (1 + c[6] ** 2 * concentration)
    expression = rate * (equilibrium - concentration) - removal

    equations = [ode_equation(expression)]

    return evaluate_equation_system(
        equations,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.7, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0],
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
