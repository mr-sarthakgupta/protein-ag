# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: dose_dependent_plateau."""

from __future__ import annotations

from typing import Any

import sympy as sp
from numpy.typing import NDArray

from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Explore the dose dependent plateau kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0] / (1 + c[1] ** 2 * inhibitor + c[2] ** 2 * inhibitor ** 2)
    rate = c[3] ** 2 + c[4] ** 2 * monomer + c[5] ** 2 * seed
    state_gate = 1 + c[6] ** 2 * concentration + c[7] ** 2 * concentration ** 2
    expression = rate * state_gate * (plateau - concentration) - c[8] ** 2 * inhibitor * concentration

    return evaluate_expression(
        expression,
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
