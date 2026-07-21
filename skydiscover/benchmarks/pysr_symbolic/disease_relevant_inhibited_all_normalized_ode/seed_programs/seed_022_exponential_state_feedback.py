# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: exponential_state_feedback."""

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
    """Explore the exponential state feedback kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    activation = sp.exp(-c[0] ** 2 * concentration)
    inhibitor_decay = sp.exp(-c[1] ** 2 * inhibitor * time)
    source = c[2] ** 2 * monomer + c[3] ** 2 * seed + c[4] ** 2
    feedback = c[5] ** 2 * concentration * activation
    expression = source * inhibitor_decay + feedback - (c[6] ** 2 + c[7] ** 2 * inhibitor) * concentration + c[8] * activation

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
