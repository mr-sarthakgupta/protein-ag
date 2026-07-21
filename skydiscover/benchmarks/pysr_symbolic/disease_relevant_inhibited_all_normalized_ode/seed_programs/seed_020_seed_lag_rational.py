# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: seed_lag_rational."""

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
    """Explore the seed lag rational kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    lag = c[0] ** 2 / (1 + c[1] ** 2 * seed + c[2] ** 2 * monomer)
    time_gate = time ** 2 / (lag ** 2 + time ** 2 + 0.01)
    source = c[3] ** 2 * monomer + c[4] ** 2 * seed + c[5] ** 2
    inhibition = 1 + c[6] ** 2 * inhibitor + c[7] ** 2 * inhibitor * time
    expression = time_gate * source * (1 + c[8] ** 2 * concentration) / inhibition - c[9] ** 2 * concentration

    return evaluate_expression(
        expression,
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
