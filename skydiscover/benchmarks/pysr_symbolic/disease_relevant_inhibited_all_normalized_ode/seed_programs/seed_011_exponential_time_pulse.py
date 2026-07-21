# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: exponential_time_pulse."""

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
    """Explore the exponential time pulse kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    pulse_rate = c[0] ** 2 + c[1] ** 2 * inhibitor + c[2] ** 2 * seed
    transient = time * sp.exp(-pulse_rate * time)
    source = c[3] ** 2 * monomer + c[4] ** 2 * seed
    autocatalysis = c[5] ** 2 * concentration * sp.exp(-c[6] ** 2 * inhibitor * time)
    expression = (source + c[7] ** 2 * transient + autocatalysis) / (1 + c[8] ** 2 * inhibitor) - c[9] ** 2 * concentration

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
