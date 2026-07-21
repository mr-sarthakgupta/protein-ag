# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: mixed_additive_multiplicative."""

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
    """Explore the mixed additive multiplicative kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    additive = (c[1] ** 2 * monomer + c[2] ** 2 * seed + c[3] ** 2) / (1 + c[4] ** 2 * inhibitor)
    multiplicative = c[5] ** 2 * concentration * (plateau - concentration) / (1 + c[6] ** 2 * inhibitor * concentration)
    transient = c[7] * time * sp.exp(-c[8] ** 2 * time)
    expression = additive + multiplicative + transient - (c[9] ** 2 + c[10] ** 2 * inhibitor) * concentration

    return evaluate_expression(
        expression,
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
