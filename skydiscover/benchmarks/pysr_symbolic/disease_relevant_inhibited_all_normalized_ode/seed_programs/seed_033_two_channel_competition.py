# EVOLVE-BLOCK-START
"""Diverse inhibited ODE seed for normalized Abeta42 aggregation kinetics."""

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
    """Single-equation inhibited aggregation ODE seed."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration
    primary_inhibition = 1 + c[1] ** 2 * inhibitor + c[2] ** 2 * inhibitor * monomer
    secondary_inhibition = 1 + c[3] ** 2 * inhibitor * concentration + c[4] ** 2 * inhibitor * seed
    primary = (c[5] ** 2 + c[6] ** 2 * monomer) / primary_inhibition
    seeded = c[7] ** 2 * seed
    secondary = c[8] ** 2 * concentration * (monomer + seed) / secondary_inhibition
    expression = capacity * (primary + seeded + secondary) + c[9] * (time + concentration) * capacity + c[10] ** 2 * capacity

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
