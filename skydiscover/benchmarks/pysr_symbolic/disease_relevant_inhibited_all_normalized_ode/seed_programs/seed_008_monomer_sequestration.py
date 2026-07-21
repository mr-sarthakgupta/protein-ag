# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: monomer_sequestration."""

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
    """Explore the monomer sequestration kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    bound_monomer = monomer * inhibitor / (c[0] ** 2 + inhibitor + c[1] ** 2)
    free_monomer = monomer / (1 + c[2] ** 2 * bound_monomer)
    source = c[3] ** 2 * free_monomer + c[4] ** 2 * seed
    secondary = c[5] ** 2 * free_monomer * concentration
    expression = (c[6] - concentration) * (source + secondary) - c[7] ** 2 * bound_monomer * concentration / (1 + c[8] ** 2 * concentration)

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
