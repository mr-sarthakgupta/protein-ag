# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: hill_autocatalysis."""

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
    """Explore the hill autocatalysis kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    state_square = concentration ** 2
    hill_gate = state_square / (c[1] ** 2 + state_square)
    source = c[2] ** 2 * monomer + c[3] ** 2 * seed + c[4] ** 2
    growth = c[5] ** 2 * monomer * hill_gate
    source_block = 1 + c[6] ** 2 * inhibitor
    growth_block = 1 + c[7] ** 2 * inhibitor + c[8] ** 2 * inhibitor * concentration
    expression = (plateau - concentration) * (source / source_block + growth / growth_block) - c[9] ** 2 * inhibitor * concentration

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
