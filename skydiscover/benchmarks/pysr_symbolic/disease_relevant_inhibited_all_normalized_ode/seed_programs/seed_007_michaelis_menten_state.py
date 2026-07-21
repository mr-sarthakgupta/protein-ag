# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: michaelis_menten_state."""

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
    """Explore the michaelis menten state kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    source = c[0] ** 2 * monomer + c[1] ** 2 * seed + c[2] ** 2
    saturated_state = concentration / (c[3] ** 2 + concentration + c[4] ** 2)
    inhibitor_scale = 1 + c[5] ** 2 * inhibitor
    production = source * (1 + c[6] ** 2 * saturated_state) / inhibitor_scale
    removal = c[7] ** 2 * concentration / (1 + c[8] ** 2 * concentration)
    expression = production - removal

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
