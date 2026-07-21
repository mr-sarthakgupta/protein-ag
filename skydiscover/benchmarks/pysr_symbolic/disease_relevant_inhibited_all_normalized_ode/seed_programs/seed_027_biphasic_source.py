# EVOLVE-BLOCK-START
"""Structurally diverse inhibited ODE seed: biphasic_source."""

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
    """Explore the biphasic source kinetic family."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    fast_source = c[0] ** 2 * sp.exp(-c[1] ** 2 * time)
    slow_source = c[2] ** 2 * (1 - sp.exp(-c[3] ** 2 * time))
    monomer_drive = c[4] ** 2 * monomer
    seed_drive = c[5] ** 2 * seed
    early_block = 1 + c[6] ** 2 * inhibitor
    late_block = 1 + c[7] ** 2 * inhibitor * time
    expression = (monomer_drive + seed_drive) * (fast_source / early_block + slow_source / late_block) - (c[8] ** 2 + c[9] ** 2 * inhibitor) * concentration + c[10] ** 2

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
