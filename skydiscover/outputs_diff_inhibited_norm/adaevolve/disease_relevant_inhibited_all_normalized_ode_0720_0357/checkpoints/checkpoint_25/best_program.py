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
    c = constant_symbols(8)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration
    free_monomer = monomer / (1 + c[1] ** 2 * inhibitor + c[2] ** 2 * inhibitor * monomer)
    source = c[3] ** 2 + c[4] ** 2 * free_monomer + c[5] ** 2 * seed
    # Autonomous secondary nucleation with surface-saturation (Meisl/Cohen/Knowles
    # amyloid master equation). The numerator keeps the crisp autocatalytic rise
    # (superlinear in fibril mass c and seed), while the saturable denominator
    # 1 + c7^2*c^2 models the fibril surface becoming a limiting catalytic
    # resource as mass accumulates -- the secondary rate rises, then rolls over.
    # This physically-grounded roll-over reshapes the sigmoid knee and shifts the
    # 25/50/75% response-crossing timings of the delayed inhibitor curves (where
    # the curve-level shape loss, ~25% of score, is concentrated) without the
    # fragile non-autonomous time*c brake used before. Denominator is >= 1 for
    # all c, so the RHS is globally smooth, finite, and singularity-free; at
    # cd=0 the inhibitor terms vanish, preserving clean uninhibited behavior.
    secondary = c[6] ** 2 * free_monomer * concentration * (1 + concentration + c[0] ** 2 * seed) / (1 + c[7] ** 2 * concentration ** 2)
    expression = capacity * (source + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3],
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
