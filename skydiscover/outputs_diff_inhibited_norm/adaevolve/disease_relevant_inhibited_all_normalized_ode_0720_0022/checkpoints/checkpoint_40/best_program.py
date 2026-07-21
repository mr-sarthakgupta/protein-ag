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
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Secondary-nucleation master-equation structure (Knowles/Cohen/Meisl) with
    # a mass-conservation ceiling (plateau - c). Two aggregation channels:
    #   * source: primary nucleation (monomer) + seed head-start; only weakly
    #     inhibitor-sensitive, so the cd=0 baseline and early lag are preserved.
    #   * secondary: surface-catalysed autocatalytic growth (monomer * fibril
    #     mass), the dominant channel that sets steepness and take-off timing.
    # Inhibitors bind the growing fibril surface, so they suppress the SECONDARY
    # channel far more than bulk primary nucleation. The data show the whole
    # curve stretches ~6x in t50 as cd rises, so the secondary gate combines a
    # dose term (c1^2*cd), a state-coupled term that strengthens as fibril
    # surface c grows (c2^2*cd*c), and a mild superlinear dose term
    # (c8^2*cd^2) to reproduce the strong high-dose delay. Every denominator is
    # 1 + (nonneg)*(nonneg feature) >= 1, so the RHS is smooth and never
    # accelerates growth; cd=0 recovers the uninhibited curve exactly.
    plateau = c[0]
    capacity = plateau - concentration
    source = c[3] ** 2 + c[4] ** 2 * monomer + c[5] ** 2 * seed
    secondary = c[6] ** 2 * monomer * concentration * (1 + c[7] ** 2 * seed)
    primary_gate = 1 + c[9] ** 2 * inhibitor
    secondary_gate = 1 + inhibitor * (c[1] ** 2 + c[2] ** 2 * concentration) + c[8] ** 2 * inhibitor ** 2
    expression = capacity * (source / primary_gate + secondary / secondary_gate)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
