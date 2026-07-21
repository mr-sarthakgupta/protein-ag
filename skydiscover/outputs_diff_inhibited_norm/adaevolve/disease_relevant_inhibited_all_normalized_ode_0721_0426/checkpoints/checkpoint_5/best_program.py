# EVOLVE-BLOCK-START
"""Mechanistic Abeta42 aggregation ODE with saturable inhibitor suppression."""

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
    """
    Knowles/Cohen/Meisl-style amyloid mass-balance RHS.

    Fibril mass c grows by (i) primary nucleation from free monomer, (ii)
    elongation from existing seed/fibril ends, and (iii) autocatalytic
    secondary nucleation catalysed by the fibril surface. The inhibitor cd
    binds the fibril surface and suppresses secondary nucleation via a
    bounded Langmuir occupancy factor 1/(1 + K*cd), which lengthens the lag
    phase without altering the cd=0 limit. A monomer-depletion capacity
    factor (plateau - c) enforces conservation of mass. All operations are
    smooth: denominators are 1 + (nonnegative), and sqrt(c^2 + eps) keeps the
    superlinear secondary term differentiable at c = 0.

    Features: x0 = normalized time, x1 = m0, x2 = M0 seed, x3 = cd inhibitor,
    x4 = current normalized fibril mass c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Bounded, smooth self-replication measure (secondary nucleation surface).
    smooth_state = sp.sqrt(concentration ** 2 + c[0] ** 2)

    # Saturable inhibitor occupancy on the fibril surface (Langmuir, bounded).
    inhibitor_gate = 1 / (1 + c[1] ** 2 * inhibitor)

    # Primary nucleation from free monomer (seeds unseeded M0=0 curves).
    primary = c[2] ** 2 * monomer

    # Elongation from existing fibril ends (seed + current mass).
    elongation = c[3] ** 2 * (seed + c[4] ** 2 * concentration)

    # Autocatalytic secondary nucleation: monomer-fed, surface-catalysed,
    # suppressed by inhibitor. Superlinear via (state + state^2).
    secondary = (
        c[5] ** 2
        * monomer
        * (smooth_state + c[6] ** 2 * smooth_state ** 2)
        * inhibitor_gate
    )

    # Monomer-depletion capacity (conservation of mass).
    plateau = c[7]
    capacity = plateau - concentration

    # Small inhibitor-driven off-pathway sink (fibril fragmentation/clearance).
    sink = c[8] ** 2 * inhibitor * concentration / (1 + c[9] ** 2 * concentration)

    expression = capacity * (primary + elongation + secondary) - sink

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.05, 1.0, 0.01, 1.0, 1.0, 1.0, 0.5, 1.0, 0.01, 1.0, 0.0],
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
