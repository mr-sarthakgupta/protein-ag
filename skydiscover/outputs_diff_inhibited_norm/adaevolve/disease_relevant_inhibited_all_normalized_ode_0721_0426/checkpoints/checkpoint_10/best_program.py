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

    Breakthrough: an explicit, smooth time-maturation gate is applied to the
    surface-catalysed fluxes (elongation + secondary nucleation) so that the
    onset timing of the sigmoidal burst is adjustable and inhibitor-delayable,
    instead of being fixed by pure state-autocatalysis. The gate

        m = x0 / (c9^2 * (1 + c10^2 * cd) + x0)

    rises smoothly from 0 at t=0 toward 1, with a half-time c9^2 that the
    inhibitor cd pushes later (larger effective half-time). This targets the
    10%/25%/50% response-crossing terms of the shape loss on delayed inhibitor
    curves. Primary nucleation is left ungated so seeded curves still leave
    zero. The denominator is >= c9^2 > 0 (constants squared), so m is bounded
    in [0, 1) and globally smooth; x0 is min-max normalized to [0, 1] so the
    gate is well scaled.

    Features: x0 = normalized time, x1 = m0, x2 = M0 seed, x3 = cd inhibitor,
    x4 = current normalized fibril mass c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
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
    sink = c[8] ** 2 * inhibitor * concentration

    # Smooth transient maturation gate: controls WHEN the surface-catalysed
    # burst turns on; the inhibitor lengthens the effective half-time c9^2.
    maturation = time / (c[9] ** 2 * (1 + c[10] ** 2 * inhibitor) + time)

    expression = (
        capacity * (primary + maturation * (elongation + secondary)) - sink
    )

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.05, 1.0, 0.01, 1.0, 1.0, 1.0, 0.5, 1.0, 0.01, 0.3, 0.1],
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
