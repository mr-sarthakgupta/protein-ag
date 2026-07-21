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
    Knowles/Cohen/Meisl-style amyloid mass-balance RHS with a fitted
    fractional secondary-nucleation reaction order.

    Fibril mass c grows by (i) primary nucleation from free monomer, (ii)
    elongation from existing seed/fibril ends, and (iii) autocatalytic
    surface-catalysed secondary nucleation. In the chemical-kinetics master
    equation for Abeta42 the secondary-nucleation flux scales as a non-integer
    power of the fibril mass (self-replication order between 1 and 2), which
    sets how sharply the sigmoidal growth burst accelerates. Rather than a
    fixed (state + state^2) approximation, this template uses a single
    power-law flux smooth_state**(1 + c6^2) with a *fitted fractional
    reaction order* 1 + c6^2 >= 1, giving each inhibitor dose one clean
    degree of freedom to match the observed burst steepness (the slope-profile
    component of the shape loss).

    The inhibitor cd binds the fibril surface and suppresses secondary
    nucleation via a bounded Langmuir occupancy factor 1/(1 + K*cd), which
    lengthens the lag phase without altering the cd=0 limit. A smooth transient
    maturation gate controls WHEN the surface-catalysed burst turns on, with
    an inhibitor-lengthened half-time that targets the 10%/25%/50% response-
    crossing terms of the shape loss on delayed inhibitor curves. A
    monomer-depletion capacity factor (plateau - c) enforces conservation of
    mass. All operations are smooth: denominators are 1 + (nonnegative), and
    sqrt(c^2 + c0^2 + 1e-4) keeps smooth_state strictly positive so the
    fractional power is always real, finite and differentiable at c = 0.

    Features: x0 = normalized time, x1 = m0, x2 = M0 seed, x3 = cd inhibitor,
    x4 = current normalized fibril mass c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Strictly-positive, smooth self-replication measure (secondary-nucleation
    # surface). sqrt(c^2 + c0^2 + 1e-4) keeps smooth_state > 0 so the
    # fractional power below is always real, finite and differentiable at c=0.
    smooth_state = sp.sqrt(concentration ** 2 + c[0] ** 2 + 1e-4)

    # Saturable inhibitor occupancy on the fibril surface (Langmuir, bounded).
    inhibitor_gate = 1 / (1 + c[1] ** 2 * inhibitor)

    # Primary nucleation from free monomer (seeds unseeded M0=0 curves).
    primary = c[2] ** 2 * monomer

    # Elongation from existing fibril ends (seed + current mass).
    elongation = c[3] ** 2 * (seed + c[4] ** 2 * concentration)

    # BREAKTHROUGH: variable-exponent self-replication flux. Instead of the
    # fixed additive (state + state^2) superlinear form, use a single
    # power-law term smooth_state**(1 + c6^2) with a *fitted fractional
    # reaction order* 1 + c6^2 >= 1. Meisl/Knowles report non-integer
    # secondary-nucleation scaling exponents for Abeta42 (typically 1-2), and
    # this exponent directly controls how sharply the growth burst
    # accelerates, i.e. the slope-profile component of the shape loss. The
    # exponent is bounded away from 0 (never < 1) so derivatives at c=0 stay
    # finite; smooth_state ~ O(1) after normalization so the power stays
    # bounded and cannot overflow.
    secondary = (
        c[5] ** 2
        * monomer
        * smooth_state ** (1 + c[6] ** 2)
        * inhibitor_gate
    )

    # Monomer-depletion capacity (conservation of mass).
    plateau = c[7]
    capacity = plateau - concentration

    # Smooth transient maturation gate: controls WHEN the surface-catalysed
    # burst turns on; the inhibitor lengthens the effective half-time c8^2.
    # This targets the 10%/25%/50% response-crossing terms of the shape loss
    # on delayed inhibitor curves while leaving the cd=0 limit intact.
    maturation = time / (c[8] ** 2 * (1 + c[9] ** 2 * inhibitor) + time)

    expression = capacity * (primary + maturation * (elongation + secondary))

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.05, 1.0, 0.01, 1.0, 1.0, 1.0, 0.5, 1.0, 0.3, 0.1],
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