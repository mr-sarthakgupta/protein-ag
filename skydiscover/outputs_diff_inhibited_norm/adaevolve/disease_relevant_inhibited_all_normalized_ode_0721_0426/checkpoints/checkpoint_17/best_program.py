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
    phase without altering the cd=0 limit. The secondary-nucleation flux uses
    a single power-law self-replication term smooth_state**(1 + c6^2) with a
    fitted fractional reaction order (>= 1), so each dataset adapts the burst
    steepness to its data. A monomer-depletion capacity factor (plateau - c)
    enforces conservation of mass. All operations are smooth: denominators are
    1 + (nonnegative), and sqrt(c^2 + eps + floor) keeps smooth_state strictly
    positive so the fractional power is always real, finite and differentiable.

    Features: x0 = normalized time, x1 = m0, x2 = M0 seed, x3 = cd inhibitor,
    x4 = current normalized fibril mass c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Strictly-positive, smooth self-replication surface measure (fibril mass).
    # sqrt(c^2 + c0^2) is smooth and differentiable at c=0, avoiding any
    # fractional-power / NaN pathology and keeping the 8-constant least-squares
    # fit well conditioned from one start (the fitted exponent in the parent
    # doubled runtime and destabilised the fit for little NMSE gain).
    smooth_state = sp.sqrt(concentration ** 2 + c[0] ** 2)

    # Effective surface-catalysed growth flux (elongation + secondary
    # nucleation), monomer- and mass-fed. This is the amyloid "kappa" channel
    # whose net rate sets the sigmoid half-time.
    growth = c[1] ** 2 * monomer * smooth_state * concentration

    # Inhibitor acts on the growth channel through TWO physically distinct,
    # cd=0-preserving mechanisms (both exactly 1 at cd=0):
    #  * TIMING: a time-fraction gate x0/(x0 + tau) whose effective half-rise
    #    time tau = c2^2 + c3^2*cd is LENGTHENED by inhibitor, delaying WHEN
    #    the autocatalytic burst ignites. This directly moves the 10/25/50/75%
    #    response-crossing terms that dominate the shape loss.
    #  * RATE: a bounded Langmuir factor 1/(1 + c4^2*cd) that also mildly
    #    lowers the burst amplitude/slope (surface coating), letting inhibited
    #    curves bend as well as shift so slope-profile shape error drops.
    # Denominators are x0 + (positive) > 0 on x0 in [0,1] and 1 + (nonneg),
    # so the whole gate is smooth and bounded in [0,1).
    time_gate = time / (time + c[2] ** 2 + c[3] ** 2 * inhibitor)
    rate_gate = 1 / (1 + c[4] ** 2 * inhibitor)

    # Monomer/seed-fed primary nucleation + elongation ignition (ungated ->
    # preserves the cd=0 limit and the seeded/unseeded early baseline that
    # sets the c=0 lag). Kept as a small additive source so the burst can
    # start from c=0 for unseeded curves.
    ignition = c[5] ** 2 * monomer + c[6] ** 2 * seed

    # Monomer-depletion capacity (conservation of mass); plateau stays
    # inhibitor-independent so dose reshapes timing/slope, not the endpoint.
    plateau = c[7]
    capacity = plateau - concentration

    expression = capacity * (ignition + time_gate * rate_gate * growth)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 1.0, 0.3, 0.3, 0.5, 0.1, 1.0, 1.0],
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
