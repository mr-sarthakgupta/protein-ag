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
    """Autonomous amyloid master-equation ODE (Cohen/Meisl/Knowles for
    Abeta42) with SATURATING surface-catalyzed secondary nucleation and a
    two-part inhibitor mechanism (onset delay + fibril-surface capping).

    The ODE is autonomous (no explicit time); lag, inflection, and plateau all
    emerge from the current fibril mass c=x4:

        dc/dt = (c0 - c) * (1 + c6^2*M0) * (c1^2*m0 + sec)
        sec   = c2^2 * m0 * c^2 / (c3^2 + c^2 + c4^2*cd + c5^2*cd*c)

    Data facts driving the design (single Abeta42 dataset, 96 curves):
    * Every curve plateaus near the global-normalized value ~1.0 regardless of
      M0, m0 or cd, so a single shared capacity ceiling c0 (fit near ~1.0) is
      appropriate; the parent's c0~=0.92 systematically under-shoots the true
      endpoints, so the initializer for c0 is raised to 1.05.
    * Curves span seeded (M0>0, higher normalized start) and unseeded (M0=0)
      conditions, and inhibitor cd ranges 0 -> 2e-6, producing both DELAYED
      onset and SLOWED growth -- the worst-curve behaviours the high-quantile
      shape loss targets.

    Terms:
    - (c0 - c): mass-conservation capacity -> smooth, non-overshooting plateau.
    - (1 + c6^2*M0): global seed acceleration for pre-seeded curves; leaves the
      cd=0 native kinetics unchanged and never negative.
    - c1^2*m0: inhibitor-INDEPENDENT primary nucleation from free monomer that
      bootstraps unseeded curves and sets baseline onset (surface-active small
      molecules do not block homogeneous primary nucleation).
    - sec: SATURATING secondary nucleation. The c^2 numerator holds the rate
      near zero during the lag; the +c^2 in the denominator saturates the
      per-mass catalytic rate at high fibril mass, yielding a sharp-then-
      bounded sigmoid that tightens the mid/late (50/75/90%) crossing timing.

    Two-part inhibition inside the SAME saturating denominator, both matching
    distinct molecular pictures and vanishing exactly at cd=0:
    - c4^2*cd     : a c-INDEPENDENT term that suppresses secondary nucleation
      even while c is still small, LENGTHENING THE LAG and delaying the early
      10%/25% crossings that dominate the worst-curve shape term. This is the
      key addition over the parent, which could only slow growth once mass had
      already accumulated and therefore mis-timed delayed inhibitor onsets.
    - c5^2*cd*c   : fibril-SURFACE CAPPING that strengthens with existing mass
      (more surface to coat), slowing the mid/late growth of inhibitor curves.

    Numerical safety: denominator = c3^2 + c^2 + (nonneg) >= c3^2 > 0 for a
    nonzero c3 initializer, so the RHS is smooth, finite, and real over the
    whole input range; capacity self-limits growth so odeint stays stable.
    Seven non-negative-coefficient constants keep least-squares reliable from a
    single initializer.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Mass-conservation capacity: trajectory relaxes smoothly to plateau c0
    # without overshooting or going negative. c0 initialized near the observed
    # universal plateau (~1.0) rather than the parent's under-shooting ~0.92.
    capacity = c[0] - concentration

    # SATURATING secondary nucleation with a two-part inhibitor denominator.
    # * c^2 numerator -> flat lag; +c^2 denominator -> Michaelis-type
    #   saturation sharpening the mid/late crossing timing.
    # * c4^2*cd (state-INDEPENDENT) delays ONSET / stretches the lag of
    #   inhibitor curves -- the dominant worst-curve failure mode.
    # * c5^2*cd*c (surface capping) slows the accelerating growth phase.
    # Both inhibitor terms are >=0, added to a strictly positive denominator,
    # so no singularity is possible and cd=0 exactly recovers native kinetics.
    secondary = (
        c[2] ** 2 * monomer * concentration ** 2
        / (
            c[3] ** 2
            + concentration ** 2
            + c[4] ** 2 * inhibitor
            + c[5] ** 2 * inhibitor * concentration
        )
    )
    growth = c[1] ** 2 * monomer + secondary
    seed_gain = 1 + c[6] ** 2 * seed
    expression = capacity * growth * seed_gain

    # Initializer: c0=1.05 targets the true ~1.0 plateau; weak primary (c1)
    # vs stronger saturating autocatalysis (c2) gives a long flat lag then a
    # sharp rise; c3=0.5 keeps the denominator strictly positive at c=0; the
    # two inhibitor coefficients (c4 onset-delay, c5 surface-capping) start
    # well-separated so the fit can distribute delayed vs slowed inhibition.
    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.05, 0.1, 0.4, 0.5, 1.2, 1.0, 0.8],
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
