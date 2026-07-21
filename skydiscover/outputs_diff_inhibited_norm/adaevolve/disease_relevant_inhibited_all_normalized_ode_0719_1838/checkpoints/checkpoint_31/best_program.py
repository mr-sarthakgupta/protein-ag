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
    """Autonomous amyloid ODE with SATURATING (Michaelis/Hill-2) self-catalysis
    in fibril mass and an inhibitor that caps the accessible fibril surface.

    Grounded in the Cohen/Meisl/Knowles Abeta42 master equation, where
    secondary nucleation SATURATES at high monomer/fibril load because the
    fibril surface has a finite density of catalytic sites; this yields a
    RATIONAL (Michaelis-type) rate law rather than an unbounded power law.
    Surface-active Abeta42 inhibitors (e.g. Brichos-type) act by binding the
    fibril surface and blocking those secondary-nucleation sites, motivating a
    surface-capping term whose strength grows with existing fibril mass.

    Autonomous RHS (no explicit time; lag, inflection and plateau all emerge
    from the current state c=x4):

        dc/dt = (c0 - c) * (c1^2*m0 + sec) * (1 + c6^2*M0)
        sec   = c2^2 * m0 * c^2 / (c3^2 + c^2 + c4^2*cd*c)

    - (c0 - c): mass-conservation capacity -> smooth plateau at c0.
    - c1^2*m0: inhibitor-independent primary nucleation setting baseline onset.
    - sec: saturating secondary nucleation. The c^2 numerator gives a flat lag;
      the +c^2 in the denominator saturates the per-mass catalytic rate at high
      fibril mass, yielding a steep-then-plateau sigmoid that sharpens the
      mid/late (50/75/90%) crossing timing dominating the shape loss.
    - c4^2*cd*c: fibril-surface capping by the inhibitor. Added to the
      denominator, it grows with existing mass (more surface to coat), delaying
      inhibited curves; it vanishes at cd=0, recovering uninhibited kinetics,
      and can never cause a singularity.
    - (1 + c6^2*M0): global seed acceleration for pre-seeded curves.

    Numerical safety: denominator = c3^2 + c^2 + nonneg >= c3^2 > 0 (c3 init
    0.5), so the RHS is smooth, finite and real everywhere; capacity self-
    limits growth so odeint stays stable. Seven non-negative-coefficient
    constants keep least-squares fitting reliable from one initializer.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Mass-conservation capacity: the trajectory relaxes smoothly to the
    # plateau c0 without overshooting or going negative.
    capacity = c[0] - concentration

    # SATURATING (Michaelis / Hill-2) SELF-CATALYSIS in fibril mass c=x4.
    # Instead of a linear-c or superlinear-c autocatalytic term (which keeps
    # accelerating), the secondary-nucleation rate here saturates at high
    # fibril mass:
    #
    #     sec = c2^2 * m0 * c^2 / (c3^2 + c^2 + c4^2 * cd * c)
    #
    # * The c^2 NUMERATOR keeps the rate near-zero while c is small, producing
    #   a genuinely flat lag phase (correct amyloid onset).
    # * The +c^2 term in the DENOMINATOR is the Michaelis-type saturation: as
    #   fibril mass grows, the per-mass catalytic rate levels off, so the
    #   mid/late phase transitions from a sharp inflection into a bounded
    #   approach to the plateau. This steep-then-saturate profile directly
    #   sharpens the 50%/75%/90% response-crossing timing that dominates the
    #   curve-level shape loss.
    # * The c4^2 * cd * c term models fibril-SURFACE CAPPING by the inhibitor:
    #   the accessible catalytic surface shrinks as more fibril mass forms
    #   (more surface to coat), so suppression grows with existing mass. It is
    #   ADDED to the denominator, so it delays inhibited curves without ever
    #   creating a singularity, and it VANISHES when cd=0, exactly recovering
    #   native uninhibited kinetics.
    #
    # Numerical safety: denominator = c3^2 + c^2 + (nonneg) >= c3^2 > 0 for a
    # nonzero c3 initializer (0.5), so the RHS is smooth, finite, and real
    # everywhere over the observed input ranges; capacity (c0-c) self-limits
    # growth so odeint stays stable.
    sec = (
        c[2] ** 2 * monomer * concentration ** 2
        / (c[3] ** 2 + concentration ** 2 + c[4] ** 2 * inhibitor * concentration)
    )

    # Primary nucleation from free monomer sets the baseline onset and is left
    # inhibitor-independent (surface-active small molecules do not block
    # homogeneous primary nucleation). The seed multiplier (1 + c6^2*M0)
    # accelerates pre-seeded curves globally without duplicating the seed
    # dependence inside the saturating channel.
    growth = c[1] ** 2 * monomer + sec
    seed_gain = 1 + c[6] ** 2 * seed
    expression = capacity * growth * seed_gain

    # Initializer: small primary nucleation (c1) versus a moderate saturating
    # autocatalysis (c2) biases toward a long flat lag then a sharp saturating
    # rise; c3=0.5 keeps the denominator strictly positive at c=0; c4 gives
    # the inhibitor surface-capping room to delay held-out inhibitor curves.
    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.05, 0.3, 2.0, 0.5, 1.5, 0.8],
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