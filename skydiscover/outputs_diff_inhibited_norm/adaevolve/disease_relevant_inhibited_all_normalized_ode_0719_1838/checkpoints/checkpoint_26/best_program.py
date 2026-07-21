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
    """Autonomous amyloid master-equation ODE with inhibitor targeting only
    the fibril-surface-catalyzed autocatalytic (secondary-nucleation) channel.

    Physical structure (Cohen/Meisl/Knowles master equation for Abeta42), with
    NO explicit time dependence so the ODE is autonomous and its lag,
    inflection, and plateau all emerge from the current state c=x4:

        dc/dt = capacity * (primary + secondary_inhibited)

      capacity = c0 - c                        (mass conservation -> plateau)
      primary  = c1^2 + c2^2*m0                (primary nucleation from free
                 monomer m0=x1; inhibitor-INDEPENDENT because surface-active
                 small molecules do not block homogeneous primary nucleation.
                 The c1^2 baseline bootstraps unseeded, delayed M0=0 curves.)
      secondary_inhibited =
          c3^2 * m0 * c * (1 + c4^2*M0)
          / (1 + c5^2*cd + c6^2*cd*c)
                 (surface-catalyzed secondary nucleation / autocatalytic
                  elongation: monomer m0 supplies material, the linear c
                  factor makes growth self-accelerating so the sigmoidal lag
                  and inflection timing appear autonomously; the seed factor
                  (1 + c4^2*M0) speeds seeded curves. The inhibitor cd=x3
                  divides ONLY this autocatalytic term, matching the molecular
                  picture that inhibitors coat/cap fibril surfaces and block
                  secondary nucleation while leaving primary nucleation
                  intact. Suppression strengthens with fibril mass c because
                  more surface becomes available to coat, producing the
                  near-proportional delay of the early response crossings that
                  dominate the high-quantile shape loss.)

    Numerical safety: the denominator is >= 1 everywhere (cd>=0 and squared
    coefficients are non-negative), so there is no singularity, overflow, or
    non-real value; when cd=0 it is exactly 1, preserving native uninhibited
    kinetics. Capacity (c0-c) self-limits the growth so c relaxes stably to
    the plateau c0 without going negative. Seven constants keep least-squares
    fitting reliable from a single initializer, and the non-physical explicit
    time term of the parent is removed to improve held-out-curve extrapolation.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Two physically distinct inhibition channels drawn from the Abeta42
    # master equation (Cohen/Meisl/Knowles). Both are >=1 denominators built
    # from squared coefficients and non-negative inputs (cd>=0, c>=0), so the
    # RHS stays smooth, finite, and singularity-free; at cd=0 both reduce to 1
    # exactly, recovering native uninhibited kinetics.
    #
    # (1) Monomer sequestration: the inhibitor binds free monomer, lowering
    # the material available to build fibrils. m_eff feeds BOTH nucleation
    # channels, so a single mechanism lowers the reachable PLATEAU and slows
    # the overall rate together, as expected when buildable monomer is removed.
    m_eff = monomer / (1 + c[3] ** 2 * inhibitor)

    # Mass-conservation capacity: trajectory relaxes smoothly to plateau c0
    # without overshooting or going negative.
    capacity = c[0] - concentration

    # (2) Surface-coating suppression of the autocatalytic (secondary-
    # nucleation) channel. Inhibitors that cap/coat fibril surfaces block
    # secondary nucleation more strongly as fibril mass grows, so the
    # suppression scales with cd*c. This term acts ONLY on the superlinear
    # channel, stretching the LAG and delaying the early 10%/25% response
    # crossings of inhibitor curves -- exactly the worst-curve behaviour that
    # dominates the high-quantile shape loss -- while leaving primary
    # nucleation (which sets baseline onset) intact.
    secondary = (
        c[2] ** 2 * m_eff * concentration ** 2 * (1 + c[4] ** 2 * seed)
        / (1 + c[5] ** 2 * inhibitor * concentration)
    )
    growth = c[1] ** 2 * m_eff + secondary
    expression = capacity * growth

    # Well-separated, physically-motivated initializer: weak primary
    # nucleation (c1) versus stronger autocatalysis (c2) biases toward a long
    # flat lag then a sharp rise (correct amyloid sigmoid), and the larger
    # inhibitor coefficients (c3, c5) let held-out inhibitor curves be delayed
    # more strongly. Spreading the vector breaks the shared local optimum the
    # fixed multi-start has been repeatedly falling into.
    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.15, 1.2, 1.5, 0.7, 1.8],
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
