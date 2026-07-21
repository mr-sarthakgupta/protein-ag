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
    c = constant_symbols(5)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Coupled monomer-sequestration inhibition (mass conservation). The
    # inhibitor binds/sequesters free monomer, so the material available to
    # build fibrils is the EFFECTIVE monomer m_eff = m0 / (1 + c3^2 * cd).
    # Because m_eff feeds BOTH primary nucleation and the superlinear
    # secondary (surface-catalyzed autocatalytic) channel, a single physical
    # mechanism slows the ONSET and lowers the reachable PLATEAU together --
    # exactly as expected when the inhibitor removes buildable monomer. When
    # cd = 0, m_eff = m0 and native uninhibited kinetics are recovered
    # exactly. The denominator (1 + c3^2 * cd) >= 1 always, so there is no
    # singularity, non-real value, or overflow during least-squares fitting.
    m_eff = monomer / (1 + c[3] ** 2 * inhibitor)

    # Capacity term (c0 - c) enforces mass-conservation saturation so the
    # trajectory relaxes smoothly to the plateau c0 without going negative.
    capacity = c[0] - concentration

    # Primary nucleation from effective monomer (inhibitor-modulated source),
    # plus superlinear (concentration**2) secondary nucleation: secondary
    # nucleation is quadratic in fibril mass, giving a genuinely flat lag
    # phase followed by a sharp autocatalytic rise. This sharpens the 10%/25%
    # response-crossing timing that dominates the curve-level shape loss. The
    # seed factor (1 + c4^2 * M0) accelerates pre-seeded curves. c2 stays
    # modest at init to keep the concentration**2 term from stiffening odeint.
    growth = (
        c[1] ** 2 * m_eff
        + c[2] ** 2 * m_eff * concentration ** 2 * (1 + c[4] ** 2 * seed)
    )
    expression = capacity * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.3, 0.8, 2.0, 0.5],
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
