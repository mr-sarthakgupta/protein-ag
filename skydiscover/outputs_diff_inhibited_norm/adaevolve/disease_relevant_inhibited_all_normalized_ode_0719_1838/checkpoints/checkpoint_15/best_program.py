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
    c = constant_symbols(7)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    capacity = c[0] - concentration
    primary = c[1] ** 2 + c[2] ** 2 * monomer
    secondary = (
        c[3] ** 2 * monomer * concentration * (1 + c[4] ** 2 * seed)
        / (1 + c[5] ** 2 * inhibitor + c[6] ** 2 * inhibitor * concentration)
    )
    expression = capacity * (primary + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
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
