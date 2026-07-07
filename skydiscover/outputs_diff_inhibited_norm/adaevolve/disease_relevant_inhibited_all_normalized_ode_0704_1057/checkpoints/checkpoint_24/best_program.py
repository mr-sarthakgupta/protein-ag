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
    """Seed-additive secondary-nucleation ODE with dual Langmuir inhibition.

    Data-driven structure. Direct inspection of the crossing-time profiles in
    this dataset shows three regimes the previous template could not separate:
      * seeded (M0>0), no inhibitor: fast sigmoid, 50% near t~0.2-0.7 (no lag);
      * unseeded (M0=0), no inhibitor: a genuine lag, 50% pushed to t~0.57;
      * unseeded + strong inhibitor (cd=2e-6): an extreme lag, 50% at t~3.0,
        90% at t~5.4 -- the worst-curve term that freezes the shape loss.
    Reproducing all three simultaneously requires (a) the seed to REMOVE the
    lag and (b) the inhibitor to suppress the lag-breaking primary channel far
    more strongly than the seed-carried autocatalytic channel, so an unseeded
    inhibited curve has almost no way to start and therefore lags for a long
    time, while a seeded inhibited curve is only moderately delayed.

    Mechanism (Cohen/Knowles secondary-nucleation reduction):
      * Seed enters ADDITIVELY into the autocatalytic state as
        (concentration + c2^2*seed). With M0=0 growth is purely self-catalytic
        in c, giving the true lag; with M0>0 the effective state is boosted
        from t=0, collapsing the lag -- exactly the seeded/unseeded contrast.
      * Two INDEPENDENT saturable Langmuir free-fractions gate the two
        channels: inhib_p = 1/(1 + c4^2*cd) on the primary (lag-breaking)
        source, inhib_s = 1/(1 + c5^2*cd) on the autocatalytic feedback.
        Letting the fit make primary suppression the stronger one produces the
        very long unseeded-inhibited lag while keeping seeded-inhibited curves
        only moderately delayed -- directly attacking the frozen shape loss.
        Both factors are bounded in (0,1], strictly monotone in cd, singular-
        free (denominator = 1 + nonnegative), and collapse to 1 at cd=0.

    capacity = c3 - concentration is the mass-conserving self-limiting sink
    (c3 may exceed 1 so capacity stays positive on normalized data). All rate
    coefficients are squared for strict positivity, so the RHS stays smooth and
    odeint-stable. Six constants -- fewer than before -- for better-conditioned
    least-squares fitting.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    inhib_p = 1 / (1 + c[4] ** 2 * inhibitor)
    inhib_s = 1 / (1 + c[5] ** 2 * inhibitor)
    primary = c[0] ** 2 * monomer * inhib_p
    secondary = c[1] ** 2 * monomer * (concentration + c[2] ** 2 * seed) * inhib_s
    capacity = c[3] - concentration
    expression = capacity * (primary + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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