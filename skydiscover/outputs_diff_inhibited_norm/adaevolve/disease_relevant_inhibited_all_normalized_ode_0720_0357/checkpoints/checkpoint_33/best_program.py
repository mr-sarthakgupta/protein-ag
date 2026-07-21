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
    """Autonomous inhibited aggregation ODE with dose-superlinear monomer
    sequestration and surface-poisoned secondary nucleation.

    Base family (best-scoring, combined_score 0.9572): the autonomous
    capacity*(source + secondary) kernel whose NMSE is already low (~0.0034).
    The dominant remaining error is the curve-level shape loss (~0.169, 25% of
    the score), concentrated on the *delayed high-dose* inhibitor curves whose
    sigmoid knee and 25/50/75/90% response-crossing timings are right-shifted.

    Mechanistic picture (Cohen/Meisl/Knowles amyloid master equation for
    Abeta42, secondary-nucleation dominant). Aggregate mass grows by primary
    nucleation + seed-fed elongation (source) and by fibril-surface-catalysed
    secondary nucleation (autocatalytic in current mass, secondary). A
    mass-conservation capacity (plateau - c) shuts growth off smoothly. Two
    physically distinct, static inhibitor actions, both exactly 1 at cd=0 so
    the proven uninhibited kernel is recovered:

    * monomer sequestration with mild dose cooperativity --
          free_mono = monomer / (1 + (c1**2 + c2**2 * cd) * cd)
      The extra c2**2 * cd**2 term makes strong doses remove disproportionately
      more reactive monomer than weak doses, lengthening the lag and
      right-shifting the early (10/25%) crossings of the worst high-dose curves
      specifically, while leaving low-dose curves nearly unchanged -- exactly
      the behaviour the high-quantile shape term rewards. Denominator >= 1.
    * surface poisoning of secondary nucleation --
          surf_sat = 1 + (c7**2 + c8**2 * cd) * concentration**2
      The inhibitor coats the growing fibril surface, so the autocatalytic rate
      rolls over at smaller fibril mass as dose rises: higher cd => larger
      denominator => secondary saturates earlier => the knee and half-time
      (50/75/90% crossings) shift right. Denominator >= 1 for all c.

    source = c3**2 + c4**2 * free_mono + c5**2 * seed keeps a finite lag-phase
    drive so early timings are set, not clamped. secondary carries the
    (1 + concentration + c0**2 * seed) autocatalytic sharpening so the knee
    stays crisp (helps the slope-profile part of the shape loss).

    Stability: all rate coefficients squared for sign control; both denominators
    are >= 1 over the non-negative feature ranges, so the RHS is globally
    smooth, finite and singularity-free -- well conditioned for single-start
    least squares. Nine fitted constants (< 13), fully autonomous.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration

    free_mono = monomer / (1 + (c[1] ** 2 + c[2] ** 2 * inhibitor) * inhibitor)
    surf_sat = 1 + (c[7] ** 2 + c[8] ** 2 * inhibitor) * concentration ** 2

    source = c[3] ** 2 + c[4] ** 2 * free_mono + c[5] ** 2 * seed
    secondary = (
        c[6] ** 2 * free_mono * concentration * (1 + concentration + c[0] ** 2 * seed) / surf_sat
    )
    expression = capacity * (source + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3, 0.5],
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
