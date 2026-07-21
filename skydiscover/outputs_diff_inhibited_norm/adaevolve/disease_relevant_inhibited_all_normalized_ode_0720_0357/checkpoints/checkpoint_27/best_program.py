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
    """Autonomous inhibited aggregation ODE with dose-scaled surface saturation.

    Approach: exploit the best-scoring family (autonomous
    capacity*(source+secondary) kernel, combined_score 0.9572, NMSE ~0.0034),
    which beat the explicit-time brake parent (0.9566). The remaining error is
    almost entirely the curve-level shape loss (~0.171, ~25% of score),
    concentrated on the delayed high-dose inhibitor curves whose sigmoid knee
    must be right-shifted in the 25/50/75/90% response-crossing timings.

    Mechanistic basis (Cohen/Meisl/Knowles amyloid master equation for Abeta42,
    secondary-nucleation dominant): the autocatalytic secondary channel
    ~ k2 * m * M sets the sigmoid steepness and half-time. Two physically
    distinct inhibitor actions are represented, both multiplicative and both
    exactly 1 at cd=0 (so the proven uninhibited kernel is recovered):

      * monomer sequestration -- competitive Langmuir binding removes reactive
        monomer from every channel:
            free_monomer = m0 / (1 + c1^2*cd + c2^2*cd*m0)
      * surface poisoning of secondary nucleation -- the inhibitor coats the
        fibril surface, lowering the number of catalytic sites so the
        autocatalytic rate rolls over at *smaller* fibril mass as dose rises:
            surf_sat = 1 + (c7^2 + c8^2*cd) * concentration^2
        Higher cd => larger denominator => secondary saturates earlier =>
        knee/half-time right-shifted, directly targeting the worst delayed
        curves that dominate the high-quantile shape term.

    source = c3^2 + c4^2*free_monomer + c5^2*seed gives a finite lag-phase drive
    (primary nucleation + seed elongation) so early crossing timings are set,
    not clamped. capacity = plateau - c is the mass-conservation sink: as c ->
    plateau growth halts smoothly, and a small overshoot flips the sign to
    restore toward plateau (no runaway).

    Stability: all rate coefficients are squared for sign control; both
    inhibitor denominators are >= 1 for the non-negative feature ranges and the
    surf_sat denominator >= 1 for all c, so the RHS is globally smooth, finite,
    and singularity-free -- well conditioned for single-start least squares.
    Nine fitted constants (< 13), autonomous (no explicit-time term).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration
    free_monomer = monomer / (1 + c[1] ** 2 * inhibitor + c[2] ** 2 * inhibitor * monomer)
    source = c[3] ** 2 + c[4] ** 2 * free_monomer + c[5] ** 2 * seed
    surf_sat = 1 + (c[7] ** 2 + c[8] ** 2 * inhibitor) * concentration ** 2
    secondary = c[6] ** 2 * free_monomer * concentration * (1 + concentration + c[0] ** 2 * seed) / surf_sat
    expression = capacity * (source + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.5],
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
