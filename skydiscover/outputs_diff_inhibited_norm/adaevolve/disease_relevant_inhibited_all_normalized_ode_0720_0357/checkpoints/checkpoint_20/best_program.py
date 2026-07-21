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
    """Autonomous inhibited aggregation ODE with surface-gated secondary nucleation.

    Approach: keep the proven capacity*(source + secondary) growth kernel that
    already delivers very low NMSE (~0.0034), but *remove the fragile explicit-
    time brake* (-c7*time*c ... ) and instead reshape the delayed inhibitor
    curves — where the curve-level shape loss (0.171, ~25% of score) is
    concentrated — through an intrinsic, mechanistically-correct inhibitor gate
    on the secondary-nucleation channel only.

    Rationale (Cohen/Meisl/Knowles Abeta42 picture): the sharp sigmoidal rise is
    driven by *secondary nucleation* — new nuclei formed catalytically on
    existing fibril surface. The dominant small-molecule inhibition mode coats
    that fibril surface, so it should suppress the secondary rate more strongly
    than primary nucleation/elongation. This right-shifts the half-time and
    stretches the 10->90% crossing timings of inhibited curves, exactly the
    shape-loss target, while leaving cd=0 curves untouched.

    Two-point inhibition, both static (fully autonomous, no explicit time, so
    odeint stays well behaved and the structure generalizes to held-out curves):
      * free_monomer = m0 / (1 + c1^2*cd + c2^2*cd*m0): competitive monomer
        sequestration reduces the reactive pool feeding all processes.
      * secondary is additionally divided by
        (1 + c7^2*cd + c8^2*cd*concentration + c9^2*cd^2): direct surface-catalysis
        suppression with a superlinear-in-dose (cd^2) term. Data show the top-dose
        curves are delayed and stretched far more than a linear-in-cd law allows,
        and those worst curves dominate the high-quantile shape term; the cd^2
        term right-shifts and stretches only the high-dose knees while leaving
        low-dose curves nearly unchanged. The cd*concentration term keeps the
        block strengthening as fibril mass (surface) grows.

    The (1 + concentration + c0^2*seed) factor gives the secondary channel a mild
    superlinear-in-c (autocatalytic) component so the sigmoid knee is crisp
    (helps the slope-profile term), while primary source keeps early drive finite.
    Capacity (plateau - c) -> 0 enforces mass-conservation saturation.

    Stability: all rate coefficients squared for sign control; both denominators
    are >= 1 for non-negative cd, concentration, so the RHS is globally smooth
    with no singularities, non-real values, or overflow. At cd=0 both inhibitor
    denominators collapse to 1, recovering clean uninhibited behavior. 9 fitted
    constants (< 13), better conditioned than the 10-constant parent.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration
    free_monomer = monomer / (1 + c[1] ** 2 * inhibitor + c[2] ** 2 * inhibitor * monomer)
    source = c[3] ** 2 + c[4] ** 2 * free_monomer + c[5] ** 2 * seed
    # Superlinear-in-dose surface-catalysis block on the secondary (autocatalytic)
    # channel. Data show the high-dose curves (cd=2e-6) are delayed far more than
    # a linear-in-cd law predicts: their half-time and 10->90% spread grow
    # disproportionately, and this worst-curve behaviour dominates the
    # high-quantile shape term (~25% of the score). Adding a saturable cd^2 term
    # (c9^2 * inhibitor^2) makes strong doses suppress the fibril-surface-catalysed
    # nucleation much more than weak doses, right-shifting and stretching only the
    # high-dose knees while leaving low-dose curves essentially unchanged. The
    # cd*concentration term keeps the block growing with fibril mass (surface).
    # Denominator is >= 1 for cd, concentration >= 0 (globally smooth, positive,
    # no singularity) and collapses to 1 at cd=0, so uninhibited behaviour and
    # NMSE are preserved. 10 fitted constants (< 13).
    secondary_gate = (
        1 + c[7] ** 2 * inhibitor + c[8] ** 2 * inhibitor * concentration + c[9] ** 2 * inhibitor ** 2
    )
    secondary = c[6] ** 2 * free_monomer * concentration * (1 + concentration + c[0] ** 2 * seed) / secondary_gate
    expression = capacity * (source + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.3],
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
