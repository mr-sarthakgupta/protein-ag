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
    """Inhibitor-modulated fibril-surface-saturation aggregation ODE.

    Approach: build on the best-scoring family (surface-saturated secondary
    nucleation, combined_score 0.9572) rather than the explicit-time or
    superlinear-cd^2 variants, and make its single new denominator carry the
    inhibitor dependence in a mechanistically-grounded, non-fragile way.

    Kernel (Cohen/Meisl/Knowles amyloid master equation, saturating secondary
    nucleation): dc/dt = capacity * (source + secondary), where the sharp
    sigmoid is driven by fibril-surface-catalysed secondary nucleation whose
    rate rises with fibril mass but *rolls over* once the finite fibril surface
    becomes saturated with monomer/nuclei.

      * capacity = plateau - c  : mass-conservation sink; smooth self-limiting
        halt at the plateau, and if c slightly overshoots the term flips sign
        and restores toward plateau (no runaway).
      * free_monomer = m0 / (1 + c1^2*cd + c2^2*cd*m0) : competitive Langmuir
        monomer sequestration by inhibitor -- reduces the reactive pool feeding
        every channel. Denominator >= 1, exactly 1 at cd=0.
      * source = c3^2 + c4^2*free_monomer + c5^2*seed : finite early drive
        (primary nucleation + seed-proportional elongation) so lag-phase slope
        is nonzero and crossing timings are set, not clamped, at t=0.
      * secondary = c6^2 * free_monomer * c * (1 + c + c0^2*seed) / surf_sat :
        autocatalytic (superlinear-in-c) secondary nucleation with a saturable
        denominator surf_sat modelling the fibril surface becoming a limiting
        catalytic resource as mass accumulates.

    Inhibitor-dependent surface saturation (the one targeted change):
        surf_sat = 1 + c7^2 * (1 + c8^2*cd) * c^2
    Physically, an inhibitor that coats/blocks fibril surface lowers the
    effective number of catalytic sites, so the secondary rate saturates
    *earlier* (at smaller fibril mass) as dose rises. This stretches and
    right-shifts precisely the high-dose delayed knees -- the worst curves that
    dominate the high-quantile shape term (~25% of the score) -- without the
    fragile explicit-time brake or the ill-conditioned polynomial cd^2 source
    term tried before. The factor is smooth and monotone in cd, and multiplies
    only the c^2 surface term, so it does not disturb early-time drive.

    Stability: all rate coefficients squared; surf_sat and both inhibitor
    denominators are >= 1 for cd, c >= 0, so the RHS is globally smooth, finite,
    singularity-free and well conditioned for single-start least squares. At
    cd=0 every inhibitor factor collapses to 1, exactly recovering the proven
    uninhibited surface-saturated kernel (preserving the low NMSE ~0.0034).
    Nine fitted constants (< 13), same count as the best parent.
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
    # Inhibitor-scaled fibril-surface saturation: higher dose coats the surface,
    # lowering effective catalytic capacity so secondary nucleation rolls over at
    # smaller fibril mass, stretching/right-shifting the high-dose knees where the
    # high-quantile shape loss concentrates. Denominator >= 1, smooth, monotone in
    # cd, collapses to the proven cd=0 surface-saturated kernel at cd=0.
    surf_sat = 1 + c[7] ** 2 * (1 + c[8] ** 2 * inhibitor) * concentration ** 2
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
