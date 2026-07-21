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
    """Autonomous secondary-nucleation ODE with cooperative dose sequestration
    and dose-dependent fibril-surface poisoning.

    Family: the best-scoring autonomous kernel dc/dt = capacity*(source +
    secondary) (combined_score ~0.957), whose pointwise NMSE is already tiny
    (~0.003). The dominant remaining error is the CURVE-LEVEL shape loss,
    concentrated on the delayed/stretched HIGH-DOSE inhibitor curves. The
    empirical crossing timings confirm two things the pure-logistic and
    sqrt-autocatalysis variants cannot both reproduce:
      * higher cd right-shifts the whole sigmoid super-linearly in dose
        (unseeded t50 grows ~2.6x from the lowest to the highest dose), and
      * higher cd also STRETCHES the growth phase (t50->t90 gap widens), i.e.
        the autocatalytic knee is not just delayed but flattened.

    Mechanistic encoding (Cohen/Meisl/Knowles amyloid master equation; both
    inhibitor terms are exactly 1 at cd=0 so the proven uninhibited kinetics
    are recovered):
      capacity  = plateau - c                 mass-conservation saturation ->1
      free_mono = monomer/(1+(c1^2+c2^2*cd)*cd)
                  competitive monomer sequestration with mild dose
                  cooperativity: the c2^2*cd^2 term removes disproportionately
                  more reactive monomer at strong doses, super-linearly
                  right-shifting the early crossings of the worst high-dose
                  curves while barely touching low doses. Denominator >= 1.
      source    = c3^2 + c4^2*free_mono + c5^2*seed
                  finite lag-phase drive (primary nucleation + seed elongation)
                  so early timing is set, not clamped; seed shortens the lag.
      surf_sat  = 1 + (c7^2 + c8^2*cd)*c^2
                  fibril-surface poisoning: inhibitor coats growing fibril
                  surface, so the autocatalytic secondary rate rolls over at
                  SMALLER fibril mass as dose rises -> flattens the knee and
                  stretches the 50/75/90% crossings of high-dose curves.
                  Denominator >= 1 for all c.
      secondary = c6^2*free_mono*c*(1 + c + c0^2*seed)/surf_sat
                  surface-catalysed autocatalysis; the (1 + c + c0^2*seed)
                  factor keeps the knee crisp at low dose and lets seed sharpen
                  it, matching the slope-profile part of the shape loss.

    Stability: all rate coefficients squared for sign control; both
    denominators are >= 1 over the non-negative feature ranges, so the RHS is
    globally smooth, finite and singularity-free -- well conditioned for
    single-start least squares. 9 fitted constants (< 13), fully autonomous.
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
