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
    """Inhibited Abeta42 aggregation ODE: autocatalytic secondary-nucleation
    core with monomer sequestration inhibition, plus a saturating quadratic
    time-gate brake that shifts response-crossing timings of delayed inhibitor
    curves (targets shape loss) while keeping NMSE and odeint stability intact.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Winning autocatalytic core (unchanged): saturating capacity term drives
    # dc/dt -> 0 at plateau (mass conservation); inhibitor sequesters free
    # monomer feeding both primary source and monomer-fed secondary nucleation.
    plateau = c[0]
    capacity = plateau - concentration
    free_monomer = monomer / (1 + c[1] ** 2 * inhibitor + c[2] ** 2 * inhibitor * monomer)
    source = c[3] ** 2 + c[4] ** 2 * free_monomer + c[5] ** 2 * seed
    secondary = c[6] ** 2 * free_monomer * concentration * (1 + concentration + c[0] ** 2 * seed)

    # Breakthrough: replace the single linear time-drag with a smoother
    # quadratic-in-time saturating brake. Physically this is an off-pathway /
    # retrograde depletion whose strength ramps up over time (c7^2*t + c8^2*t^2),
    # matching the delayed, prolonged approach of inhibited curves. The
    # denominator (1 + c9^2*c) >= 1 saturates the brake in the aggregate state,
    # so it cannot destabilize early growth (small c) nor overpower late growth
    # (capacity -> 0 near plateau damps it). This shifts the 25/50/75%
    # response-crossing timings later without hurting NMSE, targeting shape_loss.
    brake = (c[7] ** 2 * time + c[8] ** 2 * time * time) * concentration / (1 + c[9] ** 2 * concentration)
    expression = capacity * (source + secondary) - brake

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
