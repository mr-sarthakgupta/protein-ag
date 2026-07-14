# EVOLVE-BLOCK-START
"""ODE discovery seed for normalized multi-dataset amyloid aggregation kinetics."""

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
    """
    Generic bounded aggregation-rate ODE.

    This seed starts from a broad mechanism class rather than a specific
    closed-form solution: a concentration-independent source term can create
    onset from near-zero signal, while an autocatalytic state-dependent term
    can sharpen growth once aggregates are present.  Monomer and seed features
    modulate both routes, and a capacity factor keeps the trajectory saturating
    toward a fitted plateau.

    Features: x0 = normalized time, x1 = initial monomer m0, x2 = initial
    seed/aggregate M0, and x3 = current normalized concentration/state.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    # Knowles-Cohen reduced master equation in converted mass fraction c.
    # Onset (lag) is set by a small primary-nucleation source plus a seed
    # flux. Growth is autocatalytic with TWO feedbacks: elongation linear in
    # c, and secondary nucleation quadratic in c (new fibrils nucleate on
    # existing fibril surfaces). The quadratic c**2 term is the dominant
    # amplifier that sharpens the sigmoidal mid-phase slope and lengthens the
    # lag, directly targeting the shape loss the parent missed (it only had a
    # linear-in-c term). The whole flux is gated by the depleting capacity
    # (plateau - c) so mass is conserved and the trajectory saturates. All
    # rate constants are squared to stay positive; monomer**power is well
    # defined for m0 > 0; the RHS is a smooth polynomial in c with no
    # divisions, logs, or fractional powers, so it is globally stable.
    monomer_scale = monomer ** c[0]
    source_rate = c[1] ** 2 + c[2] ** 2 * seed
    elongation = c[3] ** 2 * concentration
    secondary = c[4] ** 2 * concentration ** 2 * (1 + c[5] ** 2 * seed)
    baseline_flux = c[6] ** 2
    plateau = c[7]
    capacity = plateau - concentration

    growth = baseline_flux + source_rate + monomer_scale * (elongation + secondary)
    expression = capacity * growth + c[8] * time * capacity

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 0.1, 0.3, 1.0, 1.0, 0.3, 0.01, 1.0, 0.0],
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


def _load_data():
    """Load the first dataset for local testing."""
    from evaluator import load_all_datasets

    datasets = load_all_datasets()
    name, X_train, X_val, y_train, y_val = datasets[0]
    print(f"Testing on: {name}")
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = _load_data()
    result = run_discovery(X_train, y_train, X_val, y_val)
    print("equation:", result.get("equation"))
    print("nmse_val:", result.get("nmse_val"))
    print("combined_score:", result.get("combined_score"))
