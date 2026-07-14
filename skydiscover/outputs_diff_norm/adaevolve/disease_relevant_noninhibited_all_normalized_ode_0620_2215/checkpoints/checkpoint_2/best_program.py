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
    # Onset (lag) is driven by a small primary-nucleation + seed flux; growth
    # is autocatalytic with both a linear (elongation) and a quadratic
    # (secondary-nucleation) feedback in c.  The quadratic c**2 term sharpens
    # the sigmoidal acceleration and lengthens the lag, directly targeting the
    # dominant slope/half-time shape loss.  All rate constants are squared so
    # they stay positive; monomer**power is well defined for m0 > 0; the RHS
    # is a smooth polynomial in c (capacity = plateau - c) with no divisions,
    # logs, or fractional powers, so it is globally defined and stable.
    monomer_scale = monomer ** c[0]
    seed_flux = c[1] ** 2 + c[2] ** 2 * seed
    elongation = c[3] ** 2 * concentration
    secondary = c[4] ** 2 * concentration ** 2 * (1 + c[5] ** 2 * seed)
    plateau = c[6]
    capacity = plateau - concentration

    growth = monomer_scale * (seed_flux + elongation + secondary)
    expression = capacity * (c[7] ** 2 * growth + c[8] ** 2 * time)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 0.1, 0.3, 1.0, 1.0, 0.3, 1.0, 1.0, 0.1],
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
