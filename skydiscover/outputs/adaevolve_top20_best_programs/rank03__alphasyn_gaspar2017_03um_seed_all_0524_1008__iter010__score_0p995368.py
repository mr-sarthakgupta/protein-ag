# EVOLVE-BLOCK-START
"""Symbolic regression seed for Alpha-synuclein Gaspar 2017 0.3uM seed data."""

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
    Seeded amyloid aggregation: power-law logistic with concentration-dependent
    rate and half-time, motivated by the Knowles secondary-nucleation framework.

    y = c4 / (1 + exp(-c0 * C^c1 * (t - c2 * C^(-c3)))) + c5

    where:
      c0 * C^c1      = concentration-dependent growth rate (power law)
      c2 * C^(-c3)   = concentration-dependent half-time (power law decay)
      c4             = plateau amplitude (~1)
      c5             = baseline offset (~0)

    6 constants, complexity 25, strong generalization across concentrations.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    concentration = x[1]

    # Power-law concentration-dependent growth rate
    k = c[0] * concentration ** c[1]

    # Power-law concentration-dependent half-time (decreases with C)
    t_half = c[2] * concentration ** (-c[3])

    # Plateau amplitude and baseline offset
    plateau = c[4]
    baseline = c[5]

    # Generalized logistic (sigmoidal) expression with baseline
    expression = plateau / (1 + sp.exp(-k * (time - t_half))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 1.0, 20.0, 0.5, 1.0, 0.0],
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
    """Load deterministic Alpha-synuclein Gaspar 2017 splits (matches evaluator)."""
    from evaluator import load_alphasyn_data

    return load_alphasyn_data()


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = _load_data()
    result = run_discovery(X_train, y_train, X_val, y_val)
    print("equation:", result.get("equation"))
    print("nmse_val:", result.get("nmse_val"))
    print("combined_score:", result.get("combined_score"))
