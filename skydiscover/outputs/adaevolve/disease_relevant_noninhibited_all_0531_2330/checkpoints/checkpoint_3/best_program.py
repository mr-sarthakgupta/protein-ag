# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset protein aggregation kinetics.

Generalised sigmoidal model with power-law concentration dependence and
a fitted time-power exponent that handles both symmetric and asymmetric
(slow-nucleation lag-phase) aggregation curves.
"""

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
    Generalised sigmoidal kinetics model for protein aggregation.

    Features: x0 = elapsed time, x1 = varying experimental parameter
    (concentration, pH, etc. — sequential index for single-curve datasets).

    Expression:
        y = c4 / (1 + exp(-c0 * ((x0 + c6)^c5 - c1 / x1^c2))) + c3

    Design rationale:
    - (x0 + c6)^c5: time power with offset.  c6 handles datasets where
      time does not start at zero (e.g. IAPP t_min ~ 241 s).  c5 > 1
      captures slow-nucleation lag phases (lysozyme, serum amyloid) while
      c5 = 1 recovers the standard logistic.
    - c1 / x1^c2: power-law half-time.  Higher concentration → shorter
      half-time.  c2 = 0 makes the half-time concentration-independent,
      which is correct for single-curve datasets (x1 = constant index).
    - c4 = plateau amplitude, c3 = baseline offset.
    - 7 constants total (well within the 12-constant limit).
    - Complexity ≈ 26 nodes → parsimony penalty ≈ 0.97.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    t_shifted = time + c[6]
    half_time = c[1] / parameter ** c[2]
    expression = c[4] / (1 + sp.exp(-c[0] * (t_shifted ** c[5] - half_time))) + c[3]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0],
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
