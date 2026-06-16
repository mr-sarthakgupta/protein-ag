# EVOLVE-BLOCK-START
"""Symbolic regression seed for multi-dataset amyloid aggregation kinetics."""

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
    Logistic sigmoid in log-time space with sinh-based concentration scaling.

    Works across all 60 datasets by:
    - log(x0 + c2): compresses the enormous x0 range variation, c2 shifts
      the time origin (handles near-zero and negative x0 values)
    - rate c1 * x1^c5: power-law concentration dependence of the growth rate
    - half_time c3 * sinh(c6 * log(x1)) + c7: flexible asymmetric power-law
      for the half-time vs concentration relationship; sinh(c6*log(x1)) =
      (x1^c6 - x1^(-c6))/2 which gracefully degrades when x1 is sequential
    - baseline c4 * tanh(c8 * x0): ramp from zero, handles datasets where
      y(0) > 0 without a hard offset

    Template: c0 / (1 + exp(-c1*x1^c5 * (log(x0+c2) - c3*sinh(c6*log(x1)) - c7))) + c4*tanh(c8*x0)
    Constants: 9 total (c0..c8)
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(time + c[2])
    log_param = sp.log(parameter)

    rate = c[1] * parameter ** c[5]
    half_time = c[3] * sp.sinh(c[6] * log_param) + c[7]
    plateau = c[0]
    baseline = c[4] * sp.tanh(c[8] * time)

    expression = plateau / (1 + sp.exp(-rate * (log_time - half_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        max_nfev=1000,
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
