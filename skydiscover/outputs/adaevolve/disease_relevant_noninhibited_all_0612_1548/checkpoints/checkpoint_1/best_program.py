# EVOLVE-BLOCK-START
"""Symbolic regression seed for normalized multi-dataset amyloid aggregation kinetics."""

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
    Generalized sigmoidal model for nucleation-dependent protein aggregation.

    Features: x0 = normalized time, x1 = concentration-like experimental parameter.

    Key structural choices:
    - Log-concentration scaling for both rate and half-time:
        rate     = c0 + c1*log(x1)
        half_time = c2 + c3*log(x1)
      This is physically motivated: aggregation half-times scale logarithmically
      with monomer concentration over wide concentration ranges, and log scaling
      is more numerically stable than power-law across datasets with very
      different x1 ranges.
    - Variable power c5 on the sigmoid denominator:
        y = c4 / (1 + exp(-rate*(x0 - half_time)))^c5 + c6
      This generalizes the standard logistic (c5=1) to the Richards/generalized
      logistic family. c5 > 1 produces a faster initial rise (seeded/secondary
      nucleation kinetics); c5 = 1 recovers the standard sigmoid.
      This is critical for fitting seeded aggregation datasets where the curve
      rises steeply before saturating.

    7 constants total, complexity ~31.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    rate = c[0] + c[1] * sp.log(parameter)
    half_time = c[2] + c[3] * sp.log(parameter)
    growth = sp.exp(-rate * (time - half_time))

    expression = c[4] / (1 + growth) ** c[5] + c[6]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[10.0, 0.0, 0.3, -0.05, 1.0, 2.0, 0.0],
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
