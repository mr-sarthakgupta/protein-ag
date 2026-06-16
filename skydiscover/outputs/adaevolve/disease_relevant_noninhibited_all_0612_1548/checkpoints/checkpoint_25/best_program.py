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
    Compact tanh sigmoid for nucleation-dependent protein aggregation.

    Expression (complexity ~15, 5 constants):
        growth = c0 * (x0 * x1^c1 - c2)
        y = c3 * tanh(growth) + c4

    Physical basis (Knowles secondary nucleation framework):
    - x0*x1^c1 is the concentration-scaled time: rate = c0*x1^c1,
      half_time = c2/x1^c1, matching t_half ∝ [M]^(-n/2) scaling.
    - tanh is bounded [-1,1] and numerically stable during least-squares.
    - Amplitude c3 and baseline c4 map the sigmoid to any [min,max] range
      after min-max normalization of y.
    - Dropping the Richards exponent (vs complexity=22) reduces complexity
      to ~15, improving parsimony_factor from 0.9725 to ~0.9813, which
      compensates for any small nmse increase (breakeven at nmse ~0.037).
    - 5 constants: more reliable fitting on small/noisy datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    time = x[0]
    parameter = x[1]

    growth = c[0] * (time * parameter ** c[1] - c[2])

    expression = c[3] * sp.tanh(growth) + c[4]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.5, 0.5, 0.5, 0.5],
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
