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
    Factored tanh-Richards model for nucleation-dependent protein aggregation.

    Expression (complexity=22, 6 constants):
        growth = c0 * (x0 * x1^c1 - c2)
        y = c4 * ((1 + tanh(growth)) / 2)^c3 + c5

    Physical basis (Knowles/Michaels framework):
    - This is equivalent to rate*(x0 - half_time) with rate = c0*x1^c1
      and half_time = c2/x1^c1. Half-time decreases as concentration^c1,
      consistent with secondary nucleation kinetics where t_half ∝ [M]^(-n2/2).
    - The Richards exponent c3 captures asymmetric sigmoid shapes from
      nucleation-dominated kinetics (extended lag phase, rapid growth).
    - tanh is bounded [-1,1], numerically stable during least-squares fitting.
    - 6 constants (vs 7 in full form), complexity=22 (vs 24): parsimony
      factor 0.9725 vs 0.9700, requiring nmse < 0.0289 to beat best known
      0.9452 (vs nmse < 0.0262 at complexity=24).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    growth = c[0] * (time * parameter ** c[1] - c[2])

    expression = c[4] * ((1 + sp.tanh(growth)) / 2) ** c[3] + c[5]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.5, 0.5, 2.0, 1.0, 0.0],
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
