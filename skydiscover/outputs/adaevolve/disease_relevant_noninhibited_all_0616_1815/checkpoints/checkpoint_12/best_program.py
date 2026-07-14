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
    Logistic template with monomer-dependent rate and seed-dependent half-time.

    The Knowles analytical model for nucleation-elongation aggregation predicts
    a sigmoidal time course whose half-time decreases with both higher monomer
    concentration (x1) and higher seed concentration (x2). This candidate
    incorporates x2 into the half-time term via a denominator factor
    (1 + c5*x2), which:
      - reduces the lag phase when seed is present (physically correct)
      - degrades gracefully to the seed-free form when x2=0 or c5~0
      - avoids singularities (denominator always >= 1 for x2>=0, c5>=0)

    Template (6 constants, complexity ~28):
        rate      = c0 * x1^c1
        half_time = c2 * x1^c4 / (1 + c5*x2)
        y = c3 / (1 + exp(-rate * (x0 - half_time)))

    where:
      x0 = normalized time, x1 = monomer conc, x2 = seed conc
      c0*x1^c1  : monomer-concentration-dependent growth rate
      c2*x1^c4/(1+c5*x2) : half-time shortened by both monomer and seed
      c3         : plateau amplitude
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[0] * monomer ** c[1]
    half_time = c[2] * monomer ** c[4] / (1 + c[5] * seed)
    plateau = c[3]

    expression = plateau / (1 + sp.exp(-rate * (time - half_time)))

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.5, 0.5, 1.0, -0.5, 1.0],
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
