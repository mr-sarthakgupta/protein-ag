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
    Simplified Richards (asymmetric logistic) with shared monomer exponent
    and seed-dependent lag phase offset.

    Physical basis (Knowles secondary nucleation model):
      - The elongation rate scales as x1^c1 (monomer power law)
      - The half-time scales inversely with rate: t_half ~ 1/(rate)
      - Seeds shorten the lag via a denominator factor (1 + c3*x2)
      - This gives: rate*x0 - rate*half_time = c0*x1^c1*x0 - c2/(1+c3*x2)

    The key simplification vs the previous 7-constant form: instead of
    separate exponents for rate (c1) and half_time (c4), the shared exponent
    c1 enforces the physically correct reciprocal relationship between rate
    and half-time. This reduces complexity from 30 to 26 nodes while
    preserving all essential structure.

    Template (6 constants, complexity=26):
        y = c4 / (1 + exp(-c0*x1^c1*x0 + c2/(1+c3*x2)))^c5

    where:
      x0 = normalized time, x1 = monomer conc, x2 = seed conc
      c0*x1^c1        : monomer-dependent growth rate
      c2/(1+c3*x2)    : seed-shortened lag offset (always finite for x2>=0)
      c4              : plateau amplitude
      c5              : Richards shape (asymmetry); c5=1 is standard logistic
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[0] * monomer ** c[1]
    lag = c[2] / (1 + c[3] * seed)

    expression = c[4] / (1 + sp.exp(-rate * time + lag)) ** c[5]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.5, 2.5, 1.0, 1.0, 1.0],
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
