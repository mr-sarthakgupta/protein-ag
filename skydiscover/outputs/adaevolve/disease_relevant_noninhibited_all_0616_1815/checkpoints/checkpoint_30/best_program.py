# EVOLVE-BLOCK-START
"""Symbolic regression: Gompertz double-exponential with monomer-scaled rate and lag for amyloid aggregation."""

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
    Gompertz double-exponential with monomer-scaled rate and seed-shortened lag.

    Physical basis:
      - Gompertz y = c0*exp(-c1*exp(-rate*x0 + lag)) is the natural analytical
        approximation to secondary nucleation kinetics (Knowles model): it is
        intrinsically right-skewed (slow departure from zero, fast approach to
        plateau), matching amyloid aggregation curves better than symmetric
        logistic or Richards with fitted exponent.
      - Growth rate: c5 * x1^c6 (monomer power law, secondary nucleation)
      - Lag offset: c2 * x1^c3 / (1 + c4*x2):
          * monomer-dependent: higher monomer -> shorter lag
          * seed-dependent: more seeds -> shorter lag (denominator >= 1)
      - c0 = plateau amplitude, c1 = steepness/displacement constant

    Template (7 constants, complexity=27):
        rate = c5 * x1^c6
        lag  = c2 * x1^c3 / (1 + c4*x2)
        y    = c0 * exp(-c1 * exp(-rate*x0 + lag))

    Numerically stable:
      - Inner exp(-rate*x0 + lag): large negative for large x0 -> 0
      - Outer exp(-c1 * 0) = 1 -> y -> c0 (plateau) as x0 -> inf
      - At x0=0: y = c0*exp(-c1*exp(lag)) ~ 0 when lag >> 0 and c1 > 0
      - No division by zero; no fractional powers of data variables.
    Complexity=27 saves 2 nodes vs Richards (29), improving parsimony penalty.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[5] * monomer ** c[6]
    lag = c[2] * monomer ** c[3] / (1 + c[4] * seed)

    expression = c[0] * sp.exp(-c[1] * sp.exp(-rate * time + lag))

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 3.0, -0.5, 1.0, 5.0, 0.5],
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
