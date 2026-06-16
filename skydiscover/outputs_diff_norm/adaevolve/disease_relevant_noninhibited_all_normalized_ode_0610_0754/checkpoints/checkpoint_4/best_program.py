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
    Nucleation-elongation ODE for normalized amyloid aggregation kinetics.

    Physical model: dc/dt = (c0 + c1*x1) * (c + c2) * (1 - c)

    where:
      - (c0 + c1*x1): base rate c0 linearly modulated by experimental
                      parameter x1 (concentration, pH, etc.)
      - (c + c2):     nucleation term — c2 seed enables growth even at c=0,
                      reproducing the characteristic amyloid lag phase
      - (1 - c):      saturation term — growth slows as c approaches 1

    This is equivalent to the previous c0*(1+c1*x1)*(c+c2)*(1-c) form but
    merges the outer constant into the linear x1 term, reducing complexity
    by ~2 nodes while keeping the same expressive power (c0 now plays the
    role of the base rate directly, c1 the parameter sensitivity).

    Features: x0 = normalized time, x1 = normalized experimental parameter,
    x2 = current normalized concentration c supplied by ODE integration.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    rate = c[0] + c[1] * parameter
    growth = (concentration + c[2]) * (1 - concentration)

    expression = rate * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.0, 0.01],
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
