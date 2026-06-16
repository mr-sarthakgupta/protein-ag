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

    Physical model: dc/dt = rate * (c + seed) * (1 - c)

    where:
      - rate  = c0 * (1 + c1*x1): overall rate modulated by experimental
                parameter x1 (concentration, pH, etc.)
      - seed  = c2: small nucleation constant that enables a lag phase
                when c is near zero; controls the length of the lag phase
      - (1-c): saturation term — growth slows as c approaches 1

    Compared to the logistic model c*(plateau-c), this form:
      1. Fixes the plateau at 1 (correct after min-max normalization)
      2. Adds a nucleation seed c2 so dc/dt > 0 even at c=0, reproducing
         the characteristic amyloid lag phase
      3. Has the same complexity but is more physically motivated

    Features: x0 = normalized time, x1 = normalized experimental parameter,
    x2 = current normalized concentration c supplied by ODE integration.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    rate = c[0] * (1 + c[1] * parameter)
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
