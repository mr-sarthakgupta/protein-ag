# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Key insight: protein aggregation follows a sigmoidal (logistic) curve in time.
The model y = c0 / (1 + exp(c1 - c2*x0 - c3*x1)) + c4 captures:
  - c0: plateau amplitude (~1 for rescaled data)
  - c1: positive shift ensuring sigmoid starts near 0 at t=0 (init=14)
  - c2: time-rate constant (init=9e-4 covers the wide x0 range 241-1.7M sec)
  - c3: x1 (concentration/parameter) influence on rate (init=0.001)
  - c4: baseline offset (~0)

This additive-exponent form is more numerically stable than power-law forms
because c1, c2, c3 scale independently and cannot create overflow/underflow
through interaction. When x1 is constant (single-curve datasets), c3*x1
is absorbed into the effective c1, so the model degrades gracefully to a
4-parameter logistic. Complexity=19, well below the 160-node limit.
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
    Logistic sigmoid with additive x0 and x1 terms in the exponent.

    y = c0 / (1 + exp(c1 - c2*x0 - c3*x1)) + c4

    The exponent c1 - c2*x0 - c3*x1 is a linear combination that:
    - starts large and positive (c1=14 init) so sigmoid starts near 0
    - decreases as time x0 increases (c2*x0 term)
    - is shifted by the experimental parameter x1 (c3*x1 term)

    Initial values [1, 14, 9e-4, 0.001, 0] are tuned across all 42
    disease-relevant datasets to avoid local minima from scale mismatch
    between x0 (range: 0 to 1.7M seconds) and x1 (range: 0.3 to 3950).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    time = x[0]
    parameter = x[1]

    expression = c[0] / (1 + sp.exp(c[1] - c[2] * time - c[3] * parameter)) + c[4]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 14.0, 9e-4, 0.001, 0.0],
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
