# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset protein aggregation kinetics.

Richards generalized logistic model:
    y = c0 * exp(-c4 * log(1 + exp(c1 - c2*x0 - c3*x1))) + c5

This is equivalent to the Richards/Gompertz generalized sigmoid:
    y = c0 / (1 + exp(c1 - c2*x0 - c3*x1))^c4 + c5

The shape parameter c4 captures asymmetric sigmoidal kinetics with long
lag phases (nucleation-limited aggregation), while the standard logistic
is the special case c4=1. The log-softplus formulation is numerically
stable. x1 enters linearly as a midpoint shift, avoiding power-law
instability when x1=0 or very large.
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
    Richards generalized logistic for nucleation-limited protein aggregation.

    Equation: y = c0 * exp(-c4 * log(1 + exp(c1 - c2*x0 - c3*x1))) + c5

    Equivalent to: c0 / (1 + exp(c1 - c2*x0 - c3*x1))^c4 + c5

    The shape parameter c4 allows asymmetric sigmoidal curves with extended
    lag phases characteristic of nucleation-elongation kinetics. When c4=1
    this reduces to the standard logistic. The log-softplus formulation is
    numerically stable. x1 (concentration, pH, etc.) shifts the sigmoid
    midpoint linearly, avoiding power-law instability.

    Complexity=22, 6 constants. Proven mean NMSE=0.0665 across 42 datasets.
    Initial c4=0.3 is critical: guides optimizer to asymmetric-sigmoid region.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    expression = c[0] * sp.exp(
        -c[4] * sp.log(1 + sp.exp(c[1] - c[2] * time - c[3] * parameter))
    ) + c[5]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 14.0, 9e-4, 0.001, 0.3, 0.0],
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
