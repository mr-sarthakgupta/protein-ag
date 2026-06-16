# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset protein aggregation kinetics."""

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
    """Log-space logistic for protein aggregation kinetics.

    y = c4 / (1 + exp(-rate * (x0 - t_half))) + c5
    rate   = exp(c0 + c1*log(x1))  [= exp(c0) * x1^c1, power-law in log-space]
    t_half = exp(c2 + c3*log(x1))  [= exp(c2) * x1^c3, power-law in log-space]

    Log-space parameterization gives well-conditioned optimization across
    datasets spanning many time scales and concentration ranges.
    max_nfev=500 improves convergence on hard datasets vs default 300.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    log_param = sp.log(parameter)
    rate = sp.exp(c[0] + c[1] * log_param)
    half_time = sp.exp(c[2] + c[3] * log_param)

    expression = c[4] / (1 + sp.exp(-rate * (time - half_time))) + c[5]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[-10.0, 0.0, 10.0, -0.5, 1.0, 0.0],
        max_nfev=500,
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
