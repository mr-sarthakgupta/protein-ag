# EVOLVE-BLOCK-START
"""Symbolic regression seed for multi-dataset amyloid aggregation kinetics."""

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
    Numerically stable sigmoidal model for nucleation-dependent protein aggregation.

    Uses tanh(asinh(x0)) formulation which is bounded and never overflows,
    unlike exp-based logistics. The asinh(x0) = log(x0+sqrt(x0^2+1)) compresses
    the time axis, with concentration-dependent rate (x1^c6 power law) and
    log-linear half-time (c3*log(x1)+c5). A secondary tanh(x0) ramp captures
    seeded/early-time behavior present in many datasets.

    Template: c0*tanh(c1*x1^c6*(asinh(x0) - c3*log(x1) - c5)) + c4 + c2*tanh(x0)
    7 constants, complexity ~33.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(time + sp.sqrt(time**2 + 1))

    rate = c[1] * parameter ** c[6]
    half_log_time = c[3] * sp.log(parameter) + c[5]
    sigmoid_arg = rate * (log_time - half_log_time)

    expression = c[0] * sp.tanh(sigmoid_arg) + c[4] + c[2] * sp.tanh(time)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 1.0, 0.0, 0.5, 0.5, 10.0, 0.0],
        max_nfev=1000,
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
