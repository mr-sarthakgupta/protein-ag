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
    Generalized Richards sigmoid + Michaelis-Menten drift for protein aggregation.

    This is the best-known structure (combined_score ~0.7899). The core sigmoid
    captures primary nucleation-elongation kinetics; the MM drift term captures
    secondary nucleation and late-stage growth that prevents full plateau within
    the measurement window — a well-documented feature of amyloid kinetics.

        rate      = c0 * exp(c1 * x1)          # always positive, stable for x1<=0
        half_time = c2 + c3 * x1               # linear, globally stable
        core      = exp(-rate * (x0 - half_time))
        sigmoid   = c4 / (1 + core)^c6 + c5    # Richards asymmetric sigmoid
        drift     = c7 * x0 / (x0 + c8)        # MM: bounded in [0, c7], smooth

        y = sigmoid + drift

    9 constants total. The MM drift is bounded (never diverges), smooth, and
    vanishes near x0=0, so it cannot distort the early lag phase. c8 is
    initialized to 50.0 (moderate timescale) to avoid dominating the sigmoid.
    The shape parameter c6 allows asymmetric rise/plateau (nucleation kinetics).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    parameter = x[1]

    rate = c[0] * sp.exp(c[1] * parameter)
    half_time = c[2] + c[3] * parameter
    plateau = c[4]
    baseline = c[5]
    shape = c[6]
    drift_amp = c[7]
    drift_scale = c[8]

    core = sp.exp(-rate * (time - half_time))
    sigmoid = plateau / (1 + core) ** shape + baseline
    drift = drift_amp * time / (time + drift_scale)
    expression = sigmoid + drift

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.0, 10.0, -1.0, 1.0, 0.0, 1.0, 0.0, 50.0],
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
