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
    Robust sigmoidal model for protein aggregation kinetics with 8 constants.

    Uses a generalized logistic (Richards) curve where:
      - rate depends on x1 via exp(c1*x1) — always positive, globally defined,
        avoids singularities when x1=0 or x1<0 (sequential-index datasets)
      - half_time depends on x1 via c2 + c3*x1 — linear, stable, captures
        concentration-dependent lag phase shift
      - a shape exponent c6 (Richards parameter) allows asymmetric sigmoids,
        improving fit for nucleation-dominated kinetics where rise is steeper
        than the approach to plateau
      - plateau c4 and baseline c5 adapt to rescaled [0,1] range

    The generalized logistic form:
        y = c4 / (1 + exp(-c0*exp(c1*x1)*(x0 - c2 - c3*x1)))^c6 + c5

    Using exp(c1*x1) for rate modulation:
      - Always positive regardless of x1 sign or magnitude
      - Equivalent to power-law when x1>0 but numerically safer
      - Handles both numeric concentrations and sequential-index x1

    The shape parameter c6 generalizes the symmetric logistic to allow
    faster rise / slower approach to plateau (typical of nucleation kinetics).

    8 constants total — within the 12-constant limit, moderate complexity.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    parameter = x[1]

    rate = c[0] * sp.exp(c[1] * parameter)
    half_time = c[2] + c[3] * parameter
    plateau = c[4]
    baseline = c[5]
    shape = c[6]
    rate_scale = c[7]

    core = sp.exp(-rate * (time - half_time))
    expression = plateau / (1 + core) ** shape + baseline + rate_scale * parameter

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.0, 10.0, -1.0, 1.0, 0.0, 1.0, 0.0],
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
