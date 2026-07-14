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
    Generalized Richards (asymmetric logistic) with monomer-dependent rate
    and seed-dependent half-time.

    Amyloid aggregation curves are often right-skewed: the rise from baseline
    is slow and the approach to plateau is fast. The Richards curve captures
    this via a shape exponent 1/c6 on the logistic denominator:

        rate      = c0 * x1^c1
        half_time = c2 * x1^c4 / (1 + c5*x2)
        y = c3 / (1 + exp(-rate * (x0 - half_time)))^(1/c6)

    At c6=1 this reduces exactly to the standard logistic (current best).
    c6 > 1 gives right-skewed curves (slow rise, fast plateau approach).
    c6 < 1 gives left-skewed curves.

    Template: 7 constants, complexity ~32.
    Numerically stable: initialized at c6=1 (standard logistic fallback).
    The base (1 + exp(...)) >= 1 always, so the power is real for any c6.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[0] * monomer ** c[1]
    half_time = c[2] * monomer ** c[4] / (1 + c[5] * seed)
    plateau = c[3]
    shape = c[6]

    expression = plateau / (1 + sp.exp(-rate * (time - half_time))) ** shape

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.5, 0.5, 1.0, -0.5, 1.0, 1.0],
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
