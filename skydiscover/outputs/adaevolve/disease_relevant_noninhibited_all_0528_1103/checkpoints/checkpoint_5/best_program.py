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
    Generalized Richards sigmoid for protein aggregation kinetics.

    Builds on the best-known structure (combined_score ~0.6748) and improves
    it by separating the baseline rate constant from x1-modulation:

        rate      = c0 * exp(c1 * x1)           # always positive, stable
        half_time = c2 + c3 * x1                # linear, globally stable
        shape     = c6                           # Richards asymmetry param
        y = c4 / (1 + exp(-rate*(x0 - half_time)))^c6 + c5
            + c7 * x0 / (x0 + c8)              # Michaelis-Menten drift term

    The extra Michaelis-Menten term `c7 * x0 / (x0 + c8)` captures slow
    secondary processes (secondary nucleation, late-stage growth) that cause
    the curve to not fully plateau within the measurement window. This is
    common in amyloid kinetics and adds only 2 constants (c7, c8).

    c8 is initialized to a large positive value to keep x0/(x0+c8) small
    and prevent the term from dominating the sigmoidal shape at early times.
    This term is globally smooth and bounded in [0, c7] for x0 >= 0.

    9 constants total — within the 12-constant limit.
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
        initial_values=[0.1, 0.0, 10.0, -1.0, 1.0, 0.0, 1.0, 0.0, 100.0],
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
