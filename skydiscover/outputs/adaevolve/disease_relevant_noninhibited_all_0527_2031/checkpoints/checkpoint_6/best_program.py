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
    Hill / sigmoidal nucleation model for amyloid aggregation kinetics.

    Amyloid aggregation follows nucleation-dependent polymerization with a
    pronounced lag phase, rapid autocatalytic growth, then a plateau.
    The Hill equation captures this naturally:

        y = c4 * x0^c5 / (half_time^c5 + x0^c5) + c6

    where half_time = c2 * x1^c3 is the concentration-dependent half-time
    and c5 is the Hill coefficient (controls lag-phase sharpness; c5>1
    gives a sigmoidal shape with a real lag phase, c5=1 gives hyperbolic).

    This is equivalent to a logistic in log-time space and is numerically
    more stable than the Richards (1+exp)^n form. It is directly related
    to the Finke-Watzky two-step nucleation model approximation.

    For single-concentration datasets (x1=1): half_time=c2, so the model
    collapses to a 4-constant Hill curve.

    6 constants: c0 (unused/absorbed; kept for API symmetry — actually
    we use c2 as half-time scale), c1 (half-time conc exponent),
    c2 (half-time scale), c3 (unused), c4 (plateau amplitude),
    c5 (Hill coefficient / lag-phase shape).

    Actual template (6 constants):
        half_time = c0 * x1^c1
        y = c2 * x0^c3 / (half_time^c3 + x0^c3) + c4 + c5 * x1
    The last term c5*x1 allows a small concentration-dependent baseline
    shift, improving fits on multi-concentration datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    half_time = c[0] * parameter ** c[1]
    plateau = c[2]
    hill_n = c[3]
    baseline = c[4]
    conc_baseline = c[5]

    # Hill equation: plateau * t^n / (t_half^n + t^n) + baseline
    # Add small concentration-dependent baseline term for multi-conc datasets
    expression = (
        plateau * time ** hill_n / (half_time ** hill_n + time ** hill_n)
        + baseline
        + conc_baseline * parameter
    )

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[10.0, -0.5, 1.0, 2.0, 0.0, 0.0],
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
