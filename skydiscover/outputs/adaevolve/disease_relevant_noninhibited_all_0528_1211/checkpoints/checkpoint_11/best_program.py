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
    Gompertz kinetics with power-law x1 coupling and hyperbolic baseline.

    The Gompertz function is asymmetric: it rises more steeply after the
    inflection point than it approaches the plateau, matching nucleation-
    elongation kinetics better than the symmetric logistic.

    Power-law x1 coupling for both rate and half-time captures concentration-
    dependent kinetics (nucleation order). The hyperbolic baseline c5/(x1+c6)
    captures the empirically observed ~1/x1 decay of the initial fluorescence
    with concentration (e.g. 0.231 at 1uM → 0.006 at 50uM in alphasyn data).

        rate      = c0 * x1^c1
        half_time = c2 * x1^c3
        y = c4 * exp(-exp(-rate * (x0 - half_time))) + c5/(x1 + c6)

    7 constants: sufficient expressiveness while remaining reliably fittable.
    The double-exponential is bounded: inner exp → large positive when
    x0 << half_time (y ≈ baseline), inner exp → 0 when x0 >> half_time
    (y ≈ c4 + baseline). Numerically stable for c0 > 0.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    rate = c[0] * parameter ** c[1]
    half_time = c[2] * parameter ** c[3]
    plateau = c[4]
    baseline = c[5] / (parameter + c[6])

    expression = plateau * sp.exp(-sp.exp(-rate * (time - half_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 10.0, -0.5, 1.0, 0.2, 1.0],
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
