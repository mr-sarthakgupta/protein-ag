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
    Gompertz kinetics with shifted power-law x1 coupling and flat baseline.

    The Gompertz double-exponential captures asymmetric nucleation-elongation
    kinetics (fast rise after inflection, slow approach to plateau). Using
    (x1 + c6) as the base of power laws prevents singularity/NaN when x1=0
    (sequential-index datasets) while preserving power-law concentration
    dependence for concentration datasets. A flat additive baseline c5
    handles nonzero initial fluorescence robustly across all dataset types.

        shifted   = x1 + c6
        rate      = c0 * shifted^c1
        half_time = c2 * shifted^c3
        y = c4 * exp(-exp(-rate * (x0 - half_time))) + c5

    7 constants total. The shift c6 is fitted (typically > 0) ensuring the
    power-law base is positive. Numerically stable: double-exp is bounded
    in [0,1], plateau c4 scales to data range, baseline c5 absorbs offset.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    shifted = parameter + c[6]
    rate = c[0] * shifted ** c[1]
    half_time = c[2] * shifted ** c[3]
    plateau = c[4]
    baseline = c[5]

    expression = plateau * sp.exp(-sp.exp(-rate * (time - half_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 10.0, -0.5, 1.0, 0.0, 1.0],
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
