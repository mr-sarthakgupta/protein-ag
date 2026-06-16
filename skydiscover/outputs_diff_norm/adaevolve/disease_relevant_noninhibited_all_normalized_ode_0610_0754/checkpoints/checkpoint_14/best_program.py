# EVOLVE-BLOCK-START
"""ODE discovery seed for normalized multi-dataset amyloid aggregation kinetics."""

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
    Power-law nucleation-elongation ODE with floating plateau for normalized
    amyloid aggregation kinetics.

    dx2/dt = c0 * x1^c1 * (x2 + c2) * (c3 - x2)

    - c0:       overall rate constant fitted per dataset
    - x1^c1:    power-law in raw experimental parameter (µM); exponent c1
                captures secondary nucleation scaling (Knowles/Cohen model).
                Essential: linear x1 fails across datasets with different
                concentration scales (0.3–3950 µM).
    - (c + c2): nucleation seed enabling growth from c=0 (lag phase).
                Essential: without this seed dc/dt=0 at c=0 and ODE stays stuck.
    - (c3 - c): floating plateau — c3 is fitted per dataset rather than fixed
                at 1. This relaxes the assumption that all datasets plateau at
                exactly the normalized maximum, improving fit on datasets with
                partial aggregation curves or normalization artifacts.

    Same complexity=13 as previous best. The 4th constant c3 replaces the
    fixed plateau at 1, giving one extra degree of freedom without increasing
    parsimony penalty. Wins if NMSE drops even slightly below 0.0145.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(4)

    parameter = x[1]
    concentration = x[2]

    rate = c[0] * (parameter ** c[1])
    growth = (concentration + c[2]) * (c[3] - concentration)

    expression = rate * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 2.0, 0.01, 1.0],
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
