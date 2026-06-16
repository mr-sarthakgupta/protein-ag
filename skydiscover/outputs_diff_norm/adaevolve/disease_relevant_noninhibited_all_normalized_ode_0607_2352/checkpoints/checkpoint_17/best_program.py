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
    Secondary nucleation ODE for normalized amyloid aggregation kinetics.

    dc/dt = (c0 + c1*x1) * (c2 + x2**2) * (1 - x2)

    Biological basis (Knowles/Cohen secondary nucleation framework):
    - (c0 + c1*x1): effective rate modulated by experimental parameter x1
    - (c2 + x2**2): combined nucleation driver:
        c2 = primary nucleation offset (spontaneous; enables growth from x2~0)
        x2**2 = secondary nucleation term (fibril-catalyzed; proportional to
                fibril mass squared, characteristic of surface-catalyzed
                nucleation on existing fibrils)
    - (1 - x2): saturation; growth stops as normalized concentration -> 1

    Compared to the linear (c2 + x2) form, the quadratic x2**2 term produces
    a sharper, more asymmetric sigmoidal rise concentrated in the mid-range,
    matching the steep growth phase seen in secondary-nucleation-dominated
    disease-relevant amyloid data (Abeta, alpha-synuclein, IAPP, etc.).
    Expression is always non-negative for x2 in [0,1] when c2 >= 0.
    Same 3 constants and complexity-14 as the linear form.

    Features: x0 = normalized time, x1 = normalized experimental parameter,
    x2 = current normalized concentration c (ODE state).
    Constants: c0 = base rate, c1 = parameter sensitivity, c2 = nucleation offset.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    rate = c[0] + c[1] * parameter
    nucleation_driver = c[2] + concentration**2

    expression = rate * nucleation_driver * (1 - concentration)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 0.1],
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
