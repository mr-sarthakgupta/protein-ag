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
    Nucleation-elongation ODE for normalized amyloid aggregation kinetics.

    Proposes dc/dt = (c0 + c1*x1) * (c2 + x2) * (1 - x2)

    This is a Finke-Watzky / nucleation-elongation inspired model:
    - (c0 + c1*x1): rate modulated by experimental parameter x1 (concentration)
    - (c2 + x2): nucleation offset c2 enables spontaneous growth even at low c;
      larger c2 shifts the inflection point to earlier times, matching
      primary-nucleation-dominated aggregation data
    - (1 - x2): saturation term; growth stops as c approaches the plateau (1)

    Unlike pure logistic (inflection always at c=0.5), this model can place
    the inflection at c = (1-c2)/2, covering the full range of early-to-late
    inflections seen across primary and secondary nucleation datasets.

    Features: x0 = normalized time, x1 = normalized experimental parameter,
    x2 = current normalized concentration c (ODE state).
    Constants: c0 = base rate, c1 = parameter sensitivity, c2 = nucleation offset.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    rate = c[0] + c[1] * parameter
    nucleation_offset = c[2]

    expression = rate * (nucleation_offset + concentration) * (1 - concentration)

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
