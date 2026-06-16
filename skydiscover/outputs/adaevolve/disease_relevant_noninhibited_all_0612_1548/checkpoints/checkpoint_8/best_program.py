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
    Generalized Richards sigmoidal model with power-law concentration scaling
    for nucleation-dependent protein aggregation kinetics.

    Features: x0 = normalized time, x1 = concentration-like experimental parameter.

    Key structural choices:
    - Power-law concentration scaling for rate and half-time:
        rate      = c0 * x1^c1
        half_time = c2 * x1^c3
      Physically motivated by the Knowles/Michaels kinetic framework:
      aggregation rate scales as [monomer]^(n2/2) and half-time as
      [monomer]^(-n2/2), where n2 is the secondary nucleation order.
      Power-law is superior to log-scaling for datasets spanning wide
      concentration ranges (e.g., serum amyloid: 3.7-74 uM; insulin:
      344-3444 uM), where the half-time shifts dramatically with concentration.
      When x1 is constant (single-curve datasets), both c1 and c3 are
      absorbed into c0 and c2 respectively, giving a standard sigmoid.
    - Richards/generalized logistic form:
        y = c4 / (1 + exp(-rate*(x0 - half_time)))^c5 + c6
      The free exponent c5 generalizes the standard logistic (c5=1) to
      capture asymmetric sigmoid shapes from seeded/secondary nucleation
      kinetics where aggregation accelerates rapidly once nuclei form.

    7 constants, complexity 25.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    rate = c[0] * parameter ** c[1]
    half_time = c[2] * parameter ** c[3]
    growth = sp.exp(-rate * (time - half_time))

    expression = c[4] / (1 + growth) ** c[5] + c[6]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 1.0, 0.0],
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
