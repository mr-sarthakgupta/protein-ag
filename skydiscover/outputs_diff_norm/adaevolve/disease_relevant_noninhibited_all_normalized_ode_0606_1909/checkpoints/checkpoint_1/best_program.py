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
    ODE RHS for normalized multi-dataset protein aggregation kinetics.

    Uses the nucleation-elongation form with parameter-dependent modulation:

        d(c)/dt = (c0 + c1*x2) * (1 + c2*x1) * (1 - x2)

    - (c0 + c1*x2): primary nucleation (c0) plus autocatalytic/secondary
      nucleation (c1*x2), the two dominant kinetic pathways in amyloid
      aggregation.
    - (1 + c2*x1): linear modulation by the experimental parameter x1
      (concentration, temperature, pH, etc.) to differentiate curves.
    - (1 - x2): depletion factor — rate goes to zero as monomer is consumed.

    This form is compact (complexity=16, 3 constants), smooth, globally
    defined, and robust across all 42 datasets. Constants are fitted
    independently per dataset.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    rate = (c[0] + c[1] * concentration) * (1 + c[2] * parameter)
    expression = rate * (1 - concentration)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 10.0, 2.0],
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
