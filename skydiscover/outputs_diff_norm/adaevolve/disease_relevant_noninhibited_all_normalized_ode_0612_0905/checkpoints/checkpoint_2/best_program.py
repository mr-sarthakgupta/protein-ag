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
    Finke-Watzky-inspired ODE with x1-dependent primary nucleation.

    dc/dt = (c0*x1 + c1*x2) * (1 - x2)

    Physical interpretation:
      - c0*x1*(1-x2): primary nucleation rate proportional to monomer
        concentration x1 (drives the lag phase; small c0 → long lag)
      - c1*x2*(1-x2): autocatalytic/secondary nucleation (accelerates
        once aggregates form, saturates as monomer depletes)
      - (1-x2): depletion factor ensures plateau at c=1, matching
        min-max normalized data range [0,1]

    This form is scale-invariant in x1 through c0 (each dataset fits
    its own c0), captures the asymmetric sigmoid with lag phase, and
    uses only 2 constants for robust fitting on small datasets.
    Complexity = 13 nodes.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(2)

    parameter = x[1]
    concentration = x[2]

    growth = (c[0] * parameter + c[1] * concentration) * (1 - concentration)

    return evaluate_expression(
        growth,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 5.0],
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
