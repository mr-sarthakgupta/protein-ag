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
    Extended Finke-Watzky ODE: primary nucleation + x1-scaled autocatalysis.

    dc/dt = (c0 + c1*x1*x2) * (1 - x2)

    Physical interpretation:
      - c0*(1-x2): constant primary nucleation rate; c0 sets the lag
        phase duration independently of monomer concentration x1.
      - c1*x1*x2*(1-x2): secondary/autocatalytic nucleation whose rate
        scales with monomer concentration x1 (raw µM) and existing
        aggregate x2; captures the concentration-dependent acceleration
        seen across diverse protein systems.
      - (1-x2): depletion factor pins the plateau at c=1.

    Compared to (c0*x1 + c1*x2)*(1-x2), moving x1 to the autocatalytic
    term decouples the baseline nucleation rate from the raw x1 scale
    (which spans 0.3–3950 µM), improving numerical stability while
    retaining the concentration-dependent growth rate needed for
    multi-curve validation datasets.
    Complexity = 13 nodes, 2 constants.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(2)

    parameter = x[1]
    concentration = x[2]

    growth = (c[0] + c[1] * parameter * concentration) * (1 - concentration)

    return evaluate_expression(
        growth,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 0.1],
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
