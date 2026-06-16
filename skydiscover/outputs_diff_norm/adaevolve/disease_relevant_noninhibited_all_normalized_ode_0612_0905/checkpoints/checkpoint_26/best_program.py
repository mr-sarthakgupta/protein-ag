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
    Seeded + unseeded secondary nucleation ODE for amyloid aggregation.

    dc/dt = (c0 + c1*x1*x2 + c2*x1*x2**2 + c3*exp(-c4*x0)) * (1 - x2)

    Terms:
      - c0*(1-x2): primary nucleation (concentration-independent lag phase).
      - c1*x1*x2*(1-x2): elongation proportional to monomer x1 and aggregate x2.
      - c2*x1*x2**2*(1-x2): higher-order secondary nucleation (fibril surface).
      - c3*exp(-c4*x0)*(1-x2): exponentially decaying seed-driven rate for
        seeded/pre-nucleated systems (Nielsen2001_seeded_sec, Hasecke2018_7uM_sec)
        where growth starts immediately without a lag phase. For unseeded
        datasets c3 fits near zero, leaving the other terms to dominate.
      - (1-x2): shared depletion factor pins plateau at c=1.

    The exp term is globally finite and decays safely to zero as x0→1.
    5 constants; complexity ~25 nodes; parsimony_factor ~0.969.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    time = x[0]
    parameter = x[1]
    concentration = x[2]

    growth = (c[0] + c[1] * parameter * concentration + c[2] * parameter * concentration ** 2 + c[3] * sp.exp(-c[4] * time)) * (1 - concentration)

    return evaluate_expression(
        growth,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 0.1, 0.1, 0.1, 1.0],
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
