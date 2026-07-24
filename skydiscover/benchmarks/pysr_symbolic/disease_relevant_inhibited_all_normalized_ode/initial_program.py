# EVOLVE-BLOCK-START
"""ODE discovery seed for normalized Abeta42 inhibitor aggregation kinetics."""

from __future__ import annotations

from typing import Any

import sympy as sp
from numpy.typing import NDArray

from pysr_harness.equation_session import (
    algebraic_equation,
    constant_symbols,
    evaluate_equation_system,
    feature_symbols,
    ode_equation,
)


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """
    Bounded aggregation-rate ODE with explicit inhibitor modulation.

    Features: x0 = normalized time, x1 = initial monomer m0, x2 = seed M0,
    x3 = inhibitor concentration cd, and x4 = current normalized
    concentration/state.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    inhibitor_scale = sp.Symbol("inhibitor_scale")
    source_rate = sp.Symbol("source_rate")
    autocatalytic_rate = sp.Symbol("autocatalytic_rate")
    capacity = sp.Symbol("capacity")
    equations = [
        algebraic_equation(
            inhibitor_scale,
            1 + c[5] ** 2 * inhibitor + c[6] ** 2 * inhibitor * monomer,
        ),
        algebraic_equation(
            source_rate,
            c[0] ** 2 * monomer ** c[1] + c[2] ** 2 * seed,
        ),
        algebraic_equation(
            autocatalytic_rate,
            c[3] ** 2 * monomer ** c[4] * (1 + c[7] ** 2 * seed),
        ),
        algebraic_equation(capacity, c[9] - concentration),
        ode_equation(
            capacity
            * (c[8] ** 2 + (source_rate + autocatalytic_rate * concentration) / inhibitor_scale)
            + c[10] * time * capacity
        ),
    ]

    return evaluate_equation_system(
        equations,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 0.1, 1.0, 0.5, 1.0, 0.0, 0.0, 0.01, 1.0, 0.0],
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
    """Load the inhibitor dataset for local testing."""
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
