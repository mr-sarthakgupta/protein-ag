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
    Three-pathway amyloid ODE: primary nucleation + concentration-scaled
    autocatalysis + fragmentation-driven secondary nucleation.

    dc/dt = (c0 + c1*x1*x2 + c2*x2**2) * (1 - x2)

    Physical interpretation:
      - c0*(1-x2): constant primary nucleation rate; governs lag phase length.
      - c1*x1*x2*(1-x2): surface-catalyzed secondary nucleation proportional
        to monomer concentration x1 and existing aggregate x2; captures the
        concentration-dependent sigmoid acceleration seen across diverse
        protein systems (Abeta, IAPP, htt, alpha-syn, etc.).
      - c2*x2**2*(1-x2): fragmentation/fibril-end-catalyzed secondary
        nucleation proportional to aggregate squared; captures Oosawa-Kasai
        autocatalysis independent of monomer concentration, important for
        seeded datasets and systems with strong fragmentation.
      - (1-x2): depletion factor pins the plateau at c=1.

    The c2 term adds one constant and ~4 complexity nodes but captures a
    physically distinct mechanism that the 2-constant form cannot represent.
    Complexity ~ 16 nodes, 3 constants.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    growth = (c[0] + c[1] * parameter * concentration + c[2] * concentration ** 2) * (1 - concentration)

    return evaluate_expression(
        growth,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 0.1, 0.1],
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
