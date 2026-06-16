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
    Concentration-modulated secondary nucleation ODE for amyloid aggregation.

    dc/dt = (c0 + c1*x1*x2 + c2*x1*x2**2) * (1 - x2)

    All three growth terms scale with monomer concentration x1 (via c1, c2),
    consistent with Cohen et al. 2013 / Meisl et al. 2014 integrated rate
    laws where both elongation and secondary nucleation depend on free monomer:
      - c0*(1-x2): primary nucleation (lag phase; concentration-independent).
      - c1*x1*x2*(1-x2): elongation/surface-catalyzed secondary nucleation
        proportional to monomer x1 and aggregate x2.
      - c2*x1*x2**2*(1-x2): higher-order secondary nucleation proportional
        to monomer x1 and fibril surface area x2**2; captures the n2=2
        scaling seen in Abeta, IAPP, htt systems with varying concentration.
      - (1-x2): depletion factor pins plateau at c=1.

    Making all autocatalytic terms x1-dependent improves cross-dataset
    generalization when protein concentration varies widely (Abeta/Cohen2013,
    htt/Kakkar2016). Complexity ~ 17 nodes, 3 constants.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    growth = (c[0] + c[1] * parameter * concentration + c[2] * parameter * concentration ** 2) * (1 - concentration)

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
