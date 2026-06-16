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
    Power-law concentration secondary nucleation ODE for amyloid aggregation.

    dc/dt = (c0 + c1 * x1**c4 * x2 + c2 * x1 * x2**2) * (1 - x2)

    Physical basis (Knowles et al. 2009, Cohen et al. 2013, Meisl et al. 2014):
      - c0*(1-x2): primary nucleation (concentration-independent lag phase).
      - c1*x1**c4*x2*(1-x2): elongation with power-law concentration dependence;
        c4 is the fitted nucleation/elongation exponent. In the Knowles master
        equation framework, primary nucleation scales as m^nc (nc~2-6) and
        elongation scales as m^1. By fitting c4 the model adapts to datasets
        with different nucleation orders (Abeta, htt, IAPP, lysozyme, insulin).
      - c2*x1*x2**2*(1-x2): secondary nucleation proportional to fibril surface
        area x2**2 and monomer x1; captures the n2~2 scaling seen in Abeta/IAPP.
      - (1-x2): depletion factor pins plateau at c=1 for normalized data.

    x1 is raw concentration (always > 0 in uM), so x1**c4 is well-defined for
    any real c4. Complexity ~20 nodes (lower than exp-term variant at 25 nodes),
    giving better parsimony factor. 5 constants, c4 initialized near 1.0.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    parameter = x[1]
    concentration = x[2]

    growth = (c[0] + c[1] * parameter ** c[4] * concentration + c[2] * parameter * concentration ** 2) * (1 - concentration)

    return evaluate_expression(
        growth,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 0.1, 0.1, 1.0, 1.0],
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