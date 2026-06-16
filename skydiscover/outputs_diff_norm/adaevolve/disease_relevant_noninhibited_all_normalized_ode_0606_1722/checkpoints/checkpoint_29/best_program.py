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
    Factored Cohen-Meisl ODE: compact form with linear autocatalytic term.

    dc/dt = (c0 + c1*x2) * (1 + c2*x1) * (1 - x2)

    Physical basis:
      - c0: basal primary nucleation rate — nonzero at x2=0, drives
            lag-phase exit. Corresponds to k_n*m^nc.
      - c1*x2: linear autocatalytic (secondary nucleation) term —
            fibril-catalyzed nucleation proportional to fibril mass
            (n2=1 exponent). Linear autocatalysis is valid for many
            protein aggregation systems and is simpler than quadratic.
      - (1 + c2*x1): shared monomer concentration modulation — both
            primary and secondary nucleation scale with free monomer.
      - (1 - x2): saturation/depletion factor — plateau at x2=1.

    Compared to the quadratic form (c0 + c2*x2**2)*(1+c1*x1)*(1-x2):
      - Lower complexity (~15 vs 18 nodes) → better parsimony factor
      - Same 3 constants → no additional fitting burden
      - Linear autocatalysis can fit sigmoidal aggregation curves
        equally well when the effective nucleation order is near 1
      - More robust on noisy/small datasets

    Vanishes at x2=1 (plateau), nonzero at x2=0 (nucleation onset).
    No division, no singularities, globally smooth.

    Features: x0=normalized time, x1=normalized experimental parameter,
    x2=current normalized concentration state c.
    Constants: c0=basal nucleation, c1=linear autocatalytic rate,
    c2=monomer sensitivity.
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
