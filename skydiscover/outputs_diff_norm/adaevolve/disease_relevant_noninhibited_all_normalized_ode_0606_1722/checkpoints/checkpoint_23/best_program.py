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
    Factored Cohen-Meisl ODE: monomer factor modulates all kinetic rates.

    Mechanism (factored form of secondary nucleation kinetics):
      dc/dt = (c0 + c2*x2**2) * (1 + c1*x1) * (1 - x2)

    Physical basis (Cohen et al. 2013, Meisl et al. 2014):
      - The full secondary nucleation model is:
          dc/dt = (k_n*m^nc + k2*m^n2*c^2) * (1-c)
        where m = monomer concentration (x1 in normalized form).
      - In the factored approximation, both primary nucleation (k_n)
        and secondary nucleation (k2*c^2) share the same monomer
        scaling factor (1+c1*x1), since both rates scale with free
        monomer concentration.
      - c0: basal nucleation rate (nonzero at c=0, drives lag-phase exit)
      - c1: monomer concentration sensitivity (shared by both terms)
      - c2: secondary nucleation amplitude (autocatalytic, c^2 exponent)

    Advantages over unfactored form:
      - Lower complexity (~15 nodes vs 23) → better parsimony penalty
      - Same 3 constants → no additional fitting burden
      - Monomer modulation is symmetric across all rate terms
      - Vanishes at x2=1 (plateau), nonzero at x2=0 (nucleation onset)
      - Numerically stable: fixed integer exponents, no division/log

    Features: x0=normalized time, x1=normalized experimental parameter,
    x2=current normalized concentration state c.
    Constants: c0=basal nucleation, c1=monomer sensitivity, c2=secondary rate.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    rate = (c[0] + c[2] * concentration**2) * (1 + c[1] * parameter)
    expression = rate * (1 - concentration)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 2.0, 50.0],
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
