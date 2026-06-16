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
    Refined Cohen-Meisl ODE with x1 modulation on primary nucleation.

    Mechanism:
      - Primary nucleation: c0*(1+c1*x1)*(1-x2)
        Nonzero at x2=0, drives lag-phase exit. Rate modulated by
        experimental parameter x1 (monomer concentration directly
        controls primary nucleation rate via power-law scaling).
        Vanishes at plateau x2->1.
      - Secondary nucleation: c2*x2**2*(1-x2)
        Pure autocatalytic term proportional to fibril mass squared
        (n2=2 exponent, Cohen et al. 2013 / Meisl et al. 2014).
        No x1 coupling — secondary nucleation depends on fibril surface
        area, not free monomer concentration in this factored form.

    d(c)/dt = c0*(1 + c1*x1)*(1 - x2) + c2*x2**2*(1 - x2)

    Moving x1 to the nucleation term is more physically motivated:
    primary nucleation rate scales with monomer concentration to the
    power nc, so x1 (normalized concentration/parameter) should
    modulate the nucleation onset rather than the growth rate.
    This better captures lag-phase timing dependence on x1 across
    diverse disease-relevant datasets (Abeta, IAPP, alpha-synuclein,
    htt, biofilm proteins, lysozyme, etc.).

    Same 3 constants as current best — identical parsimony penalty.
    Improvement comes purely from better structural fit quality.

    Features: x0=normalized time, x1=normalized experimental parameter,
    x2=current normalized concentration state c.
    Constants: c0=base nucleation rate, c1=x1 sensitivity of nucleation,
    c2=secondary nucleation rate.

    Stability: all terms vanish at x2=1 (plateau), nonzero at x2=0.
    Fixed integer exponents avoid numerical instability.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    nucleation = c[0] * (1 + c[1] * parameter) * (1 - concentration)
    growth = c[2] * concentration**2 * (1 - concentration)

    expression = nucleation + growth

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
