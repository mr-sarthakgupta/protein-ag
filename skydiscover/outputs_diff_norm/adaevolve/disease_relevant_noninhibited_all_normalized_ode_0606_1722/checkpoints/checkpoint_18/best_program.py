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
    Secondary nucleation ODE (Cohen/Meisl framework) for normalized
    protein aggregation kinetics.

    Mechanism:
      - Nucleation term c0*(1-x2): provides nonzero rate at c=0,
        enabling sigmoidal rise from near-zero initial states (seeded
        and unseeded conditions).
      - Autocatalytic term c1*(1+c2*x1)*x2**2*(1-x2): secondary
        nucleation rate proportional to fibril mass squared (n2=2
        exponent from Cohen et al. 2013 / Meisl et al. 2014), creating
        sharper sigmoidal transitions than the logistic x2*(1-x2).
        Rate modulated by experimental parameter x1.

    d(c)/dt = c0*(1 - x2) + c1*(1 + c2*x1)*x2**2*(1 - x2)

    The x2**2 term captures the cooperative secondary nucleation
    mechanism dominant in Abeta, IAPP, alpha-synuclein, and htt
    aggregation, producing sharper lag phases and faster transitions
    that better fit datasets with explosive nucleation kinetics.

    Features: x0=normalized time, x1=normalized experimental parameter,
    x2=current normalized concentration state c.
    Constants: c0=nucleation rate, c1=secondary nucleation rate,
    c2=parameter sensitivity.

    Stability: rate→0 as x2→1 (plateau), nonzero at x2=0 (nucleation).
    Fixed integer exponent (n2=2) avoids numerical instability.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    nucleation = c[0] * (1 - concentration)
    growth = c[1] * (1 + c[2] * parameter) * concentration**2 * (1 - concentration)

    expression = nucleation + growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 50.0, 2.0],
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
