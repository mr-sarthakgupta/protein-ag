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
    Primary + autocatalytic secondary nucleation ODE for normalized amyloid kinetics.

    dc/dt = (1 - x2) * (c2 + (c0 + c1*x1) * x2)

    Biological basis:
    - (1 - x2): saturation factor; growth halts as normalized concentration -> 1
    - c2: primary nucleation rate — spontaneous, concentration-independent;
      ensures nonzero dc/dt even at x2=0, capturing lag-phase initiation
    - (c0 + c1*x1) * x2: elongation/secondary nucleation — autocatalytic growth
      proportional to existing fibril mass x2, with rate modulated by
      experimental parameter x1 (protein concentration, seed fraction, etc.)

    Expanding: dc/dt = c2*(1-x2) + (c0+c1*x1)*x2*(1-x2)
    This separates primary nucleation (c2 term, active at x2~0) from
    autocatalytic elongation ((c0+c1*x1)*x2*(1-x2), dominant at mid-range x2),
    covering the full range from primary-nucleation-dominated to
    secondary-nucleation-dominated kinetics.

    Complexity ~13 nodes (vs 16 for x2**2 form), improving parsimony_penalty_factor
    from 0.9800 to ~0.9838. The x1 modulation of the elongation rate captures
    concentration/seed-dependent acceleration of the growth phase.

    Features: x0 = normalized time, x1 = normalized experimental parameter,
    x2 = current normalized concentration c (ODE state).
    Constants: c0 = base elongation rate, c1 = x1 sensitivity, c2 = primary nucleation.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    elongation_rate = c[0] + c[1] * parameter
    expression = (1 - concentration) * (c[2] + elongation_rate * concentration)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 0.1],
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
