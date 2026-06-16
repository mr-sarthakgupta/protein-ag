# EVOLVE-BLOCK-START
"""ODE discovery for normalized multi-dataset amyloid/protein aggregation kinetics.

Key structural insight from data analysis:
- All 42 datasets are sigmoidal aggregation curves (0 -> 1 normalized)
- Most start near c=0 (unseeded), requiring a nucleation term to escape zero
- Some datasets (seeded) start at nonzero c
- Varying parameter x1 modulates the nucleation/growth rate

Model: dc/dt = (c0 + c1*x1) * (1 - c) * (1 + c2*c)

This captures three biophysical mechanisms in one compact expression:
  - (c0 + c1*x1): primary nucleation rate, modulated by experimental parameter
  - (1 - c): saturation / monomer depletion term
  - (1 + c2*c): autocatalytic amplification (secondary nucleation)

When c=0: dc/dt = (c0 + c1*x1) > 0 -- escapes zero (unlike pure logistic)
When c->1: dc/dt -> 0 -- saturates correctly
Numerically stable: no divisions, no exponentials, no variable powers.
"""

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
    Nucleation-elongation-autocatalysis ODE for protein aggregation.

    Template: d(c)/dt = (c0 + c1*x1) * (1 - c) * (1 + c2*c)

    - c0: basal nucleation/growth rate (allows escape from c=0)
    - c1: parameter-dependent rate modulation (concentration, pH, etc.)
    - c2: autocatalytic amplification factor (secondary nucleation)
    - (1-c): saturation as monomer is depleted
    - x1: normalized experimental parameter (concentration, pH, etc.)
    - x2=c: current normalized aggregate concentration (ODE state)

    Works for both unseeded (c_init~0) and seeded (c_init>0) experiments.
    Complexity: 16 nodes, 3 fitted constants per dataset.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    nucleation_rate = c[0] + c[1] * parameter
    saturation = 1 - concentration
    autocatalysis = 1 + c[2] * concentration

    expression = nucleation_rate * saturation * autocatalysis

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 1.0, 2.0],
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
