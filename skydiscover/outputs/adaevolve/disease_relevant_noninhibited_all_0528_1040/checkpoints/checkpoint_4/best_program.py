# EVOLVE-BLOCK-START
"""Symbolic regression seed for multi-dataset amyloid aggregation kinetics."""

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
    Gompertz sigmoid with power-law x1 dependence for protein aggregation.

    The Gompertz model captures the asymmetric sigmoidal shape typical of
    nucleation-dependent protein aggregation (fast rise, slow approach to
    plateau) better than the symmetric logistic. The inner double-exponential
    structure is always real and finite for any real inputs.

    Form:
        rate      = c0 * x1^c1          (power-law rate scaling with parameter)
        half_time = c2 * x1^c3          (power-law lag-time scaling)
        y = c4 * exp(-exp(-rate * (x0 - half_time))) + c5

    Stability analysis:
    - x1^c is evaluated as exp(c * log(x1)); x1 is always positive (conc/index)
    - The inner arg -rate*(x0-half_time) is linear in x0, bounded for finite t
    - exp(-exp(...)) maps all reals to (0,1), so the full expression is bounded
    - c4 scales the plateau (~1 for rescaled data), c5 is baseline offset (~0)

    The Gompertz is asymmetric: slower approach to plateau than departure from
    baseline, which matches the nucleation-elongation mechanism where fibrils
    grow faster once nuclei form but slow as monomer is depleted.

    Power-law x1 dependence (vs affine) better captures the concentration
    scaling seen in the best-performing context solutions (score 0.5917).

    Initial values: c0=0.1 (slow rate), c1=0.5 (mild concentration scaling),
    c2=10.0 (moderate lag), c3=-0.5 (lag decreases with concentration),
    c4=1.0 (full plateau), c5=0.0 (zero baseline).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    rate = c[0] * parameter ** c[1]
    half_time = c[2] * parameter ** c[3]
    plateau = c[4]
    baseline = c[5]

    expression = plateau * sp.exp(-sp.exp(-rate * (time - half_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 0.0],
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
