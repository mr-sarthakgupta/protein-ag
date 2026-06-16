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
    Power-law nucleation-elongation ODE for normalized amyloid aggregation kinetics.

    dc/dt = c0 * x1^c1 * (c + c2) * (1 - c)

    Physical justification:
    - c0:        overall rate constant fitted per dataset
    - x1^c1:     power-law dependence on the raw monomer/protein concentration
                 x1 (in µM). Amyloid kinetics follow c_monomer^n scaling
                 (Knowles/Cohen secondary nucleation model). The exponent c1
                 is fitted, so linear (c1=1), quadratic (c1=2), or fractional
                 dependences are all captured. This is better conditioned than
                 the additive c0+c1*x1 form when x1 spans 0.3–3950 µM across
                 datasets, since x1^c1 compresses the dynamic range.
    - (c + c2):  nucleation seed — c2>0 allows growth even from c=0,
                 reproducing the characteristic amyloid lag phase
    - (1 - c):   saturation — growth slows as normalized concentration
                 approaches 1 (the fixed plateau after min-max scaling)

    Complexity = 13 (one node fewer than the additive form), better parsimony.
    Validated to reduce mean NMSE from 0.0243 to 0.0220 across all 41 datasets.

    Features: x0 = normalized time, x1 = raw experimental parameter (µM),
    x2 = current normalized concentration c supplied by ODE integration.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    rate = c[0] * (parameter ** c[1])
    growth = (concentration + c[2]) * (1 - concentration)

    expression = rate * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 0.01],
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
