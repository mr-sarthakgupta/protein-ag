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
    ODE RHS for normalized multi-dataset protein aggregation kinetics.

    Reparameterized nucleation-elongation model (complexity=15):

        d(c)/dt = c0 * (c1 + x2) * (1 + c2*x1) * (1 - x2)

    This spans the identical function subspace as (c0+c1*x2)*(1+c2*x1)*(1-x2)
    but uses one fewer node (complexity=15 vs 16), giving a parsimony gain.

    - c0: overall rate scale
    - c1: primary/secondary nucleation ratio (lag phase control)
    - (c1 + x2): primary nucleation c0*c1 plus autocatalytic secondary c0*x2
    - (1 + c2*x1): linear modulation by experimental parameter x1
    - (1 - x2): monomer depletion factor

    The Finke-Watzky / Knowles nucleation-elongation framework gives
    dc/dt = (k_n + k_2*c)*(1-c), which maps directly to this form with
    k_n = c0*c1 and k_2 = c0, both modulated by (1+c2*x1).
    Constants are fitted independently per dataset.

    Initial values chosen to cover a broader basin: c0=3 (moderate rate),
    c1=0.05 (small lag-phase ratio), c2=0.5 (moderate x1 sensitivity).
    The evaluator's multistart scales [1.0, 0.3, 3.0] then sweep:
    c0 in {0.9, 3.0, 9.0}, c1 in {0.015, 0.05, 0.15}, c2 in {0.15, 0.5, 1.5}
    providing better coverage of hard-case datasets with sharp lag phases
    or strong concentration dependence.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(3)

    parameter = x[1]
    concentration = x[2]

    growth = c[0] * (c[1] + concentration) * (1 + c[2] * parameter)
    expression = growth * (1 - concentration)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[3.0, 0.05, 0.5],
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
