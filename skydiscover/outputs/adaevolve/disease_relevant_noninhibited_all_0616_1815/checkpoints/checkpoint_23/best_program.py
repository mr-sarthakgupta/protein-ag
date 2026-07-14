# EVOLVE-BLOCK-START
"""Symbolic regression: Richards logistic with monomer-scaled rate and lag for amyloid aggregation."""

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
    Richards (asymmetric logistic) with monomer-scaled rate AND
    monomer-scaled, seed-shortened lag offset.

    Physical basis (Knowles secondary nucleation model):
      - Growth rate scales as c0 * x1^c1 (monomer power law)
      - Lag phase scales as c2 * x1^c3 / (1 + c4*x2):
          * monomer-dependent: higher monomer -> shorter lag
          * seed-dependent: more seeds -> shorter lag (denominator >= 1)
      - Richards shape exponent c5 captures right-skewed curves
        (amyloid aggregation is typically right-skewed: slow rise, fast plateau)

    This combines the best features of the top two candidates:
      - P3's Richards shape exponent (asymmetry, complexity=26)
      - P4's monomer-scaled lag term (better NMSE)
    at complexity ~28, 6 constants.

    Template:
        rate = c0 * x1^c1
        lag  = c2 * x1^c3 / (1 + c4*x2)
        y    = c5 / (1 + exp(-rate*x0 + lag))^c6

    Numerically stable: denominator (1+c4*x2) >= 1 for x2>=0, c4>=0.
    Base (1 + exp(...)) >= 1 always, so power c6 is real for any c6.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[0] * monomer ** c[1]
    lag = c[2] * monomer ** c[3] / (1 + c[4] * seed)

    expression = c[5] / (1 + sp.exp(-rate * time + lag)) ** c[6]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.5, 2.5, -0.5, 1.0, 1.0, 1.0],
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
