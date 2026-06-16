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
    Nucleation-elongation kinetics: logistic with power-law x1 scaling and
    lag-phase suppression via a saturating exponential factor.

    Physical motivation:
    - Protein aggregation (ThT fluorescence) follows a sigmoidal curve with
      a nucleation lag phase, rapid elongation, and a plateau.
    - The logistic captures the elongation + plateau. The factor
      (1 - exp(-c6*x0)) suppresses signal at early times, mimicking the
      nucleation lag. This product gives an asymmetric sigmoid with a
      longer lag than a pure logistic.
    - Rate and half-time scale as power laws of x1 (concentration/parameter):
      this is the best-performing x1-dependence structure from prior runs.
    - x1 is rescaled 0→1, so power-law x1^c is well-defined and stable.

    Template:
        y = c4 * (1 - exp(-c6*x0)) / (1 + exp(-c0*x1^c1*(x0 - c2*x1^c3))) + c5

    Constants:
      c0 = base growth rate
      c1 = power-law exponent for rate vs x1
      c2 = base half-time
      c3 = power-law exponent for half-time vs x1
      c4 = amplitude (plateau)
      c5 = baseline offset
      c6 = lag-phase decay rate (suppresses early signal)

    7 constants total. Numerically stable: all operations globally defined
    for x1 in (0,1] and x0 >= 0.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    rate = c[0] * parameter ** c[1]
    half_time = c[2] * parameter ** c[3]
    plateau = c[4]
    baseline = c[5]
    lag_rate = c[6]

    lag_factor = 1 - sp.exp(-lag_rate * time)
    sigmoid = 1 / (1 + sp.exp(-rate * (time - half_time)))
    expression = plateau * lag_factor * sigmoid + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 10.0, -0.5, 1.0, 0.0, 0.1],
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
