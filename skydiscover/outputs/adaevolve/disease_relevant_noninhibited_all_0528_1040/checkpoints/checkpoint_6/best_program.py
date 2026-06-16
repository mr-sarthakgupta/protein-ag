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
    Gompertz sigmoid with exponential x1 modulation for protein aggregation.

    The Gompertz model captures the asymmetric sigmoidal shape typical of
    nucleation-dependent protein aggregation (fast rise, slow approach to
    plateau) better than the symmetric logistic.

    Form:
        rate      = c0 * exp(c1 * x1)   (exponential rate scaling with parameter)
        half_time = c2 * exp(c3 * x1)   (exponential lag-time scaling)
        y = c4 * exp(-exp(-rate * (x0 - half_time))) + c5

    Key advantage over power-law x1^c:
    - exp(c * x1) is defined for ALL real x1, including x1=0 (sequential index)
      and negative x1, whereas x1^c fails at x1=0 or x1<0
    - exp(c * x1) > 0 always, so rate stays positive and half_time stays
      positive (when c2 > 0), ensuring physically meaningful kinetics
    - Captures exponential concentration-dependence (Arrhenius-like scaling)
      which is physically motivated for nucleation-dependent aggregation

    Stability analysis:
    - rate = c0*exp(c1*x1): always positive when c0>0, smooth for any x1
    - half_time = c2*exp(c3*x1): always positive when c2>0
    - exp(-exp(...)): double exponential maps all reals to (0,1), bounded
    - c4 scales the plateau (~1 for rescaled data), c5 is baseline offset (~0)

    Initial values: c0=0.1 (slow rate), c1=0.0 (start neutral on x1 rate),
    c2=10.0 (moderate lag), c3=0.0 (start neutral on x1 lag),
    c4=1.0 (full plateau), c5=0.0 (zero baseline).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    rate = c[0] * sp.exp(c[1] * parameter)
    half_time = c[2] * sp.exp(c[3] * parameter)
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
        initial_values=[0.1, 0.0, 10.0, 0.0, 1.0, 0.0],
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
