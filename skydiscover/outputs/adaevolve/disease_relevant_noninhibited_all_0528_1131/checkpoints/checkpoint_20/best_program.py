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
    Gompertz growth model for protein aggregation kinetics with affine x1 dependence.

    Physical motivation:
    - Protein aggregation (ThT fluorescence) follows an asymmetric sigmoidal curve
      with a nucleation lag phase, rapid elongation, and a plateau.
    - The Gompertz CDF y = A * exp(-exp(-k*(t - t0))) is the standard model for
      this asymmetric shape: it naturally has a longer lag phase and sharper rise
      than a symmetric logistic/tanh sigmoid, without needing a separate lag factor.
    - The double-exponential structure keeps predictions bounded in [0, A] for all
      real inputs, giving excellent numerical stability during least-squares fitting.
    - Rate k and half-time t0 use affine x1 dependence: k = c0 + c1*x1 and
      t0 = c2 + c3*x1. This is globally defined for all x1 (sequential indices,
      concentrations, pH, etc.) unlike power-law x1^c which can be ill-conditioned
      when x1 spans orders of magnitude or when x1 is a sequential index.
    - Only 6 constants (vs 7 in the tanh+lag model), improving fitting stability
      on small datasets.

    Template:
        y = c4 * exp(-exp(-(c0 + c1*x1) * (x0 - c2 - c3*x1))) + c5

    Constants:
      c0 = base growth rate (~0.0001, typical for time in seconds ~10000-60000)
      c1 = linear x1 modulation of rate (~0.0, conservative start)
      c2 = base half-time (~20000.0, mid-range for x0 in seconds)
      c3 = linear x1 modulation of half-time (~0.0, conservative start)
      c4 = amplitude/plateau (~1.0)
      c5 = baseline offset (~0.0)

    6 constants total. Numerically stable: inner exp argument -(c0+c1*x1)*(x0-c2-c3*x1)
    is typically negative for t > t0 and positive for t < t0, keeping outer exp
    in (0, exp(1)] and the product bounded in [0, 1].
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    rate = c[0] + c[1] * parameter
    half_time = c[2] + c[3] * parameter
    plateau = c[4]
    baseline = c[5]

    gompertz = sp.exp(-sp.exp(-rate * (time - half_time)))
    expression = plateau * gompertz + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.0001, 0.0, 20000.0, 0.0, 1.0, 0.0],
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
