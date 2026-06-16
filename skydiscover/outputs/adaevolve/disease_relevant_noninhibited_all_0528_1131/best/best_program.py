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
    Hill-function model with x1-dependent amplitude for protein aggregation kinetics.

    Physical motivation:
    - Protein aggregation follows nucleation-growth kinetics captured by a Hill
      function: y = A(x1) * t^n / (K^n + t^n) + B.
    - Key extension: amplitude A = c3 + c6*x1 varies linearly with the
      experimental parameter x1 (concentration, pH, etc.). Different conditions
      often reach different final fluorescence plateaus, so a fixed plateau
      underfits multi-condition data. Making A affine in x1 captures this.
    - The Hill coefficient n (c0) controls steepness of the sigmoidal rise.
    - Half-saturation K = c1 + c2*x1 shifts the transition time with x1.
    - Time offset c5 keeps (x0 + c5) > 0 when x0=0 (avoids 0^n issues).
    - Baseline c4 handles non-zero starting fluorescence.

    Template:
        y = (c3 + c6*x1) * (x0 + c5)^c0 / ((c1 + c2*x1)^c0 + (x0 + c5)^c0) + c4

    where:
      c0 = Hill exponent n (~2.0, controls sharpness of transition)
      c1 = base half-saturation time (~20000.0)
      c2 = linear x1 modulation of half-saturation (~0.0)
      c3 = base amplitude/plateau (~1.0)
      c4 = baseline offset (~0.0)
      c5 = small time offset to keep (x0+c5) > 0 when x0=0 (~1.0)
      c6 = linear x1 modulation of amplitude (~0.0, start near current model)

    7 constants. Numerically stable: (x0+c5) > 0 when c5 > 0,
    (c1+c2*x1) > 0 for reasonable c1, Hill ratio bounded in [0,1].
    c6 initialized to 0.0 so fitting starts from the proven 6-constant model.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    # Hill exponent (steepness)
    hill_n = c[0]
    # Half-saturation constant: shifts linearly with experimental parameter
    half_sat = c[1] + c[2] * parameter
    # Amplitude: x1-dependent plateau (key structural extension)
    plateau = c[3] + c[6] * parameter
    # Baseline offset
    baseline = c[4]
    # Small time offset to keep (time + offset) > 0 even when time=0
    time_offset = c[5]

    t_shifted = time + time_offset
    # Hill function: t^n / (K^n + t^n)
    t_pow = t_shifted ** hill_n
    k_pow = half_sat ** hill_n
    hill = t_pow / (k_pow + t_pow)

    expression = plateau * hill + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[2.0, 20000.0, 0.0, 1.0, 0.0, 1.0, 0.0],
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
