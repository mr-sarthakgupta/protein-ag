# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset protein aggregation kinetics.

Richards (generalised logistic) model with log-time using a positivity-
guaranteed time offset: log(x0 + exp(c7)).

Key insight: using exp(c7) as the time offset guarantees the log argument
is always positive (exp(c7) > 0 always), preventing the optimizer from
finding invalid solutions where log(x0 + c7) becomes undefined. With
c7_init=5.6, exp(5.6)≈270 > 256.6 = |x0_min| for the most negative
dataset (Ye2011), ensuring a valid initial point for all 42 datasets.

This fixes a critical failure mode in the current model: with log(x0+c7)
and c7_init=1.0, the Ye2011 dataset (x0_min=-256.6) starts at an invalid
point (log(-255.6)=nan) and the optimizer gets completely stuck, yielding
nmse≈1.2. With log(x0+exp(c7)) and c7_init=5.6, the initial point is
valid and the optimizer finds an excellent fit.
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
    Richards (generalised logistic) kinetics with guaranteed-positive log-time
    offset, power-law rate, and log-concentration half-time.

    Features: x0 = elapsed time, x1 = varying experimental parameter
    (concentration, pH, etc. — sequential index for single-curve datasets).

    Expression (complexity=31):
        log_t    = log(x0 + exp(c7))
        rate     = c0 * x1^c1
        half     = c2 + c3 * log(x1)
        y = c5 / (1 + exp(-rate * (log_t - half)))^c4 + c6

    Design rationale:
    - log(x0 + exp(c7)): log-time with guaranteed-positive offset. exp(c7)
      is always > 0, so the log argument is always x0 + positive_number.
      With c7_init=5.6, exp(5.6)≈270 > 256.6 = |x0_min| for Ye2011
      (the most negative dataset), ensuring a valid initial evaluation.
      For large x0 (htt datasets, x0 up to 1.7M), the optimizer reduces
      c7 so exp(c7) << x0, recovering log(x0). For x0 starting at 0,
      exp(c7) acts as a small offset that the optimizer adjusts freely.
    - c0 * x1^c1: power-law rate. c1=0 gives constant rate for single-
      concentration datasets. Physically motivated by nucleation kinetics
      where rate ∝ [M]^n_c.
    - c2 + c3*log(x1): log-linear half-time. c3=0 for constant half-time.
    - c4: Richards shape/asymmetry parameter (c4=1: standard logistic).
    - c5 = amplitude, c6 = baseline.
    - 8 constants, complexity=31 nodes → parsimony factor ≈ 0.9613.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(time + sp.exp(c[7]))
    rate = c[0] * parameter ** c[1]
    half_time = c[2] + c[3] * sp.log(parameter)
    shape = c[4]
    plateau = c[5]
    baseline = c[6]

    expression = plateau / (1 + sp.exp(-rate * (log_time - half_time))) ** shape + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.0, 8.0, 0.0, 1.0, 1.0, 0.0, 5.6],
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
