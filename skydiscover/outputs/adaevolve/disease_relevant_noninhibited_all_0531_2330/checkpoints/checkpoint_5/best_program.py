# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset protein aggregation kinetics.

Richards (generalised logistic) model with log-time and log-concentration
parameterisation. Handles the full diversity of disease-relevant aggregation
datasets: wide time ranges (seconds to days), single- and multi-concentration
experiments, and datasets where time starts at non-zero or even negative values.
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
    Richards (generalised logistic) kinetics with log-time and log-concentration.

    Features: x0 = elapsed time, x1 = varying experimental parameter
    (concentration, pH, etc. — sequential index for single-curve datasets).

    Expression:
        log_t    = log(c7 * x0 + 1)
        rate     = c0 + c1 * log(x1)
        half     = c2 + c3 * log(x1)
        y = c5 / (1 + exp(-rate * (log_t - half)))^c4 + c6

    Design rationale:
    - log(c7*x0 + 1): softplus-like time transform. c7 controls the time
      scale. With c7_init=0.001, valid for x0 > -1000 (covers all datasets
      including serum amyloid with x0_min=-257). Recovers log(x0) for large
      x0*c7, and linear for small x0*c7. Absorbs the huge variation in time
      scales (0-30 hours vs 47407-1718519 seconds).
    - c0 + c1*log(x1): concentration-dependent rate. c1=0 for single-conc
      datasets (x1 constant), which is handled gracefully.
    - c2 + c3*log(x1): concentration-dependent log half-time.
    - c4: Richards shape/asymmetry parameter. c4=1 gives standard logistic;
      small c4 gives Gompertz-like asymmetric curves (needed for IAPP, hRNPA).
    - c5 = amplitude, c6 = baseline.
    - 8 constants, complexity 36 nodes → parsimony factor ≈ 0.955.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(c[7] * time + 1)
    rate = c[0] + c[1] * sp.log(parameter)
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
        initial_values=[1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.001],
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
