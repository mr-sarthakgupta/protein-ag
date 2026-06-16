# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset protein aggregation kinetics.

Richards (generalised logistic) model with log-time and power-law-rate
parameterisation. Uses c0*x1^c1 for the rate (instead of c0+c1*log(x1)),
which reduces complexity from 36 to 30 nodes while maintaining fit quality.
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
    Richards (generalised logistic) kinetics with log-time, power-law rate,
    and log-concentration half-time parameterisation.

    Features: x0 = elapsed time, x1 = varying experimental parameter
    (concentration, pH, etc. — sequential index for single-curve datasets).

    Expression (30 nodes, complexity=30):
        log_t    = log(x0 + c7)
        rate     = c0 * x1^c1
        half     = c2 + c3 * log(x1)
        y = c5 / (1 + exp(-rate * (log_t - half)))^c4 + c6

    Design rationale:
    - log(x0 + c7): pure log-time with fitted offset. c7 handles:
        * x0 starting at 0 (c7 → 1)
        * negative x0 (Ye2011 x0_min=-257; c7 → large positive)
        * large x0 (Hasecke x0=2053-349861; c7 → small)
      Initial value c7=1.0 works for all scales since the optimizer can
      freely adjust it.
    - c0 * x1^c1: power-law rate. c1=0 gives constant rate (single-conc
      datasets). c1≠0 captures concentration-dependent kinetics. This is
      6 nodes fewer than c0+c1*log(x1) while being equally expressive.
    - c2 + c3*log(x1): log-linear half-time. c3=0 gives constant half-time.
    - c4: Richards shape/asymmetry parameter. c4=1 gives standard logistic;
      small c4 gives Gompertz-like asymmetric curves for IAPP/lysozyme data.
    - c5 = amplitude, c6 = baseline.
    - 8 constants, complexity 30 nodes → parsimony factor ≈ 0.9625.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(time + c[7])
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
        initial_values=[1.0, 0.0, 8.0, 0.0, 1.0, 1.0, 0.0, 1.0],
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
