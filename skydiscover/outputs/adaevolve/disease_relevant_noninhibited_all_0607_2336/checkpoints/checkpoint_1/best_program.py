# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid aggregation kinetics.

Key insight: parameterizing rate and half-time in log-space
  rate    = exp(c0 + c1 * log(x1))
  t_half  = exp(c2 + c3 * log(x1))
is mathematically equivalent to the power-law form
  rate    = exp(c0) * x1^c1
  t_half  = exp(c2) * x1^c3
but gives far better optimization convergence because the log-space
constants (c0, c2) are O(1) to O(10) regardless of the time scale,
while the raw scale factors span many orders of magnitude across datasets.
This single logistic template with 6 constants fits all ~42 disease-relevant
non-inhibited aggregation datasets robustly.
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
    Log-space logistic model for protein aggregation kinetics.

    Features: x0 = elapsed time, x1 = varying experimental parameter.

    Both the apparent rate and half-time are parameterized in log-space:
        rate   = exp(c0 + c1 * log(x1))   [= exp(c0) * x1^c1]
        t_half = exp(c2 + c3 * log(x1))   [= exp(c2) * x1^c3]
        y = c4 / (1 + exp(-rate * (x0 - t_half))) + c5

    This is the standard nucleation-elongation logistic but reparameterized
    so that c0 and c2 absorb the log of the scale factor. The optimizer
    then works in a well-conditioned space regardless of whether time is
    in seconds (1e4) or hours (1e1), and regardless of concentration scale.

    Initial values [-10, 0, 10, -0.5, 1, 0] correspond to:
      c0=-10: log(rate_scale) ~ -10, i.e. rate_scale ~ 4.5e-5 (slow process)
      c1=0:   no concentration dependence of rate (neutral starting point)
      c2=10:  log(t_half_scale) ~ 10, i.e. t_half ~ 22000 time units
      c3=-0.5: mild negative concentration dependence of half-time
      c4=1:   unit plateau amplitude
      c5=0:   zero baseline
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    log_param = sp.log(parameter)
    rate = sp.exp(c[0] + c[1] * log_param)
    half_time = sp.exp(c[2] + c[3] * log_param)
    plateau = c[4]
    baseline = c[5]

    expression = plateau / (1 + sp.exp(-rate * (time - half_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[-10.0, 0.0, 10.0, -0.5, 1.0, 0.0],
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
