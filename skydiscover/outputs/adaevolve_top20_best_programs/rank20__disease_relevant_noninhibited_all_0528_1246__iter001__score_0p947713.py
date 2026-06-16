# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Scale-invariant logistic in arcsinh-transformed time with
log-linear concentration dependence. The arcsinh(x0 - c2) time
transformation is globally defined (handles t=0, t<0, and t up to 10^6),
making the half-time parameter c3 interpretable on a log scale that spans
all observed dataset time ranges (seconds to hours). The x1 dependence
enters linearly on the log scale for both rate and half-time, capturing
power-law concentration effects common in nucleation-limited aggregation.
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
    Scale-invariant logistic in arcsinh-time with log-linear x1 coupling.

    Template:
        lt  = log(x0 - c2 + sqrt((x0 - c2)^2 + 1))   # asinh(x0 - c2)
        lx1 = log(x1)
        rate     = c1 + c5 * lx1
        halftime = c3 + c6 * lx1
        y = c0 / (1 + exp(-rate * (lt - halftime))) + c4

    Key design choices:
    - asinh(x0 - c2): globally defined for all real x0 (handles t=0,
      negative t, and time scales from 10 to 700000 with fixed c3 init).
    - log(x1): log-linear concentration dependence captures power-law
      kinetics (nucleation rate ~ [conc]^n becomes linear in log space).
    - c2=0 init: time shift starts at zero (no pre-shift assumed).
    - c3=3 init: arcsinh half-time ≈ sinh(3) ≈ 10 time units, which is
      near the geometric midpoint of observed time ranges.
    - c5=c6=0 init: starts with x1-independent model, optimizer adds
      concentration dependence as needed.
    - 7 constants total, well within the 12-constant limit.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    # arcsinh(time - c2) via log form — globally defined, scale-invariant
    u = time - c[2]
    log_time = sp.log(u + sp.sqrt(u ** 2 + 1))

    log_param = sp.log(parameter)

    rate = c[1] + c[5] * log_param
    half_time = c[3] + c[6] * log_param

    plateau = c[0]
    baseline = c[4]

    expression = plateau / (1 + sp.exp(-rate * (log_time - half_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0],
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
