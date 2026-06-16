# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Scale-invariant logistic in arcsinh-transformed time with
hybrid concentration dependence. Identical structure to the best-known
template (Programs 3/4, score 0.9782), but with improved initialization:
c3 is initialized to 10.0 (vs 3.0) to match the empirical median of
asinh(x0_midpoint) ≈ 10.96 across all 60 datasets, and max_nfev=2000
for better convergence on hard datasets.

Template:
    u        = x0 - c2
    lt       = log(u + sqrt(u^2 + 1))              # asinh(x0 - c2)
    lx1      = log(x1)
    rate     = c1 * x1^c5                          # power-law rate
    halftime = c3*x1^c6 + c7*lx1 + c8*lx1^2       # hybrid halftime
    y = c0 / (1 + exp(-rate * (lt - halftime))) + c4
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
    Logistic in arcsinh-time with hybrid power-law + log-quadratic x1 coupling.

    Identical structure to Programs 3/4 (best known: nmse=0.0223, score=0.9782)
    with two targeted improvements:
    1. c3 initialized to 10.0 (was 3.0): matches the empirical median of
       asinh(x0_midpoint) = 10.96 across all 60 datasets. The optimizer
       previously had to travel from c3=3 to ~11 for most datasets; now it
       starts near the typical half-time in transformed space.
    2. max_nfev=2000 (was 1000): doubled optimizer budget for harder datasets
       (incomplete sigmoidal curves, stochastic nucleation data, wide time
       scales). Runtime remains well within the 3000s total timeout.

    Template:
        u        = x0 - c2
        lt       = log(u + sqrt(u^2 + 1))              # asinh(x0 - c2)
        lx1      = log(x1)
        rate     = c1 * x1^c5                          # power-law rate
        halftime = c3*x1^c6 + c7*lx1 + c8*lx1^2       # hybrid halftime
        y = c0 / (1 + exp(-rate * (lt - halftime))) + c4

    Key design choices:
    - asinh(x0 - c2): globally defined for all real x0 including negative
      time offsets (biofilm TasA: x0 starts at -1600, serum Ye: -260).
    - rate = c1*x1^c5: power-law. c5=0 init gives x1-independent start.
    - halftime = c3*x1^c6 + c7*log(x1) + c8*log(x1)^2: hybrid form.
      When x1=1 (sequential index, single-column datasets): log(1)=0 so
      halftime = c3, rate = c1. Clean 5-parameter logistic for these cases.
    - c3=10.0 init: asinh midpoints range 2.45-14.38, median=10.96.
      Starting at 10 reduces the optimization distance for ~75% of datasets.
    - 9 constants total, complexity well within limits.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    parameter = x[1]

    # arcsinh(time - c2) via log form — globally defined, scale-invariant
    u = time - c[2]
    log_time = sp.log(u + sp.sqrt(u ** 2 + 1))

    # log(x1) for log-quadratic halftime coupling
    log_param = sp.log(parameter)

    rate = c[1] * parameter ** c[5]
    half_time = c[3] * parameter ** c[6] + c[7] * log_param + c[8] * log_param ** 2

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
        initial_values=[1.0, 2.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        max_nfev=2000,
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
