# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Generalized Richards logistic in arcsinh-transformed time with
hybrid concentration dependence. Extends the best-known template (Program 3,
score 0.9782) by adding a Richards asymmetry exponent c9, which allows the
sigmoid to be asymmetric around its inflection point. This captures datasets
where aggregation curves rise faster than they plateau, or vice versa.

Template:
    u        = x0 - c2
    lt       = log(u + sqrt(u^2 + 1))              # asinh(x0 - c2)
    lx1      = log(x1)
    rate     = c1 * x1^c5                          # power-law rate
    halftime = c3*x1^c6 + c7*lx1 + c8*lx1^2       # hybrid halftime
    y = c0 * (1 + exp(-rate * (lt - halftime)))^(-c9) + c4

With c9=1 this recovers the standard logistic (Program 3/4 structure).
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
    Richards generalized logistic in arcsinh-time with hybrid x1 coupling.

    Template:
        u        = x0 - c2
        lt       = log(u + sqrt(u^2 + 1))              # asinh(x0 - c2)
        lx1      = log(x1)
        rate     = c1 * x1^c5                          # power-law rate
        halftime = c3*x1^c6 + c7*lx1 + c8*lx1^2       # hybrid halftime
        y = c0 * (1 + exp(-rate * (lt - halftime)))^(-c9) + c4

    Key design choices:
    - asinh(x0 - c2): globally defined for all real x0. Compresses wide
      time-scale variation across 60 datasets (seconds to ~10^6 seconds).
    - rate = c1*x1^c5: power-law. c5=0 init gives x1-independent start.
    - halftime = c3*x1^c6 + c7*log(x1) + c8*log(x1)^2: hybrid form
      capturing both monotonic and non-monotonic concentration dependence.
    - c9: Richards asymmetry exponent. c9=1 init → standard logistic.
      c9>1 → faster rise (early-stage nucleation dominated).
      c9<1 → slower rise (elongation dominated, plateau approached gradually).
    - (1 + exp(-z))^(-c9) is numerically stable: argument always >= 1,
      so the base is always positive, making the power well-defined.
    - 10 constants total, well within the 13-constant limit.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

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
    richards_exp = c[9]

    # Richards generalized logistic: (1 + exp(-z))^(-nu)
    # Base (1 + exp(-z)) >= 1 always, so power is always real and positive
    sigmoid_arg = rate * (log_time - half_time)
    expression = plateau * (1 + sp.exp(-sigmoid_arg)) ** (-richards_exp) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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
