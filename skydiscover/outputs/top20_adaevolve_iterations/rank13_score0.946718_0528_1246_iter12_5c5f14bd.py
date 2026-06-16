# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Logistic/Gompertz mixture in arcsinh-transformed time with
hybrid concentration dependence. The core innovation is a shape parameter
c9 that smoothly interpolates between a symmetric logistic (c9>>0) and
an asymmetric Gompertz (c9<<0) via alpha=(1+tanh(c9))/2:

    y = c0 * [alpha/(1+exp(-z)) + (1-alpha)*exp(-exp(-z))] + c4

where z = rate*(asinh(t-c2) - halftime). This handles both symmetric
sigmoidal curves (most datasets) AND the strongly right-skewed curves
seen in e.g. lysozyme/Hasecke2018 where the logistic completely fails
(NMSE~1.0) but the Gompertz fits perfectly (NMSE~0.0).

The arcsinh time transform is globally defined for all real t. The x1
dependence uses power-law rate and hybrid log-quadratic halftime.
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
    Logistic/Gompertz mixture in arcsinh-time with hybrid x1 coupling.

    Template:
        u        = x0 - c2
        lt       = log(u + sqrt(u^2 + 1))              # asinh(x0 - c2)
        lx1      = log(x1)
        rate     = c1 * x1^c5                          # power-law rate
        halftime = c3*x1^c6 + c7*lx1 + c8*lx1^2       # hybrid halftime
        z        = rate * (lt - halftime)
        alpha    = (1 + tanh(c9)) / 2                  # mixing weight in (0,1)
        y = c0 * [alpha/(1+exp(-z)) + (1-alpha)*exp(-exp(-z))] + c4

    Key design choices:
    - Mixture logistic+Gompertz: c9 controls shape. c9>>0 → pure logistic
      (symmetric sigmoid); c9<<0 → pure Gompertz (right-skewed, asymmetric).
      c9=0 init → 50/50 mix, safe starting point for all datasets.
    - This directly fixes the Hasecke2018 lysozyme dataset where the
      logistic completely fails (NMSE~1.0) but Gompertz fits perfectly
      (NMSE~0.0) due to a strongly right-skewed aggregation curve.
    - asinh(x0 - c2): globally defined for all real x0.
    - rate = c1*x1^c5: power-law. c5=0 init → x1-independent start.
    - halftime = c3*x1^c6 + c7*log(x1) + c8*log(x1)^2: hybrid form.
    - 10 constants total, 112 nodes, well within limits.
    - max_nfev=1000 for convergence on hard datasets.
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

    z = rate * (log_time - half_time)

    # Shape parameter: alpha in (0,1) interpolates logistic <-> Gompertz
    alpha = (1 + sp.tanh(c[9])) / 2

    logistic_part = 1 / (1 + sp.exp(-z))
    gompertz_part = sp.exp(-sp.exp(-z))

    plateau = c[0]
    baseline = c[4]

    expression = plateau * (alpha * logistic_part + (1 - alpha) * gompertz_part) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        max_nfev=1000,
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
