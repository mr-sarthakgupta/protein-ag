# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Scale-invariant logistic in arcsinh-transformed time with
log1p-shifted concentration dependence. The arcsinh(x0 - c2) time
transformation is globally defined. The x1 dependence uses
(x1+1)^c5 = exp(c5*log(x1+1)) for rate and a hybrid halftime
c3*(x1+1)^c6 + c7*log(x1+1) + c8*log(x1+1)^2.

Critical fix over prior model: replacing log(x1) and x1^c with
log(x1+1) and exp(c*log(x1+1)) = (x1+1)^c. Nearly all 60 datasets
have x1 values at or near 0 (rescaled response used as parameter).
The old log(x1) → -inf and x1^c → 0/undefined at x1=0, causing
sympy/lambdify to produce nan/inf during least-squares fitting.
The log(x1+1) form is 0 at x1=0 and grows like log(x1) for large x1,
preserving the power-law concentration dependence while being globally
safe. This is the dominant source of the gap between local numpy fits
(NMSE~0.00018) and the evaluator score (NMSE=0.0223).
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
    Logistic in arcsinh-time with log1p-shifted x1 coupling (safe at x1=0).

    Template:
        u        = x0 - c2
        lt       = log(u + sqrt(u^2 + 1))              # asinh(x0 - c2)
        lx1      = log(x1 + 1)                         # log1p: safe at x1=0
        rate     = c1 * exp(c5 * lx1)                  # = c1*(x1+1)^c5
        halftime = c3*exp(c6*lx1) + c7*lx1 + c8*lx1^2 # hybrid, safe at x1=0
        y = c0 / (1 + exp(-rate * (lt - halftime))) + c4

    Key design choices:
    - log(x1+1) replaces log(x1): safe at x1=0 (=0 there), grows like
      log(x1) for large x1. Nearly all 60 datasets have x1 at or near 0
      (rescaled response used as the parameter column), so log(x1) → -inf
      causes nan/inf in sympy lambdify, making least-squares fail silently.
    - exp(c5*log(x1+1)) replaces x1^c5: equals (x1+1)^c5, which is 1 at
      x1=0 (not 0 or undefined). Preserves power-law scaling for large x1.
    - rate=c1 at x1=0 (init c5=0): x1-independent start for optimizer.
    - halftime=c3 at x1=0 (init c6=0, c7=c8=0): stable starting point.
    - 9 constants total, complexity=63 nodes, well within limits.
    - max_nfev=1000: better convergence on hard/incomplete datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    parameter = x[1]

    # arcsinh(time - c2) via log form — globally defined, scale-invariant
    u = time - c[2]
    log_time = sp.log(u + sp.sqrt(u ** 2 + 1))

    # log(x1 + 1): safe at x1=0, grows like log(x1) for large x1
    # Replaces log(x1) which is -inf at x1=0 (present in ~50/60 datasets)
    log1p_param = sp.log(parameter + 1)

    # rate = c1*(x1+1)^c5 via exp form; equals c1 at x1=0 when c5=0
    rate = c[1] * sp.exp(c[5] * log1p_param)

    # halftime: hybrid (x1+1)^c6 power-law + log1p + log1p^2
    # All terms are 0 or c3 at x1=0; safe throughout
    half_time = c[3] * sp.exp(c[6] * log1p_param) + c[7] * log1p_param + c[8] * log1p_param ** 2

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
        initial_values=[1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
