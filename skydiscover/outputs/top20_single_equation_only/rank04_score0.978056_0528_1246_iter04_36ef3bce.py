# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Scale-invariant logistic in arcsinh-transformed time with
hybrid concentration dependence. The arcsinh(x0 - c2) time transformation
is globally defined (handles t=0, t<0, and t up to 10^6). The x1
dependence uses a power-law rate (c1*x1^c5) and a hybrid halftime
(c3*x1^c6 + c7*log(x1) + c8*log(x1)^2), capturing both monotonic
power-law kinetics and non-monotonic concentration dependence seen in
some aggregation datasets (e.g. huntingtin polyQ repeats).
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

    Template:
        u        = x0 - c2
        lt       = log(u + sqrt(u^2 + 1))              # asinh(x0 - c2)
        lx1      = log(x1)
        rate     = c1 * x1^c5                          # power-law rate
        halftime = c3*x1^c6 + c7*lx1 + c8*lx1^2       # hybrid halftime
        y = c0 / (1 + exp(-rate * (lt - halftime))) + c4

    Key design choices:
    - asinh(x0 - c2): globally defined for all real x0. Compresses wide
      time-scale variation across 60 datasets (seconds to ~10^6 seconds).
    - rate = c1*x1^c5: power-law. c5=0 init gives x1-independent start.
    - halftime = c3*x1^c6 + c7*log(x1) + c8*log(x1)^2: hybrid form.
      When c7=c8=0 (init), reduces exactly to the pure power-law model.
      The log and log^2 terms capture non-monotonic x1 dependence seen
      in huntingtin polyQ datasets where halftime is non-monotonic in
      concentration. The log^2 term allows a U-shaped or inverted-U
      halftime-vs-concentration curve.
    - All extra constants init to 0: x1-independent start, optimizer adds
      x1 coupling only when data supports it.
    - 9 constants total, complexity=51 nodes, well within limits.
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
        initial_values=[1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
