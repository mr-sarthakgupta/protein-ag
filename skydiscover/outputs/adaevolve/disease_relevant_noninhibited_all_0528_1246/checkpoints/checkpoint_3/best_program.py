# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Scale-invariant logistic in arcsinh-transformed time with
multiplicative power-law concentration dependence. The arcsinh(x0 - c2)
time transformation is globally defined (handles t=0, t<0, and t up to
10^6). The x1 dependence enters as power laws: rate = c1*x1^c5 and
half_time = c3*x1^c6, capturing nucleation-limited aggregation kinetics
where both rate and lag time scale as power laws of concentration.

Key advantage over additive log coupling: when x1 is constant (many
datasets have a single x1 value), x1^c5 = const and x1^c6 = const so
the optimizer sees a well-conditioned landscape with c5/c6 as pure
scale factors. For varying x1, the power-law exponents c5 and c6 are
directly interpretable as nucleation orders. Initializing c5=c6=0 gives
x1-independent behavior at the start, letting the optimizer add x1
dependence only when the data supports it.
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
    Logistic in arcsinh-time with multiplicative power-law x1 coupling.

    Template:
        u        = x0 - c2
        lt       = log(u + sqrt(u^2 + 1))       # asinh(x0 - c2)
        rate     = c1 * x1^c5                   # power-law rate scaling
        halftime = c3 * x1^c6                   # power-law half-time scaling
        y = c0 / (1 + exp(-rate * (lt - halftime))) + c4

    Key design choices:
    - asinh(x0 - c2): globally defined for all real x0.
    - rate = c1*x1^c5: multiplicative power-law. When c5=0 (init), rate=c1
      regardless of x1, giving a stable x1-independent starting point.
      Optimizer adds x1 dependence by moving c5 away from 0.
    - halftime = c3*x1^c6: similarly, when c6=0 (init), halftime=c3.
      For insulin (x1=344-3444), c6~-0.33 gives t50 ∝ x1^(-0.33).
    - c5=c6=0 init: x1-independent start avoids bad local minima caused
      by initializing x1 coupling in the wrong direction.
    - 7 constants total, complexity=39, well within limits.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    # arcsinh(time - c2) via log form — globally defined, scale-invariant
    u = time - c[2]
    log_time = sp.log(u + sp.sqrt(u ** 2 + 1))

    rate = c[1] * parameter ** c[5]
    half_time = c[3] * parameter ** c[6]

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
