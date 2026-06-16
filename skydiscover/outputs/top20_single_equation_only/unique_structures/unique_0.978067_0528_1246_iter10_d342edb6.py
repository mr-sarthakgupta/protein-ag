# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid/protein aggregation kinetics.

Strategy: Scale-invariant logistic in arcsinh-transformed time with
numerically robust concentration dependence. Uses (1+x1)^c power laws
and log(1+x1) — both well-defined at x1=0 (affecting datasets where x1
is a sequential index starting at 0). Replacing x1^c with (1+x1)^c
removes 0^c singularities for negative exponents; replacing log(x1) with
log(1+x1) removes -inf at x1=0. Hybrid halftime (power-law + log-quadratic)
captures both monotonic and non-monotonic concentration dependence.

Research basis: Protein aggregation kinetics follow nucleation-elongation
mechanisms where the half-time t50 ~ c^(-gamma) (power law in concentration),
but non-primary nucleation pathways (secondary nucleation, fragmentation)
create non-monotonic or log-curved t50 vs concentration relationships.
The log-quadratic term in halftime captures these deviations from pure
power-law scaling observed in huntingtin polyQ and other datasets.
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
    Logistic in arcsinh-time with (1+x1)^c power laws and log(1+x1) coupling.

    Template:
        u        = x0 - c2
        lt       = log(u + sqrt(u^2 + 1))                    # asinh(x0 - c2)
        lx1      = log(1 + x1)                               # =0 at x1=0, finite always
        rate     = c1 * (1 + x1)^c5                         # =c1 at x1=0 for any c5
        halftime = c3*(1+x1)^c6 + c7*lx1 + c8*lx1^2        # =c3 at x1=0
        y = c0 / (1 + exp(-rate * (lt - halftime))) + c4

    Key design choices:
    - asinh(x0 - c2): globally defined for all real x0, compresses wide
      time-scale variation across 60 datasets (seconds to ~10^6 seconds).
    - (1+x1)^c replaces x1^c: finite at x1=0 for any exponent c.
    - log(1+x1) replaces log(x1): gives 0 at x1=0 (not -inf).
    - When c5=c6=c7=c8=0 (init), reduces to x1-independent logistic.
    - Hybrid halftime captures non-monotonic t50 vs concentration curves.
    - 9 constants, max_nfev=1000 for better convergence on hard datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    parameter = x[1]

    # arcsinh(time - c2) via log form — globally defined, scale-invariant
    u = time - c[2]
    log_time = sp.log(u + sp.sqrt(u ** 2 + 1))

    # log(1+x1): well-defined at x1=0 (gives 0), avoids log(0)=-inf
    log_param = sp.log(1 + parameter)

    # (1+x1)^c: well-defined at x1=0 for any exponent (gives 1)
    rate = c[1] * (1 + parameter) ** c[5]
    half_time = c[3] * (1 + parameter) ** c[6] + c[7] * log_param + c[8] * log_param ** 2

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
