# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid aggregation kinetics.

Tries multiple physically-motivated equation templates and returns the best.
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
    Evaluate multiple physically-motivated sigmoidal templates for amyloid
    aggregation kinetics and return the best-fitting one.

    Templates tried:
    1. Generalized Richards/logistic with concentration-dependent rate and lag:
         y = c4 / (1 + exp(-c0*(x0 - c2 - c3/x1^c1)))^(1/c5) + c6
       Collapses gracefully when x1=1 (single-concentration datasets).

    2. Hill-type nucleation model (robust, avoids exp overflow):
         y = c4 * x0^c0 / (c1^c0 + x0^c0) + c5
       where c1 scales with concentration: c1 = c2 * x1^c3

    3. Finke-Watzky inspired double-rate logistic with lag phase:
         y = c4 / (1 + exp(-c0*(x0 - c2))) * (1 - exp(-c1*x0)) + c5

    4. Stretched exponential (Avrami) nucleation-growth:
         y = c3 * (1 - exp(-c0 * x0^c1)) + c4
       where c0 scales with concentration: c0 = c5 * x1^c2

    The best result (lowest nmse_val) is returned.
    """
    x = feature_symbols(X_train.shape[1])
    t = x[0]   # time
    s = x[1]   # concentration / experimental parameter

    results = []

    # ------------------------------------------------------------------ #
    # Template 1: Generalized Richards curve with concentration-dependent
    # half-time shift.  rate = c0*s^c1, t_half = c2 + c3/s^c1
    # y = c4 / (1 + exp(-rate*(t - t_half))) + c5
    # ------------------------------------------------------------------ #
    c = constant_symbols(6)
    rate = c[0] * s ** c[1]
    t_half = c[2] + c[3] * s ** (-c[1])
    expr1 = c[4] / (1 + sp.exp(-rate * (t - t_half))) + c[5]
    r1 = evaluate_expression(
        expr1, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.2, 0.5, 5.0, 2.0, 1.0, 0.0],
    )
    results.append(r1)

    # ------------------------------------------------------------------ #
    # Template 2: Hill / sigmoidal nucleation with concentration-dependent
    # half-time.  t_half = c2 * s^c3
    # y = c4 * t^c0 / ((c2*s^c3)^c0 + t^c0) + c5
    # ------------------------------------------------------------------ #
    c = constant_symbols(6)
    n = c[0]          # Hill coefficient (cooperativity)
    t_half2 = c[2] * s ** c[3]
    expr2 = c[4] * t ** n / (t_half2 ** n + t ** n) + c[5]
    r2 = evaluate_expression(
        expr2, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[2.0, 1.0, 10.0, -0.3, 1.0, 0.0],
    )
    results.append(r2)

    # ------------------------------------------------------------------ #
    # Template 3: Logistic with lag — captures nucleation delay cleanly.
    # rate = c0 * s^c1, lag = c2 * s^(-c3)
    # y = c4 / (1 + exp(-rate*(t - lag))) + c5
    # ------------------------------------------------------------------ #
    c = constant_symbols(6)
    rate3 = c[0] * s ** c[1]
    lag3 = c[2] * s ** (-c[3])
    expr3 = c[4] / (1 + sp.exp(-rate3 * (t - lag3))) + c[5]
    r3 = evaluate_expression(
        expr3, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.3, 0.4, 8.0, 0.3, 1.0, 0.0],
    )
    results.append(r3)

    # ------------------------------------------------------------------ #
    # Template 4: Avrami / stretched-exponential nucleation-growth.
    # k = c0 * s^c1, n = c2
    # y = c3 * (1 - exp(-k * t^n)) + c4
    # ------------------------------------------------------------------ #
    c = constant_symbols(5)
    k4 = c[0] * s ** c[1]
    n4 = c[2]
    expr4 = c[3] * (1 - sp.exp(-k4 * t ** n4)) + c[4]
    r4 = evaluate_expression(
        expr4, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.01, 0.5, 2.0, 1.0, 0.0],
    )
    results.append(r4)

    # ------------------------------------------------------------------ #
    # Template 5: Double-logistic (biphasic) — secondary nucleation.
    # Some amyloid systems show two-phase kinetics.
    # y = c0/(1+exp(-c1*(t-c2))) + c3/(1+exp(-c4*(t-c5*s^c6))) + c7
    # ------------------------------------------------------------------ #
    c = constant_symbols(8)
    expr5 = (
        c[0] / (1 + sp.exp(-c[1] * (t - c[2])))
        + c[3] / (1 + sp.exp(-c[4] * (t - c[5] * s ** c[6])))
        + c[7]
    )
    r5 = evaluate_expression(
        expr5, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.5, 0.3, 5.0, 0.5, 0.2, 10.0, -0.3, 0.0],
    )
    results.append(r5)

    # ------------------------------------------------------------------ #
    # Template 6: Finke-Watzky two-step model (nucleation + autocatalysis).
    # Analytical solution: y = A*(1 - k1/(k1+k2*A)*exp(-(k1+k2*A)*t)) + B
    # Simplified: rate_eff = c0 + c1*s^c2, amplitude modulated by s
    # y = c3 * (1 - c4*exp(-rate_eff*t)) / (1 + c4*exp(-rate_eff*t)) + c5
    # (Equivalent to shifted logistic with multiplicative amplitude)
    # ------------------------------------------------------------------ #
    c = constant_symbols(6)
    rate6 = c[0] + c[1] * s ** c[2]
    expr6 = c[3] * (1 - sp.exp(-rate6 * t)) / (1 + c[4] * sp.exp(-rate6 * t)) + c[5]
    r6 = evaluate_expression(
        expr6, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.05, 0.05, 0.5, 1.0, 1.0, 0.0],
    )
    results.append(r6)

    # Return the result with the lowest validation NMSE
    valid = [r for r in results if r.get("nmse_val") is not None
             and r["nmse_val"] == r["nmse_val"]]  # exclude NaN
    if not valid:
        return results[0]
    return min(valid, key=lambda r: r["nmse_val"])


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
