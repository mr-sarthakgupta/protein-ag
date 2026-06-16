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


def _best(results: list[dict]) -> dict:
    """Return the result with the lowest finite nmse_val."""
    valid = [
        r for r in results
        if r.get("nmse_val") is not None
        and r["nmse_val"] == r["nmse_val"]  # exclude NaN
        and r["nmse_val"] < 1e15
    ]
    if not valid:
        return results[0]
    return min(valid, key=lambda r: r["nmse_val"])


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Evaluate multiple physically-motivated amyloid kinetics templates.

    Each template is fit independently per dataset (constants vary, structure
    is universal). Returns the template with the lowest validation NMSE.

    Templates cover:
    - Logistic with additive concentration-dependent lag (T1, T2)
    - Gompertz / generalized Richards asymmetric sigmoid (T3)
    - Hill-type nucleation with power-law half-time (T4)
    - Avrami stretched-exponential nucleation-growth (T5)
    - Finke-Watzky two-step nucleation+autocatalysis (T6)
    - Boltzmann sigmoidal with concentration shift (T7)
    """
    x = feature_symbols(X_train.shape[1])
    t = x[0]   # time
    s = x[1]   # concentration / experimental parameter

    results = []

    # ------------------------------------------------------------------
    # T1: Logistic with additive lag; lag shrinks with concentration.
    # rate = c0*s^c1, t_half = c2 + c3*s^(-c1)
    # y = c4 / (1 + exp(-rate*(t - t_half))) + c5
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    rate1 = c[0] * s ** c[1]
    t_half1 = c[2] + c[3] * s ** (-c[1])
    expr1 = c[4] / (1 + sp.exp(-rate1 * (t - t_half1))) + c[5]
    results.append(evaluate_expression(
        expr1, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.2, 0.5, 5.0, 2.0, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T2: Logistic with independent power-law rate and half-time.
    # rate = c0*s^c1, t_half = c2*s^c3
    # y = c4 / (1 + exp(-rate*(t - t_half))) + c5
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    rate2 = c[0] * s ** c[1]
    t_half2 = c[2] * s ** c[3]
    expr2 = c[4] / (1 + sp.exp(-rate2 * (t - t_half2))) + c[5]
    results.append(evaluate_expression(
        expr2, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.3, 0.4, 8.0, -0.3, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T3: Gompertz / asymmetric sigmoid — better for skewed lag phases.
    # y = c3 * exp(-exp(-c0*s^c1*(t - c2*s^(-c1)))) + c4
    # ------------------------------------------------------------------
    c = constant_symbols(5)
    rate3 = c[0] * s ** c[1]
    t_half3 = c[2] * s ** (-c[1])
    expr3 = c[3] * sp.exp(-sp.exp(-rate3 * (t - t_half3))) + c[4]
    results.append(evaluate_expression(
        expr3, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.2, 0.4, 8.0, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T4: Hill nucleation with concentration-dependent half-time.
    # t_half = c2*s^c3, cooperativity n=c0
    # y = c4 * t^n / (t_half^n + t^n) + c5
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    n4 = c[0]
    t_half4 = c[2] * s ** c[3]
    expr4 = c[4] * t ** n4 / (t_half4 ** n4 + t ** n4) + c[5]
    results.append(evaluate_expression(
        expr4, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[2.0, 1.0, 10.0, -0.3, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T5: Avrami stretched-exponential nucleation-growth.
    # k = c0*s^c1, n = c2
    # y = c3 * (1 - exp(-k * t^n)) + c4
    # ------------------------------------------------------------------
    c = constant_symbols(5)
    k5 = c[0] * s ** c[1]
    n5 = c[2]
    expr5 = c[3] * (1 - sp.exp(-k5 * t ** n5)) + c[4]
    results.append(evaluate_expression(
        expr5, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.01, 0.5, 2.0, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T6: Finke-Watzky two-step: nucleation + autocatalytic growth.
    # Analytical FW: y ~ A*(1 - 1/(1 + (k2/k1)*A*exp((k1+k2*A)*t)))
    # Simplified: rate_eff = c0 + c1*s^c2
    # y = c3*(1 - exp(-rate_eff*t)) / (1 + c4*exp(-rate_eff*t)) + c5
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    rate6 = c[0] + c[1] * s ** c[2]
    expr6 = (
        c[3] * (1 - sp.exp(-rate6 * t))
        / (1 + c[4] * sp.exp(-rate6 * t))
        + c[5]
    )
    results.append(evaluate_expression(
        expr6, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.05, 0.05, 0.5, 1.0, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T7: Boltzmann sigmoidal — widely used for ThT fluorescence data.
    # y = c0 + (c1 - c0) / (1 + exp((c2*s^c3 - t) / c4*s^c5))
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    t_mid7 = c[2] * s ** c[3]
    width7 = c[4] * s ** c[5]
    expr7 = c[0] + (c[1] - c[0]) / (1 + sp.exp((t_mid7 - t) / width7))
    results.append(evaluate_expression(
        expr7, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.0, 1.0, 10.0, -0.3, 3.0, -0.2],
    ))

    # ------------------------------------------------------------------
    # T8: Double-logistic (biphasic) — secondary nucleation pathways.
    # y = c0/(1+exp(-c1*(t-c2*s^c3))) + c4/(1+exp(-c5*(t-c6))) + c7
    # ------------------------------------------------------------------
    c = constant_symbols(8)
    expr8 = (
        c[0] / (1 + sp.exp(-c[1] * (t - c[2] * s ** c[3])))
        + c[4] / (1 + sp.exp(-c[5] * (t - c[6])))
        + c[7]
    )
    results.append(evaluate_expression(
        expr8, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.5, 0.3, 10.0, -0.3, 0.5, 0.2, 5.0, 0.0],
    ))

    return _best(results)


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
