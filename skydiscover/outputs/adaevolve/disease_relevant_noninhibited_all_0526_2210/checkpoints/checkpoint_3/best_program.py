# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid aggregation kinetics.

Tries multiple physically-motivated equation templates and returns the best.
Covers logistic, Gompertz, Hill, Avrami, Finke-Watzky, Boltzmann, double-logistic,
generalized Richards, log-logistic, and secondary-nucleation (Knowles) forms.
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
    """Return the result dict with the lowest finite nmse_val."""
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
    """Evaluate physically-motivated amyloid kinetics templates; return best.

    Templates tried (each fit independently per dataset):
    T1  Logistic, additive lag: rate=c0*s^c1, t_half=c2+c3*s^(-c1)
    T2  Logistic, power-law rate+half-time: rate=c0*s^c1, t_half=c2*s^c3
    T3  Gompertz asymmetric sigmoid (skewed lag phase)
    T4  Hill nucleation with concentration-dependent half-time
    T5  Avrami stretched-exponential nucleation-growth
    T6  Finke-Watzky two-step nucleation+autocatalysis
    T7  Boltzmann sigmoidal with concentration-dependent width
    T8  Double-logistic (biphasic / secondary nucleation pathways)
    T9  Generalized Richards (asymmetric, extra shape param nu)
    T10 Log-logistic (Fisk CDF) — heavy-tailed lag phase
    T11 Simplified Knowles secondary-nucleation: 1-(1+c*t)*exp(-k*t^2)
    T12 Logistic T2 with alternative initial values (multi-start)
    T13 Gompertz with independent concentration exponents
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
    # rate_eff = c0 + c1*s^c2
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
    # T7: Boltzmann sigmoidal with concentration-dependent width.
    # t_mid = c2*s^c3, width = c4*s^c5
    # y = c0 + (c1 - c0) / (1 + exp((t_mid - t) / width))
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

    # ------------------------------------------------------------------
    # T9: Generalized Richards (asymmetric sigmoid) with shape param.
    # Adds asymmetry exponent nu — generalizes logistic (nu=1).
    # rate = c0*s^c1, t_half = c2*s^c3, nu = c5 (asymmetry)
    # y = c4 / (1 + exp(-rate*(t - t_half)))^(1/nu) + c6
    # ------------------------------------------------------------------
    c = constant_symbols(7)
    rate9 = c[0] * s ** c[1]
    t_half9 = c[2] * s ** c[3]
    nu9 = c[5]
    expr9 = c[4] / (1 + sp.exp(-rate9 * (t - t_half9))) ** (1 / nu9) + c[6]
    results.append(evaluate_expression(
        expr9, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.3, 0.4, 8.0, -0.3, 1.0, 1.5, 0.0],
    ))

    # ------------------------------------------------------------------
    # T10: Log-logistic (Fisk CDF) — heavy-tailed nucleation lag.
    # t_half = c1*s^c2, shape = c0 (steepness)
    # y = c3 * t^c0 / (t_half^c0 + t^c0) + c4
    # (Identical to Hill but with different initial values for diversity)
    # ------------------------------------------------------------------
    c = constant_symbols(5)
    n10 = c[0]
    t_half10 = c[1] * s ** c[2]
    expr10 = c[3] * t ** n10 / (t_half10 ** n10 + t ** n10) + c[4]
    results.append(evaluate_expression(
        expr10, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[3.0, 15.0, -0.5, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T11: Simplified secondary nucleation (Knowles/Cohen inspired).
    # Quadratic-time exponent in Avrami captures secondary nucleation:
    # k = c0*s^c1, kappa = c2*s^c3 (secondary nucleation rate)
    # y = c4 * (1 - (1 + kappa*t) * exp(-k * t**2)) + c5
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    k11 = c[0] * s ** c[1]
    kappa11 = c[2] * s ** c[3]
    expr11 = c[4] * (1 - (1 + kappa11 * t) * sp.exp(-k11 * t ** 2)) + c[5]
    results.append(evaluate_expression(
        expr11, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.005, 0.5, 0.1, 0.3, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T12: Gompertz with independent concentration exponents for rate
    # and lag — more flexible than T3 which ties them together.
    # rate = c0*s^c1, t_lag = c2*s^c3
    # y = c4 * exp(-exp(-rate*(t - t_lag))) + c5
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    rate12 = c[0] * s ** c[1]
    t_lag12 = c[2] * s ** c[3]
    expr12 = c[4] * sp.exp(-sp.exp(-rate12 * (t - t_lag12))) + c[5]
    results.append(evaluate_expression(
        expr12, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.2, 0.4, 8.0, -0.3, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T13: T2 logistic with alternative initial values (multi-start).
    # Same structure as T2 but different starting point to escape
    # local optima — important for datasets with fast kinetics.
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    rate13 = c[0] * s ** c[1]
    t_half13 = c[2] * s ** c[3]
    expr13 = c[4] / (1 + sp.exp(-rate13 * (t - t_half13))) + c[5]
    results.append(evaluate_expression(
        expr13, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[1.0, -0.5, 3.0, 0.5, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # T14: Avrami with additive concentration term (linear + power-law).
    # k = c0*s^c1 + c2*s, n = c3
    # y = c4 * (1 - exp(-k * t^n)) + c5
    # Handles datasets where rate has both linear and nonlinear conc dep.
    # ------------------------------------------------------------------
    c = constant_symbols(6)
    k14 = c[0] * s ** c[1] + c[2] * s
    n14 = c[3]
    expr14 = c[4] * (1 - sp.exp(-k14 * t ** n14)) + c[5]
    results.append(evaluate_expression(
        expr14, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.01, 0.5, 0.001, 2.0, 1.0, 0.0],
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
