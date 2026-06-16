# EVOLVE-BLOCK-START
"""Symbolic regression seed for Alpha-synuclein Gaspar 2017 0.3uM seed data."""

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
    Fit multiple physically-motivated sigmoidal kinetics models for seeded
    alpha-synuclein aggregation and return the best-scoring one.

    Models tested:
    1. Gompertz: y = A * exp(-exp(-k*(t - t_mid))) with power-law conc dependence
    2. Logistic sigmoid with concentration-dependent half-time and rate
    3. Secondary nucleation: logistic with power-law lag and growth rate
    4. Avrami nucleation-growth with concentration-dependent exponent
    5. Double-exponential approach to plateau
    6. Hill equation with concentration-dependent t_half
    """
    x = feature_symbols(X_train.shape[1])
    time = x[0]
    conc = x[1]

    best_result = None
    best_score = -1.0

    # Model 1: Gompertz with power-law concentration dependence
    # k = c0 * conc^c1, t_mid = c2 * conc^(-c3)
    # y = c4 * exp(-exp(-k*(t - t_mid))) + c5
    c1 = constant_symbols(6)
    k1 = c1[0] * conc ** c1[1]
    t_mid1 = c1[2] * conc ** (-c1[3])
    expr1 = c1[4] * sp.exp(-sp.exp(-k1 * (time - t_mid1))) + c1[5]
    result1 = evaluate_expression(expr1, X_train, y_train, X_val, y_val, constants=c1)
    if result1.get("combined_score", 0) > best_score:
        best_score = result1.get("combined_score", 0)
        best_result = result1

    # Model 2: Logistic sigmoid with concentration-dependent half-time and rate
    # t_half = c0 * conc^(-c1), rate = c2 * conc^c3
    # y = c4 / (1 + exp(-rate * (t - t_half))) + c5
    c2 = constant_symbols(6)
    t_half2 = c2[0] * conc ** (-c2[1])
    rate2 = c2[2] * conc ** c2[3]
    expr2 = c2[4] / (1 + sp.exp(-rate2 * (time - t_half2))) + c2[5]
    result2 = evaluate_expression(expr2, X_train, y_train, X_val, y_val, constants=c2)
    if result2.get("combined_score", 0) > best_score:
        best_score = result2.get("combined_score", 0)
        best_result = result2

    # Model 3: Secondary nucleation logistic with linear conc dependence
    # t_half = c0 / (c1 + c2 * conc), rate = c3 + c4 * conc
    # y = c5 / (1 + exp(-rate * (t - t_half))) + c6
    c3 = constant_symbols(7)
    t_half3 = c3[0] / (c3[1] + c3[2] * conc)
    rate3 = c3[3] + c3[4] * conc
    expr3 = c3[5] / (1 + sp.exp(-rate3 * (time - t_half3))) + c3[6]
    result3 = evaluate_expression(expr3, X_train, y_train, X_val, y_val, constants=c3)
    if result3.get("combined_score", 0) > best_score:
        best_score = result3.get("combined_score", 0)
        best_result = result3

    # Model 4: Avrami nucleation-growth: y = 1 - exp(-k * t^n)
    # k = c0 * conc^c1, n = c2 (fixed exponent)
    # y = c3 * (1 - exp(-k * t^n)) + c4
    c4 = constant_symbols(5)
    k4 = c4[0] * conc ** c4[1]
    n4 = c4[2]
    expr4 = c4[3] * (1 - sp.exp(-k4 * time ** n4)) + c4[4]
    result4 = evaluate_expression(expr4, X_train, y_train, X_val, y_val, constants=c4)
    if result4.get("combined_score", 0) > best_score:
        best_score = result4.get("combined_score", 0)
        best_result = result4

    # Model 5: Gompertz with linear concentration dependence for t_mid
    # k = c0 + c1 * conc, t_mid = c2 / (c3 + c4 * conc)
    # y = c5 * exp(-exp(-k*(t - t_mid))) + c6
    c5 = constant_symbols(7)
    k5 = c5[0] + c5[1] * conc
    t_mid5 = c5[2] / (c5[3] + c5[4] * conc)
    expr5 = c5[5] * sp.exp(-sp.exp(-k5 * (time - t_mid5))) + c5[6]
    result5 = evaluate_expression(expr5, X_train, y_train, X_val, y_val, constants=c5)
    if result5.get("combined_score", 0) > best_score:
        best_score = result5.get("combined_score", 0)
        best_result = result5

    # Model 6: Hill equation with concentration-dependent t_half and exponent
    # t_half = c0 * exp(-c1 * conc) + c2, n = c3
    # y = c4 / (1 + (t_half / (t + 1e-6))^n) + c5
    c6 = constant_symbols(6)
    t_half6 = c6[0] * sp.exp(-c6[1] * conc) + c6[2]
    n6 = c6[3]
    hill_arg = (t_half6 / (time + sp.Rational(1, 1000))) ** n6
    expr6 = c6[4] / (1 + hill_arg) + c6[5]
    result6 = evaluate_expression(expr6, X_train, y_train, X_val, y_val, constants=c6)
    if result6.get("combined_score", 0) > best_score:
        best_score = result6.get("combined_score", 0)
        best_result = result6

    # Model 7: Logistic with sqrt concentration dependence (intermediate power)
    # t_half = c0 / sqrt(conc + c1), rate = c2 * sqrt(conc + c3)
    # y = c4 / (1 + exp(-rate * (t - t_half))) + c5
    c7 = constant_symbols(6)
    t_half7 = c7[0] / sp.sqrt(conc + c7[1])
    rate7 = c7[2] * sp.sqrt(conc + c7[3])
    expr7 = c7[4] / (1 + sp.exp(-rate7 * (time - t_half7))) + c7[5]
    result7 = evaluate_expression(expr7, X_train, y_train, X_val, y_val, constants=c7)
    if result7.get("combined_score", 0) > best_score:
        best_score = result7.get("combined_score", 0)
        best_result = result7

    return best_result


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
    """Load deterministic Alpha-synuclein Gaspar 2017 splits (matches evaluator)."""
    from evaluator import load_alphasyn_data

    return load_alphasyn_data()


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = _load_data()
    result = run_discovery(X_train, y_train, X_val, y_val)
    print("equation:", result.get("equation"))
    print("nmse_val:", result.get("nmse_val"))
    print("combined_score:", result.get("combined_score"))
