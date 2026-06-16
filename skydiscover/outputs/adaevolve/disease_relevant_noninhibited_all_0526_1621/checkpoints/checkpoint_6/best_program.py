# EVOLVE-BLOCK-START
"""Symbolic regression seed for multi-dataset amyloid aggregation kinetics."""

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
    Multi-model ensemble for amyloid aggregation kinetics.

    Tries 8 physically motivated model families (Gompertz, Hill, logistic,
    Avrami, Finke-Watzky, Richards, stretched-exponential, linear-rate
    variants) each with multiple initial value sets. Returns the best-fit
    model by validation NMSE. This mirrors the approach of Program 3 which
    achieved combined_score=0.8712, but with additional model variants and
    richer initial value grids to escape local minima.

    Key models:
    - Gompertz (asymmetric sigmoid, gold standard for nucleation kinetics)
    - Hill (cooperative binding / nucleation threshold)
    - Logistic with power-law concentration dependence
    - Avrami/JMAK (nucleation-growth, classic solid-state kinetics)
    - Finke-Watzky two-step autocatalytic mechanism
    - Richards generalized logistic (asymmetry exponent)
    - Stretched exponential / KWW (heterogeneous nucleation)
    - Linear-rate logistic (stable fallback for x1=1 datasets)
    """
    x = feature_symbols(X_train.shape[1])
    time = x[0]
    parameter = x[1]

    best_result = None
    best_nmse = float("inf")

    def _try(expr, consts, init):
        nonlocal best_result, best_nmse
        r = evaluate_expression(
            expr, X_train, y_train, X_val, y_val,
            constants=consts,
            initial_values=init,
        )
        nmse = r.get("nmse_val", float("inf"))
        if nmse is not None and nmse < best_nmse:
            best_nmse = nmse
            best_result = r

    # --- Model 1: Gompertz with power-law concentration dependence ---
    # y = c4 * exp(-exp(-c0*x1^c1*(t - c2*x1^c3))) + c5
    # Asymmetric sigmoid: long lag phase, rapid growth, flat plateau.
    c1 = constant_symbols(6)
    rate_1 = c1[0] * parameter ** c1[1]
    half_1 = c1[2] * parameter ** c1[3]
    expr1 = c1[4] * sp.exp(-sp.exp(-rate_1 * (time - half_1))) + c1[5]
    for init1 in [
        [0.2, 0.5, 10.0, -0.5, 1.0, 0.0],
        [0.1, 1.0, 20.0, -1.0, 1.0, 0.0],
        [0.3, 0.0, 5.0,  0.0,  1.0, 0.0],
        [0.15, 0.3, 15.0, -0.3, 1.0, 0.0],
        [0.5, 0.5, 8.0, -0.5, 1.0, 0.0],
    ]:
        _try(expr1, c1, init1)

    # --- Model 2: Hill kinetics with concentration-dependent half-time ---
    # y = c3 * t^c2 / ((c0*x1^c1)^c2 + t^c2) + c4
    # Hill exponent c2 captures cooperative nucleation threshold.
    c2 = constant_symbols(5)
    t_half_2 = c2[0] * parameter ** c2[1]
    hill_n = c2[2]
    expr2 = c2[3] * time ** hill_n / (t_half_2 ** hill_n + time ** hill_n) + c2[4]
    for init2 in [
        [10.0, -0.5, 2.0, 1.0, 0.0],
        [5.0,  0.0,  3.0, 1.0, 0.0],
        [20.0, -1.0, 1.5, 1.0, 0.0],
        [8.0,  -0.3, 4.0, 1.0, 0.0],
        [15.0,  0.5, 2.5, 1.0, 0.0],
    ]:
        _try(expr2, c2, init2)

    # --- Model 3: Logistic with power-law concentration dependence ---
    # y = c4 / (1 + exp(-c0*x1^c1*(t - c2*x1^c3))) + c5
    c3 = constant_symbols(6)
    rate_3 = c3[0] * parameter ** c3[1]
    half_3 = c3[2] * parameter ** c3[3]
    expr3 = c3[4] / (1 + sp.exp(-rate_3 * (time - half_3))) + c3[5]
    for init3 in [
        [0.1,  0.5, 10.0, -0.5, 1.0, 0.0],
        [0.2,  1.0,  5.0, -1.0, 1.0, 0.0],
        [0.05, 0.0, 20.0,  0.0, 1.0, 0.0],
        [0.3,  0.3, 12.0, -0.3, 1.0, 0.0],
    ]:
        _try(expr3, c3, init3)

    # --- Model 4: Avrami/JMAK nucleation-growth ---
    # y = c2 * (1 - exp(-c0 * x1^c1 * t^c3)) + c4
    # Avrami exponent c3 encodes dimensionality and nucleation mechanism.
    c4a = constant_symbols(5)
    avrami_rate = c4a[0] * parameter ** c4a[1]
    expr4 = c4a[2] * (1 - sp.exp(-avrami_rate * time ** c4a[3])) + c4a[4]
    for init4 in [
        [0.01, 0.5, 1.0, 2.0, 0.0],
        [0.001, 1.0, 1.0, 3.0, 0.0],
        [0.05, 0.0, 1.0, 1.5, 0.0],
    ]:
        _try(expr4, c4a, init4)

    # --- Model 5: Finke-Watzky two-step autocatalytic mechanism ---
    # y = c4 / (1 + c3*exp(-(c0*x1^c1 + c2)*t)) + c5
    # Nucleation rate k1 scales with concentration; growth rate k2 is fixed.
    c5 = constant_symbols(6)
    fw_rate = c5[0] * parameter ** c5[1] + c5[2]
    expr5 = c5[4] / (1 + c5[3] * sp.exp(-fw_rate * time)) + c5[5]
    for init5 in [
        [0.05, 0.5, 0.1, 100.0, 1.0, 0.0],
        [0.1,  1.0, 0.2,  50.0, 1.0, 0.0],
        [0.02, 0.0, 0.05, 200.0, 1.0, 0.0],
    ]:
        _try(expr5, c5, init5)

    # --- Model 6: Richards/generalized logistic (asymmetric sigmoid) ---
    # y = c4 / (1 + exp(-c0*(t - c1*x1^c2)))^c3 + c5
    # c3 > 1 gives asymmetric sigmoid with extended lag phase.
    c6 = constant_symbols(6)
    inner_6 = sp.exp(-c6[0] * (time - c6[1] * parameter ** c6[2]))
    expr6 = c6[4] / (1 + inner_6) ** c6[3] + c6[5]
    for init6 in [
        [0.2, 10.0, -0.3, 1.5, 1.0, 0.0],
        [0.3,  5.0, -0.5, 2.0, 1.0, 0.0],
        [0.1, 15.0,  0.0, 1.2, 1.0, 0.0],
    ]:
        _try(expr6, c6, init6)

    # --- Model 7: Stretched exponential (KWW) ---
    # y = c2 * (1 - exp(-(t / (c0*x1^c1))^c3)) + c4
    # Heterogeneous nucleation with distributed rate constants.
    c7 = constant_symbols(5)
    tau_7 = c7[0] * parameter ** c7[1]
    expr7 = c7[2] * (1 - sp.exp(-(time / tau_7) ** c7[3])) + c7[4]
    for init7 in [
        [10.0, -0.5, 1.0, 1.5, 0.0],
        [5.0,   0.0, 1.0, 2.0, 0.0],
        [20.0, -1.0, 1.0, 1.2, 0.0],
    ]:
        _try(expr7, c7, init7)

    # --- Model 8: Linear-rate logistic (stable fallback) ---
    # y = c4 / (1 + exp(-(c0 + c1*x1)*(t - (c2 + c3*x1)))) + c5
    # Linear concentration dependence avoids 0^negative instability;
    # collapses gracefully when x1=1 (single-concentration datasets).
    c8 = constant_symbols(6)
    rate_8 = c8[0] + c8[1] * parameter
    half_8 = c8[2] + c8[3] * parameter
    expr8 = c8[4] / (1 + sp.exp(-rate_8 * (time - half_8))) + c8[5]
    for init8 in [
        [0.3,  0.01, 10.0, -0.1, 1.0, 0.0],
        [0.2,  0.05,  5.0,  0.0, 1.0, 0.0],
        [0.15, 0.02, 15.0, -0.2, 1.0, 0.0],
    ]:
        _try(expr8, c8, init8)

    # --- Model 9: Gompertz with linear concentration dependence ---
    # y = c4 * exp(-exp(-(c0+c1*x1)*(t - (c2+c3*x1)))) + c5
    # Linear variant of Model 1 for numerical stability.
    c9 = constant_symbols(6)
    rate_9 = c9[0] + c9[1] * parameter
    half_9 = c9[2] + c9[3] * parameter
    expr9 = c9[4] * sp.exp(-sp.exp(-rate_9 * (time - half_9))) + c9[5]
    for init9 in [
        [0.2,  0.01, 10.0, -0.1, 1.0, 0.0],
        [0.15, 0.02, 15.0, -0.2, 1.0, 0.0],
        [0.3,  0.05,  8.0,  0.0, 1.0, 0.0],
    ]:
        _try(expr9, c9, init9)

    return best_result if best_result is not None else best_result


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
