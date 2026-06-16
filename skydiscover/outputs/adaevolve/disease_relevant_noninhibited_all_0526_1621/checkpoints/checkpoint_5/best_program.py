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
    Evaluate physically motivated kinetic equation templates and return the best fit.

    Tries eight models for amyloid aggregation (nucleation-dependent polymerization)
    and returns the one with the lowest validation NMSE. Key additions over prior
    attempts: Gompertz (asymmetric sigmoid, gold standard for growth curves),
    stretched-exponential/KWW (heterogeneous nucleation), and multiple initial
    value sets for the best-performing model families (logistic, Hill, Gompertz)
    to escape local minima.

    Models:
    1. Hill kinetics (multiple inits): y = c3*t^c2/((c0*x1^c1)^c2+t^c2) + c4
    2. Logistic (conc-dependent rate+half-time, multiple inits):
       y = c4/(1+exp(-c0*x1^c1*(t-c2*x1^c3))) + c5
    3. Gompertz (asymmetric sigmoid, conc-dependent):
       y = c2*exp(-exp(-c0*(t - c1*x1^c3))) + c4
    4. Gompertz with conc-dependent rate:
       y = c2*exp(-exp(-c0*x1^c1*(t - c3))) + c4
    5. Richards/generalized logistic:
       y = c4/(1+exp(-c0*(t-c1*x1^c2)))^c3 + c5
    6. Avrami/JMAK: y = c2*(1-exp(-c0*x1^c1*t^c3)) + c4
    7. Finke-Watzky autocatalytic:
       y = c4/(1+c3*exp(-(c0*x1^c1+c2)*t)) + c5
    8. Stretched exponential (KWW):
       y = c2*(1-exp(-(t/c0)^c1)) + c3  (heterogeneous nucleation)
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

    # --- Model 1: Hill kinetics (multiple initial value sets) ---
    # y = c3 * t^c2 / ((c0*x1^c1)^c2 + t^c2) + c4
    c1 = constant_symbols(5)
    t_half_1 = c1[0] * parameter ** c1[1]
    hill_n = c1[2]
    expr1 = c1[3] * time ** hill_n / (t_half_1 ** hill_n + time ** hill_n) + c1[4]
    for init1 in [
        [10.0, -0.5, 2.0, 1.0, 0.0],
        [5.0, 0.0, 3.0, 1.0, 0.0],
        [20.0, -1.0, 1.5, 1.0, 0.0],
    ]:
        _try(expr1, c1, init1)

    # --- Model 2: Logistic with concentration-dependent rate and half-time ---
    # y = c4 / (1 + exp(-c0*x1^c1*(t - c2*x1^c3))) + c5
    c2 = constant_symbols(6)
    rate_2 = c2[0] * parameter ** c2[1]
    half_time_2 = c2[2] * parameter ** c2[3]
    expr2 = c2[4] / (1 + sp.exp(-rate_2 * (time - half_time_2))) + c2[5]
    for init2 in [
        [0.1, 0.5, 10.0, -0.5, 1.0, 0.0],
        [0.2, 1.0, 5.0, -1.0, 1.0, 0.0],
        [0.05, 0.0, 20.0, 0.0, 1.0, 0.0],
    ]:
        _try(expr2, c2, init2)

    # --- Model 3: Gompertz with concentration-dependent half-time ---
    # y = c2 * exp(-exp(-c0*(t - c1*x1^c3))) + c4
    # Gompertz is a double-exponential sigmoid; naturally asymmetric with
    # longer lag phase — a gold standard for nucleation-growth curves.
    c3 = constant_symbols(5)
    expr3 = c3[2] * sp.exp(-sp.exp(-c3[0] * (time - c3[1] * parameter ** c3[3]))) + c3[4]
    for init3 in [
        [0.2, 10.0, 1.0, -0.3, 0.0],
        [0.1, 20.0, 1.0, -0.5, 0.0],
        [0.3, 5.0, 1.0, 0.0, 0.0],
    ]:
        _try(expr3, c3, init3)

    # --- Model 4: Gompertz with concentration-dependent rate ---
    # y = c2 * exp(-exp(-c0*x1^c1*(t - c3))) + c4
    # Rate scales with concentration via power law; fixed lag offset c3.
    c4g = constant_symbols(5)
    expr4 = c4g[2] * sp.exp(-sp.exp(-c4g[0] * parameter ** c4g[1] * (time - c4g[3]))) + c4g[4]
    for init4 in [
        [0.1, 0.5, 1.0, 10.0, 0.0],
        [0.2, 1.0, 1.0, 5.0, 0.0],
        [0.05, 0.0, 1.0, 15.0, 0.0],
    ]:
        _try(expr4, c4g, init4)

    # --- Model 5: Richards/generalized logistic (asymmetric sigmoid) ---
    # y = c4 / (1 + exp(-c0*(t - c1*x1^c2)))^c3 + c5
    c5 = constant_symbols(6)
    inner_5 = sp.exp(-c5[0] * (time - c5[1] * parameter ** c5[2]))
    expr5 = c5[4] / (1 + inner_5) ** c5[3] + c5[5]
    _try(expr5, c5, [0.2, 10.0, -0.3, 1.5, 1.0, 0.0])

    # --- Model 6: Avrami/JMAK nucleation-growth model ---
    # y = c2 * (1 - exp(-c0 * x1^c1 * t^c3)) + c4
    c6 = constant_symbols(5)
    avrami_rate = c6[0] * parameter ** c6[1]
    expr6 = c6[2] * (1 - sp.exp(-avrami_rate * time ** c6[3])) + c6[4]
    for init6 in [
        [0.01, 0.5, 1.0, 2.0, 0.0],
        [0.001, 1.0, 1.0, 3.0, 0.0],
    ]:
        _try(expr6, c6, init6)

    # --- Model 7: Finke-Watzky two-step autocatalytic mechanism ---
    # y = c4 / (1 + c3*exp(-(c0*x1^c1 + c2)*t)) + c5
    c7 = constant_symbols(6)
    k1_fw = c7[0] * parameter ** c7[1]
    fw_rate = k1_fw + c7[2]
    expr7 = c7[4] / (1 + c7[3] * sp.exp(-fw_rate * time)) + c7[5]
    _try(expr7, c7, [0.05, 0.5, 0.1, 100.0, 1.0, 0.0])

    # --- Model 8: Stretched exponential (Kohlrausch-Williams-Watts / KWW) ---
    # y = c2 * (1 - exp(-(t/c0)^c1)) + c3
    # Captures heterogeneous nucleation with distributed rate constants.
    # c0 = characteristic time (concentration-scaled), c1 = stretch exponent
    c8 = constant_symbols(5)
    tau_8 = c8[0] * parameter ** c8[1]
    expr8 = c8[2] * (1 - sp.exp(-(time / tau_8) ** c8[3])) + c8[4]
    for init8 in [
        [10.0, -0.5, 1.0, 1.5, 0.0],
        [5.0, 0.0, 1.0, 2.0, 0.0],
    ]:
        _try(expr8, c8, init8)

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
