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
    Evaluate multiple kinetic equation templates and return the best fit.

    Tries five physically motivated models for amyloid aggregation and
    returns the one with the lowest validation NMSE. Each model captures
    a different mechanistic aspect of nucleation-dependent polymerization.

    Models:
    1. Hill kinetics: y = c3 * t^c2 / ((c0*x1^c1)^c2 + t^c2) + c4
       Concentration-dependent half-time, Hill exponent captures lag phase.
    2. Logistic (concentration-dependent rate + half-time):
       y = c4 / (1 + exp(-c0*x1^c1*(t - c2*x1^c3))) + c5
    3. Generalized Richards logistic (asymmetric sigmoid):
       y = c4 / (1 + exp(-c0*(t - c1*x1^c2)))^c3 + c5
    4. Avrami/JMAK nucleation-growth:
       y = c2 * (1 - exp(-c0 * x1^c1 * t^c3)) + c4
       Classic model for nucleation and growth kinetics.
    5. Finke-Watzky two-step mechanism (autocatalytic):
       y = c4 / (1 + c3*exp(-(c0*x1^c1 + c2)*t)) + c5
       Gold standard for amyloid aggregation; nucleation + autocatalytic growth.
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

    # --- Model 1: Hill kinetics with concentration-dependent half-time ---
    # y = c3 * t^c2 / ((c0*x1^c1)^c2 + t^c2) + c4
    # 5 constants: c0=t_half_scale, c1=conc_exp, c2=hill_n, c3=plateau, c4=baseline
    c1 = constant_symbols(5)
    t_half_1 = c1[0] * parameter ** c1[1]
    hill_n = c1[2]
    expr1 = c1[3] * time ** hill_n / (t_half_1 ** hill_n + time ** hill_n) + c1[4]
    _try(expr1, c1, [10.0, -0.5, 2.0, 1.0, 0.0])

    # --- Model 2: Logistic with concentration-dependent rate and half-time ---
    # y = c4 / (1 + exp(-c0*x1^c1*(t - c2*x1^c3))) + c5
    c2 = constant_symbols(6)
    rate_2 = c2[0] * parameter ** c2[1]
    half_time_2 = c2[2] * parameter ** c2[3]
    expr2 = c2[4] / (1 + sp.exp(-rate_2 * (time - half_time_2))) + c2[5]
    _try(expr2, c2, [0.1, 0.5, 10.0, -0.5, 1.0, 0.0])

    # --- Model 3: Generalized Richards logistic (asymmetric sigmoid) ---
    # y = c4 / (1 + exp(-c0*(t - c1*x1^c2)))^c3 + c5
    # c3 > 1 gives asymmetric sigmoid with longer lag phase
    c3 = constant_symbols(6)
    inner_3 = sp.exp(-c3[0] * (time - c3[1] * parameter ** c3[2]))
    expr3 = c3[4] / (1 + inner_3) ** c3[3] + c3[5]
    _try(expr3, c3, [0.2, 10.0, -0.3, 1.5, 1.0, 0.0])

    # --- Model 4: Avrami/JMAK nucleation-growth model ---
    # y = c2 * (1 - exp(-c0 * x1^c1 * t^c3)) + c4
    # c3 = Avrami exponent (dimensionality/mechanism); c0*x1^c1 = rate constant
    c4 = constant_symbols(5)
    avrami_rate = c4[0] * parameter ** c4[1]
    expr4 = c4[2] * (1 - sp.exp(-avrami_rate * time ** c4[3])) + c4[4]
    _try(expr4, c4, [0.01, 0.5, 1.0, 2.0, 0.0])

    # --- Model 5: Finke-Watzky two-step autocatalytic mechanism ---
    # Nucleation (rate k1*x1^c1) + autocatalytic growth (rate k2)
    # y = c4 / (1 + c3*exp(-(c0*x1^c1 + c2)*t)) + c5
    # This is the FW sigmoidal: A/(1 + B*exp(-C*t)) form
    c5 = constant_symbols(6)
    k1_fw = c5[0] * parameter ** c5[1]   # concentration-dependent nucleation rate
    k2_fw = c5[2]                          # autocatalytic growth rate
    fw_rate = k1_fw + k2_fw
    expr5 = c5[4] / (1 + c5[3] * sp.exp(-fw_rate * time)) + c5[5]
    _try(expr5, c5, [0.05, 0.5, 0.1, 100.0, 1.0, 0.0])

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
