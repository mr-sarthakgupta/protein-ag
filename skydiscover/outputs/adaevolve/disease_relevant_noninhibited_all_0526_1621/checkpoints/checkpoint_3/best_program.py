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

    Tries three physically motivated models for amyloid aggregation:
    1. Hill/sigmoidal with concentration-dependent half-time and Hill exponent:
         y = c4 * x0^c2 / ((c0*x1^c1)^c2 + x0^c2) + c5
       Naturally captures lag phase (c2>1), avoids exp overflow.
    2. Logistic with concentration-dependent rate and half-time (original):
         y = c4 / (1 + exp(-c0*x1^c1*(x0 - c2*x1^c3))) + c5
       With improved initial values.
    3. Generalized logistic (Richards curve) with lag shape parameter:
         y = c4 / (1 + exp(-c0*(x0 - c1*x1^c2)))^c3 + c5
       Asymmetric sigmoid better matching nucleation kinetics.

    Returns the result with the lowest nmse_val across all three candidates.
    """
    x = feature_symbols(X_train.shape[1])
    time = x[0]
    parameter = x[1]

    best_result = None
    best_nmse = float("inf")

    # --- Model 1: Hill kinetics with concentration-dependent half-time ---
    # y = c4 * t^c2 / ((c0*x1^c1)^c2 + t^c2) + c5
    # c0*x1^c1 = concentration-dependent half-time t_half
    # c2 = Hill exponent (lag phase when >1)
    c1 = constant_symbols(6)
    t_half_1 = c1[0] * parameter ** c1[1]
    hill_n = c1[2]
    plateau_1 = c1[3]
    baseline_1 = c1[4]
    # Use abs to keep t_half and hill_n positive for stability
    expr1 = plateau_1 * time ** hill_n / (t_half_1 ** hill_n + time ** hill_n) + baseline_1
    r1 = evaluate_expression(
        expr1, X_train, y_train, X_val, y_val,
        constants=c1,
        initial_values=[10.0, -0.5, 2.0, 1.0, 0.0, 0.0],
    )
    nmse1 = r1.get("nmse_val", float("inf"))
    if nmse1 is not None and nmse1 < best_nmse:
        best_nmse = nmse1
        best_result = r1

    # --- Model 2: Logistic with concentration-dependent rate and half-time ---
    # y = c4 / (1 + exp(-c0*x1^c1*(x0 - c2*x1^c3))) + c5
    c2 = constant_symbols(6)
    rate_2 = c2[0] * parameter ** c2[1]
    half_time_2 = c2[2] * parameter ** c2[3]
    plateau_2 = c2[4]
    baseline_2 = c2[5]
    expr2 = plateau_2 / (1 + sp.exp(-rate_2 * (time - half_time_2))) + baseline_2
    r2 = evaluate_expression(
        expr2, X_train, y_train, X_val, y_val,
        constants=c2,
        initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 0.0],
    )
    nmse2 = r2.get("nmse_val", float("inf"))
    if nmse2 is not None and nmse2 < best_nmse:
        best_nmse = nmse2
        best_result = r2

    # --- Model 3: Generalized logistic (Richards) with lag shape ---
    # y = c4 / (1 + exp(-c0*(x0 - c1*x1^c2)))^c3 + c5
    # c3 > 1 gives asymmetric sigmoid with longer lag phase
    c3 = constant_symbols(6)
    rate_3 = c3[0]
    half_time_3 = c3[1] * parameter ** c3[2]
    shape_3 = c3[3]
    plateau_3 = c3[4]
    baseline_3 = c3[5]
    inner_3 = sp.exp(-rate_3 * (time - half_time_3))
    expr3 = plateau_3 / (1 + inner_3) ** shape_3 + baseline_3
    r3 = evaluate_expression(
        expr3, X_train, y_train, X_val, y_val,
        constants=c3,
        initial_values=[0.2, 10.0, -0.3, 1.5, 1.0, 0.0],
    )
    nmse3 = r3.get("nmse_val", float("inf"))
    if nmse3 is not None and nmse3 < best_nmse:
        best_nmse = nmse3
        best_result = r3

    return best_result if best_result is not None else r2


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
