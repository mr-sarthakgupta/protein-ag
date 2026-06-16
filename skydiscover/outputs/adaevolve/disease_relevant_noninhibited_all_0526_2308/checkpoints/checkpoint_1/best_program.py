# EVOLVE-BLOCK-START
"""
Improved symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: try multiple biophysically-motivated equation templates and
multiple initial-value sets; return the best-fitting result.

Key improvements over baseline:
  1. Numerically stable concentration dependence via exp(c * log(x1 + eps))
     instead of x1^c (avoids NaN for near-zero or negative x1 values).
  2. Multiple candidate templates covering different kinetic regimes:
     - Standard logistic (fast nucleation)
     - Generalized logistic with asymmetry parameter (Richards curve)
     - Stretched-exponential / Avrami (heterogeneous nucleation)
  3. Multiple initial-value guesses per template to escape local minima.
  4. Returns the template with the lowest validation NMSE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)


def _safe_log_param(parameter: sp.Expr, eps_sym: sp.Expr) -> sp.Expr:
    """Return log(parameter + eps) — stable for parameter near zero."""
    return sp.log(parameter + eps_sym)


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """
    Fit multiple biophysically-motivated sigmoidal templates and return best.

    Templates tried:
      A) Logistic with concentration-scaled rate and half-time (stable form):
           rate     = exp(c1 * log(x1 + c6))
           half_time = exp(c3 * log(x1 + c6)) * c2
           y = c4 / (1 + exp(-c0 * rate * (x0 - half_time))) + c5

      B) Richards / generalised logistic — adds asymmetry exponent c6:
           y = c4 / (1 + c6*exp(-c0*(x0 - c2)))^(1/c6) + c5
           (collapses to standard logistic when c6→1)

      C) Avrami (nucleation-growth) with concentration modulation:
           y = c4 * (1 - exp(-c0 * exp(c1*log(x1+c5)) * x0^c2)) + c3

    For each template we try several initial-value sets and keep the best.
    The overall winner (lowest nmse_val) is returned.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    x = feature_symbols(2)
    time = x[0]
    param = x[1]

    best_result: dict[str, Any] | None = None
    best_nmse = float("inf")

    # ------------------------------------------------------------------
    # Template A: stable logistic with concentration-power-law modulation
    #   rate(x1)      = c0 * (x1 + c6)^c1   [via exp(c1*log(x1+c6))]
    #   half_time(x1) = c2 * (x1 + c6)^c3
    #   y = c4 / (1 + exp(-rate * (x0 - half_time))) + c5
    # 7 constants: c0..c6
    # ------------------------------------------------------------------
    cA = constant_symbols(7)
    eps_A = cA[6]
    rate_A = cA[0] * sp.exp(cA[1] * sp.log(param + eps_A))
    half_A = cA[2] * sp.exp(cA[3] * sp.log(param + eps_A))
    expr_A = cA[4] / (1 + sp.exp(-rate_A * (time - half_A))) + cA[5]

    init_sets_A = [
        [0.1,  0.5, 10.0, -0.5, 1.0,  0.0, 1.0],
        [0.5,  1.0,  5.0,  0.0, 1.0,  0.0, 0.1],
        [0.05, 0.3, 20.0, -1.0, 0.9, -0.05, 0.5],
        [1.0,  0.0, 10.0,  0.0, 1.0,  0.0, 1.0],
    ]
    for init in init_sets_A:
        try:
            res = evaluate_expression(
                expr_A, X_train, y_train, X_val, y_val,
                constants=cA, initial_values=init,
            )
            v = float(res.get("nmse_val", float("inf")))
            if np.isfinite(v) and v < best_nmse:
                best_nmse = v
                best_result = res
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Template B: Richards (generalised logistic) — asymmetric sigmoid
    #   y = c3 / (1 + exp(-c0*(x0 - c1)))^c2 + c4
    # No explicit concentration dependence beyond fitting per dataset.
    # 5 constants, simple and robust.
    # ------------------------------------------------------------------
    cB = constant_symbols(5)
    # Richards: y = c3 / (1 + exp(-c0*(x0-c1)))^c2 + c4
    inner_B = 1 + sp.exp(-cB[0] * (time - cB[1]))
    expr_B = cB[3] / inner_B ** cB[2] + cB[4]

    init_sets_B = [
        [0.1, 10.0, 1.0, 1.0, 0.0],
        [0.5,  5.0, 2.0, 1.0, 0.0],
        [0.2, 15.0, 0.5, 0.9, 0.05],
        [1.0,  8.0, 1.5, 1.0, 0.0],
    ]
    for init in init_sets_B:
        try:
            res = evaluate_expression(
                expr_B, X_train, y_train, X_val, y_val,
                constants=cB, initial_values=init,
            )
            v = float(res.get("nmse_val", float("inf")))
            if np.isfinite(v) and v < best_nmse:
                best_nmse = v
                best_result = res
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Template C: Avrami / nucleation-growth model
    #   k(x1) = c0 * (x1 + c5)^c1
    #   y = c3 * (1 - exp(-k(x1) * x0^c2)) + c4
    # 6 constants.
    # ------------------------------------------------------------------
    cC = constant_symbols(6)
    eps_C = cC[5]
    k_C = cC[0] * sp.exp(cC[1] * sp.log(param + eps_C))
    expr_C = cC[3] * (1 - sp.exp(-k_C * time ** cC[2])) + cC[4]

    init_sets_C = [
        [0.01, 1.0, 2.0, 1.0, 0.0, 1.0],
        [0.1,  0.5, 1.5, 1.0, 0.0, 0.1],
        [0.001, 2.0, 3.0, 0.9, 0.0, 0.5],
        [0.05, 0.0, 2.0, 1.0, 0.0, 1.0],
    ]
    for init in init_sets_C:
        try:
            res = evaluate_expression(
                expr_C, X_train, y_train, X_val, y_val,
                constants=cC, initial_values=init,
            )
            v = float(res.get("nmse_val", float("inf")))
            if np.isfinite(v) and v < best_nmse:
                best_nmse = v
                best_result = res
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Template D: Double-logistic (lag + growth phases)
    #   y = c4 / (1 + exp(-c0*(x0 - c1))) - c5 / (1 + exp(-c2*(x0 - c3)))
    # Captures lag phase explicitly. 6 constants.
    # ------------------------------------------------------------------
    cD = constant_symbols(6)
    expr_D = (cD[4] / (1 + sp.exp(-cD[0] * (time - cD[1])))
              - cD[5] / (1 + sp.exp(-cD[2] * (time - cD[3]))))

    init_sets_D = [
        [0.5, 10.0, 5.0, 2.0, 1.0, 0.05],
        [0.3,  8.0, 2.0, 1.0, 1.0, 0.1],
    ]
    for init in init_sets_D:
        try:
            res = evaluate_expression(
                expr_D, X_train, y_train, X_val, y_val,
                constants=cD, initial_values=init,
            )
            v = float(res.get("nmse_val", float("inf")))
            if np.isfinite(v) and v < best_nmse:
                best_nmse = v
                best_result = res
        except Exception:
            pass

    # Fallback: if everything failed, return the baseline logistic
    if best_result is None:
        cF = constant_symbols(6)
        expr_F = cF[4] / (1 + sp.exp(-cF[0] * (time - cF[2]))) + cF[5]
        try:
            best_result = evaluate_expression(
                expr_F, X_train, y_train, X_val, y_val,
                constants=cF, initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 0.0],
            )
        except Exception:
            best_result = {
                "equation_template": "fallback",
                "equation": "fallback",
                "constants": {},
                "loss": float("inf"),
                "complexity": 0.0,
                "nmse_train": float("inf"),
                "nmse_val": float("inf"),
                "combined_score": 0.0,
            }

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
