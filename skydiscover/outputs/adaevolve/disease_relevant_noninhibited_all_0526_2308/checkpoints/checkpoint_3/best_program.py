# EVOLVE-BLOCK-START
"""
Improved symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: Evaluate a diverse set of biophysically-motivated equation templates
with multiple initial-value sets per template, returning the best fit.

Templates cover:
  A) Richards generalised logistic (asymmetric sigmoid) — linear conc shift
  B) Gompertz growth model — known to fit amyloid lag+growth curves well
  C) Avrami nucleation-growth — power-law growth, heterogeneous nucleation
  D) Hill sigmoidal — cooperative binding / nucleation
  E) Finke-Watzky inspired — nucleation + autocatalytic growth
  F) Simple logistic with additive concentration shift — robust fallback
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


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """
    Fit multiple biophysically-motivated amyloid kinetics templates.

    Tries Richards logistic, Gompertz, Avrami, Hill, Finke-Watzky, and
    simple logistic templates, each with multiple initial-value sets.
    Returns the result with the lowest validation NMSE.

    x0 = time coordinate, x1 = experimental parameter (concentration, pH, etc.)
    All templates use a concentration-dependent half-time or rate via a linear
    or log-stable parameterisation to handle both multi- and single-conc datasets.
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

    def _try(expr, consts, inits, nfev=400):
        nonlocal best_result, best_nmse
        for init in inits:
            try:
                res = evaluate_expression(
                    expr, X_train, y_train, X_val, y_val,
                    constants=consts, initial_values=init, max_nfev=nfev,
                )
                v = float(res.get("nmse_val", float("inf")))
                if np.isfinite(v) and v < best_nmse:
                    best_nmse = v
                    best_result = res
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Template A: Richards generalised logistic (asymmetric sigmoid)
    #   t_half(x1) = c1 + c2*x1   (linear concentration shift)
    #   y = c4 / (1 + exp(-c0*(x0 - t_half)))^c3 + c5
    # c3=1 → standard logistic; c3 free → asymmetric (Richards curve)
    # 6 constants. Robust across protein systems.
    # ------------------------------------------------------------------
    cA = constant_symbols(6)
    t_half_A = cA[1] + cA[2] * param
    expr_A = cA[4] / (1 + sp.exp(-cA[0] * (time - t_half_A))) ** cA[3] + cA[5]
    _try(expr_A, cA, [
        [0.1,  10.0,  0.0, 1.0, 1.0,  0.0],
        [0.3,   8.0,  0.1, 1.5, 1.0,  0.0],
        [0.5,   5.0,  0.0, 0.5, 0.9,  0.05],
        [0.1,  20.0,  0.0, 2.0, 1.0,  0.0],
        [1.0,  10.0,  0.0, 1.0, 1.0,  0.0],
        [0.2,  15.0, -0.1, 1.0, 1.0,  0.0],
        [0.05, 30.0,  0.0, 1.0, 1.0,  0.0],
        [0.5,   3.0,  0.0, 0.8, 1.0, -0.05],
    ])

    # ------------------------------------------------------------------
    # Template B: Gompertz growth model
    #   y = c3 * exp(-exp(-c0*(x0 - c1 - c2*x1))) + c4
    # Gompertz is asymmetric: slower approach to plateau than logistic.
    # Well-suited for amyloid lag-phase + rapid growth + plateau.
    # 5 constants.
    # ------------------------------------------------------------------
    cB = constant_symbols(5)
    t_half_B = cB[1] + cB[2] * param
    expr_B = cB[3] * sp.exp(-sp.exp(-cB[0] * (time - t_half_B))) + cB[4]
    _try(expr_B, cB, [
        [0.2,  10.0,  0.0, 1.0,  0.0],
        [0.5,   8.0,  0.0, 1.0,  0.0],
        [0.1,  15.0,  0.0, 1.0,  0.0],
        [0.3,   5.0,  0.1, 0.9,  0.0],
        [1.0,  10.0,  0.0, 1.0,  0.0],
        [0.2,  20.0, -0.1, 1.0,  0.0],
        [0.4,  12.0,  0.0, 1.0, -0.05],
        [0.15,  8.0,  0.0, 1.1,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template C: Avrami nucleation-growth
    #   k(x1) = c0 * (1 + c1*x1)
    #   y = c3 * (1 - exp(-k * x0^c2)) + c4
    # c2=2 → classical Avrami; c2 free → generalised nucleation order.
    # 5 constants.
    # ------------------------------------------------------------------
    cC = constant_symbols(5)
    k_C = cC[0] * (1 + cC[1] * param)
    expr_C = cC[3] * (1 - sp.exp(-k_C * time ** cC[2])) + cC[4]
    _try(expr_C, cC, [
        [0.01,  0.0, 2.0, 1.0,  0.0],
        [0.1,   0.0, 1.5, 1.0,  0.0],
        [0.001, 0.0, 2.5, 0.9,  0.0],
        [0.05,  0.0, 2.0, 1.0,  0.0],
        [0.01,  0.1, 2.0, 1.0,  0.0],
        [0.1,   0.0, 3.0, 1.0,  0.0],
        [0.005, 0.0, 1.8, 1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template D: Hill sigmoidal (cooperative nucleation)
    #   t50(x1) = c1 + c2*x1
    #   y = c3 * x0^c0 / (t50^c0 + x0^c0) + c4
    # Hill exponent c0 controls cooperativity (c0=1 → Michaelis-Menten).
    # 5 constants.
    # ------------------------------------------------------------------
    cD = constant_symbols(5)
    t50_D = cD[1] + cD[2] * param
    expr_D = cD[3] * time ** cD[0] / (t50_D ** cD[0] + time ** cD[0]) + cD[4]
    _try(expr_D, cD, [
        [2.0,  10.0,  0.0, 1.0, 0.0],
        [3.0,   8.0,  0.0, 1.0, 0.0],
        [1.5,  15.0,  0.0, 1.0, 0.0],
        [4.0,   5.0,  0.0, 0.9, 0.0],
        [2.0,  12.0,  0.1, 1.0, 0.0],
        [5.0,  10.0,  0.0, 1.0, 0.0],
        [2.0,  20.0, -0.1, 1.0, 0.0],
    ])

    # ------------------------------------------------------------------
    # Template E: Finke-Watzky inspired (nucleation + autocatalytic growth)
    #   FW: y = A / (1 + B*exp(-C*t))  where A,B,C depend on k1,k2
    #   Here: t_half(x1) = c1*(1 + c2*x1), ratio=c4 free
    #   y = c3 / (1 + c4*exp(-c0*(x0 - c1*(1+c2*x1)))) + c5
    # 6 constants.
    # ------------------------------------------------------------------
    cE = constant_symbols(6)
    t_half_E = cE[1] * (1 + cE[2] * param)
    expr_E = cE[3] / (1 + cE[4] * sp.exp(-cE[0] * (time - t_half_E))) + cE[5]
    _try(expr_E, cE, [
        [0.1,  10.0,  0.0, 1.0, 1.0,  0.0],
        [0.5,   8.0,  0.0, 1.0, 1.0,  0.0],
        [0.2,  12.0,  0.1, 0.9, 2.0,  0.0],
        [0.1,  15.0,  0.0, 1.0, 0.5,  0.0],
        [1.0,   5.0,  0.0, 1.0, 1.0,  0.0],
        [0.3,  10.0,  0.0, 1.0, 3.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template F: Logistic with power-law concentration scaling (log-stable)
    #   rate(x1)      = c0 * exp(c1 * log(x1 + c6))
    #   half_time(x1) = c2 * exp(c3 * log(x1 + c6))
    #   y = c4 / (1 + exp(-rate*(x0 - half_time))) + c5
    # 7 constants. Handles strong concentration dependence.
    # ------------------------------------------------------------------
    cF = constant_symbols(7)
    eps_F = cF[6]
    rate_F = cF[0] * sp.exp(cF[1] * sp.log(param + eps_F))
    half_F = cF[2] * sp.exp(cF[3] * sp.log(param + eps_F))
    expr_F = cF[4] / (1 + sp.exp(-rate_F * (time - half_F))) + cF[5]
    _try(expr_F, cF, [
        [0.1,  0.5, 10.0, -0.5, 1.0,  0.0, 1.0],
        [0.5,  1.0,  5.0,  0.0, 1.0,  0.0, 0.1],
        [0.05, 0.3, 20.0, -1.0, 0.9, -0.05, 0.5],
        [1.0,  0.0, 10.0,  0.0, 1.0,  0.0, 1.0],
        [0.2,  0.5, 15.0, -0.3, 1.0,  0.0, 0.5],
    ])

    # ------------------------------------------------------------------
    # Template G: Gompertz with power-law concentration scaling (log-stable)
    #   t_half(x1) = c1 * exp(c2 * log(x1 + c5))
    #   y = c3 * exp(-exp(-c0*(x0 - t_half))) + c4
    # 6 constants.
    # ------------------------------------------------------------------
    cG = constant_symbols(6)
    eps_G = cG[5]
    t_half_G = cG[1] * sp.exp(cG[2] * sp.log(param + eps_G))
    expr_G = cG[3] * sp.exp(-sp.exp(-cG[0] * (time - t_half_G))) + cG[4]
    _try(expr_G, cG, [
        [0.2, 10.0,  0.0, 1.0,  0.0, 1.0],
        [0.5,  5.0, -0.5, 1.0,  0.0, 0.1],
        [0.1, 20.0, -1.0, 1.0,  0.0, 0.5],
        [0.3, 10.0,  0.5, 0.9,  0.0, 1.0],
        [1.0,  8.0,  0.0, 1.0,  0.0, 1.0],
    ])

    # ------------------------------------------------------------------
    # Template H: Double-logistic (captures two-phase or biphasic kinetics)
    #   y = c4/(1+exp(-c0*(x0-c1))) - c5/(1+exp(-c2*(x0-c3)))
    # 6 constants. Handles lag + growth or biphasic aggregation.
    # ------------------------------------------------------------------
    cH = constant_symbols(6)
    expr_H = (cH[4] / (1 + sp.exp(-cH[0] * (time - cH[1])))
              - cH[5] / (1 + sp.exp(-cH[2] * (time - cH[3]))))
    _try(expr_H, cH, [
        [0.5, 10.0, 5.0,  2.0, 1.0, 0.05],
        [0.3,  8.0, 2.0,  1.0, 1.0, 0.1],
        [1.0, 12.0, 3.0,  3.0, 1.1, 0.1],
        [0.4, 15.0, 1.0,  5.0, 1.0, 0.05],
    ])

    # ------------------------------------------------------------------
    # Fallback: simple logistic, no concentration dependence
    # ------------------------------------------------------------------
    if best_result is None:
        cFB = constant_symbols(4)
        expr_FB = cFB[3] / (1 + sp.exp(-cFB[0] * (time - cFB[1]))) ** cFB[2]
        try:
            best_result = evaluate_expression(
                expr_FB, X_train, y_train, X_val, y_val,
                constants=cFB, initial_values=[0.1, 10.0, 1.0, 1.0],
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
