# EVOLVE-BLOCK-START
"""
Improved symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: Use data-adaptive initial value estimation combined with a focused
set of the best-performing biophysically-motivated templates. Initial guesses
are derived from data statistics (median time, time range) to dramatically
improve convergence speed and quality.

Templates:
  A) Gompertz — best for amyloid lag+growth asymmetric curves
  B) Richards generalised logistic — symmetric/asymmetric sigmoid
  C) Avrami nucleation-growth — power-law heterogeneous nucleation
  D) Logistic + Avrami hybrid — captures both nucleation and growth phases
  E) Stretched Gompertz — extra shape parameter for complex kinetics
  F) Hill cooperative — cooperative nucleation with concentration shift
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
    """Fit amyloid kinetics templates with data-adaptive initial values.

    Uses median time and time range from training data to generate smart
    initial guesses for half-time parameters, dramatically improving
    convergence. Tries 6 templates, returns lowest validation NMSE result.

    x0 = time, x1 = experimental parameter (concentration, pH, etc.)
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    x = feature_symbols(2)
    time = x[0]
    param = x[1]

    # --- Data-adaptive initial value estimation ---
    t_vals = X_train[:, 0]
    t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))
    t_range = max(t_max - t_min, 1e-6)
    t_mid = float(np.median(t_vals))
    # Estimate half-time: where y crosses 0.5
    mid_mask = np.abs(y_train - 0.5) < 0.3
    t_half_est = t_mid if not np.any(mid_mask) else float(np.median(t_vals[mid_mask]))
    # Estimate rate from 10-90% rise time
    rate_est = max(4.0 / t_range, 0.01)

    best_result: dict[str, Any] | None = None
    best_nmse = float("inf")

    def _try(expr, consts, inits, nfev=500):
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
    # Template A: Gompertz growth model (best for amyloid lag-phase curves)
    #   t_half(x1) = c1 + c2*x1
    #   y = c3 * exp(-exp(-c0*(x0 - t_half))) + c4
    # Asymmetric: fast initial growth, slow plateau approach.
    # 5 constants.
    # ------------------------------------------------------------------
    cA = constant_symbols(5)
    t_half_A = cA[1] + cA[2] * param
    expr_A = cA[3] * sp.exp(-sp.exp(-cA[0] * (time - t_half_A))) + cA[4]
    _try(expr_A, cA, [
        [rate_est,       t_half_est,  0.0,  1.0,   0.0],
        [rate_est * 2,   t_half_est,  0.0,  1.0,   0.0],
        [rate_est / 2,   t_half_est,  0.0,  1.0,   0.0],
        [rate_est,       t_half_est * 0.7, 0.0, 1.0, 0.0],
        [rate_est,       t_half_est * 1.3, 0.0, 1.0, 0.0],
        [0.2,   10.0,  0.0,  1.0,  0.0],
        [0.5,    8.0,  0.0,  1.0,  0.0],
        [0.1,   15.0,  0.0,  1.0,  0.0],
        [0.3,    5.0,  0.1,  0.9,  0.0],
        [1.0,   10.0,  0.0,  1.0,  0.0],
        [0.2,   20.0, -0.1,  1.0,  0.0],
        [0.15,   8.0,  0.0,  1.1,  0.0],
        [rate_est, t_half_est, 0.0, 1.0, -0.05],
    ])

    # ------------------------------------------------------------------
    # Template B: Richards generalised logistic (asymmetric sigmoid)
    #   t_half(x1) = c1 + c2*x1
    #   y = c4 / (1 + exp(-c0*(x0 - t_half)))^c3 + c5
    # c3=1 → standard logistic; c3 free → Richards asymmetric curve.
    # 6 constants.
    # ------------------------------------------------------------------
    cB = constant_symbols(6)
    t_half_B = cB[1] + cB[2] * param
    expr_B = cB[4] / (1 + sp.exp(-cB[0] * (time - t_half_B))) ** cB[3] + cB[5]
    _try(expr_B, cB, [
        [rate_est,       t_half_est,  0.0,  1.0,  1.0,  0.0],
        [rate_est * 2,   t_half_est,  0.0,  1.5,  1.0,  0.0],
        [rate_est,       t_half_est,  0.0,  0.5,  1.0,  0.0],
        [rate_est / 2,   t_half_est,  0.0,  2.0,  1.0,  0.0],
        [0.1,   10.0,  0.0,  1.0,  1.0,  0.0],
        [0.3,    8.0,  0.1,  1.5,  1.0,  0.0],
        [0.5,    5.0,  0.0,  0.5,  0.9,  0.05],
        [0.1,   20.0,  0.0,  2.0,  1.0,  0.0],
        [1.0,   10.0,  0.0,  1.0,  1.0,  0.0],
        [0.2,   15.0, -0.1,  1.0,  1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template C: Avrami nucleation-growth
    #   k(x1) = c0 * (1 + c1*x1)
    #   y = c3 * (1 - exp(-k * x0^c2)) + c4
    # c2 free → generalised nucleation order (c2=2 classical Avrami).
    # 5 constants.
    # ------------------------------------------------------------------
    cC = constant_symbols(5)
    k_C = cC[0] * (1 + cC[1] * param)
    expr_C = cC[3] * (1 - sp.exp(-k_C * time ** cC[2])) + cC[4]
    # Estimate Avrami k from data: k ~ 1/(t_half^n) for n~2
    k_est = max(1.0 / (t_half_est ** 2 + 1e-6), 1e-5)
    _try(expr_C, cC, [
        [k_est,   0.0,  2.0,  1.0,  0.0],
        [k_est,   0.0,  1.5,  1.0,  0.0],
        [k_est,   0.0,  2.5,  1.0,  0.0],
        [k_est,   0.0,  3.0,  1.0,  0.0],
        [k_est,   0.1,  2.0,  1.0,  0.0],
        [0.01,    0.0,  2.0,  1.0,  0.0],
        [0.1,     0.0,  1.5,  1.0,  0.0],
        [0.001,   0.0,  2.5,  0.9,  0.0],
        [0.005,   0.0,  1.8,  1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template D: Stretched Gompertz with power-law concentration scaling
    #   t_half(x1) = c1 * exp(c2 * log(x1 + c5))
    #   y = c3 * exp(-exp(-c0*(x0 - t_half))) + c4
    # Handles power-law concentration dependence (multi-conc datasets).
    # 6 constants.
    # ------------------------------------------------------------------
    cD = constant_symbols(6)
    eps_D = cD[5]
    t_half_D = cD[1] * sp.exp(cD[2] * sp.log(param + eps_D))
    expr_D = cD[3] * sp.exp(-sp.exp(-cD[0] * (time - t_half_D))) + cD[4]
    _try(expr_D, cD, [
        [rate_est,  t_half_est,  0.0,  1.0,  0.0,  1.0],
        [0.2,  10.0,   0.0,  1.0,  0.0,  1.0],
        [0.5,   5.0,  -0.5,  1.0,  0.0,  0.1],
        [0.1,  20.0,  -1.0,  1.0,  0.0,  0.5],
        [0.3,  10.0,   0.5,  0.9,  0.0,  1.0],
        [1.0,   8.0,   0.0,  1.0,  0.0,  1.0],
        [rate_est, t_half_est, -0.5, 1.0, 0.0, 0.5],
    ])

    # ------------------------------------------------------------------
    # Template E: Logistic with power-law concentration scaling (log-stable)
    #   rate(x1)      = c0 * exp(c1 * log(x1 + c6))
    #   half_time(x1) = c2 * exp(c3 * log(x1 + c6))
    #   y = c4 / (1 + exp(-rate*(x0 - half_time))) + c5
    # 7 constants. Handles strong concentration dependence.
    # ------------------------------------------------------------------
    cE = constant_symbols(7)
    eps_E = cE[6]
    rate_E = cE[0] * sp.exp(cE[1] * sp.log(param + eps_E))
    half_E = cE[2] * sp.exp(cE[3] * sp.log(param + eps_E))
    expr_E = cE[4] / (1 + sp.exp(-rate_E * (time - half_E))) + cE[5]
    _try(expr_E, cE, [
        [rate_est,  0.5,  t_half_est, -0.5,  1.0,  0.0,  1.0],
        [0.1,  0.5,  10.0, -0.5,  1.0,  0.0,  1.0],
        [0.5,  1.0,   5.0,  0.0,  1.0,  0.0,  0.1],
        [0.05, 0.3,  20.0, -1.0,  0.9, -0.05, 0.5],
        [1.0,  0.0,  10.0,  0.0,  1.0,  0.0,  1.0],
        [0.2,  0.5,  15.0, -0.3,  1.0,  0.0,  0.5],
    ])

    # ------------------------------------------------------------------
    # Template F: Hill cooperative nucleation with concentration shift
    #   t50(x1) = c1 + c2*x1
    #   y = c3 * x0^c0 / (t50^c0 + x0^c0) + c4
    # Hill exponent c0 captures nucleation cooperativity.
    # 5 constants.
    # ------------------------------------------------------------------
    cF = constant_symbols(5)
    t50_F = cF[1] + cF[2] * param
    expr_F = cF[3] * time ** cF[0] / (t50_F ** cF[0] + time ** cF[0]) + cF[4]
    _try(expr_F, cF, [
        [2.0,  t_half_est,  0.0,  1.0,  0.0],
        [3.0,  t_half_est,  0.0,  1.0,  0.0],
        [4.0,  t_half_est,  0.0,  1.0,  0.0],
        [2.0,   10.0,  0.0,  1.0,  0.0],
        [3.0,    8.0,  0.0,  1.0,  0.0],
        [1.5,   15.0,  0.0,  1.0,  0.0],
        [4.0,    5.0,  0.0,  0.9,  0.0],
        [5.0,   10.0,  0.0,  1.0,  0.0],
        [2.0,   20.0, -0.1,  1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template G: Finke-Watzky two-step (nucleation + autocatalytic growth)
    #   y = c3 / (1 + c4*exp(-c0*(x0 - c1*(1+c2*x1)))) + c5
    # Standard biophysical model for amyloid aggregation.
    # 6 constants.
    # ------------------------------------------------------------------
    cG = constant_symbols(6)
    t_half_G = cG[1] * (1 + cG[2] * param)
    expr_G = cG[3] / (1 + cG[4] * sp.exp(-cG[0] * (time - t_half_G))) + cG[5]
    _try(expr_G, cG, [
        [rate_est,  t_half_est,  0.0,  1.0,  1.0,  0.0],
        [0.1,  10.0,  0.0,  1.0,  1.0,  0.0],
        [0.5,   8.0,  0.0,  1.0,  1.0,  0.0],
        [0.2,  12.0,  0.1,  0.9,  2.0,  0.0],
        [0.1,  15.0,  0.0,  1.0,  0.5,  0.0],
        [1.0,   5.0,  0.0,  1.0,  1.0,  0.0],
        [0.3,  10.0,  0.0,  1.0,  3.0,  0.0],
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
                constants=cFB, initial_values=[rate_est, t_half_est, 1.0, 1.0],
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
