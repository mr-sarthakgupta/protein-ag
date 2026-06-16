# EVOLVE-BLOCK-START
"""
Improved symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: Data-adaptive initial value estimation + diverse biophysically-motivated
templates. Initial guesses derived from data statistics (t_half, rate, plateau)
dramatically improve convergence. Templates cover:
  A) Gompertz — best for amyloid lag+growth asymmetric curves (linear conc)
  B) Richards generalised logistic — asymmetric sigmoid (linear conc)
  C) Avrami nucleation-growth — power-law heterogeneous nucleation
  D) Finke-Watzky two-step — nucleation + autocatalytic growth
  E) Gompertz with power-law concentration scaling (log-stable)
  F) Logistic with power-law concentration scaling (log-stable)
  G) Hill cooperative — cooperative nucleation with concentration shift
  H) Stretched exponential (KWW) — complex relaxation / heterogeneous kinetics
  I) Log-logistic (Fisk) — dose-response / asymmetric sigmoid
  J) Boltzmann sigmoid — classic biophysics two-state model
  K) Gompertz with conc-dependent rate AND half-time (linear)
  L) Richards with conc-dependent rate AND half-time (linear)
  M) Softplus-lag Gompertz — smooth lag phase via log(1+exp(rate*(t-lag)))
  N) Logistic with conc-dependent rate AND half-time (power-law, log-stable)
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

    Derives t_half_est and rate_est from training data statistics to seed
    all templates with physically meaningful starting points. Tries 14
    biophysically-motivated templates (Gompertz, Richards, Avrami, FW,
    power-law variants, Hill, KWW, log-logistic, Boltzmann, and new
    templates with concentration-dependent rate AND half-time) and returns
    the one with lowest validation NMSE.

    Key additions over previous version:
    - Templates K/L: concentration modulates BOTH rate and half-time linearly
    - Template M: softplus-based smooth lag phase (log(1+exp(k*(t-lag))))
    - Template N: full power-law conc scaling on both rate and half-time
    - Better initial value seeding using concentration spread statistics

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

    # Robust half-time estimate: time where y is closest to 0.5
    mid_mask = np.abs(y_train - 0.5) < 0.3
    if np.any(mid_mask):
        t_half_est = float(np.median(t_vals[mid_mask]))
    else:
        # Fallback: use time at which cumulative fraction of y crosses 0.5
        sort_idx = np.argsort(t_vals)
        t_sorted = t_vals[sort_idx]
        y_sorted = y_train[sort_idx]
        cross = np.where(y_sorted >= 0.5)[0]
        t_half_est = float(t_sorted[cross[0]]) if len(cross) > 0 else t_mid

    # Rate estimate from 10-90% rise time
    rate_est = max(4.0 / t_range, 0.01)

    # Plateau estimate (95th percentile of y_train)
    plateau_est = float(np.percentile(y_train, 95))
    plateau_est = max(plateau_est, 0.5)

    # Avrami k estimate: k ~ 1/(t_half^n) for n~2
    k_est = max(1.0 / (t_half_est ** 2 + 1e-6), 1e-5)

    # Concentration spread: std of x1 values (used to seed conc-scaling inits)
    p_vals = X_train[:, 1]
    p_mean = float(np.mean(p_vals))
    p_std = float(np.std(p_vals)) + 1e-6
    # Estimate conc slope on half-time: if p varies, half-time may shift
    # Use simple linear regression t_half ~ a + b*p as seed
    if p_std > 1e-3 * abs(p_mean) + 1e-6:
        # Meaningful concentration variation: estimate slope
        try:
            A_lsq = np.column_stack([np.ones_like(p_vals), p_vals])
            lstsq_res = np.linalg.lstsq(A_lsq, np.full_like(p_vals, t_half_est), rcond=None)
            conc_slope_est = float(lstsq_res[0][1])
        except Exception:
            conc_slope_est = 0.0
    else:
        conc_slope_est = 0.0

    # Estimate lag time: 5th percentile of times where y > 0.05
    lag_mask = y_train > 0.05
    t_lag_est = float(np.percentile(t_vals[lag_mask], 5)) if np.any(lag_mask) else max(t_half_est * 0.2, 1e-3)
    t_lag_est = max(t_lag_est, 0.0)

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
        [rate_est,       t_half_est,        0.0,  plateau_est,   0.0],
        [rate_est * 2,   t_half_est,        0.0,  plateau_est,   0.0],
        [rate_est / 2,   t_half_est,        0.0,  plateau_est,   0.0],
        [rate_est,       t_half_est * 0.7,  0.0,  plateau_est,   0.0],
        [rate_est,       t_half_est * 1.3,  0.0,  plateau_est,   0.0],
        [rate_est,       t_half_est,        conc_slope_est,  plateau_est,   0.0],
        [0.2,   10.0,  0.0,  1.0,  0.0],
        [0.5,    8.0,  0.0,  1.0,  0.0],
        [0.1,   15.0,  0.0,  1.0,  0.0],
        [0.3,    5.0,  0.1,  0.9,  0.0],
        [1.0,   10.0,  0.0,  1.0,  0.0],
        [0.2,   20.0, -0.1,  1.0,  0.0],
        [0.15,   8.0,  0.0,  1.1,  0.0],
        [rate_est, t_half_est, 0.0, plateau_est, -0.05],
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
        [rate_est,       t_half_est,  0.0,  1.0,  plateau_est,  0.0],
        [rate_est * 2,   t_half_est,  0.0,  1.5,  plateau_est,  0.0],
        [rate_est,       t_half_est,  0.0,  0.5,  plateau_est,  0.0],
        [rate_est / 2,   t_half_est,  0.0,  2.0,  plateau_est,  0.0],
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
    _try(expr_C, cC, [
        [k_est,   0.0,  2.0,  plateau_est,  0.0],
        [k_est,   0.0,  1.5,  plateau_est,  0.0],
        [k_est,   0.0,  2.5,  plateau_est,  0.0],
        [k_est,   0.0,  3.0,  plateau_est,  0.0],
        [k_est,   0.1,  2.0,  plateau_est,  0.0],
        [0.01,    0.0,  2.0,  1.0,  0.0],
        [0.1,     0.0,  1.5,  1.0,  0.0],
        [0.001,   0.0,  2.5,  0.9,  0.0],
        [0.005,   0.0,  1.8,  1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template D: Finke-Watzky two-step (nucleation + autocatalytic growth)
    #   y = c3 / (1 + c4*exp(-c0*(x0 - c1*(1+c2*x1)))) + c5
    # Standard biophysical model for amyloid aggregation.
    # 6 constants.
    # ------------------------------------------------------------------
    cD = constant_symbols(6)
    t_half_D = cD[1] * (1 + cD[2] * param)
    expr_D = cD[3] / (1 + cD[4] * sp.exp(-cD[0] * (time - t_half_D))) + cD[5]
    _try(expr_D, cD, [
        [rate_est,  t_half_est,  0.0,  plateau_est,  1.0,  0.0],
        [0.1,  10.0,  0.0,  1.0,  1.0,  0.0],
        [0.5,   8.0,  0.0,  1.0,  1.0,  0.0],
        [0.2,  12.0,  0.1,  0.9,  2.0,  0.0],
        [0.1,  15.0,  0.0,  1.0,  0.5,  0.0],
        [1.0,   5.0,  0.0,  1.0,  1.0,  0.0],
        [0.3,  10.0,  0.0,  1.0,  3.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template E: Gompertz with power-law concentration scaling (log-stable)
    #   t_half(x1) = c1 * exp(c2 * log(x1 + c5))
    #   y = c3 * exp(-exp(-c0*(x0 - t_half))) + c4
    # Handles power-law concentration dependence (multi-conc datasets).
    # 6 constants.
    # ------------------------------------------------------------------
    cE = constant_symbols(6)
    eps_E = cE[5]
    t_half_E = cE[1] * sp.exp(cE[2] * sp.log(param + eps_E))
    expr_E = cE[3] * sp.exp(-sp.exp(-cE[0] * (time - t_half_E))) + cE[4]
    _try(expr_E, cE, [
        [rate_est,  t_half_est,  0.0,  plateau_est,  0.0,  1.0],
        [0.2,  10.0,   0.0,  1.0,  0.0,  1.0],
        [0.5,   5.0,  -0.5,  1.0,  0.0,  0.1],
        [0.1,  20.0,  -1.0,  1.0,  0.0,  0.5],
        [0.3,  10.0,   0.5,  0.9,  0.0,  1.0],
        [1.0,   8.0,   0.0,  1.0,  0.0,  1.0],
        [rate_est, t_half_est, -0.5, plateau_est, 0.0, 0.5],
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
        [rate_est,  0.5,  t_half_est, -0.5,  plateau_est,  0.0,  1.0],
        [0.1,  0.5,  10.0, -0.5,  1.0,  0.0,  1.0],
        [0.5,  1.0,   5.0,  0.0,  1.0,  0.0,  0.1],
        [0.05, 0.3,  20.0, -1.0,  0.9, -0.05, 0.5],
        [1.0,  0.0,  10.0,  0.0,  1.0,  0.0,  1.0],
        [0.2,  0.5,  15.0, -0.3,  1.0,  0.0,  0.5],
    ])

    # ------------------------------------------------------------------
    # Template G: Hill cooperative nucleation with concentration shift
    #   t50(x1) = c1 + c2*x1
    #   y = c3 * x0^c0 / (t50^c0 + x0^c0) + c4
    # Hill exponent c0 captures nucleation cooperativity.
    # 5 constants.
    # ------------------------------------------------------------------
    cG = constant_symbols(5)
    t50_G = cG[1] + cG[2] * param
    expr_G = cG[3] * time ** cG[0] / (t50_G ** cG[0] + time ** cG[0]) + cG[4]
    _try(expr_G, cG, [
        [2.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [3.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [4.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [2.0,   10.0,  0.0,  1.0,  0.0],
        [3.0,    8.0,  0.0,  1.0,  0.0],
        [1.5,   15.0,  0.0,  1.0,  0.0],
        [4.0,    5.0,  0.0,  0.9,  0.0],
        [5.0,   10.0,  0.0,  1.0,  0.0],
        [2.0,   20.0, -0.1,  1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template H: Stretched exponential / KWW (Kohlrausch-Williams-Watts)
    #   Models heterogeneous nucleation and complex relaxation kinetics.
    #   k(x1) = c0 * (1 + c1*x1)
    #   y = c3 * (1 - exp(-(k * x0)^c2)) + c4
    # c2 < 1 → stretched (heterogeneous); c2 = 1 → simple exponential.
    # 5 constants.
    # ------------------------------------------------------------------
    cH = constant_symbols(5)
    k_H = cH[0] * (1 + cH[1] * param)
    expr_H = cH[3] * (1 - sp.exp(-(k_H * time) ** cH[2])) + cH[4]
    _try(expr_H, cH, [
        [rate_est,  0.0,  0.7,  plateau_est,  0.0],
        [rate_est,  0.0,  1.0,  plateau_est,  0.0],
        [rate_est,  0.0,  1.5,  plateau_est,  0.0],
        [0.1,   0.0,  0.7,  1.0,  0.0],
        [0.05,  0.0,  0.5,  1.0,  0.0],
        [0.2,   0.0,  1.2,  1.0,  0.0],
        [0.01,  0.0,  0.8,  1.0,  0.0],
        [0.1,   0.1,  0.7,  1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template I: Log-logistic (Fisk distribution) sigmoidal
    #   Models dose-response / asymmetric sigmoid with heavy tail.
    #   t50(x1) = c1 + c2*x1
    #   y = c3 / (1 + (t50 / (x0 + 1e-10))^c0) + c4
    # c0 controls steepness; log-logistic is heavier-tailed than logistic.
    # 5 constants.
    # ------------------------------------------------------------------
    cI = constant_symbols(5)
    t50_I = cI[1] + cI[2] * param
    expr_I = cI[3] / (1 + (t50_I / (time + sp.Integer(1) / sp.Integer(10000))) ** cI[0]) + cI[4]
    _try(expr_I, cI, [
        [2.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [3.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [4.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [2.0,   10.0,  0.0,  1.0,  0.0],
        [3.0,    8.0,  0.0,  1.0,  0.0],
        [1.5,   15.0,  0.0,  1.0,  0.0],
        [4.0,    5.0,  0.0,  0.9,  0.0],
        [5.0,   10.0,  0.0,  1.0,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template J: Boltzmann sigmoid (two-state thermodynamic model)
    #   Classic biophysics: y = (A1 - A2)/(1 + exp((x0-x0_half)/dx)) + A2
    #   With concentration dependence on x0_half:
    #   x0_half(x1) = c1 + c2*x1
    #   y = c3 / (1 + exp((x0 - x0_half) / c0)) + c4
    # Note: c0 here is the width (inverse rate), not rate.
    # 5 constants.
    # ------------------------------------------------------------------
    cJ = constant_symbols(5)
    x0_half_J = cJ[1] + cJ[2] * param
    width_J = cJ[0]
    expr_J = cJ[3] / (1 + sp.exp((time - x0_half_J) / width_J)) + cJ[4]
    _try(expr_J, cJ, [
        [1.0 / max(rate_est, 0.01),  t_half_est,  0.0,  plateau_est,  0.0],
        [5.0,   10.0,  0.0,  1.0,  0.0],
        [3.0,    8.0,  0.0,  1.0,  0.0],
        [8.0,   15.0,  0.0,  1.0,  0.0],
        [2.0,   10.0,  0.1,  1.0,  0.0],
        [10.0,  20.0,  0.0,  1.0,  0.0],
        [4.0,    5.0,  0.0,  0.9,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template K: Gompertz with conc-dependent rate AND half-time (linear)
    #   rate(x1)   = c0 + c1*x1   (concentration shifts growth rate)
    #   t_half(x1) = c2 + c3*x1   (concentration shifts half-time)
    #   y = c4 * exp(-exp(-rate * (x0 - t_half))) + c5
    # Key: BOTH kinetic parameters depend on concentration simultaneously.
    # This captures the full biophysical picture where higher concentration
    # accelerates nucleation (lower t_half) AND increases elongation rate.
    # 6 constants.
    # ------------------------------------------------------------------
    cK = constant_symbols(6)
    rate_K = cK[0] + cK[1] * param
    t_half_K = cK[2] + cK[3] * param
    expr_K = cK[4] * sp.exp(-sp.exp(-rate_K * (time - t_half_K))) + cK[5]
    _try(expr_K, cK, [
        [rate_est,  0.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [rate_est,  0.0,  t_half_est,  conc_slope_est,  plateau_est,  0.0],
        [rate_est * 2,  0.0,  t_half_est,  0.0,  plateau_est,  0.0],
        [rate_est / 2,  0.0,  t_half_est * 1.3,  0.0,  plateau_est,  0.0],
        [0.2,   0.0,  10.0,  0.0,  1.0,  0.0],
        [0.3,   0.01,  8.0,  -0.1,  1.0,  0.0],
        [0.5,   0.0,   5.0,  0.0,  1.0,  0.0],
        [0.1,   0.0,  15.0,  0.0,  1.0,  0.0],
        [1.0,   0.0,  10.0,  0.0,  1.0,  0.0],
        [0.2,  -0.01,  12.0,  0.1,  1.0,  0.0],
        [rate_est, 0.001, t_half_est, conc_slope_est, plateau_est, 0.0],
    ])

    # ------------------------------------------------------------------
    # Template L: Richards logistic with conc-dependent rate AND half-time
    #   rate(x1)   = c0 + c1*x1
    #   t_half(x1) = c2 + c3*x1
    #   y = c5 / (1 + exp(-rate*(x0 - t_half)))^c4 + c6 (free baseline)
    # Asymmetric sigmoid with full concentration coupling.
    # 7 constants.
    # ------------------------------------------------------------------
    cL = constant_symbols(7)
    rate_L = cL[0] + cL[1] * param
    t_half_L = cL[2] + cL[3] * param
    expr_L = cL[5] / (1 + sp.exp(-rate_L * (time - t_half_L))) ** cL[4] + cL[6]
    _try(expr_L, cL, [
        [rate_est,  0.0,  t_half_est,  0.0,  1.0,  plateau_est,  0.0],
        [rate_est,  0.0,  t_half_est,  conc_slope_est,  1.5,  plateau_est,  0.0],
        [rate_est * 2,  0.0,  t_half_est,  0.0,  0.5,  plateau_est,  0.0],
        [0.1,  0.0,  10.0,  0.0,  1.0,  1.0,  0.0],
        [0.3,  0.01,   8.0, -0.1,  1.5,  1.0,  0.0],
        [0.5,  0.0,    5.0,  0.0,  0.5,  0.9,  0.05],
        [0.1,  0.0,   20.0,  0.0,  2.0,  1.0,  0.0],
        [1.0,  0.0,   10.0,  0.0,  1.0,  1.0,  0.0],
        [rate_est, 0.001, t_half_est, conc_slope_est, 1.0, plateau_est, 0.0],
    ])

    # ------------------------------------------------------------------
    # Template M: Softplus-lag Gompertz
    #   Smooth lag phase: effective_time = log(1 + exp(c1*(x0 - c2))) / c1
    #   This is softplus(x0 - lag) which equals ~0 before lag, ~(x0-lag) after.
    #   rate(x1) = c0 + c3*x1,  lag(x1) = c2 + c4*x1
    #   y = c5 * exp(-exp(-rate * softplus(x0 - lag))) + c6
    # Captures the sharp lag-to-growth transition in amyloid nucleation.
    # 7 constants.
    # ------------------------------------------------------------------
    cM = constant_symbols(7)
    rate_M = cM[0] + cM[3] * param
    lag_M = cM[2] + cM[4] * param
    # softplus: log(1 + exp(c1*(x0 - lag))) / c1
    # Use c1 as sharpness of the lag cutoff
    softplus_M = sp.log(1 + sp.exp(cM[1] * (time - lag_M))) / cM[1]
    expr_M = cM[5] * sp.exp(-sp.exp(-rate_M * softplus_M)) + cM[6]
    _try(expr_M, cM, [
        [rate_est,  2.0,  t_lag_est,  0.0,  0.0,  plateau_est,  0.0],
        [rate_est,  1.0,  t_lag_est,  0.0,  0.0,  plateau_est,  0.0],
        [rate_est,  3.0,  t_lag_est,  0.0,  0.0,  plateau_est,  0.0],
        [0.3,  2.0,  5.0,  0.0,  0.0,  1.0,  0.0],
        [0.2,  1.0,  3.0,  0.0,  0.0,  1.0,  0.0],
        [0.5,  2.0,  8.0,  0.0,  0.0,  1.0,  0.0],
        [0.1,  1.0, 10.0,  0.0,  0.0,  1.0,  0.0],
        [rate_est,  2.0,  t_lag_est,  0.001,  0.0,  plateau_est,  0.0],
        [rate_est * 2,  2.0,  t_lag_est * 0.8,  0.0,  0.0,  plateau_est,  0.0],
    ])

    # ------------------------------------------------------------------
    # Template N: Full power-law conc scaling on both rate and half-time
    #   rate(x1)      = c0 * exp(c1 * log(x1 + c6))
    #   half_time(x1) = c2 * exp(c3 * log(x1 + c6))
    #   y = c4 * exp(-exp(-rate*(x0 - half_time))) + c5
    # Gompertz with full power-law concentration coupling.
    # 7 constants.
    # ------------------------------------------------------------------
    cN = constant_symbols(7)
    eps_N = cN[6]
    rate_N = cN[0] * sp.exp(cN[1] * sp.log(param + eps_N))
    half_N = cN[2] * sp.exp(cN[3] * sp.log(param + eps_N))
    expr_N = cN[4] * sp.exp(-sp.exp(-rate_N * (time - half_N))) + cN[5]
    _try(expr_N, cN, [
        [rate_est,  0.0,  t_half_est,  0.0,  plateau_est,  0.0,  1.0],
        [rate_est,  0.5,  t_half_est, -0.5,  plateau_est,  0.0,  1.0],
        [rate_est,  1.0,  t_half_est, -1.0,  plateau_est,  0.0,  0.5],
        [0.2,  0.0,  10.0,  0.0,  1.0,  0.0,  1.0],
        [0.5,  0.5,   5.0, -0.5,  1.0,  0.0,  0.1],
        [0.1,  1.0,  20.0, -1.0,  1.0,  0.0,  0.5],
        [0.3,  0.5,  10.0, -0.3,  0.9,  0.0,  1.0],
        [1.0,  0.0,   8.0,  0.0,  1.0,  0.0,  1.0],
        [rate_est, 0.3, t_half_est, -0.3, plateau_est, 0.0, 0.5],
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
                constants=cFB, initial_values=[rate_est, t_half_est, 1.0, plateau_est],
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
