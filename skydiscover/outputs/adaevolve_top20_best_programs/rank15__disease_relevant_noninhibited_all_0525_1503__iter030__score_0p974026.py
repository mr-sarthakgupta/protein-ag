# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: try logistic, Gompertz, and Hill sigmoidal models with
data-adaptive initial values, return the best-fitting result.

The key insight is that amyloid aggregation datasets span many orders of
magnitude in time-scale and concentration, so fixed initial values fail
catastrophically.  By estimating the half-time and rate constant directly
from the data, we ensure convergence across all 60 datasets.
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


def _estimate_kinetic_params(
    X_train: NDArray,
    y_train: NDArray,
) -> tuple[float, float, float, float]:
    """Estimate half-time, rate, y_min, y_max from training data.

    Uses the 10-90% transition width to estimate the apparent rate constant
    and the y=50% crossing point to estimate the half-time.  These estimates
    are used as data-adaptive initial values for the least-squares optimizer,
    ensuring convergence across datasets with very different time-scales.
    """
    times = X_train[:, 0]
    sort_idx = np.argsort(times)
    t_sorted = times[sort_idx]
    y_sorted = y_train[sort_idx]

    y_min = float(y_sorted.min())
    y_max = float(y_sorted.max())
    y_range = max(y_max - y_min, 1e-10)

    # Estimate half-time: time where y is closest to midpoint
    y_half = y_min + 0.5 * y_range
    idx_half = int(np.argmin(np.abs(y_sorted - y_half)))
    t_half = float(t_sorted[idx_half])

    # Estimate rate from 10-90% transition width
    idx_10 = int(np.argmin(np.abs(y_sorted - (y_min + 0.1 * y_range))))
    idx_90 = int(np.argmin(np.abs(y_sorted - (y_min + 0.9 * y_range))))
    t_trans = abs(float(t_sorted[idx_90]) - float(t_sorted[idx_10]))
    t_range_val = abs(float(t_sorted[-1]) - float(t_sorted[0]))
    # logistic: 10-90% width = 4.394/rate; Gompertz: ~2.5/rate
    rate = 4.4 / t_trans if t_trans > 1e-30 else 4.4 / max(t_range_val, 1e-30)

    return t_half, rate, y_min, y_max


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Fit amyloid aggregation kinetics using adaptive multi-model approach.

    Tries logistic, Gompertz, and Hill sigmoidal templates with multiple
    data-adaptive initial conditions, returning the best-fitting result.

    All models include concentration-dependent rate and half-time via
    power-law scaling (x1^c), which collapses to a pure constant when
    x1 is fixed across all curves in a dataset.

    Model 1 – Logistic with concentration dependence:
        y = c4 / (1 + exp(-c0*x1^c1*(x0 - c2*x1^c3))) + c5

    Model 2 – Gompertz with concentration dependence:
        y = c4 * exp(-c6 * exp(-c0*x1^c1*(x0 - c2*x1^c3))) + c5

    Model 3 – Hill/sigmoidal (nucleation-dependent):
        y = c4 * x0^c6 / ((c2*x1^c3)^c6 + x0^c6) + c5

    The Gompertz model captures asymmetric sigmoidal curves that a symmetric
    logistic cannot fit well. The Hill model captures nucleation kinetics
    where the sigmoidal shape is governed by a cooperativity exponent.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).ravel()

    # Data-adaptive initial value estimation
    t_half, rate, y_min, y_max = _estimate_kinetic_params(X_train, y_train)
    ampl = max(y_max - y_min, 0.01)

    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    t_sym = x[0]       # time coordinate
    conc = x[1]        # concentration / experimental parameter

    # ------------------------------------------------------------------ #
    # Model 1: Logistic  y = c4/(1 + exp(-c0*conc^c1*(t - c2*conc^c3))) + c5
    # ------------------------------------------------------------------ #
    expr_logistic = (
        c[4] / (1 + sp.exp(-c[0] * conc ** c[1] * (t_sym - c[2] * conc ** c[3])))
        + c[5]
    )
    logistic_consts = c[:6]

    # Multiple initial-value sets: vary rate scale and concentration exponent
    logistic_inits = [
        [rate,      0.0,   t_half, 0.0,  ampl, y_min],
        [rate,      0.5,   t_half, -0.5, ampl, y_min],
        [rate / 5,  0.0,   t_half, 0.0,  ampl, y_min],
        [rate * 5,  0.0,   t_half, 0.0,  ampl, y_min],
        [rate,      1.0,   t_half, -1.0, ampl, y_min],
        [rate,      2.0,   t_half, -1.0, ampl, y_min],
    ]

    # ------------------------------------------------------------------ #
    # Model 2: Gompertz  y = c4*exp(-c6*exp(-c0*conc^c1*(t - c2*conc^c3))) + c5
    # ------------------------------------------------------------------ #
    expr_gompertz = (
        c[4] * sp.exp(-c[6] * sp.exp(-c[0] * conc ** c[1] * (t_sym - c[2] * conc ** c[3])))
        + c[5]
    )
    gompertz_consts = list(c)  # all 7 constants

    gompertz_inits = [
        [rate,      0.0,   t_half, 0.0,  ampl, y_min, 2.0],
        [rate,      0.5,   t_half, -0.5, ampl, y_min, 3.0],
        [rate / 5,  0.0,   t_half, 0.0,  ampl, y_min, 2.0],
        [rate * 5,  0.0,   t_half, 0.0,  ampl, y_min, 1.5],
        [rate,      1.0,   t_half, -1.0, ampl, y_min, 2.0],
    ]

    # ------------------------------------------------------------------ #
    # Model 3: Hill  y = c4 * t^c6 / ((c2*conc^c3)^c6 + t^c6) + c5
    # Nucleation-dependent sigmoidal; c6 is cooperativity (Hill exponent)
    # ------------------------------------------------------------------ #
    t_half_safe = max(abs(t_half), 1e-10)
    expr_hill = (
        c[4] * t_sym ** c[6] / ((c[2] * conc ** c[3]) ** c[6] + t_sym ** c[6])
        + c[5]
    )
    hill_consts = [c[2], c[3], c[4], c[5], c[6]]  # c0,c1 unused → use c2..c6

    hill_inits = [
        [t_half_safe, 0.0,  ampl, y_min, 2.0],
        [t_half_safe, -0.5, ampl, y_min, 3.0],
        [t_half_safe, 0.5,  ampl, y_min, 2.0],
        [t_half_safe, 0.0,  ampl, y_min, 4.0],
    ]

    best_result: dict[str, Any] | None = None
    best_nmse = float("inf")

    # Try logistic model with each initial-value set
    for init in logistic_inits:
        try:
            result = evaluate_expression(
                expr_logistic,
                X_train, y_train, X_val, y_val,
                constants=logistic_consts,
                initial_values=init,
                max_nfev=200,
            )
            nmse_v = float(result.get("nmse_val", float("inf")))
            if np.isfinite(nmse_v) and nmse_v < best_nmse:
                best_nmse = nmse_v
                best_result = result
        except Exception:
            pass

    # Try Gompertz model with each initial-value set
    for init in gompertz_inits:
        try:
            result = evaluate_expression(
                expr_gompertz,
                X_train, y_train, X_val, y_val,
                constants=gompertz_consts,
                initial_values=init,
                max_nfev=200,
            )
            nmse_v = float(result.get("nmse_val", float("inf")))
            if np.isfinite(nmse_v) and nmse_v < best_nmse:
                best_nmse = nmse_v
                best_result = result
        except Exception:
            pass

    # Try Hill model with each initial-value set
    for init in hill_inits:
        try:
            result = evaluate_expression(
                expr_hill,
                X_train, y_train, X_val, y_val,
                constants=hill_consts,
                initial_values=init,
                max_nfev=200,
            )
            nmse_v = float(result.get("nmse_val", float("inf")))
            if np.isfinite(nmse_v) and nmse_v < best_nmse:
                best_nmse = nmse_v
                best_result = result
        except Exception:
            pass

    if best_result is None:
        return {
            "equation": "",
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
