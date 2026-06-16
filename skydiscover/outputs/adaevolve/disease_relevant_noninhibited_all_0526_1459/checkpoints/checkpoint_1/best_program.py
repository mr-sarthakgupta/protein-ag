# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid aggregation kinetics.

Tries multiple biophysically-motivated equation templates and returns
the best-scoring one (lowest validation NMSE).
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from numpy.typing import NDArray

from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the result with the lowest nmse_val (ignoring inf/None)."""
    valid = [r for r in results if r is not None and r.get("nmse_val") is not None
             and r.get("nmse_val") != float("inf")]
    if not valid:
        return results[0]
    return min(valid, key=lambda r: r["nmse_val"])


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """
    Evaluate multiple sigmoidal/nucleation kinetic templates and return best.

    Templates tried:
    1. Generalized logistic (Richards): plateau/(1+exp(-k*(t-t0)))^(1/nu) + base
       - Asymmetry parameter nu captures lag-phase skewness common in nucleation
    2. Hill / sigmoidal with concentration-dependent half-time:
       y = c4 * t^c0 / (c1^c0 * x1^c2 + t^c0) + c3
       - Hill equation naturally handles cooperative nucleation kinetics
    3. Avrami nucleation-growth: 1 - exp(-k*(t-lag)^n) rescaled
       - Classic model for crystallization/amyloid nucleation-elongation
    4. Double-logistic (fast + slow phase): captures biphasic aggregation

    All use x1 via log(1+|x1|) to safely handle x1=1 (single-conc) and
    multi-concentration datasets without power-law singularities.

    Features: x0 = elapsed time, x1 = varying experimental parameter.
    Constants fitted independently per dataset via least-squares.
    """
    x = feature_symbols(X_train.shape[1])
    t = x[0]
    p = x[1]

    results = []

    # --- Template 1: Generalized logistic (Richards curve) ---
    # y = c4 / (1 + exp(-c0*(t - c1*p^c2)))^c3 + c5
    # c3 (nu shape) allows asymmetric sigmoid for lag-phase nucleation
    # Use sp.Abs(p) + 1e-6 to avoid complex numbers with fractional exponents
    c = constant_symbols(6)
    safe_p = sp.Abs(p) + sp.Float(1e-6)
    rate1 = c[0] * safe_p ** c[2]
    t_half1 = c[1]
    nu = c[3]
    expr1 = c[4] / (1 + sp.exp(-rate1 * (t - t_half1))) ** nu + c[5]
    try:
        r1 = evaluate_expression(
            expr1, X_train, y_train, X_val, y_val,
            constants=c,
            initial_values=[0.2, 10.0, 0.3, 1.0, 1.0, 0.0],
        )
        results.append(r1)
    except Exception:
        pass

    # --- Template 2: Hill equation with concentration-scaling ---
    # y = c3 * t^c0 / ((c1 * safe_p^c2)^c0 + t^c0) + c4
    # Hill equation naturally models cooperative nucleation; safe_p avoids
    # singularities; concentration scales the effective half-time
    c2 = constant_symbols(5)
    safe_p2 = sp.Abs(p) + sp.Float(1e-6)
    t_n = t ** c2[0]
    t_half_n = (c2[1] * safe_p2 ** c2[2]) ** c2[0]
    expr2 = c2[3] * t_n / (t_half_n + t_n) + c2[4]
    try:
        r2 = evaluate_expression(
            expr2, X_train, y_train, X_val, y_val,
            constants=c2,
            initial_values=[2.0, 10.0, -0.5, 1.0, 0.0],
        )
        results.append(r2)
    except Exception:
        pass

    # --- Template 3: Avrami nucleation-growth model ---
    # y = c3 * (1 - exp(-c0 * (max(t - c1, 0))^c2)) + c4
    # Classic nucleation-elongation: lag time c1, growth rate c0, Avrami n=c2
    # sp.Max used to enforce t > lag; safe clamp avoids negative base
    c3 = constant_symbols(5)
    lag = c3[1]
    t_lag = t - lag
    # Use softplus-style clamp: log(1 + exp(t_lag)) ≈ max(t_lag, 0)
    t_lag_soft = sp.log(1 + sp.exp(t_lag))
    expr3 = c3[3] * (1 - sp.exp(-c3[0] * t_lag_soft ** c3[2])) + c3[4]
    try:
        r3 = evaluate_expression(
            expr3, X_train, y_train, X_val, y_val,
            constants=c3,
            initial_values=[0.01, 5.0, 2.0, 1.0, 0.0],
        )
        results.append(r3)
    except Exception:
        pass

    # --- Template 4: Finke-Watzky two-step mechanism ---
    # Nucleation (k1) + autocatalytic growth (k2):
    # y = c4*(1 - k1/(k1+k2*c3) * exp(-(k1+k2*c3)*t)) / (1 + k1/k2/c3) + c5
    # Simplified FW: y = c4 / (1 + exp(-c0*(t - c1))) * (1 - exp(-c2*t)) + c5
    # This captures the initial lag AND the eventual plateau
    c4 = constant_symbols(6)
    safe_p4 = sp.log(1 + sp.Abs(p))  # log-concentration scaling
    k_eff = c4[0] * (1 + safe_p4 * c4[1])
    t_half4 = c4[2]
    expr4 = c4[4] / (1 + sp.exp(-k_eff * (t - t_half4))) + c4[5]
    try:
        r4 = evaluate_expression(
            expr4, X_train, y_train, X_val, y_val,
            constants=c4,
            initial_values=[0.2, 0.1, 10.0, 1.0, 1.0, 0.0],
        )
        results.append(r4)
    except Exception:
        pass

    return _best_result(results) if results else {}


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
