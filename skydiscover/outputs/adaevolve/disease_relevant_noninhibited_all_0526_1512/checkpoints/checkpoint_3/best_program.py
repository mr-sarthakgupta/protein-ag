# EVOLVE-BLOCK-START
"""Multi-model symbolic regression for amyloid aggregation kinetics.

Tries several physically-motivated model families per dataset and returns
the best-fitting one. Models cover symmetric/asymmetric sigmoids, nucleation-
growth (Avrami), and Hill-type cooperativity — all with power-law
concentration dependence so they work for both multi- and single-concentration
datasets (x1=1 collapses power-law terms to constants).
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


def _logistic(X_train, y_train, X_val, y_val):
    """Standard logistic sigmoid with power-law concentration scaling.

    y = c4 / (1 + exp(-c0*x1^c1 * (x0 - c2*x1^c3))) + c5
    Rate and half-time both scale as power-laws of x1 (concentration/pH).
    6 constants: minimal overfitting risk.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)
    time, conc = x[0], x[1]
    rate = c[0] * conc ** c[1]
    t_half = c[2] * conc ** c[3]
    expr = c[4] / (1 + sp.exp(-rate * (time - t_half))) + c[5]
    return evaluate_expression(
        expr, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 0.0],
    )


def _richards(X_train, y_train, X_val, y_val):
    """Generalised logistic (Richards curve) — handles asymmetric sigmoids.

    y = c4 / (1 + c5*exp(-c0*x1^c1*(x0 - c2*x1^c3)))^(1/c5) + c6 - c7

    Simplified to 7-constant form:
    y = c4 / (1 + exp(-c0*x1^c1*(x0 - c2*x1^c3)))^c5 + c6

    The exponent c5 on the denominator allows asymmetry (c5=1 → standard
    logistic). Captures slower-rising or faster-falling aggregation curves.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)
    time, conc = x[0], x[1]
    rate = c[0] * conc ** c[1]
    t_half = c[2] * conc ** c[3]
    denom = (1 + sp.exp(-rate * (time - t_half))) ** c[4]
    expr = c[5] / denom + c[6]
    return evaluate_expression(
        expr, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 1.0, 0.0],
    )


def _avrami(X_train, y_train, X_val, y_val):
    """Avrami / JMAK nucleation-growth model with concentration scaling.

    y = c3 * (1 - exp(-c0 * x1^c1 * x0^c2)) + c4

    Physically motivated for nucleation-dependent polymerization: Avrami
    exponent c2 encodes the dimensionality/mechanism of growth. Rate c0
    scales as power-law of concentration.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)
    time, conc = x[0], x[1]
    rate = c[0] * conc ** c[1]
    expr = c[3] * (1 - sp.exp(-rate * time ** c[2])) + c[4]
    return evaluate_expression(
        expr, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.01, 0.5, 2.0, 1.0, 0.0],
    )


def _hill(X_train, y_train, X_val, y_val):
    """Hill / cooperative sigmoidal model with concentration-dependent half-time.

    y = c3 * x0^c0 / ((c1*x1^c2)^c0 + x0^c0) + c4

    Hill exponent c0 captures cooperativity/nucleation order.
    Half-time t_half = c1*x1^c2 scales with concentration via power-law.
    Naturally bounded, no exp overflow risk.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)
    time, conc = x[0], x[1]
    t_half = c[1] * conc ** c[2]
    n = c[0]
    expr = c[3] * time ** n / (t_half ** n + time ** n) + c[4]
    return evaluate_expression(
        expr, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[2.0, 10.0, -0.5, 1.0, 0.0],
    )


def _double_logistic(X_train, y_train, X_val, y_val):
    """Double (biphasic) logistic for two-step aggregation kinetics.

    y = c4/(1+exp(-c0*(x0-c1))) + c5/(1+exp(-c2*(x0-c3))) + c6

    Captures two-phase aggregation (primary + secondary nucleation) without
    concentration dependence — useful for datasets where x1 variation is
    minimal or the kinetics show two distinct transitions.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)
    time = x[0]
    sig1 = c[4] / (1 + sp.exp(-c[0] * (time - c[1])))
    sig2 = c[5] / (1 + sp.exp(-c[2] * (time - c[3])))
    expr = sig1 + sig2 + c[6]
    return evaluate_expression(
        expr, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.1, 5.0, 0.05, 20.0, 0.5, 0.5, 0.0],
    )


def _finke_watzky(X_train, y_train, X_val, y_val):
    """Finke-Watzky two-step nucleation-autocatalytic growth model.

    y = c3 * (1 - (c0/c1) / ((c0/c1) + exp(-(c0+c1)*c2*x1^c4*x0))) + c5

    Simplified FW: y = c3*(1 - 1/(1 + (c1/c0)*exp(-(c0+c1)*rate*x0))) + c5
    where rate = c2*x1^c4.

    c0 = nucleation rate constant k1
    c1 = autocatalytic growth rate k2
    c2*x1^c4 = concentration-dependent effective rate scale
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)
    time, conc = x[0], x[1]
    k1, k2 = c[0], c[1]
    rate_scale = c[2] * conc ** c[3]
    # FW: [A]/[A]0 = (k1+k2) / (k2 + k1*exp((k1+k2)*t))
    # fraction converted = 1 - [A]/[A]0
    total = (k1 + k2) * rate_scale * time
    expr = c[4] * (1 - (k1 + k2) / (k2 + k1 * sp.exp(-total))) + c[5]
    return evaluate_expression(
        expr, X_train, y_train, X_val, y_val,
        constants=c,
        initial_values=[0.01, 0.1, 1.0, 0.5, 1.0, 0.0],
    )


def run_discovery(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Try multiple kinetic model families; return the one with lowest val NMSE.

    Models tried (in order of expected fit quality):
      1. Logistic sigmoid (symmetric, 6 constants)
      2. Avrami nucleation-growth (5 constants)
      3. Hill cooperative (5 constants)
      4. Richards generalised logistic (asymmetric, 7 constants)
      5. Finke-Watzky two-step (6 constants)
      6. Double logistic biphasic (7 constants)

    Each model uses power-law concentration dependence (x1^c) so it
    degenerates gracefully to simple constants when x1=1.
    """
    candidates = [
        _logistic,
        _avrami,
        _hill,
        _richards,
        _finke_watzky,
        _double_logistic,
    ]

    best_result = None
    best_nmse = float("inf")

    for model_fn in candidates:
        try:
            result = model_fn(X_train, y_train, X_val, y_val)
            nmse = result.get("nmse_val", float("inf"))
            if nmse is not None and nmse < best_nmse:
                best_nmse = nmse
                best_result = result
        except Exception:
            continue

    if best_result is None:
        # Fallback: plain logistic
        best_result = _logistic(X_train, y_train, X_val, y_val)

    return best_result


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
