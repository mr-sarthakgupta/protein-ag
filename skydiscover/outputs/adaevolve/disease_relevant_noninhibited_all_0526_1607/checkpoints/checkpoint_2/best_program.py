# EVOLVE-BLOCK-START
"""
Multi-model symbolic regression for amyloid aggregation kinetics.

Tries several biophysically-motivated equation templates and returns the
best-fitting one (lowest validation NMSE) for each dataset independently.

Models attempted:
  1. Richards / asymmetric logistic — generalised sigmoid with additive
     concentration coupling in rate and half-time.
  2. Avrami nucleation-growth — y = A*(1 - exp(-k*t^n)) + B, where k
     depends on x1 linearly.
  3. Gompertz growth — y = A*exp(-exp(-k*(t - t_half))) + B, a skewed
     sigmoid well-suited to nucleation lag phases.
  4. Hill sigmoidal — y = A * t^n / (t_half^n + t^n) + B with
     concentration-dependent t_half.
  5. Stretched-exponential (KWW) — y = A*(1 - exp(-(t/tau)^beta)) + B,
     captures heterogeneous nucleation kinetics.
  6. Logistic with power-law concentration coupling — the original seed
     model, kept as a robust fallback.
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


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """
    Evaluate multiple kinetic equation templates; return the best result.

    Each template is fitted independently (constants optimised via
    least-squares).  The template with the lowest validation NMSE is
    returned.  All templates share the same feature layout:
        x0 = time, x1 = experimental parameter (concentration, pH, …).

    Concentration coupling uses additive forms (c0 + c1*x1) for numerical
    stability, so single-concentration datasets (x1=1) degrade gracefully.
    """
    x = feature_symbols(X_train.shape[1])
    time = x[0]
    conc = x[1]

    candidates = []

    # ------------------------------------------------------------------
    # Model 1: Richards / asymmetric logistic
    #   y = c5 / (1 + exp(-(c0 + c1*x1)*(x0 - (c2 + c3*x1))))^(1/c4) + c6
    # c4 controls asymmetry (Richards nu); c4=1 → standard logistic.
    # ------------------------------------------------------------------
    c1 = constant_symbols(7)
    rate1 = c1[0] + c1[1] * conc
    t_half1 = c1[2] + c1[3] * conc
    nu1 = c1[4]
    plateau1 = c1[5]
    baseline1 = c1[6]
    inner1 = 1 + sp.exp(-rate1 * (time - t_half1))
    expr1 = plateau1 / (inner1 ** nu1) + baseline1
    candidates.append((expr1, c1, [0.2, 0.01, 20.0, -0.5, 1.0, 1.0, 0.0]))

    # ------------------------------------------------------------------
    # Model 2: Avrami nucleation-growth
    #   y = c4 * (1 - exp(-(c0 + c1*x1) * t^(c2 + c3*x1))) + c5
    # k = c0 + c1*x1 (rate), n = c2 + c3*x1 (Avrami exponent).
    # ------------------------------------------------------------------
    c2 = constant_symbols(6)
    k_av = c2[0] + c2[1] * conc
    n_av = c2[2] + c2[3] * conc
    plateau2 = c2[4]
    baseline2 = c2[5]
    expr2 = plateau2 * (1 - sp.exp(-k_av * sp.Abs(time) ** n_av)) + baseline2
    candidates.append((expr2, c2, [1e-3, 1e-4, 2.0, 0.0, 1.0, 0.0]))

    # ------------------------------------------------------------------
    # Model 3: Gompertz growth (skewed sigmoid, good for lag phases)
    #   y = c4 * exp(-exp(-(c0 + c1*x1)*(x0 - (c2 + c3*x1)))) + c5
    # The Gompertz curve has a longer lag and faster rise than logistic,
    # matching nucleation-elongation kinetics well.
    # ------------------------------------------------------------------
    c3 = constant_symbols(6)
    rate3 = c3[0] + c3[1] * conc
    t_half3 = c3[2] + c3[3] * conc
    plateau3 = c3[4]
    baseline3 = c3[5]
    expr3 = plateau3 * sp.exp(-sp.exp(-rate3 * (time - t_half3))) + baseline3
    candidates.append((expr3, c3, [0.15, 0.005, 25.0, -0.3, 1.0, 0.0]))

    # ------------------------------------------------------------------
    # Model 4: Hill sigmoidal with concentration-dependent half-time
    #   y = c3 * t^c0 / ((c1 + c2*x1)^c0 + t^c0) + c4
    # ------------------------------------------------------------------
    c4 = constant_symbols(5)
    n_hill = c4[0]
    t_half_h = c4[1] + c4[2] * conc
    plateau_h = c4[3]
    baseline_h = c4[4]
    expr4 = (plateau_h * sp.Abs(time) ** n_hill /
             (sp.Abs(t_half_h) ** n_hill + sp.Abs(time) ** n_hill)
             + baseline_h)
    candidates.append((expr4, c4, [2.0, 20.0, -0.5, 1.0, 0.0]))

    # ------------------------------------------------------------------
    # Model 5: Stretched-exponential (KWW / Weibull CDF)
    #   y = c4 * (1 - exp(-((x0 / (c0 + c1*x1))^(c2 + c3*x1)))) + c5
    # Captures heterogeneous nucleation; beta<1 → stretched, beta>1 → compressed.
    # ------------------------------------------------------------------
    c5 = constant_symbols(6)
    tau_kww = c5[0] + c5[1] * conc
    beta_kww = c5[2] + c5[3] * conc
    plateau5 = c5[4]
    baseline5 = c5[5]
    expr5 = (plateau5 * (1 - sp.exp(-(sp.Abs(time) / sp.Abs(tau_kww)) ** beta_kww))
             + baseline5)
    candidates.append((expr5, c5, [30.0, -0.5, 1.5, 0.0, 1.0, 0.0]))

    # ------------------------------------------------------------------
    # Model 6: Logistic with power-law concentration coupling (original seed)
    #   y = c4 / (1 + exp(-c0 * x1^c1 * (x0 - c2 * x1^c3))) + c5
    # ------------------------------------------------------------------
    c6 = constant_symbols(6)
    rate6 = c6[0] * conc ** c6[1]
    half6 = c6[2] * conc ** c6[3]
    plateau6 = c6[4]
    baseline6 = c6[5]
    expr6 = plateau6 / (1 + sp.exp(-rate6 * (time - half6))) + baseline6
    candidates.append((expr6, c6, [0.1, 0.5, 10.0, -0.5, 1.0, 0.0]))

    # ------------------------------------------------------------------
    # Evaluate all candidates; keep the best (lowest nmse_val).
    # ------------------------------------------------------------------
    best_result: dict[str, Any] | None = None
    best_nmse = float("inf")

    for expr, consts, init_vals in candidates:
        try:
            result = evaluate_expression(
                expr,
                X_train,
                y_train,
                X_val,
                y_val,
                constants=consts,
                initial_values=init_vals,
            )
            nmse = result.get("nmse_val", float("inf"))
            if nmse is not None and nmse < best_nmse:
                best_nmse = nmse
                best_result = result
        except Exception:
            continue

    if best_result is None:
        # Ultimate fallback: plain logistic, no concentration coupling
        c_fb = constant_symbols(4)
        time_fb = feature_symbols(X_train.shape[1])[0]
        expr_fb = c_fb[2] / (1 + sp.exp(-c_fb[0] * (time_fb - c_fb[1]))) + c_fb[3]
        best_result = evaluate_expression(
            expr_fb,
            X_train,
            y_train,
            X_val,
            y_val,
            constants=c_fb,
            initial_values=[0.1, 20.0, 1.0, 0.0],
        )

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
