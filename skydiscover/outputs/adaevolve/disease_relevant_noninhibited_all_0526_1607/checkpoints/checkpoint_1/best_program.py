# EVOLVE-BLOCK-START
"""
Multi-model symbolic regression for amyloid aggregation kinetics.

Tries several biophysically-motivated equation templates and returns the
best-fitting one (lowest validation NMSE) for each dataset independently.

Models attempted:
  1. Generalized Richards logistic — asymmetric sigmoid with linear
     concentration coupling in rate and half-time.
  2. Avrami nucleation-growth — y = A*(1 - exp(-k*(x0^n))) + B, where k
     depends on x1 linearly. Designed for nucleation-dependent polymerization.
  3. Double-exponential (Finke-Watzky inspired) — captures slow nucleation
     followed by fast autocatalytic growth.
  4. Hill sigmoidal — y = A * x0^n / (t_half^n + x0^n) + B with
     concentration-dependent t_half.
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

    The concentration coupling is kept numerically stable by using
    additive/linear forms (c0 + c1*x1) rather than power-laws, so that
    single-concentration datasets (x1=1) degrade gracefully.
    """
    x = feature_symbols(X_train.shape[1])
    time = x[0]
    conc = x[1]

    candidates = []

    # ------------------------------------------------------------------
    # Model 1: Generalized Richards / asymmetric logistic
    #   y = c4 / (1 + exp(-(c0 + c1*x1)*(x0 - (c2 + c3*x1))))^(1/c5) + c6
    # The exponent 1/c5 controls asymmetry (Richards parameter).
    # With c5=1 this reduces to the standard logistic.
    # ------------------------------------------------------------------
    c1 = constant_symbols(7)
    rate1 = c1[0] + c1[1] * conc
    t_half1 = c1[2] + c1[3] * conc
    nu1 = c1[4]          # Richards asymmetry exponent
    plateau1 = c1[5]
    baseline1 = c1[6]
    inner1 = 1 + sp.exp(-rate1 * (time - t_half1))
    expr1 = plateau1 / (inner1 ** nu1) + baseline1
    candidates.append((
        expr1, c1,
        [0.2, 0.01, 20.0, -0.5, 1.0, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # Model 2: Avrami nucleation-growth
    #   y = c4 * (1 - exp(-(c0 + c1*x1) * x0^(c2 + c3*x1))) + c5
    # k = c0 + c1*x1  (rate, concentration-dependent)
    # n = c2 + c3*x1  (Avrami exponent, may vary with concentration)
    # ------------------------------------------------------------------
    c2 = constant_symbols(6)
    k_av = c2[0] + c2[1] * conc
    n_av = c2[2] + c2[3] * conc
    plateau2 = c2[4]
    baseline2 = c2[5]
    # Guard: use sp.Abs to avoid complex values from negative base
    expr2 = plateau2 * (1 - sp.exp(-k_av * sp.Abs(time) ** n_av)) + baseline2
    candidates.append((
        expr2, c2,
        [1e-3, 1e-4, 2.0, 0.0, 1.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # Model 3: Double-logistic (biphasic / Finke-Watzky inspired)
    #   y = c5 / (1 + exp(-c0*(x0 - c1))) +
    #       c6 / (1 + exp(-c2*(x0 - c3))) * x1^c4 + c7
    # Two sigmoidal components capture lag + growth phases separately.
    # ------------------------------------------------------------------
    c3 = constant_symbols(8)
    sig_a = c3[5] / (1 + sp.exp(-c3[0] * (time - c3[1])))
    sig_b = c3[6] / (1 + sp.exp(-c3[2] * (time - c3[3]))) * (c3[4] * conc)
    expr3 = sig_a + sig_b + c3[7]
    candidates.append((
        expr3, c3,
        [0.1, 10.0, 0.5, 30.0, 0.01, 0.8, 0.2, 0.0],
    ))

    # ------------------------------------------------------------------
    # Model 4: Hill sigmoidal with concentration-dependent half-time
    #   y = c3 * x0^c1 / ((c2 + c4*x1)^c1 + x0^c1) + c5
    # Classic Hill equation; t_half = c2 + c4*x1 shifts with concentration.
    # ------------------------------------------------------------------
    c4 = constant_symbols(6)
    n_hill = c4[0]
    t_half_h = c4[1] + c4[2] * conc
    plateau_h = c4[3]
    baseline_h = c4[4]
    # Protect against zero/negative t_half with Abs
    expr4 = (plateau_h * sp.Abs(time) ** n_hill /
             (sp.Abs(t_half_h) ** n_hill + sp.Abs(time) ** n_hill)
             + baseline_h)
    candidates.append((
        expr4, c4,
        [2.0, 20.0, -0.5, 1.0, 0.0, 0.0],
    ))

    # ------------------------------------------------------------------
    # Evaluate all candidates; keep the best (lowest nmse_val).
    # Fall back to the next best if a candidate raises an exception.
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
        expr_fb = c_fb[2] / (1 + sp.exp(-c_fb[0] * (time - c_fb[1]))) + c_fb[3]
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
