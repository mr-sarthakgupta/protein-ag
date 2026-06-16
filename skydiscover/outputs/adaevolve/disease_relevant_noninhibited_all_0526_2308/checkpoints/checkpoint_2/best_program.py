# EVOLVE-BLOCK-START
"""
Improved symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: Fit a small set of carefully chosen, biophysically-motivated
equation templates with diverse initial guesses, and return the best.

Key design choices:
  1. Richards generalised logistic (asymmetric sigmoid) — handles both
     fast and slow nucleation, symmetric and skewed curves.
  2. Avrami nucleation-growth — physically motivated for amyloid kinetics,
     handles power-law growth in the early phase.
  3. Finke-Watzky two-step model — nucleation (k1) + autocatalytic growth
     (k2), the standard biophysical model for amyloid aggregation.
  4. Concentration enters via a simple additive offset to the time axis
     (half-time shift) rather than a complex power-law, which is more
     numerically stable and reduces the number of constants.
  5. Multiple initial-value sets per template to escape local minima.
  6. Returns the template with the lowest validation NMSE.
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
    """Fit biophysically-motivated amyloid kinetics templates; return best.

    Templates:
      A) Richards generalised logistic with concentration-shifted half-time:
           t_half = c1 + c2 * x1
           y = c4 / (1 + exp(-c0*(x0 - t_half)))^c3 + c5
         6 constants. Asymmetric sigmoid, robust across protein systems.

      B) Avrami nucleation-growth with concentration-scaled rate:
           k = c0 * (1 + c1 * x1)
           y = c3 * (1 - exp(-k * x0^c2)) + c4
         5 constants. Handles power-law growth, heterogeneous nucleation.

      C) Finke-Watzky two-step (nucleation k1 + autocatalytic growth k2):
           A = k1/k2 * exp((k1+k2)*x0)
           y = c4 * A / (1 + A) + c3
         where k1=c0, k2=c1*(1+c2*x1), t_offset=c5
           y = c4 * exp((c0+c1)*(x0-c5)) / (c1/c0 + exp((c0+c1)*(x0-c5))) + c3
         5 constants. Standard amyloid aggregation biophysical model.

      D) Logistic with concentration-dependent rate and half-time (power-law):
           rate = c0 * x1^c1  (via exp(c1*log(x1+eps)))
           t50  = c2 * x1^c3
           y = c4 / (1 + exp(-rate*(x0 - t50))) + c5
         7 constants. Explicit concentration scaling.

    Each template is tried with multiple initial guesses; best nmse_val wins.
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

    def _try(expr, consts, inits):
        """Try all initial-value sets for one template; update best in place."""
        nonlocal best_result, best_nmse
        for init in inits:
            try:
                res = evaluate_expression(
                    expr, X_train, y_train, X_val, y_val,
                    constants=consts, initial_values=init,
                )
                v = float(res.get("nmse_val", float("inf")))
                if np.isfinite(v) and v < best_nmse:
                    best_nmse = v
                    best_result = res
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Template A: Richards generalised logistic
    #   t_half(x1) = c1 + c2*x1   (linear concentration shift)
    #   y = c4 / (1 + exp(-c0*(x0 - t_half)))^c3 + c5
    # 6 constants. Asymmetric sigmoid; c3=1 → standard logistic.
    # ------------------------------------------------------------------
    cA = constant_symbols(6)
    t_half_A = cA[1] + cA[2] * param
    inner_A = 1 + sp.exp(-cA[0] * (time - t_half_A))
    expr_A = cA[4] / inner_A ** cA[3] + cA[5]
    _try(expr_A, cA, [
        [0.1,  10.0, 0.0, 1.0, 1.0,  0.0],
        [0.3,   8.0, 0.1, 1.5, 1.0,  0.0],
        [0.5,   5.0, 0.0, 0.5, 0.9,  0.05],
        [0.1,  20.0, 0.0, 2.0, 1.0,  0.0],
        [1.0,  10.0, 0.0, 1.0, 1.0,  0.0],
        [0.2,  15.0, -0.1, 1.0, 1.0, 0.0],
    ])

    # ------------------------------------------------------------------
    # Template B: Avrami nucleation-growth
    #   k(x1) = c0 * (1 + c1*x1)
    #   y = c3 * (1 - exp(-k * x0^c2)) + c4
    # 5 constants. Physically motivated; c2=2 → classical Avrami.
    # ------------------------------------------------------------------
    cB = constant_symbols(5)
    k_B = cB[0] * (1 + cB[1] * param)
    expr_B = cB[3] * (1 - sp.exp(-k_B * time ** cB[2])) + cB[4]
    _try(expr_B, cB, [
        [0.01, 0.0, 2.0, 1.0, 0.0],
        [0.1,  0.0, 1.5, 1.0, 0.0],
        [0.001, 0.01, 2.5, 0.9, 0.0],
        [0.05, 0.0, 2.0, 1.0, 0.0],
        [0.01, 0.1, 2.0, 1.0, 0.0],
    ])

    # ------------------------------------------------------------------
    # Template C: Finke-Watzky two-step aggregation
    #   Standard FW: y = A/(1 + (k1/k2)*exp(-(k1+k2)*t))
    #   Here: k_eff = c0 + c1*(1 + c2*x1),  ratio = c0/c1
    #   y = c3 / (1 + (c0/c1)*exp(-c0*(x0-c4))) + c5
    #   (simplified: ratio and rate as free constants)
    #   y = c3 / (1 + c4*exp(-c0*(x0-c1))) + c5   (4 shape + 1 offset)
    # 6 constants.
    # ------------------------------------------------------------------
    cC = constant_symbols(6)
    # FW-inspired: y = c3 / (1 + c4*exp(-c0*(x0 - c1*(1+c2*x1)))) + c5
    t_half_C = cC[1] * (1 + cC[2] * param)
    expr_C = cC[3] / (1 + cC[4] * sp.exp(-cC[0] * (time - t_half_C))) + cC[5]
    _try(expr_C, cC, [
        [0.1,  10.0, 0.0, 1.0, 1.0, 0.0],
        [0.5,   8.0, 0.0, 1.0, 1.0, 0.0],
        [0.2,  12.0, 0.1, 0.9, 2.0, 0.0],
        [0.1,  15.0, 0.0, 1.0, 0.5, 0.0],
        [1.0,   5.0, 0.0, 1.0, 1.0, 0.0],
    ])

    # ------------------------------------------------------------------
    # Template D: Logistic with concentration power-law rate + half-time
    #   rate(x1) = c0 * exp(c1 * log(x1 + c6))
    #   t50(x1)  = c2 * exp(c3 * log(x1 + c6))
    #   y = c4 / (1 + exp(-rate*(x0 - t50))) + c5
    # 7 constants. Handles strong concentration dependence.
    # ------------------------------------------------------------------
    cD = constant_symbols(7)
    eps_D = cD[6]
    rate_D = cD[0] * sp.exp(cD[1] * sp.log(param + eps_D))
    half_D = cD[2] * sp.exp(cD[3] * sp.log(param + eps_D))
    expr_D = cD[4] / (1 + sp.exp(-rate_D * (time - half_D))) + cD[5]
    _try(expr_D, cD, [
        [0.1,  0.5, 10.0, -0.5, 1.0,  0.0, 1.0],
        [0.5,  1.0,  5.0,  0.0, 1.0,  0.0, 0.1],
        [0.05, 0.3, 20.0, -1.0, 0.9, -0.05, 0.5],
        [1.0,  0.0, 10.0,  0.0, 1.0,  0.0, 1.0],
    ])

    # Fallback if all templates fail
    if best_result is None:
        cF = constant_symbols(5)
        expr_F = cF[3] / (1 + sp.exp(-cF[0] * (time - cF[1]))) ** cF[2] + cF[4]
        try:
            best_result = evaluate_expression(
                expr_F, X_train, y_train, X_val, y_val,
                constants=cF, initial_values=[0.1, 10.0, 1.0, 1.0, 0.0],
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
