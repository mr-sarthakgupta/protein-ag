# EVOLVE-BLOCK-START
"""Symbolic regression seed for Alpha-synuclein Gaspar 2017 0.3uM seed data."""

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
    Evaluate multiple biophysically-motivated sigmoidal models for seeded aSyn aggregation.

    Focuses on Knowles integrated rate law (sinh-based), Hill function, and logistic
    variants with power-law concentration dependence. Returns the best-scoring candidate.

    Key physics:
    - Sigmoidal S-shape (lag phase + rapid growth + plateau)
    - Concentration-dependent lag time (shorter at higher [monomer])
    - Concentration-dependent growth rate (power law from secondary nucleation theory)
    """
    x = feature_symbols(X_train.shape[1])
    time = x[0]
    concentration = x[1]

    best_result = None
    best_score = -1e10

    def _update(result):
        nonlocal best_score, best_result
        if result and result.get("combined_score", -1e10) > best_score:
            best_score = result["combined_score"]
            best_result = result

    # --- Candidate 1: Knowles integrated rate law (best from prior runs) ---
    # M(t)/Mtot ~ 1 - 1/(1 + B*sinh(kappa*t))^2
    # kappa = c0 * conc^c1 (secondary nucleation rate ~ conc^n2)
    # B = c2 * conc^c3 (amplitude scales with concentration)
    c1 = constant_symbols(6)
    kappa1 = c1[0] * (concentration ** c1[1])
    B1 = c1[2] * (concentration ** c1[3])
    expr1 = c1[4] * (1 - 1 / (1 + B1 * sp.sinh(kappa1 * time)) ** 2) + c1[5]
    _update(evaluate_expression(expr1, X_train, y_train, X_val, y_val, constants=c1))

    # --- Candidate 2: Knowles model with cosh-based variant ---
    # Uses tanh instead of sinh for potentially better numerical stability
    # y = A * (1 + tanh(kappa*(t - t_half))) / 2 + baseline
    # where kappa and t_half both depend on concentration
    c2 = constant_symbols(6)
    kappa2 = c2[0] * (concentration ** c2[1])
    t_half2 = c2[2] * (concentration ** c2[3])
    expr2 = c2[4] * (1 + sp.tanh(kappa2 * (time - t_half2))) / 2 + c2[5]
    _update(evaluate_expression(expr2, X_train, y_train, X_val, y_val, constants=c2))

    # --- Candidate 3: Hill function with power-law concentration dependence ---
    # y = A * t^n / (t_half^n + t^n) + baseline
    # t_half = c0 * conc^c1, n = c2
    c3 = constant_symbols(5)
    t_half3 = c3[0] * (concentration ** c3[1])
    n3 = c3[2]
    expr3 = c3[3] * (time ** n3) / (t_half3 ** n3 + time ** n3) + c3[4]
    _update(evaluate_expression(expr3, X_train, y_train, X_val, y_val, constants=c3))

    # --- Candidate 4: Logistic with power-law concentration in rate and t_half ---
    # k = c0 * conc^c1, t_half = c2 * conc^c3
    c4 = constant_symbols(6)
    k4 = c4[0] * (concentration ** c4[1])
    t_half4 = c4[2] * (concentration ** c4[3])
    expr4 = c4[4] + (1 - c4[4]) / (1 + sp.exp(-k4 * (time - t_half4))) + c4[5]
    _update(evaluate_expression(expr4, X_train, y_train, X_val, y_val, constants=c4))

    # --- Candidate 5: Knowles model with additive correction for lag phase ---
    # Extended Knowles: adds exponential decay term for pre-nucleation baseline
    c5 = constant_symbols(7)
    kappa5 = c5[0] * (concentration ** c5[1])
    B5 = c5[2] * (concentration ** c5[3])
    expr5 = c5[4] * (1 - 1 / (1 + B5 * sp.sinh(kappa5 * time)) ** 2) + c5[5] * sp.exp(-c5[6] * time)
    _update(evaluate_expression(expr5, X_train, y_train, X_val, y_val, constants=c5))

    # --- Candidate 6: Avrami with power-law concentration ---
    # y = A * (1 - exp(-k * t^n)) + baseline, k = c0 * conc^c1
    c6 = constant_symbols(5)
    k6 = c6[0] * (concentration ** c6[1])
    n6 = c6[2]
    expr6 = c6[3] * (1 - sp.exp(-k6 * time ** n6)) + c6[4]
    _update(evaluate_expression(expr6, X_train, y_train, X_val, y_val, constants=c6))

    # --- Candidate 7: Logistic with exponential concentration dependence ---
    # k = c0 * exp(c1 * conc), t_half = c2 * exp(-c3 * conc)
    c7 = constant_symbols(6)
    k7 = c7[0] * sp.exp(c7[1] * concentration)
    t_half7 = c7[2] * sp.exp(-c7[3] * concentration)
    expr7 = c7[4] + (1 - c7[4]) / (1 + sp.exp(-k7 * (time - t_half7))) + c7[5]
    _update(evaluate_expression(expr7, X_train, y_train, X_val, y_val, constants=c7))

    # --- Candidate 8: Knowles with linear+power-law kappa ---
    # kappa = (c0 + c1*conc) * conc^c2, allows more flexible concentration scaling
    c8 = constant_symbols(7)
    kappa8 = c8[0] * (concentration ** c8[1]) + c8[2] * concentration
    B8 = c8[3] * (concentration ** c8[4])
    expr8 = c8[5] * (1 - 1 / (1 + B8 * sp.sinh(kappa8 * time)) ** 2) + c8[6]
    _update(evaluate_expression(expr8, X_train, y_train, X_val, y_val, constants=c8))

    return best_result if best_result is not None else best_result


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
    """Load deterministic Alpha-synuclein Gaspar 2017 splits (matches evaluator)."""
    from evaluator import load_alphasyn_data

    return load_alphasyn_data()


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = _load_data()
    result = run_discovery(X_train, y_train, X_val, y_val)
    print("equation:", result.get("equation"))
    print("nmse_val:", result.get("nmse_val"))
    print("combined_score:", result.get("combined_score"))
