# EVOLVE-BLOCK-START
"""ODE discovery seed for normalized multi-dataset amyloid aggregation kinetics."""

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
    Generic bounded aggregation-rate ODE.

    This seed starts from a broad mechanism class rather than a specific
    closed-form solution: a concentration-independent source term can create
    onset from near-zero signal, while an autocatalytic state-dependent term
    can sharpen growth once aggregates are present.  Monomer and seed features
    modulate both routes, and a capacity factor keeps the trajectory saturating
    toward a fitted plateau.

    Features: x0 = normalized time, x1 = initial monomer m0, x2 = initial
    seed/aggregate M0, and x3 = current normalized concentration/state.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    # Autonomous, mass-conserving Knowles/Cohen-style aggregation front.  No
    # explicit-time forcing is used (that drift term is non-physical and hurts
    # held-out extrapolation); the entire sigmoid emerges from the coupling of
    # nucleation, seeded elongation, and secondary self-catalysis into a single
    # saturating moving front whose capacity (plateau - c) -> 0 at the plateau.
    #   * primary    : monomer-driven primary nucleation (baseline + c0^2*m0^c1)
    #                  breaks the flat lag phase from a near-zero baseline.
    #   * elongation : seeded growth c2^2*M0*m0, LINEAR in c, giving immediate
    #                  fast onset whenever seed M0 > 0.
    #   * secondary  : autocatalytic self-catalysis c3^2*m0^c4*c^2 reproduces the
    #                  steep explosive inflection slope, directly targeting the
    #                  shape loss (slope profile + onset/half-response timing).
    # Constants enter squared so every kinetic rate is non-negative and powers
    # act only on positive features (m0), keeping odeint and least-squares
    # numerically well behaved with no singularities or complex values.
    plateau = c[7]
    capacity = plateau - concentration

    # Single variable-order autocatalytic (secondary-nucleation) term.  The
    # reaction order in the current state c is the dominant control over the
    # inflection sharpness and half-response/onset timing that the shape loss
    # penalizes, so we expose it as a fitted exponent rather than fixing it at
    # 2.  Writing the order as (1 + c5^2) keeps it >= 1 (autocatalysis is at
    # least linear in the aggregate state) and never negative; since c stays in
    # [0, plateau] >= 0, the power c**(1+c5^2) is real, finite, and smooth with
    # no singularity at c = 0.  Replacing the previous fixed-order quadratic
    # plus redundant linear term with one tunable-order term keeps the constant
    # count at 8 while giving the optimizer direct, per-dataset control of the
    # sigmoid front shape.
    baseline_flux = c[6] ** 2
    primary = c[0] ** 2 * monomer ** c[1]
    elongation = c[2] ** 2 * seed * monomer * concentration
    secondary = c[3] ** 2 * monomer ** c[4] * concentration ** (1 + c[5] ** 2)
    growth = baseline_flux + primary + elongation + secondary

    expression = capacity * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 0.5, 1.0, 0.5, 1.0, 0.01, 1.0],
    )


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
