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
    c = constant_symbols(9)

    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    # Autonomous, mass-conserving Knowles/Cohen amyloid master-equation front.
    # The whole sigmoid emerges from coupling three physically distinct routes
    # into one saturating moving front whose capacity (plateau - c) -> 0 at the
    # plateau, so the trajectory always bounds itself and conserves convertible
    # mass.  No explicit-time forcing is used (drift hurts held-out curves).
    #   * baseline+primary : monomer-driven primary nucleation
    #                        (c6^2 + c0^2*m0^c1) breaks the flat lag phase from a
    #                        near-zero signal.
    #   * elongation       : seeded growth c2^2*M0*m0, LINEAR in c, giving an
    #                        immediate fast onset whenever seed M0 > 0 (separated
    #                        from secondary nucleation as a distinct mechanism).
    #   * secondary        : autocatalytic self-catalysis with a FITTED reaction
    #                        order in the aggregate state,
    #                        c3^2*m0^c4 * c**(1 + c5^2).  The order in c is the
    #                        dominant control over inflection sharpness and the
    #                        onset/half-response timing the shape loss penalizes,
    #                        so exposing it as a tunable exponent (kept >= 1 via
    #                        1 + c5^2, so autocatalysis is at least linear) lets
    #                        each dataset match its own sigmoid steepness.
    # Constants enter squared so every kinetic rate is non-negative; powers act
    # only on positive features (m0) and on c in [0, plateau] >= 0 with exponent
    # >= 1, so the RHS is real, finite, and smooth with no singularities,
    # keeping least-squares fitting and odeint numerically well behaved.
    plateau = c[7]
    capacity = plateau - concentration

    # Tunable-order saturation (capacity) factor.  Amyloid sigmoids are
    # markedly ASYMMETRIC: a long flat lag, a steep rise, then a soft approach
    # to the plateau.  A linear capacity (plateau - c)^1 forces a fixed,
    # symmetric saturation elbow and is the main reason the slope-profile /
    # half-response shape loss stays high.  Raising the capacity to a fitted
    # order (1 + c8^2) >= 1 lets each dataset set how sharply growth shuts off
    # near the plateau, decoupling the late-phase curvature from the early
    # inflection sharpness controlled by the secondary-nucleation order.  Since
    # c stays in [0, plateau] the base (plateau - c) >= 0 and the exponent
    # >= 1, so the power is real, finite, smooth, and vanishes at the plateau,
    # preserving mass conservation and odeint stability.
    capacity_term = capacity ** (1 + c[8] ** 2)

    baseline = c[6] ** 2
    primary = c[0] ** 2 * monomer ** c[1]
    elongation = c[2] ** 2 * seed * monomer * concentration
    secondary = c[3] ** 2 * monomer ** c[4] * concentration ** (1 + c[5] ** 2)
    growth = baseline + primary + elongation + secondary

    expression = capacity_term * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 0.5, 0.8, 0.5, 1.0, 0.01, 1.0, 0.0],
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
