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

    # Autonomous, mass-conserving Knowles/Cohen amyloid moment front.  No
    # explicit-time forcing is used: the parent's c8*time*capacity drift term
    # extrapolates poorly to held-out curves, so the whole sigmoid is driven
    # only by the convertible-mass capacity (plateau - c) -> 0 at the plateau.
    # This keeps the trajectory self-bounding and conserving convertible mass.
    #
    #   * primary    : monomer-driven primary nucleation c0^2 * m0^c1, the only
    #                  concentration-independent source.  Letting its strength
    #                  scale with a fitted power of m0 (rather than a fixed
    #                  constant) repositions the lag-phase ONSET across datasets,
    #                  which is exactly the half-response timing the shape loss
    #                  measures and a constant source cannot move.
    #   * elongation : seeded growth c2^2 * (1 + c6^2 * M0) * c, LINEAR in c, so
    #                  any seed M0 > 0 gives an immediate fast onset; the
    #                  (1 + c6^2*M0) factor makes seeded curves rise sooner than
    #                  unseeded ones without a separate additive term.
    #   * secondary  : autocatalytic self-catalysis c3^2 * m0^c4 * c^(1 + c5^2).
    #                  The fitted order in c (>= 1 via 1 + c5^2) is the dominant
    #                  control over the inflection sharpness and onset timing.
    #
    # Constants enter squared so every kinetic rate is non-negative; powers act
    # only on the positive feature m0 and on c in [0, plateau] >= 0 with
    # exponent >= 1, so the RHS is real, finite, and smooth with no
    # singularities, keeping least-squares and odeint numerically well behaved.
    plateau = c[7]
    capacity = plateau - concentration

    # Tunable-order saturation (capacity) factor.  Amyloid sigmoids are markedly
    # ASYMMETRIC: a long flat lag, a steep rise, then a soft approach to the
    # plateau.  A linear capacity (plateau - c)^1 forces a fixed, symmetric
    # saturation elbow and is the main reason the slope-profile / half-response
    # shape loss stays high.  Raising capacity to a fitted order (1 + c8^2) >= 1
    # lets each dataset set how sharply growth shuts off near the plateau,
    # decoupling late-phase curvature from the early inflection sharpness set by
    # the secondary-nucleation order.  Base (plateau - c) >= 0 with exponent
    # >= 1, so the power is real, finite, smooth, and vanishes at the plateau,
    # preserving mass conservation and odeint stability.
    capacity_term = capacity ** (1 + c[8] ** 2)

    primary = c[0] ** 2 * monomer ** c[1]
    elongation = c[2] ** 2 * (1 + c[6] ** 2 * seed) * concentration
    secondary = c[3] ** 2 * monomer ** c[4] * concentration ** (1 + c[5] ** 2)
    growth = primary + elongation + secondary

    expression = capacity_term * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 0.5, 0.8, 0.5, 1.0, 0.5, 1.0, 0.0],
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
