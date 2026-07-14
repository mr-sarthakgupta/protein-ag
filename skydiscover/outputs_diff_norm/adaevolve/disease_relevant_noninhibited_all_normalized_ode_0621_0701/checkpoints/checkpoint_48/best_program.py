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

    # SMOOTH COOPERATIVE-ONSET GATE (tanh switch on the autocatalytic burst).
    # The dominant remaining error is the kinetic SHAPE loss, not pointwise NMSE:
    # amyloid fronts are markedly ASYMMETRIC (long flat lag, steep burst, soft
    # plateau).  A fixed-order capacity power cannot sharpen the lag -> burst
    # transition the slope-profile / half-response shape loss rewards.  Instead
    # we gate ONLY the autocatalytic (secondary) route with a smooth state-driven
    # switch that is ~0 during the lag (small aggregate signal) and ~1 after the
    # cooperative threshold is crossed:
    #     gate = (1 + tanh(c8^2 * (c + c6^2*M0 - c5))) / 2  in [0, 1].
    # tanh is bounded in [-1, 1], so gate is finite, smooth, and overflow-free
    # for ANY fitted c8 (no Max/Min/Heaviside used).  c8^2 >= 0 sets the burst
    # sharpness; c5 sets the inflection location (half-response timing); and the
    # seed term c6^2*M0 ADVANCES the threshold crossing for seeded curves, moving
    # their half-response earlier exactly as the timing component of shape loss
    # measures.  Crucially the nucleation SOURCE is left UNGATED so the c = 0
    # fixed point is still broken and every trajectory leaves the lag phase.
    gate = (1 + sp.tanh(c[8] ** 2 * (concentration + c[6] ** 2 * seed - c[5]))) / 2

    # Mechanistic moment terms (Knowles/Cohen master equation), all multiplied by
    # the convertible-mass capacity (plateau - c) so growth self-bounds at the
    # plateau (mass conservation) and the RHS stays finite and smooth.
    #   * primary    : monomer-driven nucleation c0^2 * m0^c1, the only
    #                  concentration-independent source; a fitted power of m0
    #                  repositions the lag-phase onset across datasets.
    #   * elongation : linear seeded growth c2^2 * c, so any present aggregate
    #                  grows immediately.
    #   * secondary  : autocatalytic self-catalysis c3^2 * m0^c4 * c, sharpened
    #                  by the cooperative tanh gate for the steep burst.
    # Constants enter squared so every rate is non-negative; powers act only on
    # positive m0 and capacity >= 0, keeping least-squares and odeint stable.
    primary = c[0] ** 2 * monomer ** c[1]
    elongation = c[2] ** 2 * concentration
    secondary = c[3] ** 2 * monomer ** c[4] * concentration * gate
    growth = primary + elongation + secondary

    expression = capacity * growth

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 0.5, 0.8, 0.5, 0.3, 0.5, 1.0, 2.0],
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
