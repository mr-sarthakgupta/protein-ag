"""Top-10 AdaEvolve best programs by child combined_score.

Source run: disease_relevant_noninhibited_all_normalized_ode_0621_0701

Each function is renamed evaluate_symbolic_candidate_iter_<N>.
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

TOP10_ITERATIONS = [24, 36, 29, 32, 15, 9, 14, 11, 7, 8]

ITERATION_METADATA = {
    24: {
        "combined_score": 0.8894979983914554,
        "global_best_score": 0.8894979983914554,
        "program_id": '323b2853-3188-433f-bf54-634105881a02',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_24/best_program.py',
    },
    36: {
        "combined_score": 0.8894979983914554,
        "global_best_score": 0.8894979983914554,
        "program_id": 'd68e83fd-0668-4999-a971-a48fced63282',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_36/best_program.py',
    },
    29: {
        "combined_score": 0.8893426886446029,
        "global_best_score": 0.8894979983914554,
        "program_id": 'fbf3025e-fd23-41d1-bf51-69dbe0b2a98d',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_29/best_program.py',
    },
    32: {
        "combined_score": 0.8893426886446029,
        "global_best_score": 0.8894979983914554,
        "program_id": 'fae93ad4-4f5a-4bc5-82b2-1d56a4b1a506',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_32/best_program.py',
    },
    15: {
        "combined_score": 0.8890160325603259,
        "global_best_score": 0.8890160325603259,
        "program_id": 'd7d32324-5349-4e2e-b210-e9d96bde6927',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_15/best_program.py',
    },
    9: {
        "combined_score": 0.8884941964854783,
        "global_best_score": 0.8884941964854783,
        "program_id": 'a8278379-9af5-4ab4-980d-3b2867c3de76',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_9/best_program.py',
    },
    14: {
        "combined_score": 0.8884941964854783,
        "global_best_score": 0.8884941964854783,
        "program_id": '83b664c1-e674-4398-a85d-303eb2826522',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_14/best_program.py',
    },
    11: {
        "combined_score": 0.8884589520461968,
        "global_best_score": 0.8884941964854783,
        "program_id": 'a1ee62b1-12c1-4e20-bcd5-37115add6c9f',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_11/best_program.py',
    },
    7: {
        "combined_score": 0.8883707191460762,
        "global_best_score": 0.8883707191460762,
        "program_id": '3e535672-47e6-4a55-95fc-8497e67f9322',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_7/best_program.py',
    },
    8: {
        "combined_score": 0.8883707191460762,
        "global_best_score": 0.8883707191460762,
        "program_id": '607f2811-c341-463a-96aa-cdb3e6f18a5b',
        "checkpoint": '/home/mrsar/protein-ag/skydiscover/outputs_diff_norm/adaevolve/disease_relevant_noninhibited_all_normalized_ode_0621_0701/checkpoints/checkpoint_8/best_program.py',
    },
}


# ========================================================================
# Rank 1 | Iteration 24 | combined_score=0.8894979983914554
# program_id=323b2853-3188-433f-bf54-634105881a02
# checkpoint: checkpoints/checkpoint_24/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_24(
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


# ========================================================================
# Rank 2 | Iteration 36 | combined_score=0.8894979983914554
# program_id=d68e83fd-0668-4999-a971-a48fced63282
# checkpoint: checkpoints/checkpoint_36/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_36(
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


# ========================================================================
# Rank 3 | Iteration 29 | combined_score=0.8893426886446029
# program_id=fbf3025e-fd23-41d1-bf51-69dbe0b2a98d
# checkpoint: checkpoints/checkpoint_29/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_29(
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


# ========================================================================
# Rank 4 | Iteration 32 | combined_score=0.8893426886446029
# program_id=fae93ad4-4f5a-4bc5-82b2-1d56a4b1a506
# checkpoint: checkpoints/checkpoint_32/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_32(
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


# ========================================================================
# Rank 5 | Iteration 15 | combined_score=0.8890160325603259
# program_id=d7d32324-5349-4e2e-b210-e9d96bde6927
# checkpoint: checkpoints/checkpoint_15/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_15(
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

    # Mechanistic separation of saturation orders (Knowles/Cohen master eqn):
    # secondary nucleation rate physically scales with the CURRENT free monomer
    # m(t) ~ capacity, not just the fixed initial m0.  Giving the autocatalytic
    # route an extra intrinsic free-monomer factor (capacity) makes it shut off
    # one power faster than primary/elongation near the plateau.  This decouples
    # the steep autocatalytic inflection (which the slope-profile / half-response
    # shape loss penalizes) from the gentler primary onset, sharpening the lag ->
    # burst transition without adding any fitted constant.  capacity >= 0 keeps
    # the extra factor real, finite, smooth, and vanishing at the plateau.
    primary = c[0] ** 2 * monomer ** c[1]
    elongation = c[2] ** 2 * (1 + c[6] ** 2 * seed) * concentration
    secondary = (
        c[3] ** 2 * monomer ** c[4] * capacity
        * concentration ** (1 + c[5] ** 2)
    )
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


# ========================================================================
# Rank 6 | Iteration 9 | combined_score=0.8884941964854783
# program_id=a8278379-9af5-4ab4-980d-3b2867c3de76
# checkpoint: checkpoints/checkpoint_9/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_9(
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


# ========================================================================
# Rank 7 | Iteration 14 | combined_score=0.8884941964854783
# program_id=83b664c1-e674-4398-a85d-303eb2826522
# checkpoint: checkpoints/checkpoint_14/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_14(
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


# ========================================================================
# Rank 8 | Iteration 11 | combined_score=0.8884589520461968
# program_id=a1ee62b1-12c1-4e20-bcd5-37115add6c9f
# checkpoint: checkpoints/checkpoint_11/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_11(
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


# ========================================================================
# Rank 9 | Iteration 7 | combined_score=0.8883707191460762
# program_id=3e535672-47e6-4a55-95fc-8497e67f9322
# checkpoint: checkpoints/checkpoint_7/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_7(
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


# ========================================================================
# Rank 10 | Iteration 8 | combined_score=0.8883707191460762
# program_id=607f2811-c341-463a-96aa-cdb3e6f18a5b
# checkpoint: checkpoints/checkpoint_8/best_program.py
# ========================================================================

def evaluate_symbolic_candidate_iter_8(
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

