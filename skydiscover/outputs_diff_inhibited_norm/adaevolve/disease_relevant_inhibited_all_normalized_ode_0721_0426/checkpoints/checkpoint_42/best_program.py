# EVOLVE-BLOCK-START
"""Inhibited amyloid ODE: surface-gated secondary nucleation with a dose-dependent maturation time-gate for lag timing."""

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
    """Compact Knowles/Meisl amyloid mass-balance RHS with a dose-dependent
    maturation time-gate that sets the inhibitor lag timing.

    Fibril mass c(t) grows by (i) monomer/seed-fed primary nucleation +
    elongation and (ii) surface-catalysed secondary nucleation that is
    autocatalytic in c. Mass conservation is enforced by the capacity factor
    (plateau - c) so the net flux vanishes as c saturates, giving the sharp
    lag -> rise -> plateau sigmoid.

    Molecular inhibition acts on the surface-catalysed channel two ways, both
    exactly unity at cd = 0 so the uninhibited kinetics are recovered:
      * a bounded Langmuir RATE gate 1/(1 + K1*cd) suppresses the secondary
        flux (fibril-surface coating), and
      * a smooth maturation TIME-gate t/(tau*(1 + K2*cd) + t) rises from 0 at
        t = 0 toward 1 with an inhibitor-LENGTHENED half-time. Because the
        parent had no time forcing, its autonomous sigmoid could not move the
        inflection/crossing time per dose; this gate directly delays WHEN the
        autocatalytic burst ignites, targeting the 10/25/50% response-crossing
        terms that dominate the shape loss (the current bottleneck; NMSE is
        already ~0.003). Primary nucleation + elongation stay ungated so the
        c = 0 ignition baseline and seeded/unseeded early behaviour survive.

    Numerical safety: sqrt(c^2 + c0^2) is a strictly-positive, smooth surface
    measure differentiable at c = 0; every rate constant is squared for
    positivity; all denominators are 1 + (nonnegative) or tau*(...) + t > 0
    with x0 in [0,1], so the RHS is smooth, bounded, and the 8-constant
    least-squares fit stays well conditioned from one start.

    Features: x0=time, x1=m0 monomer, x2=M0 seed, x3=cd inhibitor, x4=state c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Smooth, bounded self-replication measure (fibril surface ~ mass).
    smooth_state = sp.sqrt(concentration ** 2 + c[0] ** 2)
    # Langmuir RATE suppression of the secondary channel by bound inhibitor.
    surface_gate = 1 / (1 + c[1] ** 2 * inhibitor)
    # COOPERATIVE maturation TIME-gate with a DOSE-DEPENDENT sharpness. The
    # measured response crossings show two coupled dose effects on the
    # secondary-nucleation burst: (i) the half-time t50 LENGTHENS with cd
    # (0.97 -> 2.76 as cd rises), and (ii) the transition WIDTH t90-t10
    # BROADENS strongly with cd (~1.2 at low dose -> ~4.4 at the highest
    # dose). A single Michaelis gate t/(theta+t) cannot reproduce both: it
    # fixes the shape and only shifts theta. Here the half-time
    # theta = c6^2*(1 + c7^2*cd) lengthens with dose, while the Hill exponent
    # p = 1 + c8^2/(1 + c9^2*cd) is HIGH (sharp, cooperative burst) at cd = 0
    # and DECREASES toward 1 (broad, drawn-out rise) as cd grows -- exactly
    # the observed dose-dependent broadening from a poisoned, less-cooperative
    # surface-catalysis cascade. Because time = x0 >= 0 and theta > 0, both
    # time^p and theta^p are finite, strictly nonnegative, and the denominator
    # theta^p + time^p > 0, so the gate is smooth, real, bounded in [0, 1),
    # and reduces to the uninhibited limit exactly at cd = 0.
    hill_exponent = 1 + c[8] ** 2 / (1 + c[9] ** 2 * inhibitor)
    theta = c[6] ** 2 * (1 + c[7] ** 2 * inhibitor)
    maturation = time ** hill_exponent / (theta ** hill_exponent + time ** hill_exponent)

    # Surface-catalysis site SATURATION: the finite pool of secondary
    # nucleation sites self-limits the autocatalytic flux as fibril mass
    # grows. The dose-dependent broadening role now lives in the maturation
    # cooperativity, so this term is a single inhibitor-independent
    # 1 + K10*c; denominator is 1 + (nonnegative) > 0, smooth and bounded.
    site_saturation = 1 + c[10] ** 2 * concentration

    # Monomer/seed-fed primary nucleation + elongation (ungated -> preserves
    # cd=0 limit and seeded/unseeded baselines).
    nucleation = c[2] ** 2 * monomer + c[3] ** 2 * seed
    # Autocatalytic secondary nucleation: monomer-fed, surface-catalysed,
    # rate- and time-gated by the inhibitor, with dose-dependent surface-site
    # saturation controlling the transition WIDTH.
    secondary = c[4] ** 2 * monomer * smooth_state * concentration * surface_gate * maturation / site_saturation

    # Mass-conservation capacity (monomer depletion); plateau stays
    # inhibitor-independent so dose reshapes timing, not the endpoint.
    plateau = c[5]
    capacity = plateau - concentration

    expression = capacity * (nucleation + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.05, 1.0, 0.5, 0.01, 1.0, 1.0, 0.3, 0.1, 1.0, 0.5, 0.2],
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
