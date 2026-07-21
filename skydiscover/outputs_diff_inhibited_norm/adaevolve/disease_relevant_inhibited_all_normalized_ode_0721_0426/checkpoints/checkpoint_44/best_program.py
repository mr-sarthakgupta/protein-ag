# EVOLVE-BLOCK-START
"""Inhibited amyloid ODE: surface-gated secondary nucleation with a dose-dependent maturation time-gate (delay + broadening) for inhibitor shape."""

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
    """Compact Knowles/Meisl amyloid mass-balance RHS whose inhibitor action
    both DELAYS and BROADENS the sigmoidal rise in a dose-dependent way.

    Fibril mass c(t) grows by (i) monomer/seed-fed primary nucleation +
    elongation and (ii) surface-catalysed secondary nucleation that is
    autocatalytic in c. Mass conservation is enforced by the capacity factor
    (plateau - c) so the net flux vanishes as c saturates, giving the sharp
    lag -> rise -> plateau sigmoid.

    Molecular inhibition acts on the surface-catalysed channel two ways, both
    exactly unity at cd = 0 so the uninhibited kinetics are recovered:
      * a bounded Langmuir RATE gate 1/(1 + K1*cd) suppresses the secondary
        flux (fibril-surface coating), and
      * a cooperative maturation TIME-gate t^p/(theta^p + t^p) with an
        inhibitor-LENGTHENED half-time theta AND an inhibitor-REDUCED
        cooperativity p (dose broadens the rise). Because the parent had no
        time forcing, its autonomous sigmoid could not move the inflection/
        crossing time per dose; this gate directly delays WHEN the
        autocatalytic burst ignites and how BROAD the rise is, targeting the
        10/25/50/75/90% response-crossing terms that dominate the shape loss
        (the current bottleneck; NMSE is already ~0.002). Primary nucleation +
        elongation stay ungated so the c = 0 ignition baseline and seeded/
        unseeded early behaviour survive.

    Numerical safety: sqrt(c^2 + c0^2) is a strictly-positive, smooth surface
    measure differentiable at c = 0; every rate constant is squared for
    positivity; all denominators are 1 + (nonnegative) or theta^p + t^p > 0
    with x0 in [0,1] and p >= 1, so the RHS is smooth, bounded, and the
    12-constant least-squares fit stays well conditioned from one start.

    Features: x0=time, x1=m0 monomer, x2=M0 seed, x3=cd inhibitor, x4=state c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(12)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Smooth, bounded self-replication measure (fibril surface ~ mass).
    smooth_state = sp.sqrt(concentration ** 2 + c[0] ** 2)
    # Langmuir RATE suppression of the secondary channel by bound inhibitor.
    surface_gate = 1 / (1 + c[1] ** 2 * inhibitor)
    # COOPERATIVE (Hill-power) maturation TIME-gate. A Hill form
    # time^p/(theta^p + time^p) makes the gate FLAT near t=0 (extending the
    # observed lag) then rise STEEPLY through its half-time, matching the sharp
    # sigmoid morphology of seeded amyloid mass curves.
    # The half-time theta = c6^2*(1 + c7^2*cd) is inhibitor-lengthened so
    # higher dose delays WHEN the burst fires; at cd=0 the dose term vanishes
    # and the uninhibited timing is recovered exactly.
    #
    # DOSE-DEPENDENT COOPERATIVITY (broadening). The shape loss is now the
    # bottleneck (NMSE ~0.0024 vs shape ~0.166), and its high-quantile term is
    # driven by a few visibly wrong high-dose inhibited curves. The measured
    # curves show two coupled dose effects: the rise is not only DELAYED but
    # markedly BROADENED at high inhibitor (the highest-dose curves have much
    # lower AUC and a drawn-out sigmoid). A single fixed cooperativity
    # p = 1 + c9^2 delays WHEN the burst fires (via theta) but keeps the
    # rise-width fixed, so it cannot match both the sharp uninhibited sigmoid
    # AND the broad, poisoned high-dose rise. Here the exponent
    # p = 1 + c9^2/(1 + c11^2*cd) is HIGH (sharp, cooperative) at cd=0 and
    # DECREASES toward 1 (broad, drawn-out, less-cooperative) as dose grows --
    # exactly the observed dose-dependent broadening from a poisoned surface-
    # catalysis cascade. At cd=0 the dose term vanishes so the sharp
    # uninhibited timing is recovered exactly. Because time = x0 in [0, 1],
    # theta > 0 and p >= 1, both time^p and theta^p are finite, strictly
    # nonnegative, and the denominator theta^p + time^p > 0, so the gate stays
    # smooth, real, and bounded in [0, 1) with no singularities.
    hill_exponent = 1 + c[9] ** 2 / (1 + c[11] ** 2 * inhibitor)
    theta = c[6] ** 2 * (1 + c[7] ** 2 * inhibitor)
    maturation = time ** hill_exponent / (theta ** hill_exponent + time ** hill_exponent)

    # PRIMARY-nucleation inhibitor gate. Unseeded curves (which must fire
    # primary nucleation before any surface exists) are delayed by inhibitor,
    # whereas SEEDED curves - which bypass primary nucleation via the
    # pre-formed seed - are far less affected. Mechanistically the inhibitor
    # sequesters free monomer and coats nascent primary nuclei, so primary
    # nucleation is strongly suppressed. The quadratic dose term is dropped
    # here (its constant slot c11 now drives the dose-dependent maturation
    # broadening, a higher-payoff use for the worst-curve shape term); a single
    # Langmuir dose term still delays the unseeded high-dose rise and keeps the
    # c=0 baseline exact.
    primary_gate = 1 / (1 + c[10] ** 2 * inhibitor)

    # Surface-catalysis SATURATION of the finite pool of secondary-nucleation
    # sites. A single surface-site saturation 1 + K3*c self-limits the
    # autocatalytic flux as fibril mass grows; the denominator is
    # 1 + (nonnegative) > 0 so it stays smooth and bounded.
    site_saturation = 1 + c[8] ** 2 * concentration

    # Monomer-fed PRIMARY nucleation is inhibitor-gated (unseeded curves are
    # delayed by inhibitor); the SEED-fed channel bypasses primary nucleation
    # so it stays ungated (seeded curves are barely delayed by inhibitor).
    # Both are exact at cd = 0, preserving the uninhibited limit.
    nucleation = c[2] ** 2 * monomer * primary_gate + c[3] ** 2 * seed
    # Autocatalytic secondary nucleation: monomer-fed, surface-catalysed,
    # rate- and time-gated by the inhibitor, with surface-site saturation.
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
        initial_values=[0.05, 1.0, 0.5, 0.01, 1.0, 1.0, 0.3, 0.1, 0.2, 0.7, 0.5, 0.3],
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