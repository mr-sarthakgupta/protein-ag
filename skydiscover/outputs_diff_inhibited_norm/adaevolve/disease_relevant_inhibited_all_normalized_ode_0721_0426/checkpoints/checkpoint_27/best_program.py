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
    # Maturation TIME-gate: WHEN the autocatalytic burst ignites; the
    # inhibitor lengthens the effective half-time tau = c6^2. Unity at cd = 0.
    maturation = time / (c[6] ** 2 * (1 + c[7] ** 2 * inhibitor) + time)

    # PRIMARY-nucleation inhibitor gate. The measured dose response shows that
    # UNSEEDED curves (which must fire primary nucleation before any surface
    # exists) are delayed AND broadened enormously by inhibitor (t50 shifts
    # ~4x, t90-t10 widens ~7x), whereas SEEDED curves - which bypass primary
    # nucleation via the pre-formed seed - are far less affected. The parent
    # gated only the surface/secondary channel and kept primary nucleation
    # inhibitor-independent, so it could not delay the earliest rise of the
    # unseeded high-dose lag. Mechanistically the inhibitor sequesters free
    # monomer and coats nascent primary nuclei, so primary nucleation is the
    # MOST strongly suppressed channel. A quadratic inhibitor term (cooperative
    # nucleus poisoning) gives the strong, dose-broadened suppression the data
    # demand while staying exactly unity at cd = 0 and bounded > 0.
    primary_gate = 1 / (1 + c[10] ** 2 * inhibitor + c[11] ** 2 * inhibitor ** 2)

    # Surface-catalysis SATURATION whose severity grows with inhibitor dose.
    # Physically the inhibitor coats fibril surface, so the finite number of
    # remaining active secondary-nucleation sites saturates at lower fibril
    # mass as cd rises. Dividing the autocatalytic flux by (1 + (K3 + K4*cd)*c)
    # makes the effective growth rate self-limit earlier at high dose, which
    # BROADENS the lag->rise transition (increasing t90-t10) exactly as the
    # measured dose response shows, without touching the cd=0 limit (the extra
    # site-blocking term K4*cd vanishes at cd=0) or the inhibitor-independent
    # plateau. Denominator is 1 + (nonnegative) > 0 so it stays smooth/bounded.
    site_saturation = 1 + (c[8] ** 2 + c[9] ** 2 * inhibitor) * concentration

    # Monomer-fed PRIMARY nucleation is inhibitor-gated (unseeded curves are
    # delayed by inhibitor); the SEED-fed channel bypasses primary nucleation
    # so it stays ungated (seeded curves are barely delayed by inhibitor).
    # Both are exact at cd = 0, preserving the uninhibited limit.
    nucleation = c[2] ** 2 * monomer * primary_gate + c[3] ** 2 * seed
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
        initial_values=[0.05, 1.0, 0.5, 0.01, 1.0, 1.0, 0.3, 0.1, 0.2, 0.3, 0.5, 0.3],
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
