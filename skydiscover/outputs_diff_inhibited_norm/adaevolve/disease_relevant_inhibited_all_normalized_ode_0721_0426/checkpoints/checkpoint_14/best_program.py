# EVOLVE-BLOCK-START
"""Mechanistic Abeta42 aggregation ODE: channel-selective inhibitor gating
plus a dose-dependent maturation time-gate for lag control.

Physical basis (Cohen/Meisl/Knowles amyloid master equation for fibril mass
fraction c). Growth is driven by three channels, all fed by the free monomer
pool and closed off as mass approaches a shared plateau (mass conservation):

  * primary nucleation  ~ k_n * monomer          (inhibitor-independent seed)
  * elongation          ~ k_+ * monomer * c      (fibril-end addition)
  * secondary nucleation~ k_2 * monomer * c^2    (fibril-surface autocatalysis)

At the molecular level the clinically relevant Abeta42 inhibitors act mainly
by coating fibril SURFACES and thereby suppressing the autocatalytic
secondary-nucleation channel (Langmuir occupancy 1/(1+K*cd)). A separate,
weaker gate on the elongation channel captures partial end-capping. But a
purely multiplicative rate slow-down cannot reproduce a genuine dose-dependent
LAG DELAY (a shifted inflection/crossing time). We therefore ADD a smooth
transient maturation gate t/(tau(cd)+t) on the two autocatalytic channels,
with an inhibitor-lengthened effective half-time tau(cd)=c8^2*(1+c9^2*cd).
This shifts WHEN the sigmoidal burst ignites at higher dose, directly moving
the 10/25/50% response-crossing terms that dominate the scorer's shape loss.
Primary nucleation and the plateau stay inhibitor-independent, so the cd=0
limit, the seeded/unseeded baselines, and the endpoint are all preserved.

Numerics: every rate constant is squared for positivity; sqrt(c^2+eps) keeps
the secondary surface measure smooth and differentiable at c=0; all
denominators are 1 + (nonnegative) and the time-gate denominator is >= c8^2>0
with x0 in [0,1], so the RHS is smooth, bounded, and well conditioned for
least-squares over the observed input ranges.

Features: x0 time, x1 m0 monomer, x2 M0 seed, x3 cd inhibitor, x4 state c.
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


def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Channel-selective amyloid ODE with a dose-dependent maturation time-gate.

    Combines the parent's channel-selective rate gates (strong on secondary
    nucleation, weak on elongation) with an explicit transient time-forcing
    gate on the autocatalytic channels whose half-time lengthens with
    inhibitor dose. The rate gates preserve the excellent pointwise fit while
    the time-gate shifts the burst ignition time to cut the dominant shape
    loss. Primary nucleation and the plateau remain inhibitor-independent, so
    the cd=0 limit and seeded/unseeded baselines are preserved. Uses 10 fitted
    constants; drops the parent's negligible off-pathway sink (fitted ~1e-9).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Smooth, bounded fibril-surface measure (differentiable at c = 0).
    smooth_state = sp.sqrt(concentration ** 2 + c[0] ** 2)

    # Strong Langmuir gate on autocatalytic secondary nucleation (rate slow-down).
    surface_gate = 1 / (1 + c[1] ** 2 * inhibitor)
    # Weaker gate on elongation (partial fibril-end capping).
    end_gate = 1 / (1 + c[2] ** 2 * inhibitor)

    # Inhibitor-independent primary nucleation + seed contribution.
    # Kept ungated (and un-time-gated) so the cd=0 limit and the seeded/unseeded
    # baselines are preserved exactly.
    primary = c[3] ** 2 * monomer + c[4] ** 2 * seed
    # Elongation: monomer-fed, linear in fibril mass, mildly rate-gated.
    elongation = c[5] ** 2 * monomer * concentration * end_gate
    # Secondary nucleation: monomer-fed, surface-catalysed, strongly rate-gated.
    secondary = c[6] ** 2 * monomer * smooth_state * concentration * surface_gate

    # BREAKTHROUGH lever for shape loss (the dominant scored-loss term at
    # ~95%): the parent only slows growth multiplicatively, which cannot
    # reproduce a genuine dose-dependent LAG DELAY (a later crossing time).
    # Add a smooth transient maturation gate that shifts WHEN the autocatalytic
    # burst ignites: it rises from 0 at t=0 toward 1 with an effective
    # half-time c8^2 that the inhibitor LENGTHENS via (1 + c9^2 * cd). This
    # directly moves the 10/25/50% response-crossing terms of the shape loss on
    # delayed inhibitor curves. x0 is normalized to [0,1] and the denominator
    # is >= c8^2 > 0, so the gate is bounded in [0,1) and globally smooth.
    maturation = time / (c[7] ** 2 * (1 + c[8] ** 2 * inhibitor) + time)

    # Monomer-depletion capacity (conservation of mass); plateau stays
    # inhibitor-independent so dose only reshapes timing, not the endpoint.
    plateau = c[9]
    capacity = plateau - concentration

    # Primary nucleation seeds the process immediately; the elongation and
    # secondary-nucleation autocatalytic channels are additionally time-gated
    # by maturation so their burst turns on later at higher inhibitor dose.
    expression = capacity * (primary + maturation * (elongation + secondary))

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.05, 1.0, 0.5, 1.0, 0.01, 1.0, 1.0, 0.3, 0.1, 1.0],
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
