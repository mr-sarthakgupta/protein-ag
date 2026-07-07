# EVOLVE-BLOCK-START
"""Diverse inhibited ODE seed for normalized Abeta42 aggregation kinetics."""

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
    """Seed-additive autocatalysis with a single STATE-DEPENDENT inhibitor gate.

    Breakthrough structure targeting the frozen high-quantile shape loss on
    strongly-inhibited unseeded curves (cd=2e-6: 50% at t~3.0, 90% at t~5.4).

    Research grounding (Cohen/Knowles/Meisl Abeta42 master-equation kinetics):
    aggregate mass grows via a monomer-dependent primary channel that breaks
    the lag plus a monomer-dependent secondary-nucleation channel on existing
    fibril surface (autocatalytic ~ monomer*aggregate) that sets the sharp
    rise and half-time. Surface-catalysis / secondary-nucleation inhibitors
    (e.g. chaperone- or small-molecule-class) act by DELAYING the reactive
    flux and shifting crossing times later, with potency that is largest when
    little aggregate/surface exists and is progressively titrated out as
    fibrils accumulate. That is a state-dependent gate, not a constant
    amplitude rescaling -- external paper/web tools returned no usable hits, so
    this structure is grounded in that established mechanistic principle.

    The prior best used two constant Langmuir amplitude factors 1/(1+c^2*cd),
    which only rescale the rate uniformly in time -- they cannot bend the
    crossing-time profile, so the extreme inhibited lag stayed mis-timed.

    Physical picture (mass-conserving reduction):
      * Lag-breaking primary channel c0^2*monomer and self-catalytic secondary
        channel c1^2*monomer*(concentration + c2^2*seed). With M0=0 the
        secondary channel starts near zero so the primary channel alone breaks
        the lag; with M0>0 the additive seed boosts the effective autocatalytic
        state from t=0, collapsing the lag -- the seeded/unseeded contrast.
      * A SINGLE state-dependent gate multiplies the WHOLE rate:
            gate = 1 / (1 + c4^2 * inhibitor / (c5^2 + concentration)).
        When little fibril exists (small concentration) the inhibitory burden
        c4^2*cd/(c5^2+c) is LARGE, so gate << 1 and the curve is nearly frozen
        -> a long lag. As product accumulates the inhibitor is titrated out
        (denominator grows), gate -> 1, and the curve accelerates. This bends
        crossing times later without merely lowering amplitude, directly
        attacking the worst-curve shape term while leaving cd=0 exactly
        unchanged (gate == 1).

    Stability: c5^2 + concentration >= 0 with concentration >= 0 during
    integration; the outer "+1" guarantees the gate stays finite and bounded in
    (0, 1] even if c5 fits to ~0, so the RHS is singular-free, smooth, and
    odeint-stable. capacity = c3 - concentration is the self-limiting sink
    (c3 may exceed 1 so capacity stays positive on normalized data). All rate
    coefficients are squared for strict positivity. Six constants keep the
    least-squares fit well-conditioned.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    gate = 1 / (1 + c[4] ** 2 * inhibitor / (c[5] ** 2 + concentration))
    primary = c[0] ** 2 * monomer
    secondary = c[1] ** 2 * monomer * (concentration + c[2] ** 2 * seed)
    capacity = c[3] - concentration
    expression = capacity * (primary + secondary) * gate

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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