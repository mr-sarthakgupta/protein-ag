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
    """Autocatalytic self-accelerating lag ODE with inhibitor delaying feedback.

    Grounded in the Cohen/Knowles master-equation reduction for Abeta42 fibril
    formation: the aggregate mass fraction is a sigmoid driven by (i) a small
    monomer-fed primary-nucleation source that breaks the lag without any
    explicit time term (preserving autonomy), and (ii) an autocatalytic
    secondary-nucleation feedback proportional to the existing aggregate
    concentration, which produces the sharp rise and sets the half-time.

    Surface-binding inhibitors act mechanistically by suppressing the
    secondary/autocatalytic pathway, which DELAYS the inflection (crossing
    times) rather than only lowering amplitude. Accordingly the inhibitor
    divides ONLY the autocatalytic feedback via 1 + c4^2*cd + c5^2*cd*c, so
    higher cd slows feedback build-up and pushes the 10/25/50/75/90% crossing
    times later (directly targeting the shape loss and its worst-curve term).
    At cd=0 the inhibitor factor collapses to 1, recovering the uninhibited law.

    capacity = c6 - c is a mass-conserving self-limiting sink (c6 may exceed 1
    so capacity stays positive on normalized data). All rate coefficients are
    squared for strict positivity, keeping odeint smooth and stable; the
    concentration**2 feedback cannot overflow because capacity drives c toward
    the plateau. Seven constants keep least-squares fitting reliable.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    source = c[0] ** 2 + c[1] ** 2 * monomer
    feedback = c[2] ** 2 * monomer * (1 + c[3] ** 2 * seed)
    feedback_eff = feedback / (
        1 + c[4] ** 2 * inhibitor + c[5] ** 2 * inhibitor * concentration
    )
    capacity = c[6] - concentration
    expression = capacity * (source + feedback_eff * concentration)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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