"""Ground-truth secondary-nucleation model for evaluator scoring."""

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
    """Evaluate the closed-form manual secondary-nucleation solution."""
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    m0 = x[1]
    M0 = x[2]

    eps = 1.0e-12
    kappa = c[0] ** 2 + eps
    lam = c[1] ** 2 + eps
    seed_term = c[2] ** 2 + eps
    n2 = c[3] ** 2 + eps
    nc = c[4] ** 2 + eps

    mtot = m0 + M0
    concentration_ratio = M0 / m0
    exp_kappa_t = sp.exp(kappa * time)

    Cp = seed_term + concentration_ratio / 2 + lam**2 / (2 * kappa**2)
    Cm = seed_term - concentration_ratio / 2 - lam**2 / (2 * kappa**2)
    kinf = kappa * sp.sqrt(
        2 / (n2 * (n2 + 1))
        + 2 * lam**2 / (nc * kappa**2)
        + 2 * concentration_ratio / n2
        + (2 * seed_term) ** 2
    )
    kbarinf = sp.sqrt(kinf**2 - 4 * Cp * Cm * kappa**2)
    Bp = (kinf + kbarinf) / (2 * kappa)
    Bm = (kinf - kbarinf) / (2 * kappa)

    manual_response = 1 - (1 - M0 / mtot) * (
        ((Bp + Cp) * (Bm + Cp * exp_kappa_t))
        / ((Bp + Cp * exp_kappa_t) * (Bm + Cp))
    ) ** (kinf**2 / (kbarinf * kappa)) * sp.exp(-kinf * time)
    expression = c[5] * manual_response + c[6]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.1, 0.1, 1.41421356237, 1.41421356237, 1.0, 0.0],
    )


def run_discovery(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Entry point used by the evaluator subprocess."""
    return evaluate_symbolic_candidate(X_train, y_train, X_val, y_val)
