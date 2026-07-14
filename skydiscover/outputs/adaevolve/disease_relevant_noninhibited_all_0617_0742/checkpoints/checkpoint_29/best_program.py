# EVOLVE-BLOCK-START
"""Symbolic regression seed for normalized multi-dataset amyloid aggregation kinetics."""

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
    Stretched-exponential (Johnson-Mehl-Avrami-Kolmogorov / Kohlrausch)
    approach-to-plateau template for nucleation-and-growth kinetics.

    The JMAK law for nucleation-and-growth gives the converted fraction
    f(t) = 1 - exp(-(k*t)^beta), a stretched exponential that produces a
    lag phase (Avrami exponent beta > 1), a sharp elongation, and a smooth
    saturating plateau. This is structurally distinct from the symmetric
    logistic and is the natural mechanistic law for seeded amyloid
    polymerization, where it must generalize the saturating approach and lag
    to unseen cross-protein datasets.

    The same template is used for every dataset; constants are fitted
    independently per dataset. Features: x0 = normalized elapsed time,
    x1 = m0 initial monomer concentration, x2 = static initial M0 seed.

    External research note: targeted research_papers and web_search lookups
    on this run returned no usable amyloid-kinetics content (and the cached
    JBC reference page was empty), so the structure is justified from the
    established mechanistic Avrami/JMAK nucleation-growth law plus archive
    evidence on the response shape rather than from a copied named formula.

    Numerical-stability design - the whole exponent argument is provably
    nonnegative and finite, so 1 - exp(-arg) always lies in [0, 1):
      * rate prefactor c0**2 is nonnegative (a square), never negative.
      * monomer**c1: monomer concentrations are positive, so a real power is
        always finite and positive.
      * time**beta with beta = exp(c3) > 0: x0 in [0, 1], so x0**(positive)
        stays in [0, 1] - never an imaginary/negative-base power. This is the
        Avrami stretch exponent that creates the lag (beta>1) or fast initial
        rise (beta<1).
      * seed acceleration uses exp(c5*x2), which is strictly positive for any
        real c5 and any x2 (including x2 = 0, giving factor 1), so it can only
        scale the positive rate up or down without ever turning it negative.
    Therefore arg = c0**2 * monomer**c1 * x0**beta * exp(c5*x2) >= 0 always,
    exp(-arg) in (0, 1], and growth in [0, 1). No singularities, no
    non-real values, no overflow (arg is bounded by finite positive factors).

    Initialization: c3 = 0 -> beta = 1 (pure exponential rise), c5 = 0 ->
    seed factor 1. The optimizer then introduces lag curvature (beta != 1)
    and seed dependence only where it improves the fit.

        y = c4 * (1 - exp(-c0^2 * x1^c1 * x0^exp(c3) * exp(c5*x2))) + c6
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[0] ** 2 * monomer ** c[1] * sp.exp(c[5] * seed)
    stretch = sp.exp(c[3])
    plateau = c[4]
    baseline = c[6]

    growth = 1 - sp.exp(-rate * time ** stretch)
    expression = plateau * growth + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
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