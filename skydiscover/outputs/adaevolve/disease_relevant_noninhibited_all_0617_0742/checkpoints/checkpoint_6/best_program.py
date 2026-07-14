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
    Propose an equation structure and let the harness fit its constants.

    This candidate is evaluated independently on each dataset — the same
    equation template is used everywhere, but constants are fitted separately
    per dataset.  This allows a single functional form to capture the
    universal kinetic mechanism while the constants adapt to each protein
    system's specific rates, concentrations, and timescales.

    Features: x0 = normalized elapsed time, x1 = m0 initial monomer
    concentration, x2 = static initial M0 seed/aggregate concentration.
    Units are ignored by the cleaned-data loader; leading numeric values are
    used directly.

    Generalized-logistic (Richards) template with concentration-dependent
    rate, a seed-shifted half-time, and an asymmetry shape exponent — a
    natural model for nucleation-dependent, seeded amyloid polymerization.

    Structure carried over from the proven logistic: the growth rate scales
    as a power law of the initial monomer concentration (x1), reflecting the
    concentration dependence of aggregation kinetics, and the half-time is an
    affine function of the static seed concentration (x2),
    half_time = c2*x1^c3 - c6*x2. The affine (not power-law) seed term is used
    because x2 is zero for many curves; a power x2^k would create
    singularities/non-real values, whereas a linear term stays smooth and
    finite everywhere and recovers the proven structure when c6 = 0.

    New structural element: amyloid ThT growth curves are typically
    ASYMMETRIC — the slow nucleation-dominated lag and the fast elongation /
    plateau approach are not mirror images, which a plain logistic cannot
    represent. The Richards shape exponent c7 breaks this symmetry with a
    single extra constant by raising the logistic denominator to a power.
    This is numerically safe: the base (1 + exp(z)) is always >= 1 > 0, so
    raising it to any real power c7 is always finite, real, and positive — no
    singularities or non-real values. With c7 = 1 the form reduces exactly to
    the validated logistic, so the optimizer (started at c7 = 1) can only
    improve on the previous best.

        y = c4 / (1 + exp(-c0*x1^c1 * (x0 - (c2*x1^c3 - c6*x2))))**c7 + c5
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[0] * monomer ** c[1]
    half_time = c[2] * monomer ** c[3] - c[6] * seed
    plateau = c[4]
    baseline = c[5]
    shape = c[7]

    growth = 1 / (1 + sp.exp(-rate * (time - half_time))) ** shape
    expression = plateau * growth + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 0.0, 0.0, 1.0],
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
