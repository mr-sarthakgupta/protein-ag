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
    Asymmetric tanh-power sigmoid template for nucleation-dependent amyloid
    aggregation, fitted independently per dataset (shared structure, per-dataset
    constants).

    Features: x0 = normalized elapsed time, x1 = m0 initial monomer
    concentration, x2 = static initial M0 seed/aggregate concentration.

    Structure (one expression, one scorer call):
        u = c0 * x1^c1 * (x0 - c2) + log1p(c3 * x2)
        y = c4 * (0.5 + 0.5 * tanh(u))^c6 + c5

    Backbone: a single tanh-based bounded sigmoid replaces the Gompertz/Richards
    exponential-of-exponential. s = 0.5 + 0.5*tanh(u) is in (0, 1], infinitely
    smooth, and never overflows for finite u, giving far better least-squares
    conditioning on the small/noisy single-curve evaluation-only datasets where
    the Gompertz inner exponent could blow up. A single fitted power exponent c6
    restores tunable asymmetry (c6=1 symmetric; c6>1 sharp rise / slow upper
    tail = secondary-nucleation signature; 0<c6<1 the opposite), so the model
    is NOT the symmetric tanh that previously underfit the slow upper tail.

    Numerical safety: base s in (0, 1] -> s**c6 is real and finite for any real
    c6 (no negative base, no log-of-negative, no division, no overflow). The
    additive seed term log1p(c3*x2) has argument 1 + c3*x2 >= 1 > 0 for x2 >= 0,
    c3 >= 0, and reduces to 0 when x2 = 0 so unseeded curves keep the seedless
    form. 7 fitted constants.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    rate = c[0] * monomer ** c[1]
    half_time = c[2]
    seed_shift = sp.log(1 + c[3] * seed)
    plateau = c[4]
    baseline = c[5]

    u = rate * (time - half_time) + seed_shift
    growth = (0.5 + 0.5 * sp.tanh(u)) ** c[6]
    expression = plateau * growth + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.0, 0.3, 0.1, 1.0, 0.0, 1.0],
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