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
    Generic two-channel bounded growth template.

    This seed is intentionally broad rather than tied to one closed-form law:
    one channel is a lag-capable stretched rise, the other is an immediate
    seed-sensitive rise.  A smooth seed-dependent weight lets the fitted
    constants interpolate between unseeded, lag-dominated curves and seeded,
    fast-onset curves while preserving one shared symbolic structure across
    all datasets.

    Features: x0 = normalized elapsed time, x1 = initial monomer m0, and
    x2 = static seed/aggregate M0.  All sub-curves are bounded in [0, 1) on
    the observed domain, then an affine output calibration handles per-dataset
    response normalization.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    lag_rate = c[0] ** 2 * monomer ** c[1]
    fast_rate = c[2] ** 2 * monomer ** c[3] * sp.exp(c[4] * seed)
    stretch = sp.exp(c[5])
    weight = 1 / (1 + sp.exp(-(c[6] + c[7] * seed)))
    plateau = c[8]
    baseline = c[9]

    lagged_growth = 1 - sp.exp(-lag_rate * time ** stretch)
    fast_growth = 1 - sp.exp(-fast_rate * time)
    expression = plateau * ((1 - weight) * lagged_growth + weight * fast_growth) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 1.0, 0.5, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
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
