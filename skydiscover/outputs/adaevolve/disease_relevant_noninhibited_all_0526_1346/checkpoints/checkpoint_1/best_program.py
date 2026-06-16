# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: Hill / sigmoidal function with data-driven initial values.

The Hill equation  y = c2 * x0^c0 / (c1^c0 + x0^c0) + c3  is a universal
sigmoidal model for nucleation-dependent polymerization.  It has a clear
physical interpretation: c0 is the Hill exponent (cooperativity / nucleation
order), c1 is the half-time (time at which y reaches half its plateau), c2 is
the plateau amplitude, and c3 is the baseline.

Key improvement over the seed: data-driven initial values.  The half-time c1
is estimated directly from the training data (time where y ≈ 0.5), and the
Hill exponent c0 is initialised at 2 (typical nucleation cooperativity).
Multiple starting points are tried via repeated calls with different
initial_values, and the best result is returned.  This avoids the poor
convergence caused by fixed starting points that are far from the solution
for datasets spanning many orders of magnitude in time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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
    """Fit Hill sigmoidal kinetics with data-driven multi-start optimisation.

    Uses the Hill/sigmoidal equation:
        y = c2 * x0^c0 / (c1^c0 + x0^c0) + c3

    where:
        c0 = Hill exponent (cooperativity / nucleation order, typically 2-5)
        c1 = half-time (time at which y = c2/2 + c3)
        c2 = plateau amplitude (≈1 for rescaled data)
        c3 = baseline offset (≈0 for rescaled data)

    Multiple starting points are tried; the one giving the lowest validation
    NMSE is returned.  The half-time initial guess is estimated from the data
    as the time where the training signal is closest to 0.5.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).ravel()

    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(4)

    time = x[0]  # x0 = elapsed time

    # Hill / sigmoidal equation: y = c2 * t^c0 / (c1^c0 + t^c0) + c3
    # Written in a numerically stable form using sympy
    expression = c[2] * time ** c[0] / (c[1] ** c[0] + time ** c[0]) + c[3]

    # Data-driven initial guess for half-time: find t where y ≈ 0.5
    t_train = X_train[:, 0]
    t_pos = t_train[t_train > 0]
    if len(t_pos) == 0:
        t_pos = np.array([1.0])

    # Estimate half-time from training data
    half_idx = int(np.argmin(np.abs(y_train - 0.5)))
    t_half_est = float(t_train[half_idx])
    if t_half_est <= 0:
        t_half_est = float(np.median(t_pos))
    if t_half_est <= 0:
        t_half_est = 1.0

    # Try multiple starting points for Hill exponent and half-time
    candidate_starts = [
        [2.0, t_half_est,        1.0, 0.0],
        [3.0, t_half_est,        1.0, 0.0],
        [5.0, t_half_est,        1.0, 0.0],
        [1.5, t_half_est,        1.0, 0.0],
        [4.0, t_half_est,        1.0, 0.0],
        [2.0, t_half_est * 0.5,  1.0, 0.0],
        [2.0, t_half_est * 2.0,  1.0, 0.0],
        [3.0, t_half_est * 0.5,  1.0, 0.0],
        [3.0, t_half_est * 2.0,  1.0, 0.0],
    ]

    best_result: dict[str, Any] | None = None
    best_nmse = float("inf")

    for init in candidate_starts:
        try:
            result = evaluate_expression(
                expression,
                X_train,
                y_train,
                X_val,
                y_val,
                constants=c,
                initial_values=init,
                max_nfev=500,
            )
            val_nmse = float(result.get("nmse_val", float("inf")))
            if np.isfinite(val_nmse) and val_nmse < best_nmse:
                best_nmse = val_nmse
                best_result = result
        except Exception:
            continue

    if best_result is None:
        # Fallback: return last attempt
        best_result = evaluate_expression(
            expression,
            X_train,
            y_train,
            X_val,
            y_val,
            constants=c,
            initial_values=[2.0, t_half_est, 1.0, 0.0],
            max_nfev=500,
        )

    return best_result


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
