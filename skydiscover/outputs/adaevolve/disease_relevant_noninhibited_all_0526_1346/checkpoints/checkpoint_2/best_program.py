# EVOLVE-BLOCK-START
"""Symbolic regression for multi-dataset amyloid aggregation kinetics.

Strategy: Hill / sigmoidal function with concentration-dependent half-time
and data-driven multi-start optimisation.

The Hill equation with concentration-dependent half-time:

    y = c3 * x0^c0 / ((c1 * x1^c2)^c0 + x0^c0) + c4

Physical interpretation:
  - c0: Hill / nucleation exponent (cooperativity, typically 2-5)
  - c1: half-time scale factor
  - c2: concentration exponent for half-time (power-law scaling with x1)
  - c3: plateau amplitude (~1 for rescaled data)
  - c4: baseline offset (~0 for rescaled data)

The term (c1 * x1^c2) is the effective half-time: when x2=0, it collapses
to a pure constant c1, handling single-concentration datasets; when c2!=0,
it captures how aggregation rate scales with concentration (e.g. secondary
nucleation gives c2 ~ -1 to -2, meaning higher concentration → faster).

Multiple starting points are tried (varying Hill exponent, half-time scale,
and concentration exponent), with data-driven half-time estimation, and the
best validation NMSE is returned.
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
    """Fit Hill kinetics with concentration-dependent half-time and multi-start.

    Uses the Hill equation with concentration-dependent half-time:
        y = c3 * x0^c0 / ((c1 * x1^c2)^c0 + x0^c0) + c4

    where:
        c0 = Hill exponent (cooperativity / nucleation order)
        c1 = half-time scale factor (normalised by concentration)
        c2 = concentration power-law exponent for half-time
        c3 = plateau amplitude (≈1 for rescaled data)
        c4 = baseline offset (≈0 for rescaled data)

    When x1 is constant (single-concentration datasets), (c1*x1^c2) reduces
    to a single effective constant, so the model degrades gracefully.

    Multiple starting points are tried; the one giving the lowest validation
    NMSE is returned. The half-time initial guess is estimated from the data
    as the time where the training signal is closest to 0.5, normalised by
    the median concentration.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).ravel()

    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    time = x[0]   # x0 = elapsed time
    param = x[1]  # x1 = concentration / pH / etc.

    # Hill equation with concentration-dependent half-time:
    # y = c3 * t^c0 / ((c1 * p^c2)^c0 + t^c0) + c4
    expression = c[3] * time**c[0] / ((c[1] * param**c[2])**c[0] + time**c[0]) + c[4]

    # Data-driven initial guess for half-time
    t_train = X_train[:, 0]
    p_train = X_train[:, 1]
    half_idx = int(np.argmin(np.abs(y_train - 0.5)))
    t_half_est = float(t_train[half_idx])
    t_pos = t_train[t_train > 0]
    if t_half_est <= 0:
        t_half_est = float(np.median(t_pos)) if len(t_pos) > 0 else 1.0

    # Normalise half-time by median concentration to get c1 initial value
    p_med = float(np.median(p_train))
    if p_med <= 0:
        p_med = 1.0
    t_half_norm = t_half_est / p_med  # initial guess for c1 when c2=1

    # Try diverse starting points:
    # - Different Hill exponents (1.5, 2, 3, 4)
    # - Different concentration exponents (c2 = -1, -0.5, 0, 0.5, 1)
    # - Different half-time scales
    candidate_starts = [
        # [c0,    c1,            c2,   c3,  c4]
        [2.0,  t_half_norm,    1.0,  1.0, 0.0],
        [3.0,  t_half_norm,    1.0,  1.0, 0.0],
        [4.0,  t_half_norm,    1.0,  1.0, 0.0],
        [1.5,  t_half_norm,    1.0,  1.0, 0.0],
        [2.0,  t_half_norm,    0.5,  1.0, 0.0],
        [2.0,  t_half_norm,   -0.5,  1.0, 0.0],
        [2.0,  t_half_norm,    0.0,  1.0, 0.0],
        [2.0,  t_half_norm * 0.5, 1.0, 1.0, 0.0],
        [2.0,  t_half_norm * 2.0, 1.0, 1.0, 0.0],
        [3.0,  t_half_norm,    0.5,  1.0, 0.0],
        [3.0,  t_half_norm,   -0.5,  1.0, 0.0],
        [2.0,  t_half_est,     0.0,  1.0, 0.0],  # c2=0 → conc-independent
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
        # Fallback: return last attempt with safe defaults
        best_result = evaluate_expression(
            expression,
            X_train,
            y_train,
            X_val,
            y_val,
            constants=c,
            initial_values=[2.0, t_half_norm, 1.0, 1.0, 0.0],
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
