# EVOLVE-BLOCK-START
"""Symbolic regression seed for Alpha-synuclein Gaspar 2017 0.3uM seed data."""

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

    Features are x0 = elapsed measurement coordinate X and x1 = concentration_uM.
    EvoX / AdaEvolve should improve this expression template while the harness
    fits continuous constants and scores validation NMSE.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    concentration = x[1]

    # Sigmoidal (logistic) model for seeded amyloid aggregation.
    # For nucleation-dependent polymerization, the ThT signal follows a sigmoid:
    #   y = amplitude / (1 + exp(-k*(t - t_half))) + offset
    # where:
    #   t_half = c0 / (concentration + c1)  [lag time decreases with concentration]
    #   k = c2 + c3 * concentration          [growth rate increases with concentration]
    #   amplitude = c4, offset = c5
    # This properly captures the S-shaped curves with concentration-dependent
    # lag phase and growth rate.

    t_half = c[0] / (concentration + c[1] ** 2 + 1e-3)
    k = c[2] ** 2 + c[3] ** 2 * concentration
    expression = c[4] / (1 + sp.exp(-k * (time - t_half))) + c[5]

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
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
    """Load deterministic Alpha-synuclein Gaspar 2017 splits (matches evaluator)."""
    from evaluator import load_alphasyn_data

    return load_alphasyn_data()


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = _load_data()
    result = run_discovery(X_train, y_train, X_val, y_val)
    print("equation:", result.get("equation"))
    print("nmse_val:", result.get("nmse_val"))
    print("combined_score:", result.get("combined_score"))
