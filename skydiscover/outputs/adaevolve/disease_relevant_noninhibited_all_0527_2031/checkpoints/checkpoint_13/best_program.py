# EVOLVE-BLOCK-START
"""Symbolic regression seed for multi-dataset amyloid aggregation kinetics."""

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
    Extended Gompertz model with concentration-dependent plateau for amyloid
    aggregation kinetics.

    Amyloid nucleation-polymerization curves are asymmetric sigmoids with a
    pronounced lag phase. The Gompertz model captures this asymmetry; we
    extend it with a concentration-dependent plateau amplitude, motivated by
    the physical observation that higher protein concentration yields higher
    final ThT fluorescence signal:

        y = (c4 + c5*x1) * exp(-exp(-k*(x0 - t_m))) + c6

    where:
        k   = c0 * x1^c1   (concentration-dependent growth rate)
        t_m = c2 * x1^c3   (concentration-dependent inflection time)
        c4  = plateau base amplitude
        c5  = concentration-dependent plateau scaling
        c6  = baseline offset

    For single-concentration datasets (x1=1): k=c0, t_m=c2, plateau=(c4+c5),
    collapsing to a 4-constant Gompertz curve — no extra degrees of freedom.

    7 constants: c0 (rate scale), c1 (rate-conc exponent),
    c2 (inflection-time scale), c3 (inflection-time-conc exponent),
    c4 (plateau base), c5 (plateau-conc slope), c6 (baseline offset).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    parameter = x[1]

    rate = c[0] * parameter ** c[1]
    inflection_time = c[2] * parameter ** c[3]
    plateau = c[4] + c[5] * parameter
    baseline = c[6]

    # Extended Gompertz: (c4 + c5*x1) * exp(-exp(-rate*(t - t_m))) + baseline
    expression = (
        plateau * sp.exp(-sp.exp(-rate * (time - inflection_time)))
        + baseline
    )

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.2, -0.5, 10.0, -0.3, 1.0, 0.0, 0.0],
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
