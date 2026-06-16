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
    Gompertz sigmoid with mixed x1 modulation for protein aggregation kinetics.

    The Gompertz model captures the asymmetric sigmoidal shape typical of
    nucleation-dependent protein aggregation (fast rise, slow approach to
    plateau) better than the symmetric logistic.

    Form:
        rate      = c0 * exp(c1 * x1)   (exponential rate scaling — physically
                                          motivated: Arrhenius/nucleation kinetics
                                          scale exponentially with concentration)
        half_time = c2 + c3 * x1        (affine lag-time — more robust than
                                          exponential for sequential-index x1
                                          and avoids exp blow-up at large x1)
        y = c4 * exp(-exp(-rate * (x0 - half_time))) + c5

    Design rationale:
    - Keeping exponential for rate: concentration-dependent nucleation rates
      follow Arrhenius-like scaling, so exp(c1*x1) is physically motivated
    - Switching half_time to affine (c2 + c3*x1): for datasets where x1 is a
      sequential index (1,2,...,N), the exponential c2*exp(c3*x1) can overflow
      or become degenerate; affine is well-defined for all real x1 values
    - Affine half_time still captures the linear trend in lag time vs parameter
      which is often observed in concentration-series experiments
    - Both exp(c1*x1) and (c2+c3*x1) are smooth and globally defined

    Stability:
    - rate = c0*exp(c1*x1): always finite for bounded x1
    - half_time = c2 + c3*x1: linear, never singular
    - exp(-exp(...)): double exponential maps all reals to (0,1), bounded
    - c4 plateau (~1 for rescaled data), c5 baseline offset (~0)

    Initial values: c0=0.1 (slow rate), c1=0.0 (neutral x1 rate scaling),
    c2=10.0 (moderate lag intercept), c3=0.0 (neutral x1 lag slope),
    c4=1.0 (full plateau), c5=0.0 (zero baseline).
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    parameter = x[1]

    rate = c[0] * sp.exp(c[1] * parameter)
    half_time = c[2] + c[3] * parameter
    plateau = c[4]
    baseline = c[5]

    expression = plateau * sp.exp(-sp.exp(-rate * (time - half_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.0, 10.0, 0.0, 1.0, 0.0],
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
