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
    Tanh sigmoid in asinh-time space with log-linear x1 half-time scaling,
    power-law x1 rate scaling, and a secondary tanh(x0) baseline ramp.

    Template:
        c0*tanh(c1*x1^c8*(log(x0+sqrt(x0^2+c2)) - c3*log(x1+c4) - c5))
        + c6 + c7*tanh(x0)

    Design rationale:
    - log(x0+sqrt(x0^2+c2)): this equals asinh(x0/sqrt(c2)) + 0.5*log(c2),
      which is defined for ALL real x0 (including negative baseline values
      in TasA: x0_min=-1565, SH3: x0_min=-140). For large x0 it behaves
      like log(2*x0), giving the same log-time compression as before, but
      without requiring c2 > |x0_min|. With c2=1 initial value, the
      transform is well-conditioned for x0 spanning [−1565, 1.7M].
    - c0*tanh(...): equivalent to the logistic sigmoid but uses fewer nodes
      (tanh(z) = 2*sigmoid(2z)-1). c0 scales the plateau; c6 is the offset.
      Together they span [c6-c0, c6+c0], fitting y in [0,1] naturally.
    - c1*x1^c8: power-law concentration-dependent sigmoid steepness.
      For single-x1 datasets (x1=1), x1^c8=1 so c8 is inactive.
    - c3*log(x1+c4): log-linear half-time vs concentration (power law in
      linear space). c4 offsets x1 for small concentrations (e.g. 0.3 uM).
    - c7*tanh(x0): captures the fast initial rise in seeded/prp datasets
      where y jumps from 0 quickly. For unseeded datasets c7≈0.
    - 9 constants, complexity=35, parsimony factor ~0.956.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(time + sp.sqrt(time**2 + c[2]))
    log_param = sp.log(parameter + c[4])

    rate = c[1] * parameter ** c[8]
    half_log_time = c[3] * log_param + c[5]
    sigmoid_arg = rate * (log_time - half_log_time)

    expression = c[0] * sp.tanh(sigmoid_arg) + c[6] + c[7] * sp.tanh(time)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 1.0, 1.0, 0.0, 1.0, 2.0, 0.5, 0.0, 0.0],
        max_nfev=500,
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
