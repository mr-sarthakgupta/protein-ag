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
    Tanh sigmoid in scaled-asinh-time space with log-linear x1 half-time
    scaling, power-law x1 rate scaling, and a secondary tanh(x0) baseline
    ramp. Uses c[2]^2 inside the sqrt so c[2] acts as a time-scale parameter
    (multiplicatively coupled to x0), giving a better optimizer landscape for
    datasets where x0 spans many orders of magnitude (e.g. Kar2011 x0 up to
    937000 seconds vs. CsgA x0 up to 22 hours). Dropping the c4 offset on
    log(x1) saves 2 nodes, keeping complexity=35 with only 8 constants.

    Template:
        c0*tanh(c1*x1^c7*(log(x0+sqrt(x0^2+c2^2)) - c3*log(x1) - c5))
        + c4 + c6*tanh(x0)

    Design rationale:
    - log(x0+sqrt(x0^2+c2^2)): equals log(c2) + asinh(x0/c2). c2 is a
      time-scale: log(c2) is absorbed into c5, while asinh(x0/c2) compresses
      the x0 range. Defined for ALL real x0 (negative x0 in Ye2011, SH3,
      TasA). With c2_init=1 the transform starts as pure asinh(x0), and the
      optimizer scales c2 freely to match any x0 magnitude.
    - c0*tanh(...): plateau-scaling sigmoid. c4 is the baseline offset.
      Together spans [c4-c0, c4+c0], naturally covering y in [0,1].
    - c1*x1^c7: power-law concentration-dependent sigmoid steepness.
      Consistent with nucleation-elongation theory k_app ∝ [M]^(n/2).
      For single-x1 datasets (x1=1), x1^c7=1 so c7 is inactive.
    - c3*log(x1): log-linear half-time vs concentration (power-law t_half
      ∝ x1^c3 in linear space). No c4 offset needed: for datasets with
      x1≥1 (sequential indices or uM concentrations), log(x1) is well
      defined. The optimizer adjusts c3 to handle any x1 range.
    - c6*tanh(x0): secondary ramp for seeded/prp datasets with fast initial
      rise (e.g. prp/Stoehr x0 in [1e-14, 17], insulin seeded). For large-x0
      datasets where tanh(x0)≈1, c6 becomes an extra offset that the
      optimizer sets to 0 (degenerate with c4), which is fine.
    - 8 constants (c0..c7), complexity=35, parsimony factor ~0.956.
    - max_nfev=1000 gives harder datasets more optimizer budget.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(time + sp.sqrt(time**2 + c[2]**2))
    log_param = sp.log(parameter)

    rate = c[1] * parameter ** c[7]
    half_log_time = c[3] * log_param + c[5]
    sigmoid_arg = rate * (log_time - half_log_time)

    expression = c[0] * sp.tanh(sigmoid_arg) + c[4] + c[6] * sp.tanh(time)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 1.0, 1.0, 0.5, 0.5, 2.0, 0.0, 0.0],
        max_nfev=1000,
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
