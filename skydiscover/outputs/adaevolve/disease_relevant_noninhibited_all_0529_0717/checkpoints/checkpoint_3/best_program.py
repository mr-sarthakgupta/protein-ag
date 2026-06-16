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
    Logistic sigmoid in log-time space with log-linear x1 half-time scaling
    and power-law x1 rate scaling.

    Template: c0/(1+exp(-c1*x1^c7*(log(x0+c2) - c3*log(x1+c4) - c5))) + c6

    Design rationale:
    - log(x0+c2): compresses the enormous x0 range variation across datasets
      (x0 spans from negative values to ~1.7M); c2 shifts the time origin to
      handle near-zero and negative x0 values safely.
    - c3*log(x1+c4): log-linear x1 half-time dependence, equivalent to a
      power law in linear space (t_half ~ (x1+c4)^c3); c4 offsets x1 so that
      small or fractional x1 values (e.g. 0.3 uM) remain well-behaved.
      For single-column datasets (x1=1), this term is a constant absorbed
      into c5.
    - c1*x1^c7: concentration-dependent rate (sigmoid steepness); for
      single-column datasets (x1=1), x1^c7=1 so c7 is inactive.
    - c6: baseline offset handles datasets where y(0) > 0.
    - 8 constants, complexity ~30, parsimony factor ~0.963.

    Constants: c0=plateau, c1=rate_scale, c2=time_offset, c3=halftime_x1_exp,
               c4=x1_offset, c5=log_halftime, c6=baseline, c7=rate_x1_exp
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    time = x[0]
    parameter = x[1]

    log_time = sp.log(time + c[2])
    log_param = sp.log(parameter + c[4])

    rate = c[1] * parameter ** c[7]
    half_log_time = c[3] * log_param + c[5]
    plateau = c[0]
    baseline = c[6]

    expression = plateau / (1 + sp.exp(-rate * (log_time - half_log_time))) + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0],
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
