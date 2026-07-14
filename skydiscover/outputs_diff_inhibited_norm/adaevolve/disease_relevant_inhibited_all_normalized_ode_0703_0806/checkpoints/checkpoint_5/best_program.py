# EVOLVE-BLOCK-START
"""ODE discovery seed for normalized Abeta42 inhibitor aggregation kinetics."""

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
    Physically-grounded amyloid aggregation ODE (Knowles/Cohen master-equation
    reduction) with inhibitor suppression of the secondary-nucleation channel.

    dc/dt = (plateau - c) * [ source
                              + k2*m0*c*(c + lag + kseed*M0) / (1 + ki*cd + kic*cd*c) ]

    Rationale (building on the strongest prior candidate, score 0.9349):
    - (plateau - c): mass-conservation capacity factor. Aggregation halts as the
      accessible monomer pool is consumed. Bounded and smooth; sets the plateau.
    - source = kn*m0 + ks*M0 + b0: primary nucleation + seed + baseline flux,
      active at c = 0 so it sets the lag-phase onset. Left UNINHIBITED because
      inhibitors act predominantly on secondary nucleation, not primary flux;
      keeping the source out of the denominator preserved cd = 0 behaviour and
      fit best previously.
    - autocatalytic term k2*m0*c*(c + lag): secondary-nucleation / elongation
      amplification. The c*(c + lag) form sharpens the sigmoid so the integrated
      trajectory reproduces lag-then-burst kinetics rewarded by the shape loss.
    - NEW: the onset offset gains a seed-coupled term kseed*M0. Highly-seeded
      curves skip more of the lag phase, so their amplification engages earlier;
      this single extra constant improves per-curve onset/half-response timing
      (the weaker shape-loss component) without adding variable exponents,
      explicit-time drift, or singular operations.
    - inhibitor divides only the autocatalytic channel (1 + ki*cd + kic*cd*c),
      a smooth factor >= 1 that reduces exactly to the uninhibited law at cd = 0
      and models progressive, mass-dependent blockade of secondary nucleation.

    All constants enter as squares (positivity / well-posedness). No variable
    exponents, logs, roots, or explicit time terms, so the RHS is smooth,
    finite, and well conditioned during least-squares fitting.

    Features: x0 = time, x1 = m0, x2 = M0, x3 = cd, x4 = current state c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration

    source = c[1] ** 2 * monomer + c[2] ** 2 * seed + c[3] ** 2

    onset = c[5] ** 2 + c[8] ** 2 * seed
    autocatalytic = c[4] ** 2 * monomer * concentration * (concentration + onset)

    inhibitor_scale = 1 + c[6] ** 2 * inhibitor + c[7] ** 2 * inhibitor * concentration

    expression = capacity * (source + autocatalytic / inhibitor_scale)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.1, 0.1, 0.1, 1.0, 0.3, 1.0, 0.5, 0.1],
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
    """Load the inhibitor dataset for local testing."""
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
