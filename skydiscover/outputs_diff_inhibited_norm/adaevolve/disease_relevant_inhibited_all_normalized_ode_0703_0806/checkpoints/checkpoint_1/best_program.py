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
    reduction) with inhibitor suppression of secondary nucleation.

    dc/dt = (plateau - c) * [ source + k2*m0*c*(c + lag) / (1 + ki*cd + kic*cd*c) ]

    - (plateau - c): mass-conservation capacity factor (aggregation halts as the
      accessible monomer pool is consumed); bounded and smooth.
    - source = kn*m0 + ks*M0 + b0: primary nucleation + seed + baseline flux,
      active at c = 0 so it sets lag onset.
    - autocatalytic term: secondary-nucleation/elongation amplification; the
      c*(c+lag) factor sharpens the sigmoid so the integrated trajectory
      reproduces the lag-then-burst kinetics rewarded by the shape loss.
    - inhibitor divides only the autocatalytic channel (inhibitors act mainly on
      secondary nucleation); the c-coupled term models progressive, mass-
      dependent blockade. Reduces exactly to the uninhibited law at cd = 0.

    No variable exponents, logs, or explicit-time terms, so the RHS stays
    smooth, finite, and well conditioned during least-squares fitting.
    Features: x0 = time, x1 = m0, x2 = M0, x3 = cd, x4 = current state c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(8)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Available-monomer / capacity factor (mass conservation): aggregation
    # slows and halts as the accessible pool is consumed. Bounded in [0, plateau].
    plateau = c[0]
    capacity = plateau - concentration

    # Primary nucleation + seed source term. Present even at c = 0, so it
    # sets the lag-phase onset. Weakly monomer- and seed-scaled.
    source = c[1] ** 2 * monomer + c[2] ** 2 * seed + c[3] ** 2

    # Autocatalytic (secondary-nucleation / elongation) amplification.
    # A c*(c + lag) form sharpens the sigmoid to reproduce the lag then
    # rapid-growth kinetics that the shape loss rewards. Scales with monomer.
    autocatalytic = c[4] ** 2 * monomer * concentration * (concentration + c[5] ** 2)

    # Inhibitor suppresses the secondary-nucleation channel far more strongly
    # than the primary source (established amyloid inhibition mechanism), so it
    # divides the autocatalytic term. Smooth, >= 1, and reduces to the
    # uninhibited law when cd = 0. The c-coupled term captures progressive
    # blockade of mass-dependent secondary nucleation.
    inhibitor_scale = 1 + c[6] ** 2 * inhibitor + c[7] ** 2 * inhibitor * concentration

    expression = capacity * (source + autocatalytic / inhibitor_scale)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.1, 0.1, 0.1, 1.0, 0.3, 1.0, 0.5],
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
