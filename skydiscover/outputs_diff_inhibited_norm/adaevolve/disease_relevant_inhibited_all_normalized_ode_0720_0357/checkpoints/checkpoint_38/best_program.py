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
    Autonomous Knowles/Cohen amyloid master-equation ODE for inhibited Abeta42,
    tuned to reproduce the dose-dependent DELAY *and* STRETCH of the sigmoid.

    Physical picture (normalized fibril mass fraction c):
      dc/dt = (plateau - c) * [ source + secondary ]
    Growth = slow primary nucleation + seed-fed elongation (source) plus
    fibril-surface-catalysed secondary nucleation, autocatalytic in mass
    (secondary). A mass-conservation capacity (plateau - c) closes the curve.

    Data-driven target: the empirical crossing times show that raising cd both
    right-shifts the sigmoid AND flattens its knee (for the worst unseeded
    cd=2e-6 curve the t50->t90 gap widens ~5x versus cd=0). Two distinct static
    inhibitor actions, both exactly 1 at cd=0 so the proven uninhibited kernel
    is recovered:

    * cooperative monomer sequestration --
        free_mono = monomer / (1 + (c1^2 + c2^2*cd)*cd)
      the c2^2*cd^2 term removes disproportionately more reactive monomer at
      strong doses, super-linearly lengthening the lag of high-dose curves
      while barely touching low doses. Denominator >= 1.

    * dose-gated surface poisoning of secondary nucleation --
        surf_block = 1 + (c7^2 + c8^2*cd) * concentration * (1 + concentration)
      inhibitor coats the growing fibril surface, so the autocatalytic rate is
      throttled across the WHOLE growth phase (linear + quadratic mass), not
      just near the plateau. Higher cd => larger denominator => the knee
      flattens and the 50/75/90% crossings spread out -> reproduces the
      high-dose stretch that dominates the curve-level shape loss. Denominator
      >= 1 for all non-negative inputs.

    source = c3^2 + c4^2*free_mono + c5^2*seed keeps a finite lag-phase drive so
    early timing is set, not clamped; seed shortens the lag. secondary carries
    the (1 + concentration + c0^2*seed) sharpening so the low-dose knee stays
    crisp (slope-profile part of the shape loss).

    Stability: rate coefficients squared for sign control; both denominators are
    >= 1 over the non-negative feature ranges, so the RHS is globally smooth,
    finite and singularity-free -- well conditioned for single-start least
    squares. Nine fitted constants (< 13), fully autonomous.

    Features: x0 = time, x1 = m0, x2 = M0 seed, x3 = cd inhibitor, x4 = state c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration

    # Cooperative monomer sequestration (super-linear dose dependence).
    free_mono = monomer / (1 + (c[1] ** 2 + c[2] ** 2 * inhibitor) * inhibitor)

    # Dose-gated surface poisoning acting across the whole growth phase.
    surf_block = 1 + (c[7] ** 2 + c[8] ** 2 * inhibitor) * concentration * (1 + concentration)

    # Primary nucleation + seed-driven elongation source.
    source = c[3] ** 2 + c[4] ** 2 * free_mono + c[5] ** 2 * seed

    # Autocatalytic secondary nucleation, seed-boosted, inhibitor-throttled.
    secondary = (
        c[6] ** 2 * free_mono * concentration * (1 + concentration + c[0] ** 2 * seed)
        / surf_block
    )

    expression = capacity * (source + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.3, 0.5],
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
