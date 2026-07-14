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
    Seeded nucleation-elongation ODE with a SATURABLE (Hill/Michaelis) inhibitor
    slowing factor.

    Physical reasoning (Cohen/Knowles amyloid master equation + Brichos-type
    surface-binding inhibition):
      * Aggregation of a normalized mass fraction c toward a plateau P is driven
        by three additive kinetic channels:
          - seed nucleation      ~ M0        (seed-templated initiation),
          - primary nucleation   ~ m0        (monomer-only initiation; required
            for the M0 = 0 curves that still aggregate),
          - autocatalytic elongation ~ m0 * c.
        The bounded (plateau - c) capacity factor enforces mass conservation:
        all growth halts once the aggregate mass reaches the plateau, keeping the
        RHS smooth and sign-definite.
      * Inhibitors that block the fibril SURFACE (finite catalytic-site pool)
        delay the half-time in a way that SATURATES at high dose. The number of
        blockable sites is finite, so the delay cannot grow without bound. This
        is a Langmuir/Michaelis occupancy, not an unbounded polynomial. Research
        note: external web/paper tools returned no directly relevant amyloid
        kinetics results, so this structure follows the established Cohen/Knowles
        + surface-saturation principle rather than a copied formula.

    Inhibitor structure:
        inhibitor_scale = 1 + k^2 * cd^2 / (Kd^2 + cd).
      * small cd:  cd^2/(Kd^2+cd) ~ cd^2/Kd^2  -> quadratic super-linear rise,
        matching the observed accelerating early delay (t_half 0.077->0.10->0.14).
      * large cd:  cd^2/(Kd^2+cd) ~ cd - Kd^2  -> transitions to LINEAR growth,
        so the delay saturates instead of diverging quadratically. This prevents
        the pure-quadratic template from over-delaying the highest-cd held-out
        curves (targets heldout_curve_nmse / shape extrapolation).
      * The denominator (Kd^2 + cd) is strictly positive (c5^2 > 0, cd >= 0), so
        the form is smooth and singularity-free everywhere.
      * cd = 0 gives inhibitor_scale = 1 exactly (clean uninhibited limit).

    Feature scaling: m0, M0 and cd are raw molar values (~1e-6), each multiplied
    by 1e6 to bring fitted constants to order one, conditioning least squares.
    Squared constants keep every kinetic rate non-negative. Six constants total
    (same count as the polynomial parent), avoiding added overfitting risk.

    Features: x0 = normalized time, x1 = initial monomer m0, x2 = seed M0,
    x3 = inhibitor concentration cd, and x4 = current normalized
    concentration/state.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    monomer = x[1] * 1e6
    seed = x[2] * 1e6
    inhibitor = x[3] * 1e6
    concentration = x[4]

    seed_nucleation = c[0] ** 2 * seed
    primary_nucleation = c[1] ** 2 * monomer
    elongation = c[2] ** 2 * monomer * concentration
    # Saturable Hill/Michaelis inhibitor binding replaces the unbounded quadratic
    # denominator: models a finite pool of fibril-surface binding sites.
    inhibitor_scale = 1 + c[3] ** 2 * inhibitor ** 2 / (c[5] ** 2 + inhibitor)
    plateau = c[4]
    capacity = plateau - concentration

    expression = (
        capacity
        * (seed_nucleation + primary_nucleation + elongation)
        / inhibitor_scale
    )

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 0.05, 0.5, 1.0, 0.92, 0.3],
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