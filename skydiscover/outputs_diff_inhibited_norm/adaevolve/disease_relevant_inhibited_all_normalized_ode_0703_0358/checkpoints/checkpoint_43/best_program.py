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
    Seeded nucleation-elongation ODE with a BOUNDED Langmuir-occupancy
    inhibitor suppression factor (residual escape floor).

    Physical reasoning (Cohen/Knowles amyloid master equation + Brichos-type
    surface-binding inhibition). External web/paper tools returned no directly
    relevant amyloid-kinetics results on this run, so the structure is grounded
    in the established Cohen/Knowles secondary-nucleation framework (consistent
    with the local JBC reference) rather than a copied formula:
      * Aggregation of a normalized mass fraction c toward a plateau P is driven
        by three additive kinetic channels:
          - seed nucleation      ~ M0        (seed-templated initiation),
          - primary nucleation   ~ m0        (monomer-only initiation; required
            for the M0 = 0 curves that still aggregate),
          - autocatalytic elongation ~ m0 * c.
        The bounded (plateau - c) capacity factor enforces mass conservation:
        all growth halts once the aggregate mass reaches the plateau, keeping the
        RHS smooth and sign-definite.
      * Inhibitor as a dimensionless Langmuir occupancy fraction:
            theta = cd / (Kd + cd),   theta in [0, 1),  Kd = c5^2 > 0.
        A finite pool of fibril-surface binding sites means occupancy saturates
        at high dose (theta -> 1). The occupied fraction scales the effective
        growth rate DOWN through a bounded suppression factor:
            suppression = 1 / (1 + c3^2 * theta).
      * Key advantage over the previous unbounded 1 + k^2*cd^2/(Kd^2+cd) divisor:
        - suppression is intrinsically bounded in (1/(1+c3^2), 1]; it never
          diverges, so it cannot over-delay high-cd held-out curves into a false
          non-reaction. The floor 1/(1+c3^2) is a residual "escape rate": even at
          saturating inhibitor the highest-dose curves still reach the plateau,
          which directly targets heldout_curve_nmse / shape extrapolation
          robustness on the strongest inhibitor doses.
        - suppression = 1/(1 + c3^2*theta) is smooth and singularity-free
          everywhere (denominator >= 1, Kd^2 + cd > 0), never non-real.
        - cd = 0 gives theta = 0 and suppression = 1 exactly (clean uninhibited
          limit, preserving the reference kinetics).

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
    # Langmuir occupancy fraction theta = cd / (Kd + cd) in [0, 1). A finite
    # pool of fibril-surface binding sites saturates: theta -> 1 as cd -> inf.
    occupancy = inhibitor / (c[5] ** 2 + inhibitor)
    # Bounded suppression factor: multiplies the effective growth rate DOWN
    # toward a positive floor 1/(1 + c3^2) at full occupancy. It stays in
    # (1/(1+c3^2), 1], is smooth and singularity-free, and at cd=0 equals 1
    # exactly (clean uninhibited limit). The residual floor is a physical
    # "escape rate": even at saturating inhibitor the highest-dose curves
    # still aggregate, so extrapolated high-cd trajectories cannot be
    # over-delayed to a false non-reaction, fixing the divergent-delay
    # failure mode of the previous unbounded 1 + k^2*cd^2/(Kd^2+cd) divisor.
    suppression = 1 / (1 + c[3] ** 2 * occupancy)
    plateau = c[4]
    capacity = plateau - concentration

    expression = (
        suppression
        * capacity
        * (seed_nucleation + primary_nucleation + elongation)
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