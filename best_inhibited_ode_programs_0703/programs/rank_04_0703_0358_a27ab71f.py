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
    Two-channel amyloid ODE with SELF-LIMITING logistic autocatalysis plus a
    linear nucleation floor, modulated by a bounded Langmuir-occupancy
    inhibitor suppression factor.

    Physical reasoning (Cohen/Knowles amyloid master equation + Brichos-type
    surface-binding inhibition). External research tools failed on this run
    (research_papers returned only ML/imaging papers; web_search timed out; the
    local JBC reference file is empty), so the structure is justified from the
    established Cohen/Knowles secondary-nucleation framework: the reduced master
    equation for the aggregate mass fraction is autocatalytic in existing fibril
    mass and its self-consistent solution is logistic/sigmoidal (lag -> burst ->
    plateau); surface-acting inhibitors delay the half-time while leaving the
    final plateau essentially unchanged, i.e. they suppress the RATE not the
    endpoint. The template below encodes precisely those principles.

    The novelty over the polynomial parent is that the autocatalytic channel
    carries an intrinsic logistic feedback c*(P-c) instead of relying on a
    shared capacity power. Two channels with DIFFERENT state dependence let the
    fitted weights independently set lag onset vs. burst steepness / half-time,
    directly targeting the shape loss:

      * Nucleation floor (lag seeder): seed-templated (~M0) and primary
        (monomer-only, ~m0) initiation, each times the plain linear capacity
        (P - c). Dominates the initial rise while c ~ 0, and is required for the
        M0 = 0 curves (and to start the logistic term, which vanishes at c=0).

      * Self-limiting autocatalytic elongation: c2^2 * m0 * c * (P - c). This is
        the classic Fisher / logistic autocatalysis. Being quadratic in the
        state it is slow at small c (early lag), peaks at c = P/2 (the burst /
        steepest slope), and self-arrests at the plateau (c -> P). This built-in
        asymmetric lag-burst-plateau slope profile is exactly what the shape
        loss rewards, and it comes from state feedback, not a fragile capacity
        exponent. Since c in [0, P], c*(P - c) >= 0, so the RHS stays smooth,
        bounded, and sign-definite. Both channels share the SAME plateau P to
        avoid double-counting the capacity and keep mass conservation
        consistent (all growth halts at P).

      * Inhibitor as a dimensionless Langmuir occupancy fraction:
            theta = cd / (Kd + cd),   theta in [0, 1),  Kd = c5^2 > 0,
        scaling the effective growth rate DOWN through a bounded factor
            suppression = 1 / (1 + c3^2 * theta),
        which stays in (1/(1+c3^2), 1], is smooth/singularity-free everywhere,
        equals 1 exactly at cd=0 (clean uninhibited limit), and has a residual
        escape floor so high-dose held-out curves cannot be over-delayed into a
        false non-reaction.

    Feature scaling: m0, M0 and cd are raw molar values (~1e-6), each multiplied
    by 1e6 to bring fitted constants to order one, conditioning least squares.
    Squared constants keep every kinetic rate non-negative. Six constants total,
    avoiding added overfitting risk.

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

    # Two-channel structure with DIFFERENT capacity dependence, so the fitted
    # weights can independently tune lag onset vs. burst steepness / half-time.
    #
    # (1) Nucleation floor (seed + primary): drives the INITIAL rise while the
    #     aggregate mass c is still ~0. Scaled by the plain linear capacity
    #     (P - c) so it simply relaxes toward the plateau. This is the "lag
    #     seeder": without it the logistic term (which vanishes at c=0) could
    #     never start, and the M0=0 curves would never aggregate.
    #
    # (2) Self-limiting autocatalytic elongation: c * (P - c). This is the
    #     classic Fisher / logistic autocatalysis. Its rate is quadratic in the
    #     state, so it is slow at small c (early lag), peaks at c = P/2 (the
    #     burst / steepest slope), and self-arrests at the plateau (c -> P).
    #     This built-in asymmetric lag-burst-plateau slope profile is exactly
    #     what the shape_loss (slope profile + half-time timing) rewards, and it
    #     is achieved through state feedback rather than a fragile capacity
    #     power/log. It is bounded and non-negative because c in [0, P] gives
    #     c*(P - c) >= 0, so the RHS stays smooth and sign-definite.
    #
    # Both channels share the SAME plateau P = c[4] to avoid double-counting the
    # capacity and to keep mass conservation consistent (all growth halts at P).
    seed_nucleation = c[0] ** 2 * seed
    primary_nucleation = c[1] ** 2 * monomer
    # Langmuir occupancy fraction theta = cd / (Kd + cd) in [0, 1). A finite
    # pool of fibril-surface binding sites saturates: theta -> 1 as cd -> inf.
    occupancy = inhibitor / (c[5] ** 2 + inhibitor)
    # Bounded suppression factor toward a positive floor 1/(1 + c3^2) at full
    # occupancy; stays in (1/(1+c3^2), 1], smooth, singularity-free, and equals
    # 1 exactly at cd=0 (clean uninhibited limit). Prevents divergent over-delay
    # of high-dose held-out curves.
    suppression = 1 / (1 + c[3] ** 2 * occupancy)
    plateau = c[4]
    capacity = plateau - concentration

    nucleation_flux = capacity * (seed_nucleation + primary_nucleation)
    # Self-limiting logistic autocatalysis: peaks at c = P/2, giving the burst.
    autocatalytic_flux = (
        c[2] ** 2 * monomer * concentration * capacity
    )

    expression = suppression * (nucleation_flux + autocatalytic_flux)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 0.05, 0.6, 1.0, 0.92, 0.3],
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
