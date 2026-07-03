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
    Two-channel amyloid ODE combining a linear-capacity nucleation FLOOR with a
    self-limiting autocatalytic (secondary-nucleation) burst, modulated by a
    bounded Langmuir-occupancy inhibitor suppression factor.

    Research note: on this run the external tools returned no usable amyloid
    kinetics results (snippet_search returned only unrelated ML/materials
    papers; web_search timed out; the local JBC reference file is empty), so the
    structure follows the established Cohen/Knowles secondary-nucleation master
    equation rather than a copied formula. For Abeta42 the reduced aggregate-mass
    ODE is dominated by SECONDARY NUCLEATION (autocatalytic in existing fibril
    mass c and free monomer m), giving a logistic lag->burst->plateau curve.
    Surface-acting inhibitors bind a FINITE pool of fibril-surface catalytic
    sites, so their rate suppression SATURATES with dose (Langmuir occupancy)
    and the half-time delay is bounded, while the plateau is left unchanged.

      * Nucleation floor (lag seeder): seed-templated (~M0) and primary
        (monomer-only, ~m0) initiation, each times the LINEAR capacity (P - c).
        Dominates the initial rise while c ~ 0, starts the otherwise-vanishing
        autocatalytic term, and lets the M0 = 0 curves aggregate. Two channels
        with different state dependence let the fitted weights set lag onset
        independently of burst steepness.
      * Self-limiting autocatalytic elongation: c2^2 * m0 * c * (P - c). Classic
        logistic autocatalysis: slow at small c (lag), peaks near c = P/2 (burst
        / steepest slope), self-arrests at c -> P. This asymmetric slope profile
        is what the shape loss (slope profile + half-time timing) rewards. Since
        c in [0, P], c*(P - c) >= 0, so the RHS stays smooth, bounded, and
        sign-definite. Both channels share the SAME plateau P so mass
        conservation is consistent (all growth halts at P).
      * Inhibitor as a dimensionless Langmuir occupancy fraction:
            theta = cd / (Kd + cd),   theta in [0, 1),  Kd = c5^2 > 0,
        scaling the effective growth rate DOWN through a bounded factor
            suppression = 1 / (1 + c3^2 * theta),
        which stays in (1/(1+c3^2), 1], is smooth/singularity-free, equals 1 at
        cd = 0 (clean uninhibited limit), and has a residual escape floor so
        high-dose held-out curves are not over-delayed into a false non-reaction.

    Feature scaling: m0, M0 and cd are raw molar values (~1e-6), each multiplied
    by 1e6 to bring fitted constants to order one, conditioning least squares.
    Squared constants keep every kinetic rate non-negative. Six constants total.

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
    # Langmuir occupancy fraction theta = cd / (Kd + cd) in [0, 1): finite pool
    # of fibril-surface binding sites saturates as cd grows.
    occupancy = inhibitor / (c[5] ** 2 + inhibitor)
    # Bounded suppression toward positive floor 1/(1 + c3^2); smooth, >0, and
    # exactly 1 at cd = 0. Prevents divergent over-delay of high-dose curves.
    suppression = 1 / (1 + c[3] ** 2 * occupancy)
    plateau = c[4]
    capacity = plateau - concentration

    nucleation_flux = capacity * (seed_nucleation + primary_nucleation)
    # Self-limiting logistic autocatalysis: peaks at c = P/2, giving the burst.
    autocatalytic_flux = c[2] ** 2 * monomer * concentration * capacity

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
