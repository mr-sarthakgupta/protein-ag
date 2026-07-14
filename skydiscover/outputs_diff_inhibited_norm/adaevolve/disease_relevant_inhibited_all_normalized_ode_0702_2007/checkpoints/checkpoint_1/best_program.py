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
    Bounded aggregation-rate ODE with explicit inhibitor modulation.

    Features: x0 = normalized time, x1 = initial monomer m0, x2 = seed M0,
    x3 = inhibitor concentration cd, and x4 = current normalized
    concentration/state.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(11)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Inhibitor gates every production channel (cd suppresses source and
    # secondary nucleation); the cd*monomer cross term captures stronger
    # suppression at high monomer load. inhibitor_scale >= 1 always, so it is
    # a safe bounded denominator (no singularity).
    inhibitor_scale = 1 + c[5] ** 2 * inhibitor + c[6] ** 2 * inhibitor * monomer
    # Primary-nucleation / seeded source term with a smooth monomer power.
    source_rate = (c[0] ** 2 * monomer ** c[1] + c[2] ** 2 * seed) / inhibitor_scale
    # Autocatalytic (elongation) prefactor, boosted by available seed sites.
    autocatalytic_rate = (
        c[3] ** 2 * monomer ** c[4] * (1 + c[7] ** 2 * seed) / inhibitor_scale
    )
    baseline_flux = c[8] ** 2
    plateau = c[9]
    # Monomer-depletion capacity drives every rate to zero at the plateau,
    # enforcing mass conservation and a bounded, smooth sigmoid.
    capacity = plateau - concentration

    # Aggregate growth combines a baseline flux, a monomer/seed source, and an
    # autocatalytic term with BOTH linear elongation (c[10]*c) and quadratic
    # secondary nucleation (c**2). Secondary nucleation is the dominant
    # amyloid-beta amplification mechanism (Cohen/Knowles master equation) and
    # produces the characteristic lag-then-sharp-rise kinetics. Replacing the
    # earlier non-physical explicit-time term with this autonomous
    # concentration-driven term improves both pointwise fit and shape while
    # keeping the equation autonomous and numerically well behaved.
    expression = capacity * (
        baseline_flux
        + source_rate
        + autocatalytic_rate * concentration * (c[10] + concentration)
    )

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.5, 0.1, 1.0, 0.5, 1.0, 0.0, 0.0, 0.01, 1.0, 0.5],
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
