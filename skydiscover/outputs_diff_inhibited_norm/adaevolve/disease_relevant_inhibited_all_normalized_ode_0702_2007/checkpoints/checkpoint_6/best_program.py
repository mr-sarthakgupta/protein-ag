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
    c = constant_symbols(11)  # c[4] intentionally used below as source seed weight

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Inhibitor gates the amplification channels. cd enters BOTH a linear and a
    # saturating (cd^2) term so strong inhibitor doses suppress secondary
    # nucleation super-linearly, as seen for amyloid-beta inhibitors that
    # specifically block the secondary-nucleation surface. inhibitor_scale >= 1
    # always => bounded, singularity-free denominator. Squared constants keep
    # every rate positive regardless of the least-squares sign.
    inhibitor_scale = 1 + c[5] ** 2 * inhibitor + c[6] ** 2 * inhibitor ** 2

    # Primary-nucleation / seeded source: linear in monomer and seed (no free
    # exponent, so tiny concentrations ~1e-6 cannot cause pow overflow or
    # complex values). This is the slow flux that seeds the lag phase.
    source_rate = c[0] ** 2 * monomer + c[2] ** 2 * seed + c[4] ** 2 * monomer * seed

    # Amplification prefactor: elongation/secondary nucleation scale with the
    # available monomer pool and are boosted by existing seed. Gated by the
    # inhibitor so cd delays onset and lowers the maximal growth rate.
    amp_rate = c[3] ** 2 * monomer * (1 + c[7] ** 2 * seed) / inhibitor_scale

    baseline_flux = c[8] ** 2
    plateau = c[9]
    # Monomer-depletion capacity drives every rate to zero at the plateau,
    # enforcing mass conservation and a bounded, smooth sigmoid.
    capacity = plateau - concentration

    # dc/dt = capacity * [baseline + slow source + autocatalytic amplification].
    # The amplification term combines linear elongation (c) and quadratic
    # secondary nucleation (c^2) via c*(c[1] + c[10]*c). Secondary nucleation
    # (the c^2 part) is the dominant amyloid-beta amplification mechanism
    # (Cohen/Knowles master equation) and creates the sharp lag-then-rise
    # sigmoid. Keeping the source term un-gated but the amplification gated
    # lets cd act primarily on onset timing and slope, matching the shape loss.
    expression = capacity * (
        baseline_flux
        + source_rate / inhibitor_scale
        + amp_rate * concentration * (c[1] + c[10] * concentration)
    )

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.3, 0.1, 0.1, 1.0, 0.5, 1.0, 0.2, 0.5, 0.01, 1.0, 1.0],
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
