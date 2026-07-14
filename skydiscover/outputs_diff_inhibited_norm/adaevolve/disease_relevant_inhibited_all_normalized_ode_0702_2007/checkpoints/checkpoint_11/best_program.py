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

    # DUAL-GATE inhibitor model. Abeta42 inhibitors (e.g. Brichos, small
    # molecules; Linse/Cohen/Knowles) act SELECTIVELY: they suppress the
    # fibril-surface-catalyzed SECONDARY-nucleation channel far more strongly
    # than primary nucleation / elongation. We therefore give the two channels
    # INDEPENDENT dose sensitivities instead of one shared denominator.
    #
    # Primary/source gate: mild, linear-in-cd suppression. >= 1 always.
    source_gate = 1 + c[5] ** 2 * inhibitor
    # Secondary-nucleation gate: strong, super-linear (linear + quadratic in
    # cd) suppression of the autocatalytic burst; this is what mainly delays
    # onset and lowers the maximal slope. >= 1 always => bounded, no singularity.
    amp_gate = 1 + c[6] ** 2 * inhibitor + c[4] ** 2 * inhibitor ** 2

    # Primary-nucleation / seeded source: LINEAR in monomer and seed plus a
    # monomer*seed cross term (seeded primary nucleation). Linear terms keep the
    # least-squares landscape well conditioned on normalized data (no pow
    # overflow / complex values at tiny concentrations), which empirically fits
    # and generalizes better than free monomer exponents.
    source_rate = (
        c[0] ** 2 * monomer + c[2] ** 2 * seed + c[1] ** 2 * monomer * seed
    ) / source_gate

    # Autocatalytic prefactor: elongation scales with the monomer pool and is
    # boosted by existing seed surface, gated by the strong secondary gate.
    autocatalytic_rate = (
        c[3] ** 2 * monomer * (1 + c[7] ** 2 * seed) / amp_gate
    )
    baseline_flux = c[8] ** 2
    plateau = c[9]
    # Monomer-depletion capacity drives every rate to zero at the plateau,
    # enforcing mass conservation and a bounded, smooth sigmoid.
    capacity = plateau - concentration

    # Autonomous growth: baseline + slow gated source + amplification combining
    # linear elongation (c[10]*c) and quadratic secondary nucleation (c**2),
    # the dominant Abeta42 amplification mechanism (Cohen/Knowles master
    # equation) producing the lag-then-sharp-rise sigmoid. Separating the two
    # inhibitor gates lets cd act mainly on onset timing and slope of the
    # autocatalytic phase while only weakly throttling the slow source.
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
        initial_values=[0.3, 0.3, 0.1, 1.0, 0.5, 1.0, 0.5, 0.2, 0.01, 1.0, 0.5],
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
