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

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Knowles/Cohen amyloid master-equation structure for a normalized aggregate
    # mass fraction c in [0, plateau]. This is the 0.8537-scoring template with
    # the (unphysical, non-autonomous) explicit-time drive replaced by a purely
    # concentration-driven secondary-nucleation term, so the RHS is autonomous
    # and stays physically meaningful.
    #
    #   dc/dt = (plateau - c) * [ baseline
    #                             + source_rate
    #                             + auto_rate * (c + b*c^2) ]
    #
    # (plateau - c) enforces mass conservation (monomer depletion): the growth
    # rate vanishes at the plateau so every trajectory saturates at the observed
    # 0->1 level. `source_rate` is the primary-nucleation + seed-driven lag
    # source with a free monomer reaction order (monomer**c[1]); `auto_rate`
    # is the elongation/secondary-nucleation prefactor, also with a free monomer
    # order (monomer**c[4]) and enhanced by pre-formed seed M0. The catalytic
    # factor (c + b*c^2) makes new aggregate formation grow super-linearly in
    # existing aggregate c, sharpening the sigmoid transition to match onset /
    # half-response timing and slope (shape loss), without any dependence on
    # absolute time.
    #
    # Inhibitor cd (x3) multiplicatively suppresses BOTH source and catalytic
    # pathways through one shared denominator, delaying onset and slowing growth
    # while exactly recovering the cd=0 kinetics:
    #   1 / (1 + a*cd + d*cd*c),  which is >= 1 everywhere (squared coeffs,
    # cd, c >= 0). Powers of monomer use m0 > 0 with real exponents, so the
    # expression is finite, positive and smooth -> well conditioned for LSQ.
    plateau = c[0]
    inhibition = 1 + c[8] ** 2 * inhibitor + c[9] ** 2 * inhibitor * concentration
    baseline = c[10] ** 2
    source_rate = (c[2] ** 2 * monomer ** c[1] + c[3] ** 2 * seed) / inhibition
    auto_rate = c[6] ** 2 * monomer ** c[4] * (1 + c[5] ** 2 * seed) / inhibition

    catalysis = concentration + c[7] ** 2 * concentration ** 2
    capacity = plateau - concentration

    expression = capacity * (baseline + source_rate + auto_rate * catalysis)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.5, 0.5, 0.1, 0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 0.01],
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
