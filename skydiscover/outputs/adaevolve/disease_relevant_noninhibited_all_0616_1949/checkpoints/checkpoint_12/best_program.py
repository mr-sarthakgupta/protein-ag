# EVOLVE-BLOCK-START
"""Symbolic regression seed for normalized multi-dataset amyloid aggregation kinetics."""

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
    Propose an equation structure and let the harness fit its constants.

    This candidate is evaluated independently on each dataset — the same
    equation template is used everywhere, but constants are fitted separately
    per dataset.  This allows a single functional form to capture the
    universal kinetic mechanism while the constants adapt to each protein
    system's specific rates, concentrations, and timescales.

    Features: x0 = normalized elapsed time, x1 = m0 initial monomer
    concentration, x2 = static initial M0 seed/aggregate concentration.
    Units are ignored by the cleaned-data loader; leading numeric values are
    used directly.

    Asymmetric Gompertz sigmoid with concentration-dependent rate and
    half-time — a natural model for nucleation-dependent polymerization with a
    slow plateau approach (secondary nucleation):

        y = c4 * exp(-exp(-c0 * x1^c1 * (x0 - c2 * x1^c3 / (1 + c6*x2)))) + c5
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    # Asymmetric (Gompertz) sigmoid for nucleation-dependent polymerization.
    # Amyloid growth curves are characteristically ASYMMETRIC: a relatively
    # sharp post-lag rise followed by a slow approach to the plateau (a
    # signature of secondary-nucleation / autocatalytic mechanisms in the
    # Knowles/Meisl framework). A symmetric logistic cannot match that slow
    # tail and underfits the upper portion of the curve. The Gompertz form
    #     y = plateau * exp(-exp(-rate*(t - t0))) + baseline
    # is the canonical asymmetric nucleation-growth sigmoid and adds the
    # needed tail asymmetry at the SAME constant count as the logistic.
    #
    # Concentration structure is preserved: the elongation rate follows a
    # monomer power law (x1^c1), and the lag/half-time follows a monomer power
    # law (x1^c3) shortened by the static seed concentration x2 through a
    # smooth, well-conditioned factor 1/(1 + c6*x2). When x2 = 0 (unseeded
    # curves) this reduces exactly to c2*x1^c3, so those datasets are
    # unaffected; for x2 >= 0 with non-negative c6 the denominator stays
    # >= 1 > 0 (no singularity).
    #
    # Numerical safety: for normalized t in [0,1] and a finite fitted rate,
    # the inner exponent -rate*(t - t0) is bounded, so the double exponential
    # is globally smooth and finite — no division by zero, no invalid powers.
    rate = c[0] * monomer ** c[1]
    half_time = c[2] * monomer ** c[3] / (1 + c[6] * seed)
    plateau = c[4]
    baseline = c[5]

    growth = sp.exp(-sp.exp(-rate * (time - half_time)))
    expression = plateau * growth + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0],
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
    """Load the first dataset for local testing."""
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
