# EVOLVE-BLOCK-START
"""ODE discovery seed for normalized multi-dataset amyloid aggregation kinetics."""

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
    Propose an ODE right-hand side and let the harness fit its constants.

    This candidate is evaluated independently on each dataset — the same
    ODE template is used everywhere, but constants are fitted separately
    per dataset.  This allows a single functional form to capture the
    universal kinetic mechanism while the constants adapt to each protein
    system's specific rates, concentrations, and timescales.

    Features: x0 = normalized elapsed time, x1 = m0 initial monomer
    concentration, x2 = M0 seed concentration, x3 = concentration c.
    Units are ignored by the cleaned-data loader; leading numeric values are
    used directly.

    Amyloid aggregation master-equation template (Knowles/Cohen form).

    Pure logistic growth (dc/dt = k*c*(plateau-c)) cannot leave c=0: with no
    fibril present it has zero rate, so it cannot reproduce the lag phase and
    its onset.  Disease-relevant aggregation curves are driven by three
    coupled mechanisms: primary nucleation (a small concentration-independent
    seeding rate, modulated by monomer m0 and seed M0), autocatalytic
    secondary nucleation / elongation (rate proportional to existing fibril
    mass c), all multiplied by the available monomer pool (plateau - c):

        d(c)/dt = (c3 - c) * (c0 * (1 + c4*x1 + c5*x2) + c1 * c)

    The primary term c0*(...) initiates growth even from c=0 (capturing the
    lag-to-growth transition), the c1*c term provides the autocatalytic burst
    that sets the sigmoid steepness, and (c3 - c) enforces monomer depletion /
    saturation.  The form is polynomial in c and linear in the features, so it
    is smooth, globally defined, and free of singularities or invalid powers,
    keeping the least-squares inner fit well conditioned with only 5 constants.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    plateau = c[3]
    monomer_pool = plateau - concentration
    primary = c[0] * (1 + c[2] * monomer + c[4] * seed)
    secondary = c[1] * concentration

    expression = monomer_pool * (primary + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 1.0, 0.0, 1.0, 0.0],
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
