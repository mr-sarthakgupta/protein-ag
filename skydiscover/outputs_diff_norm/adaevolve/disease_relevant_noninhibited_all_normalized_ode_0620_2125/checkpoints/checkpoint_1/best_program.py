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

    Secondary-nucleation (Knowles-Cohen) ODE template.

    Pure logistic growth is symmetric and cannot reproduce the lag phase
    seen in amyloid aggregation. The dominant mechanism is autocatalytic
    secondary nucleation: new aggregate forms at a rate proportional to
    both available monomer and existing fibril mass. We combine a small
    monomer-dependent primary-nucleation baseline (sets the onset) with an
    aggregate-mass-driven autocatalytic term (drives the sharp acceleration
    after the lag), all gated by a monomer-depletion / saturation factor
    (plateau - c) that enforces mass conservation:

        d(c)/dt = (c0 - c)
                  * [ c1*(1 + c2*x1)            # primary nucleation onset
                      + c3 * c * (c + c4*x2) ]  # secondary nucleation + seed

    The expression is polynomial in the state c, so it is smooth, globally
    defined, has no singularities or non-real values during integration,
    and remains numerically stable for least-squares fitting. The seed term
    c4*x2 gives seeded curves a non-zero initial autocatalytic rate, which
    shortens their lag, while x1 modulates the primary-nucleation onset.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    plateau = c[0]
    primary = c[1] * (1 + c[2] * monomer)
    secondary = c[3] * concentration * (concentration + c[4] * seed)

    expression = (plateau - concentration) * (primary + secondary)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.05, 0.0, 1.0, 1.0],
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
