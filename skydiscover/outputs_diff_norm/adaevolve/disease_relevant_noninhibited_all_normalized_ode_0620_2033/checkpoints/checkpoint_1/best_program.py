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

    Nucleation-augmented autocatalytic growth law (amyloid master-equation
    structure, Knowles/Meisl):

        d(c)/dt = (c4 - c) * (c0*(1 + c1*x1 + c2*x2) + c3*c)

    Rationale.  The plain logistic term c*(plateau-c) has zero derivative at
    c=0, so an integration that starts from a near-zero observed value (the
    typical sigmoidal aggregation curve) can stall in the lag phase and miss
    the onset timing that the shape loss penalizes.  Adding a primary
    nucleation term `c0*(1 + c1*x1 + c2*x2)` that is non-zero at c=0 lets the
    trajectory leave the baseline at a monomer/seed-dependent rate, while the
    autocatalytic secondary term `c3*c` reproduces the steep sigmoidal rise.
    Both source and sink share the available-monomer gate `(c4 - c)`,
    enforcing mass conservation (growth halts as monomer is consumed at the
    plateau c=c4).  The form reduces to the parent logistic when c0 -> 0, so
    it strictly generalizes the current best while fixing the onset failure
    mode.  It is a polynomial in c (smooth, globally defined, no
    singularities, powers, logs, or roots), making the five-constant
    least-squares fit well conditioned on small/noisy datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    nucleation = c[0] * (1 + c[1] * monomer + c[2] * seed)
    growth = c[3] * concentration
    plateau = c[4]

    expression = (plateau - concentration) * (nucleation + growth)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.0, 0.0, 1.0, 1.0],
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
