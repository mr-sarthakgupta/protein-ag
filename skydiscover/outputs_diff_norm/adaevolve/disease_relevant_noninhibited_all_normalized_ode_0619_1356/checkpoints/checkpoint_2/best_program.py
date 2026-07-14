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

    Nucleated autocatalytic growth (Finke-Watzky / nucleation-elongation
    structure) cast as one bounded logistic ODE with a nucleation source:

        d(c)/dt = c0 * (1 + c1*x1 + c2*x2) * (c3 + c) * (c4 - c)

    Why this structure and why it is robust:
    - The integrator seeds each curve at its first *measured* value, which is
      0 for the many response-normalized lag-phase curves. A pure logistic
      c*(P-c) has dc/dt = 0 at c = 0 and is permanently stuck there, which is
      exactly why the parent's train NMSE is ~1.9 on lag-phase data. The
      additive nucleation offset c3 gives dc/dt = c0*(...)*c3*c4 > 0 at c = 0,
      so the trajectory can initiate from zero. This is the primary-nucleation
      source of the amyloid master equation; (c3 + c) then smoothly hands over
      to autocatalytic elongation/secondary nucleation as fibril mass grows.
    - The (c4 - c) monomer-depletion factor is a hard plateau attractor: it
      changes sign at c = c4 and pulls the state back, so c stays bounded and
      cannot blow up. This is the conservation-of-mass constraint and is the
      key stability difference from a polynomial growth term, which can diverge
      during stiff integration.
    - (1 + c1*x1 + c2*x2) lets the global timescale scale gently with initial
      monomer and seed concentration without dominating the structure.

    All operations are polynomial in c and linear in the static features:
    smooth, globally defined, no singularities/overflow, and only 5 constants
    so least-squares fitting stays well conditioned on small datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(5)

    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    rate = c[0] * (1 + c[1] * monomer + c[2] * seed)
    nucleation = c[3] + concentration
    plateau = c[4] - concentration

    expression = rate * nucleation * plateau

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.0, 0.0, 0.05, 1.0],
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
