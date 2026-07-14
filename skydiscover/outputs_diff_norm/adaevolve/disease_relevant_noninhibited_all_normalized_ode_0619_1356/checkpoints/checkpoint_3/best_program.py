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

    Nucleated autocatalytic aggregation ODE (amyloid master-equation form):

        d(c)/dt = c0 * (1 + c1*x1 + c2*x2) * (c3 + c + c4*c**2) * (c5 - c)

    Structure and physical justification (Knowles et al. master equation):
    - Source/growth factor (c3 + c + c4*c**2) combines three universal
      microscopic processes:
        * c3  -> primary nucleation source. Crucially nonzero, so dc/dt > 0
          at c = 0; this lets response-normalized lag-phase curves (which
          the integrator seeds at the measured value 0) initiate growth.
          A pure logistic c*(P-c) is stuck at c=0 (the parent's train NMSE
          ~1.9 reflects exactly this failure mode).
        * c   -> linear elongation / fibril-end growth, the autocatalytic
          term that produces the steep rise.
        * c4*c**2 -> secondary nucleation, which is autocatalytic in fibril
          mass and sharpens the sigmoidal lag-to-growth transition that
          single-term logistics cannot capture. One extra constant.
    - Depletion factor (c5 - c) enforces conservation of mass: it changes
      sign at c = c5 and acts as a bounded plateau attractor, so the state
      cannot diverge during stiff integration (key stability property vs a
      bare polynomial growth term).
    - (1 + c1*x1 + c2*x2) lets the global timescale scale gently with
      initial monomer (m0) and seed (M0) concentration without dominating.

    All operations are polynomial in c and linear in static features: smooth,
    globally defined, free of singularities/overflow/non-real values, and
    only 6 fitted constants so least-squares stays well conditioned on small
    datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    rate = c[0] * (1 + c[1] * monomer + c[2] * seed)
    source = c[3] + concentration + c[4] * concentration * concentration
    plateau = c[5] - concentration

    expression = rate * source * plateau

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.0, 0.0, 0.05, 0.5, 1.0],
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
