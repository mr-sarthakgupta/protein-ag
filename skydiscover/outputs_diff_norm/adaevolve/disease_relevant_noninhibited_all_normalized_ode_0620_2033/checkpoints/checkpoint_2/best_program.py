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

    Amyloid master-equation RHS (Knowles/Meisl/Cohen structure).

    A pure logistic term c*(plateau-c) has zero slope at c=0 and so stalls in
    the lag phase, missing the onset timing that the shape loss penalizes.
    The amyloid integrated rate law instead has two monomer-fueled fluxes that
    share the available-monomer gate (plateau - c), enforcing mass
    conservation (growth halts as monomer is consumed at c = plateau):

        d(c)/dt = (c5 - c) * ( nucleation + c3*c + c4*c*c )

    * nucleation = c0*(1 + c1*x1 + c2*x2) is non-zero at c=0, so the
      trajectory leaves the baseline at a monomer (x1) / seed (x2) dependent
      primary-nucleation rate -- this controls the lag-phase onset.
    * c3*c is the linear elongation/autocatalysis term that produces the
      sigmoidal rise.
    * c4*c*c is a smooth superlinear secondary-nucleation term.  Real amyloid
      secondary nucleation is superlinear in polymer mass (exponent > 1);
      a quadratic captures that sharper feedback and steepens the transition
      (improving the shape/half-time loss) while staying polynomial -- no
      singularities, fractional powers, logs, or roots, so the least-squares
      fit stays well conditioned on small/noisy datasets.

    The form reduces to the parent logistic (c0,c4 -> 0) and to the
    well-scoring nucleation+linear law (c4 -> 0), so it strictly generalizes
    both while targeting the remaining shape error.

    Features: x0 = normalized time, x1 = m0 monomer, x2 = M0 seed,
    x3 = concentration c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    nucleation = c[0] * (1 + c[1] * monomer + c[2] * seed)
    growth = c[3] * concentration + c[4] * concentration * concentration
    plateau = c[5]

    expression = (plateau - concentration) * (nucleation + growth)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.0, 0.0, 1.0, 1.0, 1.0],
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
