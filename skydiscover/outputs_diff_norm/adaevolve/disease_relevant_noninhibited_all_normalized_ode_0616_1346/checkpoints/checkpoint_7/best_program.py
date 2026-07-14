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
    its onset.  Disease-relevant aggregation curves are governed by three
    coupled mechanisms acting on the available monomer pool (plateau - c):

      * primary nucleation: a small concentration-independent seeding rate
        that initiates growth from c=0, modulated by monomer m0 and seed M0;
      * linear elongation: rate proportional to existing fibril mass c;
      * secondary nucleation: in the Knowles theory new-fibril generation is
        superlinear in fibril mass (~c^2 at low conversion), setting the sharp
        sigmoid onset that a purely linear autocatalytic term underfits.

    This gives the polynomial RHS

        d(c)/dt = (c3 - c) * (c0*(1 + c4*x1 + c5*x2) + (c1 + c2*c)*c)

    The primary term c0*(...) initiates growth even from c=0 (lag-to-growth
    transition), the linear c1*c term gives elongation, and the quadratic
    c2*c^2 term provides the autocatalytic secondary-nucleation burst that
    controls the sigmoid steepness; (c3 - c) enforces monomer depletion.
    Every term is a low-order polynomial in c and linear in the features, so
    the form is smooth, globally defined, and free of singularities, invalid
    powers, logs, or overflow.  Six constants keep the least-squares inner fit
    well conditioned while capturing nucleation, elongation, and the
    autocatalytic burst that controls lag-to-growth steepness.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    plateau = c[3]
    monomer_pool = plateau - concentration
    # Kinetic core: primary-nucleation offset that lets growth leave c=0,
    # linear elongation (c1*c), and the autocatalytic secondary-nucleation
    # burst (c2*c^2) that sets the sigmoid steepness.
    kinetics = c[0] + (c[1] + c[2] * concentration) * concentration
    # Monomer/seed timescale modulation: in the Knowles/Meisl moment-closure
    # theory the effective elongation and secondary-nucleation rates both
    # scale with available monomer m0 (and seeding M0), so the *speed* of the
    # normalized sigmoid shifts with m0/M0 across protein systems while its
    # shape is preserved. Multiplying the whole RHS by this factor (rather
    # than only the primary term) couples m0/M0 to the full sigmoid timescale,
    # which is the dominant cross-protein effect behind the train/val gap.
    timescale = 1 + c[4] * monomer + c[5] * seed

    expression = timescale * monomer_pool * kinetics

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.01, 1.0, 1.0, 1.0, 0.0, 0.0],
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
