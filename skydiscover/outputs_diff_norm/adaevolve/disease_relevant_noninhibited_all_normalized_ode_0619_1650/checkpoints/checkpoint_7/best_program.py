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

    Nucleated secondary-nucleation aggregation ODE (sharp sigmoid + lag):

        d(c)/dt = (c0 + (c1 + c2*x2)*c + c4*c**2 + c5*c**3) * (1 - c) * (1 + c3*x1)

    Physical structure (Knowles/Cohen amyloid master equation):
    - c0            : primary nucleation source. Lets the trajectory leave the
                      baseline (c~=0), producing the observed lag phase. The
                      pure-logistic parent lacked this and got stuck at c=0,
                      explaining its high train NMSE and weak held-out fit.
    - (c1 + c2*x2)*c: elongation proportional to existing aggregate; seed M0
                      (x2) boosts it, shortening the lag exactly as seeding does.
    - c4*c**2       : surface-catalyzed SECONDARY nucleation, scaling
                      superlinearly with existing aggregate. It produces the
                      steep, delayed burst that linear autocatalysis cannot,
                      directly targeting the slope/onset-timing shape loss.
    - (1 - c)       : mass-conservation saturation; converted fraction cannot
                      exceed available monomer, so the trajectory plateaus at 1.
    - (1 + c3*x1)   : global rate scale set by initial monomer m0 (x1).

    The RHS is a degree-3 polynomial in c, linear in x1 and x2: no division,
    log, root, or variable exponent, so it is smooth, finite, and well
    conditioned over the observed input ranges. Over c in [0,1] all terms stay
    bounded, keeping the 5-constant least-squares fit stable on small datasets.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    csq = concentration * concentration
    # Rate = primary nucleation + seed-boosted elongation + quadratic and
    # cubic secondary nucleation. The cubic term raises the effective
    # autocatalytic reaction order (Cohen/Knowles n2 ~ 2), sharpening the
    # delayed burst and the slope at half-response that drive the shape loss.
    # It stays a bounded, smooth polynomial over c in [0,1] (no division,
    # root, log, or variable exponent), and can shrink to ~0 where unneeded.
    rate = (
        c[0]
        + (c[1] + c[2] * seed) * concentration
        + c[4] * csq
        + c[5] * csq * concentration
    )
    saturation = 1 - concentration
    monomer_scale = 1 + c[3] * monomer

    expression = rate * saturation * monomer_scale

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 1.0, 0.0, 0.0, 1.0, 0.0],
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
