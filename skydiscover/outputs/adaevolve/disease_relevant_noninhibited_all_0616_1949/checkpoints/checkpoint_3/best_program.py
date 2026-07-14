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

    Logistic/sigmoidal template with concentration-dependent rate and
    half-time — a natural model for nucleation-dependent polymerization:

        y = c4 / (1 + exp(-c0 * x1^c1 * (x0 - c2 * x1^c3))) + c5
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    # Logistic (sigmoidal) model for nucleation-dependent polymerization.
    # The pure 6-constant logistic with monomer power-law rate/half-time was
    # the strongest stable form found so far, so we keep it as the backbone
    # rather than the Richards generalization (whose extra (.)^shape exponent
    # is ill-conditioned and degenerate with the plateau, and regressed).
    #
    # The single new structural idea is to use the previously unused static
    # seed feature x2 (initial seed/aggregate concentration M0). Seeding is
    # known to shorten the lag phase / half-time of amyloid growth: more seed
    # means earlier onset. We encode this as a smooth, well-conditioned
    # multiplicative reduction of the half-time:
    #
    #     half_time = c2 * x1^c3 / (1 + c6 * x2)
    #
    # Numerical safety / backward compatibility:
    #   - When x2 = 0 (the many unseeded curves) this reduces EXACTLY to the
    #     winning c2 * x1^c3 form, so those datasets are unaffected.
    #   - x2 >= 0 and a non-negative fitted c6 keep the denominator >= 1 > 0,
    #     so no singularity; least-squares can still explore small/negative
    #     c6 smoothly because x2 is a small static per-curve constant.
    # This adds exactly one physically meaningful constant with a clear payoff
    # on seeded datasets and zero cost on unseeded ones.
    rate = c[0] * monomer ** c[1]
    half_time = c[2] * monomer ** c[3] / (1 + c[6] * seed)
    plateau = c[4]
    baseline = c[5]

    growth = 1 + sp.exp(-rate * (time - half_time))
    expression = plateau / growth + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.1, 0.5, 10.0, -0.5, 1.0, 0.0, 0.0],
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
