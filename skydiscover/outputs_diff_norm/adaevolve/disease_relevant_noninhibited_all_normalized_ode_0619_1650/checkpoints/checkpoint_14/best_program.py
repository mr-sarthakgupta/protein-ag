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

    Same template everywhere; constants fitted independently per dataset.

    Features: x0 = normalized elapsed time, x1 = m0 initial monomer
    concentration, x2 = M0 seed concentration, x3 = concentration c.

    Time-gated nucleated-aggregation ODE:

        d(c)/dt = (c0 + (c1 + c2*x2)*c + c4*c**2) * (1 - c) * (1 + c3*x1)
                  * (1 + tanh(c5*(x0 - c6)))

    Physical structure (Knowles/Cohen amyloid master equation). The analytic
    solution of the secondary-nucleation master equation is a sigmoid in time
    whose lag time and maximal growth rate are set by *independent*
    combinations of rate constants, and whose closed form behaves like a
    tanh-centered logistic. The autonomous polynomial parent could not
    separate lag duration from burst steepness because both were tied to a
    single rate polynomial; it had to inflate the reaction order (cubic term)
    to fake a sharp delayed onset, which hurt held-out shape loss.

    - c0            : primary nucleation source; tiny escape from c~=0 so the
                      trajectory can leave the baseline.
    - (c1 + c2*x2)*c: elongation proportional to existing aggregate; seed M0
                      (x2) boosts it, shortening the lag.
    - c4*c**2       : surface-catalyzed SECONDARY nucleation; superlinear burst.
    - (1 - c)       : mass-conservation saturation; plateaus at 1.
    - (1 + c3*x1)   : global rate scale set by initial monomer m0 (x1).
    - (1 + tanh(c5*(x0 - c6))) : smooth time gate controlling WHEN the burst
                      fires (c6 = onset/lag time) decoupled from HOW steep it
                      is (c4). Bounded in (0, 2), never zero, infinitely
                      differentiable. Since x0 is min-max normalized to [0, 1]
                      and c5 starts moderate (3.0), the tanh argument stays
                      small, so odeint and least-squares remain well
                      conditioned with no overflow or singularity. This
                      directly targets the half-response/onset-timing
                      component of the shape loss while keeping NMSE stable.

    The RHS uses only polynomial terms in c (degree 3 overall), linear x1/x2,
    and one bounded tanh of time: smooth, finite, well conditioned over the
    observed input ranges with no division, log, root, or variable exponent.
    Seven constants keep the per-dataset least-squares fit stable.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    concentration = x[3]

    csq = concentration * concentration
    rate = (
        c[0]
        + (c[1] + c[2] * seed) * concentration
        + c[4] * csq
    )
    saturation = 1 - concentration
    monomer_scale = 1 + c[3] * monomer

    # Smooth tanh time gate: decouples lag onset (c6) from burst steepness (c4).
    gate = 1 + sp.tanh(c[5] * (time - c[6]))

    expression = rate * saturation * monomer_scale * gate

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.05, 0.5, 0.0, 0.0, 1.0, 3.0, 0.3],
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