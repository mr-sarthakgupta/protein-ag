# EVOLVE-BLOCK-START
"""ODE discovery seed for normalized Abeta42 inhibitor aggregation kinetics."""

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
    Compact seeded nucleation-elongation ODE with multiplicative inhibitor slowing.

    Physical reasoning (Oosawa / Knowles amyloid master-equation reduced form):
      * Aggregation of a normalized mass fraction c toward a plateau P is
        driven by (i) seed-dependent nucleation proportional to M0 and
        (ii) autocatalytic elongation proportional to available monomer m0
        times the existing aggregate mass c. The bounded ``plateau - c``
        capacity factor enforces mass conservation (growth stops at plateau).
      * The inhibitor cd suppresses the whole effective rate multiplicatively,
        1 / (1 + k * cd). This reproduces the dominant experimental signature
        in the data: increasing cd delays the half-response / lag time without
        changing the final plateau. When cd = 0 the rate reduces cleanly to the
        uninhibited kinetics.

    Feature scaling: m0, M0 and cd are parsed as raw molar values (~1e-6), so
    each is multiplied by 1e6 to bring the fitted constants to order one. This
    conditions the least-squares fit and lets the inhibitor term ``c2**2 * cd``
    actually reach order-one magnitude (the previous template could not scale
    x3 up and therefore ignored the inhibitor delay).

    Constants use squared symbols to keep every kinetic rate non-negative,
    which keeps the RHS smooth, sign-definite, and numerically stable. Only
    four constants are fitted, avoiding over-parameterization.

    Features: x0 = normalized time, x1 = initial monomer m0, x2 = seed M0,
    x3 = inhibitor concentration cd, and x4 = current normalized
    concentration/state.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(6)

    monomer = x[1] * 1e6
    seed = x[2] * 1e6
    inhibitor = x[3] * 1e6
    concentration = x[4]

    # Reduced Cohen/Knowles amyloid master equation as a bounded logistic ODE:
    #  * seed_nucleation  ~ M0            : seed-templated initiation.
    #  * primary_nucleation ~ m0          : monomer-only initiation. This term
    #    is essential for the lag phase and for the M0 = 0 curves, which still
    #    aggregate (measured half-time ~0.57) even with no seed. Without it the
    #    only c-independent source is the seed channel, so seed-free curves
    #    cannot initiate and the onset/shape term is systematically wrong.
    #  * elongation ~ m0 * c              : autocatalytic monomer addition.
    # The bounded (plateau - c) capacity factor enforces mass conservation so
    # growth halts at the plateau and the RHS stays smooth and sign-definite.
    # The inhibitor multiplicatively slows the whole effective rate,
    # 1 / (1 + k * cd): it delays the half-time (the dominant cd signature in
    # the data, half-time rising monotonically 0.2 -> 0.7 with cd) without
    # changing the plateau, and reduces cleanly to uninhibited kinetics at
    # cd = 0. Squared constants keep every rate non-negative; the denominator
    # is >= 1 so there are no singularities. Five constants stay well-conditioned.
    # Data-driven inhibitor structure. Measured half-times for the unseeded
    # (M0 = 0) curves rise SUPER-LINEARLY with cd: t_half ~ 0.077 (cd=0),
    # 0.10 (cd=5e-7), 0.14 (cd=1e-6), 0.22 (cd=2e-6). The marginal delay grows
    # when cd is doubled, the signature of cooperative / multi-site inhibitor
    # binding. A purely linear denominator 1/(1+k*cd) cannot bend the timing
    # curve this way, which leaves residual half-response/onset shape error on
    # the high-cd curves (the dominant term in the scored loss here).
    #
    # We therefore make the multiplicative inhibitor suppression cooperative:
    #   inhibitor_scale = 1 + k3^2 * cd + k5^2 * cd^2.
    # The quadratic term captures the accelerating delay at large cd while the
    # linear term preserves the mild delay at small cd; the two are separable
    # (one linear, one quadratic in cd). Both coefficients are squared and
    # cd >= 0, so inhibitor_scale >= 1 strictly: no singularities, smooth and
    # sign-definite, and cd = 0 recovers uninhibited kinetics exactly.
    #
    # The final plateau is kept cd-INDEPENDENT: the measured mean plateau is
    # essentially flat across cd (0.94-0.95), so a cd-dependent plateau would
    # add a weakly-constrained constant and risk overfitting the timing signal
    # into an amplitude term. Only one extra constant (six total) is added,
    # keeping the least-squares fit well-conditioned.
    seed_nucleation = c[0] ** 2 * seed
    primary_nucleation = c[1] ** 2 * monomer
    elongation = c[2] ** 2 * monomer * concentration
    inhibitor_scale = 1 + c[3] ** 2 * inhibitor + c[5] ** 2 * inhibitor ** 2
    plateau = c[4]
    capacity = plateau - concentration

    expression = (
        capacity
        * (seed_nucleation + primary_nucleation + elongation)
        / inhibitor_scale
    )

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 0.05, 0.5, 1.0, 0.92, 0.2],
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
    """Load the inhibitor dataset for local testing."""
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
