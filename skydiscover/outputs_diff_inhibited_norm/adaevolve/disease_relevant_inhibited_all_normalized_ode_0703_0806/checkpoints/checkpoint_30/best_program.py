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
    Physically-grounded amyloid aggregation ODE (Cohen/Meisl/Knowles nucleated-
    polymerization master-equation reduction) with a DUAL inhibitor signature:
    amplitude suppression of secondary nucleation PLUS an inhibitor-driven
    exponential lag gate that prolongs the lag time / delays the half-time t1/2.

    dc/dt = (plateau - c) * [ source*exp(-lam*cd*t)
                              + k2*m0*c*(c + onset) / (1 + ki*cd + kic*cd*c) ]

    Research justification: in the Cohen-Knowles-Meisl framework the Abeta42
    sigmoid is characterized by two independent observables -- the half-time
    t1/2 (onset timing) and the maximum growth rate (slope). Inhibitors show two
    distinct phenotypes: (i) reducing the effective multiplication rate (flatter
    slope), and (ii) PROLONGING the lag time, which scales inversely with the
    primary-nucleation flux and shifts t1/2 later WITHOUT changing the plateau.
    (External web_search / research_papers snippet tools returned no usable
    amyloid-kinetics results in this environment, and the locally cached
    reference JBC M112.375345 was saved with 0 characters, so this structure is
    grounded in the established master-equation framework the template already
    uses plus the local metric evidence that shape/onset loss dominates.)

    - (plateau - c): mass-conservation capacity factor. Aggregation halts as the
      accessible monomer pool is consumed. Bounded, smooth; sets the plateau
      (unchanged by inhibitor, matching the observation that inhibitors delay
      but do not lower the final converted mass in these curves).
    - source = kn*m0 + ks*M0 + b0: primary nucleation + seed + baseline flux,
      the only nonzero term at c = 0, so it drives escape from the lag phase.
    - lag gate exp(-lam*cd*t): decays the nucleation source with inhibitor cd
      and elapsed normalized time t, postponing the moment the trajectory leaves
      the near-zero lag phase -> genuine t1/2 delay. The exponent is <= 0
      (squared constant, cd >= 0, t in [0,1] after min-max normalization), so
      the gate is strictly bounded in (0, 1], real, and overflow-free. At cd = 0
      the gate is exactly 1, collapsing to the uninhibited law -- a refinement.
    - autocatalytic k2*m0*c*(c + onset): secondary-nucleation / elongation
      amplification; c*(c + onset) sharpens the sigmoid for lag-then-burst
      kinetics rewarded by the shape loss.
    - inhibitor_scale (1 + ki*cd + kic*cd*c): amplitude/rate suppression of the
      secondary-nucleation channel; smooth, >= 1, reduces to uninhibited at cd=0.

    All coefficients enter squared (positivity / well-posedness). No variable
    exponents, logs, or roots; the single exp has a non-positive argument. 9
    fitted constants keep the least-squares fit well conditioned from one init.

    Features: x0 = time, x1 = m0, x2 = M0, x3 = cd, x4 = current state c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(9)

    time = x[0]
    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    # Mass-conservation capacity factor: aggregation halts as the accessible
    # monomer pool is consumed. Bounded, smooth; sets the plateau.
    plateau = c[0]
    capacity = plateau - concentration

    # Primary nucleation + seed + baseline source, active at c = 0 so it sets
    # lag-phase onset. Weakly monomer- and seed-scaled.
    source = c[1] ** 2 * monomer + c[2] ** 2 * seed + c[3] ** 2

    # Inhibitor-driven exponential LAG GATE on the primary source. exp(-c8^2 *
    # cd * t) is a smooth gate in (0, 1] because the exponent is <= 0 (squared
    # constant, non-negative cd, normalized t in [0, 1]). Higher inhibitor or
    # later time decays the effective nucleation flux, so the sigmoid ramps in
    # later -> genuine lag-phase extension / delayed half-time, directly
    # targeting the onset-timing shape loss. No overflow risk. At cd = 0 the
    # gate is exactly 1, so the RHS collapses to the uninhibited law.
    lag_gate = sp.exp(-c[8] ** 2 * inhibitor * time)
    source_gate = source * lag_gate

    # Autocatalytic (secondary-nucleation / elongation) amplification. The
    # c*(c + onset) form sharpens the sigmoid to reproduce lag-then-burst
    # kinetics rewarded by the shape loss. Scales with monomer.
    autocatalytic = c[4] ** 2 * monomer * concentration * (concentration + c[5] ** 2)

    # Inhibitor suppresses the secondary-nucleation channel (amplitude/rate
    # suppression): smooth, >= 1, reduces to the uninhibited law at cd = 0.
    inhibitor_scale = 1 + c[6] ** 2 * inhibitor + c[7] ** 2 * inhibitor * concentration

    expression = capacity * (source_gate + autocatalytic / inhibitor_scale)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.1, 0.1, 0.1, 1.0, 0.3, 1.0, 0.5, 0.5],
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