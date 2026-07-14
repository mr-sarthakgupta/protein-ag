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
    Physically-grounded amyloid aggregation ODE (Knowles/Cohen master-equation
    reduction) with inhibitor suppression of the secondary-nucleation channel
    AND inhibitor-induced lag extension.

    dc/dt = (plateau - c) * [ source
              + k2*m0*c*(c + lag + kseed*M0 + kinh*cd) / (1 + ki*cd + kic*cd*c) ]

    Rationale (building on the strongest prior candidate, score 0.9350):
    - (plateau - c): mass-conservation capacity factor. Aggregation halts as the
      accessible monomer pool is consumed. Bounded and smooth; sets the plateau.
    - source = kn*m0 + ks*M0 + b0: primary nucleation + seed + baseline flux,
      active at c = 0 so it sets the lag-phase onset. Left UNINHIBITED because
      inhibitors act predominantly on secondary nucleation, not primary flux.
    - autocatalytic term k2*m0*c*(c + onset): secondary-nucleation / elongation
      amplification. The c*(c + onset) form sharpens the sigmoid so the
      integrated trajectory reproduces lag-then-burst kinetics rewarded by the
      shape loss.
    - inhibitor_scale (1 + ki*cd + kic*cd*c) divides the autocatalytic channel:
      this encodes AMPLITUDE/RATE suppression (slower, flatter growth), a smooth
      factor >= 1 that reduces exactly to the uninhibited law at cd = 0.

    BREAKTHROUGH (this candidate): amyloid inhibitors have a second, distinct
    kinetic signature the amplitude term cannot capture -- they EXTEND the lag
    phase and DELAY the half-time t1/2 (temporal shift of the sigmoid
    inflection). In the Cohen/Meisl/Knowles master-equation reduction, damping a
    microscopic nucleation step both slows growth and postpones the inflection.
    We encode the temporal-delay signature by making the onset offset grow with
    inhibitor concentration:
        onset = c[5]^2 + c[8]^2*M0 + c[9]^2*cd
    A larger onset pushes the c*(c + onset) amplification to engage later,
    shifting the sigmoid inflection to a later time -- directly targeting the
    half-response/onset-timing shape loss (25% weight), which dominates the
    remaining error (shape 0.101 vs nmse 0.059). Seed (M0) shortens the
    effective lag and inhibitor (cd) lengthens it; both enter as positive
    squared constants so onset stays strictly positive, smooth, and finite. At
    cd = 0 the template collapses EXACTLY to the current best, so the change is
    low risk: it can only help or be fit near zero.

    All constants enter as squares (positivity / well-posedness). No variable
    exponents, logs, roots, or explicit time terms, so the RHS is smooth,
    finite, and well conditioned during least-squares fitting.

    NOTE ON RESEARCH: external web_search returned no results for every
    amyloid-kinetics query in this environment, the ML-tuned paper snippet
    search returned only unrelated ML papers, and the one locally cached
    reference (JBC M112.375345) was saved with 0 characters. This structural
    change is therefore justified from the established Cohen/Meisl/Knowles
    chemical-kinetics framework the template already uses (dual inhibition
    signature: amplitude suppression + lag/half-time delay) plus the local
    metric evidence that shape loss dominates.

    Features: x0 = time, x1 = m0, x2 = M0, x3 = cd, x4 = current state c.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(10)

    monomer = x[1]
    seed = x[2]
    inhibitor = x[3]
    concentration = x[4]

    plateau = c[0]
    capacity = plateau - concentration

    source = c[1] ** 2 * monomer + c[2] ** 2 * seed + c[3] ** 2

    # Onset offset controls when autocatalytic amplification engages in
    # c*(c + onset). Larger onset -> later sigmoid inflection (longer lag /
    # delayed half-time). Inhibitor (cd) EXTENDS the lag -- a distinct kinetic
    # signature the amplitude-suppressing inhibitor_scale cannot reproduce.
    onset = c[5] ** 2 + c[8] ** 2 * seed + c[9] ** 2 * inhibitor
    autocatalytic = c[4] ** 2 * monomer * concentration * (concentration + onset)

    inhibitor_scale = 1 + c[6] ** 2 * inhibitor + c[7] ** 2 * inhibitor * concentration

    expression = capacity * (source + autocatalytic / inhibitor_scale)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[1.0, 0.1, 0.1, 0.1, 1.0, 0.3, 1.0, 0.5, 0.1, 0.1],
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