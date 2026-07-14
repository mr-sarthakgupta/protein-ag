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

    Asymmetric Gompertz sigmoid with a monomer-power elongation rate and an
    ADDITIVE, mechanistically-correct seed lag-shift via log1p — a natural
    closed-form model for secondary-nucleation-dominated polymerization:

        y = c4 * exp(-exp(-(c0 * x1^c1 * (x0 - c2*x1^c3) + log1p(c6*x2)))) + c5
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(7)

    time = x[0]
    monomer = x[1]
    seed = x[2]

    # Asymmetric (Gompertz) sigmoid for nucleation-dependent polymerization.
    # Amyloid growth curves are characteristically ASYMMETRIC: a sharp
    # post-lag rise followed by a SLOW approach to the plateau (the documented
    # signature of secondary-nucleation / autocatalytic growth in the
    # Knowles/Cohen/Meisl integrated-rate-law framework). The Gompertz
    # double-exponential is the canonical asymmetric nucleation-growth sigmoid
    # and is what made this backbone the strongest stable variant; a symmetric
    # tanh/logistic underfits the slow upper tail (and an attempted tanh swap
    # regressed), so the asymmetry is preserved here deliberately.
    #
    # KEY STRUCTURAL CHANGE (mechanistic seed effect, supported by the
    # secondary-nucleation integrated rate law):
    # In the self-consistent closed form M(t) ~ 1 - (1 + B*exp(kappa*t))^-a,
    # the aggregate mass grows like an autocatalytic exponential and the lag
    # time scales as tau_lag ~ (1/kappa) * ln(1 / seed_offset). Seeding does
    # NOT merely rescale the half-time; it advances a curve along the
    # amplification axis by an amount proportional to the LOGARITHM of the
    # seed term, i.e. an ADDITIVE shift of the exponent argument. The compact,
    # numerically stable encoding of a multiplicative seed amplification
    # (1 + c6*x2) turned into that additive exponent offset is log1p(c6*x2):
    #   - When x2 = 0 (unseeded curves) log1p(0) = 0, so the model reduces
    #     EXACTLY to the proven seedless Gompertz c2*x1^c3 lag — those
    #     datasets are byte-for-byte unchanged.
    #   - x2 >= 0 with c6 initialized positive keeps the log1p argument
    #     1 + c6*x2 >= 1 > 0, so there is no negative-log singularity, and the
    #     residual guard rejects any transient non-finite prediction.
    # This targets generalization to held-out seed concentrations (the seeded
    # evaluation-only datasets) while keeping the SAME 7-constant count
    # (no parsimony change; one log node replaces one division node).
    #
    # Numerical safety: for normalized t in [0,1] and a finite fitted rate the
    # inner exponent is bounded, so the double exponential stays globally
    # smooth and finite — no division by zero, no invalid powers.
    rate = c[0] * monomer ** c[1]
    half_time = c[2] * monomer ** c[3]
    seed_shift = sp.log(1 + c[6] * seed)
    plateau = c[4]
    baseline = c[5]

    growth = sp.exp(-sp.exp(-(rate * (time - half_time) + seed_shift)))
    expression = plateau * growth + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.1],
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
