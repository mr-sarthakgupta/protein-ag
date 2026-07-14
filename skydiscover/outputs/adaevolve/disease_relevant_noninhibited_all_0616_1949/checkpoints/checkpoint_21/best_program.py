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

    Richards / generalized-logistic sigmoid with a monomer-power elongation
    rate, an ADDITIVE log1p seed lag-shift, and a FITTED asymmetry exponent
    that lets each dataset interpolate between symmetric-logistic (primary
    nucleation) and sharp-rise/slow-tail Gompertz-like (secondary nucleation)
    shapes — while staying compact:

        z = c0 * x1^c1 * (x0 - c2) + log1p(c3*x2)
        y = c4 * (1 + exp(-z))^(-c6) + c5
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
    # RICHARDS GENERALIZED-LOGISTIC (breakthrough): generalize the fixed
    # Gompertz asymmetry to a tunable shape exponent c6 > 0.
    #   growth = (1 + exp(-z))^(-c6)
    # c6 = 1 is the symmetric logistic; large c6 approaches the sharp-rise /
    # slow-upper-tail Gompertz-like asymmetry of secondary-nucleation growth;
    # fractional c6 gives the opposite asymmetry. Each dataset chooses its own
    # tail behaviour, directly targeting the upper-tail residual that a rigid
    # Gompertz over/under-fits on the high-weight 10/11-curve Abeta datasets.
    #
    # NUMERICAL SAFETY: the power base (1 + exp(-z)) is strictly >= 1 > 0 for
    # every real z, so the negative power is always finite and real (no
    # zero/negative base, no complex result). For normalized t in [0,1] and
    # finite fitted constants z is bounded; exp(-z) cannot underflow to make
    # the base 0, and the residual finite-check rejects transients.
    #
    # PARSIMONY COMPENSATION: the Richards wrapper plus its shape constant adds
    # a few nodes versus Gompertz. To keep the bounded parsimony penalty
    # competitive, the half-time is reduced from a monomer power law
    # (c2*x1^c3) to a single constant c2: monomer dependence of the curve is
    # already carried by the elongation-rate power law c0*x1^c1 (which sets the
    # slope AND, through the rate, the effective lag), so the second monomer
    # power law was largely redundant. This frees ~4 nodes and one constant,
    # which funds the shape exponent without net complexity growth.
    #
    # Seed effect (retained): additive log1p(c3*x2) shifts the exponent
    # argument — when x2 = 0 (unseeded) log1p(0) = 0, so those datasets reduce
    # to the seedless form; for x2 >= 0 with c3 >= 0 the log argument stays
    # >= 1 > 0 (no negative-log singularity).
    rate = c[0] * monomer ** c[1]
    half_time = c[2]
    seed_shift = sp.log(1 + c[3] * seed)
    plateau = c[4]
    baseline = c[5]

    z = rate * (time - half_time) + seed_shift
    growth = (1 + sp.exp(-z)) ** (-c[6])
    expression = plateau * growth + baseline

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[5.0, 0.0, 0.3, 0.1, 1.0, 0.0, 1.0],
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
