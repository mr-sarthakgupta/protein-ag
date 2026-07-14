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

    Features: x0 = normalized elapsed time, x1 = normalized varying
    experimental parameter (concentration, pH, etc.), x2 = concentration c.

    Finke-Watzky autocatalytic ODE template with parameter-modulated rate.

    Pure logistic growth (rate * c * (plateau - c)) cannot produce the
    characteristic lag phase of nucleation-dependent aggregation: if the
    integrated state starts near c ~ 0 the multiplicative `c` factor keeps
    the derivative at zero, so no fibril mass ever forms.  The established
    governing law for amyloid / disease-relevant aggregation is the two-step
    Finke-Watzky mechanism, which adds a *primary nucleation* term that is
    independent of the current aggregate mass:

        d(c)/dt = (c0 + c1*c) * (1 + c2*x1) * (c3 - c)

    The `c0` term seeds growth from c ~ 0 (slow primary nucleation), the
    autocatalytic `c1*c` term drives the sharp self-accelerating elongation /
    secondary-nucleation phase, and the `(c3 - c)` factor enforces monomer
    depletion / saturation toward the final plateau `c3`.  The `(1 + c2*x1)`
    factor lets the varying experimental parameter (e.g. concentration, pH)
    globally modulate the kinetic rate without coupling too strongly.  All
    terms are polynomial, so the RHS is smooth, bounded, and free of
    singularities or invalid arithmetic during least-squares fitting.  Four
    fitted constants keep the form compact while capturing the universal
    nucleation + autocatalysis + saturation structure.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(4)

    time = x[0]
    parameter = x[1]
    concentration = x[2]

    rate = (c[0] + c[1] * concentration) * (1 + c[2] * parameter)
    plateau = c[3]

    expression = rate * (plateau - concentration)

    return evaluate_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
        initial_values=[0.5, 1.0, 0.0, 1.0],
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
