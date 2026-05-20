# EVOLVE-BLOCK-START
"""Symbolic regression seed where SkyDiscover evolves the equation template."""

from __future__ import annotations

from typing import Any

import sympy as sp
from numpy.typing import NDArray

from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)


def discover(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """
    Propose an equation structure and let the harness fit its constants.

    EvoX / AdaEvolve should improve the expression template below. PySR is used
    for expression export/evaluation conveniences while SkyDiscover remains the
    evolutionary algorithm producing candidate equations.
    """
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(X_train.shape[1] + 1)

    # Conservative seed: an affine equation. The evolved program should replace
    # this with richer symbolic structure using PySR-compatible operators.
    expression = c[-1] + sum(c[i] * x[i] for i in range(X_train.shape[1]))

    return evaluate_expression(
        sp.simplify(expression),
        X_train,
        y_train,
        X_val,
        y_val,
        constants=c,
    )


def run_discovery(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    """Entry point used by the evaluator subprocess."""
    return discover(X_train, y_train, X_val, y_val)


# EVOLVE-BLOCK-END


def _load_data():
    """Load deterministic Friedman #1 splits (matches evaluator)."""
    from sklearn.datasets import make_friedman1
    from sklearn.model_selection import train_test_split

    X, y = make_friedman1(
        n_samples=400,
        n_features=5,
        noise=0.1,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, random_state=42)


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = _load_data()
    result = run_discovery(X_train, y_train, X_val, y_val)
    print("equation:", result.get("equation"))
    print("nmse_val:", result.get("nmse_val"))
    print("combined_score:", result.get("combined_score"))
