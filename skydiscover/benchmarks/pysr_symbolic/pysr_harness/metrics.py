"""Shared scoring helpers for symbolic regression benchmarks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def nmse(y_true: NDArray, y_pred: NDArray) -> float:
    """Normalized mean squared error."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        return float("inf")
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    var = float(np.var(y_true))
    if var <= 0.0:
        return float(np.mean((y_true - y_pred) ** 2))
    return float(np.mean((y_true - y_pred) ** 2) / var)


def combined_score_from_nmse(nmse_val: float) -> float:
    """Higher is better; zero for invalid or infinite NMSE."""
    if not np.isfinite(nmse_val):
        return 0.0
    return float(1.0 / (1.0 + max(nmse_val, 0.0)))
