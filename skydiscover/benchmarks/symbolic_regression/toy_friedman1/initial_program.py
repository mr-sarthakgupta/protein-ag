"""
Symbolic regression candidate: separate **structure** from **continuous parameters**.

Structure: linear model over a fixed library of unary transforms applied per feature
(Identity, Square, Sin, Cos, Log1pAbs). This defines the law's functional form.

Parameters: coefficients (including bias) are fit only on the training split using
Gaussian-process Bayesian optimization (scikit-optimize `gp_minimize`), with MSE
as the inner objective. Validation predictions use those parameters — no leakage.

Evaluator contract:
  fit_and_predict(X_train, y_train, X_test) -> dict

Returns:
  - y_pred: required
  - equation_template: SymPy expr usingSymbols p0, p1, ... (structure + parameter placeholders)
  - equation / equation_sympy: fitted numeric equation for inspection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import os


@dataclass(frozen=True)
class Transform:
    name: str

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sympy(self, s):
        raise NotImplementedError

    def to_str(self, var: str) -> str:
        raise NotImplementedError


class Identity(Transform):
    def __init__(self):
        super().__init__(name="id")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x

    def sympy(self, s):
        return s

    def to_str(self, var: str) -> str:
        return var


class Square(Transform):
    def __init__(self):
        super().__init__(name="sq")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x * x

    def sympy(self, s):
        return s**2

    def to_str(self, var: str) -> str:
        return f"({var})**2"


class Sin(Transform):
    def __init__(self):
        super().__init__(name="sin")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def sympy(self, s):
        import sympy as sp

        return sp.sin(s)

    def to_str(self, var: str) -> str:
        return f"sin({var})"


class Cos(Transform):
    def __init__(self):
        super().__init__(name="cos")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def sympy(self, s):
        import sympy as sp

        return sp.cos(s)

    def to_str(self, var: str) -> str:
        return f"cos({var})"


class Log1pAbs(Transform):
    def __init__(self, eps: float = 1e-12):
        super().__init__(name="log1pabs")
        self.eps = float(eps)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.abs(x) + self.eps)

    def sympy(self, s):
        import sympy as sp

        return sp.log(1 + sp.Abs(s) + self.eps)

    def to_str(self, var: str) -> str:
        return f"log(1 + abs({var}) + {self.eps:g})"


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def _make_library(n_features: int) -> List[Tuple[int, Transform]]:
    # Modest library keeps GP-BO on coefficients tractable; evolved code may add transforms.
    transforms: List[Transform] = [Identity(), Square(), Sin()]
    lib: List[Tuple[int, Transform]] = []
    for j in range(n_features):
        for t in transforms:
            lib.append((j, t))
    return lib


def _build_design_matrix(X: np.ndarray, lib: List[Tuple[int, Transform]]) -> np.ndarray:
    feats = []
    for j, t in lib:
        feats.append(t.apply(X[:, j]))
    Phi = np.column_stack(feats)
    Phi = np.column_stack([Phi, np.ones((Phi.shape[0], 1), dtype=float)])
    return Phi


def _ridge_theta0(Phi: np.ndarray, y_flat: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    n_feat = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(n_feat)
    b = Phi.T @ y_flat
    return np.linalg.solve(A, b).astype(float)


def _fit_params_gp_bo(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear coefficients Theta for y ~ Phi @ theta via GP-based Bayesian optimization
    on train MSE. Falls back to L-BFGS-B if scikit-optimize is unavailable.
    """
    y_flat = np.asarray(y, dtype=float).reshape(-1)
    n_feat = Phi.shape[1]

    def mse(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        pred = Phi @ theta
        return float(np.mean((pred - y_flat) ** 2))

    bounds_lo, bounds_hi = -12.0, 12.0
    theta0 = np.clip(_ridge_theta0(Phi, y_flat), bounds_lo, bounds_hi)

    try:
        from skopt import gp_minimize
        from skopt.space import Real

        n_calls = max(12, min(20, 8 + n_feat))
        if os.environ.get("SKYD_SR_GP_N_CALLS"):
            n_calls = max(5, min(80, int(os.environ["SKYD_SR_GP_N_CALLS"])))
        dimensions = [Real(bounds_lo, bounds_hi)] * n_feat
        res = gp_minimize(
            lambda t: mse(np.asarray(t, dtype=float)),
            dimensions,
            n_calls=n_calls,
            random_state=0,
            x0=[theta0.tolist()],
        )
        return np.asarray(res.x, dtype=float).reshape(-1)
    except Exception:
        from scipy.optimize import minimize

        sol = minimize(
            mse,
            theta0,
            method="L-BFGS-B",
            bounds=[(bounds_lo, bounds_hi)] * n_feat,
        )
        if not sol.success:
            return theta0
        return np.asarray(sol.x, dtype=float).reshape(-1)


def _to_sympy(
    w: np.ndarray,
    lib: List[Tuple[int, Transform]],
    feature_symbols: List,
    coef_threshold: float = 1e-8,
):
    import sympy as sp

    expr = sp.Float(0.0)
    bias = float(w[-1])
    if abs(bias) > coef_threshold:
        expr += sp.Float(bias)

    for k, (j, t) in enumerate(lib):
        ck = float(w[k])
        if abs(ck) <= coef_threshold:
            continue
        expr += sp.Float(ck) * t.sympy(feature_symbols[j])

    return sp.simplify(expr)


def _equation_template_sympy(lib: List[Tuple[int, Transform]], n_features: int):
    """Structure with parameter Symbols p0..p_{n-1} (bias is last coefficient)."""
    import sympy as sp

    feature_symbols = [sp.Symbol(f"x{i+1}") for i in range(n_features)]
    terms = []
    for k, (j, t) in enumerate(lib):
        terms.append(sp.Symbol(f"p{k}", real=True) * t.sympy(feature_symbols[j]))
    terms.append(sp.Symbol(f"p{len(lib)}", real=True))
    return sp.simplify(sp.Add(*terms))


def _to_equation_string(
    w: np.ndarray,
    lib: List[Tuple[int, Transform]],
    n_features: int,
    coef_threshold: float = 1e-8,
) -> str:
    terms: List[str] = []

    bias = float(w[-1])
    if abs(bias) > coef_threshold:
        terms.append(f"{bias:.6g}")

    for k, (j, t) in enumerate(lib):
        ck = float(w[k])
        if abs(ck) <= coef_threshold:
            continue
        var = f"x{j+1}"
        feat = t.to_str(var)
        terms.append(f"({ck:.6g})*({feat})")

    if not terms:
        return "0.0"
    return " + ".join(terms)


def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Dict[str, object]:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_test = np.asarray(X_test, dtype=float)

    Xs_train, mu, sigma = _standardize(X_train)
    Xs_test = (X_test - mu) / sigma

    lib = _make_library(Xs_train.shape[1])
    Phi_tr = _build_design_matrix(Xs_train, lib)
    Phi_te = _build_design_matrix(Xs_test, lib)

    w = _fit_params_gp_bo(Phi_tr, y_train)
    y_pred = Phi_te @ w

    eq_template = _equation_template_sympy(lib, Xs_train.shape[1])
    eq_str = _to_equation_string(w, lib, Xs_train.shape[1])

    try:
        import sympy as sp

        feature_symbols = [sp.Symbol(f"x{i+1}") for i in range(Xs_train.shape[1])]
        expr = _to_sympy(w, lib, feature_symbols)
        eq_sym = expr
    except Exception:
        eq_sym = None

    return {
        "y_pred": np.asarray(y_pred, dtype=float),
        "equation_template": eq_template,
        "equation": eq_str,
        "equation_sympy": eq_sym,
    }
