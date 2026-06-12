"""Evaluator for normalized ODE discovery across all disease-relevant non-inhibited proteins.

Discovers all fit.tsv datasets in the source data directory, evaluates a
candidate differential equation on each independently (fitting a separate set
of constants per dataset), and returns an aggregate score.
"""

from __future__ import annotations

import ast
import csv
import concurrent.futures
import importlib.util
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import importlib.util as _ilu

import numpy as np

# Import metrics directly to avoid triggering pysr_harness.__init__'s
# heavy pysr/Julia imports (only needed at candidate evaluation time).
_metrics_spec = _ilu.spec_from_file_location(
    "pysr_harness.metrics",
    str(BENCHMARK_ROOT / "pysr_harness" / "metrics.py"),
)
_metrics_mod = _ilu.module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(_metrics_mod)
combined_score_from_nmse = _metrics_mod.combined_score_from_nmse
_base_nmse = _metrics_mod.nmse

DATA_ROOT = Path("/home/mrsar/protein-ag/past-published-data/disease-relevant non-inhibited")
RANDOM_STATE = 42
TEST_SIZE = 0.35
# Entire fit.tsv datasets held out for cross-protein evaluation: all points go to
# validation and constants are fitted on that validation data at scoring time.
EVALUATION_ONLY_DATASETS = frozenset(
    {
        # Multi-curve cross-protein holdouts
        "Abeta/Cohen2013/Ab42_sec_IN_HOURS",  # 10 parameter curves
        "Abeta/Meisl2014/Ab40_sec_IN_HOURS",  # 11 parameter curves
        "htt/Kakkar2016/FIts/Kakkar2016_6kDa_polyQ_prim",  # 4 parameter curves
        "htt/Kar2011/Fits/Kar2011_Q30_without_lowest_conc_sec",  # 5 parameter curves
        "biofilm proteins/CsgA/CsgA_prim_IN_HOURS",  # 6 parameter curves
        "biofilm proteins/RdlB_Yang2017/Fits/5and20uM_prim",  # 2 parameter curves
        # Single-curve cross-protein holdouts
        "IAPP/Daval2010/Fits/Daval2010_prim",
        "IAPP/Daval2010/Fits/Daval2010_sec",
        "hnRNPA/Kim2013/Fits/Kim2013_A1_prim",
        "insulin/Fits/Nielsen2001_seeded_sec",
        "lysozyme/Fits/Hasecke2018_7uM_sec",
    }
)
GAUSSIAN_SMOOTH_SIGMA = float(os.environ.get("SKYDISCOVER_GAUSSIAN_SMOOTH_SIGMA", "1.0"))
ODE_MULTISTART_SCALES = tuple(
    float(value)
    for value in os.environ.get("SKYDISCOVER_ODE_MULTISTART_SCALES", "1.0,0.3,3.0").split(",")
    if value.strip()
)
SINGLE_EQUATION_ERROR_PREFIX = "Single-equation violation"
_ALLOWED_IMPORTS = {"sympy", "numpy"}
_ALLOWED_FROM_IMPORTS = {
    "__future__",
    "typing",
    "numpy.typing",
    "pysr_harness.equation_session",
}
_ALLOWED_HARNESS_IMPORTS = {
    "constant_symbols",
    "evaluate_expression",
    "feature_symbols",
}
_ALLOWED_EVALUATE_CALLS = {
    "constant_symbols",
    "evaluate_expression",
    "feature_symbols",
    "sp.exp",
    "sp.log",
    "sp.log1p",
    "sp.sqrt",
    "sp.sin",
    "sp.cos",
    "sp.tan",
    "sp.tanh",
    "sp.sinh",
    "sp.cosh",
}
_PROTECTED_NAMES = {
    "constant_symbols",
    "evaluate_expression",
    "feature_symbols",
    "single_equation_evaluation",
    "validate_single_equation_result",
}
_DISALLOWED_CONTROL_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.While,
    ast.If,
    ast.IfExp,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _is_module_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    if node.orelse:
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _has_postponed_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
    return False


def _function_annotations(node: ast.FunctionDef) -> list[ast.AST]:
    annotations: list[ast.AST] = []
    args = node.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        if arg.annotation is not None:
            annotations.append(arg.annotation)
    if args.vararg is not None and args.vararg.annotation is not None:
        annotations.append(args.vararg.annotation)
    if args.kwarg is not None and args.kwarg.annotation is not None:
        annotations.append(args.kwarg.annotation)
    if node.returns is not None:
        annotations.append(node.returns)
    return annotations


def _validate_top_level_function(node: ast.FunctionDef, *, postponed_annotations: bool) -> None:
    if node.name in _PROTECTED_NAMES:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: top-level function '{node.name}' "
            "shadows a protected scorer or harness name."
        )
    if node.decorator_list:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: top-level function decorators are not "
            "allowed because decorators execute during candidate import."
        )
    if node.args.defaults or node.args.kw_defaults:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: top-level function default arguments "
            "are not allowed because defaults execute during candidate import."
        )
    if _function_annotations(node) and not postponed_annotations:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: top-level function annotations require "
            "'from __future__ import annotations' because annotations otherwise "
            "execute during candidate import."
        )


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return ""


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names.update(_assigned_names(elt))
        return names
    return set()


def _validate_plain_import(alias: ast.alias) -> None:
    if alias.name not in _ALLOWED_IMPORTS:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: import '{alias.name}' is not allowed. "
            "Candidates may only import sympy/numpy and the equation harness."
        )
    if alias.name == "sympy" and alias.asname != "sp":
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: sympy must be imported canonically "
            "as 'import sympy as sp'."
        )
    if alias.name == "numpy" and alias.asname not in {None, "np"}:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: numpy may only be imported as "
            "'import numpy' or 'import numpy as np'."
        )


def _validate_evaluate_expression_data_args(call: ast.Call) -> set[int]:
    expected = ["X_train", "y_train", "X_val", "y_val"]
    if len(call.args) < 5:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: evaluate_expression() must be called "
            "as evaluate_expression(expression, X_train, y_train, X_val, y_val, ...)."
        )
    actual_data_args = call.args[1:5]
    if not all(
        isinstance(arg, ast.Name) and arg.id == expected_name
        for arg, expected_name in zip(actual_data_args, expected)
    ):
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: evaluate_expression() must use the "
            "canonical data arguments exactly as X_train, y_train, X_val, y_val. "
            "Do not fit constants on validation labels or reorder the split."
        )
    return {id(arg) for arg in call.args[1:5]}


def _x_train_shape_feature_arg_id(node: ast.AST) -> int | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "shape"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "X_train"
    ):
        return None
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and slice_node.value == 1:
        return id(node.value.value)
    return None


def _allowed_feature_symbols_data_ids(fn: ast.FunctionDef) -> set[int]:
    allowed: set[int] = set()
    for node in ast.walk(fn):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "feature_symbols"
            and len(node.args) == 1
        ):
            continue
        x_train_id = _x_train_shape_feature_arg_id(node.args[0])
        if x_train_id is not None:
            allowed.add(x_train_id)
    return allowed


def _validate_no_dataset_data_leak(fn: ast.FunctionDef, allowed_node_ids: set[int]) -> None:
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Name)
            and node.id in {"X_train", "y_train", "X_val", "y_val"}
            and id(node) not in allowed_node_ids
        ):
            raise ValueError(
                f"{SINGLE_EQUATION_ERROR_PREFIX}: dataset array '{node.id}' may only "
                "appear in feature_symbols(X_train.shape[1]) or as the canonical "
                "evaluate_expression(expression, X_train, y_train, X_val, y_val, ...) "
                "arguments. Do not use dataset values to choose expressions, mutate "
                "splits, or tune constants/initial values."
            )


def validate_candidate_source(program_path: str) -> None:
    """Reject reward-hacking program structure before importing candidate code."""
    source = Path(program_path).read_text()
    tree = ast.parse(source, filename=program_path)
    postponed_annotations = _has_postponed_annotations(tree)

    for node in tree.body:
        if _is_module_docstring(node) or _is_main_guard(node):
            continue
        if isinstance(node, ast.FunctionDef):
            _validate_top_level_function(node, postponed_annotations=postponed_annotations)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_plain_import(alias)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in _ALLOWED_FROM_IMPORTS:
                raise ValueError(
                    f"{SINGLE_EQUATION_ERROR_PREFIX}: from-import '{module}' is not allowed."
                )
            if module == "pysr_harness.equation_session":
                aliased = [alias.name for alias in node.names if alias.asname]
                if aliased:
                    raise ValueError(
                        f"{SINGLE_EQUATION_ERROR_PREFIX}: harness imports may not "
                        f"use aliases: {sorted(aliased)}."
                    )
                imported = {alias.name for alias in node.names}
                extra = imported - _ALLOWED_HARNESS_IMPORTS
                if extra:
                    raise ValueError(
                        f"{SINGLE_EQUATION_ERROR_PREFIX}: unsupported harness imports "
                        f"{sorted(extra)}. Use only feature_symbols, constant_symbols, "
                        "and evaluate_expression."
                    )
        else:
            raise ValueError(
                f"{SINGLE_EQUATION_ERROR_PREFIX}: top-level executable code is not "
                "allowed in candidate programs. Build and score the equation only "
                "inside evaluate_symbolic_candidate()."
            )

    eval_fns = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_symbolic_candidate"
    ]
    if len(eval_fns) != 1:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: program must define exactly one "
            "evaluate_symbolic_candidate() function."
        )
    fn = eval_fns[0]

    returns = [node for node in ast.walk(fn) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: evaluate_symbolic_candidate() must have "
            "exactly one return statement."
        )
    return_value = returns[0].value
    if not (
        isinstance(return_value, ast.Call)
        and isinstance(return_value.func, ast.Name)
        and return_value.func.id == "evaluate_expression"
    ):
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: evaluate_symbolic_candidate() must directly "
            "return evaluate_expression(...), with no wrapper or post-processing."
        )
    allowed_data_arg_ids = _validate_evaluate_expression_data_args(return_value)
    allowed_data_arg_ids.update(_allowed_feature_symbols_data_ids(fn))
    _validate_no_dataset_data_leak(fn, allowed_data_arg_ids)

    evaluate_calls = 0
    for node in ast.walk(fn):
        if isinstance(node, _DISALLOWED_CONTROL_NODES):
            raise ValueError(
                f"{SINGLE_EQUATION_ERROR_PREFIX}: control flow, nested definitions, "
                "comprehensions, and conditionals are not allowed inside "
                "evaluate_symbolic_candidate(). Build one straight-line equation."
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(
                f"{SINGLE_EQUATION_ERROR_PREFIX}: imports are not allowed inside "
                "evaluate_symbolic_candidate()."
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned = set().union(*(_assigned_names(target) for target in targets))
            protected = assigned & _PROTECTED_NAMES
            if protected:
                raise ValueError(
                    f"{SINGLE_EQUATION_ERROR_PREFIX}: cannot reassign protected scorer "
                    f"or harness names {sorted(protected)}."
                )
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "evaluate_expression":
                evaluate_calls += 1
            if name not in _ALLOWED_EVALUATE_CALLS:
                raise ValueError(
                    f"{SINGLE_EQUATION_ERROR_PREFIX}: call '{name or type(node.func).__name__}' "
                    "is not allowed inside evaluate_symbolic_candidate(). Allowed calls "
                    "are the symbol helpers, evaluate_expression, and basic sp.* math."
                )

    if evaluate_calls != 1:
        raise ValueError(
            f"{SINGLE_EQUATION_ERROR_PREFIX}: evaluate_symbolic_candidate() must call "
            f"evaluate_expression() exactly once; found {evaluate_calls} calls."
        )


def _parse_parameter_from_column(column_name: str) -> float | None:
    """Extract a monomer/protein concentration from a fit.tsv column header.

    Only concentration-like labels are accepted:
      - "... : 10uM"  or "10 uM"  → 10.0
      - "35uM" (bare, as in Abeta) → 35.0
      - "... : 3.95mM"             → 3950.0  (converted to uM)
      - "... : 0.3 mg/mL 22uM"    → 22.0    (prefer uM when both present)

    Seed percentages, pH values, ng/uL seed-material labels, and generic
    numeric fallbacks are intentionally ignored so x1 consistently represents
    the aggregating monomer/protein concentration when available.
    """
    text = column_name.strip()
    # Prefer the human-readable condition after the file-name prefix. Some
    # filenames contain seed concentrations that should not become x1.
    condition_text = text.rsplit(":", 1)[-1].strip().strip("\"'")

    # Prefer an explicit uM value
    m = re.search(r"(\d+(?:\.\d+)?)\s*uM", condition_text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # mM → convert to uM
    m = re.search(r"(\d+(?:\.\d+)?)\s*mM", condition_text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000.0

    # mg/mL (take the number)
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg/mL", condition_text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize one dataset column to 0..1, preserving constants as zeros."""
    values = np.asarray(values, dtype=float)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    scale = v_max - v_min
    if scale <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - v_min) / scale


def _load_fit_table(data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a wide fit.tsv and return X=[normalized time, raw parameter, c], y.

    The first column is always the time/measurement coordinate. Remaining
    columns are response curves. The varying parameter is extracted from
    concentration-like curve headers only; non-concentration columns are skipped.
    Time is min-max normalized to 0..1, but the parameter column is left in its
    parsed raw units. The response is normalized globally across the whole
    fit.tsv to 0..1. Files already rescaled to 0..1 keep that scale.
    """
    with data_file.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        response_headers = header[1:]
        parsed_params = [_parse_parameter_from_column(h) for h in response_headers]

        features: list[list[float]] = []
        targets: list[float] = []
        for row in reader:
            if not row:
                continue
            measurement_x = float(row[0])
            for param, value in zip(parsed_params, row[1:]):
                if param is None:
                    continue
                if value == "" or value.strip().lower() == "nan":
                    continue
                features.append([measurement_x, param])
                targets.append(float(value))

    if not features:
        raise ValueError(f"No regression samples loaded from {data_file}")

    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    time_norm = _minmax_normalize(X[:, 0])
    parameter_raw = X[:, 1]
    y_norm = _minmax_normalize(y)

    # x2 is the concentration state c used by ODE RHS templates. During scoring
    # it is supplied from odeint's dynamic state, not read from candidate code.
    X_ode = np.column_stack([time_norm, parameter_raw, y_norm])
    return X_ode, y_norm


def discover_datasets(root: Path = DATA_ROOT) -> list[tuple[str, Path]]:
    """Recursively find all fit.tsv files and return (dataset_name, path) pairs."""
    datasets: list[tuple[str, Path]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "fit.tsv" in filenames:
            fit_path = Path(dirpath) / "fit.tsv"
            rel = fit_path.parent.relative_to(root)
            name = str(rel).replace(os.sep, "/")
            datasets.append((name, fit_path))
    datasets.sort(key=lambda x: x[0])
    return datasets


def _trajectory_preserving_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split whole parameter curves into fit (train) and NMSE-evaluation (val) sets."""
    parameters = np.unique(X[:, 1])
    rng = np.random.default_rng(RANDOM_STATE)

    if len(parameters) < 3:
        empty_X = np.empty((0, X.shape[1]), dtype=float)
        empty_y = np.empty(0, dtype=float)
        return X, empty_X, y, empty_y

    shuffled = parameters[rng.permutation(len(parameters))]
    n_val_curves = min(
        max(1, int(round(len(parameters) * TEST_SIZE))), len(parameters) - 1
    )
    val_params = {float(p) for p in shuffled[:n_val_curves]}

    fit_indices: list[int] = []
    val_indices: list[int] = []
    for parameter in parameters:
        curve_idx = np.flatnonzero(X[:, 1] == parameter)
        if float(parameter) in val_params:
            val_indices.extend(curve_idx.tolist())
        else:
            fit_indices.extend(curve_idx.tolist())

    fit_idx = np.sort(np.asarray(fit_indices, dtype=int))
    val_idx = np.sort(np.asarray(val_indices, dtype=int))
    return X[fit_idx], X[val_idx], y[fit_idx], y[val_idx]


def _evaluation_only_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out an entire dataset for validation-only cross-protein evaluation."""
    empty_X = np.empty((0, X.shape[1]), dtype=float)
    empty_y = np.empty(0, dtype=float)
    return empty_X, X, empty_y, y


def _dataset_name_from_path(data_file: Path, root: Path = DATA_ROOT) -> str:
    return str(data_file.parent.relative_to(root)).replace(os.sep, "/")


def load_dataset(
    data_file: Path,
    *,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a single fit.tsv and return deterministic trajectory-preserving split."""
    X, y = _load_fit_table(data_file)
    name = dataset_name or _dataset_name_from_path(data_file)
    if name in EVALUATION_ONLY_DATASETS:
        return _evaluation_only_split(X, y)
    return _trajectory_preserving_split(X, y)


def load_all_datasets(
    root: Path = DATA_ROOT,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load all datasets and return list of (name, X_train, X_val, y_train, y_val)."""
    entries = discover_datasets(root)
    results = []
    for name, path in entries:
        try:
            X_train, X_val, y_train, y_val = load_dataset(path, dataset_name=name)
            results.append((name, X_train, X_val, y_train, y_val))
        except Exception as exc:
            print(f"Warning: skipping dataset {name}: {exc}")
    return results


NUM_WORKERS = int(os.environ.get("SKYDISCOVER_EVAL_WORKERS", min(8, os.cpu_count() or 4)))
PARSIMONY_WEIGHT = float(os.environ.get("SKYDISCOVER_PARSIMONY_WEIGHT", "0.20"))
PARSIMONY_COMPLEXITY_SCALE = float(
    os.environ.get("SKYDISCOVER_PARSIMONY_COMPLEXITY_SCALE", "160")
)


def _parsimony_penalty_factor(complexity: float) -> float:
    """Bounded penalty factor for expression complexity."""
    if not np.isfinite(complexity) or PARSIMONY_COMPLEXITY_SCALE <= 0:
        return 1.0
    normalized = min(max(float(complexity), 0.0) / PARSIMONY_COMPLEXITY_SCALE, 1.0)
    return max(0.0, 1.0 - PARSIMONY_WEIGHT * normalized)


def _gaussian_smooth_1d(values: np.ndarray, sigma: float = GAUSSIAN_SMOOTH_SIGMA) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 3 or sigma <= 0.0:
        return values
    radius = max(1, int(round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _curve_smoothed_target(X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Smooth targets independently for each parameter curve in time order."""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if X.shape[0] != y_true.shape[0] or X.shape[1] < 2:
        return _gaussian_smooth_1d(y_true)

    smoothed = np.empty_like(y_true, dtype=float)
    for parameter in np.unique(X[:, 1]):
        curve_idx = np.flatnonzero(X[:, 1] == parameter)
        ordered_idx = curve_idx[np.argsort(X[curve_idx, 0])]
        smoothed_curve = _gaussian_smooth_1d(y_true[ordered_idx])
        smoothed[ordered_idx] = smoothed_curve
    return smoothed


def _normalized_mse_with_reference_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_values: np.ndarray,
) -> float:
    """MSE normalized by full dataset scale, not by a flat validation segment."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    reference_values = np.asarray(reference_values, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        return float("inf")
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    var = float(np.var(reference_values))
    if var <= 0.0:
        return float(np.mean((y_true - y_pred) ** 2))
    return float(np.mean((y_true - y_pred) ** 2) / var)


def _nmse_with_gaussian_smoothed_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray | None = None,
    reference_values: np.ndarray | None = None,
) -> float:
    reference = y_true if reference_values is None else reference_values
    return _normalized_mse_with_reference_variance(y_true, y_pred, reference)


def _ode_predictions(
    rhs_fn: Any,
    X: np.ndarray,
    y_known: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Integrate dc/dt=f(t, parameter, c) per parameter curve on one split."""
    import warnings

    from scipy.integrate import odeint

    X = np.asarray(X, dtype=float)
    y_known = np.asarray(y_known, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    predictions = np.full_like(y_known, np.nan, dtype=float)

    for parameter in np.unique(X[:, 1]):
        curve_idx = np.flatnonzero(X[:, 1] == parameter)
        ordered_idx = curve_idx[np.argsort(X[curve_idx, 0])]
        times = X[ordered_idx, 0]
        observed = y_known[ordered_idx]

        if len(ordered_idx) == 1 or float(times[-1] - times[0]) <= 0.0:
            predictions[ordered_idx] = observed[0]
            continue

        def dc_dt(c_state: np.ndarray, time_value: float) -> float:
            c_value = float(np.asarray(c_state, dtype=float).reshape(-1)[0])
            value = rhs_fn(float(time_value), float(parameter), c_value, *theta)
            value = float(np.asarray(value, dtype=float).reshape(-1)[0])
            if not np.isfinite(value):
                raise FloatingPointError("non-finite ODE derivative")
            return value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrated = odeint(
                dc_dt,
                float(observed[0]),
                times,
                mxstep=1000,
            )
        curve_pred = np.asarray(integrated, dtype=float).reshape(-1)
        if curve_pred.shape != observed.shape or not np.all(np.isfinite(curve_pred)):
            raise FloatingPointError("non-finite ODE trajectory")
        predictions[ordered_idx] = curve_pred

    return predictions


def evaluate_ode_expression(
    expression: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    constants: Any | None = None,
    initial_values: Any | None = None,
    max_nfev: int = 300,
) -> dict[str, Any]:
    """Score a proposed ODE RHS by integrating predicted concentration curves."""
    import sympy as sp
    from pysr_harness import equation_session
    from scipy.optimize import least_squares

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    evaluation_only = X_train.shape[0] == 0 and X_val.shape[0] > 0
    train_only = X_val.shape[0] == 0 and X_train.shape[0] > 0
    if evaluation_only:
        X_fit, y_fit = X_val, y_val
    else:
        X_fit, y_fit = X_train, y_train
    y_reference = np.concatenate([y_train, y_val]) if y_train.size else y_val
    n_features = X_fit.shape[1] if X_fit.shape[0] else X_val.shape[1]
    feature_names = [f"x{i}" for i in range(n_features)]
    template = equation_session._as_sympy_expr(expression, feature_names)
    equation_session._record_scored_template(template)

    if constants is None:
        feature_set = set(equation_session.feature_symbols(n_features))
        constants = sorted(template.free_symbols - feature_set, key=lambda s: s.name)
    constants = list(constants)
    equation_session._validate_expression_template(template, constants)

    if initial_values is None:
        base_x0 = np.ones(len(constants), dtype=float)
    else:
        base_x0 = np.asarray(initial_values, dtype=float).reshape(-1)
        if base_x0.size != len(constants):
            raise ValueError("initial_values must match the number of constants")

    feature_syms = equation_session.feature_symbols(n_features)
    rhs_fn = sp.lambdify(feature_syms + constants, template, modules=["numpy"])

    if not constants:
        y_pred_fit = _ode_predictions(rhs_fn, X_fit, y_fit, np.asarray([], dtype=float))
        y_pred_val = _ode_predictions(rhs_fn, X_val, y_val, np.asarray([], dtype=float))
        nmse_train = (
            float("nan")
            if evaluation_only
            else _nmse_with_gaussian_smoothed_target(
                y_train, y_pred_fit, X_train, y_reference
            )
        )
        nmse_val = (
            float("nan")
            if train_only
            else _nmse_with_gaussian_smoothed_target(y_val, y_pred_val, X_val, y_reference)
        )
        result = {
            "equation_template": f"d(c)/dt = {template}",
            "equation": f"d(c)/dt = {template}",
            "constants": {},
            "loss": float(np.mean((y_pred_fit - y_fit) ** 2)),
            "complexity": equation_session._complexity(template),
            "nmse_train": nmse_train,
            "nmse_val": nmse_val,
            "n_val_points": int(len(y_val)),
            "evaluation_only": evaluation_only,
            "train_only": train_only,
            "combined_score": (
                0.0
                if train_only or not np.isfinite(nmse_val)
                else combined_score_from_nmse(nmse_val)
            ),
        }
        equation_session._record_scorer_result(result)
        return result

    def residual(theta: np.ndarray) -> np.ndarray:
        try:
            prediction = _ode_predictions(rhs_fn, X_fit, y_fit, theta)
        except Exception:
            return np.full_like(y_fit, 1e12, dtype=float)
        if prediction.shape != y_fit.shape or not np.all(np.isfinite(prediction)):
            return np.full_like(y_fit, 1e12, dtype=float)
        return prediction - y_fit

    initial_candidates: list[np.ndarray] = []
    for scale in ODE_MULTISTART_SCALES or (1.0,):
        candidate = base_x0 * float(scale)
        if not any(np.array_equal(candidate, existing) for existing in initial_candidates):
            initial_candidates.append(candidate)

    import warnings

    best_fit = None
    best_loss = float("inf")
    with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for initial_candidate in initial_candidates:
            fit = least_squares(
                residual,
                initial_candidate,
                loss="soft_l1",
                max_nfev=max_nfev,
            )
            fit_residual = residual(fit.x)
            fit_loss = float(np.mean(fit_residual**2))
            if fit_loss < best_loss:
                best_loss = fit_loss
                best_fit = fit

    if best_fit is None:
        raise RuntimeError("constant fitting did not produce a result")

    fitted_constants = {
        str(symbol): float(value) for symbol, value in zip(constants, best_fit.x)
    }
    substitutions = dict(zip(constants, best_fit.x))
    fitted_expr = template.subs(substitutions)
    y_pred_fit = _ode_predictions(rhs_fn, X_fit, y_fit, best_fit.x)
    y_pred_val = _ode_predictions(rhs_fn, X_val, y_val, best_fit.x)
    nmse_train = (
        float("nan")
        if evaluation_only
        else _nmse_with_gaussian_smoothed_target(
            y_train, y_pred_fit, X_train, y_reference
        )
    )
    nmse_val = (
        float("nan")
        if train_only
        else _nmse_with_gaussian_smoothed_target(y_val, y_pred_val, X_val, y_reference)
    )
    loss = float(np.mean((y_pred_fit - y_fit) ** 2))

    result = {
        "equation_template": f"d(c)/dt = {template}",
        "equation": f"d(c)/dt = {fitted_expr}",
        "constants": fitted_constants,
        "loss": loss,
        "complexity": equation_session._complexity(template),
        "nmse_train": nmse_train,
        "nmse_val": nmse_val,
        "n_val_points": int(len(y_val)),
        "evaluation_only": evaluation_only,
        "train_only": train_only,
        "combined_score": (
            0.0
            if train_only or not np.isfinite(nmse_val)
            else combined_score_from_nmse(nmse_val)
        ),
    }
    equation_session._record_scorer_result(result)
    return result


def _point_weighted_mean(
    values: list[float],
    weights: list[int],
) -> float:
    """Average values with one weight per evaluation point."""
    total_weight = sum(weights)
    if total_weight <= 0:
        return float("inf")
    weighted = sum(
        float(value) * float(weight)
        for value, weight in zip(values, weights)
        if np.isfinite(float(value))
    )
    if not np.isfinite(weighted):
        return float("inf")
    return float(weighted / total_weight)


def _aggregate_per_dataset_results(per_dataset_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-dataset metrics without letting failed datasets disappear."""
    scoring_results = {
        name: result
        for name, result in per_dataset_results.items()
        if int(result.get("n_val_points", 0)) > 0
    }
    raw_nmse_vals = [float(r["nmse_val"]) for r in scoring_results.values()]
    nmse_vals = list(raw_nmse_vals)
    train_nmse_vals = [
        float(r["nmse_train"])
        for r in per_dataset_results.values()
        if np.isfinite(float(r.get("nmse_train", float("inf"))))
    ]
    complexity_vals = []
    for r in per_dataset_results.values():
        try:
            value = float(r.get("complexity", float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            complexity_vals.append(value)
    aggregate_complexity = float(max(complexity_vals)) if complexity_vals else 0.0

    violation_errors = sorted(
        {
            str(r.get("error"))
            for r in per_dataset_results.values()
            if r.get("error") and SINGLE_EQUATION_ERROR_PREFIX in str(r.get("error"))
        }
    )
    failed_datasets = sorted(
        name
        for name, result in scoring_results.items()
        if result.get("error") or not np.isfinite(float(result.get("nmse_val", float("inf"))))
    )
    templates = sorted(
        {
            str(r.get("equation_template"))
            for r in per_dataset_results.values()
            if r.get("combined_score", 0.0) > 0 and r.get("equation_template")
        }
    )

    eq_template = ""
    for r in per_dataset_results.values():
        if r.get("equation"):
            eq_template = r["equation"]
            break

    hard_cases = [
        {"dataset": name, "nmse_val": float(result["nmse_val"])}
        for name, result in scoring_results.items()
        if np.isfinite(float(result.get("nmse_val", float("inf"))))
    ]
    hard_cases.sort(key=lambda item: item["nmse_val"], reverse=True)
    hard_cases = hard_cases[:5]
    hard_case_feedback = "; ".join(
        f"{case['dataset']} nmse={case['nmse_val']:.4g}" for case in hard_cases
    )

    violation_error = None
    if violation_errors:
        violation_error = violation_errors[0]
    elif failed_datasets:
        examples = ", ".join(failed_datasets[:3])
        violation_error = (
            "Single-equation violation: candidate failed on "
            f"{len(failed_datasets)}/{len(scoring_results)} validation datasets. "
            "Failed datasets are not dropped from the aggregate reward. "
            f"Examples: {examples}"
        )
    elif len(templates) > 1:
        examples = "; ".join(templates[:3])
        violation_error = (
            "Single-equation violation: evaluate_symbolic_candidate() produced "
            f"{len(templates)} distinct equation templates across datasets. "
            "The base equation must be identical for all datasets; only fitted "
            f"constants may vary. Example templates: {examples}"
        )

    if violation_error:
        agg_nmse = float("inf")
        mean_nmse = float("inf")
        agg_train_nmse = float("inf")
        agg_combined = 0.0
        n_successful = 0
    else:
        val_point_weights = [
            int(r.get("n_val_points", 0)) for r in scoring_results.values()
        ]
        mean_nmse = _point_weighted_mean(nmse_vals, val_point_weights)
        raw_mean_nmse = _point_weighted_mean(raw_nmse_vals, val_point_weights)
        eval_only_results = {
            name: result
            for name, result in scoring_results.items()
            if result.get("evaluation_only")
        }
        eval_only_nmse_vals = [
            float(result["nmse_val"])
            for result in eval_only_results.values()
            if np.isfinite(float(result.get("nmse_val", float("inf"))))
        ]
        eval_only_weights = [
            int(result.get("n_val_points", 0))
            for result in eval_only_results.values()
            if np.isfinite(float(result.get("nmse_val", float("inf"))))
        ]
        mean_nmse_eval_only = (
            _point_weighted_mean(eval_only_nmse_vals, eval_only_weights)
            if eval_only_nmse_vals
            else float("nan")
        )
        agg_nmse = mean_nmse
        agg_train_nmse = float(np.mean(train_nmse_vals)) if train_nmse_vals else float("inf")
        raw_combined = (
            float(1.0 / (1.0 + max(agg_nmse, 0.0))) if np.isfinite(agg_nmse) else 0.0
        )
        parsimony_factor = _parsimony_penalty_factor(aggregate_complexity)
        agg_combined = raw_combined * parsimony_factor
        n_successful = sum(1 for value in nmse_vals if np.isfinite(value))

    aggregated = {
        "equation": eq_template,
        "nmse_train": agg_train_nmse,
        "nmse_val": agg_nmse,
        "mean_nmse_val": mean_nmse,
        "mean_raw_nmse_val": raw_mean_nmse if not violation_error else float("inf"),
        "combined_score": agg_combined,
        "fit_score": raw_combined if not violation_error else 0.0,
        "complexity": aggregate_complexity,
        "parsimony_weight": PARSIMONY_WEIGHT,
        "parsimony_penalty_factor": (
            _parsimony_penalty_factor(aggregate_complexity) if not violation_error else 1.0
        ),
        "n_datasets": len(scoring_results),
        "n_total_datasets": len(per_dataset_results),
        "n_train_only_datasets": len(per_dataset_results) - len(scoring_results),
        "n_successful": n_successful,
        "n_evaluation_only_datasets": len(
            [name for name, result in per_dataset_results.items() if result.get("evaluation_only")]
        ),
        "mean_nmse_eval_only": (
            mean_nmse_eval_only if not violation_error else float("inf")
        ),
        "eval_workers": NUM_WORKERS,
        "hard_cases": hard_cases,
        "hard_case_feedback": hard_case_feedback,
        "per_dataset": per_dataset_results,
    }
    if violation_error:
        aggregated["error"] = violation_error
    return aggregated


def evaluate_candidate_with_timeout(program_path: str, timeout_seconds: int = 3000) -> dict:
    """Evaluate a candidate program across ALL datasets in a subprocess.

    The subprocess loads every dataset, calls the candidate's
    evaluate_symbolic_candidate() once per dataset (fitting independent
    constants each time) using parallel threads, and returns aggregated
    metrics.
    """
    validate_candidate_source(program_path)
    harness_root = str(BENCHMARK_ROOT)
    evaluator_path = str(Path(__file__).resolve())
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import importlib.util
import os
import pickle
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, {harness_root!r})

program_path = {program_path!r}
evaluator_path = {evaluator_path!r}
results_path = {temp_file.name + ".results"!r}
NUM_WORKERS = int(os.environ.get("SKYDISCOVER_EVAL_WORKERS", min(8, os.cpu_count() or 4)))

def _write_payload_and_exit(payload):
    with open(results_path, "wb") as f:
        pickle.dump(payload, f)
    sys.stdout.flush()
    sys.stderr.flush()
    # PySR/PythonCall can leave Julia runtime threads/finalizers alive at
    # interpreter shutdown. The parent only needs the pickle payload.
    os._exit(0)

try:
    evaluator_spec = importlib.util.spec_from_file_location("benchmark_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator)
    evaluator.validate_candidate_source(program_path)

    import pysr_harness.equation_session as equation_session
    equation_session.evaluate_expression = evaluator.evaluate_ode_expression
    equation_session.nmse = evaluator._nmse_with_gaussian_smoothed_target
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    from pysr_harness.equation_session import (
        single_equation_evaluation,
        validate_single_equation_result,
    )

    fn = getattr(program, "evaluate_symbolic_candidate", None) or getattr(
        program, "run_discovery", None
    )
    if fn is None:
        raise AttributeError("Program must define evaluate_symbolic_candidate() or run_discovery()")

    datasets = evaluator.load_all_datasets()

    def _eval_one(args):
        ds_name, X_train, X_val, y_train, y_val = args
        try:
            with single_equation_evaluation():
                result = fn(X_train, y_train, X_val, y_val)
                validate_single_equation_result(result)
            ds_nmse = float(result.get("nmse_val", float("inf")))
            ds_train_nmse = float(result.get("nmse_train", float("inf")))
            ds_combined = float(result.get("combined_score", 0.0))
            return ds_name, {{
                "nmse_train": ds_train_nmse,
                "nmse_val": ds_nmse,
                "n_val_points": int(result.get("n_val_points", 0)),
                "evaluation_only": bool(result.get("evaluation_only", False)),
                "combined_score": ds_combined,
                "complexity": float(result.get("complexity", 0.0)),
                "equation": str(result.get("equation", "")),
                "equation_template": str(result.get("equation_template", "")),
                "constants": result.get("constants", {{}}),
            }}
        except Exception as exc:
            return ds_name, {{
                "nmse_train": float("inf"),
                "nmse_val": float("inf"),
                "n_val_points": int(len(y_val)),
                "evaluation_only": bool(X_train.shape[0] == 0 and X_val.shape[0] > 0),
                "combined_score": 0.0,
                "error": str(exc),
            }}

    per_dataset_results = {{}}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(_eval_one, dataset) for dataset in datasets]
        for future in as_completed(futures):
            ds_name, ds_result = future.result()
            per_dataset_results[ds_name] = ds_result

    aggregated = evaluator._aggregate_per_dataset_results(per_dataset_results)

    _write_payload_and_exit({{"result": aggregated}})
except Exception as exc:
    traceback.print_exc()
    _write_payload_and_exit({{"error": str(exc)}})
"""
        temp_file.write(script.encode())
        temp_path = temp_file.name

    results_path = f"{temp_path}.results"
    try:
        process = subprocess.Popen(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        if stdout:
            print(stdout.decode())
        if stderr:
            print(stderr.decode())
        if process.returncode != 0:
            raise RuntimeError(f"Subprocess exited with code {process.returncode}")
        if not os.path.exists(results_path):
            raise RuntimeError("Results file not found")
        with open(results_path, "rb") as f:
            payload = pickle.load(f)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload["result"]
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError(f"Program execution timed out after {timeout_seconds}s")
    finally:
        for path in (temp_path, results_path):
            if os.path.exists(path):
                os.unlink(path)


def _metrics_from_result(result: dict, eval_time: float) -> dict:
    """Normalize aggregated harness output into SkyDiscover evaluator metrics."""
    nmse_val = float(result.get("nmse_val", float("inf")))
    combined = float(result.get("combined_score", combined_score_from_nmse(nmse_val)))
    if not np.isfinite(combined):
        combined = 0.0

    return {
        "equation": str(result.get("equation", "")),
        "loss": nmse_val,
        "complexity": float(result.get("complexity", 0.0)),
        "nmse_train": float(result.get("nmse_train", float("inf"))),
        "nmse_val": nmse_val,
        "mean_nmse_val": float(result.get("mean_nmse_val", nmse_val)),
        "mean_raw_nmse_val": float(result.get("mean_raw_nmse_val", nmse_val)),
        "combined_score": combined,
        "fit_score": float(result.get("fit_score", combined)),
        "parsimony_weight": float(result.get("parsimony_weight", PARSIMONY_WEIGHT)),
        "parsimony_penalty_factor": float(result.get("parsimony_penalty_factor", 1.0)),
        "eval_time": float(eval_time),
        "n_datasets": result.get("n_datasets", 0),
        "n_successful": result.get("n_successful", 0),
        "eval_workers": result.get("eval_workers", NUM_WORKERS),
        "hard_cases": result.get("hard_cases", []),
        "hard_case_feedback": str(result.get("hard_case_feedback", "")),
        "per_dataset": result.get("per_dataset", {}),
        "error": str(result.get("error", "")),
    }


def evaluate(program_path: str) -> dict:
    """Evaluate one evolved symbolic-equation candidate across all datasets."""
    try:
        start = time.time()
        result = evaluate_candidate_with_timeout(program_path, timeout_seconds=3000)
        metrics = _metrics_from_result(result, time.time() - start)
        print(
            "Evaluation: "
            f"nmse_val={metrics['nmse_val']:.6f}, "
            f"mean_nmse_val={metrics['mean_nmse_val']:.6f}, "
            f"mean_raw_nmse_val={metrics['mean_raw_nmse_val']:.6f}, "
            f"combined_score={metrics['combined_score']:.6f}, "
            f"n_datasets={metrics['n_datasets']}, "
            f"n_successful={metrics['n_successful']}, "
            f"time={metrics['eval_time']:.2f}s"
        )
        print(f"Equation template: {metrics['equation']}")
        if metrics.get("hard_case_feedback"):
            print(f"Hard-case feedback: {metrics['hard_case_feedback']}")
        if metrics.get("error"):
            print(f"Evaluator feedback: {metrics['error']}")
        return metrics
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        traceback.print_exc()
        return {
            "equation": "",
            "loss": float("inf"),
            "complexity": 0.0,
            "nmse_train": float("inf"),
            "nmse_val": float("inf"),
            "mean_nmse_val": float("inf"),
            "mean_raw_nmse_val": float("inf"),
            "combined_score": 0.0,
            "eval_time": 0.0,
            "n_datasets": 0,
            "n_successful": 0,
            "hard_cases": [],
            "hard_case_feedback": "",
            "per_dataset": {},
            "error": str(exc),
        }


def _eval_stage1_worker(args):
    """Module-level worker for stage 1 parallel evaluation."""
    name, X_train, X_val, y_train, y_val, program_path = args
    harness_root = str(BENCHMARK_ROOT)
    if harness_root not in sys.path:
        sys.path.insert(0, harness_root)
    validate_candidate_source(program_path)
    import pysr_harness.equation_session as equation_session
    equation_session.evaluate_expression = evaluate_ode_expression
    equation_session.nmse = _nmse_with_gaussian_smoothed_target
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    from pysr_harness.equation_session import (
        single_equation_evaluation,
        validate_single_equation_result,
    )

    fn = getattr(program, "evaluate_symbolic_candidate", None) or getattr(
        program, "run_discovery", None
    )
    if fn is None:
        return 0.0, "missing evaluate_symbolic_candidate()", ""
    try:
        with single_equation_evaluation():
            result = fn(X_train, y_train, X_val, y_val)
            validate_single_equation_result(result)
        return float(result.get("combined_score", 0.0)), None, str(
            result.get("equation_template", "")
        )
    except Exception as exc:
        return 0.0, str(exc), ""


def evaluate_stage1(program_path: str) -> dict:
    """Fast cascade stage: evaluate on a small subset of datasets with reduced data."""
    try:
        harness_root = str(BENCHMARK_ROOT)
        if harness_root not in sys.path:
            sys.path.insert(0, harness_root)

        all_datasets = load_all_datasets()
        scoring_datasets = [
            dataset
            for dataset in all_datasets
            if dataset[0] not in EVALUATION_ONLY_DATASETS and dataset[4].size > 0
        ]
        rng = np.random.default_rng(RANDOM_STATE)
        if len(scoring_datasets) > 8:
            indices = rng.choice(len(scoring_datasets), size=8, replace=False)
            subset = [scoring_datasets[i] for i in sorted(indices)]
        else:
            subset = scoring_datasets

        work_items = [
            (name, X_train, X_val, y_train, y_val, program_path)
            for name, X_train, X_val, y_train, y_val in subset
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(_eval_stage1_worker, item) for item in work_items]
            worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        combined_scores = [score for score, _error, _template in worker_results]
        successful_scores = [score for score in combined_scores if score > 0]
        errors = sorted({error for _score, error, _template in worker_results if error})
        violation_errors = sorted(
            {
                error
                for _score, error, _template in worker_results
                if error and SINGLE_EQUATION_ERROR_PREFIX in error
            }
        )
        if violation_errors:
            return {
                "combined_score": 0.0,
                "n_stage1_datasets": len(subset),
                "n_stage1_successful": len(successful_scores),
                "eval_workers": NUM_WORKERS,
                "error": violation_errors[0],
            }
        if errors:
            return {
                "combined_score": 0.0,
                "n_stage1_datasets": len(subset),
                "n_stage1_successful": len(successful_scores),
                "eval_workers": NUM_WORKERS,
                "error": (
                    "Single-equation violation: stage 1 candidate failed on "
                    f"{len(errors)} dataset evaluations. Example: {errors[0]}"
                ),
            }
        templates = sorted(
            {
                template
                for score, _error, template in worker_results
                if score > 0 and template
            }
        )
        if len(templates) > 1:
            examples = "; ".join(templates[:3])
            return {
                "combined_score": 0.0,
                "n_stage1_datasets": len(subset),
                "n_stage1_successful": len(successful_scores),
                "eval_workers": NUM_WORKERS,
                "error": (
                    "Single-equation violation: stage 1 produced "
                    f"{len(templates)} distinct equation templates across datasets. "
                    "The base equation must be identical for all datasets. "
                    f"Example templates: {examples}"
                ),
            }
        agg_combined = float(np.mean(combined_scores)) if combined_scores else 0.0
        return {
            "combined_score": agg_combined,
            "n_stage1_datasets": len(subset),
            "n_stage1_successful": len(successful_scores),
            "eval_workers": NUM_WORKERS,
            "stage1_full_selected_datasets": 1.0,
        }
    except Exception as exc:
        print(f"Stage 1 evaluation failed: {exc}")
        return {"combined_score": 0.0, "error": str(exc)}


def evaluate_stage2(program_path: str) -> dict:
    """Full evaluation for cascade stage 2."""
    return evaluate(program_path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list-datasets":
        datasets = discover_datasets()
        print(f"Found {len(datasets)} datasets:")
        for name, path in datasets:
            try:
                X, y = _load_fit_table(path)
                print(f"  {name}: {len(y)} samples, {X.shape[1]} features")
            except Exception as exc:
                print(f"  {name}: FAILED - {exc}")
    else:
        default_program = Path(__file__).resolve().parent / "initial_program.py"
        path = sys.argv[1] if len(sys.argv) > 1 else str(default_program)
        print(evaluate(path))
