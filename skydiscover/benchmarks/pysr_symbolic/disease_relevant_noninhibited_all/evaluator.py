"""Evaluator for multi-dataset symbolic regression across all disease-relevant non-inhibited proteins.

Discovers all fit.tsv datasets in the source data directory, evaluates a
candidate equation on each independently (fitting a separate set of constants
per dataset), and returns an aggregate score.
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

DATA_ROOT = Path("/home/mrsar/protein-ag/past-published-data/disease-relevant non-inhibited")
RANDOM_STATE = 42
TEST_SIZE = 0.25
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


def validate_candidate_source(program_path: str) -> None:
    """Reject reward-hacking program structure before importing candidate code."""
    source = Path(program_path).read_text()
    tree = ast.parse(source, filename=program_path)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    raise ValueError(
                        f"{SINGLE_EQUATION_ERROR_PREFIX}: import '{alias.name}' is not allowed. "
                        "Candidates may only import sympy/numpy and the equation harness."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in _ALLOWED_FROM_IMPORTS:
                raise ValueError(
                    f"{SINGLE_EQUATION_ERROR_PREFIX}: from-import '{module}' is not allowed."
                )
            if module == "pysr_harness.equation_session":
                imported = {alias.name for alias in node.names}
                extra = imported - _ALLOWED_HARNESS_IMPORTS
                if extra:
                    raise ValueError(
                        f"{SINGLE_EQUATION_ERROR_PREFIX}: unsupported harness imports "
                        f"{sorted(extra)}. Use only feature_symbols, constant_symbols, "
                        "and evaluate_expression."
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
    """Extract a numeric parameter value from a fit.tsv column header.

    Handles diverse header formats found across disease-relevant datasets:
      - "... : 10uM"  or "10 uM"  → 10.0
      - "35uM" (bare, as in Abeta) → 35.0
      - "... : 3.95mM"             → 3950.0  (converted to uM)
      - "... : 0.3 mg/mL 22uM"    → 22.0    (prefer uM when both present)
      - "... : pH 2.7"             → 2.7
      - "... : 5% seeds"           → 5.0
      - "20 ng/ul"                 → 20.0
      - Non-numeric labels (WT, C19S, acid 10min, ...) → None
    """
    text = column_name.strip()

    # Prefer an explicit uM value
    m = re.search(r"(\d+(?:\.\d+)?)\s*uM", text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # mM → convert to uM
    m = re.search(r"(\d+(?:\.\d+)?)\s*mM", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000.0

    # mg/mL (take the number)
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg/mL", text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # ng/ul
    m = re.search(r"(\d+(?:\.\d+)?)\s*ng/ul", text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # pH value
    m = re.search(r"pH\s+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Percentage (e.g. "5% seeds")
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))

    # Last resort: try to find any trailing number after colon/space
    m = re.search(r":\s*(\d+(?:\.\d+)?)\s*$", text)
    if m:
        return float(m.group(1))

    return None


def _load_fit_table(data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a wide fit.tsv and return X=[measurement_x, parameter], y.

    The first column is always the time/measurement coordinate. Remaining
    columns are response curves. The varying parameter (concentration, pH,
    etc.) is extracted from the header; if extraction fails, sequential
    indices 1, 2, 3, ... are used.
    """
    with data_file.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        response_headers = header[1:]
        parsed_params = [_parse_parameter_from_column(h) for h in response_headers]
        if all(p is not None for p in parsed_params):
            parameters = [float(p) for p in parsed_params]
        else:
            parameters = [float(i + 1) for i in range(len(response_headers))]

        features: list[list[float]] = []
        targets: list[float] = []
        for row in reader:
            if not row:
                continue
            measurement_x = float(row[0])
            for param, value in zip(parameters, row[1:]):
                if value == "" or value.strip().lower() == "nan":
                    continue
                features.append([measurement_x, param])
                targets.append(float(value))

    if not features:
        raise ValueError(f"No regression samples loaded from {data_file}")

    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


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


def load_dataset(data_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a single fit.tsv and return deterministic train/val split."""
    X, y = _load_fit_table(data_file)
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(y))
    split = int(round(len(y) * (1.0 - TEST_SIZE)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def load_all_datasets(
    root: Path = DATA_ROOT,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load all datasets and return list of (name, X_train, X_val, y_train, y_val)."""
    entries = discover_datasets(root)
    results = []
    for name, path in entries:
        try:
            X_train, X_val, y_train, y_val = load_dataset(path)
            results.append((name, X_train, X_val, y_train, y_val))
        except Exception as exc:
            print(f"Warning: skipping dataset {name}: {exc}")
    return results


NUM_WORKERS = int(os.environ.get("SKYDISCOVER_EVAL_WORKERS", min(8, os.cpu_count() or 4)))


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
            ds_combined = float(result.get("combined_score", 0.0))
            return ds_name, {{
                "nmse_val": ds_nmse,
                "combined_score": ds_combined,
                "equation": str(result.get("equation", "")),
                "equation_template": str(result.get("equation_template", "")),
                "constants": result.get("constants", {{}}),
            }}
        except Exception as exc:
            return ds_name, {{
                "nmse_val": float("inf"),
                "combined_score": 0.0,
                "error": str(exc),
            }}

    per_dataset_results = {{}}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(_eval_one, dataset) for dataset in datasets]
        for future in as_completed(futures):
            ds_name, ds_result = future.result()
            per_dataset_results[ds_name] = ds_result

    combined_scores = [r["combined_score"] for r in per_dataset_results.values()]
    nmse_vals = [r["nmse_val"] for r in per_dataset_results.values()]
    finite_nmse = [v for v in nmse_vals if np.isfinite(v)]
    agg_nmse = float(np.mean(finite_nmse)) if finite_nmse else float("inf")
    agg_combined = float(1.0 / (1.0 + max(agg_nmse, 0.0))) if np.isfinite(agg_nmse) else 0.0

    violation_errors = sorted(
        {{
            str(r.get("error"))
            for r in per_dataset_results.values()
            if r.get("error") and evaluator.SINGLE_EQUATION_ERROR_PREFIX in str(r.get("error"))
        }}
    )
    templates = sorted(
        {{
            str(r.get("equation_template"))
            for r in per_dataset_results.values()
            if r.get("combined_score", 0.0) > 0 and r.get("equation_template")
        }}
    )

    eq_template = ""
    for r in per_dataset_results.values():
        if r.get("equation"):
            eq_template = r["equation"]
            break

    violation_error = None
    if violation_errors:
        violation_error = violation_errors[0]
    elif len(templates) > 1:
        examples = "; ".join(templates[:3])
        violation_error = (
            "Single-equation violation: evaluate_symbolic_candidate() produced "
            f"{{len(templates)}} distinct equation templates across datasets. "
            "The base equation must be identical for all datasets; only fitted "
            f"constants may vary. Example templates: {{examples}}"
        )

    aggregated = {{
        "equation": eq_template,
        "nmse_val": float("inf") if violation_error else agg_nmse,
        "combined_score": 0.0 if violation_error else agg_combined,
        "n_datasets": len(datasets),
        "n_successful": 0 if violation_error else sum(1 for s in combined_scores if s > 0),
        "per_dataset": per_dataset_results,
    }}
    if violation_error:
        aggregated["error"] = violation_error

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
        "complexity": 0.0,
        "nmse_train": float("inf"),
        "nmse_val": nmse_val,
        "combined_score": combined,
        "eval_time": float(eval_time),
        "n_datasets": result.get("n_datasets", 0),
        "n_successful": result.get("n_successful", 0),
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
            f"combined_score={metrics['combined_score']:.6f}, "
            f"n_datasets={metrics['n_datasets']}, "
            f"n_successful={metrics['n_successful']}, "
            f"time={metrics['eval_time']:.2f}s"
        )
        print(f"Equation template: {metrics['equation']}")
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
            "combined_score": 0.0,
            "eval_time": 0.0,
            "n_datasets": 0,
            "n_successful": 0,
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
        return 0.0, "missing evaluate_symbolic_candidate()"
    n = min(200, len(X_train))
    m = min(80, len(X_val))
    try:
        with single_equation_evaluation():
            result = fn(X_train[:n], y_train[:n], X_val[:m], y_val[:m])
            validate_single_equation_result(result)
        return float(result.get("combined_score", 0.0)), None
    except Exception as exc:
        return 0.0, str(exc)


def evaluate_stage1(program_path: str) -> dict:
    """Fast cascade stage: evaluate on a small subset of datasets with reduced data."""
    try:
        harness_root = str(BENCHMARK_ROOT)
        if harness_root not in sys.path:
            sys.path.insert(0, harness_root)

        all_datasets = load_all_datasets()
        rng = np.random.default_rng(RANDOM_STATE)
        if len(all_datasets) > 8:
            indices = rng.choice(len(all_datasets), size=8, replace=False)
            subset = [all_datasets[i] for i in sorted(indices)]
        else:
            subset = all_datasets

        work_items = [
            (name, X_train, X_val, y_train, y_val, program_path)
            for name, X_train, X_val, y_train, y_val in subset
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(_eval_stage1_worker, item) for item in work_items]
            worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        combined_scores = [score for score, _error in worker_results]
        violation_errors = sorted(
            {
                error
                for _score, error in worker_results
                if error and SINGLE_EQUATION_ERROR_PREFIX in error
            }
        )
        if violation_errors:
            return {"combined_score": 0.0, "error": violation_errors[0]}
        agg_combined = float(np.mean(combined_scores)) if combined_scores else 0.0
        return {"combined_score": agg_combined}
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
