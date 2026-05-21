"""Evaluator for Alpha-synuclein Gaspar 2017 0.3uM seed symbolic regression."""

from __future__ import annotations

import csv
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

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np

from pysr_harness.metrics import combined_score_from_nmse

DATA_FILE = BENCHMARK_DIR / "fit.tsv"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def _concentration_from_column(column_name: str) -> float:
    """Extract concentration in uM from a response column label."""
    match = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)uM\s*$", column_name)
    if match is None:
        raise ValueError(f"Could not parse concentration from column: {column_name}")
    return float(match.group(1))


def _load_fit_table(data_file: Path = DATA_FILE) -> tuple[np.ndarray, np.ndarray]:
    """Load wide fit.tsv and return X=[measurement_x, concentration_uM], y."""
    features: list[list[float]] = []
    targets: list[float] = []

    with data_file.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        concentrations = [_concentration_from_column(name) for name in header[1:]]

        for row in reader:
            if not row:
                continue
            measurement_x = float(row[0])
            for concentration, value in zip(concentrations, row[1:]):
                if value == "":
                    continue
                features.append([measurement_x, concentration])
                targets.append(float(value))

    if not features:
        raise ValueError(f"No regression samples loaded from {data_file}")

    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def load_alphasyn_data():
    """Deterministic train/validation split for the copied Alpha-synuclein TSV."""
    X, y = _load_fit_table()
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(y))
    split = int(round(len(y) * (1.0 - TEST_SIZE)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def evaluate_candidate_with_timeout(program_path: str, timeout_seconds: int = 300) -> dict:
    """
    Evaluate a candidate program in a subprocess and return its metrics dict.

    The candidate's `evaluate_symbolic_candidate()` function is the evolved
    artifact: it should build an equation template and call the harness evaluator.
    This function only isolates evaluation behind a timeout; it does not run
    PySR's search loop.
    """
    harness_root = str(BENCHMARK_ROOT)
    evaluator_path = str(Path(__file__).resolve())
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import importlib.util
import pickle
import sys
import traceback

sys.path.insert(0, {harness_root!r})

program_path = {program_path!r}
evaluator_path = {evaluator_path!r}
results_path = {temp_file.name + ".results"!r}

try:
    evaluator_spec = importlib.util.spec_from_file_location("benchmark_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator)

    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    X_train, X_val, y_train, y_val = evaluator.load_alphasyn_data()

    if hasattr(program, "evaluate_symbolic_candidate"):
        result = program.evaluate_symbolic_candidate(
            X_train, y_train, X_val, y_val
        )
    elif hasattr(program, "run_discovery"):
        result = program.run_discovery(X_train, y_train, X_val, y_val)
    else:
        raise AttributeError("Program must define evaluate_symbolic_candidate() or run_discovery()")

    with open(results_path, "wb") as f:
        pickle.dump({{"result": result}}, f)
except Exception as exc:
    traceback.print_exc()
    with open(results_path, "wb") as f:
        pickle.dump({{"error": str(exc)}}, f)
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
    """Normalize harness output into SkyDiscover evaluator metrics."""
    nmse_val = float(result.get("nmse_val", float("inf")))
    combined = float(result.get("combined_score", combined_score_from_nmse(nmse_val)))
    if not np.isfinite(combined):
        combined = 0.0

    return {
        "equation": str(result.get("equation", "")),
        "loss": float(result.get("loss", float("inf"))),
        "complexity": float(result.get("complexity", 0.0)),
        "nmse_train": float(result.get("nmse_train", float("inf"))),
        "nmse_val": nmse_val,
        "combined_score": combined,
        "eval_time": float(eval_time),
    }


def evaluate(program_path: str) -> dict:
    """Evaluate one evolved symbolic-equation candidate on the Alpha-synuclein TSV."""
    try:
        start = time.time()
        result = evaluate_candidate_with_timeout(program_path, timeout_seconds=600)
        metrics = _metrics_from_result(result, time.time() - start)
        print(
            "Evaluation: "
            f"nmse_val={metrics['nmse_val']:.6f}, "
            f"complexity={metrics['complexity']:.1f}, "
            f"combined_score={metrics['combined_score']:.6f}, "
            f"time={metrics['eval_time']:.2f}s"
        )
        print(f"Best equation: {metrics['equation']}")
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
            "error": str(exc),
        }


def evaluate_stage1(program_path: str) -> dict:
    """Fast cascade stage for one evolved symbolic-equation candidate."""
    try:
        harness_root = str(BENCHMARK_ROOT)
        if harness_root not in sys.path:
            sys.path.insert(0, harness_root)

        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        fn = getattr(program, "evaluate_symbolic_candidate", None) or getattr(
            program, "run_discovery", None
        )
        if fn is None:
            return {"combined_score": 0.0, "error": "missing evaluate_symbolic_candidate()"}

        X_train, X_val, y_train, y_val = load_alphasyn_data()
        n = min(300, len(X_train))
        m = min(120, len(X_val))

        start = time.time()
        result = fn(X_train[:n], y_train[:n], X_val[:m], y_val[:m])
        metrics = _metrics_from_result(result, time.time() - start)
        return metrics
    except Exception as exc:
        print(f"Stage 1 evaluation failed: {exc}")
        return {"combined_score": 0.0, "error": str(exc)}


def evaluate_stage2(program_path: str) -> dict:
    """Full evaluation for cascade stage 2."""
    return evaluate(program_path)


if __name__ == "__main__":
    default_program = Path(__file__).resolve().parent / "initial_program.py"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default_program)
    print(evaluate(path))
