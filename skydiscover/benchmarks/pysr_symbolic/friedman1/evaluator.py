"""Evaluator for Friedman #1 symbolic regression benchmark."""

from __future__ import annotations

import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

from pysr_harness.metrics import combined_score_from_nmse, nmse

RANDOM_STATE = 42
N_SAMPLES = 400
TEST_SIZE = 0.25


def load_friedman1_data():
    """Deterministic train/validation split for Friedman #1."""
    X, y = make_friedman1(
        n_samples=N_SAMPLES,
        n_features=5,
        noise=0.1,
        random_state=RANDOM_STATE,
    )
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def evaluate_candidate_with_timeout(program_path: str, timeout_seconds: int = 300) -> dict:
    """
    Evaluate a candidate program in a subprocess and return its metrics dict.

    The candidate's `discover()` function is the evolved artifact: it should build
    an equation template and call the harness evaluator. This function only
    isolates that evaluation behind a timeout; it does not run PySR's search loop.
    """
    harness_root = str(BENCHMARK_ROOT)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import importlib.util
import pickle
import sys
import traceback

import numpy as np

sys.path.insert(0, {harness_root!r})

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

program_path = {program_path!r}
results_path = {temp_file.name + ".results"!r}

try:
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    X, y = make_friedman1(
        n_samples={N_SAMPLES},
        n_features=5,
        noise=0.1,
        random_state={RANDOM_STATE},
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size={TEST_SIZE}, random_state={RANDOM_STATE}
    )

    if hasattr(program, "discover"):
        result = program.discover(X_train, y_train, X_val, y_val)
    elif hasattr(program, "run_discovery"):
        result = program.run_discovery(X_train, y_train, X_val, y_val)
    else:
        raise AttributeError("Program must define discover() or run_discovery()")

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
    """Evaluate one evolved symbolic-equation candidate on Friedman #1."""
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

        fn = getattr(program, "discover", None) or getattr(program, "run_discovery", None)
        if fn is None:
            return {"combined_score": 0.0, "error": "missing discover()"}

        X_train, X_val, y_train, y_val = load_friedman1_data()
        # Subset for quick screening
        n = min(120, len(X_train))
        idx = np.arange(n)

        start = time.time()
        result = fn(X_train[idx], y_train[idx], X_val[:80], y_val[:80])
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
