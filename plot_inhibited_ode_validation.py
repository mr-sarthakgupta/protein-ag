#!/usr/bin/env python3
"""Plot best inhibited symbolic ODE predictions against validation curves.

This uses the same benchmark loader/split as
`disease_relevant_inhibited_all_normalized_ode`, then integrates each saved ODE
with the fitted constants stored in `best_inhibited_ode_programs_0703/metadata`.
Plots are saved one validation curve at a time, matching the notebook style of
one condition per figure with model curves overlaid on data points.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


DEFAULT_BENCHMARK_DIR = Path(
    "/home/mrsar/protein-ag/skydiscover/benchmarks/pysr_symbolic/"
    "disease_relevant_inhibited_all_normalized_ode"
)
DEFAULT_RUNS_DIR = Path("/home/mrsar/protein-ag/best_inhibited_ode_programs_0703")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "validation_prediction_plots"


RhsFn = Callable[..., float]


def safe_name(text: str) -> str:
    """Make metadata-rich labels safe for filenames."""
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_")[:180]


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_program_rows(runs_dir: Path, max_programs: int | None) -> list[dict[str, str]]:
    summary_path = runs_dir / "top10_summary.tsv"
    rows: list[dict[str, str]] = []

    if summary_path.exists():
        with summary_path.open(newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    else:
        for metadata_path in sorted((runs_dir / "metadata").glob("*.json")):
            rows.append(
                {
                    "rank": "",
                    "score": "",
                    "program_file": f"programs/{metadata_path.stem}.py",
                }
            )

    if max_programs is not None:
        rows = rows[:max_programs]
    return rows


def metadata_path_for_row(runs_dir: Path, row: dict[str, str]) -> Path:
    program_file = Path(row["program_file"])
    return runs_dir / "metadata" / f"{program_file.stem}.json"


def rhs_from_metadata(metadata: dict[str, Any]) -> tuple[RhsFn, np.ndarray, str]:
    equation_text = metadata.get("equation_template") or metadata.get("equation")
    if not equation_text or "=" not in equation_text:
        raise ValueError("metadata has no equation/equation_template RHS")

    rhs_text = equation_text.split("=", 1)[1].strip()
    constants = metadata.get("constants") or {}

    max_c_index = -1
    for key in constants:
        match = re.fullmatch(r"c(\d+)", key)
        if match:
            max_c_index = max(max_c_index, int(match.group(1)))
    for match in re.finditer(r"\bc(\d+)\b", rhs_text):
        max_c_index = max(max_c_index, int(match.group(1)))

    compiled_rhs = compile(rhs_text, "<ode_rhs>", "eval")
    theta = np.asarray(
        [float(constants[f"c{i}"]) for i in range(max_c_index + 1)],
        dtype=float,
    )

    def rhs_fn(x0: float, x1: float, x2: float, x3: float, x4: float, *params: float) -> float:
        local_values = {"x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "np": np}
        local_values.update({f"c{i}": value for i, value in enumerate(params)})
        value = eval(compiled_rhs, {"__builtins__": {}}, local_values)
        return float(np.asarray(value).reshape(-1)[0])

    return rhs_fn, theta, rhs_text


def curve_indices(base_evaluator: Any, X: np.ndarray) -> list[np.ndarray]:
    return list(base_evaluator._curve_indices(X))


def dense_prediction_for_curve(
    rhs_fn: RhsFn,
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    curve_idx: np.ndarray,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_idx = curve_idx[np.argsort(X[curve_idx, 0])]
    times = X[ordered_idx, 0]
    observed = y[ordered_idx]
    static_features = [float(value) for value in X[ordered_idx[0], 1:-1]]

    if ordered_idx.size == 1 or float(times[-1] - times[0]) <= 0.0:
        return times.copy(), np.full_like(observed, observed[0], dtype=float)

    grid = np.linspace(float(times[0]), float(times[-1]), int(n_grid))

    def dc_dt(c_state: np.ndarray, time_value: float) -> float:
        c_value = float(np.asarray(c_state, dtype=float).reshape(-1)[0])
        value = rhs_fn(float(time_value), *static_features, c_value, *theta)
        if not np.isfinite(value):
            raise FloatingPointError("non-finite ODE derivative")
        return value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        integrated = odeint(dc_dt, float(observed[0]), grid, mxstep=1000)

    prediction = np.asarray(integrated, dtype=float).reshape(-1)
    if prediction.shape != grid.shape or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("non-finite ODE trajectory")
    return grid, prediction


def plot_curve(
    output_path: Path,
    title: str,
    rhs_text: str,
    times: np.ndarray,
    observed: np.ndarray,
    dense_time: np.ndarray,
    dense_pred: np.ndarray,
    point_pred: np.ndarray,
    nmse: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dense_time, dense_pred, linewidth=2.5, color="tab:orange", label="ODE prediction")
    ax.scatter(times, observed, s=18, alpha=0.65, color="tab:blue", label="Validation points")
    ax.plot(times, point_pred, "o", markersize=3.5, alpha=0.65, color="tab:orange", label="Prediction at validation times")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Normalized aggregate mass concentration")
    ax.set_title(f"{title}\ncurve NMSE={nmse:.4g}")
    ax.legend()
    ax.text(
        0.01,
        0.01,
        f"d(c)/dt = {rhs_text}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        wrap=True,
        alpha=0.75,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-programs", type=int, default=None, help="Plot only the first N programs from top10_summary.tsv.")
    parser.add_argument("--n-grid", type=int, default=500, help="Number of dense time points per prediction curve.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_module(args.benchmark_dir / "evaluator.py", "inhibited_ode_plot_evaluator")
    base_evaluator = evaluator._base
    datasets = evaluator.load_all_datasets()
    program_rows = read_program_rows(args.runs_dir, args.max_programs)

    index_rows: list[dict[str, Any]] = []
    for row in program_rows:
        metadata_path = metadata_path_for_row(args.runs_dir, row)
        with metadata_path.open() as f:
            metadata = json.load(f)

        rhs_fn, theta, rhs_text = rhs_from_metadata(metadata)
        program_stem = Path(row["program_file"]).stem
        program_output_dir = args.output_dir / program_stem
        program_output_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, X_train, X_val, y_train, y_val in datasets:
            if X_val.size == 0:
                continue
            reference_variance = float(np.var(np.concatenate([y_train, y_val])))
            if reference_variance <= 0.0:
                reference_variance = 1.0

            try:
                point_predictions = base_evaluator._ode_predictions(rhs_fn, X_val, y_val, theta)
                point_error: str | None = None
            except Exception as exc:
                point_predictions = np.full_like(y_val, np.nan, dtype=float)
                point_error = str(exc)

            for curve_number, curve_idx in enumerate(curve_indices(base_evaluator, X_val)):
                ordered_idx = curve_idx[np.argsort(X_val[curve_idx, 0])]
                times = X_val[ordered_idx, 0]
                observed = y_val[ordered_idx]
                static = X_val[ordered_idx[0], 1:-1]
                point_pred = point_predictions[ordered_idx]
                curve_mse = float(np.mean((point_pred - observed) ** 2)) if np.all(np.isfinite(point_pred)) else float("nan")
                curve_nmse = curve_mse / reference_variance if np.isfinite(curve_mse) else float("nan")
                label = (
                    f"curve={curve_number:03d}_m0={static[0]:.4g}_M0={static[1]:.4g}_"
                    f"cd={static[2]:.4g}"
                )
                output_path = program_output_dir / f"{safe_name(dataset_name)}_{safe_name(label)}.png"

                plot_error = point_error
                try:
                    dense_time, dense_pred = dense_prediction_for_curve(
                        rhs_fn, theta, X_val, y_val, curve_idx, args.n_grid
                    )
                    title = (
                        f"{program_stem}: {dataset_name}\n"
                        f"m0={static[0]:g}, M0={static[1]:g}, cd={static[2]:g}"
                    )
                    plot_curve(
                        output_path,
                        title,
                        rhs_text,
                        times,
                        observed,
                        dense_time,
                        dense_pred,
                        point_pred,
                        curve_nmse,
                        args.dpi,
                    )
                except Exception as exc:
                    plot_error = str(exc)

                index_rows.append(
                    {
                        "program": program_stem,
                        "rank": row.get("rank", ""),
                        "score": row.get("score", metadata.get("score", "")),
                        "dataset": dataset_name,
                        "curve_number": curve_number,
                        "m0": static[0],
                        "M0": static[1],
                        "cd": static[2],
                        "n_validation_points": int(ordered_idx.size),
                        "curve_mse": curve_mse,
                        "curve_nmse": curve_nmse,
                        "plot": str(output_path) if plot_error is None else "",
                        "error": plot_error or "",
                    }
                )

    index_path = args.output_dir / "plot_index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()) if index_rows else ["error"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Saved {sum(1 for row in index_rows if row.get('plot'))} plots to {args.output_dir}")
    print(f"Wrote plot index to {index_path}")


if __name__ == "__main__":
    main()
