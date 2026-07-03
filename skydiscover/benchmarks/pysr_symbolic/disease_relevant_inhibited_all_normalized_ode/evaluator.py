"""Evaluator for normalized ODE discovery on Abeta42 inhibitor aggregation data.

This benchmark mirrors the non-inhibited normalized ODE task, but loads the
cleaned inhibitor dataset in this folder and exposes inhibitor concentration as
an additional static ODE feature.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

BASE_EVALUATOR_PATH = (
    BENCHMARK_ROOT
    / "disease_relevant_noninhibited_all_normalized_ode"
    / "evaluator.py"
)
_base_spec = importlib.util.spec_from_file_location(
    "_noninhibited_normalized_ode_evaluator",
    str(BASE_EVALUATOR_PATH),
)
_base = importlib.util.module_from_spec(_base_spec)
assert _base_spec.loader is not None
_base_spec.loader.exec_module(_base)

DATA_ROOT = BENCHMARK_ROOT / "disease_relevant_inhibited_clean"
RANDOM_STATE = _base.RANDOM_STATE
TEST_SIZE = _base.TEST_SIZE
EVALUATION_ONLY_DATASETS = frozenset()
MAX_NFEV = int(os.environ.get("SKYDISCOVER_INHIBITED_MAX_NFEV", "300"))

combined_score_from_nmse = _base.combined_score_from_nmse
validate_candidate_source = _base.validate_candidate_source
evaluate_candidate_with_timeout = _base.evaluate_candidate_with_timeout
_metrics_from_result = _base._metrics_from_result
_aggregate_per_dataset_results = _base._aggregate_per_dataset_results
_eval_stage1_worker = _base._eval_stage1_worker
_nmse_with_gaussian_smoothed_target = _base._nmse_with_gaussian_smoothed_target
SINGLE_EQUATION_ERROR_PREFIX = _base.SINGLE_EQUATION_ERROR_PREFIX
NUM_WORKERS = _base.NUM_WORKERS
SHAPE_LOSS_WEIGHT = _base.SHAPE_LOSS_WEIGHT


def _parse_metadata_number(header: str, key: str) -> float | None:
    """Extract a numeric metadata value from headers such as cd:2e-6."""
    match = re.search(
        rf"(?:^|[\s_]){re.escape(key)}:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        header,
    )
    if match is None:
        return None
    return float(match.group(1))


def _is_x_column(header: str) -> bool:
    return bool(re.search(r"\s*X\s*$", header.strip()))


def _is_y_column(header: str) -> bool:
    return bool(re.search(r"\s*Y\s*$", header.strip()))


def evaluate_ode_expression(
    expression,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    constants=None,
    initial_values=None,
    max_nfev: int = MAX_NFEV,
):
    """Score an inhibitor ODE with a bounded constant-fitting budget."""
    return _base.evaluate_ode_expression(
        expression,
        X_train,
        y_train,
        X_val,
        y_val,
        constants=constants,
        initial_values=initial_values,
        max_nfev=max_nfev,
    )


def _load_fit_table(data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load paired inhibitor data.tsv and return X=[time, m0, M0, cd, c], y.

    The first three static features follow the non-inhibited benchmark
    convention. The additional cd feature is the inhibitor concentration parsed
    from each curve header. P0 is preserved in headers but not exposed as an ODE
    feature, matching the previous benchmark's use of M0 as the seed feature.
    """
    with data_file.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = [row for row in reader if row]

    features: list[list[float]] = []
    targets: list[float] = []
    curve_ids: list[int] = []
    curve_id = 0
    for col_idx in range(len(header) - 1):
        x_header = header[col_idx]
        y_header = header[col_idx + 1]
        if not (_is_x_column(x_header) and _is_y_column(y_header)):
            continue

        monomer = _parse_metadata_number(x_header, "m0")
        seed = _parse_metadata_number(x_header, "M0")
        inhibitor = _parse_metadata_number(x_header, "cd")
        if monomer is None or seed is None or inhibitor is None:
            continue

        for row in rows:
            if col_idx + 1 >= len(row):
                continue
            time_value = row[col_idx].strip()
            response_value = row[col_idx + 1].strip()
            if (
                not time_value
                or not response_value
                or time_value.lower() == "nan"
                or response_value.lower() == "nan"
            ):
                continue
            features.append([float(time_value), monomer, seed, inhibitor])
            targets.append(float(response_value))
            curve_ids.append(curve_id)
        curve_id += 1

    if not features:
        raise ValueError(f"No regression samples loaded from {data_file}")

    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    time_norm = _base._minmax_normalize(X[:, 0])
    y_norm = _base._minmax_normalize(y)

    X_ode = np.column_stack([time_norm, X[:, 1], X[:, 2], X[:, 3], y_norm])
    curve_ids_array = np.asarray(curve_ids, dtype=int)
    return _base._remember_curve_ids(X_ode, curve_ids_array), y_norm


def discover_datasets(root: Path = DATA_ROOT) -> list[tuple[str, Path]]:
    """Recursively find cleaned inhibitor data.tsv files."""
    datasets: list[tuple[str, Path]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "data.tsv" in filenames:
            fit_path = Path(dirpath) / "data.tsv"
            rel = fit_path.parent.relative_to(root)
            name = str(rel).replace(os.sep, "/")
            datasets.append((name, fit_path))
    datasets.sort(key=lambda x: x[0])
    return datasets


def _dataset_name_from_path(data_file: Path, root: Path = DATA_ROOT) -> str:
    return str(data_file.parent.relative_to(root)).replace(os.sep, "/")


def load_dataset(
    data_file: Path,
    *,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one inhibitor data.tsv and return a trajectory-preserving split."""
    X, y = _load_fit_table(data_file)
    name = dataset_name or _dataset_name_from_path(data_file)
    if name in EVALUATION_ONLY_DATASETS:
        return _base._evaluation_only_split(X, y)
    return _base._trajectory_preserving_split(X, y)


def load_all_datasets(
    root: Path = DATA_ROOT,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load all inhibitor datasets for aggregate scoring."""
    entries = discover_datasets(root)
    results = []
    for name, path in entries:
        try:
            X_train, X_val, y_train, y_val = load_dataset(path, dataset_name=name)
            results.append((name, X_train, X_val, y_train, y_val))
        except Exception as exc:
            print(f"Warning: skipping dataset {name}: {exc}")
    return results


def _patch_base_evaluator() -> None:
    """Make reused base evaluator functions call this benchmark's data loader."""
    _base.__file__ = __file__
    _base.DATA_ROOT = DATA_ROOT
    _base.EVALUATION_ONLY_DATASETS = EVALUATION_ONLY_DATASETS
    _base._load_fit_table = _load_fit_table
    _base.discover_datasets = discover_datasets
    _base._dataset_name_from_path = _dataset_name_from_path
    _base.load_dataset = load_dataset
    _base.load_all_datasets = load_all_datasets


_patch_base_evaluator()

evaluate = _base.evaluate
evaluate_stage1 = _base.evaluate_stage1
evaluate_stage2 = _base.evaluate_stage2


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
        default_program = BENCHMARK_DIR / "initial_program.py"
        path = sys.argv[1] if len(sys.argv) > 1 else str(default_program)
        print(evaluate(path))
