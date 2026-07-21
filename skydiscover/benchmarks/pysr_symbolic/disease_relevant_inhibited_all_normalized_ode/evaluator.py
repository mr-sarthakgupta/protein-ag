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
INCLUDE_UNINHIBITED = os.environ.get(
    "SKYDISCOVER_INCLUDE_UNINHIBITED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
STATIC_CONCENTRATION_SCALE = float(
    os.environ.get("SKYDISCOVER_INHIBITED_CONCENTRATION_SCALE", "1e-6")
)
if STATIC_CONCENTRATION_SCALE <= 0.0:
    raise ValueError("SKYDISCOVER_INHIBITED_CONCENTRATION_SCALE must be positive")

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
    """Load replicate-averaged inhibitor trajectories.

    Replicates are grouped by the full experimental setting
    ``(m0, M0, P0, cd)`` and averaged in raw response space before global
    normalization. Static concentration features are expressed in units of
    ``STATIC_CONCENTRATION_SCALE`` (1 µM by default). Pointwise replicate SEM
    is retained as bounded inverse-uncertainty weights for fitting and NMSE.
    """
    with data_file.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = [row for row in reader if row]

    grouped_curves: dict[
        tuple[float, float, float | None, float],
        list[tuple[np.ndarray, np.ndarray]],
    ] = {}
    for col_idx in range(len(header) - 1):
        x_header = header[col_idx]
        y_header = header[col_idx + 1]
        if not (_is_x_column(x_header) and _is_y_column(y_header)):
            continue

        monomer = _parse_metadata_number(x_header, "m0")
        seed = _parse_metadata_number(x_header, "M0")
        seed_number = _parse_metadata_number(x_header, "P0")
        inhibitor = _parse_metadata_number(x_header, "cd")
        if monomer is None or seed is None or inhibitor is None:
            continue
        if not INCLUDE_UNINHIBITED and inhibitor <= 0.0:
            continue

        times: list[float] = []
        responses: list[float] = []
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
            times.append(float(time_value))
            responses.append(float(response_value))

        if not times:
            continue
        order = np.argsort(np.asarray(times, dtype=float))
        curve = (
            np.asarray(times, dtype=float)[order],
            np.asarray(responses, dtype=float)[order],
        )
        setting = (monomer, seed, seed_number, inhibitor)
        grouped_curves.setdefault(setting, []).append(curve)

    features: list[list[float]] = []
    targets: list[float] = []
    target_sem: list[float] = []
    target_std: list[float] = []
    curve_ids: list[int] = []
    for curve_id, (setting, replicates) in enumerate(grouped_curves.items()):
        monomer, seed, _seed_number, inhibitor = setting
        reference_time = replicates[0][0]
        aligned = []
        for times, response in replicates:
            if np.array_equal(times, reference_time):
                aligned.append(response)
            else:
                aligned.append(np.interp(reference_time, times, response))
        replicate_matrix = np.vstack(aligned)
        mean_response = np.mean(replicate_matrix, axis=0)
        if len(replicates) > 1:
            std_response = np.std(replicate_matrix, axis=0, ddof=1)
            sem_response = std_response / np.sqrt(float(len(replicates)))
        else:
            std_response = np.zeros_like(mean_response)
            sem_response = np.zeros_like(mean_response)

        for time_value, response_value, sem_value, std_value in zip(
            reference_time,
            mean_response,
            sem_response,
            std_response,
        ):
            features.append(
                [
                    float(time_value),
                    monomer / STATIC_CONCENTRATION_SCALE,
                    seed / STATIC_CONCENTRATION_SCALE,
                    inhibitor / STATIC_CONCENTRATION_SCALE,
                ]
            )
            targets.append(float(response_value))
            target_sem.append(float(sem_value))
            target_std.append(float(std_value))
            curve_ids.append(curve_id)

    if not features:
        raise ValueError(f"No regression samples loaded from {data_file}")

    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    time_norm = _base._minmax_normalize(X[:, 0])
    y_norm = _base._minmax_normalize(y)
    response_scale = float(np.ptp(y))
    std_norm = np.asarray(target_std, dtype=float) / response_scale if response_scale > 0 else np.zeros_like(y)
    sem_norm = np.asarray(target_sem, dtype=float) / response_scale if response_scale > 0 else np.zeros_like(y)
    positive_sem = sem_norm[np.isfinite(sem_norm) & (sem_norm > 0.0)]
    sem_floor = float(np.median(positive_sem)) if positive_sem.size else 1.0
    uncertainty_weights = 1.0 / (1.0 + (sem_norm / sem_floor) ** 2)
    uncertainty_weights = np.clip(
        uncertainty_weights / float(np.mean(uncertainty_weights)),
        0.25,
        4.0,
    )
    uncertainty_weights /= float(np.mean(uncertainty_weights))

    X_ode = np.column_stack([time_norm, X[:, 1], X[:, 2], X[:, 3], y_norm])
    curve_ids_array = np.asarray(curve_ids, dtype=int)
    _base._remember_curve_ids(X_ode, curve_ids_array)
    _base._remember_sample_uncertainty(
        X_ode,
        uncertainty_weights,
        std_norm,
    )
    return X_ode, y_norm


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


def _split_on_curve_ids(
    X: np.ndarray,
    y: np.ndarray,
    val_curve_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split complete trajectories while preserving evaluator metadata."""
    curves = _base._curve_indices(X)
    fit_indices: list[int] = []
    val_indices: list[int] = []
    for curve_id, curve_idx in enumerate(curves):
        target = val_indices if curve_id in val_curve_ids else fit_indices
        target.extend(curve_idx.tolist())

    fit_idx = np.sort(np.asarray(fit_indices, dtype=int))
    val_idx = np.sort(np.asarray(val_indices, dtype=int))
    curve_ids = _base._ARRAY_CURVE_IDS.get(
        id(X), np.zeros(X.shape[0], dtype=int)
    )
    sample_weights = _base._sample_weights(X)
    sample_std = _base._ARRAY_SAMPLE_STD.get(id(X))
    X_fit = X[fit_idx]
    X_val = X[val_idx]
    _base._remember_curve_ids(X_fit, curve_ids[fit_idx])
    _base._remember_curve_ids(X_val, curve_ids[val_idx])
    _base._remember_sample_uncertainty(
        X_fit,
        sample_weights[fit_idx],
        sample_std[fit_idx] if sample_std is not None else None,
    )
    _base._remember_sample_uncertainty(
        X_val,
        sample_weights[val_idx],
        sample_std[val_idx] if sample_std is not None else None,
    )
    return X_fit, X_val, y[fit_idx], y[val_idx]


def _balanced_trajectory_preserving_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a balanced cross-section of this benchmark's factorial design.

    The expected design has three seed levels and, within each seed level,
    five positive inhibitor doses, optionally plus five uninhibited monomer
    levels. Validation represents every available dose (and monomer level)
    once and distributes curves across the three seed levels.
    """
    curves = _base._curve_indices(X)
    if len(curves) < 6:
        return _base._trajectory_preserving_split(X, y)

    curve_settings = {
        curve_id: tuple(float(value) for value in X[curve_idx[0], 1:4])
        for curve_id, curve_idx in enumerate(curves)
    }
    seed_levels = sorted(
        {setting[1] for setting in curve_settings.values()}, reverse=True
    )
    if len(seed_levels) != 3:
        return _base._trajectory_preserving_split(X, y)

    inhibited_ranks = ((1, 4), (0, 3), (2,))
    uninhibited_ranks = ((1, 4), (3,), (0, 2))
    val_curve_ids: set[int] = set()
    has_uninhibited = any(
        setting[2] == 0.0 for setting in curve_settings.values()
    )

    for seed_idx, seed_level in enumerate(seed_levels):
        inhibited = sorted(
            (
                (setting[2], curve_id)
                for curve_id, setting in curve_settings.items()
                if setting[1] == seed_level and setting[2] > 0.0
            )
        )
        uninhibited = sorted(
            (
                (setting[0], curve_id)
                for curve_id, setting in curve_settings.items()
                if setting[1] == seed_level and setting[2] == 0.0
            )
        )
        if len(inhibited) != 5:
            return _base._trajectory_preserving_split(X, y)
        if has_uninhibited and len(uninhibited) != 5:
            return _base._trajectory_preserving_split(X, y)

        val_curve_ids.update(
            inhibited[rank][1] for rank in inhibited_ranks[seed_idx]
        )
        if has_uninhibited:
            val_curve_ids.update(
                uninhibited[rank][1] for rank in uninhibited_ranks[seed_idx]
            )

    return _split_on_curve_ids(X, y, val_curve_ids)


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
    return _balanced_trajectory_preserving_split(X, y)


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
