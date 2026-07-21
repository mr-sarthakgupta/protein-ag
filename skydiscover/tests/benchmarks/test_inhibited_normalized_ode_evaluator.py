"""Tests for replicate aggregation in the inhibited normalized ODE benchmark."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


EVALUATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "pysr_symbolic"
    / "disease_relevant_inhibited_all_normalized_ode"
    / "evaluator.py"
)
BASE_EVALUATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "pysr_symbolic"
    / "disease_relevant_noninhibited_all_normalized_ode"
    / "evaluator.py"
)


def _load_evaluator():
    name = "inhibited_normalized_ode_evaluator_test"
    spec = importlib.util.spec_from_file_location(name, EVALUATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_replicate_table(path: Path, n_settings: int = 6) -> None:
    header: list[str] = []
    columns: list[list[float]] = []
    times = [0.0, 1.0, 2.0]
    for setting in range(n_settings):
        for replicate, offset in ((1, 0.0), (2, 0.2)):
            metadata = (
                f"M0:0_P0:0_cd:{setting}e-7_m0:{setting + 1}e-6_rep:{replicate}"
            )
            header.extend([f"{metadata} X", f"{metadata} Y"])
            columns.extend(
                [
                    times,
                    [float(setting), float(setting + 1) + offset, float(setting + 2)],
                ]
            )

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(zip(*columns))


def test_loader_averages_replicates_and_retains_uncertainty(tmp_path):
    evaluator = _load_evaluator()
    data_file = tmp_path / "data.tsv"
    _write_replicate_table(data_file)

    X, y = evaluator._load_fit_table(data_file)
    curve_ids = evaluator._base._ARRAY_CURVE_IDS[id(X)]
    weights = evaluator._base._ARRAY_SAMPLE_WEIGHTS[id(X)]
    sample_std = evaluator._base._ARRAY_SAMPLE_STD[id(X)]

    assert len(np.unique(curve_ids)) == 6
    assert X.shape == (18, 5)
    assert np.allclose(X[curve_ids == 0, 1], 1.0)
    assert np.allclose(X[curve_ids == 5, 3], 0.5)
    assert np.isclose(float(np.mean(weights)), 1.0)
    assert np.any(sample_std > 0.0)
    assert np.all((0.0 <= y) & (y <= 1.0))


def test_split_keeps_averaged_settings_disjoint(tmp_path):
    evaluator = _load_evaluator()
    data_file = tmp_path / "data.tsv"
    _write_replicate_table(data_file)

    X_train, X_val, _y_train, _y_val = evaluator.load_dataset(
        data_file,
        dataset_name="synthetic/replicates",
    )
    train_settings = {tuple(row[1:4]) for row in X_train}
    val_settings = {tuple(row[1:4]) for row in X_val}

    assert train_settings
    assert val_settings
    assert train_settings.isdisjoint(val_settings)
    assert id(X_train) in evaluator._base._ARRAY_SAMPLE_WEIGHTS
    assert id(X_val) in evaluator._base._ARRAY_SAMPLE_WEIGHTS


def test_shape_timing_levels_are_relative_to_curve_range():
    evaluator = _load_evaluator()
    curve = np.concatenate(
        [
            np.full(10, 0.3),
            np.linspace(0.3, 0.9, 80),
            np.full(10, 0.9),
        ]
    )

    levels = evaluator._base._relative_response_levels(curve)

    assert np.allclose(levels, (0.36, 0.45, 0.6, 0.75, 0.84))
    assert all(level > 0.3 for level in levels)


def test_shape_weight_accepts_values_above_old_cap(monkeypatch):
    monkeypatch.setenv("SKYDISCOVER_SHAPE_LOSS_WEIGHT", "0.75")
    name = "noninhibited_normalized_ode_shape_weight_test"
    spec = importlib.util.spec_from_file_location(name, BASE_EVALUATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.SHAPE_LOSS_WEIGHT == 0.75
    assert module.SHAPE_FIT_WEIGHT == 0.75


def test_shape_fit_residuals_are_per_curve_and_fixed_length():
    evaluator = _load_evaluator()._base
    time = np.linspace(0.0, 1.0, 8)
    X = np.vstack(
        [
            np.column_stack(
                [time, np.ones(8), np.zeros(8), np.ones(8), np.zeros(8)]
            ),
            np.column_stack(
                [time, np.ones(8), np.ones(8), np.ones(8), np.zeros(8)]
            ),
        ]
    )
    evaluator._remember_curve_ids(X, np.repeat([11, 29], 8))
    y_true = np.concatenate([time, time**2])
    y_pred = np.concatenate([time**2, np.sqrt(time)])

    perfect = evaluator._curve_shape_fit_residuals(
        X,
        y_true,
        y_true,
        point_count=y_true.size,
        shape_weight=0.5,
    )
    mismatched = evaluator._curve_shape_fit_residuals(
        X,
        y_true,
        y_pred,
        point_count=y_true.size,
        shape_weight=0.5,
    )

    assert perfect.shape == mismatched.shape == (12,)
    assert np.allclose(perfect, 0.0)
    assert np.any(mismatched > 0.0)


def test_soft_l1_is_applied_only_through_point_residual_transform():
    evaluator = _load_evaluator()._base
    raw = np.asarray([-100.0, -2.0, -1e-12, 0.0, 1e-12, 2.0, 100.0])

    transformed = evaluator._soft_l1_point_residuals(raw)
    expected_soft_l1_cost = (
        2.0 * raw**2 / (np.sqrt(1.0 + raw**2) + 1.0)
    )

    assert np.all(np.isfinite(transformed))
    assert np.allclose(transformed**2, expected_soft_l1_cost)
    assert np.array_equal(np.signbit(transformed), np.signbit(raw))
