"""Focused tests for the observed-variable DAE candidate contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = ROOT / "benchmarks" / "pysr_symbolic"
EVALUATOR_PATH = HARNESS_ROOT / "disease_relevant_noninhibited_all_normalized_ode" / "evaluator.py"
sys.path.insert(0, str(HARNESS_ROOT))

from pysr_harness import equation_session as session


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((3, 5), dtype=float)
    y = np.zeros(3, dtype=float)
    return X, y


def _load_evaluator():
    name = "equation_system_static_guard_test"
    spec = importlib.util.spec_from_file_location(name, EVALUATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_one_equation_scalar_wrapper_is_compatible(monkeypatch):
    X, y = _data()
    x = session.feature_symbols(5)
    calls = []

    def scorer(expression, *_args, **_kwargs):
        calls.append(expression)
        return {"combined_score": 0.8, "nmse_val": 0.25}

    monkeypatch.setattr(session, "_expression_scorer", scorer)
    with session.single_equation_evaluation():
        result = session.evaluate_expression(x[4] + x[3], X, y, X, y)
        session.validate_single_equation_result(result)

    assert calls == [x[3] + x[4]]
    assert result["equation_count"] == 1
    assert result["structural_penalty"] == 0.0
    assert result["combined_score"] == result["raw_combined_score"] == 0.8


def test_five_equations_resolve_forward_references_and_apply_penalty(monkeypatch):
    X, y = _data()
    x = session.feature_symbols(5)
    c = session.constant_symbols(1)
    a, b, d, e = sp.symbols("a b d e")
    equations = [
        session.algebraic_equation(a, b + x[1]),
        session.ode_equation(e + a),
        session.algebraic_equation(b, d * c[0]),
        session.algebraic_equation(d, x[2] + x[3]),
        session.algebraic_equation(e, x[4]),
    ]
    calls = []

    def scorer(expression, *_args, **_kwargs):
        calls.append(expression)
        return {"combined_score": 0.5, "nmse_val": 1.0}

    monkeypatch.setattr(session, "_expression_scorer", scorer)
    result = session.evaluate_equation_system(
        equations,
        X,
        y,
        X,
        y,
        constants=c,
        extra_equation_penalty=0.1,
    )

    assert len(calls) == 1
    assert sp.simplify(calls[0] - (x[4] + x[1] + c[0] * (x[2] + x[3]))) == 0
    assert result["equation_count"] == 5
    assert result["equation_templates"][0].startswith("a =")
    assert result["equation_templates"][1].startswith("d(x4)/dt =")
    assert result["structural_penalty"] == pytest.approx(0.4)
    assert result["raw_combined_score"] == 0.5
    assert result["combined_score"] == pytest.approx(0.3)
    assert result["total_tree_complexity"] > 0


def test_ordered_system_fingerprint_is_deterministic_and_order_sensitive():
    x = session.feature_symbols(5)
    a, b = sp.symbols("a b")
    first = [
        session.algebraic_equation(a, x[1]),
        session.algebraic_equation(b, x[2]),
        session.ode_equation(a + b + x[4]),
    ]
    reordered = [first[1], first[0], first[2]]

    resolved_1 = session.resolve_equation_system(first)
    resolved_2 = session.resolve_equation_system(first)
    resolved_reordered = session.resolve_equation_system(reordered)

    assert resolved_1.system_fingerprint == resolved_2.system_fingerprint
    assert resolved_1.system_fingerprint != resolved_reordered.system_fingerprint
    assert resolved_1.resolved_ode == resolved_reordered.resolved_ode


@pytest.mark.parametrize(
    ("equations", "message"),
    [
        ([], "1–5 equations"),
        ([session.ode_equation(0)] * 6, "1–5 equations"),
        (
            [session.ode_equation(0), session.ode_equation(1)],
            "exactly one ODE",
        ),
        ([session.algebraic_equation("a", 1)], "exactly one ODE"),
        (
            [
                session.algebraic_equation("a", 1),
                session.algebraic_equation("a", 2),
                session.ode_equation(sp.Symbol("a")),
            ],
            "duplicate algebraic target",
        ),
        (
            [session.ode_equation(sp.Symbol("latent") + 1)],
            "undefined/free/latent",
        ),
        (
            [
                session.algebraic_equation("a", sp.Symbol("b")),
                session.algebraic_equation("b", sp.Symbol("a")),
                session.ode_equation(sp.Symbol("a")),
            ],
            "cycle",
        ),
        (
            [
                session.algebraic_equation("used", 1),
                session.algebraic_equation("dead", 2),
                session.ode_equation(sp.Symbol("used")),
            ],
            "do not contribute",
        ),
    ],
)
def test_rejects_invalid_systems(equations, message):
    with pytest.raises(session.SingleEquationViolation, match=message):
        session.resolve_equation_system(equations)


def test_static_guard_allows_system_and_rejects_two_score_attempts(tmp_path):
    evaluator = _load_evaluator()
    valid = tmp_path / "valid.py"
    valid.write_text(dedent("""
            from __future__ import annotations
            import sympy as sp
            from pysr_harness.equation_session import (
                algebraic_equation,
                constant_symbols,
                evaluate_equation_system,
                feature_symbols,
                ode_equation,
            )

            def evaluate_symbolic_candidate(X_train, y_train, X_val, y_val):
                x = feature_symbols(X_train.shape[1])
                c = constant_symbols(1)
                rate_symbol = sp.Symbol("rate")
                rate = algebraic_equation(rate_symbol, c[0] * x[1])
                ode = ode_equation(rate_symbol + x[4])
                equations = [rate, ode]
                return evaluate_equation_system(
                    equations, X_train, y_train, X_val, y_val, constants=c
                )
            """).lstrip())
    evaluator.validate_candidate_source(str(valid))

    abusive = tmp_path / "abusive.py"
    abusive.write_text(
        valid.read_text().replace(
            "return evaluate_equation_system(",
            "discarded = evaluate_equation_system(\n"
            "        equations, X_train, y_train, X_val, y_val, constants=c\n"
            "    )\n"
            "    return evaluate_equation_system(",
        )
    )
    with pytest.raises(ValueError, match="one scorer exactly once"):
        evaluator.validate_candidate_source(str(abusive))


def test_aggregate_ranking_applies_structural_penalty_to_raw_fit():
    evaluator = _load_evaluator()
    result = evaluator._aggregate_per_dataset_results(
        {
            "dataset": {
                "nmse_train": 1.0,
                "nmse_val": 1.0,
                "scored_loss_val": 1.0,
                "shape_loss_val": 0.0,
                "n_val_points": 3,
                "combined_score": 0.4,
                "equation": "d(x4)/dt = x1",
                "equation_template": "d(c)/dt = x1",
                "equation_count": 3,
                "equation_templates": ["a = x1", "b = a", "d(x4)/dt = b"],
                "resolved_ode_template": "x1",
                "system_fingerprint": "fixed",
                "total_tree_complexity": 5.0,
                "structural_penalty": 0.2,
            }
        }
    )

    assert result["fit_score"] == pytest.approx(0.5)
    assert result["raw_combined_score"] == pytest.approx(0.5)
    assert result["combined_score"] == pytest.approx(0.4)
    assert result["nmse_val"] == 1.0
    assert result["structural_penalty"] == 0.2
