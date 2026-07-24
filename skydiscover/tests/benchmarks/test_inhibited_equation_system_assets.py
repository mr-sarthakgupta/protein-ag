"""Static and lightweight runtime checks for inhibited equation-system assets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "pysr_symbolic" / "disease_relevant_inhibited_all_normalized_ode"
EVALUATOR_PATH = BENCHMARK / "evaluator.py"
CONFIG_PATH = BENCHMARK / "config.yaml"
RUN_SCRIPT = ROOT / "run_disease_relevant_inhibited_normalized_ode_experiment.sh"
MIGRATED_CANDIDATES = {
    "initial_program.py": 5,
    "seed_programs/seed_001_linear_relaxation.py": 1,
    "seed_programs/seed_005_finke_watzky.py": 2,
    "seed_programs/seed_010_competitive_channels.py": 3,
    "seed_programs/seed_016_transient_overshoot.py": 4,
    "seed_programs/seed_032_bell_shaped_state_gate.py": 5,
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("relative_path", "equation_count"), MIGRATED_CANDIDATES.items())
def test_migrated_candidates_are_valid_systems(relative_path, equation_count, monkeypatch):
    evaluator = _load_module(EVALUATOR_PATH, f"asset_evaluator_{equation_count}")
    candidate_path = BENCHMARK / relative_path
    evaluator.validate_candidate_source(str(candidate_path))

    from pysr_harness import equation_session

    monkeypatch.setattr(
        equation_session,
        "_expression_scorer",
        lambda *_args, **_kwargs: {"combined_score": 0.8},
    )
    candidate = _load_module(
        candidate_path,
        "asset_candidate_" + relative_path.replace("/", "_").replace(".", "_"),
    )
    X = np.zeros((3, 5), dtype=float)
    y = np.zeros(3, dtype=float)
    with equation_session.single_equation_evaluation():
        result = candidate.evaluate_symbolic_candidate(X, y, X, y)
        equation_session.validate_single_equation_result(result)

    assert result["equation_count"] == equation_count
    assert len(result["equation_templates"]) == equation_count
    assert result["structural_penalty"] == pytest.approx(0.01 * (equation_count - 1))


def test_prompt_preserves_mantle_and_documents_system_contract():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["llm"]["models"] == [{"name": "openai.gpt-5.6-sol", "weight": 1.0}]
    assert config["llm"]["api_base"] == ("https://bedrock-mantle.us-east-1.api.aws/openai/v1")

    prompt = " ".join(config["prompt"]["system_message"].split())
    for required in (
        "1–5 total equations",
        "exactly one concentration ODE",
        "zero to four algebraic aliases",
        "not an independently integrated state",
        "dead aliases",
        "evaluate_equation_system(...)",
        "structural_penalty = 0.01 * (equation_count - 1)",
        "add a useful alias",
        "remove one",
        "merge",
        "rewrite",
    ):
        assert required in prompt

    database = config["search"]["database"]
    assert database["fitness_key"] == "combined_score"
    assert "structural_penalty" not in database["pareto_objectives"]


def test_default_seed_checkpoint_is_equation_system_versioned():
    script = RUN_SCRIPT.read_text()
    assert "disease_relevant_inhibited_all_normalized_ode_equation_system_v2_" in script
    assert "equation_session._expression_scorer = evaluator.evaluate_ode_expression" in script
