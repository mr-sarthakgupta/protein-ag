from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skydiscover.config import Config, IterationExplanationConfig
from skydiscover.search.adaevolve.explanations import (
    IterationExplanationWriter,
    build_equation_manifest,
    parse_and_validate_explanation,
    select_iteration_programs,
)
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.base_database import Program
from skydiscover.search.utils.discovery_utils import SerializableResult


def _program(program_id="p1", iteration=3, score=0.8):
    return Program(
        id=program_id,
        solution="def evaluate_symbolic_candidate():\n    pass\n",
        iteration_found=iteration,
        metrics={
            "combined_score": score,
            "equation_templates": [
                "free = c0*x1/(x3 + 1)",
                "d(x4)/dt = free + c1*x2 - c2*x4",
            ],
            "resolved_ode_template": "c0*x1/(x3 + 1) + c1*x2 - c2*x4",
            "system_fingerprint": "abc",
            "per_dataset": {
                "dataset-a": {
                    "constants": {"c0": 1.0, "c1": 2.0, "c2": 3.0},
                    "nmse_val": 0.2,
                }
            },
        },
    )


def _complete_response(manifest):
    return json.dumps(
        {
            "summary": "Conservative summary.",
            "limitations": ["Observational fit only."],
            "explanations": [
                {
                    "manifest_id": manifest_id,
                    "mathematical_role": "A mathematical component.",
                    "scientific_interpretation": "No causal claim is established.",
                    "concentration_relationship": "Relationship is explicit in the expression.",
                    "evidence_level": "purely_mathematical",
                }
                for manifest_id in manifest["coverage_ids"]
            ],
        }
    )


class _Pool:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, *_args, **_kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)


def test_selection_detects_new_batch_members_and_deduplicates_by_id():
    winner = _program("winner", score=0.9)
    pareto = _program("pareto", score=0.7)
    old = _program("old", iteration=1, score=0.8)

    selected = select_iteration_programs(
        current_iteration_programs=[pareto, winner],
        previous_best_id=old.id,
        previous_best_score=0.8,
        current_best=winner,
        previous_pareto_ids={"old"},
        current_pareto=[old, winner, pareto],
        score_key=lambda program: program.metrics["combined_score"],
    )

    assert [program.id for program, _ in selected] == ["winner", "pareto"]
    assert selected[0][1] == [
        "iteration_winner",
        "new_global_best",
        "new_pareto_representative",
    ]
    assert selected[1][1] == ["new_pareto_representative"]


def test_manifest_ids_and_complete_preorder_are_deterministic():
    first = build_equation_manifest(_program(), observed_variables={"x3": "inhibitor"})
    second = build_equation_manifest(_program(), observed_variables={"x3": "inhibitor"})

    assert first == second
    assert [equation["id"] for equation in first["ordered_equations"]] == [
        "equation.000",
        "equation.001",
    ]
    assert len(first["coverage_ids"]) == len(set(first["coverage_ids"]))
    node_ids = {node["id"] for node in first["expression_nodes_preorder"]}
    for equation in first["ordered_equations"]:
        assert equation["root_node_id"] in node_ids
    assert first["resolved_ode"]["expression"] == "c0*x1/(x3 + 1) + c1*x2 - c2*x4"
    for node in first["expression_nodes_preorder"]:
        assert set(node["children_ids"]) <= node_ids
    assert first["fitted_constants_by_dataset"]["dataset-a"]["c0"] == 1.0


def test_coverage_validation_reports_missing_ids():
    manifest = build_equation_manifest(_program())
    complete, missing = parse_and_validate_explanation(
        _complete_response(manifest), manifest["coverage_ids"]
    )
    assert complete["summary"]
    assert missing == []

    partial = json.loads(_complete_response(manifest))
    partial["explanations"].pop()
    _, missing = parse_and_validate_explanation(json.dumps(partial), manifest["coverage_ids"])
    assert missing == [manifest["coverage_ids"][-1]]


@pytest.mark.asyncio
async def test_writer_retries_once_with_missing_ids_and_persists_complete(tmp_path):
    program = _program()
    manifest = build_equation_manifest(program)
    partial = json.loads(_complete_response(manifest))
    partial["explanations"] = partial["explanations"][:1]
    pool = _Pool([json.dumps(partial), _complete_response(manifest)])
    writer = IterationExplanationWriter(pool, IterationExplanationConfig(enabled=True), tmp_path)

    record = await writer.explain(program, 3, ["iteration_winner"])
    document = json.loads((tmp_path / "iteration_explanations/iteration_3/p1.json").read_text())

    assert pool.calls == 2
    assert record["status"] == document["status"] == "complete"
    assert document["attempts"] == 2
    assert (tmp_path / "iteration_explanations/iteration_3/p1.md").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "expected_status"),
    [
        (["not-json", "still-not-json"], "error"),
        (
            [
                json.dumps({"summary": "x", "limitations": [], "explanations": []}),
                json.dumps({"summary": "x", "limitations": [], "explanations": []}),
            ],
            "incomplete",
        ),
        ([RuntimeError("model unavailable")], "error"),
    ],
)
async def test_writer_persists_malformed_incomplete_and_model_failures(
    tmp_path, responses, expected_status
):
    writer = IterationExplanationWriter(
        _Pool(responses), IterationExplanationConfig(enabled=True), tmp_path
    )

    record = await writer.explain(_program(), 3, ["iteration_winner"])
    document = json.loads(Path(record["json_path"]).read_text())

    assert record["status"] == expected_status
    assert document["error"]


@pytest.mark.asyncio
async def test_resume_is_idempotent_and_does_not_duplicate_index(tmp_path):
    program = _program()
    manifest = build_equation_manifest(program)
    first_pool = _Pool([_complete_response(manifest)])
    writer = IterationExplanationWriter(
        first_pool, IterationExplanationConfig(enabled=True), tmp_path
    )
    await writer.explain(program, 3, ["iteration_winner"])

    second_pool = _Pool([RuntimeError("must not be called")])
    resumed = IterationExplanationWriter(
        second_pool, IterationExplanationConfig(enabled=True), tmp_path
    )
    record = await resumed.explain(program, 3, ["iteration_winner"])
    index_lines = (tmp_path / "iteration_explanations/index.jsonl").read_text().splitlines()

    assert record["reused"] is True
    assert second_pool.calls == 0
    assert len(index_lines) == 1


def test_explanation_config_defaults_disabled_and_round_trips():
    assert Config().iteration_explanations.enabled is False
    config = Config.from_dict(
        {
            "iteration_explanations": {
                "enabled": True,
                "models": [{"name": "test-model", "max_tokens": 123}],
            }
        }
    )
    assert config.iteration_explanations.models[0].name == "test-model"
    assert config.to_dict()["iteration_explanations"]["enabled"] is True


@pytest.mark.asyncio
async def test_controller_runs_explanations_after_commit_and_checkpoint():
    old = _program("old", iteration=1, score=0.5)
    child = _program("child", iteration=4, score=0.9)
    state = {"best": old}
    events = []

    class Database:
        use_paradigm_breakthrough = False

        def get_best_program(self):
            return state["best"]

        def get_program_proxy_score(self, program):
            return program.metrics["combined_score"]

        def is_multiobjective_enabled(self):
            return False

    class Writer:
        async def explain(self, program, iteration, reasons):
            events.append(("explain", program.id))
            return {"program_id": program.id, "status": "complete"}

    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.parallel_agents_per_iteration = 1
    controller.database = Database()
    controller.iteration_explanation_writer = Writer()

    async def run_step(iteration):
        return SerializableResult(
            child_program_dict=child.to_dict(),
            iteration=iteration,
            iteration_time=0.1,
            llm_generation_time=0.05,
            eval_time=0.05,
        )

    def process(*_args, **_kwargs):
        state["best"] = child
        events.append(("commit", child.id))

    controller._run_normal_step = run_step
    controller._process_result = process
    controller._checkpoint_iteration = lambda *_args: events.append(("checkpoint", child.id))
    logged = []
    controller._log_iteration_stats = lambda **kwargs: logged.append(kwargs)

    await controller._run_iteration(4, checkpoint_callback=object())

    assert events == [
        ("commit", "child"),
        ("checkpoint", "child"),
        ("explain", "child"),
    ]
    assert logged[0]["explanation_results"][0]["status"] == "complete"
