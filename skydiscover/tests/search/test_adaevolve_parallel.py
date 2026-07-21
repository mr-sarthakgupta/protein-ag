from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from skydiscover.config import Config
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.utils.discovery_utils import SerializableResult


def test_parallel_agents_per_iteration_config_round_trip(tmp_path):
    assert Config().search.parallel_agents_per_iteration == 1

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "search:\n"
        "  type: adaevolve\n"
        "  parallel_agents_per_iteration: 4\n"
    )

    config = Config.from_yaml(config_path)

    assert config.search.parallel_agents_per_iteration == 4
    assert config.to_dict()["search"]["parallel_agents_per_iteration"] == 4


@pytest.mark.asyncio
async def test_adaevolve_runs_and_commits_parallel_agent_batch():
    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.parallel_agents_per_iteration = 4
    controller.database = SimpleNamespace(use_paradigm_breakthrough=False)

    started = 0
    all_started = asyncio.Event()
    processed: list[tuple[int, int]] = []
    checkpoints: list[int] = []
    logged: list[dict] = []

    async def run_normal_step(iteration: int) -> SerializableResult:
        nonlocal started
        agent_index = started
        started += 1
        if started == controller.parallel_agents_per_iteration:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        return SerializableResult(
            child_program_dict={
                "id": f"child-{agent_index}",
                "solution": "pass",
                "language": "python",
                "metrics": {"combined_score": float(agent_index)},
                "iteration_found": iteration,
                "generation": 1,
            },
            parent_id=f"parent-{agent_index}",
            iteration=iteration,
            iteration_time=0.1,
            llm_generation_time=0.08,
            eval_time=0.02,
            sampling_mode="balanced",
            sampling_intensity=0.4,
        )

    controller._run_normal_step = run_normal_step
    controller._process_result = (
        lambda result, iteration, checkpoint_callback, **kwargs: processed.append(
            (iteration, kwargs["agent_index"])
        )
    )
    controller._checkpoint_iteration = (
        lambda iteration, checkpoint_callback: checkpoints.append(iteration)
    )
    controller._log_iteration_stats = lambda **kwargs: logged.append(kwargs)

    await controller._run_iteration(7, checkpoint_callback=object())

    assert started == 4
    assert processed == [(7, 0), (7, 1), (7, 2), (7, 3)]
    assert checkpoints == [7]
    assert len(logged) == 1
    assert len(logged[0]["agent_results"]) == 4
    assert logged[0]["iteration_time"] >= 0.0
