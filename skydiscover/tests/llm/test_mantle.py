"""Focused tests for Bedrock Mantle's OpenAI-compatible Responses API."""

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from skydiscover.config import BEDROCK_MANTLE_API_BASE, AgenticConfig, LLMConfig, LLMModelConfig
from skydiscover.llm.agentic_generator import AgenticGenerator
from skydiscover.llm.cost_tracker import CostTracker
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.llm.openai import OpenAILLM, is_openai_reasoning_model

MODEL = "openai.gpt-5.6-sol"


def _text_response(text="ok"):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=None,
    )


def _model_config(**overrides):
    values = {
        "name": MODEL,
        "api_base": BEDROCK_MANTLE_API_BASE,
        "api_key": "ABSK-test-token",
        "temperature": 0.7,
        "max_tokens": 32,
        "timeout": 10,
        "retries": 0,
        "retry_delay": 0,
        "reasoning_effort": "low",
    }
    values.update(overrides)
    return LLMModelConfig(**values)


def test_mantle_model_resolves_provider_and_bearer_token(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSK-env-token")

    config = LLMConfig(models=[LLMModelConfig(name=MODEL)])

    assert config.models[0].name == MODEL
    assert config.models[0].api_base == BEDROCK_MANTLE_API_BASE
    assert config.models[0].api_key == "ABSK-env-token"


def test_mantle_bootstraps_absk_session_token_without_logging_it(monkeypatch, tmp_path, caplog):
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[default]\n"
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n"
        "aws_secret_access_key = example\n"
        "aws_session_token = ABSK-super-secret\n"
    )
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    caplog.set_level(logging.INFO)

    config = LLMConfig(models=[LLMModelConfig(name=MODEL)])

    assert config.models[0].api_key == "ABSK-super-secret"
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "ABSK-super-secret"
    assert "ABSK-super-secret" not in caplog.text


def test_llm_pool_routes_mantle_to_openai_backend():
    with patch("skydiscover.llm.llm_pool.OpenAILLM") as openai_cls:
        pool = LLMPool([_model_config()])

    openai_cls.assert_called_once()
    assert pool.models == [openai_cls.return_value]


def test_mantle_gpt_is_a_reasoning_model():
    assert is_openai_reasoning_model(MODEL, BEDROCK_MANTLE_API_BASE)


def test_mantle_gpt_uses_published_bedrock_pricing():
    tracker = CostTracker()

    cost = tracker.record_usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=100_000,
        cache_write_tokens=100_000,
        model=MODEL,
    )

    # One million total input tokens includes the two cached 100k portions.
    assert cost == pytest.approx(0.8 * 5.50 + 33.00 + 0.1 * 0.55 + 0.1 * 6.88)


@pytest.mark.asyncio
async def test_openai_llm_uses_responses_api_for_mantle():
    client = MagicMock()
    client.responses.create.return_value = _text_response("tiny")
    with patch("skydiscover.llm.openai.openai.OpenAI", return_value=client):
        llm = OpenAILLM(_model_config())

    result = await llm.generate("system", [{"role": "user", "content": "hello"}])

    assert result.text == "tiny"
    client.chat.completions.create.assert_not_called()
    params = client.responses.create.call_args.kwargs
    assert params["model"] == MODEL
    assert params["instructions"] == "system"
    assert params["max_output_tokens"] == 32
    assert params["reasoning"] == {"effort": "low"}
    assert params["input"][0]["content"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_agentic_mantle_responses_api_extracts_tool_call():
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="read_file",
                arguments='{"path":"candidate.py"}',
            )
        ],
        usage=None,
    )
    with patch("skydiscover.llm.openai.openai.OpenAI", return_value=client):
        model = OpenAILLM(_model_config())

    pool = SimpleNamespace(
        models=[model],
        weights=[1.0],
        random_state=SimpleNamespace(choices=lambda *args, **kwargs: [0]),
    )
    generator = AgenticGenerator(pool, AgenticConfig())

    result = await generator._call_llm(
        "system",
        [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "old_call",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"pattern":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old_call", "content": "found"},
        ],
    )

    assert result["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path":"candidate.py"}',
    }
    params = client.responses.create.call_args.kwargs
    assert [item["type"] for item in params["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert params["reasoning"] == {"effort": "low"}
