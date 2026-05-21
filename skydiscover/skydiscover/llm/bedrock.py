"""AWS Bedrock LLM backend using the native Converse API."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse

logger = logging.getLogger("skydiscover.llm")


def _content_to_text(content: Any) -> str:
    """Convert OpenAI-style message content into plain text for Bedrock."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("text") is not None:
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def _bedrock_role(role: str) -> str:
    """Map common chat roles to Bedrock Converse roles."""
    return "assistant" if role == "assistant" else "user"


def _region_from_api_base(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return None
    if api_base.startswith("bedrock:"):
        return api_base.split(":", 1)[1] or None
    return None


class BedrockLLM(LLMInterface):
    """LLM backend using AWS Bedrock Runtime's Converse API."""

    def __init__(self, model_cfg: Optional[LLMModelConfig] = None):
        if model_cfg is None:
            raise ValueError("BedrockLLM requires an LLMModelConfig")

        self.model = model_cfg.name
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base

        if not self.model:
            raise ValueError("Bedrock model config requires a model name")

        self.region = (
            _region_from_api_base(self.api_base)
            or os.environ.get("BEDROCK_AWS_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "AWS Bedrock support requires boto3. Install with "
                "`pip install boto3` or `uv sync --extra bedrock`."
            ) from exc

        session_kwargs: dict[str, str] = {}
        profile = os.environ.get("AWS_PROFILE")
        if profile:
            session_kwargs["profile_name"] = profile
        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", region_name=self.region)

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        model_key = ("bedrock", self.region, self.model)
        if model_key not in logger._initialized_models:
            logger.info(f"AWS Bedrock LLM: {self.model} region={self.region}")
            logger._initialized_models.add(model_key)

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        if kwargs.get("image_output"):
            raise NotImplementedError("BedrockLLM does not support image_output")
        text = await self._generate_text(system_message, messages, **kwargs)
        return LLMResponse(text=text)

    async def _generate_text(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> str:
        params = self._build_converse_params(system_message, messages, **kwargs)
        retries, retry_delay, timeout = self._resolve_retry_options(**kwargs)

        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(self._call_api(params), timeout=timeout)
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout attempt {attempt + 1}/{retries + 1}, retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as exc:
                if attempt < retries:
                    logger.warning(
                        f"Error attempt {attempt + 1}/{retries + 1}: {exc}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    def _build_converse_params(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        bedrock_messages: list[dict[str, Any]] = []
        for message in messages:
            text = _content_to_text(message.get("content"))
            if not text:
                continue
            bedrock_messages.append(
                {
                    "role": _bedrock_role(str(message.get("role", "user"))),
                    "content": [{"text": text}],
                }
            )

        if not bedrock_messages:
            bedrock_messages.append({"role": "user", "content": [{"text": ""}]})

        inference_config: dict[str, Any] = {}
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            inference_config["maxTokens"] = int(max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            inference_config["temperature"] = float(temperature)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            inference_config["topP"] = float(top_p)

        params: Dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
        }
        if system_message:
            params["system"] = [{"text": system_message}]
        if inference_config:
            params["inferenceConfig"] = inference_config
        return params

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.client.converse(**params))
        return self._extract_text(response)

    def _extract_text(self, response: Dict[str, Any]) -> str:
        content = response.get("output", {}).get("message", {}).get("content", [])
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        return "\n".join(parts)

    def _resolve_retry_options(self, **kwargs) -> Tuple[int, int, int]:
        retries = kwargs.get("retries", self.retries)
        if retries is None:
            retries = 0
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        if retry_delay is None:
            retry_delay = 2
        timeout = kwargs.get("timeout", self.timeout)
        if timeout is None:
            timeout = 300
        return retries, retry_delay, timeout
