"""Build an `AsyncOpenAI` client pointed at a local vLLM server.

vLLM serves an OpenAI-compatible endpoint (chat.completions, tool_calls,
images), so the existing `eval_awareness_experiments` runners can drop a
local model in by swapping the `AsyncOpenAI` instance the runner builds.

Two consumer shapes:
    * `make_async_openai_client(spec)`: returns AsyncOpenAI suitable for
      `tool_calling_runner.ToolCallingRunner`.
    * `make_agentlab_chat_model_args(spec)`: returns a `BaseModelArgs`-
      compatible instance for AgentLab's GenericAgent (used by WASP /
      DoomArena runners). We use AgentLab's `SelfHostedModelArgs` with
      `backend="vllm"` since it already speaks the OpenAI protocol.
"""

from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI

from models.common.registry import LocalModelSpec

logger = logging.getLogger(__name__)


_DUMMY_API_KEY = "EMPTY"  # vLLM's default served auth — accepts any non-empty token


def make_async_openai_client(spec: LocalModelSpec) -> AsyncOpenAI:
    """Build an AsyncOpenAI client pointed at the spec's vLLM server."""
    base_url = spec.resolve_url()
    api_key = os.environ.get("LOCAL_OPENAI_API_KEY", _DUMMY_API_KEY)
    logger.info(
        f"Local OpenAI client for {spec.short_id} -> {base_url} "
        f"(served as {spec.resolve_served_name()!r})"
    )
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=0,
    )


def make_agentlab_chat_model_args(
    spec: LocalModelSpec,
    temperature: float = 0.1,
):
    """Return AgentLab `BaseModelArgs` instance for a local vLLM server.

    AgentLab's `SelfHostedModelArgs(backend="vllm")` (in
    `agentlab/llm/chat_api.py`) is exactly what we want — it builds an
    OpenAI client pointed at the configured URL.
    """
    try:
        from agentlab.llm.chat_api import SelfHostedModelArgs
    except ImportError as e:
        raise RuntimeError(
            "agentlab not installed — install via `uv pip install -e AgentLab/`"
        ) from e

    base_url = spec.resolve_url()
    return SelfHostedModelArgs(
        model_name=spec.resolve_served_name(),
        base_model_name=spec.hf_repo,
        model_url=base_url,
        max_total_tokens=spec.max_total_tokens,
        max_input_tokens=spec.max_input_tokens,
        max_new_tokens=spec.max_new_tokens,
        temperature=temperature,
        backend="vllm",
        vision_support=spec.vision,
    )
