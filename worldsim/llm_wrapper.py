"""LLM wrapper for the native OpenAI Responses API (Arm A of the A/B study).

Arm A uses ``client.responses.parse`` with ``reasoning={"effort": "none"}``
and ``text_format=<PydanticModel>`` for strict structured output.  The
reasoning tokens are separated by the API into their own output item and
never appear in the text content, so no ``<think>`` stripping is needed.

Arm B (OpenRouter pinned to OpenAI) is handled entirely in ``agent_config.py``
via ``ChatOpenRouter`` with the appropriate ``extra_body`` overrides.

Both arms use the same ``AgentOutput`` Pydantic schema and strict mode.
They differ only in the API surface:
  - Arm A: native Responses API (``/v1/responses``)
  - Arm B: OpenRouter chat completions with ``reasoning.exclude=true``
    and ``provider.only=["openai"]``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Models that use the Responses API reasoning path.
_OPENAI_REASONING_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "o4-mini",
        "o3",
        "o3-mini",
        "o1",
    }
)


def is_reasoning_model(model: str) -> bool:
    """Return True if *model* should use the Responses API reasoning path."""
    lower = model.lower()
    return any(name in lower for name in _OPENAI_REASONING_MODELS)


class ChatOpenAIResponses:
    """Wrapper around ``ChatOpenAI`` that routes structured-output calls through
    the native OpenAI Responses API instead of Chat Completions.

    For structured output (``output_format`` is not None), the wrapper calls
    ``client.responses.parse(text_format=output_format, reasoning={"effort": "none"})``
    so that:
    - Reasoning tokens are separated into their own output item (never in content).
    - No ``reasoning.summary`` is requested (avoids billing for summary tokens).
    - No ``reasoning.encrypted_content`` is included.
    - The text content arrives as clean JSON parseable by ``output_format``.

    For free-form (non-structured) calls, the call falls through to the parent
    ``ChatOpenAI.ainvoke`` unchanged.
    """

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        """Build and return a patched ``ChatOpenAI`` instance."""
        from browser_use.llm.openai.chat import ChatOpenAI

        instance = ChatOpenAI(**kwargs)
        original_ainvoke = instance.ainvoke

        async def ainvoke(messages: Any, output_format: Any = None, **kw: Any) -> Any:
            # Free-form path: fall through to the parent implementation.
            if output_format is None:
                return await original_ainvoke(messages, output_format=None, **kw)

            # Structured-output path: use the Responses API.
            from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
            from browser_use.llm.openai.serializer import OpenAIMessageSerializer
            from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
            from openai import APIConnectionError, APIStatusError, RateLimitError

            # Translate Chat Completions message format to Responses API input.
            # ``OpenAIMessageSerializer.serialize_messages`` returns a list of
            # ``ChatCompletionMessageParam`` dicts (each has ``role`` and
            # ``content``). The Responses API accepts the same dicts directly as
            # ``EasyInputMessageParam`` items because the shape is compatible:
            # both carry ``role`` and ``content`` at the top level.
            openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

            try:
                response = await instance.get_client().responses.parse(
                    model=instance.model,
                    input=openai_messages,  # type: ignore[arg-type]
                    text_format=output_format,
                    # Separate reasoning into its own output item; do NOT
                    # request summary and do NOT include encrypted_content.
                    reasoning={"effort": "none"},
                    # Store=False keeps responses out of the dashboard and
                    # avoids unintended retention of task content.
                    store=False,
                )
            except RateLimitError as e:
                raise ModelRateLimitError(message=e.message, model=instance.name) from e
            except APIConnectionError as e:
                raise ModelProviderError(message=str(e), model=instance.name) from e
            except APIStatusError as e:
                raise ModelProviderError(
                    message=e.message, status_code=e.status_code, model=instance.name
                ) from e
            except Exception as e:
                raise ModelProviderError(message=str(e), model=instance.name) from e

            parsed = response.output_parsed
            if parsed is None:
                raise ModelProviderError(
                    message="Responses API returned no parsed structured output",
                    status_code=500,
                    model=instance.name,
                )

            # Translate Responses API usage to Browser Use's ChatInvokeUsage.
            usage: ChatInvokeUsage | None = None
            if response.usage is not None:
                u = response.usage
                cached = (
                    u.input_tokens_details.cached_tokens
                    if u.input_tokens_details is not None
                    else None
                )
                usage = ChatInvokeUsage(
                    prompt_tokens=u.input_tokens,
                    prompt_cached_tokens=cached,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=u.output_tokens,
                    total_tokens=u.total_tokens,
                )

            return ChatInvokeCompletion(
                completion=parsed,
                usage=usage,
            )

        instance.ainvoke = ainvoke  # type: ignore[method-assign]
        return instance
