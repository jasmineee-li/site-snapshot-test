"""Thin wrappers around Browser Use LLM classes that strip reasoning traces.

Some models (e.g. gpt-5.4-mini via OpenRouter) prepend ``<think>...</think>``
reasoning blocks to their structured JSON output, even when ``response_format``
is set to ``json_schema``.  Browser Use's ``model_validate_json`` call rejects
this with "Invalid JSON: trailing characters" and burns a retry cycle on each
occurrence.

These wrappers intercept the raw response content, strip any ``<think>`` tags,
and then delegate to the parent's parsing logic.  The stripping is a no-op when
the response is already clean JSON, so the wrappers are safe to use for all
models.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Pre-compiled patterns matching Browser Use's own (unused) _remove_think_tags.
_THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)
_STRAY_CLOSE_TAG = re.compile(r".*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning traces from LLM output.

    Handles both well-formed pairs and stray closing tags (the model sometimes
    streams an opening ``<think>`` that gets truncated, leaving only
    ``</think>`` visible in the response).

    Returns the original text unchanged when no tags are present.
    """
    if "</think>" not in text:
        return text
    # Step 1: Remove well-formed <think>...</think>
    cleaned = re.sub(_THINK_TAGS, "", text)
    # Step 2: If there's an unmatched closing tag </think>,
    #         remove everything up to and including that.
    cleaned = re.sub(_STRAY_CLOSE_TAG, "", cleaned)
    result = cleaned.strip()
    if result != text.strip():
        logger.debug("Stripped <think> reasoning tags from LLM response")
    return result


def _patch_content(content: str | None) -> str | None:
    """Strip think tags from response content if present."""
    if content is None:
        return None
    return strip_think_tags(content)


class ChatOpenRouterClean:
    """``ChatOpenRouter`` subclass that strips ``<think>`` tags before JSON parsing.

    Constructed lazily by ``make_llm`` so the import only fires when the
    openrouter provider is actually selected.
    """

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        """Build a patched ChatOpenRouter instance."""
        from browser_use.llm.openrouter.chat import ChatOpenRouter

        base_cls = ChatOpenRouter
        instance = base_cls(**kwargs)
        # Monkey-patch the ainvoke method on this specific instance to
        # intercept and clean the raw response before JSON parsing.
        original_ainvoke = instance.ainvoke

        async def patched_ainvoke(messages, output_format=None, **kw):
            if output_format is None:
                return await original_ainvoke(messages, output_format=None, **kw)

            # For structured output: we need to intercept the raw API response
            # and clean it before model_validate_json.  The cleanest way is to
            # call the underlying OpenAI client directly, strip tags, then parse.
            from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
            from browser_use.llm.openrouter.serializer import OpenRouterMessageSerializer
            from browser_use.llm.schema import SchemaOptimizer
            from browser_use.llm.views import ChatInvokeCompletion
            from openai import APIConnectionError, APIStatusError, RateLimitError
            from openai.types.shared_params.response_format_json_schema import (
                JSONSchema,
                ResponseFormatJSONSchema,
            )

            openrouter_messages = OpenRouterMessageSerializer.serialize_messages(messages)
            extra_headers = {}
            if instance.http_referer:
                extra_headers["HTTP-Referer"] = instance.http_referer

            try:
                schema = SchemaOptimizer.create_optimized_json_schema(output_format)
                response_format_schema: JSONSchema = {
                    "name": "agent_output",
                    "strict": True,
                    "schema": schema,
                }

                response = await instance.get_client().chat.completions.create(
                    model=instance.model,
                    messages=openrouter_messages,
                    temperature=instance.temperature,
                    top_p=instance.top_p,
                    seed=instance.seed,
                    response_format=ResponseFormatJSONSchema(
                        json_schema=response_format_schema,
                        type="json_schema",
                    ),
                    extra_headers=extra_headers,
                    **(instance.extra_body or {}),
                )

                raw_content = response.choices[0].message.content
                if raw_content is None:
                    raise ModelProviderError(
                        message="Failed to parse structured output from model response",
                        status_code=500,
                        model=instance.name,
                    )

                # Strip <think> tags before parsing
                clean_content = _patch_content(raw_content)
                usage = instance._get_usage(response)
                parsed = output_format.model_validate_json(clean_content)

                return ChatInvokeCompletion(
                    completion=parsed,
                    usage=usage,
                )

            except RateLimitError as e:
                raise ModelRateLimitError(message=e.message, model=instance.name) from e
            except APIConnectionError as e:
                raise ModelProviderError(message=str(e), model=instance.name) from e
            except APIStatusError as e:
                raise ModelProviderError(
                    message=e.message, status_code=e.status_code, model=instance.name
                ) from e
            except (ModelProviderError, ModelRateLimitError):
                raise
            except Exception as e:
                raise ModelProviderError(message=str(e), model=instance.name) from e

        instance.ainvoke = patched_ainvoke
        return instance


class ChatOpenAIClean:
    """``ChatOpenAI`` subclass that strips ``<think>`` tags before JSON parsing.

    Same approach as ``ChatOpenRouterClean`` but for the direct OpenAI provider
    path, since gpt-5.4-mini can go through either depending on available keys.
    """

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        """Build a patched ChatOpenAI instance."""
        from browser_use.llm.openai.chat import ChatOpenAI

        base_cls = ChatOpenAI
        instance = base_cls(**kwargs)
        original_ainvoke = instance.ainvoke

        async def patched_ainvoke(messages, output_format=None, **kw):
            if output_format is None:
                return await original_ainvoke(messages, output_format=None, **kw)

            from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
            from browser_use.llm.openai.serializer import OpenAIMessageSerializer
            from browser_use.llm.schema import SchemaOptimizer
            from browser_use.llm.views import ChatInvokeCompletion
            from openai import APIConnectionError, APIStatusError, RateLimitError
            from openai.types.shared_params.response_format_json_schema import (
                JSONSchema,
                ResponseFormatJSONSchema,
            )

            openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

            try:
                model_params: dict[str, Any] = {}
                if instance.temperature is not None:
                    model_params["temperature"] = instance.temperature
                if instance.frequency_penalty is not None:
                    model_params["frequency_penalty"] = instance.frequency_penalty
                if instance.max_completion_tokens is not None:
                    model_params["max_completion_tokens"] = instance.max_completion_tokens
                if instance.top_p is not None:
                    model_params["top_p"] = instance.top_p
                if instance.seed is not None:
                    model_params["seed"] = instance.seed
                if instance.service_tier is not None:
                    model_params["service_tier"] = instance.service_tier

                if instance.reasoning_models and any(
                    str(m).lower() in str(instance.model).lower() for m in instance.reasoning_models
                ):
                    model_params["reasoning_effort"] = instance.reasoning_effort
                    model_params.pop("temperature", None)
                    model_params.pop("frequency_penalty", None)

                schema = SchemaOptimizer.create_optimized_json_schema(
                    output_format,
                    remove_min_items=instance.remove_min_items_from_schema,
                    remove_defaults=instance.remove_defaults_from_schema,
                )
                response_format: JSONSchema = {
                    "name": "agent_output",
                    "strict": True,
                    "schema": schema,
                }

                if (
                    instance.add_schema_to_system_prompt
                    and openai_messages
                    and openai_messages[0]["role"] == "system"
                ):
                    from collections.abc import Iterable

                    from openai.types.chat import ChatCompletionContentPartTextParam

                    schema_text = f"\n<json_schema>\n{response_format}\n</json_schema>"
                    if isinstance(openai_messages[0]["content"], str):
                        openai_messages[0]["content"] += schema_text
                    elif isinstance(openai_messages[0]["content"], Iterable):
                        openai_messages[0]["content"] = list(openai_messages[0]["content"]) + [
                            ChatCompletionContentPartTextParam(text=schema_text, type="text")
                        ]

                if instance.dont_force_structured_output:
                    response = await instance.get_client().chat.completions.create(
                        model=instance.model,
                        messages=openai_messages,
                        **model_params,
                    )
                else:
                    response = await instance.get_client().chat.completions.create(
                        model=instance.model,
                        messages=openai_messages,
                        response_format=ResponseFormatJSONSchema(
                            json_schema=response_format, type="json_schema"
                        ),
                        **model_params,
                    )

                choice = response.choices[0] if response.choices else None
                if choice is None:
                    base_url = str(instance.base_url) if instance.base_url is not None else None
                    hint = f" (base_url={base_url})" if base_url is not None else ""
                    raise ModelProviderError(
                        message=(
                            "Invalid OpenAI chat completion response: missing or empty `choices`."
                            " If you are using a proxy via `base_url`, ensure it implements the OpenAI"
                            " `/v1/chat/completions` schema and returns `choices` as a non-empty list."
                            f"{hint}"
                        ),
                        status_code=502,
                        model=instance.name,
                    )

                if choice.message.content is None:
                    raise ModelProviderError(
                        message="Failed to parse structured output from model response",
                        status_code=500,
                        model=instance.name,
                    )

                # Strip <think> tags before parsing
                clean_content = _patch_content(choice.message.content)
                usage = instance._get_usage(response)
                parsed = output_format.model_validate_json(clean_content)

                return ChatInvokeCompletion(
                    completion=parsed,
                    usage=usage,
                    stop_reason=choice.finish_reason,
                )

            except (ModelProviderError, ModelRateLimitError):
                raise
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

        instance.ainvoke = patched_ainvoke
        return instance
