"""Thin async LLM client wrapper for OpenAI, Anthropic, and OpenRouter."""

import asyncio
import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
import openai


@dataclass
class LLMConfig:
    provider: str  # "openai" | "anthropic" | "openrouter"
    model: str  # e.g. "gpt-4o", "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 1024


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.provider in ("openai", "openrouter"):
            kwargs = {}
            if config.provider == "openrouter":
                kwargs["base_url"] = "https://openrouter.ai/api/v1"
                kwargs["api_key"] = os.environ["OPENROUTER_API_KEY"]
            self._openai = openai.AsyncOpenAI(**kwargs)
        elif config.provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic()
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    async def call(self, messages: list[dict], images: list[Path] | None = None) -> str:
        """Call the LLM with messages and optional images.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            images: Optional list of image file paths to attach to the last user message.

        Returns:
            Raw text response from the model.
        """
        for attempt in range(3):
            try:
                if self.config.provider in ("openai", "openrouter"):
                    return await self._call_openai(messages, images)
                else:
                    return await self._call_anthropic(messages, images)
            except (
                openai.RateLimitError,
                openai.APIStatusError,
                anthropic.RateLimitError,
                anthropic.APIStatusError,
            ) as e:
                # Retry on rate limit (429) and server errors (5xx)
                is_retryable = False
                if isinstance(e, (openai.RateLimitError, anthropic.RateLimitError)):
                    is_retryable = True
                elif isinstance(e, (openai.APIStatusError, anthropic.APIStatusError)):
                    is_retryable = e.status_code >= 500

                if is_retryable and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                raise

        raise RuntimeError("Unreachable")

    async def _call_openai(self, messages: list[dict], images: list[Path] | None) -> str:
        formatted = _format_openai_messages(messages, images)
        resp = await self._openai.chat.completions.create(
            model=self.config.model,
            messages=formatted,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content

    async def _call_anthropic(self, messages: list[dict], images: list[Path] | None) -> str:
        system_text, formatted = _format_anthropic_messages(messages, images)
        kwargs = dict(
            model=self.config.model,
            messages=formatted,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if system_text:
            kwargs["system"] = system_text
        resp = await self._anthropic.messages.create(**kwargs)
        return resp.content[0].text


def _encode_image(path: Path) -> tuple[str, str]:
    """Read and base64-encode an image file. Returns (base64_data, media_type)."""
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8"), mime_type


def _format_openai_messages(messages: list[dict], images: list[Path] | None) -> list[dict]:
    """Format messages for OpenAI API, attaching images to the last user message."""
    formatted = []
    for msg in messages:
        formatted.append({"role": msg["role"], "content": msg["content"]})

    if images:
        # Find last user message and convert content to multimodal format
        for i in range(len(formatted) - 1, -1, -1):
            if formatted[i]["role"] == "user":
                text_content = formatted[i]["content"]
                parts = [{"type": "text", "text": text_content}]
                for img_path in images:
                    b64, media_type = _encode_image(img_path)
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        }
                    )
                formatted[i]["content"] = parts
                break

    return formatted


def _format_anthropic_messages(
    messages: list[dict], images: list[Path] | None
) -> tuple[str | None, list[dict]]:
    """Format messages for Anthropic API. Returns (system_text, messages)."""
    system_text = None
    formatted = []

    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            formatted.append({"role": msg["role"], "content": msg["content"]})

    if images:
        # Find last user message and convert content to multimodal format
        for i in range(len(formatted) - 1, -1, -1):
            if formatted[i]["role"] == "user":
                text_content = formatted[i]["content"]
                parts = []
                for img_path in images:
                    b64, media_type = _encode_image(img_path)
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        }
                    )
                parts.append({"type": "text", "text": text_content})
                formatted[i]["content"] = parts
                break

    return system_text, formatted
