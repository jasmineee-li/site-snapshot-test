"""Async LLM wrapper using OpenAI SDK + OpenRouter.

Provides a thin interface matching what the judges expect:
    result = await model.generate(prompt)
    text = result.message.text
"""

import asyncio
import base64
import os
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

# Load .env from project root if present
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            val = val.strip('"').strip("'")
            os.environ[key.strip()] = val


@dataclass
class _Message:
    """Mimics inspect_ai's message interface."""
    text: str


@dataclass
class GenerateResult:
    """Result from an LLM generate call. Matches the interface judges expect."""
    message: _Message


class LLM:
    """Async LLM client using OpenRouter via the OpenAI SDK.

    Usage:
        model = LLM("anthropic/claude-sonnet-4")
        result = await model.generate("Hello!")
        print(result.message.text)
    """

    def __init__(self, model: str, max_tokens: int = 4096, thinking: bool = False, concurrency: int = 64):
        # Strip :thinking suffix and enable thinking mode
        if model.endswith(":thinking"):
            model = model.removesuffix(":thinking")
            thinking = True
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    async def generate(
        self, prompt: str | list, response_format: dict | None = None,
    ) -> GenerateResult:
        """Generate a response from the model.

        Args:
            prompt: Either a string (sent as single user message) or a list
                of chat message objects (with .role and .content attributes).
            response_format: Optional response format constraint, e.g.
                {"type": "json_object"} for structured JSON output.

        Returns:
            GenerateResult with .message.text containing the response.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": m.role, "content": m.content} for m in prompt]

        kwargs = self._extra_kwargs()
        if response_format:
            kwargs["response_format"] = response_format

        async with self._semaphore:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        text = resp.choices[0].message.content or ""
        return GenerateResult(message=_Message(text=text))

    async def generate_json(self, prompt: str | list) -> GenerateResult:
        """Generate a JSON response from the model.

        Convenience wrapper that sets response_format to json_object.
        """
        return await self.generate(
            prompt, response_format={"type": "json_object"},
        )

    async def generate_with_images(
        self, text_prompt: str, image_paths: list[str | Path]
    ) -> GenerateResult:
        """Generate a response with one or more image inputs.

        Images are attached in order, followed by the text prompt. This matches
        how most vision models expect multi-image chat turns. Supply a single-
        element list for the common one-image case.

        Args:
            text_prompt: The text prompt to send.
            image_paths: Paths to image files, in the order the model should see them.

        Returns:
            GenerateResult with .message.text containing the response.
        """
        content: list[dict] = []
        for raw_path in image_paths:
            image_path = Path(raw_path)
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            suffix = image_path.suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
                suffix, "image/png"
            )
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": "user", "content": content}]

        async with self._semaphore:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                **self._extra_kwargs(),
            )
        text = resp.choices[0].message.content or ""
        return GenerateResult(message=_Message(text=text))

    def _extra_kwargs(self) -> dict:
        """Build extra kwargs for the API call (e.g. thinking/reasoning)."""
        if not self.thinking:
            return {}
        return {
            "extra_body": {
                "reasoning": {"effort": "high"},
            },
        }

    def __repr__(self) -> str:
        return f"LLM(model={self.model!r})"
