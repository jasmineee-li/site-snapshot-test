"""Async LLM wrapper using OpenAI SDK + OpenRouter.

Provides a thin interface matching what the judges expect:
    result = await model.generate(prompt)
    text = result.message.text
"""

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

    def __init__(self, model: str, max_tokens: int = 4096, thinking: bool = False):
        # Strip :thinking suffix and enable thinking mode
        if model.endswith(":thinking"):
            model = model.removesuffix(":thinking")
            thinking = True
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    async def generate(self, prompt: str | list) -> GenerateResult:
        """Generate a response from the model.

        Args:
            prompt: Either a string (sent as single user message) or a list
                of chat message objects (with .role and .content attributes).

        Returns:
            GenerateResult with .message.text containing the response.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": m.role, "content": m.content} for m in prompt]

        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            **self._extra_kwargs(),
        )
        text = resp.choices[0].message.content or ""
        return GenerateResult(message=_Message(text=text))

    async def generate_with_image(
        self, text_prompt: str, image_path: str | Path
    ) -> GenerateResult:
        """Generate a response with an image input (for screenshot experiments).

        Args:
            text_prompt: The text prompt to send.
            image_path: Path to the image file.

        Returns:
            GenerateResult with .message.text containing the response.
        """
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        suffix = image_path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
            suffix, "image/png"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

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
