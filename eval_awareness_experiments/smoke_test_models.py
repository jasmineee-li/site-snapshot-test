"""One-call smoke test for the 6 models in the causal eval-awareness experiment.

Verifies for each model:
1. The OpenRouter ID resolves (no 404).
2. The request returns non-empty content.
3. Reasoning mode engages — measured via `usage.completion_tokens_details.reasoning_tokens`
   for models that support OpenRouter's reasoning param. The reasoning *text*
   surfaces on `message.reasoning` for Anthropic/GLM/Gemini-Pro, but is hidden
   by OpenAI (only the token count is exposed). Both are acceptable for the
   experiment; the smoke test reports both and only fails when reasoning
   doesn't engage at all.

Uses a prompt that genuinely benefits from reasoning. A trivial prompt like
"What is 17 * 23?" is too easy — Opus 4.7 will answer it without engaging
reasoning mode at all, producing a false negative.

Usage:
    python -m eval_awareness_experiments.smoke_test_models

Cost: ~$0.10 across all 6 models.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            val = val.strip('"').strip("'")
            os.environ[key.strip()] = val


# (model_id, reasoning_mode_enabled, reasoning_text_visible)
# reasoning_mode_enabled=True → expect reasoning_tokens > 0
# reasoning_text_visible=True → expect message.reasoning to be non-empty
# OpenAI models engage reasoning but hide the text (reasoning_visible=False).
# Gemini previews route reasoning into the content rather than message.reasoning.
MODELS: list[tuple[str, bool, bool]] = [
    ("z-ai/glm-5", True, True),
    ("anthropic/claude-sonnet-4.6", True, True),  # substitute for 4.7
    ("anthropic/claude-opus-4.7", True, True),
    ("openai/gpt-5.2", True, False),  # OpenAI hides reasoning text
    ("google/gemini-3-flash-preview", False, False),  # reasoning in content
    ("google/gemini-3.1-pro-preview", False, False),  # reasoning in content
]

# Genuinely needs reasoning — trivial prompts produce false negatives
# because strong models just answer them directly.
PROMPT = (
    "A snail climbs a 30-meter wall. Each day it climbs 7 meters during "
    "the day and slides down 4 meters at night. How many days does it take "
    "to reach the top? Reason step by step before answering."
)


@dataclass
class SmokeResult:
    model: str
    ok: bool
    content_chars: int
    reasoning_chars: int
    reasoning_tokens: int
    elapsed_s: float
    error: str | None = None
    note: str | None = None


async def _smoke_one(
    client: AsyncOpenAI,
    model: str,
    reasoning_engaged: bool,
    reasoning_visible: bool,
) -> SmokeResult:
    t0 = time.monotonic()
    extra: dict = {}
    if reasoning_engaged:
        extra["extra_body"] = {"reasoning": {"effort": "high"}}
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=4096,
            **extra,
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        reasoning_text = (getattr(msg, "reasoning", None) or "").strip()
        reasoning_tokens = (
            resp.usage.completion_tokens_details.reasoning_tokens or 0
            if resp.usage and resp.usage.completion_tokens_details
            else 0
        )
        elapsed = time.monotonic() - t0

        if not content:
            return SmokeResult(
                model=model,
                ok=False,
                content_chars=0,
                reasoning_chars=len(reasoning_text),
                reasoning_tokens=reasoning_tokens,
                elapsed_s=elapsed,
                error="empty_content",
            )

        # Models with reasoning_engaged=True must show reasoning_tokens > 0,
        # otherwise reasoning mode isn't actually firing.
        if reasoning_engaged and reasoning_tokens == 0:
            return SmokeResult(
                model=model,
                ok=False,
                content_chars=len(content),
                reasoning_chars=len(reasoning_text),
                reasoning_tokens=0,
                elapsed_s=elapsed,
                error="reasoning_not_engaged (reasoning_tokens=0)",
            )

        # Models with reasoning_visible=True should also surface text on
        # message.reasoning. If text is empty we warn but don't fail — the
        # tokens engaged, that's what matters operationally.
        note = None
        if reasoning_engaged and reasoning_visible and not reasoning_text:
            note = "reasoning engaged but text hidden (reasoning_chars=0)"

        return SmokeResult(
            model=model,
            ok=True,
            content_chars=len(content),
            reasoning_chars=len(reasoning_text),
            reasoning_tokens=reasoning_tokens,
            elapsed_s=elapsed,
            note=note,
        )
    except Exception as e:
        return SmokeResult(
            model=model,
            ok=False,
            content_chars=0,
            reasoning_chars=0,
            reasoning_tokens=0,
            elapsed_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
        )


async def main() -> int:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in env or .env", file=sys.stderr)
        return 2

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=0,
    )

    print(f"Smoke-testing {len(MODELS)} models on snail-climbing prompt.")
    print()

    results: list[SmokeResult] = []
    for model, reasoning_engaged, reasoning_visible in MODELS:
        suffix = " (thinking)" if reasoning_engaged else ""
        print(f"  → {model}{suffix} ... ", end="", flush=True)
        r = await _smoke_one(client, model, reasoning_engaged, reasoning_visible)
        results.append(r)
        if r.ok:
            note = f"  [{r.note}]" if r.note else ""
            print(
                f"OK  ({r.elapsed_s:5.1f}s, content={r.content_chars}, "
                f"reasoning_tokens={r.reasoning_tokens}, "
                f"reasoning_chars={r.reasoning_chars}){note}"
            )
        else:
            print(f"FAIL  ({r.elapsed_s:5.1f}s) — {r.error}")

    print()
    print("=" * 80)
    n_pass = sum(1 for r in results if r.ok)
    print(f"Summary: {n_pass}/{len(results)} models passed")
    if n_pass < len(results):
        print("\nFailures:")
        for r in results:
            if not r.ok:
                print(f"  {r.model}: {r.error}")
        return 1
    notes = [r for r in results if r.note]
    if notes:
        print("\nNotes (informational, not failures):")
        for r in notes:
            print(f"  {r.model}: {r.note}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
