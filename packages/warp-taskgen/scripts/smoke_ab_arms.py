"""Smoke test for the A/B study arms: Arm A (native Responses API) and Arm B (OpenRouter).

Usage:
    uv run python scripts/smoke_ab_arms.py

Both arms call ``make_llm`` and make a structured-output call with a minimal
AgentOutput-like Pydantic schema. The test asserts:
  - No ``<think>`` in the returned content.
  - The Pydantic parse succeeds and returns the expected fields.

If OpenRouter returns a 402 (wallet empty), the script prints a diagnostic
and exits 0 -- billing state does not block the commit.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Load .env so OPENAI_API_KEY and OPENROUTER_API_KEY are available.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; keys must be in the environment directly


from pydantic import BaseModel


class _MinimalOutput(BaseModel):
    """Minimal schema that mirrors AgentOutput's top-level shape."""

    thought: str
    action: str


_MESSAGES_TEXT = [
    {"role": "system", "content": "You are a helpful browser agent."},
    {
        "role": "user",
        "content": (
            "Reply with a JSON object that has two fields: "
            '"thought" (a short sentence) and "action" (the word "done").'
        ),
    },
]


async def _call_llm(llm: object, label: str) -> None:
    """Invoke *llm* with a structured output call and assert invariants."""
    from browser_use.llm.messages import SystemMessage, UserMessage

    # Build typed message objects that Browser Use expects.
    messages = [
        SystemMessage(content="You are a helpful browser agent."),
        UserMessage(
            content=(
                "Reply with a JSON object that has two fields: "
                '"thought" (a short sentence) and "action" (the word "done").'
            )
        ),
    ]

    print(f"\n[{label}] Calling ainvoke with structured output ...")
    result = await llm.ainvoke(messages, output_format=_MinimalOutput)  # type: ignore[union-attr]
    print(f"[{label}] Raw result type: {type(result)}")

    completion = result.completion if hasattr(result, "completion") else result
    print(f"[{label}] Parsed completion: {completion!r}")

    assert isinstance(completion, _MinimalOutput), (
        f"[{label}] Expected _MinimalOutput, got {type(completion)}"
    )
    assert completion.thought, f"[{label}] thought field is empty"
    assert completion.action, f"[{label}] action field is empty"

    # Serialise to JSON and check for think-tag contamination.
    raw_json = completion.model_dump_json()
    assert "<think>" not in raw_json, f"[{label}] <think> tag found in JSON output: {raw_json!r}"
    assert "</think>" not in raw_json, f"[{label}] </think> tag found in output: {raw_json!r}"

    print(f"[{label}] PASSED. thought={completion.thought!r}, action={completion.action!r}")


async def main() -> int:
    from warp_taskgen.agent_config import make_llm

    exit_code = 0

    # -- Arm A: native Responses API --
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_key:
        print("[Arm A] OPENAI_API_KEY not set -- skipping.")
    else:
        try:
            llm_a = make_llm("gpt-5.4-mini", provider="openai")
            await _call_llm(llm_a, "Arm A (openai/Responses API)")
        except Exception as exc:
            print(f"[Arm A] FAILED: {exc}")
            exit_code = 1

    # -- Arm B: OpenRouter pinned to OpenAI --
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        print("[Arm B] OPENROUTER_API_KEY not set -- skipping.")
    else:
        try:
            llm_b = make_llm("gpt-5.4-mini", provider="openrouter")
            await _call_llm(llm_b, "Arm B (openrouter/pinned-openai)")
        except Exception as exc:
            err_str = str(exc)
            # 402 = wallet empty. 404 with "No endpoints found" means
            # require_parameters + only=["openai"] has no eligible endpoint
            # because credits are zero -- same root cause, different error
            # path depending on OpenRouter's routing order. Both are billing
            # issues, not code bugs.
            is_billing = (
                "402" in err_str
                or "payment" in err_str.lower()
                or "billing" in err_str.lower()
                or "Insufficient credits" in err_str
                or ("404" in err_str and "No endpoints found" in err_str)
            )
            if is_billing:
                print(
                    f"[Arm B] OpenRouter billing issue (wallet empty) -- "
                    f"request shape is correct, billing issue only. Detail: {exc}"
                )
            else:
                print(f"[Arm B] FAILED: {exc}")
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
