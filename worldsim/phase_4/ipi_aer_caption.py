"""Claude vision call that captions the IPI max-coverage screenshot.

The IPI-AER judge's caption-mode input needs a Markdown description of
the frame where the indirect-prompt-injection payload was most visible.
OS-Harm uses GPT-4o for this; we use Claude Sonnet 4.6 to stay on a
single provider. The captioner prompt text at
``worldsim/prompts/ipi-aer-caption.md`` is the OS-Harm verbatim string.

Caption-frame choice (deliberate OS-Harm deviation): we caption the IPI
max-coverage screenshot, not the terminal frame. The realization moment
is typically early; by the final step the injection modal is often
closed, tab switched, or page navigated away. The max-coverage frame
preserves what the agent actually saw when forming awareness.

The caption is persisted to ``<trajectory_dir>/ipi_aer_caption.txt`` for
offline audit and reused across all 5-25 downstream judge calls in a
single trajectory (Anthropic prompt caching further amortises the cost
within-trajectory; see ``ipi_aer_p_eval_api``).
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.phase_4.anthropic_client import (
    call_with_retry,
    classify_api_exception,
    get_client,
    normalize_model_for_auth,
)
from worldsim.phase_4.concurrency import get_api_semaphore
from worldsim.prompt_loading import load_prompt

logger = logging.getLogger(__name__)

_CAPTION_PROMPT_PATH = "ipi-aer-caption"
_CAPTION_MAX_TOKENS = 1024


def _read_png_bytes(path: Path) -> bytes:
    """Read a PNG as bytes; raise FileNotFoundError with a clear message."""
    if not path.exists():
        raise FileNotFoundError(f"screenshot not found at {path}")
    return path.read_bytes()


def _build_messages(png_bytes: bytes) -> list[dict[str, Any]]:
    """Build the vision-call message payload.

    The captioner prompt text goes first, the image block second, matching
    OS-Harm's ``format_msg_for_captioning`` ordering.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    prompt = load_prompt(_CAPTION_PROMPT_PATH).strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
            ],
        }
    ]


def _model_metadata() -> dict[str, str]:
    return {"user_id": "worldsim-v5-ipi-aer-caption"}


def _synthesize_summary(response: Any, *, sandbox_model: str, elapsed_s: float) -> str:
    """Mirror the cost-tracker summary shape from ``judge_api._synthesize_summary``.

    Kept local rather than imported so the caption sub-bucket records to
    its own key (``phase_4:aer:caption``) without cross-coupling back to
    the judge module.
    """
    import json as _json

    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    cost = (in_tok / 1_000_000) * 3.0 + (out_tok / 1_000_000) * 15.0
    return _json.dumps(
        {
            "total_cost_usd": cost,
            "num_turns": 1,
            "duration_ms": int(elapsed_s * 1000),
            "session_id": getattr(response, "id", None),
            "model_usage": {
                sandbox_model: {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0)
                    or 0,
                    "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                }
            },
        }
    )


def _extract_caption_text(response: Any) -> str | None:
    """Pull the Markdown caption out of the Messages API response.

    Claude's non-tool-use response is a list of content blocks; the first
    text block carries the caption. Returns None if the response contains
    no text blocks (unexpected — would indicate a refusal or a malformed
    return).
    """
    content = getattr(response, "content", None) or []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


async def caption_screenshot(
    screenshot_path: Path | str,
    *,
    trajectory_dir: Path | str,
    sandbox_model: str = "claude-sonnet-4-6",
    client: AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Caption one screenshot and persist the result.

    Returns a dict with:

    - ``caption``: the Markdown caption text, or ``None`` on failure
    - ``failure_class``: ``None`` on success, otherwise one of
      ``auth_invalid`` / ``insufficient_credits`` / ``quota_exceeded`` /
      ``api_error`` / ``missing_screenshot`` / ``empty_response``

    The caption is persisted at ``<trajectory_dir>/ipi_aer_caption.txt``
    when successful. If ``ipi_aer_caption.txt`` already exists the cached
    text is returned without an API call — amortises the cost when the
    Phase-4 pipeline re-runs the gate (e.g. resume after crash).
    """
    screenshot_path = Path(screenshot_path)
    trajectory_dir = Path(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    cache_path = trajectory_dir / "ipi_aer_caption.txt"
    if cache_path.exists():
        try:
            cached = cache_path.read_text(encoding="utf-8").strip()
        except OSError:
            cached = ""
        if cached:
            return {"caption": cached, "failure_class": None, "cached": True}

    try:
        png_bytes = _read_png_bytes(screenshot_path)
    except FileNotFoundError as exc:
        logger.warning("caption: %s", exc)
        return {
            "caption": None,
            "failure_class": "missing_screenshot",
            "cached": False,
        }

    client = client or get_client()
    messages = _build_messages(png_bytes)

    async def _call() -> Any:
        return await client.messages.create(
            model=normalize_model_for_auth(sandbox_model),
            max_tokens=_CAPTION_MAX_TOKENS,
            messages=messages,
            metadata=_model_metadata(),
        )

    t0 = time.monotonic()
    try:
        async with get_api_semaphore():
            response = await call_with_retry(_call, retries=3, label="aer-caption")
    except Exception as exc:  # broad — bucket via failure_class
        logger.warning("caption call failed: %s", exc)
        return {
            "caption": None,
            "failure_class": classify_api_exception(exc),
            "cached": False,
        }

    elapsed = time.monotonic() - t0
    caption = _extract_caption_text(response)

    summary = _synthesize_summary(response, sandbox_model=sandbox_model, elapsed_s=elapsed)
    cost_tracker.record("phase_4:aer:caption", summary)

    if not caption:
        return {"caption": None, "failure_class": "empty_response", "cached": False}

    try:
        cache_path.write_text(caption, encoding="utf-8")
    except OSError as exc:  # best-effort persistence
        logger.warning("could not persist ipi_aer_caption.txt at %s: %s", cache_path, exc)

    return {"caption": caption, "failure_class": None, "cached": False}


__all__ = ["caption_screenshot"]
