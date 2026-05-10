"""Text extraction helpers for Phase 4 variant trace reports."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_RETRIEVED_DATA_RE = re.compile(r'"retrieved_data"\s*:\s*\[\s*"([^"]+)"')


def compact_text(value: Any, *, limit: int = 240) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def answer_label(final_result: Any) -> str | None:
    """Extract the first retrieved_data string from common agent answer shapes."""

    if not isinstance(final_result, str) or not final_result.strip():
        return None
    direct_match = _RETRIEVED_DATA_RE.search(final_result)
    if direct_match:
        return direct_match.group(1)
    json_match = _JSON_OBJECT_RE.search(final_result)
    if not json_match:
        return None
    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None
    retrieved = payload.get("retrieved_data") if isinstance(payload, dict) else None
    if isinstance(retrieved, list) and retrieved and isinstance(retrieved[0], str):
        return retrieved[0]
    if isinstance(retrieved, str):
        return retrieved
    return None


__all__ = ["answer_label", "compact_text"]
