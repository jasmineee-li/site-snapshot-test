"""Report-only final-answer shape classification for Phase 4 analyses."""

from __future__ import annotations

import json
from typing import Any


def final_result_shape(value: Any) -> str:
    """Classify agent final answers for reporting without changing scoring."""

    if value is None:
        return "missing"
    if isinstance(value, dict | list):
        return "structured_json_value"
    text = str(value).strip()
    if not text:
        return "empty_string"
    if _is_json_text(text):
        return "json_string"
    fenced = _fenced_body(text)
    if fenced is not None and _is_json_text(fenced):
        return "fenced_json_string"
    if "retrieved_data" in text and "{" in text and "}" in text:
        return "prose_with_json_fragment"
    return "plain_text"


def _is_json_text(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _fenced_body(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or not lines[0].strip().startswith("```"):
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "```":
            return "\n".join(lines[1:index]).strip()
    return None
