"""Small diagnostics mergers for matched rewrite completion retries."""

from __future__ import annotations

import copy
from typing import Any

_LIST_FIELDS = (
    "completion_kwargs",
    "completion_responses",
    "parse_errors",
    "completion_errors",
    "retry_feedback",
    "retry_fallbacks",
)


def merge_completion_diagnostics(
    previous: dict[str, Any] | None,
    current: object,
) -> dict[str, Any]:
    """Combine independent stream calls without dropping captured attempts."""

    if not isinstance(current, dict):
        return copy.deepcopy(previous) if isinstance(previous, dict) else {}
    if not isinstance(previous, dict):
        return copy.deepcopy(current)

    merged = copy.deepcopy(previous)
    for field in _LIST_FIELDS:
        prior_items = merged.get(field)
        current_items = current.get(field)
        if isinstance(prior_items, list) and isinstance(current_items, list):
            merged[field] = prior_items + copy.deepcopy(current_items)
        elif isinstance(current_items, list):
            merged[field] = copy.deepcopy(current_items)
    for field in ("attempts", "transport_attempts", "elapsed_s"):
        prior_value = merged.get(field)
        current_value = current.get(field)
        if isinstance(prior_value, (int, float)) and isinstance(current_value, (int, float)):
            merged[field] = prior_value + current_value
        elif isinstance(current_value, (int, float)):
            merged[field] = current_value
    prior_truncation = merged.get("incomplete_output")
    current_truncation = current.get("incomplete_output")
    if prior_truncation is not None and current_truncation is not None:
        merged["incomplete_output"] = [prior_truncation, copy.deepcopy(current_truncation)]
    elif current_truncation is not None:
        merged["incomplete_output"] = copy.deepcopy(current_truncation)
    for key, value in current.items():
        if key not in _LIST_FIELDS and key not in {
            "attempts",
            "transport_attempts",
            "elapsed_s",
            "incomplete_output",
        }:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_retry_diagnostics(
    previous: dict[str, Any] | None,
    current: object,
    *,
    selected_max_tokens: int,
) -> dict[str, Any]:
    """Merge one stream attempt and project the token bound used for it."""

    merged = merge_completion_diagnostics(previous, current)
    merged["selected_max_tokens"] = selected_max_tokens
    return merged


__all__ = ["merge_completion_diagnostics", "merge_retry_diagnostics"]
