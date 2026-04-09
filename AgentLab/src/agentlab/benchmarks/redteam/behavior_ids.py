"""Behavior identifier helpers shared across redteam entrypoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

from slugify import slugify


def resolve_behavior_id(case_data: dict, case_idx: int | None = None) -> str:
    """Return the benchmark-facing behavior ID, with stable fallback for missing IDs."""

    for key in ("id", "behavior_id"):
        raw_value = case_data.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            if raw_value.strip():
                return raw_value
            continue
        rendered = str(raw_value)
        if rendered.strip():
            return rendered

    if case_idx is not None:
        return f"case{case_idx}"
    return "unknown"


def normalize_behavior_spec(case_data: dict, case_idx: int | None = None) -> dict:
    """Return a shallow copy with ``id`` populated from the shared fallback policy."""

    normalized = dict(case_data)
    normalized["id"] = resolve_behavior_id(case_data, case_idx)
    return normalized


def controller_slug_for_behavior(behavior_id: str, logical_output_dir: str | Path | None = None) -> str:
    """Return a git/path-safe controller slug while preserving the raw behavior ID elsewhere."""

    base = slugify((behavior_id or "").strip(), separator="-", lowercase=True)
    if not base:
        base = "behavior"
    logical_output = ""
    if logical_output_dir is not None:
        logical_output = Path(logical_output_dir).as_posix()
    digest = hashlib.sha256(
        f"{behavior_id or ''}\0{logical_output}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{base}-{digest}"
