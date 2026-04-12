"""Helpers for turning task identifiers into safe filesystem paths."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_UNSAFE_TASK_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_task_path_component(task_id: Any) -> str:
    """Return a deterministic, collision-safe path component for *task_id*."""
    raw = str(task_id or "unknown")
    sanitized = _UNSAFE_TASK_PATH_CHARS.sub("_", raw).strip("._-")
    if not sanitized or sanitized in {".", ".."}:
        sanitized = "task"

    if sanitized == raw:
        return sanitized

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}__{digest}"
