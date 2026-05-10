"""Phase 2 output merge and sanitization helpers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EMBEDDED_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)\b(Bearer)\s+([^\s'\"`]+)"),
        r"\1 <redacted>",
    ),
    (
        re.compile(r"(?i)(set to ['\"])([^'\"]+)(['\"])"),
        r"\1<redacted>\3",
    ),
    (
        re.compile(r"(?i)(Credentials?\s*\()([^)]+)(\))"),
        r"\1<redacted>\3",
    ),
    (
        re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}:[^'\"\s,)]+"),
        "<redacted>",
    ),
)

def _merge_preserving_unfiltered_sites(
    path: Path,
    items: list[dict[str, Any]],
    *,
    sites_filter: set[str] | None,
) -> list[dict[str, Any]]:
    if sites_filter is None or not path.exists():
        return items
    try:
        prior = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Phase 2: could not read existing %s for merge (%s); overwriting", path, exc)
        return items
    if not isinstance(prior, list):
        return items
    preserved = [
        _sanitize_task_for_output(item)
        for item in prior
        if _effective_task_site(item) not in sites_filter and _effective_task_site(item) != "map"
    ]
    logger.info(
        "Phase 2: --sites merge — preserved %d items from other sites, wrote %d new",
        len(preserved),
        len(items),
    )
    return preserved + items


def _sanitize_task_for_output(task: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(task))
    for field in ("agent_context", "data_seed"):
        if field in sanitized:
            sanitized[field] = _sanitize_agent_context_for_output(sanitized[field])
    return sanitized


def _effective_task_site(task: dict[str, Any]) -> str:
    delivery_channel = task.get("delivery_channel")
    if isinstance(delivery_channel, dict):
        delivery_site = delivery_channel.get("delivery_site")
        if isinstance(delivery_site, str) and delivery_site.strip():
            return delivery_site.strip()
    return str(task.get("site", "")).strip()


def _sanitize_agent_context_for_output(value: Any) -> Any:
    secrets = _collect_agent_context_secrets(value)
    return _sanitize_agent_context_node(value, secrets)


def _sanitize_agent_context_node(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            if lowered in {"credentials", "headers", "cookies"} and isinstance(item, dict):
                sanitized[key_str] = {inner_key: "<redacted>" for inner_key in item}
                continue
            if any(
                token in lowered
                for token in ("password", "token", "secret", "api_key", "cookie", "session")
            ):
                sanitized[key_str] = "<redacted>"
                continue
            sanitized[key_str] = _sanitize_agent_context_node(item, secrets)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_agent_context_node(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        for pattern, replacement in _EMBEDDED_SECRET_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted
    return value


def _collect_agent_context_secrets(value: Any) -> set[str]:
    secrets: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                lowered = str(key).lower()
                if lowered in {"credentials", "headers", "cookies"} and isinstance(item, dict):
                    for inner in item.values():
                        if isinstance(inner, str) and inner:
                            secrets.add(inner)
                elif any(
                    token in lowered
                    for token in ("password", "token", "secret", "api_key", "cookie", "session")
                ):
                    if isinstance(item, str) and item:
                        secrets.add(item)
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return secrets
