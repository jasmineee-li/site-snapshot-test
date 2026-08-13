"""Bounded, non-secret advisory lifecycle history for Run control."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HISTORY_FILE = "run_control_history.jsonl"
_HISTORY_SCHEMA_VERSION = 1
_HISTORY_MAX_EVENTS = 256
_HISTORY_MAX_LINE_BYTES = 2048
_EVENTS = frozenset({"pause_requested", "paused", "interrupted"})


def transition_history_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve(strict=False) / _HISTORY_FILE


def append_history_event(
    state_dir: Path,
    event: str,
    *,
    request_id: str | None = None,
    step: str | None = None,
    reason_code: str | None = None,
    signal_name: str | None = None,
) -> None:
    """Best-effort append of one whitelisted bounded advisory event.

    History never participates in lifecycle routing or checkpoint acceptance.
    Once the cap is reached, new events are omitted rather than rewriting old
    records, preserving the append-only evidence surface.
    """

    if event not in _EVENTS:
        return
    payload: dict[str, object] = {
        "schema_version": _HISTORY_SCHEMA_VERSION,
        "event": event,
        "at": datetime.now(UTC).isoformat(),
    }
    for key, value in (
        ("request_id", request_id),
        ("step", step),
        ("reason_code", reason_code),
        ("signal", signal_name),
    ):
        if isinstance(value, str) and value.strip():
            payload[key] = value.strip()[:160]
    try:
        path = transition_history_path(state_dir)
        if path.exists():
            with path.open("rb") as handle:
                if sum(1 for _ in handle) >= _HISTORY_MAX_EVENTS:
                    return
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        if len(encoded) > _HISTORY_MAX_LINE_BYTES:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(encoded)
    except OSError:
        # Explanatory history must never make a lifecycle transition fail.
        return


def load_transition_history(state_dir: Path) -> list[dict[str, object]]:
    """Read only whitelisted history fields for an explanatory status view."""

    path = transition_history_path(state_dir)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-_HISTORY_MAX_EVENTS:]
    except OSError:
        return []
    events: list[dict[str, object]] = []
    for line in lines:
        try:
            raw: Any = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or raw.get("event") not in _EVENTS:
            continue
        event: dict[str, object] = {
            "schema_version": _HISTORY_SCHEMA_VERSION,
            "event": raw["event"],
        }
        for key in ("at", "request_id", "step", "reason_code", "signal"):
            value = raw.get(key)
            if isinstance(value, str) and len(value) <= 160:
                event[key] = value
        events.append(event)
    return events


__all__ = ["append_history_event", "load_transition_history", "transition_history_path"]
