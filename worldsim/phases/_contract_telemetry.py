"""NDJSON telemetry for editor-contract events (commit 7 of the
registry refactor).

Every contract-related rejection fires a single line here; nothing in
the pipeline branches on the records. Consumers are post-run dashboards
/ ad-hoc analysis that want to see "how many plans got dropped at R6
this run, and why" without parsing validator logs.

Schema (one line per event):

    {
      "ts": ISO-8601 with timezone,
      "shard": str | None,
      "benign_task_id": str | None,
      "kind": str | None,
      "event_type": "validator_reject" | "substituter_phantom_token"
                   | "pre_shard_drop",
      "detail": dict  # event-specific payload
    }

Path: ``$WORLDSIM_STATE_DIR/phase_2/contract_events.ndjson`` (default
``logs/phase_2/contract_events.ndjson``). Append-only; every process
write opens-appends-closes so multiple shards running in parallel
don't clobber each other.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

EventType = Literal[
    "validator_reject",
    "substituter_phantom_token",
    "pre_shard_drop",
]


def _state_dir() -> Path:
    return Path(os.environ.get("WORLDSIM_STATE_DIR", "logs"))


def telemetry_path() -> Path:
    return _state_dir() / "phase_2" / "contract_events.ndjson"


def emit_contract_event(
    *,
    event_type: EventType,
    shard: str | None = None,
    benign_task_id: str | None = None,
    kind: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append one NDJSON line. Any I/O error is logged but never
    propagated — telemetry is observational, not a pipeline gate."""
    try:
        path = telemetry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "shard": shard,
            "benign_task_id": benign_task_id,
            "kind": kind,
            "event_type": event_type,
            "detail": dict(detail or {}),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("failed to emit contract_event %s", event_type)


__all__ = ["EventType", "emit_contract_event", "telemetry_path"]
