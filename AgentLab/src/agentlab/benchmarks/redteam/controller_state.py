"""Shared controller-state helpers without controller import cycles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.phase_ids import CANONICAL_PHASES


def controller_state_path(logs_dir: str | Path) -> Path:
    return Path(logs_dir) / "controller_state.json"


def controller_events_path(logs_dir: str | Path) -> Path:
    return Path(logs_dir) / "controller_events.jsonl"


def generation_phase_status_template() -> dict[str, dict[str, Any]]:
    return {
        phase: {"status": "pending", "updated_at": None}
        for phase in CANONICAL_PHASES
        if phase != "completed"
    }
