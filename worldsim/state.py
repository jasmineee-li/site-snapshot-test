"""Pipeline state persistence.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "State Persistence and Resume".

The orchestrator writes a single JSON state file before each major operation.
On crash, ``worldsim --resume`` reads this file and skips completed phases.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_DEFAULT_STATE_DIR = Path("logs")
_RESUME_POINTER = _DEFAULT_STATE_DIR / "last_run_state.json"


def get_state_dir() -> Path:
    """Return the current pipeline state directory."""
    return Path(os.environ.get("WORLDSIM_STATE_DIR", _DEFAULT_STATE_DIR))


def get_state_file() -> Path:
    """Return the current pipeline state file path."""
    return get_state_dir() / "pipeline_state.json"


# Backwards-compatible aliases. Internal code should prefer ``get_state_dir()``
# so .env-loaded overrides are honored even if this module was imported early.
STATE_DIR = get_state_dir()
STATE_FILE = get_state_file()


def save_state(step: str, iteration: int = 0, **metadata: Any) -> None:
    """Write a checkpoint before each major operation.

    Args:
        step: Identifier for the current step (e.g. "phase_0a", "phase_3_task_5").
        iteration: Loop iteration counter for retry logic. 0 means first attempt.
        **metadata: Arbitrary extra fields merged into the state blob.
    """
    state_dir = get_state_dir()
    state_file = get_state_file()
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "logs_dir": str(state_dir),
        **metadata,
    }
    # Atomic write: write to a temp file in the same directory, then rename.
    # os.replace is atomic on POSIX when src and dst are on the same filesystem.
    fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, state_file)
        _write_resume_pointer(state_dir)
    except BaseException:
        # Clean up the temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state() -> dict[str, Any] | None:
    """Return the last saved state, or ``None`` if no state file exists."""
    state_file = _resolve_state_file()
    if not state_file.exists():
        return None
    try:
        payload = json.loads(state_file.read_text())
    except json.JSONDecodeError:
        logger.warning("Corrupt pipeline state file at %s — ignoring", state_file)
        return None
    if not isinstance(payload, dict):
        logger.warning("Invalid pipeline state file at %s — expected a JSON object", state_file)
        return None
    return payload


def _resolve_state_file() -> Path:
    """Resolve the active state file, following any resume pointer."""
    if os.environ.get("WORLDSIM_STATE_DIR"):
        return get_state_file()

    pointer = _load_resume_pointer()
    if pointer is not None:
        return pointer
    return get_state_file()


def _write_resume_pointer(state_dir: Path) -> None:
    """Persist the latest state directory at the default logs location."""
    _DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "logs_dir": str(state_dir),
        "state_file": str(state_dir / "pipeline_state.json"),
        "timestamp": datetime.now().isoformat(),
    }
    fd, tmp_path = tempfile.mkstemp(dir=_DEFAULT_STATE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, _RESUME_POINTER)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_resume_pointer() -> Path | None:
    """Return the most recently saved state file path from the pointer."""
    if not _RESUME_POINTER.exists():
        return None
    try:
        payload = json.loads(_RESUME_POINTER.read_text())
    except json.JSONDecodeError:
        logger.warning("Corrupt resume pointer at %s — ignoring", _RESUME_POINTER)
        return None

    state_file = payload.get("state_file")
    logs_dir = payload.get("logs_dir")
    if state_file:
        candidate = Path(state_file)
    elif logs_dir:
        candidate = Path(logs_dir) / "pipeline_state.json"
    else:
        return None

    if not candidate.exists():
        logger.warning("Resume pointer referenced missing state file at %s", candidate)
        return None
    return candidate
