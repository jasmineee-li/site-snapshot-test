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

STATE_DIR = Path(os.environ.get("WORLDSIM_STATE_DIR", "logs"))
STATE_FILE = STATE_DIR / "pipeline_state.json"


def save_state(step: str, iteration: int = 0, **metadata: Any) -> None:
    """Write a checkpoint before each major operation.

    Args:
        step: Identifier for the current step (e.g. "phase_0a", "phase_3_task_5").
        iteration: Loop iteration counter for retry logic. 0 means first attempt.
        **metadata: Arbitrary extra fields merged into the state blob.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }
    # Atomic write: write to a temp file in the same directory, then rename.
    # os.replace is atomic on POSIX when src and dst are on the same filesystem.
    fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_FILE)
    except BaseException:
        # Clean up the temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state() -> dict[str, Any] | None:
    """Return the last saved state, or ``None`` if no state file exists."""
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        logger.warning("Corrupt pipeline state file at %s — ignoring", STATE_FILE)
        return None
