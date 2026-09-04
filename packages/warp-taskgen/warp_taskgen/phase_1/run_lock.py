"""Single-owner lock for CLI-orchestrated Phase 1 runs."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Phase1AlreadyRunning(RuntimeError):
    """Raised when another Phase 1 process owns the state root."""


@contextmanager
def phase_1_run_lock(state_dir: Path) -> Iterator[None]:
    """Exclusively lock one state root for the duration of a Phase 1 CLI run."""

    lock_path = state_dir / "phase_1" / ".phase1_run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Phase1AlreadyRunning(f"another Phase 1 run holds {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["Phase1AlreadyRunning", "phase_1_run_lock"]
