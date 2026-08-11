"""Single-owner lock for CLI-orchestrated Phase 2 runs."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Phase2AlreadyRunning(RuntimeError):
    """Raised when another Phase 2 process owns the state root."""


@contextmanager
def phase_2_run_lock(state_dir: Path) -> Iterator[None]:
    lock_path = state_dir / "phase_2" / ".phase2_run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Phase2AlreadyRunning(f"another Phase 2 run holds {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["Phase2AlreadyRunning", "phase_2_run_lock"]
