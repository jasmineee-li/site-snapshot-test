#!/usr/bin/env python3
"""Capture a detached remote-job runner's process group without shell polling."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_POLL_SECONDS = 0.01


def _linux_process_state(pid: int) -> str | None:
    """Return Linux's process state, or ``None`` when procfs is unavailable/gone.

    ``os.getpgid`` can still succeed for a zombie.  The detached runner is
    briefly a zombie when a fast child exits, so treating procfs's ``Z`` state
    as gone avoids waiting for the shell to reap it.
    """

    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return None

    # The command name is parenthesized and may contain spaces or ``)``.  The
    # state is the first field after the final closing parenthesis.
    _, separator, fields = raw.rpartition(")")
    if not separator:
        return None
    state_fields = fields.strip().split()
    return state_fields[0] if state_fields else None


def capture_process_group(
    pid: int,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> int | None:
    """Return the runner's own process group, or ``None`` once it is gone.

    ``remote_job_start.sh`` starts ``run_job.py`` in the background and the
    runner calls ``setsid`` before launching its child.  The short readiness
    window therefore needs to wait for ``pgid == pid`` for live runners, but it
    must stop immediately when a fast runner has already exited.  The previous
    shell loop launched twenty Python interpreters and could wait the full two
    seconds on a zombie; this single process handles both cases.
    """

    if pid <= 0:
        raise ValueError("pid must be positive")
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    deadline = time.monotonic() + timeout_seconds
    while True:
        # On Linux, check this before getpgid: a zombie still has a process
        # group, but it is no longer a live job we can safely stop.
        if _linux_process_state(pid) == "Z":
            return None

        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return None

        if pgid == pid:
            return pgid
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(poll_seconds, remaining))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        pgid = capture_process_group(
            args.pid,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if pgid is not None:
        print(pgid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
