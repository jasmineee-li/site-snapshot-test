"""Shared atomic file-write helpers."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from worldsim.failpoints import crash_if_enabled

# Probe the process umask exactly once at import time. ``os.umask`` is a
# process-wide mutation, so calling it inside a concurrent/async hot path (as
# Phase 2c's worker pool runs N tasks in parallel) is racy. Cache here so the
# fallback mode computation in ``write_json_atomic`` is race-free.
_PROCESS_UMASK = os.umask(0)
os.umask(_PROCESS_UMASK)
_DEFAULT_NEW_FILE_MODE = 0o666 & ~_PROCESS_UMASK


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    failpoint_base: str | None = None,
) -> None:
    """Atomically replace *path* with JSON *payload*.

    Preserves the destination file's permission bits across the swap. If the
    destination does not yet exist, the new file is created with the
    umask-respecting default (typically ``0o644`` under the standard server
    umask of ``0o022``). Without this preservation, ``tempfile.mkstemp``
    hardcodes ``0o600`` and every re-write would silently downgrade the
    file — breaking consumers that need the file world-readable (CI
    artifacts, dataset files committed as ``100644``).

    When ``failpoint_base`` is set, named crash hooks fire immediately
    before and after the final ``os.replace``:

    - ``<failpoint_base>.before_replace``
    - ``<failpoint_base>.after_replace``
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing_mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        existing_mode = None
    except OSError:
        # Unlikely — directory reachable for parent.mkdir but not for stat.
        # Fall back to the default; the write will still proceed.
        existing_mode = None
    target_mode = existing_mode if existing_mode is not None else _DEFAULT_NEW_FILE_MODE

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        try:
            os.chmod(tmp_path, target_mode)
        except OSError:
            # Exotic filesystems (e.g. tmpfs variants) may refuse chmod. The
            # write is still correct; we just accept the 0o600 leftover
            # rather than fail the whole operation.
            pass
        if failpoint_base:
            crash_if_enabled(f"{failpoint_base}.before_replace")
        os.replace(tmp_path, path)
        if failpoint_base:
            crash_if_enabled(f"{failpoint_base}.after_replace")
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
