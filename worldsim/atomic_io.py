"""Shared atomic file-write helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from worldsim.failpoints import crash_if_enabled


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    failpoint_base: str | None = None,
) -> None:
    """Atomically replace *path* with JSON *payload*.

    When ``failpoint_base`` is set, named crash hooks fire immediately
    before and after the final ``os.replace``:

    - ``<failpoint_base>.before_replace``
    - ``<failpoint_base>.after_replace``
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
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

