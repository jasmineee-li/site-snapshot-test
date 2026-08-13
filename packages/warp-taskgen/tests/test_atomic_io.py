"""Unit tests for ``warp_taskgen.atomic_io``.

Focus: the permission-preservation behavior of ``write_json_atomic``.
``tempfile.mkstemp`` hardcodes ``0o600`` on new files; without the
preservation logic, every atomic rewrite would silently downgrade the
destination's mode from ``0o644`` (the typical git-tracked mode) to
``0o600``, breaking CI/downstream tools that expect world-readable
artifacts.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from warp_taskgen import atomic_io


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_new_file_gets_umask_respecting_mode(tmp_path: Path) -> None:
    target = tmp_path / "fresh.json"
    atomic_io.write_json_atomic(target, {"hello": "world"})

    # Under the standard server umask (0o022), the default mode is 0o644.
    # The test asserts against the module's computed default so the test
    # works under any umask the CI runner happens to carry.
    assert _mode(target) == atomic_io._DEFAULT_NEW_FILE_MODE
    assert json.loads(target.read_text()) == {"hello": "world"}


def test_overwrite_preserves_existing_mode_0o644(tmp_path: Path) -> None:
    target = tmp_path / "dataset.json"
    target.write_text("[]")
    os.chmod(target, 0o644)

    atomic_io.write_json_atomic(target, [{"id": "AT-001"}])

    assert _mode(target) == 0o644
    assert json.loads(target.read_text()) == [{"id": "AT-001"}]


def test_overwrite_preserves_existing_mode_0o600(tmp_path: Path) -> None:
    """Operators who intentionally tightened permissions should keep them."""
    target = tmp_path / "secrets.json"
    target.write_text("{}")
    os.chmod(target, 0o600)

    atomic_io.write_json_atomic(target, {"token": "redacted"})

    assert _mode(target) == 0o600


def test_overwrite_preserves_group_readable_mode(tmp_path: Path) -> None:
    target = tmp_path / "shared.json"
    target.write_text("{}")
    os.chmod(target, 0o640)

    atomic_io.write_json_atomic(target, {"k": "v"})

    assert _mode(target) == 0o640


def test_no_tmp_file_leftover_on_success(tmp_path: Path) -> None:
    target = tmp_path / "ok.json"
    atomic_io.write_json_atomic(target, [1, 2, 3])

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_no_tmp_file_leftover_on_exception(monkeypatch, tmp_path: Path) -> None:
    """A BaseException raised between the chmod and the replace (e.g. via a
    failpoint crash, a KeyboardInterrupt, or a chmod OSError we don't
    swallow) must not leak the tempfile."""
    target = tmp_path / "crashy.json"
    target.write_text("[]")
    os.chmod(target, 0o644)

    def boom(_name: str) -> None:
        raise RuntimeError("synthetic crash before replace")

    monkeypatch.setattr(atomic_io, "crash_if_enabled", boom)

    with pytest.raises(RuntimeError, match="synthetic crash"):
        atomic_io.write_json_atomic(
            target,
            {"before": "replace"},
            failpoint_base="phase_2.test.crashy",
        )

    # Original file untouched, no .tmp leftovers.
    assert target.read_text() == "[]"
    assert _mode(target) == 0o644
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_default_mode_computation_respects_umask(monkeypatch, tmp_path: Path) -> None:
    """The module-level umask cache resolves at import time; this just
    documents the invariant so a reviewer understands *why* the default
    mode is what it is."""
    # Standard server umasks produce readable defaults.
    assert atomic_io._DEFAULT_NEW_FILE_MODE & 0o400, "owner must be readable"
    # Must not be more permissive than 0o666.
    assert atomic_io._DEFAULT_NEW_FILE_MODE & ~0o666 == 0
