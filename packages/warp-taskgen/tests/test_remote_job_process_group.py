from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "remote_job_process_group.py"
_SPEC = importlib.util.spec_from_file_location("remote_job_process_group", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_capture_process_group_returns_none_when_process_is_gone(monkeypatch) -> None:
    monkeypatch.setattr(_MODULE, "_linux_process_state", lambda pid: None)

    assert _MODULE.capture_process_group(12345) is None


def test_capture_process_group_returns_runner_group_without_waiting(monkeypatch) -> None:
    monkeypatch.setattr(_MODULE, "_linux_process_state", lambda pid: "S")
    monkeypatch.setattr(_MODULE.os, "getpgid", lambda pid: pid)

    assert _MODULE.capture_process_group(12345) == 12345


def test_capture_process_group_waits_for_session_group(monkeypatch) -> None:
    groups = iter((999, 999, 12345))
    sleeps: list[float] = []
    monotonic_values = iter((0.0, 0.01, 0.02, 0.03))

    monkeypatch.setattr(_MODULE, "_linux_process_state", lambda pid: "S")
    monkeypatch.setattr(_MODULE.os, "getpgid", lambda pid: next(groups))
    monkeypatch.setattr(_MODULE.time, "sleep", sleeps.append)
    monkeypatch.setattr(_MODULE.time, "monotonic", lambda: next(monotonic_values))

    assert _MODULE.capture_process_group(12345, timeout_seconds=1.0, poll_seconds=0.01) == 12345
    assert sleeps == [0.01, 0.01]


def test_capture_process_group_treats_zombie_as_gone(monkeypatch) -> None:
    monkeypatch.setattr(_MODULE, "_linux_process_state", lambda pid: "Z")
    getpgid_called = False

    def fail_getpgid(pid: int) -> int:
        nonlocal getpgid_called
        getpgid_called = True
        raise AssertionError("zombie state should short-circuit before getpgid")

    monkeypatch.setattr(_MODULE.os, "getpgid", fail_getpgid)

    assert _MODULE.capture_process_group(12345) is None
    assert not getpgid_called
