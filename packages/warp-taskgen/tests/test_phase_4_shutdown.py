from __future__ import annotations

import asyncio
import os
import time

from worldsim import browser_use_agent
from worldsim.main import (
    _phase4_async_shutdown_timeout,
    _run_phase4_with_bounded_async_shutdown,
)


def test_phase4_bounded_shutdown_returns_when_background_task_ignores_cancel():
    created_tasks: list[asyncio.Task[None]] = []

    async def phase4_like_run() -> int:
        async def stubborn_background_task() -> None:
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                while True:
                    await asyncio.sleep(3600)

        created_tasks.append(
            asyncio.create_task(stubborn_background_task(), name="browser-use-watchdog")
        )
        return 7

    started = time.monotonic()

    rc = _run_phase4_with_bounded_async_shutdown(
        phase4_like_run(),
        shutdown_timeout_s=0.01,
    )

    assert rc == 7
    assert len(created_tasks) == 1
    assert time.monotonic() - started < 1.0


def test_phase4_shutdown_timeout_env_defaults_for_invalid_values(monkeypatch):
    monkeypatch.delenv("WORLDSIM_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_S", raising=False)
    assert _phase4_async_shutdown_timeout() == 10.0

    monkeypatch.setenv("WORLDSIM_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_S", "0")
    assert _phase4_async_shutdown_timeout() == 10.0

    monkeypatch.setenv("WORLDSIM_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_S", "0.25")
    assert _phase4_async_shutdown_timeout() == 0.25


def test_browser_use_runtime_env_defaults_disable_cloud_telemetry(monkeypatch):
    for key in ("ANONYMIZED_TELEMETRY", "BROWSER_USE_CLOUD_SYNC", "POSTHOG_DISABLED"):
        monkeypatch.delenv(key, raising=False)

    browser_use_agent._ensure_browser_use_runtime_env()

    assert os.environ["ANONYMIZED_TELEMETRY"] == "false"
    assert os.environ["BROWSER_USE_CLOUD_SYNC"] == "false"
    assert os.environ["POSTHOG_DISABLED"] == "true"


def test_browser_use_runtime_env_preserves_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "true")
    monkeypatch.setenv("BROWSER_USE_CLOUD_SYNC", "true")
    monkeypatch.setenv("POSTHOG_DISABLED", "false")

    browser_use_agent._ensure_browser_use_runtime_env()

    assert os.environ["ANONYMIZED_TELEMETRY"] == "true"
    assert os.environ["BROWSER_USE_CLOUD_SYNC"] == "true"
    assert os.environ["POSTHOG_DISABLED"] == "false"
