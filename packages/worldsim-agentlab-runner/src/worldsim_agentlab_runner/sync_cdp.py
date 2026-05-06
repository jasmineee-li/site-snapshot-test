from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import queue
import signal
import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any


def sync_cdp_deadline(timeout_s: float, label: str):
    """Bound sync Playwright CDP calls while preserving an outer SIGALRM."""

    return _SyncCdpDeadline(timeout_s, label)


class _SyncCdpDeadline:
    def __init__(self, timeout_s: float, label: str) -> None:
        self.timeout_s = timeout_s
        self.label = label
        self.previous_handler: Any = None
        self.previous_timer: tuple[float, float] = (0.0, 0.0)
        self.started_at = 0.0

    def __enter__(self) -> None:
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        self.previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
        self.started_at = time.monotonic()

        def _raise_timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError(f"{self.label} exceeded timeout {self.timeout_s:g}s")

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_s)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        previous_remaining = float(self.previous_timer[0] or 0)
        if previous_remaining > 0:
            elapsed = time.monotonic() - self.started_at
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_remaining - elapsed),
                float(self.previous_timer[1] or 0),
            )


class CdpCallPump:
    def __init__(self, *, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.calls: queue.Queue[tuple[Any, str, dict[str, Any], concurrent.futures.Future[Any]]] = (
            queue.Queue()
        )

    async def send(self, session: Any, method: str, params: dict[str, Any]) -> dict[str, Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self.calls.put((session, method, params, future))
        result = await asyncio.wrap_future(future)
        return result if isinstance(result, dict) else {}

    def service_once(self, *, timeout: float = 0.01) -> None:
        try:
            session, method, params, future = self.calls.get(timeout=timeout)
        except queue.Empty:
            return
        if future.cancelled():
            return
        try:
            with sync_cdp_deadline(self.timeout_s, f"PVPO CDP {method}"):
                result = session.send(method, params)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)


class PumpedSyncCdpSession:
    def __init__(self, session: Any, pump: CdpCallPump) -> None:
        self.session = session
        self.pump = pump

    async def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.pump.send(self.session, method, params or {})


class SyncCdpWorker:
    """Persistent async worker for sync Playwright CDP calls.

    Playwright's sync API must be called from the owning thread. The worker keeps
    async PVPO state, especially the beginFrame coordinator, on one event loop
    while the caller thread services actual sync ``session.send`` calls.
    """

    def __init__(self, *, timeout_s: float, name: str) -> None:
        self.pump = CdpCallPump(timeout_s=timeout_s)
        self._ready = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run_loop, name=name, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        if self._loop is None:
            raise RuntimeError("sync CDP worker loop did not start")

    def run(self, build: Callable[[CdpCallPump], Coroutine[Any, Any, Any]]) -> Any:
        if self._closed or self._loop is None:
            raise RuntimeError("sync CDP worker is closed")
        future = asyncio.run_coroutine_threadsafe(build(self.pump), self._loop)
        while not future.done():
            self.pump.service_once()
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=1)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
