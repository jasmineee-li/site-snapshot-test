"""CLI adapter for cooperative Run lifecycle control."""

from __future__ import annotations

import contextlib
import signal
import sys
import threading
from argparse import Namespace
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path

from worldsim.run_control import (
    PauseBoundaryReached,
    RunInterrupted,
    acknowledge_pause,
    mark_interrupted,
    request_pause,
)


def dispatch_pause(args: Namespace) -> int:
    try:
        request = request_pause(getattr(args, "state_dir", None))
    except (OSError, ValueError) as exc:
        print(f"pause failed: {exc}", file=sys.stderr)
        return 2
    print(f"Pause requested for {request.step} at the next safe checkpoint ({request.request_id}).")
    return 0


def dispatch_phase_with_run_control(
    *,
    phase: str,
    state_dir: Path,
    operation: Callable[[], int],
    lifecycle_guard: Callable[[], AbstractContextManager[None]],
) -> int:
    """Run a phase and persist cooperative lifecycle after stack unwind."""

    with lifecycle_guard():
        try:
            with _phase_interrupt_signals(enabled=phase == "4"):
                return operation()
        except PauseBoundaryReached:
            if phase not in {"2", "2c", "4"}:
                raise
            try:
                with _ignore_lifecycle_transition_signals(enabled=True):
                    acknowledge_pause(state_dir)
            except (OSError, ValueError) as exc:
                print(f"Pause acknowledgement failed: {exc}", file=sys.stderr)
                return 2
            print(f"Phase {phase} paused after admitted work reached a safe checkpoint.")
            return 0
        except RunInterrupted as exc:
            try:
                with _ignore_lifecycle_transition_signals(enabled=True):
                    mark_interrupted(state_dir, signal_name=exc.signal_name)
            except (OSError, ValueError) as state_exc:
                print(f"Could not persist interrupted state: {state_exc}", file=sys.stderr)
            return 128 + int(getattr(signal, exc.signal_name, signal.SIGTERM))
        except KeyboardInterrupt:
            if phase != "4":
                raise
            try:
                with _ignore_lifecycle_transition_signals(enabled=True):
                    mark_interrupted(state_dir, signal_name="SIGINT")
            except (OSError, ValueError) as exc:
                print(f"Could not persist interrupted state: {exc}", file=sys.stderr)
            return 130


@contextlib.contextmanager
def _phase_interrupt_signals(*, enabled: bool) -> Iterator[None]:
    """Convert catchable termination into a phase-boundary interruption."""

    if not enabled or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(_signum: int, _frame: object) -> None:
        raise RunInterrupted("SIGTERM")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextlib.contextmanager
def _ignore_lifecycle_transition_signals(*, enabled: bool) -> Iterator[None]:
    """Keep the short atomic lifecycle write from being interrupted midway."""

    if not enabled or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


__all__ = ["dispatch_pause", "dispatch_phase_with_run_control"]
