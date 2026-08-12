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
from worldsim.run_control_wait import current_pause_stage, wait_for_pause


def dispatch_pause(args: Namespace) -> int:
    try:
        request = request_pause(getattr(args, "state_dir", None))
    except (OSError, ValueError) as exc:
        print(f"pause failed: {exc}", file=sys.stderr)
        return 2
    print(f"Pause requested for {request.step} at the next safe checkpoint ({request.request_id}).")
    if not getattr(args, "wait", False):
        return 0
    progress_stage = current_pause_stage(getattr(args, "state_dir", None)) or request.step
    print(f"Pause progress: pausing request={request.request_id} stage={progress_stage}.")
    try:
        result = wait_for_pause(
            getattr(args, "state_dir", None),
            request.request_id,
            timeout=float(getattr(args, "timeout", 300.0)),
            poll_interval=float(getattr(args, "poll_interval", 0.25)),
            expected_request=request,
        )
    except (OSError, ValueError) as exc:
        print(f"pause wait failed: {exc}", file=sys.stderr)
        return 2
    if result.status == "paused":
        print(f"Pause acknowledged ({request.request_id}).")
        return 0
    if result.status == "terminal":
        print(f"Pause ended because the pipeline is terminal ({result.state_status or 'unknown'}).")
        return 0
    if result.status == "timed_out":
        print(
            f"Pause still pending after {result.elapsed_seconds:.1f}s "
            f"({request.request_id}; reason={result.reason_code}).",
            file=sys.stderr,
        )
        return 1
    print(
        f"Pause wait {result.status}: {result.reason_code} ({request.request_id}).",
        file=sys.stderr,
    )
    return 2


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
            # Handled signals begin only after the caller's lifecycle guard
            # owns the Phase 2/4 run lock. Pre-lock termination remains the
            # existing crash-compatible running state.
            with _phase_interrupt_signals(enabled=phase in {"2", "2c", "4"}):
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
            if phase not in {"2", "2c", "4"}:
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
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigterm(_signum: int, _frame: object) -> None:
        raise RunInterrupted("SIGTERM")

    def _handle_sigint(_signum: int, _frame: object) -> None:
        raise RunInterrupted("SIGINT")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


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
