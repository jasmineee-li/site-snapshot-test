#!/usr/bin/env python3
"""Run one bounded Classifieds canary with local host lifecycle ownership.

The command validates the configured host, starts one sanitized Remote Job,
reports its status, and cleans up the exact host it owns. It always removes
the sweep marker and attempts to park that host when the run exits.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from warp_taskgen.classifieds_canary import (
    ClassifiedsCanaryConfig,
    build_remote_job_start_args,
    load_canary_config,
    validate_classifieds_run_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_TAG_KEY = "worldsim:sweep-in-progress"
SWEEP_TAG_VALUE = "true"
OWNER_TAG_KEY = "warp:classifieds-canary-owner"
REMOTE_JOB_NAME = "classifieds-listing-reply-canary"
DEFAULT_TIMEOUT_SECONDS = 3_600.0
DEFAULT_POLL_INTERVAL_SECONDS = 15.0
STATUS_COMMAND_TIMEOUT_SECONDS = 60.0
STOP_COMMAND_TIMEOUT_SECONDS = 330.0
CLEANUP_COMMAND_TIMEOUT_SECONDS = 300.0

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ConfigLoader = Callable[..., ClassifiedsCanaryConfig]


class CanaryInterrupted(Exception):
    """Raised by the signal handler so the lifecycle ``finally`` can run."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"Classifieds canary interrupted by signal {signum}")


class CanaryCleanupError(RuntimeError):
    """Raised when the run ended normally but lifecycle cleanup failed."""


@contextmanager
def _exclusive_operator_lock(instance_id: str):  # type: ignore[no-untyped-def]
    """Serialize canonical canary ownership on this operator workstation."""

    lock_path = Path(tempfile.gettempdir()) / f"warp-classifieds-{instance_id}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another Classifieds canary operator already owns this host"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def _completed(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    check: bool,
    timeout: float,
    start_new_session: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one local operator command with a stable, injectable seam."""

    kwargs: dict[str, Any] = {
        "cwd": REPO_ROOT,
        "capture_output": True,
        "text": True,
        "check": check,
        "timeout": timeout,
    }
    if start_new_session:
        kwargs["start_new_session"] = True
    return runner(list(argv), **kwargs)


def _job_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("job_id="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise RuntimeError("remote_job_start did not return a job_id")


def _status_payload(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("remote_job_status --json returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise RuntimeError("remote_job_status --json returned no status")
    return payload


def _require_lifecycle_identity(config: ClassifiedsCanaryConfig) -> None:
    if not config.host.region or not config.host.instance_id:
        raise ValueError("classifieds canary host config requires region and instance_id")


def _host_resume_command(config: ClassifiedsCanaryConfig) -> tuple[str, ...]:
    return (
        str(REPO_ROOT / "scripts" / "host_resume.sh"),
        "--host-config",
        str(config.host_config),
        "--no-tag",
    )


def _claim_host_tags_command(config: ClassifiedsCanaryConfig, owner_token: str) -> tuple[str, ...]:
    if not config.host.region or not config.host.instance_id:
        raise ValueError("classifieds canary host config requires region and instance_id")
    return (
        "aws",
        "ec2",
        "create-tags",
        "--region",
        config.host.region,
        "--resources",
        config.host.instance_id,
        "--tags",
        f"Key={SWEEP_TAG_KEY},Value={SWEEP_TAG_VALUE}",
        f"Key={OWNER_TAG_KEY},Value={owner_token}",
    )


def _host_availability_command(config: ClassifiedsCanaryConfig) -> tuple[str, ...]:
    if not config.host.region or not config.host.instance_id:
        raise ValueError("classifieds canary host config requires region and instance_id")
    return (
        "aws",
        "ec2",
        "describe-instances",
        "--region",
        config.host.region,
        "--instance-ids",
        config.host.instance_id,
        "--query",
        (
            "Reservations[0].Instances[0]."
            "{state:State.Name,"
            "sweep:Tags[?Key=='worldsim:sweep-in-progress'].Value|[0],"
            "owner:Tags[?Key=='warp:classifieds-canary-owner'].Value|[0]}"
        ),
        "--output",
        "json",
    )


def _require_exclusive_stopped_host(
    config: ClassifiedsCanaryConfig, *, runner: CommandRunner, timeout: float
) -> None:
    result = _completed(
        runner,
        _host_availability_command(config),
        check=True,
        timeout=timeout,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("host availability query returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("state") != "stopped":
        raise RuntimeError("Classifieds canary requires the exclusive stopped host")
    if payload.get("sweep") not in (None, "") or payload.get("owner") not in (None, ""):
        raise RuntimeError("Classifieds canary host already has a lifecycle owner")


def _claim_exclusive_host(
    config: ClassifiedsCanaryConfig,
    owner_token: str,
    *,
    runner: CommandRunner,
    timeout: float,
) -> None:
    _completed(
        runner,
        _claim_host_tags_command(config, owner_token),
        check=True,
        timeout=timeout,
    )
    _require_host_owner(
        config,
        owner_token,
        runner=runner,
        timeout=timeout,
        require_stopped=True,
    )


def _require_host_owner(
    config: ClassifiedsCanaryConfig,
    owner_token: str,
    *,
    runner: CommandRunner,
    timeout: float,
    require_stopped: bool = False,
) -> None:
    result = _completed(
        runner,
        _host_availability_command(config),
        check=True,
        timeout=timeout,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("host ownership query returned invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("sweep") != SWEEP_TAG_VALUE
        or payload.get("owner") != owner_token
    ):
        raise RuntimeError("Classifieds canary lost its exclusive host ownership")
    if require_stopped and payload.get("state") != "stopped":
        raise RuntimeError("Classifieds canary host changed state during ownership claim")


def _remote_status_command(config: ClassifiedsCanaryConfig, job_id: str) -> tuple[str, ...]:
    return (
        str(REPO_ROOT / "scripts" / "remote_job_status.sh"),
        "--host-config",
        str(config.host_config),
        "--remote-dir",
        config.remote_dir,
        "--job-id",
        job_id,
        "--json",
    )


def _remote_list_command(config: ClassifiedsCanaryConfig) -> tuple[str, ...]:
    return (
        str(REPO_ROOT / "scripts" / "remote_job_list.sh"),
        "--host-config",
        str(config.host_config),
        "--remote-dir",
        config.remote_dir,
        "--name",
        REMOTE_JOB_NAME,
        "--limit",
        "10",
        "--json",
    )


def _recover_started_job_id(
    config: ClassifiedsCanaryConfig,
    *,
    run_dir: str,
    runner: CommandRunner,
) -> str | None:
    """Recover one detached job after ambiguous local start failure."""

    for _attempt in range(3):
        try:
            result = _completed(
                runner,
                _remote_list_command(config),
                check=False,
                timeout=STATUS_COMMAND_TIMEOUT_SECONDS,
            )
            payload = json.loads(result.stdout)
        except Exception:
            continue
        rows = payload.get("jobs") if isinstance(payload, dict) else None
        matches = [
            row
            for row in rows or []
            if isinstance(row, dict)
            and row.get("name") == REMOTE_JOB_NAME
            and row.get("state_dir") == run_dir
            and isinstance(row.get("job_id"), str)
            and row["job_id"]
        ]
        if len(matches) == 1:
            return str(matches[0]["job_id"])
    return None


def _remote_stop_command(
    config: ClassifiedsCanaryConfig, job_id: str, *, graceful: bool
) -> tuple[str, ...]:
    mode = ("--graceful", "--pause-timeout", "300") if graceful else ("--force",)
    return (
        str(REPO_ROOT / "scripts" / "remote_job_stop.sh"),
        "--host-config",
        str(config.host_config),
        "--remote-dir",
        config.remote_dir,
        "--job-id",
        job_id,
        *mode,
    )


def _clear_sweep_tag_command(config: ClassifiedsCanaryConfig) -> tuple[str, ...]:
    if not config.host.region or not config.host.instance_id:
        raise ValueError("classifieds canary host config requires region and instance_id")
    return (
        "aws",
        "ec2",
        "delete-tags",
        "--region",
        config.host.region,
        "--resources",
        config.host.instance_id,
        "--tags",
        f"Key={SWEEP_TAG_KEY},Value={SWEEP_TAG_VALUE}",
    )


def _clear_owner_tag_command(config: ClassifiedsCanaryConfig, owner_token: str) -> tuple[str, ...]:
    if not config.host.region or not config.host.instance_id:
        raise ValueError("classifieds canary host config requires region and instance_id")
    return (
        "aws",
        "ec2",
        "delete-tags",
        "--region",
        config.host.region,
        "--resources",
        config.host.instance_id,
        "--tags",
        f"Key={OWNER_TAG_KEY},Value={owner_token}",
    )


def _host_park_command(config: ClassifiedsCanaryConfig) -> tuple[str, ...]:
    return (
        str(REPO_ROOT / "scripts" / "host_park.sh"),
        "--host-config",
        str(config.host_config),
    )


def _cleanup_host(
    config: ClassifiedsCanaryConfig,
    *,
    owner_token: str,
    runner: CommandRunner,
) -> list[str]:
    """Park the exact owned host, retaining owner evidence until parking succeeds."""

    errors: list[str] = []
    previous_handlers: dict[int, Any] = {}
    # Do not let a second operator signal interrupt the cleanup sequence after
    # the run's signal handler has already entered its ``finally`` block.
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, signal.SIG_IGN)
    try:
        try:
            _require_host_owner(
                config,
                owner_token,
                runner=runner,
                timeout=STATUS_COMMAND_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            return [f"verify sweep owner: {exc}; host was not parked"]
        try:
            result = _completed(
                runner,
                _clear_sweep_tag_command(config),
                check=False,
                timeout=CLEANUP_COMMAND_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            return [f"clear sweep tag: {exc}; host was not parked"]
        if result.returncode != 0:
            return [f"clear sweep tag exited with {result.returncode}; host was not parked"]

        # The canonical sweep guard is now clear so host_park may run, while
        # the distinct owner tag continues to fence this exact operator.
        try:
            owner = _completed(
                runner,
                _host_availability_command(config),
                check=True,
                timeout=STATUS_COMMAND_TIMEOUT_SECONDS,
            )
            payload = json.loads(owner.stdout)
            if not isinstance(payload, dict) or payload.get("owner") != owner_token:
                raise RuntimeError("host owner changed before parking")
        except BaseException as exc:
            return [f"verify owner before parking: {exc}; host was not parked"]

        for label, command in (
            ("park host", _host_park_command(config)),
            ("clear owner tag", _clear_owner_tag_command(config, owner_token)),
        ):
            try:
                result = _completed(
                    runner,
                    command,
                    check=False,
                    timeout=CLEANUP_COMMAND_TIMEOUT_SECONDS,
                )
            except BaseException as exc:  # cleanup must still attempt the next step
                errors.append(f"{label}: {exc}")
                if label == "park host":
                    break
                continue
            if result.returncode != 0:
                errors.append(f"{label} exited with {result.returncode}")
                if label == "park host":
                    break
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return errors


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        raise CanaryInterrupted(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _stop_job_if_needed(
    config: ClassifiedsCanaryConfig,
    job_id: str | None,
    *,
    runner: CommandRunner,
) -> None:
    if job_id is None:
        return
    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, signal.SIG_IGN)
    try:
        try:
            result = _completed(
                runner,
                _remote_stop_command(config, job_id, graceful=True),
                check=False,
                timeout=STOP_COMMAND_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            print(
                f"warning: graceful stop for remote job {job_id} failed: {exc}",
                file=sys.stderr,
            )
            result = None
        if result is None or result.returncode != 0:
            detail = "raised" if result is None else f"exited with {result.returncode}"
            print(
                f"warning: graceful stop for {job_id} {detail}; "
                "using explicit abrupt fallback before host parking",
                file=sys.stderr,
            )
            try:
                forced = _completed(
                    runner,
                    _remote_stop_command(config, job_id, graceful=False),
                    check=False,
                    timeout=STOP_COMMAND_TIMEOUT_SECONDS,
                )
            except BaseException as exc:
                print(f"warning: abrupt fallback for {job_id} failed: {exc}", file=sys.stderr)
            else:
                if forced.returncode != 0:
                    print(
                        f"warning: abrupt fallback for {job_id} exited with {forced.returncode}",
                        file=sys.stderr,
                    )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_canary(
    config_path: str | Path,
    *,
    run_dir: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    runner: CommandRunner = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    config_loader: ConfigLoader | None = None,
) -> int:
    """Resume, launch, and observe one canary; return zero on success."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not math.isfinite(poll_interval_seconds) or poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be finite and non-negative")

    loader = load_canary_config if config_loader is None else config_loader
    config = loader(config_path, repo_root=REPO_ROOT)
    _require_lifecycle_identity(config)
    selected_run_dir = run_dir or f"{config.run_root}/{uuid.uuid4().hex}"
    selected_run_dir = validate_classifieds_run_dir(selected_run_dir)
    assert config.host.instance_id is not None
    with _exclusive_operator_lock(config.host.instance_id):
        return _run_canary_owned(
            config,
            selected_run_dir=selected_run_dir,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            runner=runner,
            sleeper=sleeper,
            clock=clock,
        )


def _run_canary_owned(
    config: ClassifiedsCanaryConfig,
    *,
    selected_run_dir: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    runner: CommandRunner,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
) -> int:
    """Run after the local operator has atomically acquired its host lock."""

    owner_token = f"classifieds-{uuid.uuid4().hex}"
    job_id: str | None = None
    owns_host = False
    previous_handlers = _install_signal_handlers()
    primary_error: BaseException | None = None
    deadline = clock() + timeout_seconds

    def remaining(stage: str, *, cap: float | None = None) -> float:
        value = deadline - clock()
        if value <= 0:
            raise TimeoutError(f"Classifieds canary timed out during {stage}")
        return min(value, cap) if cap is not None else value

    try:
        _require_exclusive_stopped_host(
            config,
            runner=runner,
            timeout=remaining("host availability", cap=STATUS_COMMAND_TIMEOUT_SECONDS),
        )
        # From this point onward a timed-out/interrupted create-tags request
        # may already have applied our exact token. Cleanup re-verifies that
        # token before parking, so containment is both safe and required.
        owns_host = True
        _claim_exclusive_host(
            config,
            owner_token,
            runner=runner,
            timeout=remaining("host ownership claim", cap=STATUS_COMMAND_TIMEOUT_SECONDS),
        )
        _completed(
            runner,
            _host_resume_command(config),
            check=True,
            timeout=remaining("host resume"),
        )
        # Treat detached-job registration as an atomic acquisition. The SSH
        # helper runs in a separate local session so terminal Ctrl-C cannot
        # kill it after remote detach. The wrapper records an operator signal,
        # obtains the exact job ID, then immediately stops that job.
        start_handlers: dict[int, Any] = {}
        pending_signals: list[int] = []

        def record_start_signal(signum: int, _frame: Any) -> None:
            pending_signals.append(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            start_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, record_start_signal)
        try:
            try:
                started = _completed(
                    runner,
                    build_remote_job_start_args(config, run_dir=selected_run_dir),
                    check=True,
                    timeout=remaining("remote job start"),
                    start_new_session=True,
                )
                job_id = _job_id(started.stdout)
            except BaseException:
                job_id = _recover_started_job_id(
                    config,
                    run_dir=selected_run_dir,
                    runner=runner,
                )
                if job_id is None:
                    print(
                        "warning: remote start identity could not be recovered; "
                        "parking the exact owned host as the containment fallback",
                        file=sys.stderr,
                    )
                raise
        finally:
            for signum, handler in start_handlers.items():
                signal.signal(signum, handler)
        if pending_signals:
            raise CanaryInterrupted(pending_signals[0])
        _require_host_owner(
            config,
            owner_token,
            runner=runner,
            timeout=remaining("post-resume host ownership", cap=STATUS_COMMAND_TIMEOUT_SECONDS),
        )
        status_command = _remote_status_command(config, job_id)
        while True:
            _require_host_owner(
                config,
                owner_token,
                runner=runner,
                timeout=remaining("poll host ownership", cap=STATUS_COMMAND_TIMEOUT_SECONDS),
            )
            status_result = _completed(
                runner,
                status_command,
                check=False,
                timeout=remaining("remote job status", cap=STATUS_COMMAND_TIMEOUT_SECONDS),
            )
            if status_result.returncode != 0:
                raise RuntimeError(f"remote_job_status exited with {status_result.returncode}")
            payload = _status_payload(status_result.stdout)
            print(
                "[classifieds-canary] "
                f"job={job_id} status={payload['status']} "
                f"run_status={payload.get('run_status', 'unknown')} "
                f"run_step={payload.get('run_step', 'unknown')}",
                flush=True,
            )
            if payload["status"] != "running":
                returncode = payload.get("returncode")
                if returncode == 0:
                    return 0
                raise RuntimeError(
                    f"Classifieds canary remote job ended with status={payload['status']} "
                    f"returncode={returncode}"
                )
            if clock() >= deadline:
                raise TimeoutError(f"Classifieds canary timed out after {timeout_seconds:g}s")
            sleeper(min(poll_interval_seconds, remaining("poll sleep")))
    except BaseException as exc:
        primary_error = exc
        _stop_job_if_needed(config, job_id, runner=runner)
        raise
    finally:
        cleanup_errors = (
            _cleanup_host(config, owner_token=owner_token, runner=runner) if owns_host else []
        )
        _restore_signal_handlers(previous_handlers)
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            if primary_error is None:
                raise CanaryCleanupError(message)
            print(f"warning: canary cleanup incomplete: {message}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded Classifieds canary on the configured sandbox host.",
        epilog=(
            "Inputs: the ignored operator host configuration and, optionally, a fresh Run root. "
            "Action: claim the stopped host, start one sanitized Remote Job, and poll it. "
            "Output: the Run artifacts, including completion.json, under the Run root. "
            "Safety: this is the only listed command that changes host/container state; it "
            "uses the exclusive lifecycle guard and always attempts job cleanup and host parking."
        ),
    )
    parser.add_argument(
        "--host-config",
        required=True,
        type=Path,
        help="Ignored operator host YAML for the selected sandbox host.",
    )
    parser.add_argument(
        "--run-dir",
        help="Fresh Classifieds Run root; defaults to a generated path under the configured run root.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Whole-canary timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Remote status polling interval in seconds (default: {DEFAULT_POLL_INTERVAL_SECONDS:g}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_canary(
            args.host_config,
            run_dir=args.run_dir,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except CanaryInterrupted as exc:
        return 128 + exc.signum
    except KeyboardInterrupt:
        return 130
    except (CanaryCleanupError, TimeoutError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
