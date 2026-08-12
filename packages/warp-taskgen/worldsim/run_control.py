"""Cooperative pause requests and lifecycle acknowledgements."""

from __future__ import annotations

import asyncio
import fcntl
import json
import re
import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from worldsim.atomic_io import write_json_atomic
from worldsim.run_definition import define_run
from worldsim.state import get_state_dir, transition_pipeline_status

_SCHEMA_VERSION = 1
_REQUEST_FILE = "pause_request.json"
_REQUEST_LOCK_FILE = ".pause_request.lock"
_SUPPORTED_PAUSE_STAGES = {
    "phase_2": frozenset({"planning", "text_fill"}),
    "phase_4": frozenset({"initial_evaluation", "postprocessing"}),
}
_PROCESS_POOL_PAUSE_STAGE = "process_pool_dispatch"


class PauseBoundaryReached(BaseException):
    """Raised after admitted atomic work drains at a cooperative boundary."""


class RunInterrupted(BaseException):
    """Raised by a handled process signal at the outer phase boundary."""

    def __init__(self, signal_name: str) -> None:
        super().__init__(signal_name)
        self.signal_name = signal_name


@dataclass(frozen=True)
class PauseRequest:
    request_id: str
    requested_at: str
    run_id: str | None
    definition_digest: str | None
    step: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "request_id": self.request_id,
            "requested_at": self.requested_at,
            "run_id": self.run_id,
            "definition_digest": self.definition_digest,
            "step": self.step,
        }


def pause_request_path(state_dir: Path | None = None) -> Path:
    return (state_dir or get_state_dir()).expanduser().resolve(strict=False) / _REQUEST_FILE


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"pause control at {path} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"pause control at {path} must contain an object")
    return payload


def _parse_request(payload: dict[str, object]) -> PauseRequest:
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != _SCHEMA_VERSION:
        raise ValueError("pause request has an unsupported schema")
    request_id = payload.get("request_id")
    requested_at = payload.get("requested_at")
    step = payload.get("step")
    if not all(
        isinstance(value, str) and value.strip() for value in (request_id, requested_at, step)
    ):
        raise ValueError("pause request identity is malformed")
    if re.fullmatch(r"pause-[0-9a-f]{32}", request_id) is None:
        raise ValueError("pause request id is malformed")
    try:
        parsed_requested_at = datetime.fromisoformat(requested_at)
    except ValueError as exc:
        raise ValueError("pause request timestamp is malformed") from exc
    if parsed_requested_at.tzinfo is None:
        raise ValueError("pause request timestamp must include a timezone")
    if step not in _SUPPORTED_PAUSE_STAGES:
        raise ValueError("pause request step is unsupported")
    run_id = payload.get("run_id")
    digest = payload.get("definition_digest")
    if run_id is not None and not isinstance(run_id, str):
        raise ValueError("pause request run_id is malformed")
    if digest is not None and not isinstance(digest, str):
        raise ValueError("pause request definition_digest is malformed")
    return PauseRequest(
        request_id=request_id,
        requested_at=requested_at,
        run_id=run_id,
        definition_digest=digest,
        step=step,
    )


def load_pause_request(state_dir: Path | None = None) -> PauseRequest | None:
    path = pause_request_path(state_dir)
    if not path.exists():
        return None
    return _parse_request(_load_json_object(path))


def pause_requested(state_dir: Path | None = None) -> bool:
    """Return true for valid or malformed markers so schedulers fail closed."""

    path = pause_request_path(state_dir)
    if not path.exists():
        return False
    try:
        load_pause_request(state_dir)
    except ValueError:
        return True
    return True


def _pauseable_definition(root: Path):
    state_path = root / "pipeline_state.json"
    state = _load_json_object(state_path)
    if state.get("status") != "running":
        raise ValueError("cooperative pause requires a running pipeline phase")
    step = state.get("step")
    if step not in _SUPPORTED_PAUSE_STAGES:
        raise ValueError(f"pipeline step {step!r} is not pause-aware")
    if step == "phase_2":
        stage = state.get("phase_2_stage")
    else:
        stage = state.get("pause_stage")
    if step == "phase_4" and state.get("process_pool"):
        if stage != _PROCESS_POOL_PAUSE_STAGE:
            raise ValueError(f"Phase 4 process-pool stage {stage!r} is not pause-aware")
        supported_stages = {_PROCESS_POOL_PAUSE_STAGE}
    else:
        supported_stages = _SUPPORTED_PAUSE_STAGES[str(step)]
    if stage not in supported_stages:
        raise ValueError(f"{step} stage {stage!r} is not pause-aware")
    return define_run(state)


def validate_active_pause_request(
    state: dict[str, object],
    request: PauseRequest,
) -> None:
    """Validate a marker against one active, pause-aware checkpoint."""

    if not isinstance(state, dict):
        raise ValueError("pause request state must contain an object")
    if state.get("status") != "running":
        raise ValueError("pause request does not match a running pipeline phase")
    step = state.get("step")
    if request.step != step or step not in _SUPPORTED_PAUSE_STAGES:
        raise ValueError("pause request does not match an active pause-aware Run")
    if step == "phase_2":
        stage = state.get("phase_2_stage")
    else:
        stage = state.get("pause_stage")
    supported_stages = (
        {_PROCESS_POOL_PAUSE_STAGE}
        if step == "phase_4" and state.get("process_pool")
        else _SUPPORTED_PAUSE_STAGES[str(step)]
    )
    if stage not in supported_stages:
        raise ValueError("pause request does not match an active pause-aware Run")
    definition = define_run(state)
    if (
        request.run_id != definition.run_id
        or request.definition_digest != definition.definition_digest
    ):
        raise ValueError("pause request does not match an active pause-aware Run")


@contextmanager
def pause_control_lock(root: Path) -> Iterator[None]:
    """Serialize pause requests with state-owned Phase 4 stage transitions."""

    handle = (root / _REQUEST_LOCK_FILE).open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def request_pause(state_dir: Path | None = None) -> PauseRequest:
    root = (state_dir or get_state_dir()).expanduser().resolve(strict=False)
    with pause_control_lock(root):
        state = _load_json_object(root / "pipeline_state.json")
        if state.get("status") == "paused":
            request = load_pause_request(root)
            if request is not None:
                return request
            raise ValueError("pipeline is already paused")
        definition = _pauseable_definition(root)
        existing = load_pause_request(root)
        if existing is not None:
            if (
                existing.step != state.get("step")
                or existing.run_id != definition.run_id
                or existing.definition_digest != definition.definition_digest
            ):
                raise ValueError("existing pause request targets a different Run")
            return existing
        request = PauseRequest(
            request_id=f"pause-{uuid.uuid4().hex}",
            requested_at=datetime.now(UTC).isoformat(),
            run_id=definition.run_id,
            definition_digest=definition.definition_digest,
            step=str(state.get("step")),
        )
        write_json_atomic(
            pause_request_path(root),
            request.to_dict(),
            failpoint_base="run_control.pause_request",
        )
        try:
            observed_definition = _pauseable_definition(root)
        except ValueError:
            clear_pause_request(root)
            raise
        if observed_definition != definition:
            clear_pause_request(root)
            raise ValueError("active Run changed while the pause request was being recorded")
        return request


def acknowledge_pause(state_dir: Path | None = None) -> dict[str, object]:
    root = (state_dir or get_state_dir()).expanduser().resolve(strict=False)
    with pause_control_lock(root):
        request = load_pause_request(root)
        if request is None:
            raise ValueError("pause boundary reached without a pause request")
        state = _load_json_object(root / "pipeline_state.json")
        definition = define_run(state)
        if (
            request.step != state.get("step")
            or request.run_id != definition.run_id
            or request.definition_digest != definition.definition_digest
        ):
            raise ValueError("pause request does not match the active Run")
        if state.get("status") == "paused" and state.get("pause_request_id") == request.request_id:
            clear_pause_request(root)
            return state
        validate_active_pause_request(state, request)
        updated = transition_pipeline_status(
            "paused",
            expected_statuses={"running"},
            state_dir=root,
            metadata={
                "paused_from_status": "running",
                "pause_request_id": request.request_id,
                "pause_requested_at": request.requested_at,
                "paused_at": datetime.now(UTC).isoformat(),
                "reason": "operator_requested_pause",
            },
        )
        clear_pause_request(root)
        return updated


def mark_interrupted(state_dir: Path | None = None, *, signal_name: str) -> dict[str, object]:
    root = (state_dir or get_state_dir()).expanduser().resolve(strict=False)
    with pause_control_lock(root):
        updated = transition_pipeline_status(
            "interrupted",
            expected_statuses={"running"},
            state_dir=root,
            metadata={
                "interrupted_from_status": "running",
                "interrupted_at": datetime.now(UTC).isoformat(),
                "interrupt_signal": signal_name,
                "reason": "abrupt_process_interruption",
            },
        )
        clear_pause_request(root)
        return updated


def clear_pause_request(state_dir: Path | None = None) -> None:
    try:
        pause_request_path(state_dir).unlink()
    except FileNotFoundError:
        pass


async def pause_aware_map[T, R](
    items: Sequence[T],
    operation: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    state_dir: Path | None = None,
) -> list[R | BaseException]:
    """Run admitted items to completion while stopping dequeue on pause."""

    if concurrency <= 0:
        raise ValueError("pause-aware concurrency must be positive")
    queue: asyncio.Queue[tuple[int, T]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    missing = object()
    results: list[object] = [missing] * len(items)
    paused = asyncio.Event()

    async def worker() -> None:
        while True:
            if pause_requested(state_dir):
                paused.set()
                return
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                try:
                    results[index] = await operation(item)
                except Exception as exc:
                    results[index] = exc
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
    if workers:
        await asyncio.gather(*workers)
    if paused.is_set():
        raise PauseBoundaryReached()
    if any(result is missing for result in results):
        raise RuntimeError("pause-aware scheduler lost an item without a pause request")
    return [result for result in results if result is not missing]  # type: ignore[misc]


__all__ = [
    "PauseBoundaryReached",
    "PauseRequest",
    "RunInterrupted",
    "acknowledge_pause",
    "clear_pause_request",
    "load_pause_request",
    "mark_interrupted",
    "pause_aware_map",
    "pause_control_lock",
    "pause_request_path",
    "pause_requested",
    "request_pause",
]
