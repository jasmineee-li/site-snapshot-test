"""Checkpoint-aligned pause scheduling for Phase 2a planning shards."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.run_control import PauseBoundaryReached, pause_control_lock, pause_requested
from warp_taskgen.run_definition import define_run
from warp_taskgen.run_definition_contracts import RunDefinition
from warp_taskgen.state import get_state_dir, load_state_for_current_root

_SCHEMA_VERSION = 1
_MISSING_PAYLOAD = object()


@dataclass(frozen=True)
class PlanningShardCheckpointRead:
    """Single-read result shared by checkpoint status and reuse admission."""

    status: Literal["missing", "stale", "malformed", "compatible"]
    reason_code: str
    payload: list[dict] | None = None
    manifest: dict[str, Any] | None = None


async def run_planning_shards[T, R](
    items: Sequence[T],
    operation: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    state_dir: Path | None = None,
) -> list[R | BaseException]:
    """Drain admitted shards and stop claiming work after a pause request."""

    if concurrency <= 0:
        raise ValueError("Phase 2 planning concurrency must be positive")
    root = (state_dir or get_state_dir()).expanduser().resolve(strict=False)
    queue: asyncio.Queue[tuple[int, T]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    missing = object()
    results: list[object] = [missing] * len(items)
    paused = asyncio.Event()
    claim_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            async with claim_lock:
                with pause_control_lock(root):
                    if pause_requested(root):
                        paused.set()
                        return
                    try:
                        index, item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
            try:
                try:
                    results[index] = await operation(item)
                except PauseBoundaryReached:
                    paused.set()
                    return
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
        raise RuntimeError("Phase 2 planning scheduler lost a shard")
    return [result for result in results if result is not missing]  # type: ignore[misc]


def write_planning_shard_checkpoint(
    path: Path,
    payload: list[dict],
    *,
    label: str,
    input_task_ids: list[str],
) -> None:
    """Atomically persist a shard and its non-secret Run binding."""

    normalized_input_ids = _validated_task_ids(input_task_ids)
    output_task_ids = _payload_task_ids(payload)
    state = load_state_for_current_root()
    definition = define_run(state or {})
    write_json_atomic(
        path,
        payload,
        failpoint_base="phase_2.planning_shard.output",
    )
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "stage": "phase_2_planning",
        "label": label,
        "run_id": definition.run_id,
        "definition_digest": definition.definition_digest,
        "legacy": definition.legacy,
        "input_task_ids": normalized_input_ids,
        "output_task_ids": output_task_ids,
        "output_sha256": _payload_digest(payload),
    }
    write_json_atomic(
        _manifest_path(path),
        manifest,
        failpoint_base="phase_2.planning_shard.manifest",
    )


def planning_shard_checkpoint_matches(
    path: Path,
    payload: list[dict],
    *,
    definition: RunDefinition,
    expected_input_task_ids: Sequence[object] | None = None,
) -> bool:
    """Return whether a shard manifest binds the payload to this exact Run."""

    return (
        _read_planning_shard_checkpoint(
            path,
            definition=definition,
            expected_input_task_ids=expected_input_task_ids,
            payload=payload,
        ).status
        == "compatible"
    )


def _read_planning_shard_checkpoint(
    path: Path,
    *,
    definition: RunDefinition,
    expected_input_task_ids: Sequence[object] | None = None,
    payload: object = _MISSING_PAYLOAD,
) -> PlanningShardCheckpointRead:
    """Read and classify one checkpoint without rereading its sidecars.

    A caller that already parsed the payload can pass it through ``payload``;
    otherwise this helper reads the payload and manifest exactly once each.
    The returned payload is the same in-memory value used for compatibility
    checks, so reuse and status cannot disagree because of a second read.
    """

    if definition.legacy or not definition.run_id:
        return PlanningShardCheckpointRead("stale", "checkpoint_definition_unavailable")

    if payload is _MISSING_PAYLOAD:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return PlanningShardCheckpointRead("missing", "checkpoint_missing")
        except UnicodeError:
            return PlanningShardCheckpointRead("malformed", "checkpoint_unreadable")
        except (OSError, json.JSONDecodeError):
            return PlanningShardCheckpointRead("malformed", "checkpoint_json_invalid")
    if not isinstance(payload, list):
        return PlanningShardCheckpointRead("malformed", "checkpoint_output_invalid")

    try:
        manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PlanningShardCheckpointRead("malformed", "manifest_missing")
    except UnicodeError:
        return PlanningShardCheckpointRead("malformed", "manifest_unreadable")
    except (OSError, json.JSONDecodeError):
        return PlanningShardCheckpointRead("malformed", "manifest_invalid")
    if not isinstance(manifest, dict) or not _manifest_shape_matches(
        manifest,
        label=path.stem,
    ):
        return PlanningShardCheckpointRead("malformed", "manifest_invalid")

    try:
        expected_input_ids = (
            _validated_task_ids(expected_input_task_ids)
            if expected_input_task_ids is not None
            else None
        )
    except (TypeError, ValueError):
        return PlanningShardCheckpointRead("malformed", "checkpoint_input_invalid")

    if (
        manifest.get("run_id") != definition.run_id
        or manifest.get("definition_digest") != definition.definition_digest
        or manifest.get("legacy") is not False
        or (expected_input_ids is not None and manifest.get("input_task_ids") != expected_input_ids)
    ):
        return PlanningShardCheckpointRead(
            "stale", "checkpoint_binding_mismatch", manifest=manifest
        )

    try:
        output_task_ids = _payload_task_ids(payload)
        output_digest = _payload_digest(payload)
    except (TypeError, ValueError):
        return PlanningShardCheckpointRead("malformed", "checkpoint_output_invalid")
    if (
        manifest.get("output_task_ids") != output_task_ids
        or manifest.get("output_sha256") != output_digest
    ):
        return PlanningShardCheckpointRead("malformed", "checkpoint_output_mismatch")
    return PlanningShardCheckpointRead(
        "compatible",
        "checkpoint_compatible",
        payload=payload,
        manifest=manifest,
    )


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _manifest_shape_matches(manifest: Mapping[str, Any], *, label: str) -> bool:
    output_sha256 = manifest.get("output_sha256")
    definition_digest = manifest.get("definition_digest")

    def _is_sha256(value: object) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    return bool(
        type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == _SCHEMA_VERSION
        and manifest.get("stage") == "phase_2_planning"
        and manifest.get("label") == label
        and isinstance(manifest.get("run_id"), str)
        and bool(manifest.get("run_id"))
        and _is_sha256(definition_digest)
        and isinstance(manifest.get("legacy"), bool)
        and isinstance(manifest.get("input_task_ids"), list)
        and all(isinstance(value, str) and value.strip() for value in manifest["input_task_ids"])
        and isinstance(manifest.get("output_task_ids"), list)
        and all(isinstance(value, str) and value.strip() for value in manifest["output_task_ids"])
        and _is_sha256(output_sha256)
    )


def _payload_digest(payload: list[dict]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_task_ids(payload: list[dict]) -> list[str]:
    if any(not isinstance(task, dict) for task in payload):
        raise ValueError("Phase 2 planning shard payload entries must be objects")
    return _validated_task_ids([task.get("id") for task in payload])


def _validated_task_ids(values: Sequence[object]) -> list[str]:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Phase 2 planning shard task IDs must be non-empty strings")
    normalized = sorted(value.strip() for value in values if isinstance(value, str))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Phase 2 planning shard task IDs must be unique")
    return normalized
