"""Pipeline state persistence.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "State Persistence and Resume".
The filename is legacy; the spec now describes WARP Taskgen.

The orchestrator writes a single JSON state file before each major operation.
On crash, ``warp-taskgen resume`` reads this file and skips completed phases.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic

logger = logging.getLogger(__name__)
_DEFAULT_STATE_DIR = Path("logs")
_RESUME_POINTER = _DEFAULT_STATE_DIR / "last_run_state.json"
_PATHISH_METADATA_KEYS = {"feasibility_instances"}
STATE_DIR_ENV = "WARP_TASKGEN_STATE_DIR"
LEGACY_STATE_DIR_ENV = "WORLDSIM_STATE_DIR"


@dataclass(frozen=True)
class _StateCandidate:
    payload: dict[str, Any]
    source: str
    authoritative: bool
    path: Path | None = None


def get_state_dir() -> Path:
    """Return the current pipeline state directory."""
    return _canonical_path(_state_dir_override() or _DEFAULT_STATE_DIR)


def get_state_file() -> Path:
    """Return the current pipeline state file path."""
    return get_state_dir() / "pipeline_state.json"


def save_state(step: str, iteration: int = 0, **metadata: Any) -> None:
    """Write a checkpoint before each major operation.

    Args:
        step: Identifier for the current step (e.g. "phase_0a", "phase_3_task_5").
        iteration: Loop iteration counter for retry logic. 0 means first attempt.
        **metadata: Arbitrary extra fields merged into the state blob.
    """
    state_dir = get_state_dir()
    state_file = get_state_file()
    state_dir.mkdir(parents=True, exist_ok=True)
    normalized_metadata = _normalize_state_metadata(metadata)
    state = {
        "step": step,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "logs_dir": str(state_dir),
        **normalized_metadata,
    }
    # Discovery pointer first: if a crash lands between the two writes for a
    # brand-new custom logs dir, ``resume`` can still recover from the mirrored
    # snapshot instead of losing the run entirely.
    status = str(normalized_metadata.get("status", "unknown")).strip() or "unknown"
    _write_resume_pointer(state)
    # Atomic write: write to a temp file in the same directory, then rename.
    # os.replace is atomic on POSIX when src and dst are on the same filesystem.
    write_json_atomic(state_file, state, failpoint_base=f"state.save_state.{step}.{status}")


def load_state() -> dict[str, Any] | None:
    """Return the last saved state, or ``None`` if no state file exists."""
    primary = _load_state_candidate(get_state_file(), source="authoritative state")
    mirror_candidates = _load_resume_pointer_candidates()
    if _state_dir_override():
        return _select_best_state(
            [candidate for candidate in [primary, *mirror_candidates] if candidate is not None],
            expected_logs_dir=get_state_dir(),
        )

    return _select_best_state(
        [candidate for candidate in [primary, *mirror_candidates] if candidate is not None]
    )


def _state_dir_override() -> str | None:
    """Return the canonical state-dir override, falling back to legacy env."""
    return os.environ.get(STATE_DIR_ENV) or os.environ.get(LEGACY_STATE_DIR_ENV)


def _write_resume_pointer(state: dict[str, Any]) -> None:
    """Persist the latest full state snapshot at the default logs location."""
    pointer_path = _resume_pointer_path()
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(pointer_path, {**state, "state_file": str(_resume_pointer_target(state))})


def _load_resume_pointer_candidates() -> list[_StateCandidate]:
    """Return authoritative + snapshot candidates surfaced by the pointer."""
    payload = _load_state_file(_resume_pointer_path(), label="resume state mirror")
    if payload is None:
        return []
    candidates: list[_StateCandidate] = []
    candidate = _resume_pointer_target(payload)
    target_exists = False
    target_valid = False
    if candidate is not None:
        if not candidate.exists():
            logger.warning("Resume pointer referenced missing state file at %s", candidate)
        else:
            target_exists = True
            authoritative = _load_state_candidate(candidate, source="resume pointer target")
            if authoritative is not None:
                target_valid = True
                candidates.append(authoritative)

    snapshot = _mirror_snapshot_candidate(
        payload,
        target=candidate,
        target_exists=target_exists,
        target_valid=target_valid,
    )
    if snapshot is not None:
        candidates.append(snapshot)
    return candidates


def _resume_pointer_target(payload: dict[str, Any]) -> Path | None:
    """Return the authoritative state file path referenced by a mirror payload."""
    state_file = payload.get("state_file")
    logs_dir = payload.get("logs_dir")
    if state_file:
        return _canonical_path(state_file)
    if logs_dir:
        return _canonical_path(logs_dir) / "pipeline_state.json"
    return None


def _load_state_file(path: Path, *, label: str = "pipeline state file") -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        logger.warning("Unreadable %s at %s — ignoring (%s)", label, path, exc)
        return None
    except json.JSONDecodeError:
        logger.warning("Corrupt %s at %s — ignoring", label, path)
        return None
    if not isinstance(payload, dict):
        logger.warning("Invalid %s at %s — expected a JSON object", label, path)
        return None
    return _normalize_loaded_state(payload)


def _load_state_candidate(path: Path, *, source: str) -> _StateCandidate | None:
    payload = _load_state_file(path)
    if payload is None:
        return None
    if "step" not in payload or "status" not in payload:
        logger.warning("Invalid %s at %s — missing required resume keys", source, path)
        return None
    return _StateCandidate(payload=payload, source=source, authoritative=True, path=path)


def _mirror_snapshot_candidate(
    payload: dict[str, Any],
    *,
    target: Path | None,
    target_exists: bool,
    target_valid: bool,
) -> _StateCandidate | None:
    if "step" not in payload:
        return None
    if target_exists and not target_valid:
        return None
    if target is not None and not target_exists:
        logs_dir = payload.get("logs_dir")
        if not logs_dir or not _canonical_path(logs_dir).exists():
            return None
    return _StateCandidate(
        payload={
            **payload,
            "status": str(payload.get("status", "running")).strip() or "running",
        },
        source="resume pointer snapshot",
        authoritative=False,
        path=_resume_pointer_path(),
    )


def _select_best_state(
    candidates: list[_StateCandidate],
    *,
    expected_logs_dir: Path | None = None,
) -> dict[str, Any] | None:
    if expected_logs_dir is not None:
        expected = _canonical_path(expected_logs_dir)
        expected_state_file = expected / "pipeline_state.json"
        candidates = [
            candidate
            for candidate in candidates
            if _candidate_matches_expected_dir(
                candidate,
                expected_logs_dir=expected,
                expected_state_file=expected_state_file,
            )
        ]
    if not candidates:
        return None
    best = max(candidates, key=_candidate_sort_key)
    return best.payload


def _candidate_sort_key(candidate: _StateCandidate) -> tuple[datetime, int, int]:
    timestamp = _parse_state_timestamp(candidate.payload.get("timestamp"))
    return (timestamp, int(candidate.authoritative), int(bool(candidate.payload.get("status"))))


def _parse_state_timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                return parsed.astimezone(UTC).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass
    return datetime.min


def _payload_logs_dir(payload: dict[str, Any]) -> Path | None:
    logs_dir = payload.get("logs_dir")
    if not logs_dir:
        return None
    return _canonical_path(logs_dir)


def _candidate_matches_expected_dir(
    candidate: _StateCandidate,
    *,
    expected_logs_dir: Path,
    expected_state_file: Path,
) -> bool:
    if _payload_logs_dir(candidate.payload) != expected_logs_dir:
        return False
    if candidate.authoritative and candidate.path != expected_state_file:
        return False
    if not candidate.authoritative and _resume_pointer_target(candidate.payload) != expected_state_file:
        return False
    return True


def _normalize_state_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if _is_pathish_metadata(key, value):
            normalized[key] = str(_canonical_path(value))
        else:
            normalized[key] = value
    return normalized


def _normalize_loaded_state(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    logs_dir = normalized.get("logs_dir")
    if logs_dir:
        normalized["logs_dir"] = str(_canonical_path(logs_dir))
    state_file = normalized.get("state_file")
    if state_file:
        normalized["state_file"] = str(_canonical_path(state_file))
    elif logs_dir:
        normalized["state_file"] = str(_canonical_path(logs_dir) / "pipeline_state.json")
    for key, value in list(normalized.items()):
        if key in {"logs_dir", "state_file"}:
            continue
        if _is_pathish_metadata(key, value):
            normalized[key] = str(_canonical_path(value))
    return normalized


def _is_pathish_metadata(key: str, value: Any) -> bool:
    if not isinstance(value, (str, os.PathLike)):
        return False
    return key.endswith(("_path", "_dir", "_root")) or key in _PATHISH_METADATA_KEYS


def _canonical_path(value: str | os.PathLike[str] | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _resume_pointer_path() -> Path:
    return _canonical_path(_RESUME_POINTER)


# Backwards-compatible aliases. Internal code should prefer ``get_state_dir()``
# so .env-loaded overrides are honored even if this module was imported early.
STATE_DIR = get_state_dir()
STATE_FILE = get_state_file()
