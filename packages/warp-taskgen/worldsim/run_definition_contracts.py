"""Immutable value objects used by the read-only Run Definition seam."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

_SCHEMA_VERSION = 1

CheckpointAction = Literal["reuse", "rerun", "reject", "not_inspected"]
LifecycleAction = Literal["advance_phase", "rerun_phase", "finished", "reject"]
ResumeMode = Literal["exact", "legacy", "derived_required", "rejected"]


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("frozen contract keys must be non-empty strings")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (set, frozenset)):
        items = [_freeze(item) for item in value]
        return tuple(sorted(items, key=lambda item: json.dumps(_thaw(item), sort_keys=True)))
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("frozen contract values must be JSON-shaped and finite")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _digest(contributions: Mapping[str, object]) -> str:
    payload = json.dumps(
        _thaw(contributions),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optional_identity(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"run definition {field} must be a non-empty string or null")
    return value.strip()


@dataclass(frozen=True)
class RunDefinition:
    """Immutable, non-secret projection of one Run's effective inputs."""

    schema_version: int
    run_id: str | None
    source_run_id: str | None
    definition_digest: str
    contributions: Mapping[str, Mapping[str, object]]
    legacy: bool

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"run definition schema_version must be {_SCHEMA_VERSION}")
        _optional_identity(self.run_id, field="run_id")
        _optional_identity(self.source_run_id, field="source_run_id")
        if not isinstance(self.definition_digest, str) or len(self.definition_digest) != 64:
            raise ValueError("run definition digest must be a SHA-256 hex string")
        try:
            int(self.definition_digest, 16)
        except ValueError as exc:
            raise ValueError("run definition digest must be a SHA-256 hex string") from exc
        if not isinstance(self.legacy, bool):
            raise ValueError("run definition legacy must be boolean")
        frozen = _freeze(self.contributions)
        if not isinstance(frozen, Mapping):
            raise ValueError("run definition contributions must be a mapping")
        if any(not isinstance(values, Mapping) for values in frozen.values()):
            raise ValueError("run definition contributor values must be mappings")
        if _digest(frozen) != self.definition_digest:
            raise ValueError("run definition digest does not match contributions")
        if self.legacy and self.run_id is not None:
            raise ValueError("legacy run definition cannot declare a run_id")
        if not self.legacy and self.run_id is None:
            raise ValueError("non-legacy run definition requires a run_id")
        if self.source_run_id is not None and self.run_id is None:
            raise ValueError("source_run_id requires a persisted run_id")
        object.__setattr__(self, "contributions", frozen)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe serialization boundary for this frozen value."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "source_run_id": self.source_run_id,
            "definition_digest": self.definition_digest,
            "contributions": _thaw(self.contributions),
            "legacy": self.legacy,
        }


@dataclass(frozen=True)
class CheckpointDecision:
    checkpoint_id: str
    owner: str
    action: CheckpointAction
    reason_code: str
    path: str | None = None

    def __post_init__(self) -> None:
        for field in ("checkpoint_id", "owner", "reason_code"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"checkpoint decision {field} must be non-empty")
        if self.action not in {"reuse", "rerun", "reject", "not_inspected"}:
            raise ValueError("checkpoint decision has unsupported action")
        if self.path is not None and (not isinstance(self.path, str) or not self.path.strip()):
            raise ValueError("checkpoint decision path must be non-empty or null")

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "owner": self.owner,
            "action": self.action,
            "reason_code": self.reason_code,
            "path": self.path,
        }


@dataclass(frozen=True)
class ResumePlan:
    """Advisory explanation of current resume behavior and checkpoint scope."""

    schema_version: int
    mode: ResumeMode
    lifecycle_action: LifecycleAction
    source_run_id: str | None
    source_digest: str
    requested_digest: str
    current_step: str
    target_step: str | None
    state_status: str
    drift_fields: tuple[str, ...]
    checkpoint_decisions: tuple[CheckpointDecision, ...]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"resume plan schema_version must be {_SCHEMA_VERSION}")
        if self.mode not in {"exact", "legacy", "derived_required", "rejected"}:
            raise ValueError("resume plan has unsupported mode")
        if self.lifecycle_action not in {
            "advance_phase",
            "rerun_phase",
            "finished",
            "reject",
        }:
            raise ValueError("resume plan has unsupported lifecycle action")
        for field in ("source_digest", "requested_digest"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"resume plan {field} must be a SHA-256 hex string")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"resume plan {field} must be a SHA-256 hex string") from exc
        object.__setattr__(self, "drift_fields", tuple(self.drift_fields))
        object.__setattr__(self, "checkpoint_decisions", tuple(self.checkpoint_decisions))
        object.__setattr__(self, "errors", tuple(self.errors))
        if any(not isinstance(row, CheckpointDecision) for row in self.checkpoint_decisions):
            raise ValueError("resume plan checkpoint_decisions must contain decisions")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "lifecycle_action": self.lifecycle_action,
            "source_run_id": self.source_run_id,
            "source_digest": self.source_digest,
            "requested_digest": self.requested_digest,
            "current_step": self.current_step,
            "target_step": self.target_step,
            "state_status": self.state_status,
            "drift_fields": list(self.drift_fields),
            "checkpoint_decisions": [row.to_dict() for row in self.checkpoint_decisions],
            "errors": list(self.errors),
        }
