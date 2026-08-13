"""Durable, task-local checkpoints for Phase 2c feasibility verification.

Phase 2c deliberately treats one verification as an atomic work unit: seed,
render/readback/reachability, cleanup, then outcome.  This module owns the
small persistence seam for that unit.  The runner owns execution and this
module owns only the checkpoint envelope, atomic IO, and compatibility checks.

Checkpoints are intentionally run-bound.  A checkpoint without an opaque Run
ID and Definition Digest is legacy evidence and is never reused.  The
topology is represented by hashes/labels only; instance dictionaries (and in
particular auth material) never appear in a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.phase_2.output import _sanitize_task_for_output

# Bump these values when the semantics of the corresponding verifier/policy
# contract changes.  They are deliberately explicit instead of inferred from
# Python package metadata so a deployed run can explain why evidence was
# rejected after a code update.
CHECKPOINT_SCHEMA_VERSION = 1
VERIFIER_VERSION = "phase_2c-verifier-v1"
POLICY_CATALOG_VERSION = "phase_2c-policy-catalog-v1"
SITE_CATALOG_VERSION = "phase_2c-site-catalog-v1"
CHECKPOINT_KIND = "phase_2c_verification"
CHECKPOINT_FAILPOINT = "phase_2c.checkpoint"

_TASK_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_REWARD_PROBE_ANNOTATION_KEYS = frozenset(
    {
        "requires_attribution",
        "excluded_comment_ids",
        "seed_comment_id",
        "project_id",
        "project_path",
        "issue_iid",
    }
)


class CheckpointValidationError(ValueError):
    """Raised when a task checkpoint cannot be trusted for this run."""


@dataclass(frozen=True)
class Phase2cCheckpointContext:
    """Immutable expected identity for one Phase 2c verification run."""

    run_id: str
    definition_digest: str
    task_id: str
    task_content_hash: str
    task_fingerprint: str
    topology_fingerprint: Mapping[str, str]
    verifier_version: str = VERIFIER_VERSION
    policy_version: str = POLICY_CATALOG_VERSION
    catalog_version: str = SITE_CATALOG_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "definition_digest",
            "task_id",
            "task_content_hash",
            "task_fingerprint",
            "verifier_version",
            "policy_version",
            "catalog_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"checkpoint context {name} must be a non-empty string")
        if not isinstance(self.topology_fingerprint, Mapping):
            raise TypeError("checkpoint topology_fingerprint must be a mapping")
        for key, value in self.topology_fingerprint.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("checkpoint topology_fingerprint must contain string values")


@dataclass(frozen=True)
class CheckpointLoad:
    """Result of loading one checkpoint, including a human-readable reason."""

    result: dict[str, Any] | None
    path: Path
    reason: str
    completed_at: str | None = None
    cleanup_warnings: tuple[str, ...] = ()

    @property
    def reusable(self) -> bool:
        return self.result is not None


def task_identity(task: Mapping[str, Any]) -> str:
    """Return the stable task ID used for checkpoint filenames and binding."""

    raw = task.get("id") or task.get("task_id")
    return str(raw).strip() if raw is not None else ""


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Hash task content while excluding mutable feasibility annotations.

    The seed and exposure contract are the operational content of a task, but
    hashing the complete task projection also catches benign-target or reward
    drift before a stale outcome can be promoted into canonical artifacts.
    """

    # Verification adds read-surface metadata and attribution anchors to the
    # task before it is written back to ``adversarial_tasks.json``.  Keep those
    # derived values out of the source fingerprint, otherwise every ordinary
    # resume would look like task drift and reseed the benchmark.
    projection: dict[str, Any] = {}
    for raw_key, value in task.items():
        key = str(raw_key)
        if key in {
            "feasibility",
            "last_reverify_skipped_at",
            "read_surface_urls",
            "read_surface_provenance",
            "attribution_contract",
        }:
            continue
        if key == "exposure_contract" and isinstance(value, Mapping):
            projection[key] = {
                str(contract_key): contract_value
                for contract_key, contract_value in value.items()
                if str(contract_key) not in {"anchors", "attribution_contract"}
            }
        elif key == "reward_function" and isinstance(value, Mapping):
            projection[key] = _reward_fingerprint_projection(value)
        else:
            projection[key] = value
    # Hash the same sanitized projection that is written to canonical task
    # artifacts. This prevents credential redaction during artifact promotion
    # from masquerading as source-task drift on the next resume.
    sanitized_projection = _sanitize_task_for_output(projection)
    canonical = _canonical_json(sanitized_projection)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checkpoint_path(checkpoint_dir: Path, task_id: str) -> Path:
    """Return a collision-resistant, inspectable filename for *task_id*."""

    clean = _TASK_ID_RE.sub("_", str(task_id).strip()).strip("._") or "task"
    clean = clean[:96]
    suffix = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:16]
    return checkpoint_dir / f"{clean}-{suffix}.json"


def checkpoint_context(
    *,
    run_id: str | None,
    definition_digest: str | None,
    task: Mapping[str, Any],
    task_content_hash: str,
    topology_fingerprint: Mapping[str, str],
    verifier_version: str = VERIFIER_VERSION,
    policy_version: str = POLICY_CATALOG_VERSION,
    catalog_version: str = SITE_CATALOG_VERSION,
) -> Phase2cCheckpointContext | None:
    """Build a context, returning ``None`` for legacy/unidentified runs."""

    task_id = task_identity(task)
    if not run_id or not definition_digest or not task_id:
        return None
    return Phase2cCheckpointContext(
        run_id=str(run_id),
        definition_digest=str(definition_digest),
        task_id=task_id,
        task_content_hash=str(task_content_hash),
        task_fingerprint=task_fingerprint(task),
        topology_fingerprint=dict(topology_fingerprint),
        verifier_version=str(verifier_version),
        policy_version=str(policy_version),
        catalog_version=str(catalog_version),
    )


def write_checkpoint(
    checkpoint_dir: Path,
    *,
    context: Phase2cCheckpointContext,
    result: Mapping[str, Any],
    cleanup_warnings: list[str] | tuple[str, ...] = (),
    seed_applied: bool | None = None,
    render_completed: bool | None = None,
    reachability_completed: bool | None = None,
) -> Path:
    """Atomically persist one completed verification work unit.

    Callers must invoke this only after the seed cleanup handle has been
    attempted.  The explicit ``cleanup_completed`` marker makes that ordering
    auditable during crash/failpoint tests and prevents an outcome-only file
    from being mistaken for a complete unit.
    """

    if not isinstance(result, Mapping):
        raise TypeError("checkpoint result must be a mapping")
    if task_identity(result) != context.task_id:
        raise CheckpointValidationError("checkpoint result task does not match context")
    if task_fingerprint(result) != context.task_fingerprint:
        raise CheckpointValidationError("checkpoint result task fingerprint drifted")
    if not isinstance(result.get("feasibility"), Mapping):
        raise CheckpointValidationError("checkpoint result is missing feasibility stanza")
    status = _result_status(result)
    if status not in {"verified", "infeasible"}:
        raise CheckpointValidationError(
            f"checkpoint outcome must be verified or infeasible, got {status!r}"
        )
    warnings = [str(item) for item in cleanup_warnings if str(item)]
    seed_applied = status == "verified" if seed_applied is None else bool(seed_applied)
    render_completed = False if render_completed is None else bool(render_completed)
    reachability_completed = (
        False if reachability_completed is None else bool(reachability_completed)
    )
    path = checkpoint_path(checkpoint_dir, context.task_id)
    completed_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Task records may carry agent-context metadata. Reuse the Phase 2 output
    # sanitizer before writing a durable sidecar so credentials/cookie-like
    # values never cross the checkpoint boundary. The task fingerprint remains
    # computed from the source task and therefore still detects source drift.
    safe_result = _sanitize_task_for_output(dict(result))
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_id": f"{context.task_id}:{context.task_fingerprint[:16]}",
        "completed_at": completed_at,
        "run_id": context.run_id,
        "definition_digest": context.definition_digest,
        "task_id": context.task_id,
        "task_content_hash": context.task_content_hash,
        "task_fingerprint": context.task_fingerprint,
        "verifier_version": context.verifier_version,
        "policy_version": context.policy_version,
        "catalog_version": context.catalog_version,
        "topology_fingerprint": dict(context.topology_fingerprint),
        "cleanup_completed": True,
        "cleanup_warnings": warnings,
        "work_unit": {
            "seed_applied": seed_applied,
            "render_completed": render_completed,
            "reachability_completed": reachability_completed,
            "cleanup_completed": True,
            "outcome": status,
        },
        # ``result`` is the exact non-secret task evidence needed to rebuild
        # the canonical verified/infeasible artifacts.  No instance object is
        # copied into this envelope.
        "result": _copy_json(safe_result),
    }
    payload["checkpoint_digest"] = _payload_digest(payload)
    write_json_atomic(path, payload, failpoint_base=CHECKPOINT_FAILPOINT)
    return path


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    context: Phase2cCheckpointContext,
) -> CheckpointLoad:
    """Load and validate a task checkpoint against the expected context."""

    path = checkpoint_path(checkpoint_dir, context.task_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return CheckpointLoad(None, path, "missing")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return CheckpointLoad(None, path, "malformed")
    try:
        _validate_payload(raw, context)
    except CheckpointValidationError as exc:
        return CheckpointLoad(None, path, str(exc))
    return CheckpointLoad(
        dict(raw["result"]),
        path,
        "compatible",
        raw["completed_at"],
        tuple(raw["cleanup_warnings"]),
    )


def checkpoint_is_fresh(checkpoint: CheckpointLoad, *, ttl_hours: float | None) -> bool:
    """Return whether a compatible checkpoint satisfies an optional TTL."""

    if not checkpoint.reusable or ttl_hours is None:
        return checkpoint.reusable
    if ttl_hours < 0:
        return False
    completed_at = _parse_completed_at(checkpoint.completed_at)
    if completed_at is None:
        return False
    age_hours = (datetime.now(tz=UTC) - completed_at).total_seconds() / 3600.0
    return age_hours <= ttl_hours


def validate_checkpoint_payload(
    payload: Any,
    *,
    context: Phase2cCheckpointContext,
) -> dict[str, Any]:
    """Public strict validator used by aggregate promotion and tests."""

    _validate_payload(payload, context)
    return dict(payload["result"])


def _validate_payload(payload: Any, context: Phase2cCheckpointContext) -> None:
    if not isinstance(payload, dict):
        raise CheckpointValidationError("malformed: checkpoint must be an object")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointValidationError("legacy_or_schema_mismatch")
    if payload.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise CheckpointValidationError("malformed: checkpoint kind mismatch")
    # Verify the envelope integrity before interpreting nested evidence. A
    # user-edited status/outcome should be reported as tampered rather than as
    # an apparently valid but semantically inconsistent checkpoint.
    try:
        payload_digest = _payload_digest(payload)
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError("malformed: checkpoint is not JSON-shaped") from exc
    if payload.get("checkpoint_digest") != payload_digest:
        raise CheckpointValidationError("tampered")
    for field in (
        "checkpoint_id",
        "completed_at",
        "run_id",
        "definition_digest",
        "task_id",
        "task_content_hash",
        "task_fingerprint",
        "verifier_version",
        "policy_version",
        "catalog_version",
    ):
        expected = (
            f"{context.task_id}:{context.task_fingerprint[:16]}"
            if field == "checkpoint_id"
            else None
        )
        if field == "completed_at":
            if _parse_completed_at(payload.get(field)) is None:
                raise CheckpointValidationError("malformed: completed_at")
            continue
        if payload.get(field) != (expected if expected is not None else getattr(context, field)):
            reason = "unbound" if field in {"run_id", "definition_digest"} else f"{field}_drift"
            raise CheckpointValidationError(reason)
    if payload.get("topology_fingerprint") != dict(context.topology_fingerprint):
        raise CheckpointValidationError("topology_drift")
    if payload.get("cleanup_completed") is not True:
        raise CheckpointValidationError("cleanup_incomplete")
    if not isinstance(payload.get("cleanup_warnings"), list) or not all(
        isinstance(warning, str) for warning in payload["cleanup_warnings"]
    ):
        raise CheckpointValidationError("malformed: cleanup_warnings")
    work_unit = payload.get("work_unit")
    if not isinstance(work_unit, dict) or work_unit.get("cleanup_completed") is not True:
        raise CheckpointValidationError("cleanup_incomplete")
    for field in ("seed_applied", "render_completed", "reachability_completed"):
        if not isinstance(work_unit.get(field), bool):
            raise CheckpointValidationError(f"malformed: work_unit.{field}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise CheckpointValidationError("malformed: result")
    if task_identity(result) != context.task_id:
        raise CheckpointValidationError("task_drift")
    if task_fingerprint(result) != context.task_fingerprint:
        raise CheckpointValidationError("task_drift")
    result_status = _result_status(result)
    work_unit_outcome = work_unit.get("outcome")
    allowed_outcomes = {"verified", "infeasible"}
    if result_status not in allowed_outcomes or work_unit_outcome not in allowed_outcomes:
        raise CheckpointValidationError("malformed: invalid outcome")
    if result_status != work_unit_outcome:
        raise CheckpointValidationError("malformed: outcome mismatch")
    if not isinstance(result.get("feasibility"), dict):
        raise CheckpointValidationError("malformed: missing feasibility stanza")


def _result_status(result: Mapping[str, Any]) -> str | None:
    stanza = result.get("feasibility")
    return str(stanza.get("status")) if isinstance(stanza, Mapping) else None


def _parse_completed_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _reward_fingerprint_projection(
    value: Any,
    *,
    in_state_probe: bool = False,
) -> Any:
    """Copy reward config while dropping Phase 2c's derived probe annotations."""

    if isinstance(value, Mapping):
        projection: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if in_state_probe and key in _REWARD_PROBE_ANNOTATION_KEYS:
                continue
            projection[key] = _reward_fingerprint_projection(
                child,
                in_state_probe=in_state_probe or key == "state_probe",
            )
        return projection
    if isinstance(value, list):
        return [
            _reward_fingerprint_projection(item, in_state_probe=in_state_probe) for item in value
        ]
    return value


def _payload_digest(payload: Mapping[str, Any]) -> str:
    without_digest = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    return hashlib.sha256(_canonical_json(without_digest).encode("utf-8")).hexdigest()


def _copy_json(value: Any) -> Any:
    """Copy JSON-shaped evidence while rejecting accidental non-JSON values."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise TypeError("checkpoint result must be JSON serializable") from exc


__all__ = [
    "CHECKPOINT_FAILPOINT",
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "POLICY_CATALOG_VERSION",
    "SITE_CATALOG_VERSION",
    "VERIFIER_VERSION",
    "CheckpointLoad",
    "CheckpointValidationError",
    "Phase2cCheckpointContext",
    "checkpoint_context",
    "checkpoint_is_fresh",
    "checkpoint_path",
    "load_checkpoint",
    "task_fingerprint",
    "task_identity",
    "validate_checkpoint_payload",
    "write_checkpoint",
]
