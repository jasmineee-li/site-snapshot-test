"""Read-only inspection of one Phase 2b text-fill checkpoint envelope."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from warp_taskgen.run_definition_contracts import RunDefinition


@dataclass(frozen=True)
class TextFillCheckpointInspection:
    """One feature-owned, read-only checkpoint inspection result."""

    status: Literal["pending", "stale", "malformed", "compatible"]
    reason_code: str
    task: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None


def inspect_text_fill_checkpoint(
    path: Path,
    plan: dict[str, Any],
    *,
    definition: RunDefinition,
    text_model: str,
    texts_per_plan: int,
    settings: Mapping[str, Any] | None = None,
) -> TextFillCheckpointInspection:
    """Inspect one envelope using the exact resume validator authority.

    The result is advisory for status only.  ``load_text_fill_checkpoint``
    remains the boolean-compatible resume seam and projects the same single
    validation pass into its historical ``(task, diagnostics)`` value.

    Imports from ``checkpoints`` stay inside this function so the reader can
    be imported by that owner without creating a module cycle.  The helpers
    are still the production loader's own validation and hashing authority.
    """

    from warp_taskgen.phase_2.text_fill.checkpoints import (
        _json_copy,
        _sha256_json,
        _validate_completed_task,
        _validation_records,
    )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _checkpoint_inspection("pending", "checkpoint_missing")
    except (OSError, UnicodeError):
        return _checkpoint_inspection("malformed", "checkpoint_unreadable")
    except json.JSONDecodeError:
        return _checkpoint_inspection("malformed", "checkpoint_json_invalid")

    if not isinstance(payload, dict):
        return _checkpoint_inspection("malformed", "checkpoint_envelope_invalid")
    if _missing_checkpoint_fields(payload):
        return _checkpoint_inspection("malformed", "checkpoint_envelope_invalid")
    shape_reason = _checkpoint_envelope_shape_reason(payload)
    if shape_reason is not None:
        return _checkpoint_inspection("malformed", shape_reason)

    header_reason = _checkpoint_header_reason(
        payload,
        plan,
        definition=definition,
        text_model=text_model,
        texts_per_plan=texts_per_plan,
        settings=settings,
    )
    if header_reason is not None:
        return _checkpoint_inspection("stale", header_reason)
    if payload.get("outcome") != "complete":
        return _checkpoint_inspection("malformed", "checkpoint_outcome_incomplete")
    if payload.get("validation_errors"):
        return _checkpoint_inspection("malformed", "checkpoint_validation_error")

    task = payload.get("task")
    diagnostics = payload.get("diagnostics")
    validation = payload.get("validation")
    if not isinstance(task, dict) or not isinstance(diagnostics, dict):
        return _checkpoint_inspection("malformed", "checkpoint_envelope_invalid")
    if not isinstance(validation, list):
        return _checkpoint_inspection("malformed", "checkpoint_validation_invalid")
    try:
        if payload.get("task_sha256") != _sha256_json(task):
            return _checkpoint_inspection("malformed", "checkpoint_task_hash_mismatch")
        payload_ordinals = payload.get("payload_ordinals")
        if not isinstance(payload_ordinals, list):
            return _checkpoint_inspection("malformed", "checkpoint_ordinals_invalid")
        if payload_ordinals != task.get("payload_texts"):
            return _checkpoint_inspection("malformed", "checkpoint_ordinals_mismatch")
        if payload.get("payload_ordinals_sha256") != _sha256_json(payload_ordinals):
            return _checkpoint_inspection("malformed", "checkpoint_ordinals_hash_mismatch")
        if payload.get("selected_payload_index") != task.get("selected_payload_index"):
            return _checkpoint_inspection("malformed", "checkpoint_selected_ordinal_mismatch")
        if payload.get("selected_seed") != task.get("adversarial_data_seed"):
            return _checkpoint_inspection("malformed", "checkpoint_selected_seed_mismatch")
        if payload.get("diagnostics_sha256") != _sha256_json(diagnostics):
            return _checkpoint_inspection("malformed", "checkpoint_diagnostics_hash_mismatch")
        if payload.get("validation_sha256") != _sha256_json(validation):
            return _checkpoint_inspection("malformed", "checkpoint_validation_hash_mismatch")
        if validation != _validation_records(task, texts_per_plan=texts_per_plan):
            return _checkpoint_inspection("malformed", "checkpoint_validation_mismatch")
        _validate_completed_task(task, plan=plan, texts_per_plan=texts_per_plan)
    except (TypeError, ValueError, KeyError) as exc:
        return _checkpoint_inspection("malformed", _validation_reason_code(str(exc)))
    return TextFillCheckpointInspection(
        status="compatible",
        reason_code="checkpoint_compatible",
        task=_json_copy(task),
        diagnostics=_json_copy(diagnostics),
    )


def _checkpoint_header_reason(
    payload: object,
    plan: dict[str, Any],
    *,
    definition: RunDefinition,
    text_model: str,
    texts_per_plan: int,
    settings: Mapping[str, Any] | None,
) -> str | None:
    from warp_taskgen.phase_2.text_fill.checkpoints import (
        CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_STAGE,
        _checkpoint_settings,
        _sha256_json,
        _task_id,
        text_fill_input_digest,
    )

    if not isinstance(payload, dict):
        return "checkpoint_envelope_invalid"
    if not isinstance(definition, RunDefinition) or definition.legacy or not definition.run_id:
        return "checkpoint_run_definition_unavailable"
    if type(payload.get("schema_version")) is not int:
        return "checkpoint_schema_mismatch"
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return "checkpoint_schema_mismatch"
    if payload.get("stage") != CHECKPOINT_STAGE:
        return "checkpoint_stage_mismatch"
    try:
        task_id = _task_id(plan)
    except ValueError:
        return "checkpoint_input_invalid"
    checkpoint_settings = _checkpoint_settings(
        settings,
        text_model=text_model,
        texts_per_plan=texts_per_plan,
    )
    if payload.get("task_id") != task_id:
        return "checkpoint_task_id_mismatch"
    if payload.get("run_id") != definition.run_id:
        return "checkpoint_run_mismatch"
    if payload.get("definition_digest") != definition.definition_digest:
        return "checkpoint_definition_mismatch"
    if payload.get("legacy") is not False:
        return "checkpoint_legacy"
    if payload.get("input_sha256") != text_fill_input_digest(plan):
        return "checkpoint_input_mismatch"
    if payload.get("text_model") != text_model:
        return "checkpoint_model_mismatch"
    if type(payload.get("texts_per_plan")) is not int:
        return "checkpoint_count_mismatch"
    if payload.get("texts_per_plan") != texts_per_plan:
        return "checkpoint_count_mismatch"
    if payload.get("settings") != checkpoint_settings:
        return "checkpoint_settings_mismatch"
    if payload.get("settings_sha256") != _sha256_json(checkpoint_settings):
        return "checkpoint_settings_hash_mismatch"
    return None


_REQUIRED_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "task_id",
        "run_id",
        "definition_digest",
        "legacy",
        "input_sha256",
        "text_model",
        "texts_per_plan",
        "settings",
        "outcome",
        "task",
        "payload_ordinals",
        "selected_payload_index",
        "selected_seed",
        "diagnostics",
        "validation",
        "validation_errors",
        "task_sha256",
        "payload_ordinals_sha256",
        "diagnostics_sha256",
        "validation_sha256",
        "settings_sha256",
    }
)


def _missing_checkpoint_fields(payload: Mapping[str, Any]) -> set[str]:
    return _REQUIRED_CHECKPOINT_FIELDS.difference(payload)


def _checkpoint_envelope_shape_reason(payload: Mapping[str, Any]) -> str | None:
    """Reject non-envelope JSON shapes before classifying value drift."""

    if type(payload.get("schema_version")) is not int:
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("stage"), str):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("task_id"), str):
        return "checkpoint_envelope_invalid"
    if payload.get("run_id") is not None and not isinstance(payload.get("run_id"), str):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("definition_digest"), str):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("legacy"), bool):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("input_sha256"), str):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("text_model"), str):
        return "checkpoint_envelope_invalid"
    if type(payload.get("texts_per_plan")) is not int:
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("settings"), Mapping):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("outcome"), str):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("payload_ordinals"), list):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("diagnostics"), Mapping):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("validation"), list):
        return "checkpoint_envelope_invalid"
    if not isinstance(payload.get("validation_errors"), list):
        return "checkpoint_envelope_invalid"
    return None


def _checkpoint_inspection(
    status: Literal["pending", "stale", "malformed", "compatible"],
    reason_code: str,
) -> TextFillCheckpointInspection:
    return TextFillCheckpointInspection(status=status, reason_code=reason_code)


def _validation_reason_code(message: str) -> str:
    lowered = message.lower()
    if "selected seed" in lowered:
        return "checkpoint_selected_seed_mismatch"
    if "selected payload index" in lowered:
        return "checkpoint_selected_ordinal_invalid"
    if "payload ordinal" in lowered:
        return "checkpoint_ordinal_invalid"
    if "validation" in lowered or "failed validation" in lowered:
        return "checkpoint_validation_mismatch"
    return "checkpoint_task_invalid"


__all__ = ["TextFillCheckpointInspection", "inspect_text_fill_checkpoint"]
