"""Run-bound, task-local checkpoints for Phase 2b text fill.

Text fill is intentionally the owner of this checkpoint contract.  A
checkpoint is an envelope around one complete plan: every requested payload
ordinal is generated and validated before the envelope is replaced atomically.
The envelope is not a generic manifest and is never accepted as a source of
truth for Phase 2c.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.phase_2.text_fill.checkpoint_reader import (
    TextFillCheckpointInspection,
    inspect_text_fill_checkpoint,
)
from warp_taskgen.phase_2.text_fill.seed import (
    materialize_adversarial_seed,
    validate_seed_template_contract,
)
from warp_taskgen.phase_2.text_fill.validation import validate_text_post_hoc
from warp_taskgen.run_definition import define_run
from warp_taskgen.run_definition_contracts import RunDefinition
from warp_taskgen.state import load_state_for_current_root

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STAGE = "phase_2b_text_fill"

_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def text_fill_checkpoint_path(checkpoints_dir: Path, task_id: str) -> Path:
    """Return the deterministic path for one plan/task checkpoint.

    Normal task IDs are retained for operator visibility.  IDs that cannot be
    represented safely as one filename use a digest, so a malformed task can
    never escape the checkpoint directory.
    """

    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("text-fill checkpoint task_id must be a non-empty string")
    candidate = task_id.strip()
    filename = candidate if _SAFE_TASK_ID.fullmatch(candidate) else _sha256_text(candidate)
    return checkpoints_dir / f"{filename}.json"


def text_fill_task_id(plan: Mapping[str, Any]) -> str:
    """Return the normalized identifier used by text-fill checkpoints."""

    return _task_id(plan)


def text_fill_input_digest(plan: dict[str, Any]) -> str:
    """Hash the complete validated 2a plan consumed by text fill."""

    if not isinstance(plan, dict):
        raise ValueError("text-fill checkpoint input must be an object")
    return _sha256_json(plan)


def write_text_fill_checkpoint(
    path: Path,
    plan: dict[str, Any],
    task: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    *,
    text_model: str,
    texts_per_plan: int,
    settings: Mapping[str, Any] | None = None,
    definition: RunDefinition | None = None,
) -> dict[str, Any]:
    """Atomically persist one completed (or failed) text-fill unit.

    Failed units retain diagnostics but deliberately have no reusable task;
    the validator will make them run again on continuation.  Legacy runs may
    write evidence for inspection, but their envelopes are never reusable.
    """

    if not isinstance(plan, dict):
        raise ValueError("text-fill checkpoint input must be an object")
    task_id = _task_id(plan)
    if not isinstance(diagnostics, dict):
        raise ValueError("text-fill checkpoint diagnostics must be an object")
    if not isinstance(text_model, str) or not text_model.strip():
        raise ValueError("text-fill checkpoint text_model must be a non-empty string")
    if type(texts_per_plan) is not int or texts_per_plan <= 0:
        raise ValueError("text-fill checkpoint texts_per_plan must be positive")
    checkpoint_settings = _checkpoint_settings(
        settings,
        text_model=text_model,
        texts_per_plan=texts_per_plan,
    )

    if definition is None:
        definition = define_run(load_state_for_current_root() or {})
    if not isinstance(definition, RunDefinition):
        raise ValueError("text-fill checkpoint definition must be a RunDefinition")

    validation_errors: list[str] = []
    validation = []
    if task is not None:
        if str(task.get("id") or "") != task_id:
            raise ValueError("text-fill checkpoint task id does not match its plan")
        validation = _validation_records(task, texts_per_plan=texts_per_plan)
        validation_errors = [error for record in validation for error in record["errors"]]
        try:
            _validate_completed_task(task, plan=plan, texts_per_plan=texts_per_plan)
        except (TypeError, ValueError, KeyError) as exc:
            # The normal service has already performed post-hoc validation.
            # Keep a diagnostic checkpoint for compatibility callers that
            # synthesize legacy-shaped tasks, but never make that evidence
            # reusable: the loader below rejects non-empty validation_errors.
            validation_errors.append(str(exc))

    envelope: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": CHECKPOINT_STAGE,
        "task_id": task_id,
        "run_id": definition.run_id,
        "definition_digest": definition.definition_digest,
        "legacy": definition.legacy,
        "input_sha256": text_fill_input_digest(plan),
        "text_model": text_model,
        "texts_per_plan": texts_per_plan,
        "settings": checkpoint_settings,
        "outcome": "complete" if task is not None and not validation_errors else "failed",
        "task": _json_copy(task) if task is not None else None,
        "payload_ordinals": (
            _json_copy(task.get("payload_texts"))
            if task is not None and isinstance(task.get("payload_texts"), list)
            else []
        ),
        "selected_payload_index": task.get("selected_payload_index") if task is not None else None,
        "selected_seed": task.get("adversarial_data_seed") if task is not None else None,
        "diagnostics": _json_copy(diagnostics),
        "validation": validation,
        "validation_errors": validation_errors,
    }
    envelope["task_sha256"] = _sha256_json(task) if task is not None else None
    envelope["payload_ordinals_sha256"] = _sha256_json(envelope["payload_ordinals"])
    envelope["diagnostics_sha256"] = _sha256_json(diagnostics)
    envelope["validation_sha256"] = _sha256_json(validation)
    envelope["settings_sha256"] = _sha256_json(checkpoint_settings)
    write_json_atomic(
        path,
        envelope,
        failpoint_base="phase_2.text_fill.checkpoint",
    )
    return envelope


def load_text_fill_checkpoint(
    path: Path,
    plan: dict[str, Any],
    *,
    definition: RunDefinition,
    text_model: str,
    texts_per_plan: int,
    settings: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load a checkpoint only when every Run/input/settings guard matches."""

    inspection = inspect_text_fill_checkpoint(
        path,
        plan,
        definition=definition,
        text_model=text_model,
        texts_per_plan=texts_per_plan,
        settings=settings,
    )
    if inspection.status != "compatible":
        return None
    if inspection.task is None or inspection.diagnostics is None:
        return None
    return (_json_copy(inspection.task), _json_copy(inspection.diagnostics))


def text_fill_checkpoint_matches(
    path: Path,
    plan: dict[str, Any],
    *,
    definition: RunDefinition,
    text_model: str,
    texts_per_plan: int,
    settings: Mapping[str, Any] | None = None,
) -> bool:
    """Boolean compatibility projection used by resume tests and tooling."""

    return (
        load_text_fill_checkpoint(
            path,
            plan,
            definition=definition,
            text_model=text_model,
            texts_per_plan=texts_per_plan,
            settings=settings,
        )
        is not None
    )


def _validate_completed_task(
    task: dict[str, Any],
    *,
    plan: dict[str, Any],
    texts_per_plan: int,
) -> None:
    if not isinstance(task, dict):
        raise ValueError("text-fill checkpoint task must be an object")
    if str(task.get("id") or "") != _task_id(plan):
        raise ValueError("text-fill checkpoint task id is not bound to its plan")
    seed_template = task.get("seed_template")
    validate_seed_template_contract(seed_template)
    payloads = task.get("payload_texts")
    if not isinstance(payloads, list) or len(payloads) < texts_per_plan:
        raise ValueError("text-fill checkpoint does not contain every payload ordinal")
    selected_index = task.get("selected_payload_index")
    if type(selected_index) is not int or not (0 <= selected_index < len(payloads)):
        raise ValueError("text-fill checkpoint selected payload index is invalid")
    for payload in payloads:
        if not isinstance(payload, dict):
            raise ValueError("text-fill checkpoint payload ordinal is not an object")
        errors = validate_text_post_hoc(payload, task)
        if errors:
            raise ValueError("text-fill checkpoint payload failed validation: " + "; ".join(errors))
    selected = payloads[selected_index]
    if not isinstance(selected, dict) or not isinstance(selected.get("rendered_payload"), str):
        raise ValueError("text-fill checkpoint selected payload is malformed")
    expected_seed = materialize_adversarial_seed(
        seed_template,
        selected["rendered_payload"],
    )
    if task.get("adversarial_data_seed") != expected_seed:
        raise ValueError("text-fill checkpoint selected seed is stale")


def _validation_records(
    task: dict[str, Any],
    *,
    texts_per_plan: int,
) -> list[dict[str, Any]]:
    payloads = task.get("payload_texts")
    if not isinstance(payloads, list):
        return []
    records: list[dict[str, Any]] = []
    for ordinal, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            errors = ["payload ordinal is not an object"]
        else:
            errors = list(validate_text_post_hoc(payload, task))
        records.append({"ordinal": ordinal, "errors": errors})
    return records


def _task_id(plan: Mapping[str, Any]) -> str:
    task_id = plan.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("text-fill plan id must be a non-empty string")
    return task_id.strip()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint_settings(
    settings: Mapping[str, Any] | None,
    *,
    text_model: str,
    texts_per_plan: int,
) -> dict[str, Any]:
    """Normalize result-affecting and scheduler settings into the envelope."""

    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("text-fill checkpoint settings must be a mapping")
    normalized = _json_copy(dict(settings))
    if not isinstance(normalized, dict):
        raise ValueError("text-fill checkpoint settings must be JSON-shaped")
    normalized.setdefault("text_model", text_model)
    normalized.setdefault("texts_per_plan", texts_per_plan)
    return normalized


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STAGE",
    "TextFillCheckpointInspection",
    "inspect_text_fill_checkpoint",
    "load_text_fill_checkpoint",
    "text_fill_checkpoint_matches",
    "text_fill_checkpoint_path",
    "text_fill_input_digest",
    "text_fill_task_id",
    "write_text_fill_checkpoint",
]
