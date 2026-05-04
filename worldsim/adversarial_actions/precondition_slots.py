"""Host-owned precondition slots for action-capability task generation.

Precondition slots describe the benign workflow role that page content may play.
They are not payloads, rewards, selectors, endpoints, fixtures, or cleanup
instructions. Keeping the schema here lets adapters and task cards share the
same fail-closed validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PRECONDITION_SLOT_SCHEMA_VERSION = "worldsim-precondition-slot-v1"

PRECONDITION_SLOT_ROLES = frozenset(
    {
        "task_local_prerequisite",
        "maintenance_prerequisite",
        "public_followup_instruction",
    }
)

PRECONDITION_SLOT_FRAMES = frozenset(
    {
        "stale_resource_cleanup",
        "duplicate_resource_cleanup",
        "repository_maintenance_record",
        "issue_triage_record",
        "public_followup_context",
        "discussion_followup_context",
    }
)

PRECONDITION_FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "fixture_project",
        "fixture_path",
        "fixture_content",
        "destructive_action",
        "attacker_url",
        "action_nonce",
        "endpoint_path",
        "selector",
        "reward_logic",
        "cleanup_behavior",
    }
)

_MODEL_VISIBLE_FORBIDDEN_SUBSTRINGS = (
    "__gitlab__",
    "__reddit__",
    "attacker.invalid",
    "worldsim",
    "fixture",
    "nonce",
    "/api/",
    "selector",
    "evaluator",
    "reward",
)


@dataclass(frozen=True)
class PreconditionSlot:
    """Benchmark-neutral role for page content in a benign workflow."""

    role: str
    allowed_frames: tuple[str, ...]
    model_visible_summary: str
    required_benign_bridge_terms: tuple[str, ...] = ()
    forbidden_exact_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRECONDITION_SLOT_SCHEMA_VERSION,
            "role": self.role,
            "allowed_frames": list(self.allowed_frames),
            "model_visible_summary": self.model_visible_summary,
            "required_benign_bridge_terms": list(self.required_benign_bridge_terms),
            "forbidden_exact_fields": list(self.forbidden_exact_fields),
        }


def precondition_slot_from_mapping(value: Mapping[str, Any]) -> PreconditionSlot:
    """Parse and validate a precondition slot mapping."""

    role = str(value.get("role") or "").strip()
    allowed_frames = _string_tuple(value.get("allowed_frames"))
    summary = str(value.get("model_visible_summary") or "").strip()
    required_terms = _string_tuple(value.get("required_benign_bridge_terms"))
    forbidden_fields = _string_tuple(value.get("forbidden_exact_fields"))
    slot = PreconditionSlot(
        role=role,
        allowed_frames=allowed_frames,
        model_visible_summary=summary,
        required_benign_bridge_terms=required_terms,
        forbidden_exact_fields=forbidden_fields,
    )
    problem = validate_precondition_slot(slot.to_dict())
    if problem is not None:
        raise ValueError(problem)
    return slot


def validate_precondition_slot(value: Any) -> str | None:
    """Return a fail-closed validation problem for a slot-like value."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        return "precondition_slot must be an object"
    role = str(value.get("role") or "").strip()
    if role not in PRECONDITION_SLOT_ROLES:
        return f"precondition_slot.role must be one of {sorted(PRECONDITION_SLOT_ROLES)}"
    frames = _string_tuple(value.get("allowed_frames"))
    if not frames:
        return "precondition_slot.allowed_frames must be a non-empty array of strings"
    unknown_frames = sorted(set(frames) - PRECONDITION_SLOT_FRAMES)
    if unknown_frames:
        return f"precondition_slot.allowed_frames contains unsupported frame(s): {unknown_frames}"
    summary = str(value.get("model_visible_summary") or "").strip()
    if not summary:
        return "precondition_slot.model_visible_summary must be a non-empty string"
    summary_folded = summary.casefold()
    for forbidden in _MODEL_VISIBLE_FORBIDDEN_SUBSTRINGS:
        if forbidden in summary_folded:
            return (
                "precondition_slot.model_visible_summary contains forbidden "
                f"benchmark/evaluator detail {forbidden!r}"
            )
    terms = _string_tuple(value.get("required_benign_bridge_terms"))
    if value.get("required_benign_bridge_terms") is not None and not terms:
        return "precondition_slot.required_benign_bridge_terms must be an array of strings"
    forbidden_fields = _string_tuple(value.get("forbidden_exact_fields"))
    if value.get("forbidden_exact_fields") is not None and not forbidden_fields:
        return "precondition_slot.forbidden_exact_fields must be an array of strings"
    unknown_fields = sorted(set(forbidden_fields) - PRECONDITION_FORBIDDEN_EXACT_FIELDS)
    if unknown_fields:
        return (
            "precondition_slot.forbidden_exact_fields contains unsupported "
            f"field(s): {unknown_fields}"
        )
    return None


def precondition_slot_from_task(task: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return host-owned precondition slot metadata attached to a task."""

    if not isinstance(task, Mapping):
        return None
    slot = task.get("precondition_slot")
    if isinstance(slot, Mapping):
        return slot
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        slot = provenance.get("precondition_slot")
        if isinstance(slot, Mapping):
            return slot
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, list):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if isinstance(value, tuple):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()
