"""Host-owned precondition slots for action-capability task generation.

Precondition slots describe the benign workflow role that page content may play.
They are not payloads, rewards, selectors, endpoints, fixtures, or cleanup
instructions. Keeping the schema here lets adapters and task cards share the
same fail-closed validation.
"""

from __future__ import annotations

import re
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

PRECONDITION_SLOT_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "allowed_frames",
        "model_visible_summary",
        "required_benign_bridge_terms",
        "forbidden_exact_fields",
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
_MODEL_VISIBLE_FORBIDDEN_PATH_RE = re.compile(
    r"(?i)(?:^|[\s`'\"])(?:/[A-Za-z0-9_.~:@!$&'()*+,;=-]+){1,}(?:[/?#][^\s`'\"]*)?"
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

    def to_model_dict(self) -> dict[str, Any]:
        """Return the prompt-safe slot projection."""

        return {
            "schema_version": PRECONDITION_SLOT_SCHEMA_VERSION,
            "role": self.role,
            "allowed_frames": list(self.allowed_frames),
            "model_visible_summary": self.model_visible_summary,
            "required_benign_bridge_terms": list(self.required_benign_bridge_terms),
        }


def precondition_slot_from_mapping(value: Mapping[str, Any]) -> PreconditionSlot:
    """Parse and validate a precondition slot mapping."""

    problem = validate_precondition_slot(value)
    if problem is not None:
        raise ValueError(problem)
    role = str(value.get("role") or "").strip()
    allowed_frames = _string_tuple(value.get("allowed_frames"))
    summary = str(value.get("model_visible_summary") or "").strip()
    required_terms = _string_tuple(value.get("required_benign_bridge_terms"))
    forbidden_fields = _string_tuple(value.get("forbidden_exact_fields"))
    return PreconditionSlot(
        role=role,
        allowed_frames=allowed_frames,
        model_visible_summary=summary,
        required_benign_bridge_terms=required_terms,
        forbidden_exact_fields=forbidden_fields,
    )


def validate_precondition_slot(value: Any) -> str | None:
    """Return a fail-closed validation problem for a slot-like value."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        return "precondition_slot must be an object"
    unknown_slot_fields = sorted(set(value) - PRECONDITION_SLOT_FIELDS)
    if unknown_slot_fields:
        return f"precondition_slot contains unsupported field(s): {unknown_slot_fields}"
    schema_version = value.get("schema_version")
    if schema_version is not None and schema_version != PRECONDITION_SLOT_SCHEMA_VERSION:
        return f"precondition_slot.schema_version must be {PRECONDITION_SLOT_SCHEMA_VERSION!r}"
    role = str(value.get("role") or "").strip()
    if role not in PRECONDITION_SLOT_ROLES:
        return f"precondition_slot.role must be one of {sorted(PRECONDITION_SLOT_ROLES)}"
    frames_problem = _validate_string_array_field(value, "allowed_frames")
    if frames_problem is not None:
        return f"precondition_slot.{frames_problem}"
    frames = _string_tuple(value.get("allowed_frames"))
    if not frames:
        return "precondition_slot.allowed_frames must be a non-empty array of strings"
    duplicate_frames = _duplicate_folded_values(frames)
    if duplicate_frames:
        return f"precondition_slot.allowed_frames contains duplicate frame(s): {duplicate_frames}"
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
    if _MODEL_VISIBLE_FORBIDDEN_PATH_RE.search(summary):
        return "precondition_slot.model_visible_summary contains forbidden endpoint/path detail"
    terms_problem = _validate_string_array_field(
        value,
        "required_benign_bridge_terms",
        allow_empty=True,
    )
    if terms_problem is not None:
        return f"precondition_slot.{terms_problem}"
    terms = _string_tuple(value.get("required_benign_bridge_terms"))
    duplicate_terms = _duplicate_folded_values(terms)
    if duplicate_terms:
        return (
            "precondition_slot.required_benign_bridge_terms contains duplicate "
            f"term(s): {duplicate_terms}"
        )
    for term in terms:
        term_folded = term.casefold()
        for forbidden in _MODEL_VISIBLE_FORBIDDEN_SUBSTRINGS:
            if forbidden in term_folded:
                return (
                    "precondition_slot.required_benign_bridge_terms contains forbidden "
                    f"benchmark/evaluator detail {forbidden!r}"
                )
        if _MODEL_VISIBLE_FORBIDDEN_PATH_RE.search(term):
            return (
                "precondition_slot.required_benign_bridge_terms contains forbidden "
                "endpoint/path detail"
            )
    fields_problem = _validate_string_array_field(
        value,
        "forbidden_exact_fields",
        allow_empty=True,
    )
    if fields_problem is not None:
        return f"precondition_slot.{fields_problem}"
    forbidden_fields = _string_tuple(value.get("forbidden_exact_fields"))
    duplicate_forbidden_fields = _duplicate_folded_values(forbidden_fields)
    if duplicate_forbidden_fields:
        return (
            "precondition_slot.forbidden_exact_fields contains duplicate field(s): "
            f"{duplicate_forbidden_fields}"
        )
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
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        slot = provenance.get("precondition_slot")
        if isinstance(slot, Mapping):
            return slot
    return None


def precondition_slot_model_projection_from_task(
    task: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return validated prompt-safe precondition slot metadata.

    Top-level slot fields are intentionally ignored: once slots influence prompt
    text, the only trusted source is host-owned task provenance.
    """

    slot = precondition_slot_from_task(task)
    if slot is None:
        return None
    return precondition_slot_from_mapping(slot).to_model_dict()


def _validate_string_array_field(
    value: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list):
        return f"{key} must be an array of strings"
    if (not raw and not allow_empty) or not all(
        isinstance(item, str) and item.strip() for item in raw
    ):
        return f"{key} must be an array of strings"
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, list):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if isinstance(value, tuple):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _duplicate_folded_values(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        folded = value.casefold()
        if folded in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(folded)
    return duplicates
