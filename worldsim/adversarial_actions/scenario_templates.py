"""Host-owned scenario templates for capability-aligned IPI tasks.

Scenario templates describe the benign workflow shape that makes an
adversarial action meaningful. They are not route contracts, selectors,
fixtures, rewards, endpoints, or cleanup recipes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from worldsim.adversarial_actions.precondition_slots import (
    PreconditionSlot,
    precondition_slot_from_mapping,
    validate_precondition_slot,
)

SCENARIO_TEMPLATE_SCHEMA_VERSION = "worldsim-scenario-template-v1"

SCENARIO_TEMPLATE_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "workflow_family",
        "user_visible_goal_summary",
        "allowed_frames",
        "model_visible_context_terms",
        "required_instruction_evidence",
        "forbidden_instruction_evidence",
        "precondition_slot",
    }
)

SCENARIO_TEMPLATE_ALLOWED_FRAMES = frozenset(
    {
        "public_followup",
        "discussion_reply",
        "repository_maintenance",
        "task_local_prerequisite",
        "stale_resource_cleanup",
        "duplicate_resource_cleanup",
        "navigation_reference",
    }
)

_FORBIDDEN_PROMPT_VISIBLE_SUBSTRINGS = (
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
    "cleanup",
    "graphql",
)
_FORBIDDEN_PATH_RE = re.compile(
    r"(?i)(?:^|[\s`'\"])(?:/[A-Za-z0-9_.~:@!$&'()*+,;=-]+){1,}(?:[/?#][^\s`'\"]*)?"
)
_FORBIDDEN_ENDPOINT_RE = re.compile(r"(?i)\b(?:api|rest)/[A-Za-z0-9_.~:@!$&'()*+,;=/?#-]+")
_FORBIDDEN_SELECTOR_RE = re.compile(
    r"(?:^|[\s`'\"])(?:#[A-Za-z][\w-]*|\.[A-Za-z][\w-]*|\[[A-Za-z_][\w-]*(?:[*^$|~]?=)[^\]]+\])"
)
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*$")


@dataclass(frozen=True)
class ScenarioTemplate:
    """Benchmark-neutral benign workflow scaffold for action IPI."""

    id: str
    workflow_family: str
    user_visible_goal_summary: str
    allowed_frames: tuple[str, ...]
    model_visible_context_terms: tuple[str, ...] = ()
    required_instruction_evidence: tuple[dict[str, Any], ...] = ()
    forbidden_instruction_evidence: tuple[str, ...] = ()
    precondition_slot: PreconditionSlot | dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": SCENARIO_TEMPLATE_SCHEMA_VERSION,
            "id": self.id,
            "workflow_family": self.workflow_family,
            "user_visible_goal_summary": self.user_visible_goal_summary,
            "allowed_frames": list(self.allowed_frames),
        }
        if self.model_visible_context_terms:
            result["model_visible_context_terms"] = list(self.model_visible_context_terms)
        if self.required_instruction_evidence:
            result["required_instruction_evidence"] = [
                _copy_evidence_group(group) for group in self.required_instruction_evidence
            ]
        if self.forbidden_instruction_evidence:
            result["forbidden_instruction_evidence"] = list(self.forbidden_instruction_evidence)
        if self.precondition_slot is not None:
            result["precondition_slot"] = (
                self.precondition_slot.to_dict()
                if isinstance(self.precondition_slot, PreconditionSlot)
                else dict(self.precondition_slot)
            )
        return result

    def to_model_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": SCENARIO_TEMPLATE_SCHEMA_VERSION,
            "id": self.id,
            "workflow_family": self.workflow_family,
            "user_visible_goal_summary": self.user_visible_goal_summary,
            "allowed_frames": list(self.allowed_frames),
            "model_visible_context_terms": list(self.model_visible_context_terms),
            "required_instruction_evidence": [
                _copy_evidence_group(group) for group in self.required_instruction_evidence
            ],
        }
        if self.precondition_slot is not None:
            slot = (
                self.precondition_slot
                if isinstance(self.precondition_slot, PreconditionSlot)
                else precondition_slot_from_mapping(self.precondition_slot)
            )
            result["precondition_slot"] = slot.to_model_dict()
        return result


def scenario_template_from_mapping(value: Mapping[str, Any]) -> ScenarioTemplate:
    problem = validate_scenario_template(value)
    if problem is not None:
        raise ValueError(problem)
    slot_value = value.get("precondition_slot")
    slot = precondition_slot_from_mapping(slot_value) if isinstance(slot_value, Mapping) else None
    return ScenarioTemplate(
        id=str(value["id"]).strip(),
        workflow_family=str(value["workflow_family"]).strip(),
        user_visible_goal_summary=str(value["user_visible_goal_summary"]).strip(),
        allowed_frames=_string_tuple(value.get("allowed_frames")),
        model_visible_context_terms=_string_tuple(value.get("model_visible_context_terms")),
        required_instruction_evidence=tuple(
            _copy_evidence_group(group)
            for group in value.get("required_instruction_evidence") or []
            if isinstance(group, Mapping)
        ),
        forbidden_instruction_evidence=_string_tuple(value.get("forbidden_instruction_evidence")),
        precondition_slot=slot,
    )


def validate_scenario_template(value: Any) -> str | None:
    """Return a fail-closed validation problem for scenario-template metadata."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        return "scenario_template must be an object"
    unknown = sorted(set(value) - SCENARIO_TEMPLATE_FIELDS)
    if unknown:
        return f"scenario_template contains unsupported field(s): {unknown}"
    schema_version = value.get("schema_version")
    if schema_version is not None and schema_version != SCENARIO_TEMPLATE_SCHEMA_VERSION:
        return f"scenario_template.schema_version must be {SCENARIO_TEMPLATE_SCHEMA_VERSION!r}"
    for key in ("id", "workflow_family"):
        raw = value.get(key)
        if (
            not isinstance(raw, str)
            or not raw.strip()
            or _IDENTIFIER_RE.fullmatch(raw.strip()) is None
        ):
            return f"scenario_template.{key} must be a lowercase identifier"
    summary = value.get("user_visible_goal_summary")
    if not isinstance(summary, str) or not summary.strip():
        return "scenario_template.user_visible_goal_summary must be a non-empty string"
    problem = _prompt_visible_text_problem("user_visible_goal_summary", summary)
    if problem is not None:
        return problem
    frames = _string_tuple(value.get("allowed_frames"))
    if not frames:
        return "scenario_template.allowed_frames must be a non-empty array of strings"
    duplicate_frames = _duplicate_folded_values(frames)
    if duplicate_frames:
        return f"scenario_template.allowed_frames contains duplicate frame(s): {duplicate_frames}"
    unknown_frames = sorted(set(frames) - SCENARIO_TEMPLATE_ALLOWED_FRAMES)
    if unknown_frames:
        return f"scenario_template.allowed_frames contains unsupported frame(s): {unknown_frames}"
    terms_problem = _validate_string_array_field(
        value,
        "model_visible_context_terms",
        allow_empty=True,
    )
    if terms_problem is not None:
        return f"scenario_template.{terms_problem}"
    for term in _string_tuple(value.get("model_visible_context_terms")):
        problem = _prompt_visible_text_problem("model_visible_context_terms", term)
        if problem is not None:
            return problem
    evidence_problem = _validate_required_instruction_evidence(
        value.get("required_instruction_evidence")
    )
    if evidence_problem is not None:
        return evidence_problem
    forbidden_problem = _validate_regex_array(value, "forbidden_instruction_evidence")
    if forbidden_problem is not None:
        return forbidden_problem
    slot_problem = validate_precondition_slot(value.get("precondition_slot"))
    if slot_problem is not None:
        return f"scenario_template.{slot_problem}"
    return None


def scenario_template_from_task(task: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(task, Mapping):
        return None
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        template = provenance.get("scenario_template")
        if isinstance(template, Mapping):
            return template
    return None


def scenario_template_model_projection_from_task(
    task: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    template = scenario_template_from_task(task)
    if template is None:
        return None
    return scenario_template_from_mapping(template).to_model_dict()


def _validate_required_instruction_evidence(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return "scenario_template.required_instruction_evidence must be an array"
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            return f"scenario_template.required_instruction_evidence[{index}] must be an object"
        unknown = sorted(set(item) - {"name", "any_regex"})
        if unknown:
            return (
                "scenario_template.required_instruction_evidence"
                f"[{index}] contains unsupported field(s): {unknown}"
            )
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            return (
                f"scenario_template.required_instruction_evidence[{index}].name must be non-empty"
            )
        name_problem = _prompt_visible_text_problem(
            f"required_instruction_evidence[{index}].name",
            name,
        )
        if name_problem is not None:
            return name_problem
        folded = name.casefold()
        if folded in names:
            return f"scenario_template.required_instruction_evidence[{index}].name is duplicated"
        names.add(folded)
        regexes = item.get("any_regex")
        if not isinstance(regexes, list) or not regexes:
            return (
                "scenario_template.required_instruction_evidence"
                f"[{index}].any_regex must be a non-empty array of strings"
            )
        if not all(isinstance(pattern, str) and pattern.strip() for pattern in regexes):
            return (
                "scenario_template.required_instruction_evidence"
                f"[{index}].any_regex must be a non-empty array of strings"
            )
        for pattern in regexes:
            try:
                re.compile(pattern)
            except re.error as exc:
                return (
                    "scenario_template.required_instruction_evidence"
                    f"[{index}].any_regex contains invalid regex {pattern!r}: {exc}"
                )
            pattern_problem = _prompt_visible_regex_problem(
                f"required_instruction_evidence[{index}].any_regex",
                pattern,
            )
            if pattern_problem is not None:
                return pattern_problem
    return None


def _validate_regex_array(value: Mapping[str, Any], key: str) -> str | None:
    problem = _validate_string_array_field(value, key, allow_empty=True)
    if problem is not None:
        return f"scenario_template.{problem}"
    for pattern in _string_tuple(value.get(key)):
        try:
            re.compile(pattern)
        except re.error as exc:
            return f"scenario_template.{key} contains invalid regex {pattern!r}: {exc}"
        pattern_problem = _prompt_visible_regex_problem(key, pattern)
        if pattern_problem is not None:
            return pattern_problem
    return None


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


def _prompt_visible_text_problem(field: str, text: str) -> str | None:
    folded = text.casefold()
    if _FORBIDDEN_PATH_RE.search(text) or _FORBIDDEN_ENDPOINT_RE.search(text):
        return f"scenario_template.{field} contains forbidden endpoint/path detail"
    if _FORBIDDEN_SELECTOR_RE.search(text):
        return f"scenario_template.{field} contains forbidden selector detail"
    for forbidden in _FORBIDDEN_PROMPT_VISIBLE_SUBSTRINGS:
        if forbidden in folded:
            return (
                f"scenario_template.{field} contains forbidden "
                f"benchmark/evaluator detail {forbidden!r}"
            )
    return None


def _prompt_visible_regex_problem(field: str, pattern: str) -> str | None:
    """Reject private details even when they are regex-escaped."""

    for candidate in {pattern, _unescape_regex_for_private_scan(pattern)}:
        problem = _prompt_visible_text_problem(field, candidate)
        if problem is not None:
            return problem
    return None


def _unescape_regex_for_private_scan(pattern: str) -> str:
    """Expose literal private tokens hidden behind common regex escaping."""

    return re.sub(r"\\([/._:#\[\]=~-])", r"\1", pattern)


def _copy_evidence_group(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(group.get("name") or "").strip(),
        "any_regex": [
            str(pattern).strip()
            for pattern in group.get("any_regex") or []
            if isinstance(pattern, str) and pattern.strip()
        ],
    }


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
