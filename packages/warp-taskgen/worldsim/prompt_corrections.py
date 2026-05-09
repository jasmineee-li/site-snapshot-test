"""Shared prompt repair feedback helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    expected: Any | None = None
    actual: Any | None = None
    repair_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        if self.repair_hint:
            payload["repair_hint"] = self.repair_hint
        return payload


def normalize_validation_errors(
    errors: Sequence[str | Mapping[str, Any] | ValidationIssue],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for error in errors:
        if isinstance(error, ValidationIssue):
            normalized.append(error.to_dict())
        elif isinstance(error, Mapping):
            normalized.append(_normalize_mapping_error(error))
        else:
            normalized.append(
                {
                    "code": "VALIDATION_ERROR",
                    "path": "$",
                    "message": str(error),
                }
            )
    return normalized


def render_validation_feedback(
    *,
    artifact_name: str,
    errors: Sequence[str | Mapping[str, Any] | ValidationIssue],
    summary: str | None = None,
    instruction: str | None = None,
    extra_guidance: str | None = None,
) -> str:
    """Render compact model-facing validation feedback for retry prompts."""
    normalized = normalize_validation_errors(errors)
    payload: dict[str, Any] = {
        "valid": False,
        "artifact": artifact_name,
        "summary": summary or f"{artifact_name} failed validation with {len(normalized)} error(s).",
        "errors": normalized,
    }
    if extra_guidance and extra_guidance.strip():
        payload["extra_guidance"] = extra_guidance.strip()

    retry_instruction = (
        instruction
        or "Fix only the listed issues, preserve valid content where possible, and return the complete artifact again. Do not include markdown or commentary."
    )
    return (
        "\n\n<validation_feedback>\n"
        f"The previous `{artifact_name}` failed validation. {retry_instruction}\n\n"
        "```json\n"
        f"{json.dumps(payload, indent=2)}\n"
        "```\n"
        "</validation_feedback>"
    )


def _normalize_mapping_error(error: Mapping[str, Any]) -> dict[str, Any]:
    code = error.get("code") or "VALIDATION_ERROR"
    path = error.get("path") or "$"
    message = error.get("message") or str(error)
    payload: dict[str, Any] = {
        "code": str(code),
        "path": str(path),
        "message": str(message),
    }
    for key in ("expected", "actual", "repair_hint"):
        if key in error and error[key] is not None:
            payload[key] = error[key]
    return payload
