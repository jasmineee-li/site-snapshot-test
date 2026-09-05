"""Structured error exports for Phase 1 novel-task validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeneratedTaskValidationError:
    code: str
    path: str
    message: str
    expected: Any | None = None
    actual: Any | None = None
    repair_hint: str | None = None

    def render(self) -> str:
        parts = [f"{self.code} at {self.path}: {self.message}"]
        if self.expected is not None:
            parts.append(f"expected={self.expected!r}")
        if self.actual is not None:
            parts.append(f"actual={self.actual!r}")
        if self.repair_hint:
            parts.append(f"repair={self.repair_hint}")
        return "; ".join(parts)

    def legacy_render(self) -> str:
        return self.message

    def __contains__(self, text: object) -> bool:
        return isinstance(text, str) and text in self.render()

    def __str__(self) -> str:
        return self.render()

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


def _field_error(
    task_index: int,
    code: str,
    field_path: str,
    message: str,
    *,
    expected: Any | None = None,
    actual: Any | None = None,
    repair_hint: str | None = None,
) -> GeneratedTaskValidationError:
    return GeneratedTaskValidationError(
        code=code,
        path=f"$[{task_index}].{field_path}",
        message=message,
        expected=expected,
        actual=actual,
        repair_hint=repair_hint,
    )


__all__ = [
    "GeneratedTaskValidationError",
    "_field_error",
]
