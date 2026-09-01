"""Exact response decoding and grading for Rocket.Chat decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from warp_taskgen.phase_1.rocket_chat_contracts import (
    DECISION_FIELDS,
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_EVALUATOR_AUTHORITY,
    RocketChatContractError,
    RocketChatDecision,
    validate_rocket_chat_benchmark,
)
from warp_taskgen.phase_1.rocket_chat_decisions import resolve_rocket_chat_evaluator_authority


def _decode_response(response: object) -> tuple[dict[str, object] | None, str | None]:
    """Decode only the bounded direct and retrieve-envelope response shapes."""

    value: object = response
    if isinstance(response, str):
        try:
            value = json.loads(response)
        except (TypeError, json.JSONDecodeError):
            return None, "response is not valid JSON"
    if not isinstance(value, Mapping):
        # In particular, a reader observation is not an answer.  It cannot be
        # used as a substitute for the acceptance seam's reader provenance.
        return None, "response must be a structured object"
    data = dict(value)
    direct_keys = set(DECISION_FIELDS)
    envelope_keys = {"task_type", "status", "retrieved_data"}
    if envelope_keys & set(data):
        missing = envelope_keys - set(data)
        extra = set(data) - envelope_keys
        if missing:
            return None, (
                "response envelope is missing fields: "
                f"{', '.join(sorted(str(item) for item in missing))}"
            )
        if extra:
            return None, (
                "response envelope has extra fields: "
                f"{', '.join(sorted(str(item) for item in extra))}"
            )
        if data["task_type"] != "retrieve":
            return None, "response task_type must be retrieve"
        if data["status"] != "SUCCESS":
            return None, "response status must be SUCCESS"
        retrieved = data["retrieved_data"]
        if isinstance(retrieved, (list, tuple)):
            if not retrieved:
                return None, "response retrieved_data must contain the decision fields"
            if all(isinstance(item, Mapping) for item in retrieved):
                if len(retrieved) != 1:
                    return None, "response retrieved_data must contain exactly one decision object"
                retrieved = retrieved[0]
            elif all(isinstance(item, str) for item in retrieved):
                entries: dict[str, str] = {}
                for item in retrieved:
                    field, separator, field_value = item.partition("=")
                    field = field.strip()
                    field_value = field_value.strip()
                    if not separator or not field or not field_value:
                        return None, "response retrieved_data entries must be field=value strings"
                    if field in entries:
                        return None, f"response retrieved_data repeats field {field!r}"
                    if field not in DECISION_FIELDS:
                        return None, f"response retrieved_data has extra field {field!r}"
                    entries[field] = field_value
                if set(entries) != set(DECISION_FIELDS):
                    missing = sorted(set(DECISION_FIELDS) - set(entries))
                    return None, f"response retrieved_data is missing fields: {', '.join(missing)}"
                return entries, None
            else:
                return (
                    None,
                    "response retrieved_data must contain one decision object or field=value entries",
                )
        if not isinstance(retrieved, Mapping):
            return None, "response retrieved_data must be a decision object"
        return dict(retrieved), None
    missing = direct_keys - set(data)
    extra = set(data) - direct_keys
    if missing:
        return (
            None,
            f"response is missing fields: {', '.join(sorted(str(item) for item in missing))}",
        )
    if extra:
        return None, f"response has extra fields: {', '.join(sorted(str(item) for item in extra))}"
    return data, None


@dataclass(frozen=True)
class RocketChatDecisionGrade:
    """Finite, structured result for one Rocket.Chat response grading call."""

    ok: bool
    reason: str
    expected: Mapping[str, str]
    actual: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise RocketChatContractError("decision grade ok must be boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise RocketChatContractError("decision grade reason must be bounded text")
        expected = RocketChatDecision.from_mapping(self.expected).as_dict()
        object.__setattr__(self, "expected", MappingProxyType(expected))
        if self.actual is not None:
            actual = RocketChatDecision.from_mapping(self.actual).as_dict()
            object.__setattr__(self, "actual", MappingProxyType(actual))

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "expected": dict(self.expected),
            "actual": None if self.actual is None else dict(self.actual),
        }

    def __bool__(self) -> bool:
        return self.ok


def grade_rocket_chat_decision(
    response: object,
    expected: RocketChatDecision | Mapping[str, object],
) -> RocketChatDecisionGrade:
    """Grade exact owner/date fields and reject malformed structures."""

    try:
        expected_decision = (
            expected
            if isinstance(expected, RocketChatDecision)
            else RocketChatDecision.from_mapping(expected)
        )
    except (TypeError, RocketChatContractError) as exc:
        raise RocketChatContractError(f"invalid expected Rocket.Chat decision: {exc}") from exc
    decoded, decode_error = _decode_response(response)
    if decoded is None:
        return RocketChatDecisionGrade(
            ok=False,
            reason=decode_error or "response could not be decoded",
            expected=expected_decision.as_dict(),
        )
    try:
        actual_decision = RocketChatDecision.from_mapping(decoded)
    except RocketChatContractError as exc:
        return RocketChatDecisionGrade(
            ok=False,
            reason=str(exc),
            expected=expected_decision.as_dict(),
        )
    actual = actual_decision.as_dict()
    expected_values = expected_decision.as_dict()
    if actual != expected_values:
        differing = [field for field in DECISION_FIELDS if actual[field] != expected_values[field]]
        return RocketChatDecisionGrade(
            ok=False,
            reason=f"decision mismatch in fields: {', '.join(differing)}",
            expected=expected_values,
            actual=actual,
        )
    return RocketChatDecisionGrade(
        ok=True,
        reason="exact Rocket.Chat decision matches expected fields",
        expected=expected_values,
        actual=actual,
    )


@dataclass(frozen=True)
class RocketChatEvaluator:
    """Task-id-less WARP evaluator authority for this response family."""

    benchmark: str = ROCKET_CHAT_BENCHMARK
    task_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        resolve_rocket_chat_evaluator_authority(self.benchmark, task_id=self.task_id)

    @property
    def authority(self) -> str:
        return ROCKET_CHAT_EVALUATOR_AUTHORITY

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "authority": self.authority,
        }

    def grade(
        self,
        response: object,
        expected: RocketChatDecision | Mapping[str, object],
    ) -> RocketChatDecisionGrade:
        return grade_rocket_chat_decision(response, expected)


__all__ = ["RocketChatDecisionGrade", "RocketChatEvaluator", "grade_rocket_chat_decision"]
