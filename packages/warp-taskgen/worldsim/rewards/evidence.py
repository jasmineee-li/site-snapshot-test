from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidencePolicy:
    required: frozenset[str]
    allowed_source: frozenset[str]

    @property
    def requires_state_readback(self) -> bool:
        return "state_readback" in self.required

    @property
    def allows_ui_state_transition(self) -> bool:
        return "ui_state_transition" in self.allowed_source

    @property
    def allows_network_event(self) -> bool:
        return "network_event" in self.allowed_source


_SUPPORTED_REQUIRED = frozenset({"source_event", "state_readback"})
_SUPPORTED_ALLOWED_SOURCE = frozenset({"network_event", "ui_state_transition"})


def parse_evidence_policy(raw: Any) -> tuple[EvidencePolicy, str | None]:
    """Parse explicit reward evidence policy.

    Missing policy preserves legacy WebArena-style semantics: a matching source
    network event is required before state readback.
    """

    if raw is None:
        return EvidencePolicy(
            required=frozenset({"source_event"}),
            allowed_source=frozenset({"network_event"}),
        ), None
    if not isinstance(raw, Mapping):
        return EvidencePolicy(frozenset(), frozenset()), "evidence_policy must be an object"

    unsupported = set(raw) - {"required", "allowed_source"}
    if unsupported:
        return (
            EvidencePolicy(frozenset(), frozenset()),
            "evidence_policy unsupported fields: "
            + ", ".join(sorted(str(key) for key in unsupported)),
        )

    required, required_error = _parse_string_set(raw.get("required"), field="required")
    if required_error:
        return EvidencePolicy(frozenset(), frozenset()), required_error
    allowed, allowed_error = _parse_string_set(raw.get("allowed_source"), field="allowed_source")
    if allowed_error:
        return EvidencePolicy(frozenset(), frozenset()), allowed_error

    required = required or frozenset({"source_event"})
    allowed = allowed or frozenset({"network_event"})
    unsupported_required = required - _SUPPORTED_REQUIRED
    if unsupported_required:
        return (
            EvidencePolicy(frozenset(), frozenset()),
            "evidence_policy.required unsupported values: "
            + ", ".join(sorted(unsupported_required)),
        )
    unsupported_allowed = allowed - _SUPPORTED_ALLOWED_SOURCE
    if unsupported_allowed:
        return (
            EvidencePolicy(frozenset(), frozenset()),
            "evidence_policy.allowed_source unsupported values: "
            + ", ".join(sorted(unsupported_allowed)),
        )
    if "source_event" in required and "network_event" not in allowed:
        return (
            EvidencePolicy(frozenset(), frozenset()),
            "evidence_policy requiring source_event must allow network_event",
        )
    return EvidencePolicy(required=required, allowed_source=allowed), None


def _parse_string_set(raw: Any, *, field: str) -> tuple[frozenset[str], str | None]:
    if raw is None:
        return frozenset(), None
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return frozenset(), f"evidence_policy.{field} must be a string or list"
    parsed = frozenset(str(item).strip() for item in values if str(item).strip())
    if not parsed:
        return frozenset(), f"evidence_policy.{field} must not be empty"
    return parsed, None
