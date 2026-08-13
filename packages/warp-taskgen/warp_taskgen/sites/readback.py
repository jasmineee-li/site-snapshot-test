"""Pure Site-owned interpretation of readback observations."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

ReadbackKind = Literal["resource_identity", "resource_signature", "comment_visibility"]


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, Set) and not isinstance(value, (str, bytes)):
        return frozenset(_freeze_payload(item) for item in value)
    return value


def identity_token_text(value: Any) -> str | None:
    """Normalize an opaque scalar resource ID without accepting containers."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class ReadbackObservation:
    kind: ReadbackKind
    identity_tokens: Mapping[str, Any]
    payload: Any
    signature: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"resource_identity", "resource_signature", "comment_visibility"}:
            raise ValueError("unsupported readback observation kind")
        if not isinstance(self.identity_tokens, Mapping):
            raise ValueError("readback observation requires identity-token metadata")
        if self.signature is not None and not isinstance(self.signature, str):
            raise ValueError("readback observation signature must be text")
        object.__setattr__(self, "identity_tokens", _freeze_payload(self.identity_tokens))
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True)
class ReadbackDecision:
    verified: bool
    reason: str
    matched_signature: str | None = None
    rendered_text: str | None = None


@dataclass(frozen=True)
class ReadbackFailure:
    site: str
    reason: str
    detail: str


class BoundReadback:
    _adapter: Any
    _context: Any

    def interpret_readback(
        self,
        observation: ReadbackObservation,
    ) -> ReadbackDecision | ReadbackFailure:
        if not isinstance(observation, ReadbackObservation):
            return ReadbackFailure(
                self._context.site,
                "invalid_readback_observation",
                "readback interpretation requires a typed observation",
            )
        interpreter = getattr(self._adapter, "interpret_readback", None)
        if not callable(interpreter):
            return ReadbackFailure(
                self._context.site,
                "unsupported_readback",
                "Site does not provide readback interpretation",
            )
        supported = getattr(self._adapter, "supported_benchmarks", frozenset())
        if self._context.benchmark not in supported:
            return ReadbackFailure(
                self._context.site,
                "unsupported_benchmark",
                f"benchmark {self._context.benchmark!r} is not supported by this Site",
            )
        try:
            decision = interpreter(observation)
        except Exception as exc:
            return ReadbackFailure(
                self._context.site,
                "readback_adapter_error",
                f"{exc.__class__.__name__}: {exc}",
            )
        if not isinstance(decision, ReadbackDecision):
            return ReadbackFailure(
                self._context.site,
                "invalid_readback_decision",
                "Site returned an unsupported readback decision",
            )
        return decision


__all__ = [
    "BoundReadback",
    "ReadbackDecision",
    "ReadbackFailure",
    "ReadbackKind",
    "ReadbackObservation",
    "identity_token_text",
]
