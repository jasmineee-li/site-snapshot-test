"""Classifieds anonymous-reader contract for the live canary.

The writer and reader are deliberately different browser contexts for the
Classifieds proof.  This feature-local contract says exactly what the reader
is allowed to carry: an explicit ``reader_auth: {"type": "none"}`` declaration
and no storage state, cookies, HTTP auth, or auth headers.  The runtime owns
creating the context; this module only validates the inputs and records the
small evidence needed to prove isolation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

CLASSIFIEDS_READER_AUTH_TYPE = "none"
CLASSIFIEDS_READER_CONTRACT_SCHEMA_VERSION = "classifieds-reader-contract-v1"
_READER_AUTH_FIELDS = frozenset({"type"})
_AUTH_CONTEXT_KEYS = frozenset(
    {
        "storage_state",
        "cookies",
        "extra_http_headers",
        "http_credentials",
        "proxy",
    }
)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _has_items(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping | Sequence) and not isinstance(value, str | bytes):
        return bool(value)
    return True


@dataclass(frozen=True)
class ClassifiedsReaderPreflight:
    """Result of validating the ordinary-reader context contract."""

    ok: bool
    reason: str | None = None
    detail: str = ""
    reader_auth: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({"type": CLASSIFIEDS_READER_AUTH_TYPE})
    )
    browser_context_kwargs: Mapping[str, Any] = field(default_factory=dict)
    fresh_context_required: bool = True
    writer_context_reuse_forbidden: bool = True

    def __post_init__(self) -> None:
        if self.ok and self.reason is not None:
            raise ValueError("successful Classifieds reader preflight cannot have a reason")
        if not self.ok and not self.reason:
            raise ValueError("failed Classifieds reader preflight needs a reason")
        if not isinstance(self.reader_auth, Mapping):
            raise ValueError("reader_auth must be a mapping")
        if not isinstance(self.browser_context_kwargs, Mapping):
            raise ValueError("browser_context_kwargs must be a mapping")
        object.__setattr__(self, "reader_auth", _freeze_mapping(self.reader_auth))
        object.__setattr__(
            self,
            "browser_context_kwargs",
            _freeze_mapping(self.browser_context_kwargs),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return safe evidence metadata without runtime IDs or auth material."""

        return {
            "reader_auth": dict(self.reader_auth),
            "fresh_context_required": self.fresh_context_required,
            "writer_context_reuse_forbidden": self.writer_context_reuse_forbidden,
        }


def preflight_classifieds_reader(
    instance: Mapping[str, Any],
    *,
    reader_context_kwargs: Mapping[str, Any] | None = None,
    writer_context: object | None = None,
    reader_context: object | None = None,
    reader_cookies: Sequence[Mapping[str, Any]] | None = None,
) -> ClassifiedsReaderPreflight:
    """Validate one Classifieds independent-reader setup.

    ``instance`` must explicitly declare ``reader_auth: {"type": "none"}``.
    A missing declaration is not silently treated as anonymous because that
    would make a live canary unable to distinguish an omitted reader policy
    from a deliberate no-auth reader.  Context options and observed cookies
    are checked separately so the runtime can call this both before and after
    creating a fresh Playwright context.
    """

    if not isinstance(instance, Mapping):
        return _failure("invalid_instance", "Classifieds reader instance must be a mapping")
    declared = instance.get("reader_auth")
    if not isinstance(declared, Mapping):
        return _failure(
            "missing_reader_auth",
            'Classifieds live canary requires explicit reader_auth: {"type": "none"}',
        )
    if set(declared) != _READER_AUTH_FIELDS or declared.get("type") != CLASSIFIEDS_READER_AUTH_TYPE:
        return _failure(
            "non_anonymous_reader_auth",
            'Classifieds reader_auth must contain only {"type": "none"}',
        )

    context_kwargs = {} if reader_context_kwargs is None else reader_context_kwargs
    if not isinstance(context_kwargs, Mapping):
        return _failure("invalid_reader_context", "reader context options must be a mapping")
    forbidden = sorted(
        key
        for key in _AUTH_CONTEXT_KEYS
        if key in context_kwargs and (key == "storage_state" or _has_items(context_kwargs[key]))
    )
    if forbidden:
        return _failure(
            "reader_context_auth",
            "anonymous Classifieds reader cannot carry " + ", ".join(forbidden),
        )
    if (
        writer_context is not None
        and reader_context is not None
        and writer_context is reader_context
    ):
        return _failure(
            "writer_context_reused",
            "Classifieds reader must use a fresh browser context",
        )
    if reader_cookies is not None and len(reader_cookies) > 0:
        return _failure(
            "reader_context_has_cookies",
            "anonymous Classifieds reader must start without cookies",
        )

    return ClassifiedsReaderPreflight(
        ok=True,
        reader_auth={"type": CLASSIFIEDS_READER_AUTH_TYPE},
        browser_context_kwargs={},
    )


def _failure(reason: str, detail: str) -> ClassifiedsReaderPreflight:
    return ClassifiedsReaderPreflight(ok=False, reason=reason, detail=detail)


__all__ = [
    "CLASSIFIEDS_READER_AUTH_TYPE",
    "CLASSIFIEDS_READER_CONTRACT_SCHEMA_VERSION",
    "ClassifiedsReaderPreflight",
    "preflight_classifieds_reader",
]
