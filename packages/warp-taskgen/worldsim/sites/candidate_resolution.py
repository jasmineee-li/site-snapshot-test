"""Pure L3 candidate contracts for the Site Targeting seam.

This module owns the value objects and fail-closed bridge from intent/probe
evidence to a bound Site target.  Contract and evidence helpers live in their
feature-owned modules so the catalog remains a thin registry facade.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from worldsim.sites.contracts import (
    ResolvedTarget,
    SiteTargetingDefinitionError,
    TargetingFailure,
)
from worldsim.sites.task_evidence import _matches_origin


def _definition_error(message: str) -> ValueError:
    return SiteTargetingDefinitionError(message)


@dataclass(frozen=True)
class TargetCandidate:
    """Validated evidence supplied by an intent/probe resolver.

    ``kind`` may be either a Site's local kind or its prefixed compatibility
    kind.  ``probe_query`` lets a Site reject an incoherent API/kind pair.
    ``fallback_url`` is diagnostic input only and is never used to construct a
    target when reconstruction fails.
    """

    kind: str
    anchors: Mapping[str, Any]
    probe_query: Mapping[str, Any] = field(default_factory=dict)
    evidence_url: str | None = None
    fallback_url: str | None = None
    layer: str = "L3"

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip()
        if not kind:
            raise _definition_error("candidate kind is required")
        if not isinstance(self.anchors, Mapping):
            raise _definition_error("candidate anchors must be a mapping")
        if not isinstance(self.probe_query, Mapping):
            raise _definition_error("candidate probe_query must be a mapping")
        layer = str(self.layer or "").strip().upper()
        if not layer:
            raise _definition_error("candidate layer is required")
        try:
            anchors = MappingProxyType(dict(self.anchors))
            probe_query = MappingProxyType(dict(self.probe_query))
        except Exception as exc:
            raise _definition_error(
                f"candidate mappings must be materializable: {type(exc).__name__}"
            ) from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "probe_query", probe_query)
        object.__setattr__(self, "layer", layer)
        for name in ("evidence_url", "fallback_url"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, str(value).strip() or None)


@dataclass(frozen=True)
class SourceListing:
    """Site-owned provenance for the listing that yielded a candidate."""

    kind: str
    start_url: str

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip()
        start_url = str(self.start_url or "").strip()
        if not kind or not start_url:
            raise _definition_error("source listing kind and URL are required")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "start_url", start_url)


def _adapter_validation(
    adapter: Any,
    kind: str,
    probe_query: Mapping[str, Any],
    anchors: Mapping[str, Any],
    context: Any,
) -> tuple[str, str] | None:
    validate_candidate = getattr(adapter, "validate_candidate", None)
    if not callable(validate_candidate):
        return None
    try:
        result = validate_candidate(kind, probe_query, anchors, context)
    except Exception as exc:
        return (
            "adapter_error",
            f"Site adapter validate_candidate failed ({type(exc).__name__})",
        )
    if result is None:
        return None
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and all(isinstance(value, str) and value.strip() for value in result)
    ):
        return result
    return ("invalid_candidate", "Site adapter returned an invalid candidate error")


def _is_absolute_http_url(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def validate_probe(
    *,
    site: str,
    adapter: Any,
    context: Any,
    route_for_identifier: Callable[[str, Mapping[str, Any]], Any],
    kind: str,
    probe_query: Mapping[str, Any],
) -> Any:
    """Return a structured failure for an incoherent API/kind pair."""

    if not isinstance(probe_query, Mapping):
        return TargetingFailure(
            site,
            "invalid_probe_query",
            "probe_query must be a mapping",
            layer="L3",
        )
    try:
        probe_snapshot = dict(probe_query)
    except Exception as exc:
        return TargetingFailure(
            site,
            "invalid_probe_query",
            f"probe_query could not be read: {type(exc).__name__}",
            layer="L3",
        )
    route = route_for_identifier(str(kind or ""), {})
    if route is None:
        return TargetingFailure(
            site,
            "unknown_route",
            f"no canonical route for resource kind {kind!r}",
            layer="L3",
            evidence={"kind": kind},
        )
    validation = _adapter_validation(adapter, route.kind, probe_snapshot, {}, context)
    if validation is None:
        return None
    reason, message = validation
    return TargetingFailure(
        site,
        reason,
        message,
        layer="L3",
        evidence={"kind": kind, "probe_query": probe_snapshot},
    )


def materialize_candidate(
    *,
    site: str,
    benchmark: str,
    adapter: Any,
    context: Any,
    route_for_identifier: Callable[[str, Mapping[str, Any]], Any],
    candidate: TargetCandidate,
) -> Any:
    """Materialize a candidate using only the bound Site's route grammar."""

    if not isinstance(candidate, TargetCandidate):
        return TargetingFailure(
            site,
            "invalid_candidate",
            "candidate must be a TargetCandidate",
            layer="L3",
        )
    if candidate.layer != "L3":
        return TargetingFailure(
            site,
            "unsupported_resolution_layer",
            f"materialize only accepts L3 candidates, got {candidate.layer!r}",
            layer=candidate.layer,
        )
    supported = getattr(adapter, "supported_benchmarks", frozenset())
    if benchmark not in supported:
        return TargetingFailure(
            site,
            "unsupported_benchmark",
            f"benchmark {benchmark!r} is not supported by this Site",
            layer="L3",
            evidence={"kind": candidate.kind, "anchors": dict(candidate.anchors)},
        )
    origin = context.site_origin()
    if origin is None:
        return TargetingFailure(
            site,
            "missing_origin",
            "materialize requires an explicit site origin or placeholder",
            layer="L3",
            evidence={"kind": candidate.kind, "anchors": dict(candidate.anchors)},
        )
    if candidate.evidence_url is not None and (
        not _is_absolute_http_url(candidate.evidence_url)
        or not _matches_origin(candidate.evidence_url, origin)
    ):
        return TargetingFailure(
            site,
            "foreign_origin",
            "candidate evidence_url is not an absolute URL on the bound Site origin",
            layer="L3",
            evidence={"kind": candidate.kind},
        )
    route = route_for_identifier(candidate.kind, candidate.anchors)
    if route is None:
        return TargetingFailure(
            site,
            "unknown_route",
            f"no canonical route for resource kind {candidate.kind!r}",
            layer="L3",
            evidence={"kind": candidate.kind, "anchors": dict(candidate.anchors)},
            evidence_url=candidate.evidence_url,
        )
    validation = _adapter_validation(
        adapter, route.kind, candidate.probe_query, candidate.anchors, context
    )
    if validation is not None:
        reason, message = validation
        return TargetingFailure(
            site,
            reason,
            message,
            layer="L3",
            evidence={
                "kind": candidate.kind,
                "anchors": dict(candidate.anchors),
                "probe_query": dict(candidate.probe_query),
            },
            evidence_url=candidate.evidence_url,
        )
    try:
        reconstructed = adapter.reconstruct(route.kind, candidate.anchors, context)
    except Exception as exc:
        return TargetingFailure(
            site,
            "adapter_error",
            f"Site adapter reconstruct failed ({type(exc).__name__})",
            layer="L3",
            evidence={"kind": candidate.kind, "anchors": dict(candidate.anchors)},
            evidence_url=candidate.evidence_url,
        )
    if not reconstructed:
        return TargetingFailure(
            site,
            "missing_anchor",
            f"resource kind {candidate.kind!r} has insufficient anchors for canonical route",
            layer="L3",
            evidence={"kind": candidate.kind, "anchors": dict(candidate.anchors)},
            evidence_url=candidate.evidence_url,
        )
    if not _is_absolute_http_url(reconstructed):
        return TargetingFailure(
            site,
            "invalid_target_url",
            "adapter reconstruction must be an absolute http(s) URL",
            layer="L3",
            evidence={"kind": candidate.kind, "anchors": dict(candidate.anchors)},
            evidence_url=candidate.evidence_url,
        )
    if not _matches_origin(reconstructed, origin):
        return TargetingFailure(
            site,
            "foreign_origin",
            "adapter reconstruction did not stay on the bound Site origin",
            layer="L3",
            evidence={
                "kind": candidate.kind,
                "anchors": dict(candidate.anchors),
                "reconstructed": reconstructed,
            },
            evidence_url=candidate.evidence_url,
        )
    return ResolvedTarget(
        site=site,
        kind=route.kind,
        anchors=dict(candidate.anchors),
        start_url_resolved=reconstructed,
        layer="L3",
        canonical_route=route,
        evidence_url=candidate.evidence_url,
    )


def source_listing_for_candidate(
    *,
    site: str,
    adapter: Any,
    context: Any,
    route_for_identifier: Callable[[str, Mapping[str, Any]], Any],
    candidate: TargetCandidate,
) -> Any:
    """Resolve adapter-owned source-listing facts for a candidate."""

    if not isinstance(candidate, TargetCandidate):
        return TargetingFailure(
            site,
            "invalid_candidate",
            "candidate must be a TargetCandidate",
            layer="L3",
        )
    route = route_for_identifier(candidate.kind, candidate.anchors)
    if route is None:
        return TargetingFailure(
            site,
            "unknown_route",
            f"no canonical route for resource kind {candidate.kind!r}",
            layer="L3",
            evidence={"kind": candidate.kind},
        )
    source_listing = getattr(adapter, "source_listing", None)
    if not callable(source_listing):
        return None
    try:
        result = source_listing(route.kind, candidate.probe_query, candidate.anchors, context)
    except Exception as exc:
        return TargetingFailure(
            site,
            "adapter_error",
            f"Site adapter source_listing failed ({type(exc).__name__})",
            layer="L3",
            evidence={"kind": candidate.kind},
        )
    if result is None:
        return None
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], str)
        or not isinstance(result[1], str)
    ):
        return TargetingFailure(
            site,
            "invalid_source_listing",
            "Site adapter returned an invalid source listing",
            layer="L3",
            evidence={"kind": candidate.kind},
        )
    source_kind, source_url = result
    source_route = route_for_identifier(source_kind, {})
    origin = context.site_origin()
    if (
        source_route is None
        or origin is None
        or not _is_absolute_http_url(source_url)
        or not _matches_origin(source_url, origin)
    ):
        return TargetingFailure(
            site,
            "invalid_source_listing",
            "source listing is unknown or outside the bound Site origin",
            layer="L3",
            evidence={"kind": source_kind, "url": source_url},
        )
    return SourceListing(
        kind=source_route.compatibility_kind or source_route.kind,
        start_url=source_url,
    )


__all__ = [
    "SourceListing",
    "TargetCandidate",
    "materialize_candidate",
    "source_listing_for_candidate",
    "validate_probe",
]
