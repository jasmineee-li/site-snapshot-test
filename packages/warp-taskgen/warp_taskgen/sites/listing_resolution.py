"""Pure L4 listing-entry contracts for the Site Targeting seam.

Phase 2 still owns listing probes, encounter policy, and artifact composition.
This module only translates one already-fetched listing row through the bound
Site's route grammar.  It deliberately has no HTTP, browser, editor, or
admission dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from warp_taskgen.sites.candidate_resolution import (
    _is_absolute_http_url,
    _materialize_route_target,
)
from warp_taskgen.sites.contracts import (
    ResolvedTarget,
    SiteTargetingDefinitionError,
    TargetingFailure,
)
from warp_taskgen.sites.task_evidence import _matches_origin


def _definition_error(message: str) -> ValueError:
    return SiteTargetingDefinitionError(message)


@runtime_checkable
class ListingSiteAdapter(Protocol):
    """Optional Site capability for deterministic listing-entry expansion."""

    expandable_listing_kinds: frozenset[str]

    def listing_item_kind(
        self,
        source_kind: str,
        item_kind: str,
        context: Any,
    ) -> str | None: ...

    def listing_item_anchors(
        self,
        source_kind: str,
        item_kind: str,
        payload: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class ListingItemCandidate:
    """One raw row returned by a Site-owned listing probe.

    ``source_kind`` and ``item_kind`` accept either the Site-local or the
    historical prefixed spelling.  The top-level ``payload`` mapping is
    snapshotted so callers cannot replace its fields after it crosses the
    Site seam.
    ``evidence_url`` is the already-known listing page and is diagnostic only;
    it is never used as a fallback for item reconstruction.
    """

    source_kind: str
    item_kind: str
    payload: Mapping[str, Any]
    evidence_url: str | None = None
    layer: str = "L4"

    def __post_init__(self) -> None:
        source_kind = str(self.source_kind or "").strip()
        item_kind = str(self.item_kind or "").strip()
        layer = str(self.layer or "").strip().upper()
        if not source_kind:
            raise _definition_error("listing candidate source_kind is required")
        if not item_kind:
            raise _definition_error("listing candidate item_kind is required")
        if not isinstance(self.payload, Mapping):
            raise _definition_error("listing candidate payload must be a mapping")
        if not layer:
            raise _definition_error("listing candidate layer is required")
        try:
            payload = MappingProxyType(dict(self.payload))
        except Exception as exc:
            raise _definition_error(
                f"listing candidate payload must be materializable: {type(exc).__name__}"
            ) from exc
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "item_kind", item_kind)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "layer", layer)
        if self.evidence_url is not None:
            object.__setattr__(self, "evidence_url", str(self.evidence_url).strip() or None)


def _adapter_item_kind(
    adapter: Any,
    candidate: ListingItemCandidate,
    context: Any,
) -> tuple[str | None, TargetingFailure | None]:
    hook = getattr(adapter, "listing_item_kind", None)
    if not callable(hook):
        return None, TargetingFailure(
            context.site,
            "unsupported_listing_entry",
            "Site adapter does not validate listing source/item kinds",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    try:
        item_kind = hook(candidate.source_kind, candidate.item_kind, context)
    except Exception as exc:
        return None, TargetingFailure(
            context.site,
            "adapter_error",
            f"Site adapter listing_item_kind failed ({type(exc).__name__})",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    if item_kind is None:
        return None, TargetingFailure(
            context.site,
            "unknown_route",
            f"listing source {candidate.source_kind!r} cannot yield item kind "
            f"{candidate.item_kind!r}",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    try:
        item_kind = str(item_kind).strip()
    except Exception as exc:
        return None, TargetingFailure(
            context.site,
            "invalid_candidate",
            f"Site adapter returned an unreadable listing item kind: {type(exc).__name__}",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    if not item_kind:
        return None, TargetingFailure(
            context.site,
            "unknown_route",
            "Site adapter returned an empty listing item kind",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    return item_kind, None


def _adapter_item_anchors(
    adapter: Any,
    candidate: ListingItemCandidate,
    item_kind: str,
    context: Any,
) -> tuple[Mapping[str, Any] | None, TargetingFailure | None]:
    hook = getattr(adapter, "listing_item_anchors", None)
    if not callable(hook):
        return None, TargetingFailure(
            context.site,
            "unsupported_listing_entry",
            "Site adapter does not project listing item anchors",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    try:
        anchors = hook(candidate.source_kind, item_kind, candidate.payload, context)
    except Exception as exc:
        return None, TargetingFailure(
            context.site,
            "adapter_error",
            f"Site adapter listing_item_anchors failed ({type(exc).__name__})",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    if not isinstance(anchors, Mapping):
        return None, TargetingFailure(
            context.site,
            "invalid_candidate",
            "Site adapter listing_item_anchors must return a mapping",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    if not anchors:
        return None, TargetingFailure(
            context.site,
            "missing_anchor",
            "listing item has no reconstructable anchors",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    try:
        return dict(anchors), None
    except Exception as exc:
        return None, TargetingFailure(
            context.site,
            "invalid_candidate",
            f"listing item anchors could not be read: {type(exc).__name__}",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": item_kind,
            },
            evidence_url=candidate.evidence_url,
        )


def materialize_listing_entry(
    *,
    site: str,
    benchmark: str,
    adapter: Any,
    context: Any,
    route_for_identifier: Callable[[str, Mapping[str, Any]], Any],
    candidate: ListingItemCandidate,
) -> ResolvedTarget | TargetingFailure:
    """Materialize one listing row through a bound Site adapter.

    The helper performs no policy decisions and never promotes an evidence or
    listing URL into an item target.  A Site must explicitly accept the
    source/item-kind pair and project raw row anchors before the common route
    reconstruction checks run.
    """

    if not isinstance(candidate, ListingItemCandidate):
        return TargetingFailure(
            site,
            "invalid_candidate",
            "candidate must be a ListingItemCandidate",
            layer="L4",
        )
    if candidate.layer != "L4":
        return TargetingFailure(
            site,
            "unsupported_resolution_layer",
            f"materialize_listing_entry only accepts L4 candidates, got {candidate.layer!r}",
            layer=candidate.layer,
        )
    supported = getattr(adapter, "supported_benchmarks", frozenset())
    if benchmark not in supported:
        return TargetingFailure(
            site,
            "unsupported_benchmark",
            f"benchmark {benchmark!r} is not supported by this Site",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    origin = context.site_origin()
    if origin is None:
        return TargetingFailure(
            site,
            "missing_origin",
            "materialize_listing_entry requires an explicit site origin or placeholder",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    if candidate.evidence_url is not None and (
        not _is_absolute_http_url(candidate.evidence_url)
        or not _matches_origin(candidate.evidence_url, origin)
    ):
        return TargetingFailure(
            site,
            "foreign_origin",
            "candidate evidence_url is not an absolute URL on the bound Site origin",
            layer="L4",
            evidence={"source_kind": candidate.source_kind},
        )
    source_route = route_for_identifier(candidate.source_kind, {})
    if source_route is None:
        return TargetingFailure(
            site,
            "unknown_route",
            f"no canonical source listing route for kind {candidate.source_kind!r}",
            layer="L4",
            evidence={"source_kind": candidate.source_kind},
            evidence_url=candidate.evidence_url,
        )
    try:
        source_is_listing = bool(adapter.is_listing(source_route.kind))
    except Exception as exc:
        return TargetingFailure(
            site,
            "adapter_error",
            f"Site adapter is_listing failed ({type(exc).__name__})",
            layer="L4",
            evidence={"source_kind": candidate.source_kind},
            evidence_url=candidate.evidence_url,
        )
    if not source_is_listing:
        return TargetingFailure(
            site,
            "invalid_source_listing",
            f"source kind {candidate.source_kind!r} is not a listing route",
            layer="L4",
            evidence={"source_kind": candidate.source_kind},
            evidence_url=candidate.evidence_url,
        )
    item_kind, failure = _adapter_item_kind(adapter, candidate, context)
    if failure is not None:
        return failure
    assert item_kind is not None
    item_route = route_for_identifier(item_kind, {})
    if item_route is None:
        return TargetingFailure(
            site,
            "unknown_route",
            f"no canonical item route for kind {item_kind!r}",
            layer="L4",
            evidence={
                "source_kind": candidate.source_kind,
                "item_kind": candidate.item_kind,
            },
            evidence_url=candidate.evidence_url,
        )
    anchors, failure = _adapter_item_anchors(adapter, candidate, item_kind, context)
    if failure is not None:
        return failure
    assert anchors is not None
    target = _materialize_route_target(
        site=site,
        benchmark=benchmark,
        adapter=adapter,
        context=context,
        route_for_identifier=route_for_identifier,
        kind=item_kind,
        anchors=anchors,
        probe_query={},
        evidence_url=candidate.evidence_url,
        layer="L4",
    )
    return target


__all__ = ["ListingItemCandidate", "ListingSiteAdapter", "materialize_listing_entry"]
