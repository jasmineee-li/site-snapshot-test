"""One Site adapter bound to one immutable targeting context.

``BoundSite`` is the seam every caller reaches after binding a benchmark and a
Site: it owns route classification, L1/L2 resolution, candidate
materialization, and the optional probe/listing adapter hooks. Profile routes,
read-surface planning, and readback come from the mixins it composes, so
core-surface and active-carrier policy stay Site-owned and a Site without the
capability binds closed. The catalog that constructs it lives in
``warp_taskgen.sites.catalog``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.sites.candidate_resolution import (
    SourceListing,
    TargetCandidate,
    materialize_candidate,
    source_listing_for_candidate,
    validate_probe,
)
from warp_taskgen.sites.contracts import (
    CanonicalRoute,
    ResolvedTarget,
    SiteAdapter,
    SiteTargetingDefinitionError,
    TargetingContext,
    TargetingFailure,
)
from warp_taskgen.sites.listing_resolution import (
    ListingItemCandidate,
    ListingSiteAdapter,
    materialize_listing_entry,
)
from warp_taskgen.sites.probe_hooks import (
    ListingTaskProjectPathCapability,
    ProbeItemAnchorsCapability,
)
from warp_taskgen.sites.profile_binding import BoundProfileRoutes
from warp_taskgen.sites.read_surface import BoundReadSurface
from warp_taskgen.sites.readback import BoundReadback
from warp_taskgen.sites.task_evidence import (
    _iter_eval_urls,
    _iter_start_urls,
    _matches_origin,
    _metadata_failure,
    _normalise_url,
    _task_site_metadata,
)


class BoundSite(BoundProfileRoutes, BoundReadSurface, BoundReadback):
    """A Site adapter bound to one immutable targeting context."""

    def __init__(self, adapter: SiteAdapter, context: TargetingContext) -> None:
        self._adapter = adapter
        self._context = context
        routes = tuple(adapter.routes(context))
        route_ids = [route.id for route in routes]
        if len(set(route_ids)) != len(route_ids):
            raise SiteTargetingDefinitionError(
                f"duplicate route ids for site {context.site!r}: {route_ids!r}"
            )
        if any(route.site != context.site for route in routes):
            raise SiteTargetingDefinitionError(
                f"route site does not match bound site {context.site!r}"
            )
        self._routes = tuple(sorted(routes, key=lambda route: route.id))
        self._routes_by_kind: dict[str, tuple[CanonicalRoute, ...]] = {}
        self._routes_by_compatibility_kind: dict[str, tuple[CanonicalRoute, ...]] = {}
        for route in self._routes:
            self._routes_by_kind.setdefault(route.kind, ())
            self._routes_by_kind[route.kind] += (route,)
            if route.compatibility_kind:
                self._routes_by_compatibility_kind.setdefault(route.compatibility_kind, ())
                self._routes_by_compatibility_kind[route.compatibility_kind] += (route,)

    def routes(self) -> tuple[CanonicalRoute, ...]:
        return self._routes

    def resolve(
        self,
        task: Mapping[str, Any],
        *,
        allow_layers: tuple[str, ...] = ("L1", "L2"),
    ) -> ResolvedTarget | TargetingFailure:
        if not isinstance(task, Mapping):
            return TargetingFailure(self._context.site, "invalid_task", "task must be a mapping")
        unsupported_layers = set(allow_layers) - {"L1", "L2"}
        if unsupported_layers:
            return TargetingFailure(
                self._context.site,
                "unsupported_resolution_layer",
                f"Site Targeting only owns L1/L2; unsupported={sorted(unsupported_layers)!r}",
            )
        metadata_failure = _metadata_failure(task)
        if metadata_failure is not None:
            reason, message = metadata_failure
            return TargetingFailure(self._context.site, reason, message)
        adapter_failure = self._adapter.validate_task(task)
        if adapter_failure is not None:
            reason, message = adapter_failure
            return TargetingFailure(self._context.site, reason, message)
        supported = getattr(self._adapter, "supported_benchmarks", frozenset())
        if self._context.benchmark not in supported:
            return TargetingFailure(
                self._context.site,
                "unsupported_benchmark",
                f"benchmark {self._context.benchmark!r} is not supported by this Site",
            )
        task_site = _task_site_metadata(task).task_site
        if task_site != self._context.site:
            return TargetingFailure(
                self._context.site,
                "unsupported_site",
                f"task site {task_site!r} is not bound site {self._context.site!r}",
            )
        origin = self._context.site_origin()
        if origin is None:
            return TargetingFailure(
                self._context.site,
                "missing_origin",
                "resolve requires an explicit site origin or placeholder",
            )
        resolved_start = next(
            (
                resolved
                for raw in _iter_start_urls(task)
                if (resolved := _normalise_url(raw, self._context.placeholders))
                and _matches_origin(resolved, origin)
            ),
            None,
        )
        if "L1" in allow_layers:
            for raw in _iter_eval_urls(task):
                resolved = _normalise_url(raw, self._context.placeholders)
                if not resolved or not _matches_origin(resolved, origin):
                    continue
                hit = self._adapter.match(resolved, task, self._context)
                if hit is not None:
                    return self._resolved(hit, resolved, resolved_start, "L1")
        if "L2" in allow_layers and resolved_start:
            hit = self._adapter.match(resolved_start, task, self._context)
            if hit is not None:
                return self._resolved(hit, resolved_start, resolved_start, "L2")
        return TargetingFailure(
            self._context.site,
            "unresolved_evidence",
            "L1+L2 found no concrete resource; intent-only task pending L3",
            pending_layer="L3",
            evidence_url=resolved_start,
        )

    def validate_probe(
        self,
        kind: str,
        probe_query: Mapping[str, Any],
    ) -> TargetingFailure | None:
        return validate_probe(
            site=self._context.site,
            adapter=self._adapter,
            context=self._context,
            route_for_identifier=self._route_for_identifier,
            kind=kind,
            probe_query=probe_query,
        )

    def materialize(self, candidate: TargetCandidate) -> ResolvedTarget | TargetingFailure:
        return materialize_candidate(
            site=self._context.site,
            benchmark=self._context.benchmark,
            adapter=self._adapter,
            context=self._context,
            route_for_identifier=self._route_for_identifier,
            candidate=candidate,
        )

    def source_listing(
        self,
        candidate: TargetCandidate,
    ) -> SourceListing | TargetingFailure | None:
        return source_listing_for_candidate(
            site=self._context.site,
            adapter=self._adapter,
            context=self._context,
            route_for_identifier=self._route_for_identifier,
            candidate=candidate,
        )

    def materialize_listing_entry(
        self,
        candidate: ListingItemCandidate,
    ) -> ResolvedTarget | TargetingFailure:
        """Materialize one raw listing row through this bound Site."""

        return materialize_listing_entry(
            site=self._context.site,
            benchmark=self._context.benchmark,
            adapter=self._adapter,
            context=self._context,
            route_for_identifier=self._route_for_identifier,
            candidate=candidate,
        )

    def is_expandable_listing_kind(self, kind: str) -> bool:
        """Return whether ``kind`` is an adapter-approved L4 listing."""

        route = self._route_for_identifier(str(kind or ""), {})
        if route is None:
            return False
        if not isinstance(self._adapter, ListingSiteAdapter):
            return False
        return route.kind in self._adapter.expandable_listing_kinds

    def supports_benchmark(self) -> bool:
        """Return whether this bound Site declares the requested Benchmark."""

        return self._context.benchmark in getattr(
            self._adapter,
            "supported_benchmarks",
            frozenset(),
        )

    def has_materialization_origin(self) -> bool:
        """Return whether canonical reconstruction has an explicit bound origin."""

        return self._context.site_origin() is not None

    def is_listing_kind(self, kind: str) -> bool:
        """Compatibility alias for callers that only need route classification."""

        route = self._route_for_identifier(str(kind or ""), {})
        if route is None:
            return False
        try:
            return bool(self._adapter.is_listing(route.kind))
        except Exception:
            return False

    def reconstruct(self, kind: str, anchors: Mapping[str, Any]) -> str | None:
        """Reconstruct the canonical start URL for a local or compatibility kind.

        Unknown kinds and insufficient anchors yield ``None``; the bound
        context supplies the origin, so no deployment host is guessed.
        """

        route = self._route_for_identifier(str(kind or ""), anchors)
        if route is None:
            return None
        return self._adapter.reconstruct(route.kind, anchors, self._context)

    def probe_item_anchors(
        self,
        item: Mapping[str, Any],
        *,
        kind_hint: str,
        forum_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Project one L3 probe row through the adapter's optional hook."""

        if not isinstance(self._adapter, ProbeItemAnchorsCapability):
            return None
        anchors = self._adapter.probe_item_anchors(item, kind_hint=kind_hint, forum_name=forum_name)
        return dict(anchors) if isinstance(anchors, Mapping) else None

    def project_path_from_listing_task(
        self,
        instruction: str,
        *,
        resolved_start: str | None,
    ) -> str | None:
        """Return the project path a listing task names, or ``None`` without the hook."""

        if not isinstance(self._adapter, ListingTaskProjectPathCapability):
            return None
        return self._adapter.project_path_from_listing_task(
            instruction, resolved_start=resolved_start
        )

    def _resolved(
        self,
        hit: tuple[str, dict[str, Any]],
        evidence_url: str,
        fallback_url: str | None,
        layer: str,
    ) -> ResolvedTarget | TargetingFailure:
        kind, anchors = hit
        route = self._route_for(kind, anchors)
        if route is None:
            return TargetingFailure(
                self._context.site,
                "unknown_route",
                f"no canonical route for resource kind {kind!r}",
                layer=layer,
                evidence_url=evidence_url,
                evidence={"kind": kind, "anchors": dict(anchors)},
            )
        reconstructed = self._adapter.reconstruct(kind, anchors, self._context)
        if self._adapter.is_listing(kind):
            start_url = self._adapter.listing_start_url(kind, evidence_url, fallback_url)
        else:
            start_url = reconstructed
        if not start_url:
            return TargetingFailure(
                self._context.site,
                "missing_anchor",
                f"resource kind {kind!r} has insufficient anchors for canonical route",
                layer=layer,
                evidence_url=evidence_url,
                evidence={"kind": kind, "anchors": dict(anchors)},
            )
        return ResolvedTarget(
            site=self._context.site,
            kind=kind,
            anchors=dict(anchors),
            start_url_resolved=start_url,
            layer=layer,
            canonical_route=route,
            evidence_url=evidence_url,
        )

    def _route_for(self, kind: str, anchors: Mapping[str, Any]) -> CanonicalRoute | None:
        routes = self._routes_by_kind.get(kind, ())
        if len(routes) == 1:
            return routes[0]
        if not routes:
            return None
        matches: list[CanonicalRoute] = []
        for route in routes:
            required = {
                key
                for example in route.anchor_examples
                for key in example
                if key not in {"start_url", "route_variant", "existing_comment_count"}
            }
            if required and required.issubset(anchors):
                matches.append(route)
        return matches[0] if len(matches) == 1 else None

    def _route_for_identifier(self, kind: str, anchors: Mapping[str, Any]) -> CanonicalRoute | None:
        """Resolve either a local or compatibility kind without guessing."""

        local = self._route_for(kind, anchors)
        if local is not None:
            return local
        routes = self._routes_by_compatibility_kind.get(kind, ())
        if len(routes) == 1:
            return routes[0]
        if not routes:
            return None
        matches: list[CanonicalRoute] = []
        for route in routes:
            required = {
                key
                for example in route.anchor_examples
                for key in example
                if key not in {"start_url", "route_variant", "existing_comment_count"}
            }
            if required and required.issubset(anchors):
                matches.append(route)
        return matches[0] if len(matches) == 1 else None


__all__ = ["BoundSite"]
