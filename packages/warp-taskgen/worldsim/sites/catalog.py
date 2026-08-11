"""The deterministic Site Targeting catalog.

The catalog binds a benchmark/profile projection to a named Site and delegates
URL grammar to that Site's feature module. It remains the compatibility facade
for the contracts and task-evidence modules while callers migrate to the
feature-owned seams.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from worldsim.sites.candidate_resolution import (
    SourceListing,
    TargetCandidate,
    materialize_candidate,
    source_listing_for_candidate,
    validate_probe,
)
from worldsim.sites.contracts import (
    CanonicalRoute,
    ResolvedTarget,
    SiteAdapter,
    SiteTargetingDefinitionError,
    TargetingContext,
    TargetingFailure,
)
from worldsim.sites.task_evidence import (
    _MISSING,  # noqa: F401
    _iter_eval_urls,
    _iter_start_urls,
    _matches_origin,
    _metadata_failure,
    _normalise_site_token,  # noqa: F401
    _normalise_url,
    _path_and_query,
    _site_kind_for_task,
    _strip_json_suffix,
    _strip_regex_anchors,
    _task_site_metadata,
    _TaskSiteMetadata,  # noqa: F401
    _url_with_expected_query_params,
)


class BoundSite:
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


class SiteCatalog:
    """Explicit catalog of Site Targeting adapters."""

    def __init__(
        self,
        adapters: Mapping[str, SiteAdapter] | Iterable[SiteAdapter] | None = None,
    ) -> None:
        if adapters is None:
            from worldsim.sites.gitlab import GitLabSite
            from worldsim.sites.reddit import RedditSite

            adapters = (GitLabSite(), RedditSite())
        if isinstance(adapters, Mapping):
            candidates = list(adapters.items())
        else:
            candidates = [(adapter.site, adapter) for adapter in adapters]
        normalized: dict[str, SiteAdapter] = {}
        for key, adapter in candidates:
            site = str(key or getattr(adapter, "site", "")).strip().lower()
            adapter_site = str(getattr(adapter, "site", "")).strip().lower()
            if not site or site != adapter_site:
                raise SiteTargetingDefinitionError("adapter key and site identity must agree")
            if site in normalized:
                raise SiteTargetingDefinitionError(f"duplicate Site adapter {site!r}")
            supported = getattr(adapter, "supported_benchmarks", None)
            if not isinstance(supported, frozenset) or not supported:
                raise SiteTargetingDefinitionError(
                    f"Site adapter {site!r} needs supported_benchmarks"
                )
            validate = getattr(adapter, "validate", None)
            if not callable(validate):
                raise SiteTargetingDefinitionError(f"Site adapter {site!r} has no validate()")
            required_methods = (
                "validate_task",
                "routes",
                "match",
                "reconstruct",
                "is_listing",
                "listing_start_url",
            )
            missing_methods = [
                name for name in required_methods if not callable(getattr(adapter, name, None))
            ]
            if missing_methods:
                raise SiteTargetingDefinitionError(
                    f"Site adapter {site!r} is missing methods: {missing_methods!r}"
                )
            validate()
            validation_benchmark = sorted(supported)[0]
            BoundSite(
                adapter,
                TargetingContext(benchmark=validation_benchmark, site=site),
            )
            normalized[site] = adapter
        if not normalized:
            raise SiteTargetingDefinitionError("catalog requires at least one Site adapter")
        self._adapters = normalized

    @property
    def sites(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def bind(
        self,
        context: TargetingContext | Mapping[str, Any] | None = None,
        *,
        benchmark: str = "webarena_verified",
        site: str | None = None,
        profile: Mapping[str, Any] | None = None,
        origin: str | None = None,
        placeholders: Mapping[str, str] | None = None,
    ) -> BoundSite:
        targeting = TargetingContext.from_input(
            context,
            benchmark=benchmark,
            site=site,
            profile=profile,
            origin=origin,
            placeholders=placeholders,
        )
        adapter = self._adapters.get(targeting.site)
        if adapter is None:
            raise SiteTargetingDefinitionError(f"unknown Site adapter {targeting.site!r}")
        return BoundSite(adapter, targeting)


_DEFAULT_CATALOG: SiteCatalog | None = None


def default_catalog() -> SiteCatalog:
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = SiteCatalog()
    return _DEFAULT_CATALOG


__all__ = [
    "BoundSite",
    "CanonicalRoute",
    "ResolvedTarget",
    "SiteAdapter",
    "SiteCatalog",
    "SiteTargetingDefinitionError",
    "SourceListing",
    "TargetCandidate",
    "TargetingContext",
    "TargetingFailure",
    "_iter_eval_urls",
    "_iter_start_urls",
    "_normalise_url",
    "_path_and_query",
    "_site_kind_for_task",
    "_strip_json_suffix",
    "_strip_regex_anchors",
    "_url_with_expected_query_params",
    "default_catalog",
]
