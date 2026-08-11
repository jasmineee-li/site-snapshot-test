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
from worldsim.sites.listing_resolution import (
    ListingItemCandidate,
    ListingSiteAdapter,
    materialize_listing_entry,
)
from worldsim.sites.profile_binding import BoundProfileRoutes
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


class BoundSite(BoundProfileRoutes):
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
            bound = BoundSite(
                adapter,
                TargetingContext(benchmark=validation_benchmark, site=site),
            )
            listing_members = (
                "expandable_listing_kinds",
                "listing_item_kind",
                "listing_item_anchors",
            )
            if any(hasattr(adapter, name) for name in listing_members):
                if not isinstance(adapter, ListingSiteAdapter):
                    missing = [name for name in listing_members if not hasattr(adapter, name)]
                    raise SiteTargetingDefinitionError(
                        f"Site adapter {site!r} has an incomplete L4 listing capability: "
                        f"missing {missing!r}"
                    )
                expandable = adapter.expandable_listing_kinds
                if not isinstance(expandable, frozenset) or not all(
                    isinstance(kind, str) and kind.strip() for kind in expandable
                ):
                    raise SiteTargetingDefinitionError(
                        f"Site adapter {site!r} expandable_listing_kinds must be a "
                        "frozenset of non-empty local kinds"
                    )
                invalid = [
                    kind
                    for kind in sorted(expandable)
                    if not bound.is_listing_kind(kind) or not bound.is_expandable_listing_kind(kind)
                ]
                if invalid:
                    raise SiteTargetingDefinitionError(
                        f"Site adapter {site!r} has invalid expandable listing kinds: {invalid!r}"
                    )
            normalized[site] = adapter
        if not normalized:
            raise SiteTargetingDefinitionError("catalog requires at least one Site adapter")
        self._adapters = normalized

    @property
    def sites(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def site_for_task(
        self,
        task: Mapping[str, Any],
        *,
        fallback_kind: str | None = None,
        benchmark: str = "webarena_verified",
    ) -> str | None:
        """Resolve a task Site without guessing through malformed metadata.

        An explicit, valid task Site wins.  When task metadata omits a Site,
        ``fallback_kind`` may identify one unique route owner for legacy
        resource records.  Invalid or ambiguous task metadata returns
        ``None`` rather than silently selecting that fallback.
        """

        if not isinstance(task, Mapping):
            return None
        metadata = _task_site_metadata(task)
        if metadata.failure_reason is not None:
            return None
        if metadata.task_site is not None:
            return metadata.task_site if metadata.task_site in self._adapters else None
        if fallback_kind is None:
            return None
        return self.site_for_kind(fallback_kind, benchmark=benchmark)

    def site_for_kind(self, kind: str, *, benchmark: str = "webarena_verified") -> str | None:
        """Return the unique Site owner for a local or compatibility kind.

        This lookup exists for direct Phase 2 compatibility callers whose
        resource record predates explicit Site metadata.  Ambiguous or
        unsupported kinds fail closed instead of selecting an adapter by
        prefix.
        """

        normalized_kind = str(kind or "").strip()
        if not normalized_kind:
            return None
        owners: list[str] = []
        try:
            normalized_benchmark = TargetingContext(
                benchmark=benchmark,
                site=next(iter(self._adapters)),
            ).benchmark
        except (StopIteration, SiteTargetingDefinitionError):
            return None
        for site, adapter in self._adapters.items():
            if normalized_benchmark not in getattr(adapter, "supported_benchmarks", ()):
                continue
            try:
                routes = adapter.routes(TargetingContext(benchmark=normalized_benchmark, site=site))
            except Exception:
                continue
            if any(normalized_kind in {route.kind, route.compatibility_kind} for route in routes):
                owners.append(site)
        return owners[0] if len(owners) == 1 else None

    def is_expandable_listing_kind(
        self,
        kind: str,
        *,
        benchmark: str = "webarena_verified",
    ) -> bool:
        """Return whether one registered Site explicitly admits ``kind`` for L4."""

        site = self.site_for_kind(kind, benchmark=benchmark)
        if site is None:
            return False
        try:
            return self.bind(benchmark=benchmark, site=site).is_expandable_listing_kind(kind)
        except SiteTargetingDefinitionError:
            return False

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
    "ListingItemCandidate",
    "ListingSiteAdapter",
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
