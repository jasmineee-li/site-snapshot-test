"""The deterministic Site Targeting catalog.

The catalog binds a benchmark/profile projection to a named Site and delegates
URL grammar to that Site's feature module. It remains the compatibility facade
for the contracts and task-evidence modules while callers migrate to the
feature-owned seams.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from warp_taskgen.sites.bound_site import BoundSite
from warp_taskgen.sites.candidate_resolution import (
    SourceListing,
    TargetCandidate,
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
)
from warp_taskgen.sites.task_evidence import (
    _MISSING,  # noqa: F401
    _iter_eval_urls,
    _iter_start_urls,
    _matches_origin,  # noqa: F401
    _metadata_failure,  # noqa: F401
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


class SiteCatalog:
    """Explicit catalog of Site Targeting adapters."""

    def __init__(
        self,
        adapters: Mapping[str, SiteAdapter] | Iterable[SiteAdapter] | None = None,
    ) -> None:
        if adapters is None:
            from warp_taskgen.sites.gitlab import GitLabSite
            from warp_taskgen.sites.reddit import RedditSite

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
