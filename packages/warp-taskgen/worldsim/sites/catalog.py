"""The deterministic Site Targeting seam.

The catalog binds a benchmark/profile projection to a named Site and delegates
URL grammar to that Site's feature module.  It deliberately has no knowledge
of Phase 2 contracts, editors, authentication, browsers, exposure, or reward
evaluation.  Those concerns consume the returned target through their own
interfaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from worldsim.benchmark_capabilities import normalize_benchmark_name
from worldsim.placeholders import apply_placeholders
from worldsim.sites.candidate_resolution import (
    SourceListing,
    TargetCandidate,
    materialize_candidate,
    source_listing_for_candidate,
    validate_probe,
)


class SiteTargetingDefinitionError(ValueError):
    """Raised when a Site adapter or catalog definition is invalid."""


@dataclass(frozen=True)
class TargetingContext:
    """Immutable benchmark/profile projection used by deterministic routing.

    ``origin`` and ``placeholders`` are optional while callers inspect route
    descriptors.  A resolution requires either an explicit origin or the
    Site's placeholder resolving to an HTTP(S) origin; no deployment host is
    inferred from a Site name.
    """

    benchmark: str
    site: str
    profile: Mapping[str, Any] = field(default_factory=dict)
    origin: str | None = None
    placeholders: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        benchmark = normalize_benchmark_name(self.benchmark)
        site = str(self.site or "").strip().lower()
        if not benchmark:
            raise SiteTargetingDefinitionError("benchmark is required")
        if not site:
            raise SiteTargetingDefinitionError("site is required")
        if not isinstance(self.profile, Mapping):
            raise SiteTargetingDefinitionError("profile must be a mapping")
        if not isinstance(self.placeholders, Mapping):
            raise SiteTargetingDefinitionError("placeholders must be a mapping")
        profile_site = str(
            self.profile.get("site_name") or self.profile.get("site") or ""
        ).strip().lower()
        if profile_site and profile_site != site:
            raise SiteTargetingDefinitionError(
                f"profile site {profile_site!r} does not match bound site {site!r}"
            )
        profile_benchmark = normalize_benchmark_name(
            self.profile.get("benchmark_name") or self.profile.get("benchmark")
        )
        if profile_benchmark and profile_benchmark != benchmark:
            raise SiteTargetingDefinitionError(
                f"profile benchmark {profile_benchmark!r} does not match {benchmark!r}"
            )
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(
            self,
            "placeholders",
            {str(key): str(value) for key, value in self.placeholders.items()},
        )
        if self.origin is not None:
            origin = str(self.origin).strip()
            object.__setattr__(self, "origin", origin.rstrip("/") or None)

    @classmethod
    def from_input(
        cls,
        context: TargetingContext | Mapping[str, Any] | None = None,
        *,
        benchmark: str = "webarena_verified",
        site: str | None = None,
        profile: Mapping[str, Any] | None = None,
        origin: str | None = None,
        placeholders: Mapping[str, str] | None = None,
    ) -> TargetingContext:
        if isinstance(context, TargetingContext):
            if any(value is not None for value in (site, profile, origin, placeholders)):
                raise SiteTargetingDefinitionError(
                    "context cannot be combined with site/profile/origin/placeholders"
                )
            return context
        if context is not None:
            if not isinstance(context, Mapping):
                raise SiteTargetingDefinitionError("context must be a TargetingContext or mapping")
            benchmark = str(context.get("benchmark") or benchmark)
            site = str(context.get("site") or context.get("site_name") or site or "")
            profile = context.get("profile", profile)
            origin = context.get("origin", context.get("site_url", origin))
            placeholders = context.get("placeholders", placeholders)
        if site is None:
            raise SiteTargetingDefinitionError("site is required")
        return cls(
            benchmark=benchmark,
            site=site,
            profile=profile or {},
            origin=origin,
            placeholders=placeholders or {},
        )

    def site_origin(self) -> str | None:
        """Return a validated origin supplied explicitly by the caller."""

        token = f"__{self.site.upper()}__"

        def normalize(raw: object) -> str | None:
            if not raw:
                return None
            try:
                resolved = apply_placeholders(str(raw), dict(self.placeholders), strict=True)
                parts = urlsplit(resolved)
            except (TypeError, ValueError):
                return None
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                return None
            if parts.query or parts.fragment:
                return None
            return resolved.rstrip("/")

        explicit = normalize(self.origin)
        placeholder = normalize(self.placeholders.get(token))
        if self.origin is not None and explicit is None:
            return None
        if token in self.placeholders and placeholder is None:
            return None
        if explicit and placeholder:
            explicit_parts = urlsplit(explicit)
            placeholder_parts = urlsplit(placeholder)
            if (explicit_parts.scheme, explicit_parts.netloc) != (
                placeholder_parts.scheme,
                placeholder_parts.netloc,
            ):
                return None
        return explicit or placeholder


@dataclass(frozen=True)
class CanonicalRoute:
    """One host-independent route descriptor owned by a Site feature."""

    id: str
    site: str
    kind: str
    allowed_start_url_patterns: tuple[str, ...]
    compatibility_kind: str | None = None
    anchor_examples: tuple[Mapping[str, Any], ...] = ()
    route_variant: str | None = None
    requires_inventory_backed_start_url: bool = False

    def __post_init__(self) -> None:
        route_id = str(self.id or "").strip()
        site = str(self.site or "").strip().lower()
        kind = str(self.kind or "").strip()
        patterns = tuple(str(pattern).strip() for pattern in self.allowed_start_url_patterns)
        if not route_id or not site or not kind:
            raise SiteTargetingDefinitionError("route id, site, and kind are required")
        if not patterns or any(not pattern for pattern in patterns):
            raise SiteTargetingDefinitionError(f"route {route_id!r} needs URL patterns")
        if len(set(patterns)) != len(patterns):
            raise SiteTargetingDefinitionError(f"duplicate URL patterns for route {route_id!r}")
        if any(not isinstance(example, Mapping) for example in self.anchor_examples):
            raise SiteTargetingDefinitionError(f"invalid anchor example for route {route_id!r}")
        object.__setattr__(self, "id", route_id)
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "allowed_start_url_patterns", patterns)
        object.__setattr__(
            self, "anchor_examples", tuple(dict(example) for example in self.anchor_examples)
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "site": self.site,
            "resource_kind": self.kind,
            "allowed_start_url_patterns": list(self.allowed_start_url_patterns),
        }
        if self.route_variant is not None:
            result["route_variant"] = self.route_variant
        if self.requires_inventory_backed_start_url:
            result["requires_inventory_backed_start_url"] = True
        if self.anchor_examples:
            result["anchor_examples"] = [dict(example) for example in self.anchor_examples]
        return result


@dataclass(frozen=True)
class ResolvedTarget:
    """A deterministic Resource Kind and Canonical Route bound to one Site."""

    site: str
    kind: str
    anchors: Mapping[str, Any]
    start_url_resolved: str | None
    layer: str
    canonical_route: CanonicalRoute | None = None
    evidence_url: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": self.kind,
            "anchors": dict(self.anchors),
            "start_url_resolved": self.start_url_resolved,
            "layer": self.layer,
        }
        if self.canonical_route is not None:
            record["canonical_route_id"] = self.canonical_route.id
        if self.evidence_url:
            record["evidence_url"] = self.evidence_url
        record.update(dict(self.extra))
        return record

    def __getitem__(self, key: str) -> Any:
        return self.as_record()[key]


@dataclass(frozen=True)
class TargetingFailure:
    """Structured, fail-closed result for unsupported task evidence."""

    site: str
    reason: str
    message: str
    layer: str | None = None
    evidence_url: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    pending_layer: str | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": None,
            "anchors": {},
            "start_url_resolved": None,
            "layer": self.layer,
            "reason": self.message,
            "targeting_failure": self.reason,
        }
        if self.pending_layer:
            record["pending_layer"] = self.pending_layer
        if self.evidence_url:
            record["evidence_url"] = self.evidence_url
        if self.evidence:
            record["evidence"] = dict(self.evidence)
        return record

    def __getitem__(self, key: str) -> Any:
        return self.as_record()[key]


class SiteAdapter(Protocol):
    """Pure feature-owned grammar consumed by :class:`SiteCatalog`."""

    site: str
    supported_benchmarks: frozenset[str]

    def validate(self) -> None: ...

    def validate_task(self, task: Mapping[str, Any]) -> tuple[str, str] | None: ...

    def routes(self, context: TargetingContext) -> Iterable[CanonicalRoute]: ...

    def match(
        self, url: str, task: Mapping[str, Any], context: TargetingContext
    ) -> tuple[str, dict[str, Any]] | None: ...

    def reconstruct(
        self,
        kind: str,
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> str | None: ...

    def is_listing(self, kind: str) -> bool: ...

    def listing_start_url(
        self, kind: str, resolved_url: str, fallback_url: str | None
    ) -> str | None: ...

def _strip_regex_anchors(url: str) -> str:
    if not url:
        return ""
    stripped = url.strip()
    if stripped.startswith("^"):
        stripped = stripped[1:]
    if stripped.endswith(".*$"):
        stripped = stripped[:-3]
    elif stripped.endswith("$"):
        stripped = stripped[:-1]
    if stripped.endswith(".*"):
        stripped = stripped[:-2]
    return stripped


def _strip_json_suffix(url: str) -> str:
    return url[: -len(".json")] if url.endswith(".json") else url


def _normalise_url(url: str, placeholders: Mapping[str, str]) -> str | None:
    if not url:
        return None
    stripped = _strip_json_suffix(_strip_regex_anchors(url))
    try:
        return apply_placeholders(stripped, dict(placeholders), strict=True)
    except ValueError:
        return None


def _path_and_query(url: str) -> str:
    if not url:
        return ""
    if "://" not in url:
        return url if url.startswith("/") else "/" + url
    parts = urlsplit(url)
    path = parts.path or "/"
    return f"{path}?{parts.query}" if parts.query else path


def _matches_origin(url: str, origin: str) -> bool:
    """Accept relative evidence or an absolute URL on the bound origin."""

    try:
        candidate = urlsplit(url)
        expected = urlsplit(origin)
    except ValueError:
        return False
    if not candidate.scheme and not candidate.netloc:
        return True
    return (candidate.scheme, candidate.netloc) == (expected.scheme, expected.netloc)


def _url_with_expected_query_params(url: str, expected: Mapping[str, Any]) -> str:
    query_params = expected.get("query_params")
    if not isinstance(query_params, Mapping) or not query_params:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    merged = parse_qs(parts.query, keep_blank_values=True)
    for key, raw in query_params.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(raw, list):
            values = [str(value) for value in raw if value is not None]
        elif raw is None:
            values = []
        else:
            values = [str(raw)]
        if values:
            merged[key] = values
    return urlunsplit(parts._replace(query=urlencode(merged, doseq=True)))


def _iter_eval_urls(task: Mapping[str, Any]) -> list[str]:
    """Return expected URLs with NetworkEvent entries ranked first."""

    reward = task.get("reward_function") or {}
    if not isinstance(reward, Mapping):
        return []
    evals = reward.get("eval") or []
    ranked: list[tuple[int, int, str]] = []
    for sequence, evaluator in enumerate(evals):
        if not isinstance(evaluator, Mapping):
            continue
        name = str(evaluator.get("evaluator") or "")
        priority = 0 if "NetworkEvent" in name else 1
        expected = evaluator.get("expected") or {}
        if not isinstance(expected, Mapping):
            continue
        raw = expected.get("url") or expected.get("reference_url")
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, list):
            candidates = [candidate for candidate in raw if isinstance(candidate, str)]
        else:
            continue
        for candidate in candidates:
            ranked.append(
                (priority, sequence, _url_with_expected_query_params(candidate, expected))
            )
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [url for _, _, url in ranked]


def _iter_start_urls(task: Mapping[str, Any]) -> list[str]:
    start = task.get("start_urls") or []
    if isinstance(start, str):
        return [start]
    if isinstance(start, Sequence):
        return [url for url in start if isinstance(url, str)]
    return []


@dataclass(frozen=True)
class _TaskSiteMetadata:
    """Validated task/delivery identity used by the L1/L2 resolver.

    ``task_site`` is the page/benign-task identity.  ``delivery_site`` is
    deliberately kept separate because a payload may be delivered to another
    Site (for example, an admin task whose mutation is on a storefront).  A
    multi-Site ``sites`` list is only accepted when the explicit ``site`` and
    ``delivery_channel.delivery_site`` explain the additional entry; callers
    must never select the first list item as a guess.
    """

    task_site: str | None
    delivery_site: str | None = None
    failure_reason: str | None = None
    failure_message: str | None = None

    @classmethod
    def failure(cls, reason: str, message: str) -> _TaskSiteMetadata:
        return cls(None, failure_reason=reason, failure_message=message)


_MISSING = object()


def _normalise_site_token(
    value: object,
    *,
    field: str,
    allow_none_token: bool = False,
) -> tuple[str | None, tuple[str, str] | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, ("malformed_site_metadata", f"{field} must be a string")
    token = value.strip().lower()
    if not token:
        return None, ("malformed_site_metadata", f"{field} must not be empty")
    if allow_none_token and token == "none":
        return None, None
    if token == "none":
        return None, ("malformed_site_metadata", f"{field} cannot be 'none'")
    return token, None


def _task_site_metadata(task: Mapping[str, Any]) -> _TaskSiteMetadata:
    """Validate site-bearing task metadata without choosing an arbitrary Site."""

    explicit_site: str | None = None
    if task.get("site", _MISSING) is not _MISSING:
        explicit_site, error = _normalise_site_token(task.get("site"), field="task.site")
        if error is not None:
            return _TaskSiteMetadata.failure(*error)

    sites_value = task.get("sites", _MISSING)
    declared_sites: list[str] = []
    if sites_value is not _MISSING and sites_value is not None:
        if isinstance(sites_value, str):
            sites_iterable: Sequence[object] = (sites_value,)
        elif isinstance(sites_value, Sequence) and not isinstance(
            sites_value, (bytes, bytearray)
        ):
            sites_iterable = sites_value
        else:
            return _TaskSiteMetadata.failure(
                "malformed_site_metadata", "task.sites must be a string or sequence of strings"
            )
        for index, value in enumerate(sites_iterable):
            token, error = _normalise_site_token(value, field=f"task.sites[{index}]")
            if error is not None:
                return _TaskSiteMetadata.failure(*error)
            if token is not None and token not in declared_sites:
                declared_sites.append(token)

    delivery_site: str | None = None
    delivery = task.get("delivery_channel", _MISSING)
    if delivery is not _MISSING and delivery is not None:
        if not isinstance(delivery, Mapping):
            return _TaskSiteMetadata.failure(
                "malformed_metadata", "task.delivery_channel must be a mapping"
            )
        if "delivery_site" in delivery:
            delivery_site, error = _normalise_site_token(
                delivery.get("delivery_site"),
                field="task.delivery_channel.delivery_site",
                allow_none_token=True,
            )
            if error is not None:
                return _TaskSiteMetadata.failure(*error)

    if explicit_site is not None:
        if declared_sites and explicit_site not in declared_sites:
            return _TaskSiteMetadata.failure(
                "conflicting_site_metadata",
                f"task.site {explicit_site!r} is absent from task.sites {declared_sites!r}",
            )
        extra_sites = set(declared_sites) - {explicit_site}
        if extra_sites and extra_sites != {delivery_site}:
            return _TaskSiteMetadata.failure(
                "ambiguous_site_metadata",
                "task.sites contains additional Sites without a matching delivery_site",
            )
        return _TaskSiteMetadata(explicit_site, delivery_site)

    if len(declared_sites) > 1:
        return _TaskSiteMetadata.failure(
            "ambiguous_site_metadata",
            "task.sites declares multiple Sites but task.site is absent",
        )
    if declared_sites:
        return _TaskSiteMetadata(declared_sites[0], delivery_site)
    if delivery_site is not None:
        return _TaskSiteMetadata.failure(
            "missing_task_site",
            "delivery_site is present but the page/task Site is not declared",
        )
    return _TaskSiteMetadata(None, delivery_site)


def _metadata_failure(task: Mapping[str, Any]) -> tuple[str, str] | None:
    """Validate nested L1/L2 metadata before any adapter can infer a target."""

    site_metadata = _task_site_metadata(task)
    if site_metadata.failure_reason:
        return site_metadata.failure_reason, site_metadata.failure_message or "invalid site metadata"

    reward = task.get("reward_function", _MISSING)
    if reward is not _MISSING and reward is not None:
        if not isinstance(reward, Mapping):
            return "malformed_metadata", "task.reward_function must be a mapping"
        evals = reward.get("eval", _MISSING)
        if evals is not _MISSING and evals is not None:
            if not isinstance(evals, Sequence) or isinstance(evals, (str, bytes, bytearray)):
                return "malformed_metadata", "task.reward_function.eval must be a sequence"
            for index, evaluator in enumerate(evals):
                if not isinstance(evaluator, Mapping):
                    return (
                        "malformed_metadata",
                        f"task.reward_function.eval[{index}] must be a mapping",
                    )
                expected = evaluator.get("expected", _MISSING)
                if expected is not _MISSING and expected is not None:
                    if not isinstance(expected, Mapping):
                        return (
                            "malformed_metadata",
                            f"task.reward_function.eval[{index}].expected must be a mapping",
                        )
                    for url_key in ("url", "reference_url"):
                        raw = expected.get(url_key, _MISSING)
                        if raw is _MISSING or raw is None:
                            continue
                        if isinstance(raw, str):
                            continue
                        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                            continue
                        return (
                            "malformed_metadata",
                            f"task.reward_function.eval[{index}].expected.{url_key} "
                            "must be a string or list of strings",
                        )
                    query_params = expected.get("query_params", _MISSING)
                    if query_params is not _MISSING and query_params is not None and not isinstance(
                        query_params, Mapping
                    ):
                        return (
                            "malformed_metadata",
                            f"task.reward_function.eval[{index}].expected.query_params "
                            "must be a mapping",
                        )

    start_urls = task.get("start_urls", _MISSING)
    if start_urls is not _MISSING and start_urls is not None:
        if isinstance(start_urls, str):
            pass
        elif isinstance(start_urls, Sequence) and not isinstance(
            start_urls, (bytes, bytearray)
        ):
            if not all(isinstance(url, str) for url in start_urls):
                return "malformed_metadata", "task.start_urls must contain only strings"
        else:
            return "malformed_metadata", "task.start_urls must be a string or sequence"

    agent_context = task.get("agent_context", _MISSING)
    if agent_context is not _MISSING and agent_context is not None:
        if not isinstance(agent_context, Mapping):
            return "malformed_metadata", "task.agent_context must be a mapping"
        for context_name, site_context in agent_context.items():
            if site_context is None:
                continue
            if not isinstance(site_context, Mapping):
                return (
                    "malformed_metadata",
                    f"task.agent_context.{context_name} must be a mapping",
                )
    return None


def _site_kind_for_task(task: Mapping[str, Any]) -> str | None:
    """Return the explicit page/task Site, never an arbitrary list member."""

    metadata = _task_site_metadata(task)
    return metadata.task_site if metadata.failure_reason is None else None


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

    def _route_for_identifier(
        self, kind: str, anchors: Mapping[str, Any]
    ) -> CanonicalRoute | None:
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
