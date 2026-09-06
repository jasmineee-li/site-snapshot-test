"""Value contracts for the deterministic Site Targeting seam.

The contracts are intentionally independent from the catalog registry and
task-resolution policy.  Site feature modules use them to describe routes and
bound contexts; callers consume the resulting targets and failures without
reaching into an adapter implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.placeholders import apply_placeholders
from warp_taskgen.sites.carrier_policy import SiteCarrierPolicy


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
        profile_sites: set[str] = set()
        for profile_field in ("site_name", "site"):
            if profile_field not in self.profile:
                continue
            value = self.profile[profile_field]
            if not isinstance(value, str) or not value.strip():
                raise SiteTargetingDefinitionError(
                    f"profile {profile_field} must be a non-empty string"
                )
            profile_sites.add(value.strip().lower())
        if len(profile_sites) > 1:
            raise SiteTargetingDefinitionError("profile site fields disagree")
        profile_site = next(iter(profile_sites), "")
        if profile_site and profile_site != site:
            raise SiteTargetingDefinitionError(
                f"profile site {profile_site!r} does not match bound site {site!r}"
            )
        profile_benchmarks: set[str] = set()
        for profile_field in ("benchmark_name", "benchmark"):
            if profile_field not in self.profile:
                continue
            value = self.profile[profile_field]
            if not isinstance(value, str) or not value.strip():
                raise SiteTargetingDefinitionError(
                    f"profile {profile_field} must be a non-empty string"
                )
            normalized = normalize_benchmark_name(value)
            if not normalized:
                raise SiteTargetingDefinitionError(f"profile {profile_field} is unknown")
            profile_benchmarks.add(normalized)
        if len(profile_benchmarks) > 1:
            raise SiteTargetingDefinitionError("profile benchmark fields disagree")
        profile_benchmark = next(iter(profile_benchmarks), "")
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
            profile={} if profile is None else profile,
            origin=origin,
            placeholders={} if placeholders is None else placeholders,
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
class SurfaceResolution:
    """A Site-owned mapping from a profile surface to a canonical carrier.

    Surface identity is deliberately kept separate from core/active-carrier
    policy.  A Site can explain what a profile field means, while Phase 1/2
    remains responsible for deciding whether that carrier is eligible.
    """

    benchmark: str
    site: str
    canonical_surface_id: str
    profile_surface_id: str
    profile_surface: Mapping[str, Any]
    evidence: str
    source_field: str | None = None
    editor_surface_id: str | None = None

    def __post_init__(self) -> None:
        benchmark = normalize_benchmark_name(self.benchmark)
        site = str(self.site or "").strip().lower()
        canonical = str(self.canonical_surface_id or "").strip()
        profile_id = str(self.profile_surface_id or "").strip()
        evidence = str(self.evidence or "").strip()
        if not benchmark or not site or not canonical or not profile_id or not evidence:
            raise SiteTargetingDefinitionError(
                "surface resolution needs benchmark, site, surface ids, and evidence"
            )
        if not isinstance(self.profile_surface, Mapping):
            raise SiteTargetingDefinitionError("profile_surface must be a mapping")
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "canonical_surface_id", canonical)
        object.__setattr__(self, "profile_surface_id", profile_id)
        object.__setattr__(self, "profile_surface", dict(self.profile_surface))
        object.__setattr__(self, "evidence", evidence)

    def as_record(self) -> dict[str, Any]:
        """Serialize the resolution using the historical artifact shape."""

        payload: dict[str, Any] = {
            "benchmark": self.benchmark,
            "site": self.site,
            "canonical_surface_id": self.canonical_surface_id,
            "profile_surface_id": self.profile_surface_id,
            "evidence": self.evidence,
        }
        if self.source_field:
            payload["source_field"] = self.source_field
        if self.editor_surface_id:
            payload["editor_surface_id"] = self.editor_surface_id
        return payload


@dataclass(frozen=True)
class SiteRouteContractFacts:
    """Inventory/profile facts needed to describe one Phase 1 route.

    These facts are intentionally narrower than a Phase 1 route contract.
    They carry only Site grammar and inventory evidence; editor eligibility,
    exposure, instruction, and answer policy remain generic Phase 1 policy.
    """

    allowed_start_url_patterns: tuple[str, ...] = ()
    anchor_examples: tuple[Mapping[str, Any], ...] = ()
    requires_inventory_backed_start_url: bool = False
    route_variant: str | None = None

    def __post_init__(self) -> None:
        patterns = tuple(str(pattern).strip() for pattern in self.allowed_start_url_patterns)
        if any(not pattern for pattern in patterns):
            raise SiteTargetingDefinitionError("route facts need non-empty URL patterns")
        if len(set(patterns)) != len(patterns):
            raise SiteTargetingDefinitionError("route facts contain duplicate URL patterns")
        if any(not isinstance(example, Mapping) for example in self.anchor_examples):
            raise SiteTargetingDefinitionError("route facts contain an invalid anchor example")
        object.__setattr__(self, "allowed_start_url_patterns", patterns)
        object.__setattr__(
            self,
            "anchor_examples",
            tuple(dict(example) for example in self.anchor_examples),
        )
        if self.route_variant is not None:
            variant = str(self.route_variant).strip()
            object.__setattr__(self, "route_variant", variant or None)

    @property
    def start_url_patterns(self) -> tuple[str, ...]:
        """Short alias used by Site feature callers."""

        return self.allowed_start_url_patterns

    @property
    def inventory_backed(self) -> bool:
        """Whether the route requires concrete profile inventory anchors."""

        return self.requires_inventory_backed_start_url


@runtime_checkable
class SiteProfileRouteCapability(Protocol):
    """Optional Site feature capability for profile identity and route facts.

    The capability is intentionally independent of :class:`SiteAdapter` so a
    compatibility/test adapter can opt in without acquiring editor, core,
    active-carrier, exposure, or reward policy.
    """

    def canonicalize_surface_id(
        self,
        *,
        benchmark: str,
        raw_surface_id: str | None,
    ) -> str | None: ...

    def resolve_profile_surface(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, Any],
        target_surface_id: str,
        kind: str | None = None,
        method: str | None = None,
        editor_surface_id: str | None = None,
    ) -> SurfaceResolution | None: ...

    def route_contract_facts(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, Any],
        kind: str,
    ) -> SiteRouteContractFacts: ...


@runtime_checkable
class SiteCarrierPolicyCapability(Protocol):
    """Optional Site feature capability for core-surface and carrier policy.

    Separate from :class:`SiteProfileRouteCapability` so a Site can explain
    profile identity without deciding which carriers are admissible.  A Site
    without this capability binds to ``SiteCarrierPolicy.closed``.
    """

    def carrier_policy(self, *, benchmark: str) -> SiteCarrierPolicy | None: ...


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


__all__ = [
    "CanonicalRoute",
    "ResolvedTarget",
    "SiteAdapter",
    "SiteCarrierPolicy",
    "SiteCarrierPolicyCapability",
    "SiteProfileRouteCapability",
    "SiteRouteContractFacts",
    "SiteTargetingDefinitionError",
    "SurfaceResolution",
    "TargetingContext",
    "TargetingFailure",
]
