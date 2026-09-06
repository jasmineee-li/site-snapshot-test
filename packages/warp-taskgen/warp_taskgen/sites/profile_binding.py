"""Bound operations for the optional Site profile-route capability."""

from __future__ import annotations

from typing import Any

from warp_taskgen.sites.contracts import (
    SiteCarrierPolicy,
    SiteRouteContractFacts,
    SiteTargetingDefinitionError,
    SurfaceResolution,
    TargetingContext,
)


class BoundProfileRoutes:
    """Mixin exposing profile/route facts through one immutable context."""

    _adapter: Any
    _context: TargetingContext

    def canonicalize_surface_id(self, raw_surface_id: str | None) -> str | None:
        resolver = getattr(self._adapter, "canonicalize_surface_id", None)
        if not callable(resolver):
            return None
        try:
            return resolver(
                benchmark=self._context.benchmark,
                raw_surface_id=raw_surface_id,
            )
        except Exception:
            return None

    def resolve_profile_surface(
        self,
        target_surface_id: str,
        *,
        kind: str | None = None,
        method: str | None = None,
        editor_surface_id: str | None = None,
    ) -> SurfaceResolution | None:
        resolver = getattr(self._adapter, "resolve_profile_surface", None)
        if not callable(resolver):
            return None
        try:
            resolution = resolver(
                benchmark=self._context.benchmark,
                profile=self._context.profile,
                target_surface_id=target_surface_id,
                kind=kind,
                method=method,
                editor_surface_id=editor_surface_id,
            )
        except Exception:
            return None
        if not isinstance(resolution, SurfaceResolution):
            return None
        if resolution.benchmark != self._context.benchmark or resolution.site != self._context.site:
            return None
        return resolution

    def route_contract_facts(self, kind: str) -> SiteRouteContractFacts:
        resolver = getattr(self._adapter, "route_contract_facts", None)
        if not callable(resolver):
            return SiteRouteContractFacts()
        try:
            facts = resolver(
                benchmark=self._context.benchmark,
                profile=self._context.profile,
                kind=kind,
            )
        except Exception:
            return SiteRouteContractFacts()
        if not isinstance(facts, SiteRouteContractFacts):
            raise SiteTargetingDefinitionError(
                f"Site adapter {self._context.site!r} returned invalid route facts"
            )
        return facts

    def carrier_policy(self) -> SiteCarrierPolicy:
        """Return the Site's carrier policy for the bound benchmark, else closed."""

        closed = SiteCarrierPolicy.closed(self._context.benchmark)
        resolver = getattr(self._adapter, "carrier_policy", None)
        if not callable(resolver):
            return closed
        try:
            policy = resolver(benchmark=self._context.benchmark)
        except Exception:
            return closed
        if not isinstance(policy, SiteCarrierPolicy):
            return closed
        if policy.benchmark != self._context.benchmark:
            return closed
        return policy

    def supports_profile_routes(self) -> bool:
        supported = getattr(self._adapter, "supported_benchmarks", frozenset())
        return self._context.benchmark in supported and all(
            callable(getattr(self._adapter, name, None))
            for name in (
                "canonicalize_surface_id",
                "resolve_profile_surface",
                "route_contract_facts",
            )
        )


__all__ = ["BoundProfileRoutes"]
