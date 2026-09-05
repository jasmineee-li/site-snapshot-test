"""Mechanics for a Site-owned carrier policy.

A Site decides which of its canonical surfaces are Path A core surfaces and
which of those may carry an injected payload on a mainline run.  This module
only implements the matching; the tables live beside each Site feature
(``gitlab_profile.py``, ``reddit_profile.py``) and Phase 1/2/4 reach them
through ``BoundSite.carrier_policy()``.

A surface kind is core if and only if a regular authenticated non-admin user
can write it through the platform's public API and an agent doing a typical
benign task naturally traverses it.  Active carrier policy is stricter:
retired title fields stay canonical for old artifacts and benign labels, but
new Phase 1/2/4 mainline runs must not use them as injected payload carriers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SiteCarrierPolicy:
    """Core-surface allowlist and active-carrier rules for one benchmark/Site."""

    benchmark: str
    surface_aliases: Mapping[str, str] = field(default_factory=dict)
    core_surfaces: frozenset[str] = frozenset()
    retired_carrier_surfaces: frozenset[str] = frozenset()
    unsupported_carrier_kinds: frozenset[str] = frozenset()
    unsupported_carrier_methods: frozenset[str] = frozenset()
    unsupported_carrier_surfaces: frozenset[str] = frozenset()
    retired_reason: str = "retired_title_carrier_surface"
    unsupported_reason: str = "unsupported_merge_request_carrier_surface"

    def __post_init__(self) -> None:
        for name in (
            "core_surfaces",
            "retired_carrier_surfaces",
            "unsupported_carrier_kinds",
            "unsupported_carrier_methods",
            "unsupported_carrier_surfaces",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    @classmethod
    def closed(cls, benchmark: str) -> SiteCarrierPolicy:
        """A policy with no core surfaces: nothing canonicalizes, nothing carries."""

        return cls(benchmark=benchmark)

    def canonical_surface(self, raw_surface_id: str | None) -> str | None:
        """Return the canonical surface id for a raw editor/profile surface id.

        Aliases match exactly; an unknown non-empty id is returned unchanged.
        """

        if not raw_surface_id:
            return None
        surface = raw_surface_id.strip()
        if not surface:
            return None
        return self.surface_aliases.get(surface, surface)

    def is_core_surface(self, raw_surface_id: str | None) -> bool:
        """Return True iff the surface is in this Site's core allowlist."""

        canonical = self.canonical_surface(raw_surface_id)
        if not canonical:
            return False
        return canonical in self.core_surfaces

    def retired_reason_for(self, raw_surface_id: str | None) -> str | None:
        """Return the active-carrier retirement reason for a canonicalized surface."""

        canonical = self.canonical_surface(raw_surface_id)
        if not canonical:
            return None
        if canonical in self.retired_carrier_surfaces:
            return self.retired_reason
        return None

    def ineligible_reason(
        self,
        raw_surface_id: str | None,
        *,
        kind: str | None = None,
        method: str | None = None,
    ) -> str | None:
        """Return why this route cannot be used as a mainline active IPI carrier."""

        retired = self.retired_reason_for(raw_surface_id)
        if retired is not None:
            return retired
        canonical = self.canonical_surface(raw_surface_id)
        kind_key = kind.strip().lower() if isinstance(kind, str) else ""
        method_key = method.strip() if isinstance(method, str) else ""
        if kind_key in self.unsupported_carrier_kinds:
            return self.unsupported_reason
        if method_key in self.unsupported_carrier_methods:
            return self.unsupported_reason
        if canonical in self.unsupported_carrier_surfaces:
            return self.unsupported_reason
        return None

    def is_active_carrier(
        self,
        raw_surface_id: str | None,
        *,
        kind: str | None = None,
        method: str | None = None,
    ) -> bool:
        """Return True iff the surface is currently admissible as an IPI carrier."""

        return (
            self.is_core_surface(raw_surface_id)
            and self.ineligible_reason(raw_surface_id, kind=kind, method=method) is None
        )


def _frozen(values: Iterable[str]) -> frozenset[str]:
    return frozenset(str(value) for value in values)


__all__ = ["SiteCarrierPolicy"]
