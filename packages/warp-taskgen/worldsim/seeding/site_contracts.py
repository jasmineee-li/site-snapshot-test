"""Explicit contracts for site-owned seed execution.

The active editor registry predates the Site cutover and remains a patchable
compatibility surface.  ``SeedSiteRegistry`` is the small, immutable seam used
by new callers: a seed run can bind the editor factories it is allowed to use
without changing process-wide registration state.

The result facts in this module deliberately describe only generic write and
read evidence.  Site editors still own authentication, HTTP, mutation, and
cleanup semantics; Phase 2/2c still owns exposure and admission policy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from worldsim.benchmark_capabilities import get_benchmark_capabilities, normalize_benchmark_name

EditorFactory = Callable[[dict[str, Any], Any], Any]
EditorKey = tuple[str, str]


@dataclass(frozen=True)
class SeedSiteRegistration:
    """One immutable benchmark/Site editor factory binding."""

    benchmark: str
    site: str
    editor_factory: EditorFactory

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, str) or not self.benchmark.strip():
            raise ValueError("seed site registration requires a non-empty benchmark")
        if not isinstance(self.site, str) or not self.site.strip():
            raise ValueError("seed site registration requires a non-empty site")
        benchmark = get_benchmark_capabilities(self.benchmark).canonical_name
        site = self.site.strip().lower()
        if not callable(self.editor_factory):
            raise TypeError("seed site registration editor_factory must be callable")
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "site", site)

    @property
    def key(self) -> EditorKey:
        return self.benchmark, self.site

    def create(self, instance: dict[str, Any], session: Any) -> Any:
        """Construct an editor through the bound factory."""
        return self.editor_factory(instance, session)


@dataclass(frozen=True)
class SeedSiteRegistry:
    """Immutable per-run registry of Site seed editor factories.

    ``from_editor_registry`` is the compatibility adapter for the historical
    mutable ``worldsim.editors.EDITOR_REGISTRY``.  Callers that need an
    isolated test Site should construct this registry directly and pass it to
    :func:`worldsim.seeding.apply_data_seed`; no global mapping is changed.
    """

    registrations: Mapping[EditorKey, SeedSiteRegistration]

    def __post_init__(self) -> None:
        normalized: dict[EditorKey, SeedSiteRegistration] = {}
        for raw_key, registration in self.registrations.items():
            if not isinstance(registration, SeedSiteRegistration):
                raise TypeError("seed registry values must be SeedSiteRegistration instances")
            key = registration.key
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError(f"invalid seed registry key: {raw_key!r}")
            raw_benchmark, raw_site = raw_key
            if not isinstance(raw_benchmark, str) or not isinstance(raw_site, str):
                raise ValueError(f"invalid seed registry key: {raw_key!r}")
            requested_key = (
                normalize_benchmark_name(raw_benchmark),
                raw_site.strip().lower(),
            )
            if requested_key != key:
                raise ValueError(
                    "seed registry key does not match its registration: "
                    f"{requested_key!r} != {key!r}"
                )
            if key in normalized:
                raise ValueError(f"duplicate seed registry registration: {key!r}")
            normalized[key] = registration
        object.__setattr__(self, "registrations", MappingProxyType(normalized))

    @classmethod
    def from_editor_registry(
        cls,
        editor_registry: Mapping[EditorKey, type | EditorFactory],
    ) -> SeedSiteRegistry:
        """Snapshot a legacy editor mapping without retaining mutable state."""
        registrations: dict[EditorKey, SeedSiteRegistration] = {}
        for raw_key, editor_factory in editor_registry.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError(f"invalid editor registry key: {raw_key!r}")
            benchmark, site = raw_key
            if not isinstance(benchmark, str) or not isinstance(site, str):
                raise ValueError(f"invalid editor registry key: {raw_key!r}")
            registration = SeedSiteRegistration(benchmark, site, editor_factory)
            if registration.key in registrations:
                raise ValueError(f"duplicate seed registry registration: {registration.key!r}")
            registrations[registration.key] = registration
        return cls(registrations)

    @classmethod
    def from_registrations(
        cls,
        registrations: Iterable[SeedSiteRegistration],
    ) -> SeedSiteRegistry:
        by_key: dict[EditorKey, SeedSiteRegistration] = {}
        for registration in registrations:
            if not isinstance(registration, SeedSiteRegistration):
                raise TypeError("seed registry values must be SeedSiteRegistration instances")
            if registration.key in by_key:
                raise ValueError(f"duplicate seed registry registration: {registration.key!r}")
            by_key[registration.key] = registration
        return cls(by_key)

    def get(self, benchmark: str, site: str) -> SeedSiteRegistration | None:
        """Return a registered binding, or ``None`` for unsupported Sites."""
        return self.registrations.get((normalize_benchmark_name(benchmark), site.strip().lower()))


@dataclass(frozen=True)
class CreatedResourceFact:
    """Generic created-resource evidence emitted by a Site editor."""

    url: str
    role: str = "created_resource"
    kind: str | None = None
    id: str | None = None
    parent_url: str | None = None
    editor_method: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        editor_method: str | None = None,
    ) -> CreatedResourceFact | None:
        url = value.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        raw_id = value.get("id")
        resource_id = None if raw_id in (None, "") else str(raw_id)
        return cls(
            url=url.strip(),
            role=str(value.get("role") or "created_resource").strip() or "created_resource",
            kind=(str(value["kind"]).strip() if value.get("kind") not in (None, "") else None),
            id=resource_id,
            parent_url=(
                str(value["parent_url"]).strip()
                if value.get("parent_url") not in (None, "")
                else None
            ),
            editor_method=editor_method,
        )

    def as_mapping(self) -> dict[str, Any]:
        value: dict[str, Any] = {"role": self.role, "url": self.url}
        for key, field_value in (
            ("kind", self.kind),
            ("id", self.id),
            ("parent_url", self.parent_url),
            ("editor_method", self.editor_method),
        ):
            if field_value not in (None, ""):
                value[key] = field_value
        return value


@dataclass(frozen=True)
class EditorSeedResult:
    """Normalized generic result while preserving legacy editor dictionaries."""

    write_tokens: Mapping[str, Any]
    created_resources: tuple[CreatedResourceFact, ...]
    read_surface_urls: tuple[str, ...]
    read_surface_provenance_source: str | None = None
    editor_method: str | None = None

    @property
    def read_surfaces(self) -> tuple[ReadSurfaceFact, ...]:
        """Typed read-surface facts projected from the legacy URL tuple."""
        return tuple(
            ReadSurfaceFact(
                url=url,
                provenance_source=self.read_surface_provenance_source,
                editor_method=self.editor_method,
            )
            for url in self.read_surface_urls
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        editor_method: str | None = None,
    ) -> EditorSeedResult:
        raw_items: list[Any] = []
        raw_single = value.get("created_resource")
        if isinstance(raw_single, Mapping):
            raw_items.append(raw_single)
        raw_many = value.get("created_resources")
        if isinstance(raw_many, list):
            raw_items.extend(raw_many)
        resources = tuple(
            resource
            for item in raw_items
            if isinstance(item, Mapping)
            for resource in (CreatedResourceFact.from_mapping(item, editor_method=editor_method),)
            if resource is not None
        )
        raw_urls = value.get("read_surface_urls")
        urls = (
            tuple(item.strip() for item in raw_urls if isinstance(item, str) and item.strip())
            if isinstance(raw_urls, list)
            else ()
        )
        write_tokens = {
            key: value[key]
            for key in (
                "note_id",
                "issue_iid",
                "project_id",
                "comment_id",
                "submission_id",
                "review_id",
            )
            if value.get(key) not in (None, "")
        }
        source = value.get("read_surface_provenance_source")
        return cls(
            write_tokens=MappingProxyType(write_tokens),
            created_resources=resources,
            read_surface_urls=urls,
            read_surface_provenance_source=(
                source.strip() if isinstance(source, str) and source.strip() else None
            ),
            editor_method=editor_method,
        )


@dataclass(frozen=True)
class ReadSurfaceFact:
    """Generic URL evidence emitted by a Site editor."""

    url: str
    provenance_source: str | None = None
    editor_method: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("read-surface fact requires a non-empty URL")
        object.__setattr__(self, "url", self.url.strip())

    def as_mapping(self) -> dict[str, str]:
        value = {"url": self.url}
        if self.provenance_source:
            value["provenance_source"] = self.provenance_source
        if self.editor_method:
            value["editor_method"] = self.editor_method
        return value


__all__ = [
    "CreatedResourceFact",
    "EditorFactory",
    "EditorKey",
    "EditorSeedResult",
    "ReadSurfaceFact",
    "SeedSiteRegistration",
    "SeedSiteRegistry",
]
