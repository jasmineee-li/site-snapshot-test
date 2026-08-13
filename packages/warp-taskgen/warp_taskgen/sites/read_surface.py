"""Pure Site-owned planning for Phase 2c read-surface verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urljoin, urlsplit

from warp_taskgen.seeding.site_contracts import EditorSeedResult, ReadSurfaceFact

VerificationMode = Literal["body_text", "seed_resource"]


@dataclass(frozen=True)
class ReadSurfacePlanFailure:
    """Structured failure returned when Site evidence cannot form a safe plan."""

    site: str
    reason: str
    detail: str


@dataclass(frozen=True)
class ReadSurfaceVerificationPlan:
    """Immutable browser-neutral inputs for one rendered read-surface check."""

    site: str
    surfaces: tuple[ReadSurfaceFact, ...]
    signature: str
    verification_mode: VerificationMode
    identity_tokens: Mapping[str, Any]
    provenance_source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or not self.site.strip():
            raise ValueError("read-surface plan requires a non-empty Site")
        try:
            surfaces = tuple(self.surfaces)
        except TypeError as exc:
            raise ValueError("read-surface plan requires typed surface facts") from exc
        if not surfaces or any(not isinstance(item, ReadSurfaceFact) for item in surfaces):
            raise ValueError("read-surface plan requires typed surface facts")
        if not isinstance(self.signature, str) or not self.signature.strip():
            raise ValueError("read-surface plan requires a non-empty signature")
        if self.verification_mode not in {"body_text", "seed_resource"}:
            raise ValueError("read-surface plan has an unsupported verification mode")
        object.__setattr__(self, "site", self.site.strip().lower())
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "signature", self.signature.strip())
        object.__setattr__(self, "identity_tokens", MappingProxyType(dict(self.identity_tokens)))

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(surface.url for surface in self.surfaces)


@runtime_checkable
class SiteReadSurfaceCapability(Protocol):
    """Optional Site capability for deterministic read-surface planning."""

    def build_read_surface_plan(
        self,
        *,
        seed_result: EditorSeedResult,
        signature: str,
        origin: str,
    ) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure: ...


def build_read_surface_plan(
    *,
    site: str,
    seed_result: EditorSeedResult,
    signature: str,
    origin: str,
    identity_keys: tuple[str, ...],
) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure:
    """Build a safe plan from typed editor evidence without browser behavior."""

    parsed_origin = urlsplit(str(origin or "").strip())
    if parsed_origin.scheme not in {"http", "https"} or not parsed_origin.netloc:
        return ReadSurfacePlanFailure(site, "missing_origin", "read-surface plan needs an origin")
    canonical_origin = f"{parsed_origin.scheme}://{parsed_origin.netloc}"

    safe: list[ReadSurfaceFact] = []
    seen: set[str] = set()
    foreign_seen = False
    for surface in seed_result.read_surfaces:
        raw = surface.url.strip()
        parsed = urlsplit(raw)
        if raw.startswith("//"):
            foreign_seen = True
            continue
        if parsed.scheme or parsed.netloc:
            if parsed.scheme != parsed_origin.scheme or parsed.netloc != parsed_origin.netloc:
                foreign_seen = True
                continue
            resolved = raw
        elif raw.startswith("/"):
            resolved = urljoin(canonical_origin + "/", raw)
        else:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        safe.append(
            ReadSurfaceFact(
                url=resolved,
                provenance_source=surface.provenance_source,
                editor_method=surface.editor_method,
            )
        )

    if not safe:
        reason = "foreign_read_surface" if foreign_seen else "missing_read_surface"
        return ReadSurfacePlanFailure(
            site,
            reason,
            "no same-origin or path-local read surface was emitted for the payload call",
        )

    tokens = {
        key: seed_result.write_tokens[key]
        for key in identity_keys
        if seed_result.write_tokens.get(key) not in (None, "")
    }
    mode: VerificationMode = "seed_resource" if tokens else "body_text"
    try:
        return ReadSurfaceVerificationPlan(
            site=site,
            surfaces=tuple(safe),
            signature=signature,
            verification_mode=mode,
            identity_tokens=tokens,
            provenance_source=seed_result.read_surface_provenance_source,
        )
    except (TypeError, ValueError) as exc:
        return ReadSurfacePlanFailure(site, "invalid_read_surface_plan", str(exc))


class BoundReadSurface:
    """Mixin exposing the optional capability through one bound Site context."""

    _adapter: Any
    _context: Any

    def read_surface_plan(
        self,
        *,
        seed_result: EditorSeedResult,
        signature: str,
    ) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure:
        builder = getattr(self._adapter, "build_read_surface_plan", None)
        if not callable(builder):
            return ReadSurfacePlanFailure(
                self._context.site,
                "unsupported_read_surface",
                "Site does not provide read-surface verification planning",
            )
        supported = getattr(self._adapter, "supported_benchmarks", frozenset())
        if self._context.benchmark not in supported:
            return ReadSurfacePlanFailure(
                self._context.site,
                "unsupported_benchmark",
                f"benchmark {self._context.benchmark!r} is not supported by this Site",
            )
        origin = self._context.site_origin()
        if origin is None:
            return ReadSurfacePlanFailure(
                self._context.site,
                "missing_origin",
                "read-surface planning requires an explicit bound origin",
            )
        try:
            plan = builder(
                seed_result=seed_result,
                signature=signature,
                origin=origin,
            )
        except Exception as exc:
            return ReadSurfacePlanFailure(
                self._context.site,
                "read_surface_adapter_error",
                f"{exc.__class__.__name__}: {exc}",
            )
        if not isinstance(plan, (ReadSurfaceVerificationPlan, ReadSurfacePlanFailure)):
            return ReadSurfacePlanFailure(
                self._context.site,
                "invalid_read_surface_plan",
                "Site returned an unsupported read-surface plan value",
            )
        if isinstance(plan, ReadSurfaceVerificationPlan) and plan.site != self._context.site:
            return ReadSurfacePlanFailure(
                self._context.site,
                "invalid_read_surface_plan",
                "read-surface plan Site does not match the bound Site",
            )
        if isinstance(plan, ReadSurfaceVerificationPlan):
            parsed_origin = urlsplit(origin)
            for surface in plan.surfaces:
                parsed_surface = urlsplit(surface.url)
                if (
                    parsed_surface.scheme != parsed_origin.scheme
                    or parsed_surface.netloc != parsed_origin.netloc
                ):
                    return ReadSurfacePlanFailure(
                        self._context.site,
                        "foreign_read_surface",
                        "Site plan returned a read surface outside the bound origin",
                    )
        return plan


__all__ = [
    "BoundReadSurface",
    "ReadSurfacePlanFailure",
    "ReadSurfaceVerificationPlan",
    "SiteReadSurfaceCapability",
    "VerificationMode",
    "build_read_surface_plan",
]
