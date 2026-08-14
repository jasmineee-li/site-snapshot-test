"""Classifieds read-surface planning capability."""

from __future__ import annotations

from urllib.parse import parse_qsl, urljoin, urlsplit

from warp_taskgen.seeding.site_contracts import EditorSeedResult
from warp_taskgen.sites.read_surface import (
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
    build_read_surface_plan,
)


class ClassifiedsReadSurfaceCapability:
    """Build a browser-neutral plan for an ordinary-reader listing page."""

    def build_read_surface_plan(
        self,
        *,
        seed_result: EditorSeedResult,
        signature: str,
        origin: str,
    ) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure:
        identity_keys = ("listing_id", "reply_id", "actor_name", "reply_body_sha256")
        identity = seed_result.write_tokens
        if any(identity.get(key) in (None, "") for key in identity_keys):
            return ReadSurfacePlanFailure(
                "classifieds",
                "missing_listing_reply_identity",
                "Classifieds readback requires listing, reply, actor, and body evidence",
            )
        listing_id = str(identity["listing_id"]).strip()
        surfaces = seed_result.read_surfaces
        if len(surfaces) != 1 or not _is_exact_listing_surface(
            surfaces[0].url,
            origin=origin,
            listing_id=listing_id,
        ):
            return ReadSurfacePlanFailure(
                "classifieds",
                "invalid_listing_read_surface",
                "Classifieds readback requires exactly the created reply's listing route",
            )
        return build_read_surface_plan(
            site="classifieds",
            seed_result=seed_result,
            signature=signature,
            origin=origin,
            identity_keys=identity_keys,
        )


def _is_exact_listing_surface(url: str, *, origin: str, listing_id: str) -> bool:
    try:
        origin_parts = urlsplit(origin)
        resolved = urljoin(f"{origin_parts.scheme}://{origin_parts.netloc}/", url)
        parts = urlsplit(resolved)
        query = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
    except (TypeError, ValueError):
        return False
    return (
        origin_parts.scheme in {"http", "https"}
        and bool(origin_parts.netloc)
        and (parts.scheme, parts.netloc) == (origin_parts.scheme, origin_parts.netloc)
        and parts.path == "/index.php"
        and not parts.fragment
        and query == [("page", "item"), ("id", listing_id)]
    )


__all__ = ["ClassifiedsReadSurfaceCapability"]
