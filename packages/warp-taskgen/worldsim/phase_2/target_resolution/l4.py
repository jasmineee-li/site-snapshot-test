"""Phase 2 target resolution l4."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from worldsim.phase_2.target_resolution.constants import DEFAULT_L4_TOP_N
from worldsim.phase_2.target_resolution.listing_probes import _default_listing_probe
from worldsim.phase_2.target_resolution.listing_records import compose_listing_record
from worldsim.phase_2.target_resolution.types import ListingProbeFn
from worldsim.phase_2.target_resolution.url_matching import _empty_record
from worldsim.sites import (
    ListingItemCandidate,
    SiteCatalog,
    TargetingFailure,
    default_catalog,
)

logger = logging.getLogger(__name__)


def _l4_top_n_default() -> int:
    raw = os.environ.get("WORLDSIM_L4_TOP_N", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_L4_TOP_N


async def resolve_l4(
    resource: Mapping[str, Any],
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
    *,
    probe_fn: ListingProbeFn | None = None,
    top_n: int | None = None,
    placeholders: Mapping[str, str] | None = None,
    benchmark: str = "webarena_verified",
    catalog: SiteCatalog | None = None,
) -> list[dict[str, Any]]:
    """Expand a listing-kind resource into N concrete item records.

    For non-listing kinds returns ``[resource]`` unchanged so the caller
    can use a single dispatcher regardless of kind. Empty probe result
    returns ``[]`` so the caller can exclude the task (no items to
    attack means no Option-A placement exists for this listing).
    """
    kind = resource.get("kind")
    catalog = catalog or default_catalog()
    site_kind = catalog.site_for_task(
        task,
        fallback_kind=str(kind or ""),
        benchmark=benchmark,
    )
    bound = _bind_listing_site(
        site_kind,
        catalog=catalog,
        benchmark=benchmark,
        placeholders=placeholders,
        instance=instance,
    )
    if bound is None:
        # A known Site resource that cannot be bound must not appear to have
        # completed L4. Unknown/non-Site records remain on the identity path.
        known_site = catalog.site_for_kind(str(kind or ""), benchmark=benchmark)
        if known_site is not None:
            error = _empty_record(
                f"L4 Site Targeting bind failed for site {known_site!r}",
                pending_layer="L4",
            )
            error["layer"] = "L4"
            error["start_url_resolved"] = resource.get("start_url_resolved")
            error["targeting_failure"] = "site_context_unavailable"
            return [error]
        return [dict(resource)]
    if not bound.supports_benchmark():
        error = _empty_record(
            f"L4 Site Targeting does not support benchmark {benchmark!r}",
            pending_layer="L4",
        )
        error["layer"] = "L4"
        error["start_url_resolved"] = resource.get("start_url_resolved")
        error["targeting_failure"] = "unsupported_benchmark"
        return [error]
    if not bound.is_expandable_listing_kind(str(kind or "")):
        return [dict(resource)]
    if not bound.has_materialization_origin():
        error = _empty_record("L4 requires an explicit Site origin", pending_layer="L4")
        error["layer"] = "L4"
        error["start_url_resolved"] = resource.get("start_url_resolved")
        error["targeting_failure"] = "missing_origin"
        return [error]

    probe_fn = probe_fn or _default_listing_probe
    limit = top_n if top_n is not None else _l4_top_n_default()
    try:
        if probe_fn is _default_listing_probe:
            items = await probe_fn(resource, task, instance, limit=limit)
        else:
            items = await probe_fn(resource, task, instance)
    except Exception as exc:
        logger.exception("L4 listing probe failed for kind=%r", kind)
        error = _empty_record(f"L4 probe raised: {type(exc).__name__}: {exc}", pending_layer="L4")
        error["layer"] = "L4"
        error["start_url_resolved"] = resource.get("start_url_resolved")
        error["l4_error"] = str(exc)
        return [error]
    if not items:
        return []

    records: list[dict[str, Any]] = []
    for item in items[:limit]:
        candidate = _listing_candidate(resource, item)
        if candidate is None:
            continue
        target = bound.materialize_listing_entry(candidate)
        if isinstance(target, TargetingFailure):
            logger.debug(
                "L4 listing item rejected for source=%r item=%r: %s",
                candidate.source_kind,
                candidate.item_kind,
                target.reason,
            )
            continue
        record = compose_listing_record(
            resource,
            candidate,
            target,
            benchmark=benchmark,
        )
        if record is not None:
            records.append(record)
    return records


def _absolute_origin(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _bind_listing_site(
    site_kind: str | None,
    *,
    catalog: SiteCatalog,
    benchmark: str,
    placeholders: Mapping[str, str] | None,
    instance: Mapping[str, Any],
) -> Any:
    if not site_kind:
        return None
    origin = _absolute_origin(instance.get("site_url"))
    effective_placeholders = dict(placeholders or {})
    try:
        return catalog.bind(
            benchmark=benchmark,
            site=site_kind,
            origin=origin,
            placeholders=effective_placeholders,
        )
    except Exception as exc:
        logger.debug("L4 Site Targeting bind failed for %r: %s", site_kind, exc)
        return None


def _listing_candidate(
    resource: Mapping[str, Any], item: Mapping[str, Any] | object
) -> ListingItemCandidate | None:
    if not isinstance(item, Mapping):
        return None
    item_kind = item.get("_item_kind")
    if not isinstance(item_kind, str) or not item_kind.strip():
        return None
    evidence_url = resource.get("start_url_resolved")
    if evidence_url is not None and not isinstance(evidence_url, str):
        evidence_url = None
    try:
        return ListingItemCandidate(
            source_kind=str(resource.get("kind") or ""),
            item_kind=item_kind,
            payload=item,
            evidence_url=evidence_url,
        )
    except (TypeError, ValueError):
        return None
