"""Phase 2 target resolution l4."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2.target_resolution._context import install_context

install_context(globals())

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
) -> list[dict[str, Any]]:
    """Expand a listing-kind resource into N concrete item records.

    For non-listing kinds returns ``[resource]`` unchanged so the caller
    can use a single dispatcher regardless of kind. Empty probe result
    returns ``[]`` so the caller can exclude the task (no items to
    attack means no Option-A placement exists for this listing).
    """
    kind = resource.get("kind")
    if kind not in _LISTING_KINDS:
        return [dict(resource)]

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
        record = _project_item_to_record(
            resource,
            item,
            placeholders,
            benchmark=benchmark,
        )
        if record is not None:
            records.append(record)
    return records

