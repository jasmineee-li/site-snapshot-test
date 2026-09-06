"""Seed execution result metadata."""

from __future__ import annotations

import urllib.parse
from typing import Any

from warp_taskgen.seeding.site_contracts import EditorSeedResult


def _created_resources_from_editor_result(
    result: dict[str, Any],
    *,
    editor_method: str,
) -> list[dict[str, Any]]:
    """Extract generic created-resource descriptors from an editor result.

    Editors own site-specific write semantics. Callers should not need to
    know that a Postmill-created resource is called a submission or that a
    GitLab-created resource is called an issue. This helper preserves the
    editor-declared, generic transition targets that Phase 2c can later use
    for exposure verification.
    """
    normalized = EditorSeedResult.from_mapping(result, editor_method=editor_method)
    return [resource.as_mapping() for resource in normalized.created_resources]


def _editor_call_result_record(
    result: dict[str, Any],
    *,
    call_index: int,
    editor_site_name: str,
    method_name: str,
    benchmark: str | None = None,
    logical_record_key: object = None,
) -> dict[str, Any]:
    """Return per-call write/read metadata for call-aware verification.

    The aggregate ``read_surface_urls`` / ``issue_iid`` metadata intentionally
    preserves older callers' simple shape. Phase 2c also needs to prove that a
    rendered signature came from the same editor call that produced the read
    surface being checked, especially for self-contained seeds that preserve a
    benign setup call before appending the adversarial write.
    """
    record: dict[str, Any] = {
        "call_index": call_index,
        "site": editor_site_name,
        "method": method_name,
        "editor_method": f"{editor_site_name}.{method_name}",
    }
    if benchmark is not None and benchmark.strip():
        record["benchmark"] = benchmark.strip()
    if isinstance(logical_record_key, str) and logical_record_key.strip():
        record["logical_record_key"] = logical_record_key.strip()
    normalized = EditorSeedResult.from_mapping(
        result,
        editor_method=f"{editor_site_name}.{method_name}",
    )
    if normalized.read_surface_urls:
        record["read_surface_urls"] = list(normalized.read_surface_urls)
    if normalized.read_surface_provenance_source is not None:
        record["read_surface_provenance_source"] = normalized.read_surface_provenance_source
    created_resources = [resource.as_mapping() for resource in normalized.created_resources]
    if created_resources:
        record["created_resources"] = created_resources
        record["created_resource"] = _primary_created_resource(created_resources)
    if normalized.write_tokens:
        record["write_tokens"] = dict(normalized.write_tokens)
    return record


def _dedupe_created_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for resource in resources:
        key = (
            str(resource.get("role") or ""),
            str(resource.get("kind") or ""),
            str(resource.get("url") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(resource))
    return deduped


def _primary_created_resource(resources: list[dict[str, Any]]) -> dict[str, Any]:
    for resource in reversed(resources):
        if resource.get("role") == "seed_render_surface":
            return dict(resource)
    return dict(resources[-1])


def _call_reference(call: dict[str, Any]) -> str | None:
    raw_path = call.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        return raw_path
    raw_url = call.get("url")
    if isinstance(raw_url, str) and raw_url.strip():
        return raw_url
    return None


def _concrete_call_path(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    return path


__all__ = [
    "_call_reference",
    "_concrete_call_path",
    "_created_resources_from_editor_result",
    "_dedupe_created_resources",
    "_editor_call_result_record",
    "_primary_created_resource",
]
