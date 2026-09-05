"""Exposure mode inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from warp_taskgen.phase_2.exposure_contract.constants import (
    CREATE_CHILD_LISTING_KINDS,
    DIRECT_KINDS,
    LISTING_SOURCE_KINDS,
    TRANSITIVE_EXISTING_SOURCE_KINDS,
)


def _benign_read_url(resource: Mapping[str, Any]) -> str | None:
    value = resource.get("benign_read_url") or resource.get("start_url_resolved")
    return str(value) if isinstance(value, str) and value.strip() else None


def _mode_for_resource(resource: Mapping[str, Any], kind: str) -> tuple[str, str | None]:
    source_kind = resource.get("source_listing_kind")
    if isinstance(source_kind, str) and source_kind in TRANSITIVE_EXISTING_SOURCE_KINDS:
        if _transitive_entry_supported(source_kind, resource):
            return "bounded_transitive_existing", None
        return (
            "ineligible",
            f"unsupported_transitive_entry:{source_kind}",
        )
    if kind in DIRECT_KINDS:
        return "direct_detail", None
    if kind in CREATE_CHILD_LISTING_KINDS:
        if _created_child_listing_row_forced(resource):
            return "inline_listing_created_child", None
        return "bounded_transitive_created_child", None
    if kind in LISTING_SOURCE_KINDS:
        return "inline_listing", None
    return "ineligible", f"kind_not_supported_for_exposure:{kind}"


def _effective_mode_for_seeded_surface(
    *,
    base_mode: str,
    site: str,
    kind: str,
    editor_method: str,
    target_surface_id: str | None,
) -> str:
    """Return the encounter route for the selected seeded surface.

    Listing-level resources can support multiple write surfaces. A created
    title is paint-visible on the listing row; a created body is only visible
    after opening the child detail page. Keeping this distinction here prevents
    the old false-admission failure where a body/comment was treated like a
    listing-row title.
    """
    if target_surface_id in {"submission.body", "issue.description", "mr.description"}:
        if base_mode in {"inline_listing", "inline_listing_created_child"} and (
            _creates_child_detail_surface(site=site, kind=kind, editor_method=editor_method)
        ):
            return "bounded_transitive_created_child"
    if target_surface_id == "note.body" and base_mode == "inline_listing":
        if (
            site == "gitlab"
            and kind == "gitlab_search_result"
            and editor_method
            in {
                "create_issue_note",
                "create_mr_note",
            }
        ):
            return "bounded_transitive_existing"
    return base_mode


def _creates_child_detail_surface(*, site: str, kind: str, editor_method: str) -> bool:
    if site == "reddit" and kind == "reddit_forum" and editor_method == "create_submission":
        return True
    if (
        site == "gitlab"
        and kind == "gitlab_search_result"
        and editor_method
        in {
            "create_issue_description",
            "create_mr_description",
        }
    ):
        return True
    return False


def _transitive_entry_supported(source_kind: str, resource: Mapping[str, Any]) -> bool:
    url = resource.get("benign_read_url") or resource.get("start_url_resolved")
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    path = parsed.path or "/"
    if source_kind == "gitlab_search_result":
        # A project root is not a search/list surface. Admitting arbitrary
        # project-root -> issue links reintroduces the commit-count bug:
        # the agent can complete the benign task without reading issues.
        return path == "/search" or path.endswith("/-/issues") or path.endswith("/-/merge_requests")
    if source_kind == "gitlab_dashboard_list":
        return path.startswith("/dashboard/")
    if source_kind == "reddit_dashboard_list":
        return path.startswith("/user/")
    if source_kind == "reddit_forum":
        return path.startswith("/f/")
    return False


def _created_child_listing_row_forced(resource: Mapping[str, Any]) -> bool:
    requirements = resource.get("encounter_requirements")
    return isinstance(requirements, Mapping) and requirements.get("must_appear_on_list") is True


__all__ = [
    "_benign_read_url",
    "_created_child_listing_row_forced",
    "_creates_child_detail_surface",
    "_effective_mode_for_seeded_surface",
    "_mode_for_resource",
    "_transitive_entry_supported",
]
