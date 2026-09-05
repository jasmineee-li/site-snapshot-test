"""Exposure verification contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_2.exposure_contract.editor_args import _selector_hint


def _verification_contract(
    resource: Mapping[str, Any],
    mode: str,
    benign_read_url: str,
    kind: str,
) -> dict[str, Any]:
    verification: dict[str, Any] = {
        "url": benign_read_url,
        "entry": {"url": benign_read_url, "role": "benign_entry"},
        "mode": mode,
        "selector_hint": _selector_hint(kind, mode),
        "witness_policy": "signature_plus_second_witness",
    }
    if mode == "bounded_transitive_existing":
        target_url = resource.get("seeded_detail_url") or resource.get("start_url_resolved")
        if isinstance(target_url, str) and target_url.strip():
            verification["target"] = {
                "url": target_url,
                "role": "seed_render_surface",
            }
        verification["transition"] = {
            "type": "visible_link",
            "max_depth": 1,
            "same_origin": True,
            "edge_role": _edge_role_for_kind(kind),
        }
    elif mode == "inline_listing_created_child":
        verification["target"] = {
            "url_source": "seed_metadata.created_resource.parent_url",
            "role": "seed_render_surface",
        }
        verification["transition"] = {
            "type": "inline_listing_row",
            "max_depth": 0,
            "same_origin": True,
            "edge_role": "created_child_listing_row",
        }
    elif mode == "bounded_transitive_created_child":
        verification["target"] = {
            "url_source": "seed_metadata.created_resource.url",
            "role": "seed_render_surface",
        }
        verification["transition"] = {
            "type": "visible_link",
            "max_depth": 1,
            "same_origin": True,
            "edge_role": "created_child_detail",
        }
    return verification


def _edge_role_for_kind(kind: str) -> str:
    if kind == "gitlab_issue":
        return "issue_detail"
    if kind == "gitlab_mr":
        return "merge_request_detail"
    if kind == "reddit_submission":
        return "submission_detail"
    if kind == "gitlab_user_profile":
        return "user_profile_detail"
    if kind == "gitlab_snippet":
        return "snippet_detail"
    if kind == "gitlab_project_milestone":
        return "milestone_detail"
    if kind == "gitlab_group":
        return "group_detail"
    return "detail"


__all__ = [
    "_edge_role_for_kind",
    "_verification_contract",
]
