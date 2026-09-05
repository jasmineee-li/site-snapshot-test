"""Task metadata labels used by Phase 4 summaries."""

from __future__ import annotations

from typing import Any


def task_site(result: dict[str, Any], task: dict[str, Any] | None) -> str:
    if isinstance(task, dict) and isinstance(task.get("site"), str):
        return str(task["site"])
    if isinstance(result.get("site"), str):
        return str(result["site"])
    return "unknown"


def task_origin(result: dict[str, Any], task: dict[str, Any] | None) -> str:
    if isinstance(task, dict) and task.get("origin") in {"existing_task", "new_task"}:
        return str(task["origin"])
    if result.get("origin") in {"existing_task", "new_task"}:
        return str(result["origin"])
    task_id = str(result.get("task_id", ""))
    if task_id.startswith("novel_") or task_id.startswith("adv_novel_"):
        return "new_task"
    if task_id:
        return "existing_task"
    return "unknown"


def task_scenario_template(task: dict[str, Any] | None) -> str:
    if not isinstance(task, dict):
        return "unknown"
    provenance = task.get("task_provenance")
    if isinstance(provenance, dict):
        template = provenance.get("scenario_template")
        if isinstance(template, dict):
            template_id = template.get("id")
            if isinstance(template_id, str) and template_id.strip():
                return template_id.strip()
    return "unknown"


def task_surface(task: dict[str, Any] | None) -> str:
    contract = task.get("exposure_contract") if isinstance(task, dict) else None
    if not isinstance(contract, dict):
        return "unknown"
    surface_id = contract.get("target_surface_id")
    if isinstance(surface_id, str) and surface_id:
        return surface_id
    surface = contract.get("target_surface")
    if isinstance(surface, dict):
        sid = surface.get("id")
        if isinstance(sid, str) and sid:
            return sid
    if isinstance(surface, str) and surface:
        return surface
    return "unknown"


def task_editor_method(task: dict[str, Any] | None) -> str:
    contract = task.get("exposure_contract") if isinstance(task, dict) else None
    if isinstance(contract, dict) and isinstance(contract.get("editor_method"), str):
        return str(contract["editor_method"])
    return "unknown"


def task_route_variant(task: dict[str, Any] | None) -> str:
    contract = task.get("exposure_contract") if isinstance(task, dict) else None
    if not isinstance(contract, dict):
        return "unknown"
    surface_route = contract.get("surface_route")
    surface_route = surface_route if isinstance(surface_route, dict) else {}
    for value in (
        contract.get("route_variant"),
        surface_route.get("route_variant"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    site = contract.get("site")
    kind = contract.get("kind")
    target_surface_id = task_surface(task)
    entry_surface = surface_route.get("entry_surface") or contract.get("source_listing_kind")
    requires_transition = surface_route.get("requires_transition") is True or (
        isinstance(contract.get("phase4_exposure"), dict)
        and contract["phase4_exposure"].get("requires_transition") is True
    )
    if site == "reddit":
        if kind == "reddit_forum" or entry_surface == "reddit_forum":
            if requires_transition or target_surface_id == "submission.body":
                return "forum_to_submission_detail"
            if target_surface_id == "submission.title":
                return "forum_listing"
            return "forum_listing"
        if kind == "reddit_submission":
            if entry_surface == "reddit_forum":
                return "forum_to_submission_detail"
            if target_surface_id == "comment.body" or target_surface_id.endswith(".comment"):
                return "submission_detail_comment_region"
            return "submission_detail"
        if kind == "reddit_dashboard_list" or entry_surface == "reddit_dashboard_list":
            return "dashboard_listing"
    anchors = contract.get("anchors")
    if (
        site == "gitlab"
        and kind == "gitlab_search_result"
        and isinstance(anchors, dict)
        and isinstance(anchors.get("project_path"), str)
        and anchors["project_path"].strip()
    ):
        return "project_issue_list"
    return "unknown"


__all__ = [
    "task_editor_method",
    "task_origin",
    "task_route_variant",
    "task_scenario_template",
    "task_site",
    "task_surface",
]
