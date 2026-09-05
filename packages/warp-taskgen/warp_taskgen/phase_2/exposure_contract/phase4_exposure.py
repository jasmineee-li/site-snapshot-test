"""Phase 4 exposure admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_2.exposure_contract.modes import _created_child_listing_row_forced


def _phase4_exposure_capability(
    mode: str,
    *,
    reason: str | None = None,
    transition_forced_by_task: bool = False,
    runtime_hook_available: bool = False,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "seed_timing": "pre_task",
    }
    if mode == "direct_detail":
        base.update(
            {
                "admissible": True,
                "encounter_surface": "benign_read_surface",
                "requires_transition": False,
                "transition_forced_by_task": True,
                "requires_runtime_hook": False,
            }
        )
        return base
    if mode == "inline_listing":
        base.update(
            {
                "admissible": True,
                "encounter_surface": "benign_read_surface",
                "requires_transition": False,
                "transition_forced_by_task": True,
                "requires_runtime_hook": False,
                "requires_visual_evidence": True,
            }
        )
        return base
    if mode == "inline_listing_created_child":
        base.update(
            {
                "admissible": True,
                "encounter_surface": "seed_created_child_listing_row",
                "requires_transition": False,
                "transition_forced_by_task": True,
                "requires_runtime_hook": False,
                "requires_visual_evidence": True,
            }
        )
        return base
    if mode == "bounded_transitive_existing":
        admissible = transition_forced_by_task or runtime_hook_available
        base.update(
            {
                "admissible": admissible,
                "encounter_surface": "seed_existing_child_detail",
                "requires_transition": True,
                "transition_forced_by_task": transition_forced_by_task,
                "requires_runtime_hook": runtime_hook_available and not transition_forced_by_task,
            }
        )
        if not admissible:
            base["reason"] = reason or "unforced_transitive_surface"
        return base
    if mode == "bounded_transitive_created_child":
        admissible = transition_forced_by_task or runtime_hook_available
        base.update(
            {
                "admissible": admissible,
                "encounter_surface": "seed_created_child_detail",
                "requires_transition": True,
                "transition_forced_by_task": transition_forced_by_task,
                "requires_runtime_hook": runtime_hook_available and not transition_forced_by_task,
            }
        )
        if not admissible:
            base["reason"] = reason or "unforced_transitive_child_surface"
        return base
    base.update(
        {
            "admissible": False,
            "reason": reason or "unsupported_exposure_topology",
            "encounter_surface": "none",
            "requires_transition": False,
            "transition_forced_by_task": False,
            "requires_runtime_hook": False,
        }
    )
    return base


def _phase4_exposure_for_seeded_surface(
    *,
    mode: str,
    site: str,
    kind: str,
    editor_method: str,
    target_surface_id: str,
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    transition_forced = _transition_forced_by_task(resource)
    runtime_hook = _phase4_runtime_hook_available(resource)
    if _is_title_surface(target_surface_id) and not _title_surface_forced_by_task(resource):
        return _phase4_exposure_capability(
            "ineligible",
            reason="task_does_not_force_title_surface",
        )
    if _is_unanchored_appended_comment_surface(
        site=site,
        kind=kind,
        editor_method=editor_method,
        target_surface_id=target_surface_id,
    ):
        return _appended_comment_exposure_capability(
            mode,
            site=site,
            transition_forced_by_task=transition_forced,
            exact_comment_region_forced=_exact_comment_region_forced_by_task(resource),
            visual_comment_region_forced=_visual_comment_region_forced_by_task(resource),
            seeded_comment_visible_forced=_seeded_comment_visible_forced_by_task(resource),
            runtime_hook_available=runtime_hook,
        )
    return _phase4_exposure_capability(
        mode,
        transition_forced_by_task=transition_forced,
        runtime_hook_available=runtime_hook,
    )


def _is_title_surface(target_surface_id: str | None) -> bool:
    return target_surface_id in {"issue.title", "mr.title", "submission.title"}


def _appended_comment_exposure_capability(
    mode: str,
    *,
    site: str,
    transition_forced_by_task: bool,
    exact_comment_region_forced: bool,
    visual_comment_region_forced: bool,
    seeded_comment_visible_forced: bool,
    runtime_hook_available: bool,
) -> dict[str, Any]:
    requires_visual_comment_region = site == "reddit"
    requires_seeded_comment_visibility = site == "reddit"
    visual_gate_satisfied = not requires_visual_comment_region or visual_comment_region_forced
    seeded_visibility_gate_satisfied = (
        not requires_seeded_comment_visibility or seeded_comment_visible_forced
    )
    task_forces_required_comment_exposure = (
        exact_comment_region_forced and visual_gate_satisfied and seeded_visibility_gate_satisfied
    )
    admissible = task_forces_required_comment_exposure or runtime_hook_available
    base: dict[str, Any] = {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": admissible,
        "encounter_surface": "seed_appended_comment_region",
        "requires_transition": mode != "direct_detail",
        "transition_forced_by_task": transition_forced_by_task,
        "requires_exact_comment_region": True,
        "exact_comment_region_forced_by_task": exact_comment_region_forced,
        "requires_runtime_hook": runtime_hook_available
        and not task_forces_required_comment_exposure,
    }
    if requires_visual_comment_region:
        base["requires_visual_comment_region"] = True
        base["visual_comment_region_forced_by_task"] = visual_comment_region_forced
    if requires_seeded_comment_visibility:
        base["requires_seeded_comment_visibility"] = True
        base["seeded_comment_visible_forced_by_task"] = seeded_comment_visible_forced
    if not admissible:
        if (
            exact_comment_region_forced
            and requires_visual_comment_region
            and not visual_gate_satisfied
        ):
            base["reason"] = "reddit_comment_region_not_visually_forced"
        elif (
            exact_comment_region_forced
            and visual_gate_satisfied
            and requires_seeded_comment_visibility
            and not seeded_visibility_gate_satisfied
        ):
            base["reason"] = "reddit_seeded_comment_not_visibility_forced"
        else:
            base["reason"] = "unanchored_appended_comment_surface"
    return base


def _is_unanchored_appended_comment_surface(
    *,
    site: str,
    kind: str,
    editor_method: str,
    target_surface_id: str,
) -> bool:
    # Appended discussion entries are not equivalent to the primary read
    # surface. A benign task can force the agent to an issue/submission page
    # without forcing it through a newly seeded sibling note/comment.
    if (
        site == "reddit"
        and kind in {"reddit_submission", "reddit_dashboard_list"}
        and editor_method == "create_comment"
        and target_surface_id == "comment.body"
    ):
        return True
    return (
        site == "gitlab"
        and kind in {"gitlab_issue", "gitlab_mr", "gitlab_search_result", "gitlab_dashboard_list"}
        and editor_method in {"create_issue_note", "create_mr_note"}
        and target_surface_id == "note.body"
    )


def _eligibility_from_capabilities(
    seed_capability: Mapping[str, Any],
    phase4_exposure: Mapping[str, Any],
) -> dict[str, str]:
    if seed_capability.get("status") != "supported":
        return {
            "status": "ineligible",
            "reason": str(seed_capability.get("reason") or "seed_capability_unsupported"),
        }
    if phase4_exposure.get("admissible") is not True:
        reason = str(phase4_exposure.get("reason") or "phase4_exposure_inadmissible")
        return {
            "status": "ineligible",
            "reason": f"phase4_exposure:{reason}",
        }
    return {"status": "eligible"}


def _transition_forced_by_task(resource: Mapping[str, Any]) -> bool:
    return resource.get("transition_forced_by_task") is True


def _title_surface_forced_by_task(resource: Mapping[str, Any]) -> bool:
    return resource.get(
        "title_surface_forced_by_task"
    ) is True or _created_child_listing_row_forced(resource)


def _exact_comment_region_forced_by_task(resource: Mapping[str, Any]) -> bool:
    return resource.get("exact_comment_region_forced_by_task") is True


def _visual_comment_region_forced_by_task(resource: Mapping[str, Any]) -> bool:
    return resource.get("visual_comment_region_forced_by_task") is True


def _seeded_comment_visible_forced_by_task(resource: Mapping[str, Any]) -> bool:
    return resource.get("seeded_comment_visible_forced_by_task") is True


def _phase4_runtime_hook_available(resource: Mapping[str, Any]) -> bool:
    return resource.get("phase4_runtime_hook_available") is True


__all__ = [
    "_appended_comment_exposure_capability",
    "_eligibility_from_capabilities",
    "_exact_comment_region_forced_by_task",
    "_is_title_surface",
    "_is_unanchored_appended_comment_surface",
    "_phase4_exposure_capability",
    "_phase4_exposure_for_seeded_surface",
    "_phase4_runtime_hook_available",
    "_seeded_comment_visible_forced_by_task",
    "_title_surface_forced_by_task",
    "_transition_forced_by_task",
    "_visual_comment_region_forced_by_task",
]
