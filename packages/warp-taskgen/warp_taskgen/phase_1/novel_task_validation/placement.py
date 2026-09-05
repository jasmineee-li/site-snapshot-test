"""Phase 2 placement-target validation exports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.novel_task_validation.answer_stability import (
    _task_route_targets_reddit_comment_body,
)
from warp_taskgen.phase_2.exposure_contract import build_exposure_contract
from warp_taskgen.phase_2.target_resolution.constants import _REDDIT_COMMENT_VISUAL_REGION_RE
from warp_taskgen.phase_2.target_resolution.runner import derive_benign_target_resource
from warp_taskgen.placeholders import placeholder_for_site

_DISALLOWED_SAMPLE_EDITOR_VALUES = frozenset({"Seeded title", "Seeded body"})


def _validate_phase2_placement_target(
    task: dict[str, Any],
    *,
    site_name: str,
    route_index: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Require generated tasks to target pages Phase 2 can seed and verify.

    generate-new-tasks is useful only when the benign task naturally traverses the
    same surface Phase 2 can seed. This calls the same deterministic
    resolver and exposure-contract builder Phase 2 uses, so future
    benchmark support extends by adding placement contracts rather than
    broadening this validator by hand.
    """
    if site_name not in {"gitlab", "reddit"}:
        return None
    placeholder = placeholder_for_site(site_name)
    if placeholder is None:
        return f"site {site_name!r} has no placeholder mapping"
    placeholders = {placeholder: f"https://{site_name}.local"}
    resource = derive_benign_target_resource(task, placeholders)
    _merge_route_contract_editor_anchors(resource, task, route_index)
    if site_name == "reddit" and _task_route_targets_reddit_comment_body(task):
        # Route selection can prove that the benign task makes the comments
        # region salient. Seed-specific visibility is admitted only when the
        # selected inventory anchor carries host-owned empty-thread evidence;
        # Phase 2c/runtime evidence remains the authority for counted rows.
        resource["exact_comment_region_forced_by_task"] = True
        resource["visual_comment_region_forced_by_task"] = True
        if _resource_has_reddit_seed_visibility_anchor_evidence(resource):
            resource["seeded_comment_visible_forced_by_task"] = True
    elif (
        site_name == "reddit"
        and resource.get("kind") in {"reddit_submission", "reddit_dashboard_list"}
        and _REDDIT_COMMENT_VISUAL_REGION_RE.search(str(task.get("instruction") or ""))
    ):
        # Phase 1 admission for novel benign read tasks without a comment-body
        # data_seed: the instruction forces the agent into the visual comment
        # region, so Phase 2's future seed placement is structurally admissible.
        # Phase 2c remains the runtime authority for verifying that the actual
        # seeded comment is in the painted region during the agent trajectory.
        resource["seeded_comment_visible_forced_by_task"] = True
    contract = build_exposure_contract(
        benign_task_id=str(task.get("id") or ""),
        site=site_name,
        benchmark=str(task.get("benchmark") or "webarena_verified"),
        benign_target_resource=resource,
    )
    eligibility = contract.get("eligibility") if isinstance(contract, dict) else None
    if isinstance(eligibility, dict) and eligibility.get("status") == "eligible":
        return None
    reason = "unknown"
    if isinstance(eligibility, dict):
        reason = str(eligibility.get("reason") or reason)
    elif isinstance(contract, dict):
        reason = str(contract.get("reason") or reason)
    return (
        "start_urls must resolve to an eligible Phase 2 exposure contract; "
        f"resolver kind={resource.get('kind')!r}, reason={reason!r}, "
        f"start_urls={task.get('start_urls')!r}"
    )


def _validate_no_literal_sample_editor_values(
    task: dict[str, Any],
) -> tuple[str, str, Any] | None:
    seed = task.get("data_seed")
    if not isinstance(seed, dict) or seed.get("mechanism") != "editor":
        return None
    calls = seed.get("editor_calls")
    if not isinstance(calls, list):
        return None
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        site = str(call.get("site") or task.get("site") or "").strip().lower()
        if site not in {"gitlab", "reddit"}:
            continue
        args = call.get("args")
        if not isinstance(args, dict):
            continue
        for arg_name, value in args.items():
            if value in _DISALLOWED_SAMPLE_EDITOR_VALUES:
                path = f"data_seed.editor_calls[{call_index}].args.{arg_name}"
                return (
                    path,
                    f"{path} must not copy literal route-contract sample value {value!r}",
                    value,
                )
    return None


def _merge_route_contract_editor_anchors(
    resource: dict[str, Any],
    task: dict[str, Any],
    route_index: dict[str, dict[str, Any]] | None,
) -> None:
    if route_index is None:
        return
    route_id = task.get("route_id")
    if not isinstance(route_id, str):
        return
    route = route_index.get(route_id)
    if not isinstance(route, dict):
        return

    anchors = dict(resource.get("anchors") or {})
    selected_anchor = _selected_route_anchor_example(route, task)
    if selected_anchor is not None:
        for key in (
            "project_id",
            "project_path",
            "issue_iid",
            "mr_iid",
            "forum_name",
            "submission_id",
        ):
            value = selected_anchor.get(key)
            if value not in (None, ""):
                anchors[key] = str(value)
        if _anchor_example_has_reddit_seed_visibility_evidence(selected_anchor):
            resource["reddit_seed_comment_visibility_anchor_evidence"] = {
                "source": "phase0c_empty_submission_inventory",
                "existing_comment_count": int(selected_anchor.get("existing_comment_count") or 0),
                "max_existing_comments_for_comment_seed": int(
                    selected_anchor.get("max_existing_comments_for_comment_seed") or 0
                ),
            }
    token_to_anchor = {
        "{benign_project_id}": "project_id",
        "{benign_project_path}": "project_path",
        "{benign_issue_iid}": "issue_iid",
        "{benign_mr_iid}": "mr_iid",
        "{benign_forum_name}": "forum_name",
        "{benign_submission_id}": "submission_id",
    }
    editor_arg_templates = route.get("editor_arg_templates")
    if isinstance(editor_arg_templates, dict):
        for template_args in editor_arg_templates.values():
            if not isinstance(template_args, dict):
                continue
            for value in template_args.values():
                if not isinstance(value, str):
                    continue
                anchor = token_to_anchor.get(value)
                if anchor is not None:
                    anchors.setdefault(anchor, "1")
    resource["anchors"] = anchors

    allowed = [
        method
        for method in route.get("allowed_editor_methods", [])
        if isinstance(method, str) and method.strip()
    ]
    if allowed:
        resource["allowed_editor_methods"] = allowed


def _selected_route_anchor_example(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    start_urls = task.get("start_urls")
    if not isinstance(start_urls, list):
        return None
    wanted = {url for url in start_urls if isinstance(url, str)}
    if not wanted:
        return None
    examples = route.get("anchor_examples")
    if not isinstance(examples, list):
        return None
    for example in examples:
        if not isinstance(example, Mapping):
            continue
        start_url = example.get("start_url")
        if isinstance(start_url, str) and start_url in wanted:
            return example
    return None


def _anchor_example_has_reddit_seed_visibility_evidence(anchor: Mapping[str, Any]) -> bool:
    if str(anchor.get("seeded_comment_visibility_candidate") or "").strip().lower() != "true":
        return False
    existing = _nonnegative_int(anchor.get("existing_comment_count"))
    limit = _nonnegative_int(anchor.get("max_existing_comments_for_comment_seed"))
    return existing is not None and limit is not None and existing <= limit


def _resource_has_reddit_seed_visibility_anchor_evidence(resource: Mapping[str, Any]) -> bool:
    evidence = resource.get("reddit_seed_comment_visibility_anchor_evidence")
    if not isinstance(evidence, Mapping):
        return False
    existing = _nonnegative_int(evidence.get("existing_comment_count"))
    limit = _nonnegative_int(evidence.get("max_existing_comments_for_comment_seed"))
    return existing is not None and limit is not None and existing <= limit


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


__all__ = [
    "_merge_route_contract_editor_anchors",
    "_validate_no_literal_sample_editor_values",
    "_validate_phase2_placement_target",
]
