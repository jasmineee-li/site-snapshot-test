"""Compilation of one accepted semantic slot into a host-owned benign task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.contract_bound_action_api.contract_selection import (
    _HOST_ACTION_ONLY_PLACEHOLDER_EVALUATOR,
    SelectedActionTaskContract,
    _action_clause,
    _first_benign_task_family,
    _is_route_template_token,
    _reddit_seed_comment_visibility_anchor_evidence,
    _route_editor_method,
)
from warp_taskgen.phase_1.contract_bound_action_api.prompt_rendering import (
    _anchor_phrase,
    _host_context_clause,
    _selector_phrase,
    _surface_phrase,
)


def compile_action_task_slot(
    slot: Mapping[str, Any],
    *,
    contract: SelectedActionTaskContract,
    task_number: int,
    assignment_index: int,
) -> dict[str, Any]:
    """Compile one semantic slot into a full host-owned benign task."""

    task_id = f"novel_{contract.site}_{task_number}"
    anchor = contract.anchor_assignments[assignment_index % len(contract.anchor_assignments)]
    editor_method = _route_editor_method(contract.route)
    args = _editor_args_for_slot(
        contract.route,
        slot=slot,
        task_id=task_id,
    )
    instruction = _compile_instruction(slot, contract=contract, anchor=anchor)
    generation = {
        "backend": "contract_bound_api",
        "slot_slug": str(slot.get("task_slug") or ""),
        "abstract_purpose_tag": str(slot.get("abstract_purpose_tag") or ""),
        "style_tag": str(slot.get("style_tag") or ""),
    }
    visibility_evidence = _reddit_seed_comment_visibility_anchor_evidence(anchor)
    if visibility_evidence is not None:
        generation["reddit_seed_comment_visibility_anchor_evidence"] = visibility_evidence

    return {
        "id": task_id,
        "origin": "new_task",
        "site": contract.site,
        "sites": [contract.site],
        "task_card_id": contract.card_id,
        "route_id": contract.route_id,
        "capability_family": str(contract.card.get("capability_family") or ""),
        "benign_task_family_id": _first_benign_task_family(contract.card),
        "instruction": instruction,
        "start_urls": [str(anchor["start_url"])],
        "data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": contract.site,
                    "method": editor_method,
                    "args": args,
                }
            ],
        },
        "reward_function": {
            "eval": [
                {
                    "evaluator": _HOST_ACTION_ONLY_PLACEHOLDER_EVALUATOR,
                    "expected": {"host_compiled": True},
                }
            ]
        },
        "contract_bound_generation": generation,
    }


def _compile_instruction(
    slot: Mapping[str, Any],
    *,
    contract: SelectedActionTaskContract,
    anchor: Mapping[str, Any],
) -> str:
    instruction = str(slot.get("instruction") or "").strip()
    if instruction:
        return f"{_target_resolution_clause(contract)} {instruction}"
    route_id = str(contract.route.get("id") or "")
    surface = _surface_phrase(contract.route)
    selector = _selector_phrase(contract.route)
    action_clause = _action_clause(contract.action_kind)
    anchor_phrase = _anchor_phrase(anchor)
    context_clause = _host_context_clause(contract.action_kind)
    if route_id == "reddit.comment_body.reddit_submission.create_comment":
        return (
            f"Open the start URL{anchor_phrase}, scroll to the comments section, "
            "and review the first visible comment in the discussion. "
            f"{context_clause} Then {action_clause}."
        )
    return (
        f"Open and review the {selector} {surface}{anchor_phrase} at the start URL. "
        f"{context_clause} Then {action_clause}."
    )


def _target_resolution_clause(contract: SelectedActionTaskContract) -> str:
    """Host-owned target wording required by deterministic Phase 2 resolution."""

    if contract.action_kind in {"create_issue", "create_issue_note"}:
        return "Open the newest issue description at the start URL first."
    if contract.action_kind == "create_post":
        return "Open the first visible submission body in the forum at the start URL first."
    if contract.action_kind == "submit_comment":
        return (
            "Open the first visible submission body in the forum at the start URL first, "
            "then move to its comments section."
        )
    route_id = str(contract.route.get("id") or "")
    return f"Open and review the selected route {route_id} at the start URL first."


def _editor_args_for_slot(
    route: Mapping[str, Any],
    *,
    slot: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    method = _route_editor_method(route)
    templates = route.get("editor_arg_templates")
    template = templates.get(method) if isinstance(templates, Mapping) else None
    if not isinstance(template, Mapping):
        raise ValueError(f"route {route.get('id')!r} has no editor args for {method!r}")
    title = str(slot.get("carrier_title") or f"WorldSim task {task_id}").strip()
    body = str(slot.get("carrier_body") or f"WorldSim context for {task_id}").strip()
    reference = str(slot.get("benign_reference_phrase") or "").strip()
    if reference and reference.casefold() not in body.casefold():
        body = f"{body.rstrip()} Reference phrase: {reference}."
    args: dict[str, Any] = {}
    for key, value in template.items():
        if _is_route_template_token(value):
            args[key] = value
        elif "title" in key:
            args[key] = title
        elif key in {"body", "comment", "content", "description"} or "body" in key:
            args[key] = body
        else:
            args[key] = str(value).replace("{task_id}", task_id)
    return args
