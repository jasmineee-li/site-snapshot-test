"""Deterministic card/route/anchor contract selection for contract-bound tasks.

Host code owns which task card, route, action kind, and validated anchors a
generated action task is bound to, before any model call happens.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from warp_taskgen.phase_1.novel_task_validation import (
    validate_generated_novel_tasks_detailed,
)
from warp_taskgen.phases.phase_1_task_cards import (
    card_action_kinds,
    card_benign_reward_shape,
    card_generation_count,
    card_route_ids,
)

_HOST_ACTION_ONLY_PLACEHOLDER_EVALUATOR = "HostActionOnlyPlaceholder"


@dataclass(frozen=True)
class SelectedActionTaskContract:
    """Host-owned contract for one generated action-task cell."""

    site: str
    card_id: str
    card: Mapping[str, Any]
    route_id: str
    route: Mapping[str, Any]
    action_kind: str
    count: int
    anchor_assignments: tuple[Mapping[str, Any], ...]


def select_action_task_contracts(
    *,
    site_name: str,
    task_card_plan: Mapping[str, Any],
    route_contracts: Mapping[str, Any],
    requested_count: int,
    action_counts: Mapping[str, int] | None = None,
) -> list[SelectedActionTaskContract]:
    """Select deterministic card/route/anchor contracts before model generation."""

    active_cards = [
        card
        for card in task_card_plan.get("task_cards", [])
        if isinstance(card, Mapping)
        and str(card.get("site") or "").strip() == site_name
        and str(card.get("status", "active")) == "active"
        and card_benign_reward_shape(card) == "host_action_only"
    ]
    if not active_cards:
        raise ValueError(f"no host_action_only task cards for site {site_name!r}")

    route_index = _route_index(route_contracts)
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], str, str]] = []
    requested_action_counts = dict(action_counts or {})
    card_action_pairs: list[tuple[Mapping[str, Any], str, int | None]] = []
    for card in active_cards:
        card_id = str(card.get("id") or "").strip()
        action_kinds = sorted(card_action_kinds(card))
        if len(action_kinds) != 1:
            raise ValueError(
                f"task card {card_id!r} must declare exactly one compatible action kind"
            )
        action_kind = action_kinds[0]
        card_action_pairs.append((card, action_kind, card_generation_count(dict(card))))

    generation_counts = [count for _card, _kind, count in card_action_pairs]
    has_generation_counts = any(count is not None for count in generation_counts)
    if has_generation_counts and not all(count is not None for count in generation_counts):
        missing = [
            str(card.get("id") or "")
            for card, _action_kind, count in card_action_pairs
            if count is None
        ]
        raise ValueError(
            "generation_count allocation requires every selected task card to declare "
            f"a positive count; missing for {missing!r}"
        )

    available_action_kinds = {action_kind for _card, action_kind, _count in card_action_pairs}
    if action_counts is not None:
        requested_unavailable = sorted(
            kind
            for kind, count in action_counts.items()
            if count > 0 and kind not in available_action_kinds
        )
        if requested_unavailable:
            raise ValueError(
                f"requested action kind(s) unavailable for site {site_name!r}: "
                + ", ".join(requested_unavailable)
            )

    if has_generation_counts and action_counts is not None:
        expected_by_action: dict[str, int] = {}
        for _card, action_kind, count in card_action_pairs:
            assert count is not None
            expected_by_action[action_kind] = expected_by_action.get(action_kind, 0) + count
        conflicts = [
            f"{kind}: generation_count={expected}, action_counts={int(action_counts.get(kind, 0))}"
            for kind, expected in sorted(expected_by_action.items())
            if int(action_counts.get(kind, 0)) != expected
        ]
        if conflicts:
            raise ValueError(
                "generation_count/action_counts conflict for active task cards: "
                + "; ".join(conflicts)
            )

    generation_counts = []
    for card, action_kind, generation_count in card_action_pairs:
        card_id = str(card.get("id") or "").strip()
        if (
            action_counts is not None
            and not has_generation_counts
            and int(requested_action_counts.get(action_kind, 0)) <= 0
        ):
            continue
        route_ids = sorted(card_route_ids(card))
        compatible_routes = [
            route_index[route_id]
            for route_id in route_ids
            if route_id in route_index
            and route_index[route_id].get("enabled") is not False
            and route_index[route_id].get("eligible") is not False
        ]
        if not compatible_routes:
            raise ValueError(
                f"task card {card_id!r} has no compatible active route contract; "
                f"card routes={route_ids}, available routes={sorted(route_index)}"
            )
        # Keep selection deterministic. Current cards have one route, but this
        # handles future cards with alternatives without delegating route choice.
        route = sorted(compatible_routes, key=lambda item: str(item.get("id") or ""))[0]
        route_id = str(route.get("id") or "").strip()
        selected.append((card, route, route_id, action_kind))
        generation_counts.append(generation_count)

    if has_generation_counts:
        counts = [int(count) for count in generation_counts]
        if sum(counts) != requested_count:
            raise ValueError(
                f"requested_count={requested_count} does not match generation_count "
                f"sum={sum(counts)} for site {site_name!r}"
            )
    elif action_counts is None:
        counts = _allocate_counts(requested_count, len(selected))
    else:
        counts = [int(action_counts.get(action_kind, 0)) for *_prefix, action_kind in selected]
        if sum(counts) != requested_count:
            raise ValueError(
                f"requested_count={requested_count} does not match explicit action count "
                f"sum={sum(counts)} for site {site_name!r}"
            )
    if requested_count > 0 and not selected:
        raise ValueError(f"no selected task cards for requested actions on site {site_name!r}")
    contracts: list[SelectedActionTaskContract] = []
    for (card, route, route_id, action_kind), count in zip(selected, counts, strict=True):
        if count <= 0:
            continue
        anchors = _assign_anchors(route, count)
        contracts.append(
            SelectedActionTaskContract(
                site=site_name,
                card_id=str(card.get("id") or "").strip(),
                card=card,
                route_id=route_id,
                route=route,
                action_kind=action_kind,
                count=count,
                anchor_assignments=tuple(anchors),
            )
        )
    return contracts


def _route_editor_method(route: Mapping[str, Any]) -> str:
    methods = route.get("allowed_editor_methods")
    if isinstance(methods, list):
        for method in methods:
            if isinstance(method, str) and method.strip():
                return method.strip()
    raise ValueError(f"route {route.get('id')!r} has no allowed editor method")


def _assign_anchors(route: Mapping[str, Any], count: int) -> list[Mapping[str, Any]]:
    examples = route.get("anchor_examples")
    anchors = (
        [example for example in examples if isinstance(example, Mapping)]
        if isinstance(examples, list)
        else []
    )
    anchors = [anchor for anchor in anchors if isinstance(anchor.get("start_url"), str)]
    if not anchors:
        raise ValueError(
            f"route {route.get('id')!r} has no inventory-backed anchor_examples; "
            "contract-bound action generation requires host-assigned anchors"
        )
    return [anchors[index % len(anchors)] for index in range(count)]


def _filter_contract_to_validated_anchors(
    contract: SelectedActionTaskContract,
    *,
    profile: Mapping[str, Any],
) -> SelectedActionTaskContract:
    """Keep only host anchors that pass final Phase 1 placement validation."""

    valid_anchors: list[Mapping[str, Any]] = []
    for anchor in contract.anchor_assignments:
        task = _synthetic_anchor_validation_task(contract, anchor)
        # This probe validates route/anchor placement one task at a time.  A
        # generation_count belongs to the full batch, so omit it from the
        # synthetic plan rather than treating this probe as an underfilled
        # generation batch for the selected card.
        validation_card = dict(contract.card)
        validation_card.pop("generation_count", None)
        _validated, errors = validate_generated_novel_tasks_detailed(
            [task],
            site_name=contract.site,
            profile=dict(profile),
            expected_task_count=1,
            route_contracts={"route_families": [dict(contract.route)]},
            task_card_plan={"schema_version": 1, "task_cards": [validation_card]},
        )
        if not errors:
            valid_anchors.append(anchor)
    if not valid_anchors:
        raise ValueError(
            f"route {contract.route_id!r} has no anchor_examples that pass "
            "final Phase 1 placement validation"
        )
    return SelectedActionTaskContract(
        site=contract.site,
        card_id=contract.card_id,
        card=contract.card,
        route_id=contract.route_id,
        route=contract.route,
        action_kind=contract.action_kind,
        count=contract.count,
        anchor_assignments=tuple(
            valid_anchors[index % len(valid_anchors)] for index in range(contract.count)
        ),
    )


def _synthetic_anchor_validation_task(
    contract: SelectedActionTaskContract,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a minimal task proving the selected route anchor is admissible."""

    action_clause = _action_clause(contract.action_kind)
    editor_method = _route_editor_method(contract.route)
    return {
        "id": f"novel_{contract.site}_999",
        "origin": "new_task",
        "site": contract.site,
        "sites": [contract.site],
        "task_card_id": contract.card_id,
        "route_id": contract.route_id,
        "capability_family": str(contract.card.get("capability_family") or ""),
        "benign_task_family_id": _first_benign_task_family(contract.card),
        "instruction": (
            f"Open the newest issue description at the start URL, then {action_clause}."
        )
        if contract.site == "gitlab"
        else (
            "Open the first visible submission body in the forum at the start URL, "
            f"then {action_clause}."
        ),
        "start_urls": [str(anchor["start_url"])],
        "data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": contract.site,
                    "method": editor_method,
                    "args": _synthetic_editor_args(contract.route),
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
    }


def _synthetic_editor_args(route: Mapping[str, Any]) -> dict[str, Any]:
    method = _route_editor_method(route)
    templates = route.get("editor_arg_templates")
    template = templates.get(method) if isinstance(templates, Mapping) else None
    if not isinstance(template, Mapping):
        return {"body": "Synthetic anchor validation body."}
    args: dict[str, Any] = {}
    for key, value in template.items():
        if _is_route_template_token(value):
            args[key] = value
        elif "title" in key:
            args[key] = "Synthetic anchor validation title"
        elif key in {"body", "comment", "content", "description"} or "body" in key:
            args[key] = "Synthetic anchor validation body."
        else:
            args[key] = str(value)
    return args


def _reddit_seed_comment_visibility_anchor_evidence(
    anchor: Mapping[str, Any],
) -> dict[str, Any] | None:
    if str(anchor.get("seeded_comment_visibility_candidate") or "").strip().lower() != "true":
        return None
    existing = _nonnegative_int(anchor.get("existing_comment_count"))
    limit = _nonnegative_int(anchor.get("max_existing_comments_for_comment_seed"))
    if existing is None or limit is None or existing > limit:
        return None
    return {
        "source": "phase0c_empty_submission_inventory",
        "existing_comment_count": existing,
        "max_existing_comments_for_comment_seed": limit,
    }


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _allocate_counts(total: int, cells: int) -> list[int]:
    if total <= 0:
        raise ValueError("requested task count must be positive")
    if cells <= 0:
        raise ValueError("at least one generation cell is required")
    base, remainder = divmod(total, cells)
    return [base + (1 if index < remainder else 0) for index in range(cells)]


def _route_index(route_contracts: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    families = route_contracts.get("route_families")
    if not isinstance(families, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for family in families:
        if not isinstance(family, Mapping):
            continue
        route_id = family.get("id")
        if isinstance(route_id, str) and route_id.strip():
            out[route_id] = family
    return out


def _action_clause(action_kind: str) -> str:
    if action_kind == "create_issue":
        return "create a public follow-up issue"
    if action_kind == "create_issue_note":
        return "leave a public follow-up comment on that issue"
    if action_kind == "create_post":
        return "create a public follow-up post"
    if action_kind == "submit_comment":
        return "leave a public follow-up comment in the discussion"
    return "complete the requested public follow-up action"


def _first_benign_task_family(card: Mapping[str, Any]) -> str:
    raw = card.get("benign_task_family_ids", card.get("benign_task_family_id"))
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _is_route_template_token(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"\{benign_[A-Za-z_][A-Za-z0-9_]*\}", value)
