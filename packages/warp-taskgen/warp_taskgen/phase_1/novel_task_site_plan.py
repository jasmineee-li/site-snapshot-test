"""Phase 1 novel-task site plan: eligible sites, site filters, and per-site counts."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warp_taskgen.phases.phase_1_route_contracts import build_task_route_contracts
from warp_taskgen.phases.phase_1_task_cards import (
    card_action_kinds,
    task_card_generation_counts,
    task_card_plan_for_site,
)
from warp_taskgen.placeholders import normalize_site_name
from warp_taskgen.profile_validation import load_and_validate_profile

logger = logging.getLogger(__name__)

DEFAULT_NOVEL_TASKS_PER_SITE = 30


@dataclass(frozen=True)
class EligibleSiteProfile:
    site_name: str
    profile_path: Path
    profile: dict[str, Any]


def load_generate_new_tasks_eligible_sites(
    *,
    profiles_dir: Path,
    manifest_eval_types: list[str],
    site_filter: Iterable[str] | None = None,
) -> list[EligibleSiteProfile]:
    """Return profiles with Phase 4-admissible carrier route families."""
    eligible: list[EligibleSiteProfile] = []
    site_filter_set = _normalize_site_filter(site_filter)

    for profile_path in sorted(profiles_dir.glob("BENCHMARK_PROFILE_*.json")):
        site_name = profile_path.stem.removeprefix("BENCHMARK_PROFILE_")
        if site_filter_set is not None and site_name not in site_filter_set:
            logger.info(
                "Phase 1 (generate-new-tasks): skipping site %r due to --sites filter", site_name
            )
            continue
        profile = load_and_validate_profile(
            site_name,
            profile_path,
            manifest_eval_types=manifest_eval_types,
        )
        route_contracts = build_task_route_contracts(site_name=site_name, profile=profile)
        if route_contracts.get("route_families"):
            eligible.append(
                EligibleSiteProfile(
                    site_name=site_name,
                    profile_path=profile_path,
                    profile=profile,
                )
            )
        else:
            logger.info(
                "Phase 1 (generate-new-tasks): skipping site %r (no Phase 4-admissible carrier route families)",
                site_name,
            )

    return eligible


def _normalize_site_filter(site_filter: Iterable[str] | None) -> set[str] | None:
    if site_filter is None:
        return None
    normalized = {
        normalize_site_name(str(site).strip()) for site in site_filter if str(site).strip()
    }
    return normalized or None


def _fail_if_requested_sites_ineligible(
    *,
    site_filter: Iterable[str] | None,
    eligible_sites: list[EligibleSiteProfile],
) -> None:
    requested = _normalize_site_filter(site_filter)
    if requested is None:
        return
    eligible = {site.site_name for site in eligible_sites}
    missing = sorted(requested - eligible)
    if not missing:
        return
    raise RuntimeError(
        "Phase 1 (generate-new-tasks) cannot satisfy the requested site filter because "
        "the following selected site(s) have no Phase 4-admissible carrier route "
        f"families: {', '.join(missing)}. This usually means Phase 0c did not produce "
        "live inventory for inventory-backed routes, or the site has no strict-WASP "
        "carrier surface under the current route contracts. Fix the profile/inventory "
        "source and rerun Phase 0c rather than silently generating fewer sites."
    )


def _fail_if_task_card_plan_missing_sites(
    *,
    task_card_plan: dict[str, Any] | None,
    eligible_sites: list[EligibleSiteProfile],
) -> None:
    """Reject card-guided generation that would mix with legacy site generation."""
    if task_card_plan is None:
        return
    missing = [
        site.site_name
        for site in eligible_sites
        if task_card_plan_for_site(task_card_plan, site.site_name) is None
    ]
    if not missing:
        return
    active_sites = sorted(
        {
            str(card.get("site")).strip()
            for card in task_card_plan.get("task_cards", [])
            if isinstance(card, dict)
            and str(card.get("status", "active")) == "active"
            and str(card.get("site") or "").strip()
        }
    )
    raise RuntimeError(
        "Phase 1 task-card-guided generation cannot silently fall back to legacy "
        "generation for site(s) without active task cards: "
        + ", ".join(sorted(missing))
        + ". Requested/eligible generated sites must be a subset of the task-card "
        f"plan sites: {', '.join(active_sites) or '<none>'}."
    )


def _action_counts_for_site(
    task_card_plan: dict[str, Any] | None,
    action_counts: dict[str, int] | None,
) -> dict[str, int] | None:
    if action_counts is None:
        return None
    if not isinstance(task_card_plan, dict):
        return {}
    available: set[str] = set()
    for card in task_card_plan.get("task_cards", []):
        if not isinstance(card, dict) or str(card.get("status", "active")) != "active":
            continue
        available.update(card_action_kinds(card))
    return {kind: count for kind, count in action_counts.items() if kind in available}


def _fail_if_action_counts_unavailable(
    *,
    site_plans: dict[str, dict[str, Any] | None],
    action_counts: dict[str, int] | None,
) -> None:
    if action_counts is None:
        return
    available: set[str] = set()
    for plan in site_plans.values():
        if not isinstance(plan, dict):
            continue
        for card in plan.get("task_cards", []):
            if not isinstance(card, dict) or str(card.get("status", "active")) != "active":
                continue
            available.update(card_action_kinds(card))
    unavailable = sorted(
        kind for kind, count in action_counts.items() if count > 0 and kind not in available
    )
    if unavailable:
        raise ValueError(
            "requested action kind(s) unavailable for selected sites/task-card plan: "
            + ", ".join(unavailable)
        )


def _site_requested_count(
    task_card_plan: dict[str, Any] | None,
    *,
    novel_tasks_per_site: int,
    action_counts: dict[str, int] | None,
) -> int:
    generation_counts = task_card_generation_counts(task_card_plan)
    if generation_counts is not None:
        _validate_generation_count_action_counts(
            task_card_plan=task_card_plan,
            action_counts=action_counts,
        )
        return sum(generation_counts.values())
    if action_counts is None:
        return novel_tasks_per_site
    return sum(_action_counts_for_site(task_card_plan, action_counts).values())


def _validate_generation_count_action_counts(
    *,
    task_card_plan: dict[str, Any] | None,
    action_counts: dict[str, int] | None,
) -> None:
    """Fail closed when explicit action counts disagree with card quotas."""
    if action_counts is None:
        return
    generation_counts = task_card_generation_counts(task_card_plan)
    if generation_counts is None:
        return

    expected_by_action: dict[str, int] = {}
    for card in (task_card_plan or {}).get("task_cards", []):
        if not isinstance(card, dict) or str(card.get("status", "active")) != "active":
            continue
        card_id = str(card.get("id") or "")
        count = generation_counts.get(card_id)
        if count is None:
            continue
        action_kinds = card_action_kinds(card)
        if len(action_kinds) > 1:
            raise ValueError(
                "generation_count/action_counts conflict: multi-action task card "
                f"{card_id!r} maps one quota to multiple action kinds "
                f"{list(action_kinds)!r}; action_counts cannot disambiguate it"
            )
        if len(action_kinds) == 1:
            expected_by_action[action_kinds[0]] = expected_by_action.get(action_kinds[0], 0) + count

    conflicts: list[str] = []
    for action_kind, expected in sorted(expected_by_action.items()):
        actual = int(action_counts.get(action_kind, 0))
        if actual != expected:
            conflicts.append(f"{action_kind}: generation_count={expected}, action_counts={actual}")
    if conflicts:
        raise ValueError(
            "generation_count/action_counts conflict for active task cards: "
            + "; ".join(conflicts)
            + ". Remove action_counts or make each action-kind total match its card quotas."
        )
