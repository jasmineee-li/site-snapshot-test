"""Phase 2 target resolution encounter metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldsim.editors._registry import attach_surfaces_for_kind as _registry_attach_surfaces
from worldsim.editors._registry import kind_contract as _registry_kind_contract
from worldsim.phase_2.target_resolution.constants import (
    _EXACT_DISCUSSION_REGION_RE,
    _LATEST_DISCUSSION_REGION_RE,
    _LISTING_DETAIL_FORCING_RE,
    _LISTING_PAGE_ONLY_RE,
    _LISTING_ROW_ACTION_RE,
    _REDDIT_COMMENT_VISUAL_REGION_RE,
    _TITLE_CONTENT_FORCING_RE,
    VIEWPORT_BUDGET_CHARS,
)
from worldsim.phase_2.target_resolution.types import (
    ResolverContractDriftError,
    ResourceKind,
)


def _assert_anchor_contract_conformance(
    record: Mapping[str, Any],
    *,
    benchmark: str = "webarena_verified",
    site: str | None = None,
) -> None:
    kind = record.get("kind")
    if kind is None:
        return  # pending/empty records - nothing to verify yet
    contract = _registry_kind_contract(str(kind), benchmark=benchmark, site=site)
    if not contract.valid_methods:
        raise ResolverContractDriftError(
            f"resolver emitted kind {kind!r} but no editor method addresses "
            f"it in the contract registry. Either add an @editor_method "
            f"with this kind in its `kinds` set, or stop emitting this kind."
        )


def _attach_surfaces_for(
    kind: ResourceKind,
    *,
    benchmark: str = "webarena_verified",
    site: str | None = None,
) -> list[dict[str, Any]]:
    return [
        dict(surface) for surface in _registry_attach_surfaces(kind, benchmark=benchmark, site=site)
    ]


def _route_evidence_flags(kind: ResourceKind | str, task: Mapping[str, Any]) -> dict[str, bool]:
    instruction = str(task.get("instruction") or "")
    if not instruction.strip():
        return {}
    flags: dict[str, bool] = {}
    if _title_surface_forced_by_instruction(instruction):
        flags["title_surface_forced_by_task"] = True
    if kind in {"reddit_forum", "gitlab_search_result", "gitlab_dashboard_list"}:
        if _LISTING_DETAIL_FORCING_RE.search(instruction):
            flags["transition_forced_by_task"] = True
        if _EXACT_DISCUSSION_REGION_RE.search(instruction) or _LATEST_DISCUSSION_REGION_RE.search(
            instruction
        ):
            flags["transition_forced_by_task"] = True
            flags["exact_comment_region_forced_by_task"] = True
    if kind in {"reddit_submission", "gitlab_issue", "gitlab_mr"}:
        if _EXACT_DISCUSSION_REGION_RE.search(instruction) or _LATEST_DISCUSSION_REGION_RE.search(
            instruction
        ):
            flags["transition_forced_by_task"] = True
            flags["exact_comment_region_forced_by_task"] = True
    if kind in {"reddit_submission", "reddit_dashboard_list"}:
        if _REDDIT_COMMENT_VISUAL_REGION_RE.search(instruction):
            flags["transition_forced_by_task"] = True
            flags["exact_comment_region_forced_by_task"] = True
            flags["visual_comment_region_forced_by_task"] = True
            if not _task_already_seeds_reddit_comment_body(task):
                flags["seeded_comment_visible_forced_by_task"] = True
    return flags


def _task_already_seeds_reddit_comment_body(task: Mapping[str, Any]) -> bool:
    data_seed = task.get("data_seed")
    if not isinstance(data_seed, Mapping):
        return False
    raw_calls = data_seed.get("editor_calls")
    if raw_calls is None:
        raw_calls = data_seed.get("calls")
    if not isinstance(raw_calls, list):
        return False
    for call in raw_calls:
        if not isinstance(call, Mapping):
            continue
        if str(call.get("site") or "").strip().lower() != "reddit":
            continue
        if str(call.get("method") or "").strip() == "create_comment":
            return True
    return False


def _title_surface_forced_by_instruction(instruction: str) -> bool:
    """Return True when a title row is part of the benign task's goal."""
    if _TITLE_CONTENT_FORCING_RE.search(instruction):
        return True
    if _LISTING_PAGE_ONLY_RE.search(instruction):
        return False
    return _LISTING_ROW_ACTION_RE.search(instruction) is not None


def _benign_user_handle(task: Mapping[str, Any]) -> str | None:
    agent_ctx = task.get("agent_context") or {}
    auth = agent_ctx.get("authentication") or {}
    creds = auth.get("credentials") or {}
    handle = creds.get("username")
    if isinstance(handle, str) and handle.strip():
        return handle.strip()
    return None


def _reddit_forum_choice_is_ambiguous(task: Mapping[str, Any]) -> bool:
    instruction = str(task.get("instruction") or "").casefold()
    # "Most appropriate forum" tasks ask the agent to choose the forum at
    # runtime. A resolver-selected candidate forum is only a plausible answer,
    # not a forced encounter path for a pre-seeded forum listing row.
    return "most appropriate forum" in instruction


def _encounter_requirements(
    kind: ResourceKind, task: Mapping[str, Any], anchors: Mapping[str, Any]
) -> dict[str, Any]:
    """Populate encounter_requirements per handoff encounter-derivation notes."""
    requirements: dict[str, Any] = {"viewport_budget_chars": VIEWPORT_BUDGET_CHARS}
    if kind == "gitlab_dashboard_list":
        handle = _benign_user_handle(task)
        if handle:
            requirements["requires_at_mention"] = handle
        requirements["must_appear_on_list"] = True
    elif kind == "gitlab_search_result":
        query = anchors.get("query")
        scope = anchors.get("scope") or "issues"
        if query:
            requirements["requires_search_index"] = {"query": query, "scope": scope}
    elif kind == "reddit_forum":
        requirements["requires_post_sort_order"] = "recent"
        if _reddit_forum_choice_is_ambiguous(task):
            requirements["forum_choice_ambiguous"] = True
        else:
            requirements["must_appear_on_list"] = True
    elif kind == "reddit_dashboard_list":
        handle = _benign_user_handle(task)
        if handle:
            requirements["requires_at_mention"] = handle
    elif kind in ("gitlab_snippets_index", "gitlab_project_labels"):
        # Inline-listing surfaces: the seed's visible artifact must appear
        # on the listing page so the agent encounters it during the
        # benign read.
        requirements["must_appear_on_list"] = True
    return requirements
