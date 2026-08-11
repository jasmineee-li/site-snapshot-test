"""Phase 2 target resolution runner."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any, Literal

from worldsim.phase_2.target_resolution.constants import (
    DEFAULT_L3_CONCURRENCY,
    DEFAULT_L4_CONCURRENCY,
)
from worldsim.phase_2.target_resolution.encounter import (
    _assert_anchor_contract_conformance,
    _attach_surfaces_for,
    _benign_user_handle,
    _encounter_requirements,
    _reddit_forum_choice_is_ambiguous,
    _route_evidence_flags,
    _title_surface_forced_by_instruction,
)
from worldsim.phase_2.target_resolution.http_probes import (
    _admission_filter_resolved_record,
    _benign_probe_instance,
    _default_probe,
    _normalise_sort_direction,
    _postmill_submission_comment_count_from_html,
    _probe_http_json,
)
from worldsim.phase_2.target_resolution.l3 import resolve_l3
from worldsim.phase_2.target_resolution.l4 import resolve_l4
from worldsim.phase_2.target_resolution.listing_intent import (
    _gitlab_issue_listing_intent,
    _label_names_from_gitlab_issue_listing_instruction,
    _project_path_from_gitlab_listing_task,
)
from worldsim.phase_2.target_resolution.listing_probes import (
    _dashboard_query,
    _default_listing_probe,
    _filter_visible_gitlab_dashboard_items,
    _first_query_value,
    _gitlab_item_url,
    _gitlab_visible_dashboard_hrefs,
    _list_gitlab_dashboard,
    _list_gitlab_search,
    _list_reddit_forum,
    _normalize_href_path,
)
from worldsim.phase_2.target_resolution.reconstruction import (
    _anchors_from_gitlab_item,
    _anchors_from_reddit_submission,
    _clean_project_path,
    _project_item_to_record,
    _reconstruct_start_url_from_anchors,
)
from worldsim.phase_2.target_resolution.resolver import derive_benign_target_resource
from worldsim.phase_2.target_resolution.types import (
    ClassifierFn,
    ListingProbeFn,
    ProbeFn,
    RedditCommentCountFn,
)
from worldsim.phase_2.target_resolution.url_matching import (
    _canonicalize_project_path,
    _disambiguate_root_segment,
    _empty_record,
    _is_listing_kind,
    _iter_eval_urls,
    _iter_start_urls,
    _listing_start_url,
    _literalize_regex_value,
    _match_gitlab,
    _match_reddit,
    _normalise_url,
    _path_and_query,
    _site_kind_for_task,
    _strip_json_suffix,
    _strip_regex_anchors,
    _url_with_expected_query_params,
)

logger = logging.getLogger(__name__)
_SHARED_L3_SEM: asyncio.Semaphore | None = None
_SHARED_L3_SEM_KEY: tuple[int, asyncio.AbstractEventLoop | None] | None = None
_SHARED_L4_SEM: asyncio.Semaphore | None = None
_SHARED_L4_SEM_KEY: tuple[int, asyncio.AbstractEventLoop | None] | None = None

__all__ = [
    "ClassifierFn",
    "ListingProbeFn",
    "ProbeFn",
    "RedditCommentCountFn",
    "_admission_filter_resolved_record",
    "_anchors_from_gitlab_item",
    "_anchors_from_reddit_submission",
    "_assert_anchor_contract_conformance",
    "_attach_surfaces_for",
    "_benign_probe_instance",
    "_benign_user_handle",
    "_canonicalize_project_path",
    "_clean_project_path",
    "_dashboard_query",
    "_default_listing_probe",
    "_default_probe",
    "_disambiguate_root_segment",
    "_empty_record",
    "_encounter_requirements",
    "_filter_visible_gitlab_dashboard_items",
    "_first_query_value",
    "_gitlab_issue_listing_intent",
    "_gitlab_item_url",
    "_gitlab_visible_dashboard_hrefs",
    "_is_listing_kind",
    "_iter_eval_urls",
    "_iter_start_urls",
    "_label_names_from_gitlab_issue_listing_instruction",
    "_list_gitlab_dashboard",
    "_list_gitlab_search",
    "_list_reddit_forum",
    "_listing_start_url",
    "_literalize_regex_value",
    "_match_gitlab",
    "_match_reddit",
    "_normalise_sort_direction",
    "_normalise_url",
    "_normalize_href_path",
    "_path_and_query",
    "_postmill_submission_comment_count_from_html",
    "_probe_http_json",
    "_project_item_to_record",
    "_project_path_from_gitlab_listing_task",
    "_reconstruct_start_url_from_anchors",
    "_reddit_forum_choice_is_ambiguous",
    "_route_evidence_flags",
    "_site_kind_for_task",
    "_strip_json_suffix",
    "_strip_regex_anchors",
    "_title_surface_forced_by_instruction",
    "_url_with_expected_query_params",
    "derive_benign_target_resource",
    "resolve_l3",
    "resolve_l4",
    "resolve_tasks",
]


def _l3_concurrency_default() -> int:
    raw = os.environ.get("WORLDSIM_L3_CONCURRENCY", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_L3_CONCURRENCY


def _l4_concurrency_default() -> int:
    raw = os.environ.get("WORLDSIM_L4_CONCURRENCY", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_L4_CONCURRENCY


def _shared_l3_sem(limit: int) -> asyncio.Semaphore:
    global _SHARED_L3_SEM, _SHARED_L3_SEM_KEY
    loop = asyncio.get_event_loop()
    key = (limit, loop)
    if _SHARED_L3_SEM is None or _SHARED_L3_SEM_KEY != key:
        _SHARED_L3_SEM = asyncio.Semaphore(limit)
        _SHARED_L3_SEM_KEY = key
    return _SHARED_L3_SEM


def _shared_l4_sem(limit: int) -> asyncio.Semaphore:
    global _SHARED_L4_SEM, _SHARED_L4_SEM_KEY
    loop = asyncio.get_event_loop()
    key = (limit, loop)
    if _SHARED_L4_SEM is None or _SHARED_L4_SEM_KEY != key:
        _SHARED_L4_SEM = asyncio.Semaphore(limit)
        _SHARED_L4_SEM_KEY = key
    return _SHARED_L4_SEM


async def resolve_tasks(
    tasks: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    placeholders: Mapping[str, str],
    instance: Mapping[str, Any] | None,
    *,
    allow_layers: tuple[Literal["L1", "L2", "L3", "L4"], ...] = ("L1", "L2", "L3", "L4"),
    l3_concurrency: int | None = None,
    l4_concurrency: int | None = None,
    top_n: int | None = None,
    classifier: ClassifierFn | None = None,
    probe_fn: ProbeFn | None = None,
    listing_probe_fn: ListingProbeFn | None = None,
    reddit_comment_count_fn: RedditCommentCountFn | None = None,
    benchmark: str = "webarena_verified",
) -> dict[str, list[dict[str, Any]]]:
    """Resolve benign_target_resource records for a batch of benign tasks.

    The four layers run cheap-first: every task gets L1/L2 synchronously;
    tasks whose L1/L2 record is tagged ``pending_layer="L3"`` fall back
    to :func:`resolve_l3`; resolved records pass through Site Targeting's
    L4 capability, where declared listings fan out and other kinds pass
    through unchanged.

    Returns ``{task_id: [record, ...]}``. The list is ≥ 1 for every task
    except those whose L4 listing probe returned zero items — those
    tasks are omitted from the output dict so the caller's shard-builder
    sees "drop this task" rather than "attach to a stub".

    ``instance`` is required whenever ``allow_layers`` includes ``"L3"``
    or ``"L4"``; a ``ValueError`` fires at call time so misconfigured
    callers fail loudly instead of silently falling back to L1/L2.
    When ``allow_layers`` is ``("L1", "L2")`` this function is a
    sync-equivalent wrapper over :func:`derive_benign_target_resource`
    (kept async for uniform caller plumbing).

    Failure handling is graceful at the per-task level: L3 classifier /
    probe failures return the same stub record
    :func:`derive_benign_target_resource` emits for unresolved tasks so
    the downstream eligibility filter drops them; never raises into the
    caller. ``classifier`` / ``probe_fn`` / ``listing_probe_fn`` exist
    purely so tests can inject stubs without hitting the network.
    """
    needs_instance = ("L3" in allow_layers) or ("L4" in allow_layers)
    if needs_instance and instance is None:
        raise ValueError(
            "resolve_tasks: instance is required when allow_layers includes "
            "'L3' or 'L4'; pass allow_layers=('L1','L2') for the offline path"
        )

    l3_sem = _shared_l3_sem(l3_concurrency or _l3_concurrency_default())
    l4_sem = _shared_l4_sem(l4_concurrency or _l4_concurrency_default())

    l1_l2_layers: tuple[Literal["L1", "L2"], ...] = tuple(
        layer for layer in allow_layers if layer in ("L1", "L2")
    )  # type: ignore[assignment]
    if not l1_l2_layers:
        # At minimum we need L1 regex; without it the intent-only path
        # (L3) has nothing to fall back on.
        l1_l2_layers = ("L1", "L2")

    async def _resolve_one(task: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        task_id = str(task.get("id") or "")
        base = derive_benign_target_resource(
            task,
            placeholders,
            allow_layers=l1_l2_layers,
            benchmark=benchmark,
        )

        record: dict[str, Any] = dict(base)
        if "L3" in allow_layers and record.get("pending_layer") == "L3" and instance is not None:
            async with l3_sem:
                try:
                    record = await resolve_l3(
                        task,
                        placeholders,
                        instance,
                        classifier=classifier,
                        probe_fn=probe_fn,
                        benchmark=benchmark,
                    )
                except Exception as exc:
                    logger.warning("resolve_tasks: L3 raised for task=%r: %s", task_id, exc)
                    record = _empty_record(
                        f"L3 raised: {type(exc).__name__}: {exc}",
                        pending_layer="L3",
                    )

        if instance is not None:
            record = await _admission_filter_resolved_record(
                record,
                instance,
                reddit_comment_count_fn=reddit_comment_count_fn,
            )

        if (
            "L4" in allow_layers
            and record.get("kind")
            and not record.get("skip_l4_expansion")
            and instance is not None
        ):
            async with l4_sem:
                try:
                    expanded = await resolve_l4(
                        record,
                        task,
                        instance,
                        probe_fn=listing_probe_fn,
                        top_n=top_n,
                        placeholders=placeholders,
                        benchmark=benchmark,
                    )
                except Exception as exc:
                    logger.warning("resolve_tasks: L4 raised for task=%r: %s", task_id, exc)
                    expanded = []
            return task_id, expanded

        return task_id, [record]

    results = await asyncio.gather(*(_resolve_one(t) for t in tasks))
    # Preserve input order in the output dict (Python dicts preserve
    # insertion order); omit tasks whose resolver produced no records.
    out: dict[str, list[dict[str, Any]]] = {}
    for task_id, records in results:
        if not task_id:
            continue
        if records:
            out[task_id] = records
    return out


__all__ = [name for name in globals() if not name.startswith("__")]
