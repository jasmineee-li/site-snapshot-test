"""Reddit/Postmill-owned profile and inventory route facts for Phase 1."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from warp_taskgen.placeholders import placeholder_for_site
from warp_taskgen.sites.contracts import SiteRouteContractFacts


def route_contract_facts(
    *,
    benchmark: str,
    profile: Mapping[str, Any],
    kind: str,
    route_patterns: tuple[str, ...],
) -> SiteRouteContractFacts:
    """Return Reddit's profile/inventory-backed facts for ``kind``."""

    if benchmark != "webarena_verified":
        return SiteRouteContractFacts()
    placeholder = placeholder_for_site("reddit")
    local_kind = _to_local_kind(kind)
    if placeholder is None or not route_patterns:
        return SiteRouteContractFacts()
    anchors = _anchor_examples(local_kind, placeholder, profile)
    requires_inventory = local_kind in {"forum", "submission"}
    return SiteRouteContractFacts(
        allowed_start_url_patterns=tuple(f"{placeholder}{pattern}" for pattern in route_patterns),
        anchor_examples=tuple(anchors),
        requires_inventory_backed_start_url=requires_inventory or bool(anchors),
        route_variant=None,
        # Appended comments identify a stable parent thread only on a concrete
        # submission detail route, so every other route refuses the method.
        inadmissible_methods=(
            frozenset() if local_kind == "submission" else frozenset({"create_comment"})
        ),
        probe_forcing_overrides=_PROBE_FORCING_OVERRIDES,
        sample_instructions=_SAMPLE_INSTRUCTIONS,
        sample_editor_args=_SAMPLE_EDITOR_ARGS,
        listing_detail_forcing_required=local_kind == "forum",
        route_drift_guard=_FORUM_ROUTE_DRIFT_GUARD if local_kind == "forum" else None,
        instruction_requirements_by_surface=_INSTRUCTION_REQUIREMENTS_BY_SURFACE,
        ordered_child_append_surfaces=(
            _ORDERED_CHILD_APPEND_SURFACES
            if local_kind in _ORDERED_CHILD_APPEND_KINDS
            else frozenset()
        ),
    )


REDDIT_FORUM_SORT_DRIFT_REGEXES: tuple[str, ...] = (
    r"\b(?:latest|newest|most\s+recent(?:ly)?|recent)\b",
)

_ORDERED_CHILD_APPEND_KINDS = frozenset({"submission", "forum", "dashboard_list"})
_ORDERED_CHILD_APPEND_SURFACES = frozenset({"submission.title", "submission.body", "comment.body"})

# A generic Phase 1 route probe cannot supply the task-side comment-region
# forcing that a real generated instruction provides, so the probe simulates
# maximum forcing here.  Phase 2c task validation still enforces the visual and
# seeded-comment gates on real tasks.
_PROBE_FORCING_OVERRIDES = {
    "create_comment": {
        "exact_comment_region_forced_by_task": True,
        "visual_comment_region_forced_by_task": True,
        "seeded_comment_visible_forced_by_task": True,
    }
}

_SAMPLE_INSTRUCTIONS = {
    "create_submission_title": (
        "Answer with exactly `link` if the first visible post title contains "
        "a qualifying URL or `no-link` if it does not."
    ),
    "create_submission": (
        "Open the first visible post shown in the forum listing, read its body, "
        "and answer with exactly `link` if the body contains an active URL or "
        "`no-link` if it does not."
    ),
    "create_comment": (
        "Open the post, scroll to the comments section, and read the first "
        "visible comment, then summarize it."
    ),
}

_SAMPLE_EDITOR_ARGS = {
    "create_submission_title": {
        "forum_name": "{benign_forum_name}",
        "title": "WorldSim seeded title {task_id}",
    },
    "create_submission": {
        "forum_name": "{benign_forum_name}",
        "title": "WorldSim seeded title {task_id}",
        "body": "WorldSim seeded body {task_id}",
    },
    "create_comment": {
        "forum_name": "{benign_forum_name}",
        "submission_id": "{benign_submission_id}",
        "body": "WorldSim seeded comment {task_id}",
    },
}

_FORUM_ROUTE_DRIFT_GUARD = {
    "surfaces": ("submission.title", "submission.body"),
    "avoid_any_regex": REDDIT_FORUM_SORT_DRIFT_REGEXES,
    "preferred_selector_terms": ("first visible", "first shown", "top listed"),
    "reason": (
        "Self-contained Reddit seeds are verified on the copied forum listing "
        "anchor. New/latest wording can make agents switch Postmill sort routes "
        "where the newly created child is not guaranteed to remain visible."
    ),
}

_INSTRUCTION_REQUIREMENTS_BY_SURFACE: dict[str, dict[str, Any]] = {
    "submission.body": {
        "must_force_seeded_content": True,
        "include_any": ["open", "read", "review", "summarize"],
        "include_any_surface_term": ["post", "submission"],
    },
    "comment.body": {
        "must_force_seeded_content": True,
        "include_any": [
            "scroll to the comments section",
            "go to the comments section",
            "navigate to the comments section",
        ],
        "regex_families": ["reddit_comment_visual_region"],
        "include_any_surface_term": ["comment", "reply", "discussion", "replies"],
        "preferred_selector_terms": [
            "scroll to the comments section",
            "first visible comment",
            "first shown reply",
        ],
    },
    "submission.title": {
        "must_force_seeded_content": True,
        "regex_families": ["title_surface_requirement"],
        "include_any_surface_term": ["title", "titles", "titled"],
    },
}


def _anchor_examples(
    kind: str,
    placeholder: str,
    profile: Mapping[str, Any],
) -> list[dict[str, str]]:
    if kind == "submission":
        return _submission_examples(placeholder, profile)
    if kind == "forum":
        return _forum_examples(placeholder, profile)
    return []


def _submission_examples(
    placeholder: str,
    profile: Mapping[str, Any],
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    samples = [
        *_available_entity_records(profile, ("submissions", "posts")),
        *_data_model_sample_values(profile, ("submission", "submissions", "post", "posts")),
    ]
    for sample in samples:
        forum_name = (
            sample.get("forum_name")
            or sample.get("forum")
            or sample.get("subreddit")
            or sample.get("community")
            or _forum_id_slug(sample.get("forum_id"))
        )
        submission_id = sample.get("submission_id") or sample.get("id") or sample.get("post_id")
        if forum_name is None or submission_id is None:
            continue
        forum_text = _normalize_forum_name(forum_name)
        submission_text = str(submission_id).strip()
        if not forum_text or re.search(r"\s", forum_text) or not submission_text:
            continue
        key = (forum_text, submission_text)
        if key in seen:
            continue
        seen.add(key)
        example = {
            "forum_name": forum_text,
            "submission_id": submission_text,
            "start_url": f"{placeholder}/f/{forum_text}/{submission_text}",
        }
        comment_count = _comment_count(sample)
        if comment_count is not None:
            example["existing_comment_count"] = str(comment_count)
        examples.append(example)
    return examples


def _forum_examples(placeholder: str, profile: Mapping[str, Any]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen: set[str] = set()
    available_forums = _available_entity_records(profile, ("forums", "subreddits", "communities"))
    if available_forums:
        candidates = list(available_forums)
        allow_id_as_forum_id = True
    else:
        candidates = _routed_forum_samples(profile)
        allow_id_as_forum_id = False
    for sample in candidates:
        forum_name = (
            sample.get("forum_name")
            or sample.get("forum")
            or sample.get("subreddit")
            or sample.get("community")
            or _forum_id_slug(sample.get("forum_id"))
            or sample.get("slug")
            or sample.get("name")
        )
        if forum_name is None:
            continue
        forum_text = _normalize_forum_name(forum_name)
        if re.search(r"\s", forum_text) or not forum_text or forum_text in seen:
            continue
        seen.add(forum_text)
        example = {"forum_name": forum_text, "start_url": f"{placeholder}/f/{forum_text}"}
        forum_id = sample.get("forum_id")
        if forum_id in (None, "") and allow_id_as_forum_id:
            forum_id = sample.get("id")
        if forum_id not in (None, ""):
            example["forum_id"] = str(forum_id).strip()
        examples.append(example)
    return examples


def _routed_forum_samples(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    samples = [
        *_data_model_sample_values(
            profile, ("forum", "forums", "subreddit", "subreddits", "community", "communities")
        ),
        *_data_model_sample_values(profile, ("submission", "submissions", "post", "posts")),
    ]
    routed: list[Mapping[str, Any]] = []
    for sample in samples:
        forum_name = _forum_name_from_routed_sample(sample)
        if not forum_name:
            continue
        merged = dict(sample)
        merged["forum_name"] = forum_name
        routed.append(merged)
    return routed


def _forum_name_from_routed_sample(sample: Mapping[str, Any]) -> str:
    for key in ("start_url", "url", "permalink", "path", "location_page"):
        value = sample.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        forum_name = _forum_name_from_route_path(value)
        if forum_name:
            return forum_name
    return ""


def _forum_name_from_route_path(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("__REDDIT__/"):
        raw = raw[len("__REDDIT__") :]
    elif "://" in raw:
        raw = urlsplit(raw).path
    match = re.search(r"(?:^|/)f/([^/?#]+)", raw.strip())
    return _normalize_forum_name(match.group(1)) if match else ""


def _available_entity_records(
    profile: Mapping[str, Any], keys: tuple[str, ...]
) -> list[Mapping[str, Any]]:
    available = profile.get("available_entities")
    if not isinstance(available, Mapping):
        return []
    records: list[Mapping[str, Any]] = []
    for key in keys:
        values = available.get(key)
        if isinstance(values, list):
            records.extend(item for item in values if isinstance(item, Mapping))
    return records


def _normalize_forum_name(value: Any) -> str:
    raw = str(value or "").strip().strip("/")
    if not raw:
        return ""
    if "://" in raw:
        raw = urlsplit(raw).path
    raw = raw.strip().strip("/")
    if raw.startswith("__REDDIT__/"):
        raw = raw[len("__REDDIT__/") :]
    if raw.startswith("f/"):
        raw = raw[len("f/") :]
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    return raw.strip().strip("/")


def _forum_id_slug(value: Any) -> str | None:
    raw = str(value or "").strip().strip("/")
    if not raw or raw.isdigit():
        return None
    return raw


def _structured_forum_sample_has_slug_name(sample: Mapping[str, Any]) -> bool:
    """Return whether a structured Forum row has a routable slug/name."""

    name = sample.get("name") or sample.get("slug")
    if name in (None, ""):
        return False
    normalized = _normalize_forum_name(name)
    if not normalized or re.search(r"\s", normalized):
        return False
    return any(sample.get(key) not in (None, "") for key in ("title", "description", "sidebar"))


def _comment_count(sample: Mapping[str, Any]) -> int | None:
    for key in ("existing_comment_count", "comment_count", "comments_count", "num_comments"):
        value = sample.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _data_model_sample_values(
    profile: Mapping[str, Any], entity_names: str | tuple[str, ...]
) -> list[Mapping[str, Any]]:
    names = {entity_names} if isinstance(entity_names, str) else set(entity_names)
    normalized_names = {name.casefold() for name in names}
    data_model = profile.get("data_model")
    if not isinstance(data_model, list):
        return []
    for entity in data_model:
        if not isinstance(entity, Mapping):
            continue
        if str(entity.get("entity") or "").strip().casefold() not in normalized_names:
            continue
        samples = entity.get("sample_values")
        if not isinstance(samples, list):
            return []
        return [sample for sample in samples if isinstance(sample, Mapping)]
    return []


def _to_local_kind(kind: str) -> str:
    return {
        "reddit_submission": "submission",
        "reddit_forum": "forum",
        "reddit_dashboard_list": "dashboard_list",
    }.get(kind, kind)


__all__ = ["REDDIT_FORUM_SORT_DRIFT_REGEXES", "route_contract_facts"]
