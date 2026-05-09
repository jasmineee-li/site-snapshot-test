"""Resolver tests for the six gitlab new_task kinds.

These extend the inventory beyond the existing 812 benchmark tasks. The
URL shapes are derived from ``logs/phase_1/novel_tasks_gitlab.json``;
the disambiguator tests cover the user-vs-group-vs-neither matrix.
"""

from __future__ import annotations

from typing import Any

from worldsim.phase_2.target_resolution.resolver import derive_benign_target_resource
from worldsim.phase_2.target_resolution.url_matching import (
    _disambiguate_root_segment,
)
from worldsim.placeholders import placeholders_for_site_urls

PLACEHOLDERS = placeholders_for_site_urls(
    [
        ("gitlab", "https://gitlab.local"),
        ("reddit", "https://reddit.local"),
    ]
)


def _gitlab_task(
    *,
    task_id: str = "t",
    eval_url: Any | None = None,
    start_urls: list[str] | None = None,
    user_handles: list[str] | None = None,
    group_handles: list[str] | None = None,
    evaluator: str = "NetworkEventEvaluator",
) -> dict[str, Any]:
    agent_context: dict[str, Any] = {
        "authentication": {"credentials": {"username": "byteblaze"}},
    }
    if user_handles is not None or group_handles is not None:
        agent_context["gitlab"] = {
            "user_handles": list(user_handles or []),
            "group_handles": list(group_handles or []),
        }
    task: dict[str, Any] = {
        "id": task_id,
        "sites": ["gitlab"],
        "start_urls": start_urls if start_urls is not None else ["__GITLAB__"],
        "agent_context": agent_context,
        "reward_function": {"eval": []},
    }
    if eval_url is not None:
        task["reward_function"]["eval"] = [{"evaluator": evaluator, "expected": {"url": eval_url}}]
    return task


# ---------------------------------------------------------------------------
# Disambiguator unit tests
# ---------------------------------------------------------------------------


def test_disambiguate_segment_in_user_handles_only():
    task = _gitlab_task(user_handles=["byteblaze", "root"], group_handles=["a11yproject"])
    assert _disambiguate_root_segment(task, "byteblaze") == "user"


def test_disambiguate_segment_in_group_handles_only():
    task = _gitlab_task(user_handles=["byteblaze"], group_handles=["a11yproject"])
    assert _disambiguate_root_segment(task, "a11yproject") == "group"


def test_disambiguate_segment_in_both_lists_returns_none():
    task = _gitlab_task(user_handles=["overlap"], group_handles=["overlap"])
    assert _disambiguate_root_segment(task, "overlap") is None


def test_disambiguate_segment_in_neither_returns_none():
    task = _gitlab_task(user_handles=["byteblaze"], group_handles=["a11yproject"])
    assert _disambiguate_root_segment(task, "unknown") is None


def test_disambiguate_missing_gitlab_block_returns_none():
    task = _gitlab_task()
    assert _disambiguate_root_segment(task, "byteblaze") is None


def test_disambiguate_empty_segment_returns_none():
    task = _gitlab_task(user_handles=["byteblaze"])
    assert _disambiguate_root_segment(task, "") is None


# ---------------------------------------------------------------------------
# gitlab_user_profile
# ---------------------------------------------------------------------------


def test_gitlab_user_profile_l2_from_root_segment_with_handles():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/byteblaze"],
        user_handles=["root", "byteblaze"],
        group_handles=["a11yproject"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L2"
    assert result["kind"] == "gitlab_user_profile"
    assert result["anchors"] == {"username": "byteblaze"}
    assert result["start_url_resolved"] == "https://gitlab.local/byteblaze"


def test_gitlab_user_profile_l1_from_eval_url():
    task = _gitlab_task(
        eval_url="__GITLAB__/root",
        user_handles=["root", "byteblaze"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_user_profile"
    assert result["anchors"]["username"] == "root"


def test_root_segment_unresolved_falls_through_when_handles_missing():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/byteblaze"],
        # No user_handles / group_handles supplied.
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] is None


# ---------------------------------------------------------------------------
# gitlab_group
# ---------------------------------------------------------------------------


def test_gitlab_group_l2_from_root_segment():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/a11yproject"],
        user_handles=["byteblaze"],
        group_handles=["a11yproject", "design"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_group"
    assert result["anchors"] == {"group_path": "a11yproject"}
    assert result["start_url_resolved"] == "https://gitlab.local/a11yproject"


# ---------------------------------------------------------------------------
# gitlab_snippet / gitlab_snippets_index
# ---------------------------------------------------------------------------


def test_gitlab_snippet_l2_basic():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/-/snippets/42"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_snippet"
    assert result["anchors"] == {"snippet_id": "42"}
    assert result["start_url_resolved"] == "https://gitlab.local/-/snippets/42"


def test_gitlab_snippets_index_l2():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/-/snippets"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_snippets_index"
    assert result["start_url_resolved"] == "https://gitlab.local/-/snippets"
    # Listings carry the must_appear_on_list encounter requirement.
    assert result["encounter_requirements"].get("must_appear_on_list") is True


def test_gitlab_snippet_l1_from_eval_url():
    task = _gitlab_task(eval_url="__GITLAB__/-/snippets/7")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_snippet"
    assert result["anchors"]["snippet_id"] == "7"


# ---------------------------------------------------------------------------
# gitlab_project_milestone
# ---------------------------------------------------------------------------


def test_gitlab_project_milestone_l1():
    task = _gitlab_task(
        eval_url="__GITLAB__/byteblaze/dotfiles/-/milestones/1",
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_project_milestone"
    assert result["anchors"]["project_path"] == "byteblaze/dotfiles"
    assert result["anchors"]["milestone_iid"] == "1"
    assert result["start_url_resolved"] == "https://gitlab.local/byteblaze/dotfiles/-/milestones/1"


def test_gitlab_project_milestone_l2_fallback():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/byteblaze/empathy-prompts/-/milestones/3"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_project_milestone"
    assert result["anchors"]["milestone_iid"] == "3"


# ---------------------------------------------------------------------------
# gitlab_project_labels
# ---------------------------------------------------------------------------


def test_gitlab_project_labels_l2():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/primer/design/-/labels"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_project_labels"
    assert result["anchors"] == {"project_path": "primer/design"}
    assert result["start_url_resolved"] == "https://gitlab.local/primer/design/-/labels"
    assert result["encounter_requirements"].get("must_appear_on_list") is True


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


def test_existing_kinds_unaffected_by_new_root_segment_pattern():
    """Existing /<project>/-/issues/<iid> tasks must still resolve to gitlab_issue."""
    task = _gitlab_task(eval_url="__GITLAB__/byteblaze/dotfiles/-/issues/7")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_issue"


def test_dashboard_unchanged_after_root_segment_addition():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/dashboard/todos"],
        user_handles=["byteblaze"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_dashboard_list"
