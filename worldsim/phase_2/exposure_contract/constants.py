"""Exposure contract policy constants."""

from __future__ import annotations

PREFERRED_PAYLOAD_ARGS: tuple[str, ...] = ("body", "description", "message", "text", "content")
PREFERRED_TOKEN_ORDER: tuple[str, ...] = (
    "{benign_project_id}",
    "{benign_project_path}",
    "{benign_label_names}",
    "{benign_issue_iid}",
    "{benign_mr_iid}",
    "{benign_milestone_iid}",
    "{benign_submission_id}",
    "{benign_forum_name}",
    "{benign_snippet_id}",
    "{benign_group_path}",
    "{benign_user_handle}",
)

DIRECT_KINDS = frozenset(
    {
        "gitlab_issue",
        "gitlab_mr",
        "reddit_submission",
        "gitlab_user_profile",
        "gitlab_snippet",
        "gitlab_project_milestone",
        "gitlab_group",
    }
)
LISTING_SOURCE_KINDS = frozenset(
    {
        "gitlab_search_result",
        "gitlab_dashboard_list",
        "reddit_forum",
        "reddit_dashboard_list",
        "gitlab_snippets_index",
        "gitlab_project_labels",
    }
)
TRANSITIVE_EXISTING_SOURCE_KINDS = frozenset(
    {"gitlab_search_result", "gitlab_dashboard_list", "reddit_forum", "reddit_dashboard_list"}
)
CREATE_CHILD_LISTING_KINDS = frozenset({"reddit_forum"})
ORDERED_CREATED_CHILD_PRE_CALL_DELAY_S = 1.25

__all__ = [
    "CREATE_CHILD_LISTING_KINDS",
    "DIRECT_KINDS",
    "LISTING_SOURCE_KINDS",
    "ORDERED_CREATED_CHILD_PRE_CALL_DELAY_S",
    "PREFERRED_PAYLOAD_ARGS",
    "PREFERRED_TOKEN_ORDER",
    "TRANSITIVE_EXISTING_SOURCE_KINDS",
]
