"""Phase 2 target resolution types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

ResourceKind = Literal[
    "gitlab_issue",
    "gitlab_mr",
    "gitlab_search_result",
    "gitlab_dashboard_list",
    "gitlab_user_profile",
    "gitlab_snippet",
    "gitlab_snippets_index",
    "gitlab_project_milestone",
    "gitlab_project_labels",
    "gitlab_group",
    "reddit_submission",
    "reddit_forum",
    "reddit_dashboard_list",
]

ProbeFn = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, str]],
    Awaitable[dict[str, Any] | None],
]
RedditCommentCountFn = Callable[
    [Mapping[str, Any], str, str],
    Awaitable[int | None],
]
ClassifierFn = Callable[
    [Mapping[str, Any], Mapping[str, str]],
    Awaitable[dict[str, Any] | None],
]
ListingProbeFn = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Awaitable[list[dict[str, Any]]],
]


class ResolverContractDriftError(RuntimeError):
    """Raised when :func:`derive_benign_target_resource` produces a
    record whose ``kind`` isn't addressable by any registered editor
    method — a sign the resolver and the editor-method contract registry
    have drifted apart. Fix: either add the missing kind to an existing
    ``@editor_method`` decorator, or stop emitting the kind here.
    """
