"""GitLab's deterministic Site Targeting grammar."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from urllib.parse import quote as urlquote
from urllib.parse import urlsplit

from worldsim.sites.contracts import CanonicalRoute, TargetingContext
from worldsim.sites.task_evidence import _path_and_query

GitLabResourceKind = Literal[
    "issue",
    "merge_request",
    "search_result",
    "dashboard_list",
    "user_profile",
    "snippet",
    "snippets_index",
    "project_milestone",
    "project_labels",
    "group",
]

_LOCAL_TO_LEGACY: dict[str, str] = {
    "issue": "gitlab_issue",
    "merge_request": "gitlab_mr",
    "search_result": "gitlab_search_result",
    "dashboard_list": "gitlab_dashboard_list",
    "user_profile": "gitlab_user_profile",
    "snippet": "gitlab_snippet",
    "snippets_index": "gitlab_snippets_index",
    "project_milestone": "gitlab_project_milestone",
    "project_labels": "gitlab_project_labels",
    "group": "gitlab_group",
}
_LEGACY_TO_LOCAL = {legacy: local for local, legacy in _LOCAL_TO_LEGACY.items()}

_ISSUE_RE = re.compile(r"/(?P<project_path>(?:[^/?#]+/)+[^/?#]+)/-/issues/(?P<issue_iid>\d+)")
_ISSUE_LISTING_RE = re.compile(r"/(?P<project_path>(?:[^/?#]+/)+[^/?#]+)/-/issues(?:/?(?:\?|$))")
_MR_RE = re.compile(r"/(?P<project_path>(?:[^/?#]+/)+[^/?#]+)/-/merge_requests/(?P<mr_iid>\d+)")
_MILESTONE_RE = re.compile(
    r"/(?P<project_path>(?:[^/?#]+/)+[^/?#]+)/-/milestones/(?P<milestone_iid>\d+)"
)
_LABELS_RE = re.compile(r"/(?P<project_path>(?:[^/?#]+/)+[^/?#]+)/-/labels(?:/?(?:\?|$))")
_SNIPPET_RE = re.compile(r"/-/snippets/(?P<snippet_id>\d+)")
_SNIPPETS_INDEX_RE = re.compile(r"/-/snippets(?:/?(?:\?|$))")
_SEARCH_RE = re.compile(
    r"/search\?(?=[^#]*\bsearch=(?P<q>[^&]+))"
    r"(?=[^#]*\bscope=(?P<scope>issues|merge_requests))"
)
_DASHBOARD_RE = re.compile(r"/dashboard/(?P<dash>todos|merge_requests|issues)\b")
_PROJECT_ISSUES_API_RE = re.compile(r"/api/v4/projects/(?P<project_id>\d+)/issues\b")
_ROOT_SEGMENT_RE = re.compile(r"^/(?P<segment>[A-Za-z][A-Za-z0-9_.\-]*)(?:/?(?:\?|$))")
_GITLAB_PROJECT_ROOT_RE = re.compile(
    r"^/(?P<project_path>[A-Za-z0-9_.-]+/[A-Za-z0-9_.\-/.-]+?)(?:/?(?:\?|$))"
)
_GITLAB_PROJECT_PATH_IN_TEXT_RE = re.compile(
    r"\b(?P<project_path>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)\b"
)
_HOSTPREFIX_RE = re.compile(
    r"^(?:https?://[^/]+/|gitlab\.local/|localhost(?::\d+)?/)",
    re.IGNORECASE,
)

_LISTING_KINDS = frozenset({"search_result", "dashboard_list", "snippets_index", "project_labels"})

_L3_PROBE_KINDS_FOR_API: dict[str, frozenset[str]] = {
    "list_user_todos": frozenset({"gitlab_dashboard_list"}),
    "list_user_merge_requests": frozenset({"gitlab_dashboard_list", "gitlab_mr"}),
    "list_user_issues": frozenset({"gitlab_dashboard_list", "gitlab_issue"}),
    "search_user_issues": frozenset({"gitlab_issue", "gitlab_search_result"}),
    "search_user_mrs": frozenset({"gitlab_mr", "gitlab_search_result"}),
    "search_project_issues": frozenset({"gitlab_issue", "gitlab_search_result"}),
    "search_project_mrs": frozenset({"gitlab_mr", "gitlab_search_result"}),
    "list_project_issues_recent": frozenset({"gitlab_issue", "gitlab_search_result"}),
    "list_project_mrs_recent": frozenset({"gitlab_mr", "gitlab_search_result"}),
    "find_project_by_path": frozenset(
        {
            "gitlab_issue",
            "gitlab_mr",
            "gitlab_search_result",
            "gitlab_user_profile",
            "gitlab_snippet",
            "gitlab_project_milestone",
            "gitlab_group",
            "gitlab_snippets_index",
            "gitlab_project_labels",
        }
    ),
    "none": frozenset(),
}
_L3_LISTING_SOURCE_FOR_API: dict[str, dict[str, str]] = {
    "list_user_issues": {"gitlab_issue": "gitlab_dashboard_list"},
    "list_user_merge_requests": {
        "gitlab_issue": "gitlab_dashboard_list",
        "gitlab_mr": "gitlab_dashboard_list",
    },
    "search_user_issues": {"gitlab_issue": "gitlab_search_result"},
    "search_user_mrs": {"gitlab_mr": "gitlab_search_result"},
    "search_project_issues": {"gitlab_issue": "gitlab_search_result"},
    "search_project_mrs": {"gitlab_mr": "gitlab_search_result"},
    "list_project_issues_recent": {"gitlab_issue": "gitlab_search_result"},
    "list_project_mrs_recent": {"gitlab_mr": "gitlab_search_result"},
}


def to_local_kind(kind: str) -> str:
    return _LEGACY_TO_LOCAL.get(kind, kind)


def to_legacy_kind(kind: str) -> str:
    return _LOCAL_TO_LEGACY.get(kind, kind)


def _anchors(match: re.Match[str]) -> dict[str, str]:
    return {key: value for key, value in match.groupdict().items() if value}


def _clean_project_path(project_path: str) -> str:
    if not project_path:
        return ""
    path = str(project_path).strip()
    while True:
        stripped = _HOSTPREFIX_RE.sub("", path, count=1)
        if stripped == path:
            break
        path = stripped
    return path.strip("/")


def _disambiguate_root_segment(task: Mapping[str, Any], segment: str) -> str | None:
    if not isinstance(segment, str) or not segment:
        return None
    agent_context = task.get("agent_context") or {}
    if not isinstance(agent_context, Mapping):
        return None
    gitlab_context = agent_context.get("gitlab")
    if not isinstance(gitlab_context, Mapping):
        return None
    users = {str(value).strip() for value in gitlab_context.get("user_handles") or []}
    groups = {str(value).strip() for value in gitlab_context.get("group_handles") or []}
    in_users = segment in users
    in_groups = segment in groups
    if in_users and not in_groups:
        return "user"
    if in_groups and not in_users:
        return "group"
    return None


def _route(
    kind: str,
    patterns: tuple[str, ...],
    *examples: Mapping[str, Any],
    variant: str | None = None,
) -> CanonicalRoute:
    return CanonicalRoute(
        id=f"gitlab.{kind}",
        site="gitlab",
        kind=kind,
        allowed_start_url_patterns=patterns,
        compatibility_kind=to_legacy_kind(kind),
        anchor_examples=tuple(examples),
        route_variant=variant,
    )


class GitLabSite:
    site = "gitlab"
    supported_benchmarks = frozenset({"webarena_verified"})

    def validate(self) -> None:
        # Route construction is deterministic and validated by BoundSite.
        return None

    def validate_task(self, task: Mapping[str, Any]) -> tuple[str, str] | None:
        agent_context = task.get("agent_context") or {}
        gitlab_context = agent_context.get("gitlab")
        if gitlab_context is None:
            return None
        for field_name in ("user_handles", "group_handles"):
            handles = gitlab_context.get(field_name)
            if handles is None:
                continue
            if (
                not isinstance(handles, Sequence)
                or isinstance(handles, (str, bytes, bytearray))
                or not all(isinstance(handle, str) for handle in handles)
            ):
                return (
                    "malformed_metadata",
                    f"task.agent_context.gitlab.{field_name} must be a sequence of strings",
                )
        return None

    def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
        del context
        return (
            _route(
                "issue",
                ("/{project_path}/-/issues/{issue_iid}",),
                {"project_path": "namespace/project", "issue_iid": "1"},
            ),
            _route(
                "merge_request",
                ("/{project_path}/-/merge_requests/{mr_iid}",),
                {"project_path": "namespace/project", "mr_iid": "1"},
            ),
            _route(
                "search_result",
                ("/{project_path}/-/issues",),
                {"project_path": "namespace/project"},
            ),
            _route(
                "dashboard_list",
                ("/dashboard/todos", "/dashboard/merge_requests", "/dashboard/issues"),
                {"dashboard": "todos"},
                {"dashboard": "merge_requests"},
                {"dashboard": "issues"},
            ),
            _route("user_profile", ("/{username}",), {"username": "user"}),
            _route("group", ("/{group_path}",), {"group_path": "group"}),
            _route("snippet", ("/-/snippets/{snippet_id}",), {"snippet_id": "1"}),
            _route("snippets_index", ("/-/snippets",), {}),
            _route(
                "project_milestone",
                ("/{project_path}/-/milestones/{milestone_iid}",),
                {"project_path": "namespace/project", "milestone_iid": "1"},
            ),
            _route(
                "project_labels",
                ("/{project_path}/-/labels",),
                {"project_path": "namespace/project"},
            ),
        )

    def match(
        self, url: str, task: Mapping[str, Any], context: TargetingContext
    ) -> tuple[str, dict[str, Any]] | None:
        del context
        path_and_query = _path_and_query(url)
        patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
            ("issue", _ISSUE_RE),
            ("search_result", _ISSUE_LISTING_RE),
            ("merge_request", _MR_RE),
            ("project_milestone", _MILESTONE_RE),
            ("project_labels", _LABELS_RE),
            ("snippet", _SNIPPET_RE),
            ("snippets_index", _SNIPPETS_INDEX_RE),
            ("search_result", _SEARCH_RE),
            ("dashboard_list", _DASHBOARD_RE),
            ("search_result", _PROJECT_ISSUES_API_RE),
            ("user_profile", _ROOT_SEGMENT_RE),
        )
        for kind, pattern in patterns:
            match = pattern.search(path_and_query)
            if not match:
                continue
            anchors = _anchors(match)
            if kind == "dashboard_list":
                anchors["dashboard"] = anchors.pop("dash", "")
            if kind == "search_result" and "q" in anchors:
                anchors["query"] = anchors.pop("q")
            if "segment" in anchors:
                segment = anchors.pop("segment")
                resolved = _disambiguate_root_segment(task, segment)
                if resolved == "user":
                    return "user_profile", {"username": segment}
                if resolved == "group":
                    return "group", {"group_path": segment}
                continue
            return kind, anchors
        return None

    def reconstruct(
        self,
        kind: str,
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> str | None:
        origin = context.site_origin()
        if origin is None:
            return None
        base = origin.rstrip("/")
        local_kind = to_local_kind(kind)
        if local_kind == "issue":
            project_path, iid = anchors.get("project_path"), anchors.get("issue_iid")
            return (
                f"{base}/{_clean_project_path(str(project_path))}/-/issues/{iid}"
                if project_path and iid
                else None
            )
        if local_kind == "merge_request":
            project_path, iid = anchors.get("project_path"), anchors.get("mr_iid")
            return (
                f"{base}/{_clean_project_path(str(project_path))}/-/merge_requests/{iid}"
                if project_path and iid
                else None
            )
        if local_kind == "search_result":
            project_path = anchors.get("project_path")
            query = anchors.get("query")
            scope = anchors.get("scope") or "issues"
            if project_path:
                return f"{base}/{_clean_project_path(str(project_path))}/-/{scope}"
            if query:
                return f"{base}/search?search={urlquote(str(query), safe='+')}&scope={scope}"
            return None
        if local_kind == "dashboard_list":
            dashboard = anchors.get("dashboard")
            return f"{base}/dashboard/{dashboard}" if dashboard else None
        if local_kind == "user_profile":
            username = anchors.get("username")
            return f"{base}/{username}" if username else None
        if local_kind == "group":
            group_path = anchors.get("group_path")
            return f"{base}/{group_path}" if group_path else None
        if local_kind == "snippet":
            snippet_id = anchors.get("snippet_id")
            return f"{base}/-/snippets/{snippet_id}" if snippet_id else None
        if local_kind == "snippets_index":
            return f"{base}/-/snippets"
        if local_kind == "project_milestone":
            project_path, iid = anchors.get("project_path"), anchors.get("milestone_iid")
            return (
                f"{base}/{_clean_project_path(str(project_path))}/-/milestones/{iid}"
                if project_path and iid
                else None
            )
        if local_kind == "project_labels":
            project_path = anchors.get("project_path")
            return (
                f"{base}/{_clean_project_path(str(project_path))}/-/labels"
                if project_path
                else None
            )
        return None

    def is_listing(self, kind: str) -> bool:
        return to_local_kind(kind) in _LISTING_KINDS

    def listing_start_url(
        self, kind: str, resolved_url: str, fallback_url: str | None
    ) -> str | None:
        if not self.is_listing(kind):
            return fallback_url
        path = urlsplit(resolved_url).path or ""
        return fallback_url if path.startswith("/api/") else resolved_url

    def validate_candidate(
        self,
        kind: str,
        probe_query: Mapping[str, Any],
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> tuple[str, str] | None:
        del anchors, context
        api = str(probe_query.get("api") or "").strip()
        if not api:
            return None
        allowed = _L3_PROBE_KINDS_FOR_API.get(api)
        if allowed is None:
            return None
        compatibility_kind = to_legacy_kind(to_local_kind(kind))
        if not allowed:
            return (
                "probe_kind_mismatch",
                f"api={api!r} is not a real probe; pair it with kind=null or "
                "kind=out_of_scope_for_option_a",
            )
        if compatibility_kind not in allowed:
            return (
                "probe_kind_mismatch",
                f"api={api!r} cannot fill anchors for kind={compatibility_kind!r}; "
                f"allowed kinds: {sorted(allowed)}",
            )
        return None

    def source_listing(
        self,
        kind: str,
        probe_query: Mapping[str, Any],
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> tuple[str, str] | None:
        api = str(probe_query.get("api") or "").strip()
        compatibility_kind = to_legacy_kind(to_local_kind(kind))
        source_kind = _L3_LISTING_SOURCE_FOR_API.get(api, {}).get(compatibility_kind)
        origin = context.site_origin()
        if source_kind is None or origin is None:
            return None
        if source_kind == "gitlab_dashboard_list":
            dashboard = "merge_requests" if compatibility_kind == "gitlab_mr" else "issues"
            if compatibility_kind == "gitlab_dashboard_list":
                dashboard = str(anchors.get("dashboard") or "issues")
            return source_kind, f"{origin}/dashboard/{dashboard}"
        project_path = _clean_project_path(
            str(probe_query.get("project_path") or anchors.get("project_path") or "")
        )
        if project_path:
            scope = "merge_requests" if compatibility_kind == "gitlab_mr" else "issues"
            return source_kind, f"{origin}/{project_path}/-/{scope}"
        return source_kind, f"{origin}/search"

    @staticmethod
    def canonicalize_project_path(project_path: str) -> str:
        return _clean_project_path(project_path)

    @staticmethod
    def clean_project_path(project_path: str) -> str:
        return _clean_project_path(project_path)

    @staticmethod
    def disambiguate_root_segment(task: Mapping[str, Any], segment: str) -> str | None:
        return _disambiguate_root_segment(task, segment)

    @staticmethod
    def anchors_from_item(item: Mapping[str, Any], *, kind_hint: str) -> dict[str, Any]:
        anchors: dict[str, Any] = {}
        project_id = item.get("project_id")
        if project_id is not None:
            anchors["project_id"] = str(project_id)
        iid = item.get("iid")
        if iid is not None:
            anchors["mr_iid" if "mr" in kind_hint else "issue_iid"] = str(iid)
        web_url = str(item.get("web_url") or "")
        if web_url:
            match = _ISSUE_RE.search(web_url) or _MR_RE.search(web_url)
            if match:
                anchors["project_path"] = match.group("project_path")
        return anchors

    @staticmethod
    def project_path_from_listing_task(
        instruction: str,
        *,
        resolved_start: str | None,
    ) -> str | None:
        if resolved_start:
            parsed = urlsplit(resolved_start)
            path = parsed.path if parsed.scheme else _path_and_query(resolved_start)
            issue_listing = _ISSUE_LISTING_RE.search(path)
            if issue_listing:
                return _clean_project_path(issue_listing.group("project_path"))
            match = _GITLAB_PROJECT_ROOT_RE.match(path)
            if match and "/-/" not in path:
                return _clean_project_path(match.group("project_path"))
        match = _GITLAB_PROJECT_PATH_IN_TEXT_RE.search(instruction)
        return _clean_project_path(match.group("project_path")) if match else None


__all__ = [
    "GitLabResourceKind",
    "GitLabSite",
    "to_legacy_kind",
    "to_local_kind",
]
