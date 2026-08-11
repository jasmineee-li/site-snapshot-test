"""Reddit/Postmill's deterministic Site Targeting grammar."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from worldsim.sites.catalog import CanonicalRoute, TargetingContext, _path_and_query

RedditResourceKind = Literal["submission", "forum", "dashboard_list"]

_LOCAL_TO_LEGACY: dict[str, str] = {
    "submission": "reddit_submission",
    "forum": "reddit_forum",
    "dashboard_list": "reddit_dashboard_list",
}
_LEGACY_TO_LOCAL = {legacy: local for local, legacy in _LOCAL_TO_LEGACY.items()}

_SUBMISSION_RE = re.compile(r"/f/(?P<forum_name>[^/?#]+)/(?P<submission_id>\d+)(?:/|$|\b)")
_FORUM_RE = re.compile(r"/f/(?P<forum_name>[^/?#\d][^/?#]*)(?:/?(?:\?|$))")
_REDDIT_SUBMIT_RE = re.compile(r"/submit/(?P<forum_name>[^/?#]+)")
_REDDIT_USER_DASH_RE = re.compile(r"/user/(?P<user>[^/?#]+)/(?P<dash>submitted|comments)\b")

_L3_PROBE_KINDS_FOR_API: dict[str, frozenset[str]] = {
    "list_user_submitted": frozenset({"reddit_dashboard_list"}),
    "list_user_comments": frozenset({"reddit_dashboard_list"}),
    "find_submission_by_title": frozenset({"reddit_submission"}),
    "list_forum_submissions_recent": frozenset({"reddit_submission", "reddit_forum"}),
    "none": frozenset(),
}
_L3_LISTING_SOURCE_FOR_API: dict[str, dict[str, str]] = {
    "find_submission_by_title": {"reddit_submission": "reddit_dashboard_list"},
    "list_forum_submissions_recent": {"reddit_submission": "reddit_forum"},
}


def to_local_kind(kind: str) -> str:
    return _LEGACY_TO_LOCAL.get(kind, kind)


def to_legacy_kind(kind: str) -> str:
    return _LOCAL_TO_LEGACY.get(kind, kind)


def _route(
    kind: str,
    patterns: tuple[str, ...],
    *examples: Mapping[str, Any],
) -> CanonicalRoute:
    return CanonicalRoute(
        id=f"reddit.{kind}",
        site="reddit",
        kind=kind,
        allowed_start_url_patterns=patterns,
        compatibility_kind=to_legacy_kind(kind),
        anchor_examples=tuple(examples),
    )


class RedditSite:
    site = "reddit"
    supported_benchmarks = frozenset({"webarena_verified"})

    def validate(self) -> None:
        return None

    def validate_task(self, task: Mapping[str, Any]) -> tuple[str, str] | None:
        del task
        return None

    def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
        del context
        return (
            _route(
                "submission",
                ("/f/{forum_name}/{submission_id}",),
                {"forum_name": "books", "submission_id": "1"},
            ),
            _route("forum", ("/f/{forum_name}",), {"forum_name": "books"}),
            _route(
                "dashboard_list",
                ("/user/{username}/submitted", "/user/{username}/comments"),
                {"username": "user", "dashboard": "submitted"},
                {"username": "user", "dashboard": "comments"},
            ),
        )

    def match(
        self, url: str, task: Mapping[str, Any], context: TargetingContext
    ) -> tuple[str, dict[str, Any]] | None:
        del task, context
        path_and_query = _path_and_query(url)
        patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
            ("submission", _SUBMISSION_RE),
            ("dashboard_list", _REDDIT_USER_DASH_RE),
            ("forum", _FORUM_RE),
            ("forum", _REDDIT_SUBMIT_RE),
        )
        for kind, pattern in patterns:
            match = pattern.search(path_and_query)
            if not match:
                continue
            anchors = {key: value for key, value in match.groupdict().items() if value}
            if kind == "dashboard_list":
                anchors["dashboard"] = anchors.pop("dash", "")
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
        if local_kind == "submission":
            forum, submission_id = anchors.get("forum_name"), anchors.get("submission_id")
            return f"{base}/f/{forum}/{submission_id}" if forum and submission_id else None
        if local_kind == "forum":
            forum = anchors.get("forum_name")
            return f"{base}/f/{forum}" if forum else None
        if local_kind == "dashboard_list":
            user = (
                anchors.get("user")
                or anchors.get("username")
                or context.profile.get("username")
            )
            dashboard = anchors.get("dashboard")
            return f"{base}/user/{user}/{dashboard}" if user and dashboard else None
        return None

    def is_listing(self, kind: str) -> bool:
        return to_local_kind(kind) == "dashboard_list"

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
        forum_name = str(
            probe_query.get("forum_name") or anchors.get("forum_name") or ""
        ).strip()
        if not forum_name:
            return None
        return source_kind, f"{origin}/f/{forum_name}"

    @staticmethod
    def anchors_from_submission(entry: Mapping[str, Any], forum_name: str) -> dict[str, Any]:
        submission_id = entry.get("id") or entry.get("submission_id")
        anchors: dict[str, Any] = {"forum_name": forum_name}
        if submission_id is not None:
            anchors["submission_id"] = str(submission_id)
        return anchors


__all__ = [
    "RedditResourceKind",
    "RedditSite",
    "to_legacy_kind",
    "to_local_kind",
]
