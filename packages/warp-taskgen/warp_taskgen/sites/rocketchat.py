"""Rocket.Chat's small static Site Targeting grammar.

The adapter describes only the channel/group room route needed by the
response-only TAC task.  It has no browser client, authentication behavior,
message mutation methods, or authenticated reader/writer behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from warp_taskgen.sites.contracts import (
    CanonicalRoute,
    SiteRouteContractFacts,
    TargetingContext,
)

RocketChatResourceKind = Literal["room"]
_ROOM_RE = re.compile(
    r"^/(?:channel|group)/(?P<room_id>[A-Za-z0-9][A-Za-z0-9_.:-]{0,127})"
    r"(?:/thread/(?P<thread_id>[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}))?/?$"
)
_ROOM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _route() -> CanonicalRoute:
    return CanonicalRoute(
        id="rocketchat.room",
        site="rocketchat",
        kind="room",
        allowed_start_url_patterns=("/channel/{room_id}", "/group/{room_id}"),
        compatibility_kind="rocketchat_room",
        anchor_examples=({"room_id": "project-alpha"},),
    )


class RocketChatSite:
    """Pure route adapter for generated Rocket.Chat room targets."""

    site = "rocketchat"
    supported_benchmarks = frozenset({"theagentcompany"})

    def validate(self) -> None:
        return None

    def validate_task(self, task: Mapping[str, Any]) -> tuple[str, str] | None:
        if not isinstance(task, Mapping):
            return "malformed_metadata", "Rocket.Chat task must be a mapping"
        conversation = task.get("conversation")
        if conversation is not None and not isinstance(conversation, Mapping):
            return "malformed_metadata", "Rocket.Chat conversation metadata must be a mapping"
        return None

    def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
        del context
        return (_route(),)

    def route_contract_facts(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, Any],
        kind: str,
    ) -> SiteRouteContractFacts:
        del profile
        if kind not in {"room", "rocketchat_room"}:
            return SiteRouteContractFacts()
        route = _route()
        return SiteRouteContractFacts(
            allowed_start_url_patterns=route.allowed_start_url_patterns,
            anchor_examples=route.anchor_examples,
            route_variant="channel_or_group",
        )

    def match(
        self, url: str, task: Mapping[str, Any], context: TargetingContext
    ) -> tuple[str, dict[str, Any]] | None:
        del task, context
        try:
            parsed = urlsplit(url)
        except ValueError:
            return None
        if parsed.query or parsed.fragment:
            return None
        match = _ROOM_RE.fullmatch(parsed.path)
        if match is None:
            return None
        anchors = {"room_id": match.group("room_id")}
        if match.group("thread_id") is not None:
            anchors["thread_id"] = match.group("thread_id")
        return "room", anchors

    def reconstruct(
        self,
        kind: str,
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> str | None:
        if kind not in {"room", "rocketchat_room"}:
            return None
        room_id = str(anchors.get("room_id") or "").strip()
        origin = context.site_origin()
        if origin is None or _ROOM_ID_RE.fullmatch(room_id) is None:
            return None
        thread_id = str(anchors.get("thread_id") or "").strip()
        if thread_id and _ROOM_ID_RE.fullmatch(thread_id) is None:
            return None
        suffix = f"/thread/{thread_id}" if thread_id else ""
        return f"{origin}/channel/{room_id}{suffix}"

    def is_listing(self, kind: str) -> bool:
        del kind
        return False

    def listing_start_url(
        self, kind: str, resolved_url: str, fallback_url: str | None
    ) -> str | None:
        del kind, resolved_url
        return fallback_url


__all__ = ["RocketChatResourceKind", "RocketChatSite"]
