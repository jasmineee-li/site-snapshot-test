"""Explicit experimental Phase 2c policy for the Classifieds POC."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from warp_taskgen.phase_2.phase_2c.policy import PreflightClassification, ProbeTarget
from warp_taskgen.phase_2.phase_2c.webarena_policy import (
    WebArenaFeasibilityPolicy,
    classify_webarena_probe,
    dedupe_targets,
    render_anchor_tokens,
    task_probe_url,
)
from warp_taskgen.phases.phase_2_reachability import resolve_start_url

_LISTING_ID_RE = re.compile(r"[1-9][0-9]*")
_CLASSIFIEDS_LOGIN_MARKERS = (
    "page=login",
    "action=login",
    'name="s_email"',
    'name="s_password"',
    'name="username"',
    'name="password"',
    "sign in",
    "log in",
)


def _exact_listing_id(url: str, instance_site_url: str) -> str | None:
    """Return an id only for the canonical same-origin item route."""

    try:
        parsed = urlsplit(url)
        origin = urlsplit(instance_site_url.rstrip("/"))
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except (TypeError, ValueError):
        return None
    if (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc):
        return None
    if parsed.path != "/index.php" or parsed.fragment:
        return None
    if len(query) != 2 or query[0] != ("page", "item") or query[1][0] != "id":
        return None
    listing_id = query[1][1]
    return listing_id if _LISTING_ID_RE.fullmatch(listing_id) else None


def _looks_like_classifieds_login_form(body: str) -> bool:
    if not isinstance(body, str) or "<form" not in body.lower():
        return False
    lowered = body.lower()
    return any(marker in lowered for marker in _CLASSIFIEDS_LOGIN_MARKERS)


def _location_is_classifieds_login(location: str | None) -> bool:
    if not isinstance(location, str) or not location.strip():
        return False
    lowered = location.casefold()
    if any(marker in lowered for marker in ("page=login", "action=login", "/login")):
        return True
    try:
        query = dict(parse_qsl(urlsplit(location).query, keep_blank_values=True))
    except (TypeError, ValueError):
        return False
    return any(str(value).casefold() in {"login", "signin", "sign_in"} for value in query.values())


@dataclass(frozen=True)
class ClassifiedsFeasibilityPolicy(WebArenaFeasibilityPolicy):
    """Probe one public listing through the fresh anonymous reader lane."""

    site: str = "classifieds"
    auth_path: str | None = None
    benchmark: str = "visualwebarena"

    def requires_authenticated_preflight(self) -> bool:
        return False

    def probe_targets(self, task: dict[str, Any], instance_site_url: str) -> list[ProbeTarget]:
        targets: list[ProbeTarget] = []
        start_url = task_probe_url(task, instance_site_url)

        resource = task.get("benign_target_resource")
        anchors_raw = resource.get("anchors") if isinstance(resource, Mapping) else None
        anchors = dict(anchors_raw) if isinstance(anchors_raw, Mapping) else {}
        seed = task.get("adversarial_data_seed")
        calls = seed.get("editor_calls") if isinstance(seed, Mapping) else None
        if not isinstance(calls, list):
            return []
        exact_listing_ids: set[str] = set()
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                continue
            if str(call.get("method") or "").strip() != "create_listing_reply":
                continue
            args = call.get("args")
            if not isinstance(args, Mapping):
                continue
            listing_id = render_anchor_tokens(args.get("listing_id"), anchors)
            if not listing_id or _LISTING_ID_RE.fullmatch(listing_id) is None:
                continue
            exact_listing_ids.add(listing_id)
            url = resolve_start_url(
                f"/index.php?page=item&id={listing_id}",
                instance_site_url,
            )
            targets.append(
                ProbeTarget(url, f"editor_call[{index}].classifieds.create_listing_reply")
            )
        if (
            start_url
            and (listing_id := _exact_listing_id(start_url, instance_site_url)) in exact_listing_ids
        ):
            targets.insert(0, ProbeTarget(start_url, "benign_start_url"))
        return dedupe_targets(targets)

    def classify_probe(
        self,
        *,
        status: int | None,
        headers: dict[str, str] | None,
        body_snippet: str,
        exception_name: str | None,
    ) -> PreflightClassification:
        """Treat Classifieds login redirects/forms as authentication failures."""

        classification = classify_webarena_probe(
            status=status,
            headers=headers,
            body_snippet=body_snippet,
            exception_name=exception_name,
        )
        location = headers.get("location") if isinstance(headers, Mapping) else None
        login_surface = (
            classification.kind == "login_redirect"
            or _looks_like_classifieds_login_form(body_snippet)
            or (300 <= (status or 0) < 400 and _location_is_classifieds_login(location))
        )
        if not login_surface:
            return classification
        return PreflightClassification(
            kind="auth_missing",
            quarantine=True,
            http_status=status,
            detail=(
                "Classifieds public listing unexpectedly requires authentication; "
                "the anonymous-reader contract is unavailable"
            ),
        )


__all__ = ["ClassifiedsFeasibilityPolicy"]
