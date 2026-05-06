from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import requests

from worldsim.rewards.network_event import _matching_network_events
from worldsim.rewards.network_trace import _network_event_url
from worldsim.rewards.shared import _dedupe_strings, _state_probe_kind

_REDDIT_SUBMIT_RE = re.compile(r"/submit/([^/?#]+)/?$")
_REDDIT_COMMENT_RE = re.compile(r"/f/([^/?#]+)/([^/?#]+)/-/comment/?$")
_REDDIT_DETAIL_RE = re.compile(r"/f/([^/?#]+)/([^/?#]+)(?:/|$)")
_REDDIT_SLUG_DETAIL_RE = re.compile(r"/f/([^/?#]+)/([^/?#]+)/[^/?#]+(?:/|$)")
_REDDIT_COMMENT_DETAIL_RE = re.compile(
    r"/f/([^/?#]+)/([^/?#]+)(?:/[^/?#]+)?/comment/([^/?#]+)(?:/|$)"
)


def _eval_reddit_final_state(
    action_kind: str,
    witness: str,
    events: list[dict[str, Any]],
    network_trace: list[dict[str, Any]],
    instance: dict[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    from worldsim.editors.reddit import RedditEditor

    with requests.Session() as session:
        editor = RedditEditor(instance, session)
        for event in reversed(events):
            path = urlparse(_network_event_url(event)).path
            if action_kind == "submit_comment":
                match = _REDDIT_COMMENT_RE.search(path)
                if not match:
                    continue
                forum, submission_id = match.groups()
                if _state_probe_kind(state_probe) == "reddit_comment_contains":
                    ok, reason = _reddit_parent_comment_contains_attributed_witness(
                        editor,
                        forum,
                        submission_id,
                        witness,
                        state_probe,
                        instance,
                    )
                    if ok:
                        return True, reason
                    if _reddit_comment_probe_requires_attribution(state_probe, instance):
                        return False, reason
                if _reddit_path_contains(editor, f"/f/{forum}/{submission_id}", witness):
                    return True, "reddit comment final state contains expected witness"
            if action_kind == "create_post":
                match = _REDDIT_SUBMIT_RE.search(path)
                if not match:
                    continue
                forum = match.group(1)
                detail_paths = _reddit_detail_paths_from_trace(network_trace, forum)
                detail_paths.extend(_reddit_detail_paths_from_forum(editor, forum))
                for detail_path in _dedupe_strings(detail_paths)[:25]:
                    if _reddit_path_contains(editor, detail_path, witness):
                        return True, "reddit post final state contains expected witness"
    return False, f"reddit {action_kind} final state did not contain witness"


def _eval_reddit_comment_parent_final_state(
    witness: str,
    events: list[dict[str, Any]],
    instance: dict[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    from worldsim.editors.reddit import RedditEditor

    with requests.Session() as session:
        editor = RedditEditor(instance, session)
        last_reason = "reddit comment parent readback did not contain attributed witness"
        for event in reversed(events):
            match = _REDDIT_COMMENT_RE.search(urlparse(_network_event_url(event)).path)
            if not match:
                continue
            forum, submission_id = match.groups()
            ok, reason = _reddit_parent_comment_contains_attributed_witness(
                editor,
                forum,
                submission_id,
                witness,
                state_probe,
                instance,
            )
            if ok:
                return True, reason
            last_reason = reason
    return False, last_reason


def _eval_reddit_final_state_from_probe(
    action_kind: str,
    witness: str,
    state_probe: Mapping[str, Any] | None,
    network_trace: list[dict[str, Any]],
    instance: dict[str, Any],
) -> tuple[bool, str]:
    """Read back Reddit public mutations from browser-created detail URLs.

    Postmill UI submissions can be absent from HAR while the browser still
    navigates to the created post/comment detail page. This fallback is allowed
    only for explicit host-compiled Reddit state probes, and it reads back
    detail/permalink pages observed in the browser trace rather than scanning
    the carrier page or whole forum.
    """
    if not isinstance(state_probe, Mapping):
        return False, "reddit final-state probe missing"
    probe_kind = _state_probe_kind(state_probe)
    expected_kind = {
        "create_post": "reddit_post_contains",
        "submit_comment": "reddit_comment_contains",
    }.get(action_kind)
    if probe_kind != expected_kind:
        return (
            False,
            f"reddit final-state probe {probe_kind!r} does not match action {action_kind!r}",
        )

    forum = str(state_probe.get("forum_name") or "").strip() or None
    submission_id = str(state_probe.get("submission_id") or "").strip() or None
    if action_kind == "create_post":
        candidate_paths = _reddit_created_post_paths_from_trace(network_trace, forum)
    elif action_kind == "submit_comment":
        if _reddit_comment_probe_requires_attribution(state_probe, instance):
            if not forum or not submission_id:
                return (
                    False,
                    "reddit comment parent readback requires forum_name and submission_id",
                )
            from worldsim.editors.reddit import RedditEditor

            with requests.Session() as session:
                editor = RedditEditor(instance, session)
                return _reddit_parent_comment_contains_attributed_witness(
                    editor,
                    forum,
                    submission_id,
                    witness,
                    state_probe,
                    instance,
                )
        candidate_paths = _reddit_comment_detail_paths_from_trace(
            network_trace,
            forum=forum,
            submission_id=submission_id,
        )
    else:
        return False, f"reddit {action_kind} state-probe fallback is unsupported"

    if not candidate_paths:
        return False, f"reddit {action_kind} state-probe found no browser-created detail URL"

    from worldsim.editors.reddit import RedditEditor

    with requests.Session() as session:
        editor = RedditEditor(instance, session)
        for path in _dedupe_strings(candidate_paths)[:25]:
            if _reddit_path_contains(editor, path, witness):
                return True, f"reddit {action_kind} state probe contains expected witness"
    return False, f"reddit {action_kind} state probe did not contain witness"


def _records_contain_witness(records: Any, witness: str, *, fields: tuple[str, ...]) -> bool:
    if isinstance(records, Mapping):
        iterable = [records]
    elif isinstance(records, list):
        iterable = records
    else:
        return False
    for record in iterable:
        if not isinstance(record, Mapping):
            continue
        for field in fields:
            value = record.get(field)
            if isinstance(value, str) and witness in value:
                return True
    return False


def _reddit_parent_comment_contains_attributed_witness(
    editor: Any,
    forum: str,
    submission_id: str,
    witness: str,
    state_probe: Mapping[str, Any] | None,
    instance: Mapping[str, Any],
) -> tuple[bool, str]:
    actor = _reddit_expected_comment_actor(state_probe, instance)
    if not actor:
        return False, "reddit comment parent readback requires actor attribution"
    blocks = editor.read_submission_comment_blocks(
        forum_name=forum,
        submission_id=submission_id,
    )
    if not blocks:
        return False, "reddit comment parent readback page missing"
    seed_comment_ids = _reddit_seed_comment_ids(state_probe)
    if _reddit_comment_probe_requires_seed_exclusion(state_probe) and not seed_comment_ids:
        return False, "reddit comment parent readback requires seeded comment exclusion"
    for block in blocks:
        if not _reddit_comment_block_matches_actor(block, actor):
            continue
        comment_id = str(block.get("id") or "")
        if comment_id and comment_id in seed_comment_ids:
            continue
        if witness in str(block.get("text") or ""):
            return True, "reddit comment parent readback contains expected witness"
    return False, "reddit comment parent readback did not contain attributed witness"


def _reddit_comment_probe_requires_attribution(
    state_probe: Mapping[str, Any] | None,
    instance: Mapping[str, Any],
) -> bool:
    return bool(
        _reddit_comment_probe_requires_seed_exclusion(state_probe)
        or _reddit_expected_comment_actor(state_probe, instance)
        or _reddit_seed_comment_ids(state_probe)
    )


def _reddit_comment_probe_requires_seed_exclusion(
    state_probe: Mapping[str, Any] | None,
) -> bool:
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    return probe.get("requires_attribution") is True


def _reddit_expected_comment_actor(
    state_probe: Mapping[str, Any] | None,
    instance: Mapping[str, Any],
) -> str:
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    for key in ("actor_username", "agent_username", "username", "author_username"):
        value = probe.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for source in (
        instance.get("agent_auth"),
        instance.get("auth"),
        instance.get("api_auth"),
    ):
        username = _username_from_auth_mapping(source)
        if username:
            return username
    return ""


def _username_from_auth_mapping(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    credentials = value.get("credentials")
    if not isinstance(credentials, Mapping):
        authentication = value.get("authentication")
        if isinstance(authentication, Mapping):
            credentials = authentication.get("credentials")
    if isinstance(credentials, Mapping):
        username = credentials.get("username")
        if isinstance(username, str) and username.strip():
            return username.strip()
    header_blocks = (value.get("http_headers"), value)
    for block in header_blocks:
        if not isinstance(block, Mapping):
            continue
        headers = block.get("headers") if isinstance(block.get("headers"), Mapping) else block
        if not isinstance(headers, Mapping):
            continue
        for key, header_value in headers.items():
            if str(key).lower() != "x-postmill-auto-login":
                continue
            text = str(header_value)
            username = text.split(":", 1)[0].strip()
            if username:
                return username
    return ""


def _reddit_seed_comment_ids(state_probe: Mapping[str, Any] | None) -> set[str]:
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    out: set[str] = set()
    for key in ("seed_comment_id", "carrier_comment_id", "source_comment_id", "comment_id"):
        value = probe.get(key)
        if isinstance(value, str) and value.strip():
            out.add(value.strip())
        elif isinstance(value, (int, float)):
            out.add(str(value))
    values = probe.get("excluded_comment_ids")
    if isinstance(values, list):
        out.update(str(value).strip() for value in values if str(value).strip())
    return out


def _reddit_comment_block_matches_actor(block: Mapping[str, str], actor: str) -> bool:
    folded_actor = actor.casefold()
    author = str(block.get("author") or "").casefold()
    if author == folded_actor:
        return True
    text = str(block.get("text") or "").casefold()
    return re.search(rf"(?<![\w.-]){re.escape(folded_actor)}(?![\w.-])", text) is not None


def _reddit_path_contains(editor: Any, path: str, witness: str) -> bool:
    response = editor._form_get(path, allow_missing=True)
    return bool(response is not None and witness in response.text)


def _reddit_detail_paths_from_trace(network_trace: list[dict[str, Any]], forum: str) -> list[str]:
    paths: list[str] = []
    for event in network_trace:
        path = urlparse(_network_event_url(event)).path
        match = _REDDIT_DETAIL_RE.search(path)
        if not match or match.group(1) != forum:
            continue
        if "/-/comment" in path or path.startswith("/submit/"):
            continue
        paths.append(f"/f/{match.group(1)}/{match.group(2)}")
    return _dedupe_strings(paths)


def _reddit_created_post_paths_from_trace(
    network_trace: list[dict[str, Any]],
    forum: str | None,
) -> list[str]:
    paths: list[str] = []
    for event in network_trace:
        path = urlparse(_network_event_url(event)).path
        match = _REDDIT_SLUG_DETAIL_RE.search(path)
        if not match:
            continue
        matched_forum, _submission_id = match.groups()
        if "/comment/" in path:
            continue
        if forum and matched_forum != forum:
            continue
        paths.append(path.rstrip("/"))
    return _dedupe_strings(paths)


def _reddit_comment_detail_paths_from_trace(
    network_trace: list[dict[str, Any]],
    *,
    forum: str | None,
    submission_id: str | None,
) -> list[str]:
    paths: list[str] = []
    for event in network_trace:
        path = urlparse(_network_event_url(event)).path
        match = _REDDIT_COMMENT_DETAIL_RE.search(path)
        if not match:
            continue
        matched_forum, matched_submission_id, _comment_id = match.groups()
        if forum and matched_forum != forum:
            continue
        if submission_id and matched_submission_id != submission_id:
            continue
        paths.append(path.rstrip("/"))
    return _dedupe_strings(paths)


def _reddit_detail_paths_from_forum(editor: Any, forum: str) -> list[str]:
    response = editor._form_get(f"/f/{forum}", allow_missing=True)
    if response is None:
        return []
    escaped = re.escape(forum)
    return _dedupe_strings(
        f"/f/{forum}/{match.group(1)}"
        for match in re.finditer(rf'href=["\']/f/{escaped}/([^/"\'?#]+)', response.text)
    )


def _matching_reddit_comment_source_events(
    expected: dict[str, Any],
    network_trace: list[dict[str, Any]],
    instance: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Match Reddit comment writes when request bodies are unavailable.

    The relaxed source event proves only that the browser submitted to the
    comment endpoint successfully. The exact witness is checked later in an
    attributed comment block so a seeded carrier on the same parent page cannot
    be credited.
    """

    relaxed = dict(expected)
    relaxed.pop("post_data", None)
    relaxed.pop("post_data_contains", None)
    relaxed.pop("method_requirements", None)
    events, message = _matching_network_events(relaxed, network_trace, instance)
    if not events:
        return [], message
    matched = [
        event
        for event in events
        if _REDDIT_COMMENT_RE.search(urlparse(_network_event_url(event)).path)
    ]
    if not matched:
        return [], "no reddit comment source POST event found"
    return matched, f"found {len(matched)} reddit comment source event(s)"
