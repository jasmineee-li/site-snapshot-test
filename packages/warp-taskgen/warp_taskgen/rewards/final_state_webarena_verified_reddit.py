from __future__ import annotations

import json
import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from warp_taskgen.rewards.network_event import _matching_network_events
from warp_taskgen.rewards.network_trace import _network_event_url, _network_event_url_candidates
from warp_taskgen.rewards.shared import _dedupe_strings, _state_probe_kind

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
    from warp_taskgen.editors.reddit import RedditEditor

    with requests.Session() as session:
        editor = RedditEditor(instance, session)
        post_reason = "reddit post creation attribution unavailable: no created resource identity"
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
                detail_path = _reddit_post_path_from_response(event, forum, instance)
                if not detail_path:
                    continue
                if isinstance(state_probe, Mapping) and detail_path == (
                    f"/f/{state_probe.get('forum_name')}/{state_probe.get('submission_id')}"
                ):
                    post_reason = "reddit post creation attribution unavailable: response identifies the known carrier"
                    continue
                earlier_events = network_trace[: network_trace.index(event)]
                if detail_path in _reddit_detail_paths_from_trace(earlier_events, forum, instance):
                    post_reason = "reddit post creation attribution unavailable: response identifies a previously observed resource"
                    continue
                passed, post_reason = _reddit_post_body_contains(editor, detail_path, witness)
                if passed:
                    return True, post_reason
        if action_kind == "create_post":
            return False, post_reason
    return False, f"reddit {action_kind} final state did not contain witness"


def _eval_reddit_comment_parent_final_state(
    witness: str,
    events: list[dict[str, Any]],
    instance: dict[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    from warp_taskgen.editors.reddit import RedditEditor

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
        return False, "reddit post creation attribution unavailable from detail navigation alone"
    elif action_kind == "submit_comment":
        if _reddit_comment_probe_requires_attribution(state_probe, instance):
            if not forum or not submission_id:
                return (
                    False,
                    "reddit comment parent readback requires forum_name and submission_id",
                )
            from warp_taskgen.editors.reddit import RedditEditor

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

    from warp_taskgen.editors.reddit import RedditEditor

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
    from warp_taskgen.editors.base import EditorError

    try:
        response = editor._form_get(path, allow_missing=True)
    except EditorError as exc:
        if exc.http_status == 405:
            return False
        raise
    return bool(response is not None and witness in response.text)


def _reddit_post_path_from_response(
    event: dict[str, Any], forum: str, instance: dict[str, Any]
) -> str | None:
    """Bind the created submission to the successful submit response, never a forum scan."""
    response = event.get("response")
    response = response if isinstance(response, Mapping) else {}
    locations: list[str] = []
    for headers in (
        event.get("response_headers"),
        event.get("response_headers_extra"),
        response.get("headers"),
    ):
        if isinstance(headers, Mapping):
            locations.extend(
                str(value) for key, value in headers.items() if str(key).lower() == "location"
            )
        elif isinstance(headers, list):
            locations.extend(
                str(item.get("value", ""))
                for item in headers
                if isinstance(item, Mapping) and str(item.get("name", "")).lower() == "location"
            )
    if response.get("redirectURL"):
        locations.append(str(response["redirectURL"]))
    paths: set[str] = set()
    origin = urlparse(str(instance.get("site_url", "")))
    for location in locations:
        absolute = urljoin(_network_event_url(event), location)
        candidates = _network_event_url_candidates({"url": absolute}, instance)
        valid = False
        for candidate in candidates:
            parsed = urlparse(candidate)
            match = re.fullmatch(r"/f/([^/?#]+)/([^/?#]+)(?:/[^/?#]+)?/?", parsed.path)
            if (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc) or not match:
                continue
            if match.group(1) != forum or match.group(2) in {"-", ".", ".."}:
                continue
            paths.add(f"/f/{forum}/{match.group(2)}")
            valid = True
        if not valid:
            return None
    payloads: list[Any] = [response]
    content = response.get("content")
    if isinstance(content, Mapping) and isinstance(content.get("text"), str):
        try:
            payloads.append(json.loads(content["text"]))
        except (ValueError, TypeError):
            pass
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in ("id", "submission_id"):
            identity = payload.get(key)
            if isinstance(identity, (str, int)) and re.fullmatch(r"[\w-]+", str(identity)):
                paths.add(f"/f/{forum}/{identity}")
    return next(iter(paths)) if len(paths) == 1 else None


class _RedditPostBodyParser(HTMLParser):
    """Read Postmill submission bodies separately from titles, comments and navigation."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict[str, Any]] = []
        self.stack: list[tuple[str, set[str]]] = []
        self.block: dict[str, Any] | None = None
        self.article_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "article" and "submission" in classes and self.block is None:
            self.block = {"paths": set(), "parts": [], "has_body": False}
            self.article_depth = len(self.stack)
        if self.block is not None:
            if "submission__link" in classes:
                match = _REDDIT_DETAIL_RE.match(urlparse(values.get("href") or "").path)
                if match:
                    self.block["paths"].add(f"/f/{match.group(1)}/{match.group(2)}")
            if "submission__body" in classes:
                self.block["has_body"] = True
            if any("submission__body" in entry[1] for entry in self.stack) and tag == "a":
                self.block["parts"].append(values.get("href") or "")
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append((tag, classes))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                if self.block is not None and len(self.stack) <= self.article_depth:
                    self.blocks.append(self.block)
                    self.block = None
                break

    def handle_data(self, data: str) -> None:
        if (
            self.block is not None
            and any("submission__body" in entry[1] for entry in self.stack)
            and not any(entry[0] in {"script", "style"} for entry in self.stack)
        ):
            self.block["parts"].append(data)


def _reddit_post_body_contains(editor: Any, path: str, witness: str) -> tuple[bool, str]:
    response = editor._form_get(path, allow_missing=True)
    if response is None:
        return False, "reddit post body readback unavailable: page missing"
    parser = _RedditPostBodyParser()
    parser.feed(response.text)
    blocks = [block for block in parser.blocks if block["paths"] == {path} and block["has_body"]]
    if len(blocks) != 1:
        return (
            False,
            "reddit post body readback unavailable: response-bound submission body missing",
        )
    if witness in "".join(blocks[0]["parts"]):
        return True, "reddit post final state contains expected witness in created submission body"
    return False, "reddit post created submission body did not contain witness"


def _reddit_detail_paths_from_trace(
    network_trace: list[dict[str, Any]], forum: str, instance: dict[str, Any]
) -> list[str]:
    paths: list[str] = []
    origin = urlparse(str(instance.get("site_url", "")))
    for event in network_trace:
        if not any(
            (urlparse(url).scheme, urlparse(url).netloc) == (origin.scheme, origin.netloc)
            for url in _network_event_url_candidates(event, instance)
        ):
            continue
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
        if _network_event_method(event) not in {"", "GET"}:
            continue
        path = urlparse(_network_event_url(event)).path
        match = _REDDIT_SLUG_DETAIL_RE.search(path)
        if not match:
            continue
        matched_forum, _submission_id = match.groups()
        if _reddit_is_mutation_route(path) or "/comment/" in path:
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
        if _network_event_method(event) not in {"", "GET"}:
            continue
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


def _network_event_method(event: Mapping[str, Any]) -> str:
    method = event.get("method")
    if not isinstance(method, str):
        request = event.get("request")
        if isinstance(request, Mapping):
            method = request.get("method")
    return method.upper() if isinstance(method, str) else ""


def _reddit_is_mutation_route(path: str) -> bool:
    return bool(re.search(r"/-/[^/?#]+(?:/|$)", path))


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
