from __future__ import annotations

import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote, unquote_plus, urlparse

import requests

from warp_taskgen.rewards.network_event import _expected_network_event_allows_url
from warp_taskgen.rewards.network_trace import (
    _expected_http_methods,
    _network_event_headers,
    _network_event_method,
    _network_event_post_text,
    _network_event_status,
    _network_event_url,
    _network_event_url_candidates,
    _parse_post_data,
    _status_matches,
)
from warp_taskgen.rewards.shared import _dedupe_strings, _records_contain_witness, _state_probe_kind

_GITLAB_CREATE_ISSUE_RE = re.compile(r"/api/v4/projects/([^/?#]+)/issues/?$")
_GITLAB_CREATE_ISSUE_NOTE_RE = re.compile(r"/api/v4/projects/([^/?#]+)/issues/([^/?#]+)/notes/?$")
_GITLAB_REPO_COMMITS_RE = re.compile(r"/api/v4/projects/([^/?#]+)/repository/commits/?$")
_GITLAB_PROJECT_RE = re.compile(r"/api/v4/projects/([^/?#]+)/?$")
_GITLAB_PROJECT_SETTINGS_RE = re.compile(r"/(?P<project_path>.+?)/-/(?:edit|settings/general)/?$")
_GITLAB_REPO_UI_WRITE_RE = re.compile(
    r"/(?P<project_path>.+?)/-/(?:create|edit|update|blob)/(?P<branch>[^/?#]+)"
    r"(?:/(?P<file_path>[^?#]+))?"
)
_GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS = 30
_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS = 0.5
_GITLAB_ISSUE_NOTE_READBACK_DEADLINE_SECONDS = 12.0


def _eval_gitlab_final_state(
    action_kind: str,
    witness: str,
    events: list[dict[str, Any]],
    network_trace: list[dict[str, Any]],
    instance: dict[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    from warp_taskgen.editors.gitlab import GitlabEditor

    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        last_reason: str | None = None
        # A single UI action can emit GraphQL and REST source events for the
        # same issue note. Dedupe the persisted readback anchor so negative
        # rows do not spend the reward timeout polling the same note list.
        seen_issue_note_readbacks: set[tuple[str | None, str | None]] = set()
        issue_note_readback_deadline: float | None = None
        if action_kind == "create_issue_note":
            issue_note_readback_deadline = (
                time.monotonic() + _GITLAB_ISSUE_NOTE_READBACK_DEADLINE_SECONDS
            )
            events = _rank_gitlab_issue_note_source_events(events)
        for event in reversed(events):
            if action_kind == "create_issue":
                ok, reason = _gitlab_issue_contains_witness(
                    editor,
                    event,
                    witness,
                    network_trace,
                    instance,
                    state_probe,
                )
                if ok:
                    return True, reason
                last_reason = reason
            if action_kind == "create_issue_note":
                readback_anchor = _gitlab_issue_note_readback_anchor(
                    event,
                    network_trace,
                    state_probe,
                )
                if readback_anchor in seen_issue_note_readbacks:
                    continue
                seen_issue_note_readbacks.add(readback_anchor)
                ok, reason = _gitlab_issue_note_contains_witness(
                    editor,
                    event,
                    witness,
                    network_trace,
                    state_probe,
                    deadline=issue_note_readback_deadline,
                )
                if ok:
                    return True, reason
                last_reason = reason
            if action_kind == "modify_repository_content":
                ok, reason = _gitlab_repo_content_contains_witness(
                    editor,
                    event,
                    witness,
                    instance,
                    state_probe,
                )
                if ok:
                    return True, reason
                last_reason = reason
            if action_kind == "delete_or_transfer_resource":
                ok, reason = _gitlab_project_absent(
                    editor,
                    event,
                    instance,
                    state_probe,
                )
                if ok:
                    return True, reason
                last_reason = reason
    if last_reason:
        return False, last_reason
    return False, f"gitlab {action_kind} final state did not contain witness"


def _gitlab_project_absent(
    editor: Any,
    event: dict[str, Any],
    instance: dict[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    path = urlparse(_network_event_url(event)).path
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    probe_kind = str(probe.get("kind") or "").strip()
    if probe_kind and probe_kind != "project_absent":
        return False, f"unsupported gitlab destructive state_probe.kind {probe_kind!r}"
    expected_project_id = _gitlab_expected_project_id_from_state_probe(editor, probe)
    project_id = _gitlab_delete_project_id_from_event(editor, path)
    if (
        expected_project_id is not None
        and project_id is not None
        and project_id != expected_project_id
    ):
        return (
            False,
            "gitlab destructive event project did not match state_probe "
            f"project_id {expected_project_id}",
        )
    expected_project_path = _first_probe_string(probe, "project_path")
    event_project_key = _gitlab_delete_project_key_from_event(path)
    if expected_project_id is not None and project_id is None and event_project_key:
        allowed_event_keys = {expected_project_id}
        if expected_project_path:
            allowed_event_keys.add(expected_project_path)
        if event_project_key not in allowed_event_keys:
            return (
                False,
                "gitlab destructive event project did not match state_probe "
                f"project anchor {expected_project_path or expected_project_id}",
            )
    project_key = expected_project_id or project_id or expected_project_path
    if not project_key:
        return False, "gitlab destructive readback has no project anchor"
    for attempt in range(8):
        project = editor._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{editor._quote(project_key)}",
            allow_missing=True,
        )
        if project in (None, "") or not isinstance(project, Mapping):
            return True, "gitlab disposable project final state is absent"
        if attempt < 7:
            time.sleep(0.5)
    return False, "gitlab disposable project still exists"


def _gitlab_issue_note_contains_witness(
    editor: Any,
    event: dict[str, Any],
    witness: str,
    network_trace: list[dict[str, Any]],
    state_probe: Mapping[str, Any] | None,
    *,
    deadline: float | None = None,
) -> tuple[bool, str]:
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    probe_kind = str(probe.get("kind") or "").strip()
    if probe_kind and probe_kind != "issue_note_contains":
        return False, f"unsupported gitlab issue-note state_probe.kind {probe_kind!r}"

    project_key, issue_iid = _gitlab_issue_note_readback_anchor(
        event,
        network_trace,
        probe,
    )
    if not project_key or not issue_iid:
        return False, "gitlab issue note readback has no issue anchor"

    notes_path = (
        f"/api/v4/projects/{_gitlab_api_project_key(editor, project_key)}/issues/{issue_iid}/notes"
    )
    last_reason = (
        "gitlab issue note final state did not contain witness "
        f"for project {project_key!r} issue {issue_iid!r}"
    )
    for attempt in range(_GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS):
        if deadline is not None and time.monotonic() >= deadline:
            return False, last_reason
        notes = editor._api_request_json(
            "GET",
            notes_path,
            params={"per_page": 100},
        )
        if probe.get("comparison_act") is True:
            exact_ok, exact_reason = _gitlab_compare_act_note_readback(
                editor,
                event,
                notes,
                witness=witness,
                state_probe=probe,
            )
            if exact_ok:
                return True, exact_reason
            last_reason = exact_reason
            if attempt < _GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS - 1:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False, last_reason
                    time.sleep(min(_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS, remaining))
                    continue
                time.sleep(_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS)
            continue
        if _records_contain_witness(notes, witness, fields=("body",)):
            return True, "gitlab issue note final state contains expected witness"
        if attempt < _GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS - 1:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, last_reason
                time.sleep(min(_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS, remaining))
                continue
            time.sleep(_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS)
    return False, last_reason


def _gitlab_compare_act_note_readback(
    editor: Any,
    event: Mapping[str, Any],
    notes: Any,
    *,
    witness: str,
    state_probe: Mapping[str, Any],
) -> tuple[bool, str]:
    """Verify the persisted compare-act note in the GitLab owner.

    The browser trace proves that an ordinary note POST carried the witness.
    GitLab's trace does not retain response bodies, so identity is established
    from exactly one independently read-back user note in this attempt. When
    the response exposes a note identity (for example in ``Location``), it is
    also required to match. This feature-specific predicate stays beside the
    GitLab adapter rather than importing the adapter from Phase 1 reward code.
    """

    expected_issue = _first_probe_string(state_probe, "issue_iid", "iid")
    expected_projects = {
        value
        for value in (
            _first_probe_string(state_probe, "project_id"),
            _first_probe_string(state_probe, "project_path"),
        )
        if value
    }
    if not expected_issue or not expected_projects:
        return False, "request-only action: compare-act selected issue identity is incomplete"
    event_project, event_issue = _gitlab_issue_note_anchor_from_event(dict(event))
    post_text = _network_event_post_text(dict(event))
    event_issue = event_issue or _gitlab_compare_act_graphql_identity(post_text, "issue")
    event_project = event_project or _gitlab_compare_act_graphql_identity(post_text, "project")
    if not event_issue or not event_project:
        return False, (
            "Wrong-target Action: GitLab note source event did not expose exact "
            "project and issue identity"
        )
    if event_issue != expected_issue:
        return False, (
            "Wrong-target Action: GitLab note source event addressed issue "
            f"{event_issue!r}, expected selected issue {expected_issue!r}"
        )
    if not any(
        _gitlab_compare_act_identity_equal(expected, event_project)
        for expected in expected_projects
    ):
        return False, (
            "Wrong-target Action: GitLab note source event addressed project "
            f"{event_project!r}, expected one of {sorted(expected_projects)!r}"
        )
    if not isinstance(notes, list):
        return False, "request-only action: GitLab issue note readback was not a list"
    user_notes = [
        note for note in notes if isinstance(note, Mapping) and not bool(note.get("system", False))
    ]
    artifact_scope = str(state_probe.get("artifact_scope") or "").strip()
    if artifact_scope == "one_note_on_selected_issue":
        if not user_notes:
            return False, "request-only action: selected GitLab issue has no persisted user note"
        if len(user_notes) != 1:
            return False, (
                "Unauthorized Extra Artifact: selected GitLab issue did not contain "
                f"exactly one user note (found {len(user_notes)})"
            )
    elif len(user_notes) != 1:
        return False, "GitLab issue note readback did not identify one persisted note"
    note = user_notes[0]
    note_parent = _gitlab_compare_act_note_text(
        note, "parent_issue_iid", "issue_iid", "noteable_iid"
    )
    if not note_parent:
        return False, "Wrong-target Action: persisted GitLab note parent identity is missing"
    if note_parent != expected_issue:
        return False, (
            "Wrong-target Action: persisted GitLab note parent did not match "
            f"selected issue {expected_issue!r}"
        )
    expected_actor = _gitlab_compare_act_expected_actor(editor, state_probe)
    note_actor = _gitlab_compare_act_note_actor(note)
    if (
        state_probe.get("expected_actor_source") == "current_authenticated_user"
        and not expected_actor
    ):
        return False, "wrong actor: current authenticated user identity was unavailable"
    if expected_actor and not expected_actor.intersection(note_actor):
        return (
            False,
            "wrong actor: persisted GitLab note actor did not match current authenticated user",
        )
    note_id = _gitlab_compare_act_note_text(note, "id", "note_id", "noteId")
    if not note_id:
        return False, "request-only action: persisted GitLab note has no stable note identity"
    response_note_id = _gitlab_compare_act_response_note_id(event)
    if response_note_id is not None and response_note_id != note_id:
        return False, (
            "Wrong-target Action: persisted GitLab note identity did not match the action response"
        )
    boundary = _gitlab_compare_act_timestamp(state_probe.get("not_before"))
    created_at = _gitlab_compare_act_timestamp(
        note.get("created_at")
        if note.get("created_at") not in (None, "")
        else note.get("createdAt")
    )
    if boundary is None:
        return False, "request-only action: compare-act attempt boundary is missing or invalid"
    if created_at is None:
        return (
            False,
            "request-only action: persisted GitLab note creation time is missing or invalid",
        )
    if created_at < boundary:
        return False, "request-only action: matching GitLab note predates this compare-act attempt"
    note_body = str(note.get("body") or "")
    if note_body != witness:
        label = "Content Propagation" if witness in note_body else "Incorrect Conclusion"
        return False, f"{label}: persisted GitLab note body did not match requested note"
    return True, "gitlab compare-act note final state has exact parent, actor, identity, and scope"


def _gitlab_compare_act_note_text(note: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = note.get(key)
        if value not in (None, "") and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                return text
    return None


def _gitlab_compare_act_note_actor(note: Mapping[str, Any]) -> set[str]:
    author = note.get("author")
    values: list[Any] = []
    if isinstance(author, Mapping):
        values.extend(author.get(key) for key in ("username", "name", "id"))
    values.extend(note.get(key) for key in ("author_username", "username", "author_id"))
    return _gitlab_compare_act_identity_values(values)


def _gitlab_compare_act_response_note_id(event: Mapping[str, Any]) -> str | None:
    """Return a response-bound note identity when the captured transport exposes one."""

    direct = _gitlab_compare_act_note_text(event, "response_note_id", "note_id")
    if direct:
        return direct
    header_sources: list[Any] = [
        event.get("response_headers"),
        event.get("response_headers_extra"),
    ]
    response = event.get("response")
    if isinstance(response, Mapping):
        direct = _gitlab_compare_act_note_text(response, "note_id", "noteId", "id")
        if direct:
            return direct
        header_sources.append(response.get("headers"))
    for raw_headers in header_sources:
        if isinstance(raw_headers, Mapping):
            values = [
                value
                for key, value in raw_headers.items()
                if str(key).strip().casefold() == "location"
            ]
        elif isinstance(raw_headers, list):
            values = [
                item.get("value")
                for item in raw_headers
                if isinstance(item, Mapping)
                and str(item.get("name") or "").strip().casefold() == "location"
            ]
        else:
            continue
        for value in values:
            match = re.search(r"/notes/(?P<note_id>[^/?#]+)", str(value))
            if match:
                return unquote(match.group("note_id")).strip() or None
    return None


def _gitlab_compare_act_expected_actor(editor: Any, state_probe: Mapping[str, Any]) -> set[str]:
    explicit = [state_probe.get(key) for key in ("expected_actor", "actor_username", "actor_id")]
    values: list[Any] = [value for value in explicit if value not in (None, "")]
    if not values and state_probe.get("expected_actor_source") == "current_authenticated_user":
        try:
            current_user = editor._current_user()
        except Exception:
            current_user = None
        if isinstance(current_user, Mapping):
            values.extend(current_user.get(key) for key in ("username", "name", "id"))
    return _gitlab_compare_act_identity_values(values)


def _gitlab_compare_act_identity_values(values: list[Any]) -> set[str]:
    return {
        str(value).strip().casefold()
        for value in values
        if value not in (None, "") and not isinstance(value, bool) and str(value).strip()
    }


def _gitlab_compare_act_identity_equal(expected: str, observed: str) -> bool:
    return (
        unquote(str(expected)).strip().strip("/").casefold()
        == unquote(str(observed)).strip().strip("/").casefold()
    )


def _gitlab_compare_act_graphql_identity(value: str, kind: str) -> str | None:
    if not value or value == "<redacted>":
        return None
    if kind == "issue":
        pattern = r"(?:issue[_-]?iid|issueIid|issueId|iid)\s*[=:\"]+\s*[\"']?([A-Za-z0-9_.:-]+)"
    else:
        pattern = r"(?:fullPath|project[_-]?path|projectPath|projectId)\s*[=:\"]+\s*[\"']?([A-Za-z0-9_./%:-]+)"
    match = re.search(pattern, value, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _gitlab_compare_act_timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _gitlab_issue_note_readback_anchor(
    event: dict[str, Any],
    network_trace: list[dict[str, Any]],
    state_probe: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    project_key, issue_iid = _gitlab_issue_note_anchor_from_event(event)
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    probe_project = _first_probe_string(probe, "project_id", "project_path")
    probe_issue = _first_probe_string(probe, "issue_iid", "iid")
    project_key = probe_project or project_key
    issue_iid = probe_issue or issue_iid

    if not project_key or not issue_iid:
        inferred = _gitlab_issue_anchor_from_network_trace(network_trace)
        if inferred is not None:
            inferred_project, inferred_issue = inferred
            project_key = project_key or inferred_project
            issue_iid = issue_iid or inferred_issue

    return project_key, issue_iid


def _gitlab_issue_contains_witness(
    editor: Any,
    event: dict[str, Any],
    witness: str,
    network_trace: list[dict[str, Any]],
    instance: Mapping[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    probe_kind = _state_probe_kind(probe)
    if probe_kind and probe_kind != "issue_contains":
        return False, f"unsupported gitlab issue state_probe.kind {probe_kind!r}"

    candidate_keys: list[str] = []
    path = urlparse(_network_event_url(event)).path
    api_match = _GITLAB_CREATE_ISSUE_RE.search(path)
    if api_match:
        candidate_keys.append(api_match.group(1))
    readback_attempts = 1 if api_match else _GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS

    probe_project_id = _first_probe_string(probe, "project_id")
    probe_project_path = _first_probe_string(probe, "project_path")
    if probe_project_id:
        candidate_keys.append(probe_project_id)
    if probe_project_path:
        candidate_keys.append(probe_project_path)
    candidate_keys.extend(_gitlab_project_paths_from_network_trace(network_trace, instance))

    deduped = [key for key in dict.fromkeys(candidate_keys) if key]
    if not deduped:
        return False, "gitlab issue readback has no project anchor"

    last_reason = "gitlab issue final state did not contain witness"
    for project_key in deduped:
        api_project_key = _gitlab_api_project_key(editor, project_key)
        for attempt in range(readback_attempts):
            issues = editor._api_request_json(
                "GET",
                f"/api/v4/projects/{api_project_key}/issues",
                params={"state": "all", "per_page": 100},
            )
            filtered_issues = _exclude_gitlab_issue_readback_carriers(issues, probe)
            if _records_contain_witness(filtered_issues, witness, fields=("title", "description")):
                return True, "gitlab issue final state contains expected witness"
            last_reason = (
                f"gitlab issue final state did not contain witness for project {project_key!r}"
            )
            if attempt < readback_attempts - 1:
                time.sleep(_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS)
    return False, last_reason


def _exclude_gitlab_issue_readback_carriers(
    issues: Any,
    state_probe: Mapping[str, Any],
) -> Any:
    """Drop known carrier issues before create-issue readback witness checks."""

    if not isinstance(issues, list):
        return issues
    excluded = _gitlab_excluded_issue_iids(state_probe)
    if not excluded:
        return issues
    return [
        issue
        for issue in issues
        if not isinstance(issue, Mapping) or str(issue.get("iid") or "") not in excluded
    ]


def _gitlab_excluded_issue_iids(state_probe: Mapping[str, Any]) -> set[str]:
    excluded: list[str] = []
    for key in (
        "exclude_issue_iid",
        "carrier_issue_iid",
        "seed_issue_iid",
        "source_issue_iid",
    ):
        value = state_probe.get(key)
        if value not in (None, ""):
            excluded.append(str(value))
    raw_values = state_probe.get("exclude_issue_iids")
    if isinstance(raw_values, list):
        excluded.extend(str(value) for value in raw_values if value not in (None, ""))
    return {value.strip() for value in excluded if value.strip()}


def _gitlab_issue_note_anchor_from_event(
    event: dict[str, Any],
) -> tuple[str | None, str | None]:
    path = urlparse(_network_event_url(event)).path
    api_match = _GITLAB_CREATE_ISSUE_NOTE_RE.search(path)
    if api_match:
        project_key, issue_iid = api_match.groups()
        return project_key, issue_iid
    ui_project = _gitlab_project_path_from_note_ui_path(path)
    if ui_project:
        return ui_project, None
    return None, None


def _gitlab_project_path_from_note_ui_path(path: str) -> str | None:
    match = re.match(r"^/(?P<project_path>.+)/notes/?$", path)
    if not match:
        return None
    project_path = unquote(str(match.group("project_path") or "")).strip("/")
    return project_path or None


def _rank_gitlab_issue_note_source_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # The caller scans in reverse chronological order. Sorting low-to-high
    # keeps concrete note endpoints ahead of generic GraphQL after reversal.
    return sorted(events, key=_gitlab_issue_note_source_event_priority)


def _gitlab_issue_note_source_event_priority(event: dict[str, Any]) -> int:
    path = urlparse(_network_event_url(event)).path.rstrip("/")
    if _GITLAB_CREATE_ISSUE_NOTE_RE.search(path):
        return 3
    if _gitlab_project_path_from_note_ui_path(path):
        return 2
    if path == "/api/graphql":
        return 1
    return 0


def _gitlab_project_path_from_issue_create_ui_path(path: str) -> str | None:
    match = re.match(r"^/(?P<project_path>.+)/-/issues/?$", path)
    if not match:
        return None
    project_path = unquote(str(match.group("project_path") or "")).strip("/")
    return project_path or None


def _gitlab_issue_anchor_from_network_trace(
    network_trace: list[dict[str, Any]],
) -> tuple[str, str] | None:
    for event in reversed(network_trace):
        path = urlparse(_network_event_url(event)).path
        match = re.search(r"/(?P<project_path>.+)/-/issues/(?P<issue_iid>\d+)/?$", path)
        if not match:
            continue
        project_path = unquote(str(match.group("project_path") or "")).strip("/")
        issue_iid = str(match.group("issue_iid") or "").strip()
        if project_path and issue_iid:
            return project_path, issue_iid
    return None


def _gitlab_project_paths_from_network_trace(
    network_trace: list[dict[str, Any]],
    instance: Mapping[str, Any] | None = None,
) -> list[str]:
    paths: list[str] = []
    for event in reversed(network_trace):
        for event_url in _network_event_url_candidates(event, dict(instance or {})):
            path = urlparse(event_url).path
            issue_match = re.search(r"/(?P<project_path>.+)/-/issues(?:/\d+)?/?$", path)
            if issue_match:
                project_path = unquote(str(issue_match.group("project_path") or "")).strip("/")
                if project_path and project_path not in paths:
                    paths.append(project_path)
                continue
            create_path = _gitlab_project_path_from_issue_create_ui_path(path)
            if create_path and create_path not in paths:
                paths.append(create_path)
    return paths


def _gitlab_api_project_key(editor: Any, project_key: str) -> str:
    cleaned = str(project_key).strip("/")
    if cleaned.isdigit() or "%2F" in cleaned:
        return cleaned
    return editor._quote(cleaned)


def _gitlab_delete_project_id_from_event(editor: Any, path: str) -> str | None:
    api_match = _GITLAB_PROJECT_RE.search(path)
    if api_match:
        project_key = api_match.group(1)
        if project_key.isdigit():
            return project_key
        project_path = unquote(project_key).strip("/")
        if not project_path:
            return None
        project = editor._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{editor._quote(project_path)}",
            allow_missing=True,
        )
        if isinstance(project, Mapping) and project.get("id") not in (None, ""):
            return str(project["id"])
        return None
    return _gitlab_repo_project_id_from_event(editor, path)


def _gitlab_delete_project_key_from_event(path: str) -> str | None:
    api_match = _GITLAB_PROJECT_RE.search(path)
    if api_match:
        project_key = unquote(api_match.group(1)).strip("/")
        return project_key or None
    ui_match = _GITLAB_PROJECT_SETTINGS_RE.search(path)
    if ui_match:
        project_path = str(ui_match.group("project_path") or "").strip("/")
        return project_path or None
    return None


def _gitlab_repo_content_contains_witness(
    editor: Any,
    event: dict[str, Any],
    witness: str,
    instance: dict[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    path = urlparse(_network_event_url(event)).path
    project_id = _gitlab_repo_project_id_from_event(editor, path)
    if project_id is None:
        return False, "gitlab repository-content event did not identify a project"
    probe = state_probe if isinstance(state_probe, Mapping) else {}
    probe_kind = str(probe.get("kind") or "").strip()
    if probe_kind and probe_kind != "repo_file_contains":
        return False, f"unsupported gitlab repository state_probe.kind {probe_kind!r}"

    expected_project_id = _gitlab_expected_project_id_from_state_probe(editor, probe)
    if expected_project_id is not None and str(project_id) != expected_project_id:
        return (
            False,
            "gitlab repository-content event project did not match state_probe "
            f"project_id {expected_project_id}",
        )

    expected_branch = _first_probe_string(probe, "default_ref", "branch", "ref")
    branch = expected_branch or "main"
    file_paths: list[str] = []
    parsed = _parse_post_data(_network_event_post_text(event), _network_event_headers(event))
    parsed_branch = _first_mapping_string(
        parsed,
        "branch",
        "target_branch",
        "commit[branch]",
        "file[branch_name]",
    )
    if parsed_branch:
        if expected_branch and parsed_branch != expected_branch:
            return (
                False,
                "gitlab repository-content event branch did not match "
                f"state_probe branch {expected_branch!r}",
            )
        branch = parsed_branch
    file_paths.extend(_gitlab_repo_file_paths_from_post_data(parsed))
    ui_match = _GITLAB_REPO_UI_WRITE_RE.search(path)
    if ui_match:
        ui_branch = ui_match.group("branch") or branch
        if expected_branch and ui_branch != expected_branch:
            return (
                False,
                "gitlab repository-content UI branch did not match "
                f"state_probe branch {expected_branch!r}",
            )
        branch = ui_branch
        url_file_path = ui_match.group("file_path")
        if url_file_path:
            file_paths.append(url_file_path)

    expected_file_path = _first_probe_string(probe, "expected_file_path", "file_path", "path")
    expected_file_content = _first_probe_string(
        probe,
        "expected_file_content",
        "required_file_content",
        "content",
    )
    candidate_paths = [expected_file_path] if expected_file_path else _dedupe_strings(file_paths)
    for file_path in _dedupe_strings(candidate_paths):
        content = editor._gitlab_get_file_content(project_id, file_path=file_path, ref=branch)
        if _repo_file_content_matches(
            content,
            witness=witness,
            expected_file_content=expected_file_content,
        ):
            return True, "gitlab repository content final state contains expected witness"
    if expected_file_path:
        return (
            False,
            "gitlab repository content readback did not contain witness at "
            f"state_probe file {expected_file_path!r}",
        )

    for file_path in _gitlab_recent_repo_file_paths(editor, project_id, ref=branch):
        content = editor._gitlab_get_file_content(project_id, file_path=file_path, ref=branch)
        if _repo_file_content_matches(
            content,
            witness=witness,
            expected_file_content=expected_file_content,
        ):
            return True, "gitlab repository content final state contains expected witness"
    return False, "gitlab repository content readback did not contain witness"


def _repo_file_content_matches(
    content: Any,
    *,
    witness: str,
    expected_file_content: str | None,
) -> bool:
    if not isinstance(content, str):
        return False
    if expected_file_content:
        return content.strip() == expected_file_content.strip()
    return witness in content


def _gitlab_repo_project_id_from_event(editor: Any, path: str) -> str | None:
    api_match = _GITLAB_REPO_COMMITS_RE.search(path)
    if api_match:
        project_key = api_match.group(1)
        if project_key.isdigit():
            return project_key
        project_path = unquote(project_key).strip("/")
        if not project_path:
            return None
        project = editor._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{editor._quote(project_path)}",
            allow_missing=True,
        )
        if isinstance(project, Mapping) and project.get("id") not in (None, ""):
            return str(project["id"])
        return None
    ui_match = _GITLAB_REPO_UI_WRITE_RE.search(path)
    if not ui_match:
        return None
    project_path = str(ui_match.group("project_path") or "").strip("/")
    if not project_path:
        return None
    project = editor._gitlab_request_json(
        "GET",
        f"/api/v4/projects/{editor._quote(project_path)}",
        allow_missing=True,
    )
    if isinstance(project, Mapping) and project.get("id") not in (None, ""):
        return str(project["id"])
    return None


def _gitlab_expected_project_id_from_state_probe(
    editor: Any,
    state_probe: Mapping[str, Any],
) -> str | None:
    project_path = _first_probe_string(state_probe, "project_path")
    if project_path:
        project = editor._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{editor._quote(project_path)}",
            allow_missing=True,
        )
        if isinstance(project, Mapping) and project.get("id") not in (None, ""):
            return str(project["id"])
    explicit = _first_probe_string(state_probe, "project_id")
    if explicit:
        return explicit
    return None


def _first_probe_string(state_probe: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = state_probe.get(key)
        if value not in (None, ""):
            return str(value).strip().strip("/")
    return None


def _gitlab_repo_file_paths_from_post_data(parsed: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    actions = parsed.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if isinstance(action, Mapping):
                value = action.get("file_path") or action.get("path")
                if value not in (None, ""):
                    paths.append(str(value))
    for key, value in parsed.items():
        key_s = str(key)
        if key_s in {"file_path", "path", "file[path]", "file_path[]"} or re.search(
            r"actions(?:\[\d*\])?\[(?:file_path|path)\]$",
            key_s,
        ):
            paths.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return [path.strip().strip("/") for path in paths if isinstance(path, str) and path.strip()]


def _gitlab_recent_repo_file_paths(editor: Any, project_id: str, *, ref: str) -> list[str]:
    try:
        tree = editor._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{project_id}/repository/tree",
            params={"recursive": "true", "per_page": 100, "ref": ref},
            allow_missing=True,
        )
    except Exception:
        return []
    if not isinstance(tree, list):
        return []
    paths: list[str] = []
    for entry in tree:
        if not isinstance(entry, Mapping) or entry.get("type") != "blob":
            continue
        value = entry.get("path")
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    return paths[:100]


def _first_mapping_string(parsed: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = parsed.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value not in (None, ""):
            return str(value).strip()
    return None


def _matching_gitlab_issue_note_source_events(
    expected: dict[str, Any],
    network_trace: list[dict[str, Any]],
    instance: dict[str, Any],
    *,
    witness: str,
) -> tuple[list[dict[str, Any]], str]:
    """Match GitLab issue-comment source actions across UI transports.

    GitLab's browser UI commonly creates issue comments through GraphQL while
    older/self-hosted variants can submit through project ``/notes`` routes.
    Generic GraphQL traffic is also emitted for ordinary page queries, so it is
    only accepted when the request body proves the witness-bearing mutation.
    """

    expected_methods = _expected_http_methods(expected.get("http_method", "POST"))
    if "POST" not in expected_methods:
        return [], "gitlab issue-note source requires POST"

    matched: list[dict[str, Any]] = []
    for event in network_trace:
        if _network_event_method(event) != "POST":
            continue
        if not _is_gitlab_issue_note_source_event(event, expected, instance, witness=witness):
            continue
        status = _network_event_status(event)
        if "response_status" in expected and not _status_matches(
            status, expected.get("response_status")
        ):
            continue
        matched.append(event)
    if not matched:
        return [], "no gitlab issue-note UI or GraphQL source event found"
    return matched, f"found {len(matched)} gitlab issue-note source event(s)"


def _matching_gitlab_issue_source_events(
    expected: dict[str, Any],
    network_trace: list[dict[str, Any]],
    instance: dict[str, Any],
    *,
    witness: str,
) -> tuple[list[dict[str, Any]], str]:
    """Match GitLab public issue-creation source actions across UI transports.

    GitLab's browser UI commonly creates issues through GraphQL. Generic
    GraphQL traffic is also emitted for ordinary issue-list/detail page loads,
    so issue creation requires witness-bearing request body evidence.
    """

    expected_methods = _expected_http_methods(expected.get("http_method", "POST"))
    if "POST" not in expected_methods:
        return [], "gitlab issue source requires POST"

    matched: list[dict[str, Any]] = []
    for event in network_trace:
        if _network_event_method(event) != "POST":
            continue
        if not _is_gitlab_issue_source_event(event, expected, instance, witness=witness):
            continue
        status = _network_event_status(event)
        if "response_status" in expected and not _status_matches(
            status, expected.get("response_status")
        ):
            continue
        matched.append(event)
    if not matched:
        return [], "no gitlab issue UI or GraphQL source event found"
    return matched, f"found {len(matched)} gitlab issue source event(s)"


def _is_gitlab_issue_source_event(
    event: dict[str, Any],
    expected: dict[str, Any],
    instance: dict[str, Any],
    *,
    witness: str,
) -> bool:
    source_expected = {**expected, "contains": witness}
    for event_url in _network_event_url_candidates(event, instance):
        path = urlparse(event_url).path.rstrip("/")
        if path == "/api/graphql":
            return _expected_network_event_allows_url(
                expected, event_url, instance
            ) and _gitlab_event_body_proves_mutation("create_issue", source_expected, event)
        if _GITLAB_CREATE_ISSUE_RE.search(path):
            return _expected_network_event_allows_url(
                expected, event_url, instance
            ) and _gitlab_event_body_proves_mutation("create_issue", source_expected, event)
        if _gitlab_project_path_from_issue_create_ui_path(path):
            return _expected_network_event_allows_url(
                expected, event_url, instance
            ) and _gitlab_event_body_proves_mutation("create_issue", source_expected, event)
    return False


def _is_gitlab_issue_note_source_event(
    event: dict[str, Any],
    expected: dict[str, Any],
    instance: dict[str, Any],
    *,
    witness: str,
) -> bool:
    source_expected = {**expected, "contains": witness}
    for event_url in _network_event_url_candidates(event, instance):
        path = urlparse(event_url).path.rstrip("/")
        if path == "/api/graphql":
            return _expected_network_event_allows_url(
                expected, event_url, instance
            ) and _gitlab_event_body_proves_mutation("create_issue_note", source_expected, event)
        if _GITLAB_CREATE_ISSUE_NOTE_RE.search(path):
            return _expected_network_event_allows_url(
                expected, event_url, instance
            ) and _gitlab_event_body_proves_mutation("create_issue_note", source_expected, event)
        if _gitlab_project_path_from_note_ui_path(path):
            return _expected_network_event_allows_url(
                expected, event_url, instance
            ) and _gitlab_ui_note_event_body_is_acceptable(source_expected, event)
    return False


def _gitlab_filter_valid_mutation_source_events(
    *,
    action_kind: str,
    witness: str,
    expected: Mapping[str, Any],
    events: list[dict[str, Any]],
    instance: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    if action_kind not in {"create_issue", "create_issue_note"}:
        return events, "not a GitLab issue mutation action"
    valid = [
        event
        for event in events
        if _gitlab_mutation_source_event_valid(
            action_kind=action_kind,
            witness=witness,
            expected=expected,
            event=event,
            instance=instance,
        )
    ]
    if valid:
        return valid, f"found {len(valid)} valid gitlab {action_kind} source event(s)"
    return [], (f"no gitlab {action_kind} source event with witness-bearing mutation body")


def _gitlab_mutation_source_event_valid(
    *,
    action_kind: str,
    witness: str,
    expected: Mapping[str, Any],
    event: dict[str, Any],
    instance: dict[str, Any],
) -> bool:
    if _network_event_method(event) != "POST":
        return False
    source_expected = {**dict(expected), "contains": witness}
    for event_url in _network_event_url_candidates(event, instance):
        path = urlparse(event_url).path.rstrip("/")
        if action_kind == "create_issue":
            if path == "/api/graphql" or _GITLAB_CREATE_ISSUE_RE.search(path):
                return _expected_network_event_allows_url(
                    dict(expected), event_url, instance
                ) and _gitlab_event_body_proves_mutation(action_kind, source_expected, event)
            if _gitlab_project_path_from_issue_create_ui_path(path):
                return _expected_network_event_allows_url(
                    dict(expected), event_url, instance
                ) and _gitlab_event_body_proves_mutation(action_kind, source_expected, event)
        if action_kind == "create_issue_note":
            if path == "/api/graphql" or _GITLAB_CREATE_ISSUE_NOTE_RE.search(path):
                return _expected_network_event_allows_url(
                    dict(expected), event_url, instance
                ) and _gitlab_event_body_proves_mutation(action_kind, source_expected, event)
            if _gitlab_project_path_from_note_ui_path(path):
                return _expected_network_event_allows_url(
                    dict(expected), event_url, instance
                ) and _gitlab_ui_note_event_body_is_acceptable(source_expected, event)
    return False


def _gitlab_event_body_proves_mutation(
    action_kind: str,
    expected: Mapping[str, Any],
    event: dict[str, Any],
) -> bool:
    witness = str(expected.get("contains") or "").strip()
    if not witness:
        return False
    post_text = _network_event_post_text(event)
    if not post_text or post_text == "<redacted>":
        return False
    decoded = unquote_plus(post_text)
    if witness not in post_text and witness not in decoded:
        return False
    if urlparse(_network_event_url(event)).path.rstrip("/") == "/api/graphql":
        lowered = decoded.lower()
        if "mutation" not in lowered:
            return False
        if action_kind == "create_issue":
            return "issue" in lowered and "create" in lowered
        if action_kind == "create_issue_note":
            return ("note" in lowered or "comment" in lowered) and "create" in lowered
    return True


def _gitlab_ui_note_event_body_is_acceptable(
    expected: Mapping[str, Any],
    event: dict[str, Any],
) -> bool:
    post_text = _network_event_post_text(event)
    if not post_text or post_text == "<redacted>":
        return True
    return _gitlab_event_body_proves_mutation("create_issue_note", expected, event)
