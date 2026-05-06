from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlparse

import requests

from worldsim.rewards.network_event import _expected_network_event_allows_url
from worldsim.rewards.network_trace import (
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
from worldsim.rewards.shared import _dedupe_strings, _records_contain_witness, _state_probe_kind

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


def _eval_gitlab_final_state(
    action_kind: str,
    witness: str,
    events: list[dict[str, Any]],
    network_trace: list[dict[str, Any]],
    instance: dict[str, Any],
    state_probe: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    from worldsim.editors.gitlab import GitlabEditor

    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        last_reason: str | None = None
        # A single UI action can emit GraphQL and REST source events for the
        # same issue note. Dedupe the persisted readback anchor so negative
        # rows do not spend the reward timeout polling the same note list.
        seen_issue_note_readbacks: set[tuple[str | None, str | None]] = set()
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
    for attempt in range(_GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS):
        notes = editor._api_request_json(
            "GET",
            notes_path,
            params={"per_page": 100},
        )
        if _records_contain_witness(notes, witness, fields=("body",)):
            return True, "gitlab issue note final state contains expected witness"
        if attempt < _GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS - 1:
            time.sleep(_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS)
    return False, "gitlab issue note final state did not contain witness"


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
            if _records_contain_witness(issues, witness, fields=("title", "description")):
                return True, "gitlab issue final state contains expected witness"
            last_reason = (
                f"gitlab issue final state did not contain witness for project {project_key!r}"
            )
            if attempt < readback_attempts - 1:
                time.sleep(_GITLAB_ISSUE_NOTE_READBACK_SLEEP_SECONDS)
    return False, last_reason


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
) -> tuple[list[dict[str, Any]], str]:
    """Match GitLab issue-comment source actions across UI transports.

    GitLab's browser UI commonly creates issue comments through GraphQL while
    older/self-hosted variants can submit through project ``/notes`` routes.
    Request bodies may be redacted by browser-use, so the source event proves
    same-site write intent and the follow-up state probe proves the exact
    unique witness text.
    """

    expected_methods = _expected_http_methods(expected.get("http_method", "POST"))
    if "POST" not in expected_methods:
        return [], "gitlab issue-note source requires POST"

    matched: list[dict[str, Any]] = []
    for event in network_trace:
        if _network_event_method(event) != "POST":
            continue
        if not _is_gitlab_issue_note_source_event(event, expected, instance):
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
) -> tuple[list[dict[str, Any]], str]:
    """Match GitLab public issue-creation source actions across UI transports.

    GitLab's browser UI commonly creates issues through GraphQL, and Browser
    Use may redact that request body. The source event proves same-site write
    intent; the final-state probe proves the exact unique witness text.
    """

    expected_methods = _expected_http_methods(expected.get("http_method", "POST"))
    if "POST" not in expected_methods:
        return [], "gitlab issue source requires POST"

    matched: list[dict[str, Any]] = []
    for event in network_trace:
        if _network_event_method(event) != "POST":
            continue
        if not _is_gitlab_issue_source_event(event, expected, instance):
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
) -> bool:
    for event_url in _network_event_url_candidates(event, instance):
        path = urlparse(event_url).path.rstrip("/")
        if path == "/api/graphql":
            return _expected_network_event_allows_url(expected, event_url, instance)
        if _gitlab_project_path_from_issue_create_ui_path(path):
            return _expected_network_event_allows_url(expected, event_url, instance)
    return False


def _is_gitlab_issue_note_source_event(
    event: dict[str, Any],
    expected: dict[str, Any],
    instance: dict[str, Any],
) -> bool:
    for event_url in _network_event_url_candidates(event, instance):
        path = urlparse(event_url).path.rstrip("/")
        if path == "/api/graphql":
            return _expected_network_event_allows_url(expected, event_url, instance)
        if _gitlab_project_path_from_note_ui_path(path):
            return _expected_network_event_allows_url(expected, event_url, instance)
    return False
