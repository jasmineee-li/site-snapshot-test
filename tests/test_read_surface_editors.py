"""Unit tests for the editor read-surface contract (Commit 2 of C1 migration).

Each editor method returns a dict containing ``read_surface_urls`` (both
host-qualified and path-only forms) + ``read_surface_provenance_source``.
The accumulator in ``seeding.apply_data_seed`` threads the URLs onto the
Phase 4 task via ``(handle, metadata)`` tuple return.
"""

from __future__ import annotations

import types
from typing import Any, ClassVar

from worldsim import seeding
from worldsim.editors._read_surface import (
    collect_platform_urls,
    host_and_path_forms,
    normalize_surface_urls,
)
from worldsim.editors.gitlab import GitlabEditor
from worldsim.editors.reddit import RedditEditor


def test_collect_platform_urls_extracts_nested_paths():
    response = {
        "web_url": "http://gitlab.test/owner/repo/-/issues/42",
        "_links": {"self": "http://gitlab.test/api/v4/projects/174/issues/42"},
    }
    urls = collect_platform_urls(response, ["web_url", "_links.self"])
    assert urls == [
        "http://gitlab.test/owner/repo/-/issues/42",
        "http://gitlab.test/api/v4/projects/174/issues/42",
    ]


def test_collect_platform_urls_skips_non_strings():
    response = {"web_url": None, "_links": {"self": 123}}
    assert collect_platform_urls(response, ["web_url", "_links.self"]) == []


def test_host_and_path_forms_emits_both_for_full_url():
    forms = host_and_path_forms("http://gitlab.test", "http://gitlab.test/owner/-/issues/42?utm=1")
    assert forms == [
        "http://gitlab.test/owner/-/issues/42?utm=1",
        "/owner/-/issues/42?utm=1",
    ]


def test_host_and_path_forms_path_only_attaches_host():
    forms = host_and_path_forms("http://reddit.test", "/f/books/59421")
    assert forms == ["http://reddit.test/f/books/59421", "/f/books/59421"]


def test_host_and_path_forms_no_site_url_still_yields_path():
    forms = host_and_path_forms("", "/path")
    assert forms == ["/path"]


def test_normalize_surface_urls_dedupes_preserving_order():
    assert normalize_surface_urls(
        ["/a", "/b", "/a", "", None, "/c"]  # type: ignore[list-item]
    ) == ["/a", "/b", "/c"]


# ---------------------------------------------------------------------------
# GitLab editor emits surface URLs for issues + MRs (the primary C1 targets)
# ---------------------------------------------------------------------------


def _gitlab_editor(instance: dict | None = None) -> GitlabEditor:
    return GitlabEditor(
        instance
        or {
            "site_url": "http://gitlab.test",
            "agent_auth": {"authentication": {"credentials": {"username": "byteblaze"}}},
        },
        session=None,  # type: ignore[arg-type]
    )


def test_gitlab_create_issue_emits_surface_urls_from_web_url(monkeypatch):
    editor = _gitlab_editor()
    monkeypatch.setattr(
        editor,
        "_ensure_project",
        lambda **kwargs: {
            "project_id": 174,
            "project_path": "byteblaze/secret-project",
            "default_branch": "main",
        },
    )
    monkeypatch.setattr(editor, "_find_existing_issue", lambda **kwargs: None)
    monkeypatch.setattr(
        editor,
        "_gitlab_request_json",
        lambda method, path, **kwargs: {
            "iid": 42,
            "web_url": "http://gitlab.test/byteblaze/secret-project/-/issues/42",
        },
    )

    result = editor.create_issue(title_template="Seeded", body_template="body")

    assert result["issue_iid"] == 42
    assert result["read_surface_provenance_source"] == "editor_api_response"
    # Both host-qualified and path-only forms for each URL the helper sees.
    urls = result["read_surface_urls"]
    assert "http://gitlab.test/byteblaze/secret-project/-/issues/42" in urls
    assert "/byteblaze/secret-project/-/issues/42" in urls


def test_gitlab_create_issue_falls_back_to_constructed_path_when_web_url_missing(monkeypatch):
    editor = _gitlab_editor()
    monkeypatch.setattr(
        editor,
        "_ensure_project",
        lambda **kwargs: {
            "project_id": 174,
            "project_path": "byteblaze/secret-project",
            "default_branch": "main",
        },
    )
    monkeypatch.setattr(editor, "_find_existing_issue", lambda **kwargs: None)
    monkeypatch.setattr(
        editor,
        "_gitlab_request_json",
        lambda method, path, **kwargs: {"iid": 42},  # no web_url
    )

    result = editor.create_issue(title_template="Seeded", body_template="body")

    urls = result["read_surface_urls"]
    # _issue_surface_urls emits the HTML page + the /discussions.json
    # plaintext source (worldsim/editors/gitlab.py:1417-1428) — the
    # HTML alone only exposes ~8 KB while notes lazy-load, so the JSON
    # feed is what Phase 2c's render verifier can dependably grep.
    assert urls == [
        "http://gitlab.test/byteblaze/secret-project/-/issues/42",
        "/byteblaze/secret-project/-/issues/42",
        "http://gitlab.test/byteblaze/secret-project/-/issues/42/discussions.json",
        "/byteblaze/secret-project/-/issues/42/discussions.json",
    ]


# ---------------------------------------------------------------------------
# Reddit editor emits surface URLs for submissions + comments
# ---------------------------------------------------------------------------


def _reddit_editor() -> RedditEditor:
    return RedditEditor(
        {
            "site_url": "http://reddit.test",
            "agent_auth": {"credentials": {"username": "user"}},
        },
        session=types.SimpleNamespace(),  # type: ignore[arg-type]
    )


def test_reddit_create_submission_emits_host_and_path_forms(monkeypatch):
    editor = _reddit_editor()
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *a, **k: {
            "action": "",
            "fields": {
                "submission[_token]": "csrf",
                "submission[forum]": "",
                "submission[title]": "",
                "submission[body]": "",
            },
            "select_options": {
                "submission[forum]": [{"data-name": "books", "value": "7"}],
            },
        },
    )
    monkeypatch.setattr(
        editor,
        "_submit_exact_form",
        lambda *a, **k: types.SimpleNamespace(
            headers={"Location": "/f/books/59421"}, json=lambda: {}
        ),
    )

    result = editor.create_submission(forum_name="books", title_template="Thread")

    assert result["read_surface_urls"] == [
        "http://reddit.test/f/books/59421",
        "/f/books/59421",
    ]
    assert result["read_surface_provenance_source"] == "editor_constructed"


# Shopping editor tests removed 2026-04-21 with the WASP-aligned scoping
# decision (see docs/handoffs/wasp-aligned-scoping-decision.md).


# ---------------------------------------------------------------------------
# apply_data_seed tuple return + multi-editor-call accumulation (§12.9)
# ---------------------------------------------------------------------------


class _RecordingFakeEditor:
    """Two-method fake that demonstrates multi-call accumulation."""

    site_name = "reddit"
    supported_methods = frozenset({"create_submission", "create_comment"})
    instances: ClassVar[list[_RecordingFakeEditor]] = []

    def __init__(self, instance: dict[str, Any], session: Any) -> None:
        self.instance = instance
        self.session = session
        self.cleaned = False
        _RecordingFakeEditor.instances.append(self)

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        return None

    def preview_context(self, method_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {}

    def create_submission(self, *, forum_name: str, title_template: str) -> dict[str, Any]:
        return {
            "forum_name": forum_name,
            "submission_id": "9999",
            "read_surface_urls": [
                f"http://fake.test/f/{forum_name}/9999",
                f"/f/{forum_name}/9999",
            ],
            "read_surface_provenance_source": "editor_constructed",
        }

    def create_comment(self, *, forum_name: str, submission_id: str, body: str) -> dict[str, Any]:
        return {
            "forum_name": forum_name,
            "submission_id": submission_id,
            "comment_id": "c1",
            "read_surface_urls": [
                f"http://fake.test/f/{forum_name}/{submission_id}",
                f"/f/{forum_name}/{submission_id}",
                # add a DIFFERENT URL to prove we don't clobber
                f"http://fake.test/f/{forum_name}/{submission_id}#comment-c1",
            ],
            "read_surface_provenance_source": "editor_constructed",
        }

    def cleanup(self) -> None:
        self.cleaned = True


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_apply_data_seed_returns_tuple_with_metadata(monkeypatch):
    _RecordingFakeEditor.instances.clear()
    fake_session = _FakeSession()
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        seeding.EDITOR_REGISTRY,
        ("webarena_verified", "reddit"),
        _RecordingFakeEditor,
    )

    handle, metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert handle is not None
    assert metadata["read_surface_urls"] == [
        "http://fake.test/f/books/9999",
        "/f/books/9999",
    ]
    assert metadata["read_surface_provenance"]["source"] == "editor_constructed"
    # §12.9: editor_method is a list so multi-call seeds can attribute each
    # contribution; single-call seeds land as a one-element list.
    assert metadata["read_surface_provenance"]["editor_method"] == ["reddit.create_submission"]
    assert "captured_at" in metadata["read_surface_provenance"]
    handle.cleanup()


def test_apply_data_seed_accumulates_across_multi_editor_calls(monkeypatch):
    """§12.9 — two editor calls in one seed must both contribute URLs.

    This is correct-by-construction today even though no shipped task exercises
    it (verified 2026-04-18: 0/174 adversarial_tasks have len(editor_calls) >= 2).
    """
    _RecordingFakeEditor.instances.clear()
    fake_session = _FakeSession()
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        seeding.EDITOR_REGISTRY,
        ("webarena_verified", "reddit"),
        _RecordingFakeEditor,
    )

    _handle, metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                },
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "{forum_name}",
                        "submission_id": "{submission_id}",
                        "body": "payload",
                    },
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    urls = metadata["read_surface_urls"]
    # Submission surface comes first; comment surface second.
    # Duplicates deduped preserving first-occurrence order.
    assert "http://fake.test/f/books/9999" in urls
    assert "/f/books/9999" in urls
    # Anchored comment URL is distinct and preserved.
    assert "http://fake.test/f/books/9999#comment-c1" in urls
    # §12.9: both contributing methods recorded in first-occurrence order.
    assert metadata["read_surface_provenance"]["editor_method"] == [
        "reddit.create_submission",
        "reddit.create_comment",
    ]
    _handle.cleanup()


def test_apply_data_seed_returns_tuple_even_without_editor_calls():
    # state_push mechanism has no editors.
    import requests

    class _FakeResp:
        def raise_for_status(self):
            return None

    saved = requests.put

    def fake_put(*args, **kwargs):
        return _FakeResp()

    requests.put = fake_put  # type: ignore[assignment]
    try:
        handle, metadata = seeding.apply_data_seed(
            {"mechanism": "state_push", "state": {}},
            {"site_url": "http://site.test"},
        )
        assert handle is None
        assert metadata == {}
    finally:
        requests.put = saved  # type: ignore[assignment]


def test_apply_data_seed_surface_keys_do_not_bleed_into_seed_context(monkeypatch):
    """The C1b ``read_surface_urls`` list must not flow through seed_context
    as a placeholder source for subsequent calls (§5.4 design invariant)."""

    captured_contexts: list[dict[str, Any]] = []

    class _ContextWatchingEditor(_RecordingFakeEditor):
        pass

    _ContextWatchingEditor.instances = []
    fake_session = _FakeSession()
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        seeding.EDITOR_REGISTRY,
        ("webarena_verified", "reddit"),
        _ContextWatchingEditor,
    )

    original_render = seeding._render_editor_seed_call

    def capturing_render(call: dict[str, Any], seed_context: dict[str, Any]) -> dict[str, Any]:
        captured_contexts.append(dict(seed_context))
        return original_render(call, seed_context)

    monkeypatch.setattr(seeding, "_render_editor_seed_call", capturing_render)

    seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                },
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "{forum_name}",
                        "submission_id": "{submission_id}",
                        "body": "payload",
                    },
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    # The second call's seed_context must NOT contain read_surface_urls /
    # read_surface_provenance_source — those are C1b-only.
    assert len(captured_contexts) == 2
    second_context = captured_contexts[1]
    assert "read_surface_urls" not in second_context
    assert "read_surface_provenance_source" not in second_context


# Shopping admin editor test removed 2026-04-21 with the WASP-aligned
# scoping decision (see docs/handoffs/wasp-aligned-scoping-decision.md).
