from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest
import requests

from warp_taskgen.editors.base import BaseSiteEditor, EditorError
from warp_taskgen.sites.classifieds_editor import (
    CLASSIFIEDS_FIXED_TITLE,
    ClassifiedsEditor,
    ClassifiedsListingReplyForm,
    build_listing_reply_form,
    extract_listing_reply_id,
)

ORIGIN = "https://classifieds.test"
LISTING_URL = f"{ORIGIN}/index.php?page=item&id=17"
BODY = "Please read UNIQUE-SIGNATURE"
ACTOR = "alice"


def _form_html(
    *,
    action: str = "/index.php",
    csrf: str = "csrf-17",
    listing_id: str = "17",
    include_rating: bool = False,
) -> str:
    token = f'<input type="hidden" name="octoken" value="{csrf}">' if csrf else ""
    rating = '<input type="hidden" name="rating" value="">' if include_rating else ""
    return f"""
    <form action="{action}" method="post">
      {token}
      <input type="hidden" name="action" value="add_comment">
      <input type="hidden" name="page" value="item">
      <input type="hidden" name="id" value="{listing_id}">
      <input type="hidden" name="replyId" value="">
      <input type="hidden" name="authorName" value="alice">
      <input type="hidden" name="authorEmail" value="alice@example.test">
      <input type="text" name="title" value="">
      {rating}
      <textarea name="body"></textarea>
    </form>
    """


def _comment_html(*, reply_id: str = "88", actor: str = ACTOR, body: str = BODY) -> str:
    return f"""
    <div class="comment">
      <h3><strong>Additional listing details</strong> <em>by {actor}:</em></h3>
      <p>{body}</p>
      <p class="comment-reply-row">
        <a class="comment-reply" data-id="{reply_id}">Reply</a>
      </p>
    </div>
    """


class _Response:
    def __init__(
        self,
        *,
        status_code: int,
        text: str = "",
        headers: Mapping[str, str] | None = None,
        url: str = "",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = dict(headers or {})
        self.url = url

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, *, form_html: str, rendered_html: str, post_response: _Response) -> None:
        self._get_responses = [
            _Response(status_code=200, text=form_html, url=LISTING_URL),
            _Response(status_code=200, text=rendered_html, url=LISTING_URL),
        ]
        self._post_response = post_response
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(("GET", url, kwargs))
        if not self._get_responses:
            raise AssertionError("unexpected extra GET")
        return self._get_responses.pop(0)

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls.append((method, url, kwargs))
        assert method == "POST"
        return self._post_response


def _editor(
    *,
    form_html: str | None = None,
    rendered_html: str | None = None,
    post_response: _Response | None = None,
) -> tuple[ClassifiedsEditor, _FakeSession]:
    session = _FakeSession(
        form_html=form_html or _form_html(),
        rendered_html=rendered_html or _comment_html(),
        post_response=post_response
        or _Response(
            status_code=302,
            headers={"Location": "/index.php?page=item&id=17"},
            url=LISTING_URL,
        ),
    )
    return ClassifiedsEditor({"site_url": ORIGIN}, session), session


def test_listing_reply_form_is_regular_participant_and_body_only() -> None:
    form = ClassifiedsListingReplyForm(
        listing_id="17",
        csrf_token="csrf-17",
        body=BODY,
        signature="UNIQUE-SIGNATURE",
        author_name=ACTOR,
        author_email="alice@example.test",
    )

    assert form.method == "create_listing_reply"
    assert form.action == LISTING_URL.removeprefix(ORIGIN)
    assert dict(form.fields) == {
        "action": "add_comment",
        "page": "item",
        "id": "17",
        "replyId": "",
        "authorName": "alice",
        "authorEmail": "alice@example.test",
        "title": CLASSIFIEDS_FIXED_TITLE,
        "rating": "0",
        "body": BODY,
        "octoken": "csrf-17",
    }
    assert "description" not in form.fields


def test_form_builder_rejects_missing_csrf_fixed_id_signature_or_actor() -> None:
    kwargs = {
        "listing_id": "17",
        "csrf_token": "csrf",
        "body": BODY,
        "signature": "UNIQUE-SIGNATURE",
        "author_name": ACTOR,
        "author_email": "alice@example.test",
    }
    with pytest.raises(ValueError, match="CSRF"):
        build_listing_reply_form(**{**kwargs, "csrf_token": ""})
    with pytest.raises(ValueError, match="listing"):
        build_listing_reply_form(**{**kwargs, "listing_id": "17/18"})
    with pytest.raises(ValueError, match="signature"):
        build_listing_reply_form(**{**kwargs, "signature": "MARKER"})
    with pytest.raises(ValueError, match="author name"):
        build_listing_reply_form(**{**kwargs, "author_name": ""})


def test_editor_subclasses_base_editor_and_exposes_local_listing_contract() -> None:
    editor = ClassifiedsEditor()

    assert isinstance(editor, BaseSiteEditor)
    assert editor.site_name == "classifieds"
    assert editor.benchmark == "visualwebarena"
    assert editor.supported_methods == frozenset({"create_listing_reply"})
    spec = editor.create_listing_reply._editor_method_spec
    assert spec["http"] == ("POST", "/index.php?page=item&id={listing_id}")
    assert spec["kinds"] == frozenset({"listing"})
    assert spec["surface_id_per_kind"] == {"listing": "listing_reply.body"}
    assert spec["required_editor_args"] == ("listing_id", "body")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            _Response(
                status_code=200,
                text='<nav><a href="/index.php?page=logout">Log out</a></nav>',
                url=f"{ORIGIN}/index.php?page=user&action=dashboard",
            ),
            True,
        ),
        (
            _Response(
                status_code=200,
                text='<nav><a href="/index.php?page=logout">Log out</a></nav>',
                url=f"{ORIGIN}/index.php?page=user&action=items",
            ),
            True,
        ),
        (
            _Response(
                status_code=302,
                headers={"Location": "/index.php?page=login"},
                url=f"{ORIGIN}/index.php?page=user&action=dashboard",
            ),
            False,
        ),
        (
            _Response(
                status_code=200,
                text='<form action="/index.php?page=login"><input name="password"></form>',
                url=f"{ORIGIN}/index.php?page=login",
            ),
            False,
        ),
    ],
)
def test_auth_probe_requires_dashboard_logout_evidence(
    response: _Response,
    expected: bool,
) -> None:
    class _ProbeSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get(self, url: str, **kwargs: object) -> _Response:
            self.calls.append((url, kwargs))
            return response

    session = _ProbeSession()
    editor = ClassifiedsEditor({"site_url": ORIGIN}, session)

    assert editor.probe_authenticated() is expected
    assert session.calls == [
        (
            f"{ORIGIN}/index.php?page=user&action=dashboard",
            {"timeout": 10, "allow_redirects": True},
        )
    ]


def test_reply_id_comes_from_exact_real_comment_data_id() -> None:
    assert (
        extract_listing_reply_id(
            _comment_html(),
            actor=ACTOR,
            body=BODY,
            signature="UNIQUE-SIGNATURE",
            listing_id="17",
        )
        == "88"
    )


def test_reply_id_accepts_matching_delete_identity_without_latest_fallback() -> None:
    html = """
    <div class="comment">
      <h3>Title by alice</h3><p>Please read UNIQUE-SIGNATURE</p>
      <a class="comment-reply" data-id="88">Reply</a>
      <a href="/index.php?page=item&amp;action=delete_comment&amp;id=17&amp;comment=88&amp;octoken=x">Delete</a>
    </div>
    """
    assert (
        extract_listing_reply_id(
            html,
            actor=ACTOR,
            body=BODY,
            signature="UNIQUE-SIGNATURE",
            listing_id="17",
        )
        == "88"
    )


def test_nested_comment_does_not_cross_contaminate_outer_identity() -> None:
    html = """
    <div class="comment">
      <h3>Title by alice:</h3><p>Please read UNIQUE-SIGNATURE</p>
      <div class="comment reply">
        <h3>Nested by alice:</h3><p>Please read UNIQUE-SIGNATURE</p>
        <a class="comment-reply" data-id="99">Reply</a>
      </div>
      <a class="comment-reply" data-id="88">Reply</a>
    </div>
    """
    assert (
        extract_listing_reply_id(
            html,
            actor=ACTOR,
            body=BODY,
            signature="UNIQUE-SIGNATURE",
        )
        == "88"
    )


@pytest.mark.parametrize(
    "html",
    [
        "<div class='comment-reply' data-id='88'><p>UNIQUE-SIGNATURE</p></div>",
        "<div class='comment'><h3>Title by mallory</h3><p>UNIQUE-SIGNATURE</p>"
        "<a class='comment-reply' data-id='88'>Reply</a></div>",
        "<div class='comment'><h3>Title by alice</h3><p>UNIQUE-SIGNATURE</p></div>",
        "<div class='comment'><h3>Title by alice</h3><p>UNIQUE-SIGNATURE</p>"
        "<a class='comment-reply' data-id='88'>Reply</a></div>"
        "<div class='comment'><h3>Other by alice</h3><p>UNIQUE-SIGNATURE</p>"
        "<a class='comment-reply' data-id='89'>Reply</a></div>",
        "<div class='latest-comment' data-id='88'><p>UNIQUE-SIGNATURE</p></div>",
    ],
)
def test_reply_id_extraction_fails_closed_without_one_exact_rendered_identity(html: str) -> None:
    assert (
        extract_listing_reply_id(
            html,
            actor=ACTOR,
            body=BODY,
            signature="UNIQUE-SIGNATURE",
        )
        is None
    )


def test_editor_records_exact_get_post_get_and_returns_identity_tokens() -> None:
    editor, session = _editor()

    result = editor.create_listing_reply(
        listing_id="17",
        body=BODY,
        signature="UNIQUE-SIGNATURE",
    )

    assert [call[0] for call in session.calls] == ["GET", "POST", "GET"]
    assert session.calls[0][1] == LISTING_URL
    assert session.calls[2][1] == LISTING_URL
    method, post_url, kwargs = session.calls[1]
    assert method == "POST"
    assert post_url == f"{ORIGIN}/index.php"
    assert kwargs["data"] == {
        "action": "add_comment",
        "page": "item",
        "id": "17",
        "replyId": "",
        "authorName": ACTOR,
        "authorEmail": "alice@example.test",
        "title": CLASSIFIEDS_FIXED_TITLE,
        "rating": "0",
        "body": BODY,
        "octoken": "csrf-17",
    }
    assert result["identity_tokens"] == {
        "listing_id": "17",
        "reply_id": "88",
        "actor_name": ACTOR,
        "reply_body_sha256": hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
    }
    assert result["read_surface_urls"] == [LISTING_URL]


def test_editor_cleanup_deletes_exact_writer_reply_and_proves_absence() -> None:
    class _CleanupSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__(
                form_html=_form_html(),
                rendered_html=_comment_html(),
                post_response=_Response(
                    status_code=302,
                    headers={"Location": "/index.php?page=item&id=17"},
                    url=LISTING_URL,
                ),
            )

        def get(self, url: str, **kwargs: object) -> _Response:
            params = kwargs.get("params")
            if isinstance(params, dict) and params.get("action") == "delete_comment":
                self.calls.append(("GET", url, kwargs))
                return _Response(
                    status_code=302,
                    headers={"Location": "/index.php?page=item&id=17"},
                    url=url,
                )
            if len(self.calls) >= 4:
                self.calls.append(("GET", url, kwargs))
                return _Response(status_code=200, text=_form_html(), url=LISTING_URL)
            return super().get(url, **kwargs)

    session = _CleanupSession()
    editor = ClassifiedsEditor({"site_url": ORIGIN}, session)
    editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    editor.cleanup()

    delete = session.calls[3]
    assert delete[1] == f"{ORIGIN}/index.php"
    assert delete[2]["params"] == {
        "page": "item",
        "action": "delete_comment",
        "id": "17",
        "comment": "88",
        "octoken": "csrf-17",
    }
    assert [call[0] for call in session.calls] == ["GET", "POST", "GET", "GET", "GET"]


@pytest.mark.parametrize(
    "witness_html",
    [
        "<main>temporary soft error</main>",
        "<form action='/index.php?page=login' method='post'><input name='email'></form>",
        "<form action='/index.php' method='post'><input name='id' value='17'></form>",
    ],
)
def test_editor_cleanup_rejects_non_listing_success_pages(witness_html: str) -> None:
    class _SoftErrorCleanupSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__(
                form_html=_form_html(),
                rendered_html=_comment_html(),
                post_response=_Response(
                    status_code=302,
                    headers={"Location": "/index.php?page=item&id=17"},
                    url=LISTING_URL,
                ),
            )

        def get(self, url: str, **kwargs: object) -> _Response:
            params = kwargs.get("params")
            if isinstance(params, dict) and params.get("action") == "delete_comment":
                self.calls.append(("GET", url, kwargs))
                return _Response(
                    status_code=302,
                    headers={"Location": "/index.php?page=item&id=17"},
                    url=url,
                )
            if len(self.calls) >= 4:
                self.calls.append(("GET", url, kwargs))
                return _Response(status_code=200, text=witness_html, url=LISTING_URL)
            return super().get(url, **kwargs)

    session = _SoftErrorCleanupSession()
    editor = ClassifiedsEditor({"site_url": ORIGIN}, session)
    editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    with pytest.raises(EditorError, match="exact listing surface") as raised:
        editor.cleanup()

    assert raised.value.kind == "cleanup_failed"


def test_editor_fails_when_form_csrf_is_missing() -> None:
    editor, session = _editor(form_html=_form_html(csrf=""))

    with pytest.raises(EditorError) as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    assert raised.value.kind == "form_missing"
    assert [call[0] for call in session.calls] == ["GET"]


def test_editor_rejects_foreign_form_action_before_post() -> None:
    editor, session = _editor(form_html=_form_html(action="https://attacker.test/index.php"))

    with pytest.raises(EditorError) as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    assert raised.value.kind == "cross_origin_form_action"
    assert [call[0] for call in session.calls] == ["GET"]


@pytest.mark.parametrize(
    "rendered_html",
    [
        "<div class='comment'><h3>Title by alice</h3><p>UNIQUE-SIGNATURE</p></div>",
        "<div class='comment'><h3>Title by mallory</h3><p>UNIQUE-SIGNATURE</p>"
        "<a class='comment-reply' data-id='88'>Reply</a></div>",
        "<div class='comment'><h3>Title by alice</h3><p>UNIQUE-SIGNATURE</p>"
        "<a class='comment-reply' data-id='88'>Reply</a></div>"
        "<div class='comment'><h3>Other by alice</h3><p>UNIQUE-SIGNATURE</p>"
        "<a class='comment-reply' data-id='89'>Reply</a></div>",
    ],
)
def test_editor_rejects_missing_id_wrong_actor_or_ambiguous_matches(rendered_html: str) -> None:
    editor, session = _editor(rendered_html=rendered_html)

    with pytest.raises(EditorError) as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    assert raised.value.kind in {"schema_mismatch", "mutation_unreconciled"}
    assert [call[0] for call in session.calls] == ["GET", "POST", "GET"]


def test_editor_rejects_unexpected_post_status() -> None:
    editor, session = _editor(post_response=_Response(status_code=201, url=LISTING_URL))

    with pytest.raises(EditorError) as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    assert raised.value.kind == "request_failed"
    # HTTP 201 is not an accepted Classifieds form response, but it may have
    # committed the reply. The post-write GET must therefore reconcile the
    # exact delta before surfacing the primary request error.
    assert [call[0] for call in session.calls] == ["GET", "POST", "GET"]
    assert len(editor._cleanup_stack) == 1


def test_editor_reconciles_timeout_after_post_commit_before_raising() -> None:
    class _TimeoutAfterPostSession(_FakeSession):
        def request(self, method: str, url: str, **kwargs: object) -> _Response:
            self.calls.append((method, url, kwargs))
            raise requests.Timeout("synthetic timeout after server commit")

    session = _TimeoutAfterPostSession(
        form_html=_form_html(),
        rendered_html=_comment_html(),
        post_response=_Response(status_code=302, url=LISTING_URL),
    )
    editor = ClassifiedsEditor({"site_url": ORIGIN}, session)

    with pytest.raises(EditorError) as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    assert raised.value.kind == "request_failed"
    assert isinstance(raised.value.__cause__, requests.Timeout)
    assert [call[0] for call in session.calls] == ["GET", "POST", "GET"]
    # The recovery GET observed the one newly rendered ID, so the strict seed
    # cleanup boundary can still delete the writer-owned reply.
    assert len(editor._cleanup_stack) == 1


def test_editor_registers_cleanup_for_one_new_reply_when_refetch_identity_is_ambiguous() -> None:
    # The form GET is also the pre-write ID witness. The POST commits reply 88,
    # but the immediate refetch contains an unrelated reply as well, so exact
    # body/actor readback is ambiguous. The one-ID delta still identifies the
    # writer mutation and must be cleaned up by the normal seed boundary.
    class _AmbiguousRefetchSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__(
                form_html=_form_html() + _comment_html(reply_id="89"),
                rendered_html=_comment_html() + _comment_html(reply_id="89"),
                post_response=_Response(
                    status_code=302,
                    headers={"Location": "/index.php?page=item&id=17"},
                    url=LISTING_URL,
                ),
            )

        def get(self, url: str, **kwargs: object) -> _Response:
            params = kwargs.get("params")
            if isinstance(params, dict) and params.get("action") == "delete_comment":
                self.calls.append(("GET", url, kwargs))
                return _Response(
                    status_code=302,
                    headers={"Location": "/index.php?page=item&id=17"},
                    url=LISTING_URL,
                )
            if len(self.calls) >= 4:
                self.calls.append(("GET", url, kwargs))
                return _Response(status_code=200, text=_form_html(), url=LISTING_URL)
            return super().get(url, **kwargs)

    session = _AmbiguousRefetchSession()
    editor = ClassifiedsEditor({"site_url": ORIGIN}, session)

    with pytest.raises(EditorError, match="one exact rendered reply id") as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    assert raised.value.kind == "schema_mismatch"
    editor.cleanup()
    assert session.calls[-2][2]["params"]["comment"] == "88"


def test_editor_fails_closed_when_post_submit_delta_is_not_reconcilable() -> None:
    class _UnreconcilableSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__(
                form_html=_form_html(),
                rendered_html=(
                    _comment_html(reply_id="88")
                    + _comment_html(reply_id="89", actor="alice", body=BODY)
                ),
                post_response=_Response(
                    status_code=302,
                    headers={"Location": "/index.php?page=item&id=17"},
                    url=LISTING_URL,
                ),
            )

    session = _UnreconcilableSession()
    editor = ClassifiedsEditor({"site_url": ORIGIN}, session)

    with pytest.raises(EditorError, match="one exact new reply ID") as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    assert raised.value.kind == "mutation_unreconciled"
    assert len(editor._cleanup_stack) == 0


def test_classifieds_http_error_does_not_expose_form_secrets() -> None:
    csrf_marker = "csrf-fixture-marker"
    email_marker = "fixture-marker@example.test"
    form = _form_html(csrf=csrf_marker).replace("alice@example.test", email_marker)

    class _HttpErrorResponse(_Response):
        def raise_for_status(self) -> None:
            raise requests.HTTPError("synthetic HTTP error")

    editor, _session = _editor(
        form_html=form,
        post_response=_HttpErrorResponse(status_code=500, text=form),
    )

    with pytest.raises(EditorError) as raised:
        editor.create_listing_reply(listing_id="17", body=BODY, signature="UNIQUE-SIGNATURE")

    error = raised.value
    assert error.response_snippet is None
    assert csrf_marker not in str(error)
    assert email_marker not in str(error)
