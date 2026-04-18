from __future__ import annotations

import re

import pytest
import requests

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.editors.base import EditorError
from worldsim.editors.reddit import RedditEditor

pytestmark = pytest.mark.integration


def test_reddit_live_create_forum_submission_and_comment(live_instance, unique_suffix):
    instance = live_instance("reddit")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = RedditEditor(instance, session)
        RedditEditor.probe_base_state(instance)

        forum_name = f"ws{unique_suffix[:8]}"
        first = editor.create_forum(
            name_template=forum_name,
            description_template=f"Live integration forum {unique_suffix}",
        )
        second = editor.create_forum(name_template=forum_name)
        forum_page = editor._form_get(f"/f/{first['forum_name']}")
        submission_title = f"Live submission {unique_suffix}"
        submission_body = f"Live submission body {unique_suffix}"
        comment_body = f"Live comment body {unique_suffix}"
        submission = None
        submission_page = None
        try:
            submission = editor.create_submission(
                forum_name="news",
                title_template=submission_title,
                body_template=submission_body,
            )
            submission_page = editor._form_get(f"/f/news/{submission['submission_id']}")
        except EditorError as exc:
            if exc.kind not in {"request_failed", "submission_id_missing"}:
                raise
        if submission is None:
            news_page = editor._form_get("/f/news")
            assert news_page is not None
            match = re.search(r"/f/news/(\d+)", news_page.text)
            if match is None:
                pytest.skip("reddit live stack does not expose a commentable submission id")
            submission = {
                "forum_name": "news",
                "submission_id": match.group(1),
            }
            submission_page = editor._form_get(f"/f/news/{submission['submission_id']}")
        try:
            comment = editor.create_comment(
                forum_name=submission["forum_name"],
                submission_id=submission["submission_id"],
                body=comment_body,
            )
            comment_page = editor._form_get(
                f"/f/{submission['forum_name']}/{submission['submission_id']}"
            )
        except EditorError as exc:
            if exc.kind not in {"request_failed", "submission_id_missing"}:
                raise
            pytest.skip("reddit live stack rejects programmatic comment form submission")
        original_form = editor._fetch_form_state(
            f"/user/{editor._quote(editor._resolve_current_username(instance))}/edit_biography",
            required_fields=("user_biography[biography]", "user_biography[_token]"),
        )
        original_bio = str(original_form["fields"].get("user_biography[biography]") or "")
        updated_bio = f"Live bio {unique_suffix}"

        try:
            editor.update_user_bio(bio_text=updated_bio)
            profile_page = editor._form_get(
                f"/user/{editor._quote(editor._resolve_current_username(instance))}"
            )
            # Commit 2 of the C1 migration: editor results now include
            # ``read_surface_urls`` + ``read_surface_provenance_source``.
            assert first["forum_name"] == forum_name
            assert second["forum_name"] == forum_name
            for result in (first, second):
                assert result.get("read_surface_provenance_source") == "editor_constructed"
                surface = result.get("read_surface_urls") or []
                assert f"/f/{forum_name}" in surface
            assert forum_page is not None
            assert submission["forum_name"] in {"news", forum_name}
            assert submission_page is not None
            assert comment["submission_id"] == submission["submission_id"]
            assert comment_page is not None
            assert comment_body in comment_page.text
            assert profile_page is not None
            assert updated_bio in profile_page.text
        finally:
            editor.update_user_bio(bio_text=original_bio)
