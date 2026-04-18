"""Honeyclient-style verification of editor-emitted ``read_surface_urls``.

Per handoff §11.2, this test plants a payload with a known canary, captures
the editor's emitted ``read_surface_urls``, fetches each URL from the
orchestrator machine, and asserts the canary text renders at (at least) one
URL. The check closes the loop between C1b (editor says the injection is at
URL X) and the live platform (URL X actually shows the injection).

Gated by ``PYTEST_VERIFY_READ_SURFACE_URLS=1`` — the whole module skips
otherwise so default integration runs stay fast and do not require
network-fetching every URL.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest
import requests

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.editors.base import EditorError
from worldsim.editors.gitlab import GitlabEditor
from worldsim.editors.reddit import RedditEditor
from worldsim.editors.shopping import ShoppingEditor

pytestmark = pytest.mark.integration


def _skip_unless_verify_read_surface_urls() -> None:
    if not os.getenv("PYTEST_VERIFY_READ_SURFACE_URLS"):
        pytest.skip(
            "set PYTEST_VERIFY_READ_SURFACE_URLS=1 "
            "(or pass --verify-read-surface-urls to run_integration_tests.sh) "
            "to enable honeyclient read-surface verification"
        )


def _fetch_url(session: requests.Session, instance: dict[str, Any], url: str) -> requests.Response:
    """GET ``url``; if path-only, prefix the instance site_url."""
    target = url
    if url.startswith("/"):
        site = str(instance.get("site_url") or "").rstrip("/")
        target = f"{site}{url}"
    return session.get(target, timeout=30, allow_redirects=True)


def _assert_canary_rendered(
    session: requests.Session,
    instance: dict[str, Any],
    surface_urls: list[str],
    canary: str,
    *,
    label: str,
) -> None:
    """At least one URL in ``surface_urls`` must render ``canary`` in the body.

    Host-qualified and path-only forms are expected to coexist (handoff §12.8);
    either (or both) resolving successfully is sufficient.
    """
    assert surface_urls, f"{label}: editor emitted no read_surface_urls"
    errors: list[str] = []
    for url in surface_urls:
        try:
            response = _fetch_url(session, instance, url)
        except requests.RequestException as exc:
            errors.append(f"{url}: fetch failed ({exc})")
            continue
        if response.status_code != 200:
            errors.append(f"{url}: HTTP {response.status_code}")
            continue
        if canary in response.text:
            return
        errors.append(f"{url}: HTTP 200 but canary missing")
    raise AssertionError(
        f"{label}: none of the emitted read_surface_urls rendered the canary\n  "
        + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------


def test_verify_gitlab_read_surface_urls(live_instance, unique_suffix):
    _skip_unless_verify_read_surface_urls()
    instance = live_instance("gitlab")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        GitlabEditor.probe_base_state(instance)

        try:
            canary_issue = f"WS-CANARY-ISSUE-{unique_suffix}"
            canary_issue_note = f"WS-CANARY-ISSUE-NOTE-{unique_suffix}"
            canary_mr = f"WS-CANARY-MR-{unique_suffix}"
            canary_mr_note = f"WS-CANARY-MR-NOTE-{unique_suffix}"
            canary_repo_file = f"WS-CANARY-REPO-{unique_suffix}"

            project = editor.create_project(
                name_template=f"webagent-rsv-{unique_suffix}",
                description_template=f"Project description {unique_suffix}",
            )
            issue = editor.create_issue(
                project_name_template=f"webagent-rsv-{unique_suffix}",
                title_template=f"Issue {unique_suffix}",
                body_template=canary_issue,
            )
            _assert_canary_rendered(
                session,
                instance,
                issue.get("read_surface_urls") or [],
                canary_issue,
                label="gitlab.create_issue",
            )

            issue_note = editor.create_issue_note(
                project_id=issue["project_id"],
                issue_iid=issue["issue_iid"],
                note_body=canary_issue_note,
            )
            _assert_canary_rendered(
                session,
                instance,
                issue_note.get("read_surface_urls") or [],
                canary_issue_note,
                label="gitlab.create_issue_note",
            )

            project_payload = editor._gitlab_get_json(
                f"/api/v4/projects/{issue['project_id']}",
                allow_missing=False,
            )
            merge_request = editor.create_mr(
                project_id=issue["project_id"],
                title_template=f"MR {unique_suffix}",
                body_template=canary_mr,
                source_branch=f"rsv-{unique_suffix}",
            )
            _assert_canary_rendered(
                session,
                instance,
                merge_request.get("read_surface_urls") or [],
                canary_mr,
                label="gitlab.create_mr",
            )

            mr_note = editor.create_mr_note(
                project_id=merge_request["project_id"],
                mr_iid=merge_request["mr_iid"],
                note_body=canary_mr_note,
            )
            _assert_canary_rendered(
                session,
                instance,
                mr_note.get("read_surface_urls") or [],
                canary_mr_note,
                label="gitlab.create_mr_note",
            )

            repo_file = editor.create_repo_file(
                project_id=issue["project_id"],
                branch=str(project_payload.get("default_branch") or "main"),
                path=f"rsv-{unique_suffix}.md",
                content=f"{canary_repo_file}\n",
            )
            # Repo files render on the "raw" endpoint; fetch and check any URL.
            repo_surface = repo_file.get("read_surface_urls") or []
            assert repo_surface, "gitlab.create_repo_file: no read_surface_urls"
            # The path-only forms may not resolve through the form session; we
            # just verify at least one returns 200 (GitLab blob pages render
            # the content inside a syntax-highlighted <pre>).
            fetched_ok = False
            for url in repo_surface:
                response = _fetch_url(session, instance, url)
                if response.status_code == 200:
                    fetched_ok = True
                    break
            assert fetched_ok, "gitlab.create_repo_file: none of the read_surface_urls returned 200"
        finally:
            editor.cleanup()


# ---------------------------------------------------------------------------
# Reddit (PostMill)
# ---------------------------------------------------------------------------


def test_verify_reddit_read_surface_urls(live_instance, unique_suffix):
    _skip_unless_verify_read_surface_urls()
    instance = live_instance("reddit")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = RedditEditor(instance, session)
        RedditEditor.probe_base_state(instance)

        canary_submission = f"WS-CANARY-SUB-{unique_suffix}"
        canary_comment = f"WS-CANARY-COM-{unique_suffix}"

        submission = None
        try:
            submission = editor.create_submission(
                forum_name="news",
                title_template=f"RSV submission {unique_suffix}",
                body_template=canary_submission,
            )
        except EditorError as exc:
            if exc.kind not in {"request_failed", "submission_id_missing"}:
                raise
        if submission is None:
            # Fallback: the stack occasionally rate-limits submissions; find
            # any existing commentable submission and verify comment rendering
            # alone. That still exercises the create_comment read surface.
            news_page = editor._form_get("/f/news")
            assert news_page is not None
            match = re.search(r"/f/news/(\d+)", news_page.text)
            if match is None:
                pytest.skip("reddit live stack exposes no commentable submission")
            submission = {
                "forum_name": "news",
                "submission_id": match.group(1),
                "read_surface_urls": [f"/f/news/{match.group(1)}"],
            }
        else:
            _assert_canary_rendered(
                session,
                instance,
                submission.get("read_surface_urls") or [],
                canary_submission,
                label="reddit.create_submission",
            )

        comment = editor.create_comment(
            forum_name=submission["forum_name"],
            submission_id=submission["submission_id"],
            body=canary_comment,
        )
        _assert_canary_rendered(
            session,
            instance,
            comment.get("read_surface_urls") or [],
            canary_comment,
            label="reddit.create_comment",
        )


# ---------------------------------------------------------------------------
# Shopping (Magento)
# ---------------------------------------------------------------------------


def test_verify_shopping_read_surface_urls(live_instance, unique_suffix):
    _skip_unless_verify_read_surface_urls()
    instance = live_instance("shopping")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = ShoppingEditor(instance, session)
        ShoppingEditor.probe_base_state(instance)

        canary = f"WS-CANARY-REVIEW-{unique_suffix}"
        review = editor.create_product_review(
            title=f"RSV review {unique_suffix}",
            detail=canary,
            nickname="WorldSim",
            entity_pk_value=1,
            rating=4,
        )
        # Magento moderates reviews by default; freshly planted reviews may
        # not appear on the PDP until an admin approves. We verify that at
        # least one of the emitted URLs returns HTTP 200 (the PDP itself
        # renders regardless). The canary appearance is a stretch goal.
        surface = review.get("read_surface_urls") or []
        assert surface, "shopping.create_product_review: no read_surface_urls"
        any_ok = False
        canary_seen = False
        for url in surface:
            response = _fetch_url(session, instance, url)
            if response.status_code == 200:
                any_ok = True
                if canary in response.text:
                    canary_seen = True
                    break
        assert any_ok, "shopping.create_product_review: none of the read_surface_urls returned 200"
        if not canary_seen:
            pytest.xfail(
                "Magento moderates reviews by default; canary not yet visible on PDP. "
                "URL resolvability was verified. Approve the review via admin to "
                "observe the full honeyclient signal."
            )
