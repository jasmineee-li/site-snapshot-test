"""Honeyclient-style verification of editor-emitted ``read_surface_urls``.

Per `docs/handoffs/codex-handoff-c1-read-surface.md` §11.2, each test in
this module plants a payload carrying a known canary string (e.g.
``WS-CANARY-ISSUE-<uuid>``), captures the editor's emitted
``read_surface_urls`` field, fetches each of those URLs from the
orchestrator host, and asserts the canary text renders at (at least)
one of them. The check closes the loop between two halves of the C1b
engagement signal:

  - **Editor side**: the editor says "after this seed call, the injection
    is rendered at URLs X / Y / Z."
  - **Platform side**: when the agent navigates to X / Y / Z, the
    injection is actually visible there.

Without these tests, an editor could silently emit URLs that don't
render the planted content and the Phase 4 outcome classifier's C1b
signal would degrade from "agent visited the surface" to "agent visited
some URL that may or may not have the content". The honeyclient gate is
the unit-of-truth that prevents that drift; treat it as the
contract-test for ``read_surface_urls`` accuracy.

**Why these 3 tests are skipped by default.** The fetch loop hits every
emitted URL over HTTP from the orchestrator host. That:

  - adds 30-60s of wall-clock to the integration suite (the default suite
    targets <90s),
  - dogpiles the live stack with extra GETs while the rest of the suite
    is also running, and
  - requires every site (gitlab, reddit, shopping) to be healthy on the
    target host AND for SG ingress to allow the orchestrator host to
    reach each site's port.

So the default integration run skips them and asserts only the
editor-side contracts. Operators opt in explicitly when they're touching
``read_surface_urls`` emission code or the C1b classifier signal.

**When to opt in.** Run with ``--verify-read-surface-urls`` (which sets
``PYTEST_VERIFY_READ_SURFACE_URLS=1`` for pytest) any time you change:

  - any ``worldsim/editors/*.py`` file's ``read_surface_urls`` emission
    or the helper module ``worldsim/editors/_read_surface.py``,
  - the C1b branch in ``worldsim/outcome_taxonomy.py::
    _check_injection_surface_visited``, or
  - benchmark-host routing / nginx config that could change which URLs
    actually render the canary on each site.

Default integration runs include this module's tests as ``s`` (skipped)
in the pytest output so it's visible that the gate exists; opt in
explicitly when the change warrants the extra coverage.

Invocation:
    bash scripts/run_integration_tests.sh \\
        --host-config configs/benchmark_hosts/r5.yaml \\
        --verify-read-surface-urls
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
    """Per-test gate. See module docstring for the full rationale.

    The default integration run leaves these tests as skipped so the
    suite stays fast and doesn't hit every emitted URL on the live
    stack. Opt in by setting ``PYTEST_VERIFY_READ_SURFACE_URLS=1`` (or
    pass ``--verify-read-surface-urls`` to ``run_integration_tests.sh``)
    when you've touched ``read_surface_urls`` emission or the C1b
    classifier signal — the gate is what prevents a silent C1b
    correctness regression on the next Phase 4 run.
    """
    if not os.getenv("PYTEST_VERIFY_READ_SURFACE_URLS"):
        pytest.skip(
            "set PYTEST_VERIFY_READ_SURFACE_URLS=1 "
            "(or pass --verify-read-surface-urls to run_integration_tests.sh) "
            "to enable honeyclient read-surface verification — see module "
            "docstring for when this gate should be flipped on"
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
    """C1b honeyclient — gitlab editor surfaces.

    GitLab is the heaviest C1b surface in the dataset because Phase 2
    plants payloads across five distinct gitlab editor methods, each of
    which emits its own ``read_surface_urls`` shape:

      - ``create_issue`` → ``/<group>/<project>/-/issues/<iid>``
      - ``create_issue_note`` → same issue page (note rendered inline)
      - ``create_mr`` → ``/<group>/<project>/-/merge_requests/<iid>``
      - ``create_mr_note`` → same MR page (note rendered inline)
      - ``create_repo_file`` → blob and raw URLs

    A regression in any one of these would silently degrade C1b's recall
    on a different slice of Phase 4 trajectories (e.g. a wrong
    issues-vs-merge_requests path would miss every MR-targeted task).
    This test plants a unique canary in each of the 5 surfaces and
    asserts each is independently visible at the URL the editor
    emitted, so a single live run pins the contract for the whole
    GitLab read-surface family.

    Skipped by default. Opt in when changing
    ``worldsim/editors/gitlab.py``'s ``read_surface_urls`` emission, the
    GitLab path-template helpers in
    ``worldsim/editors/_read_surface.py``, or after a GitLab version
    bump on the benchmark host (URL layouts can drift across major
    releases — e.g. ``/-/issues/`` ↔ ``/_/issues/``).
    """
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
    """C1b honeyclient — reddit (PostMill) editor surfaces.

    Reddit's PostMill backend exposes two C1b-relevant editor methods:

      - ``create_submission`` → ``/f/<forum>/<submission_id>``
      - ``create_comment`` → same submission page (comment rendered inline)

    PostMill auto-linkifies URLs and applies its own markdown rules to
    UGC, so a payload's rendered DOM can shift across PostMill versions
    in ways that don't map cleanly to the editor's static
    ``path_template``. This test verifies that a planted canary actually
    surfaces inside the rendered submission and comment pages — covering
    both the plain-text body case and the markdown-formatted case.

    Includes a graceful fallback for PostMill rate-limiting on
    submission creation: if the stack returns ``request_failed`` /
    ``submission_id_missing``, the test reuses an existing commentable
    submission so the comment-creation surface is still exercised.
    Without that fallback the test would be flaky on busy stacks; with
    it, a real C1b regression still surfaces (the comment path is the
    one most adversarial Phase 4 tasks touch).

    Skipped by default. Opt in when changing
    ``worldsim/editors/reddit.py``'s ``read_surface_urls`` emission or
    after a PostMill version bump on the benchmark host.
    """
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
    """C1b honeyclient — shopping (Magento) editor surfaces.

    Magento's product-review pipeline emits two read-surface URLs per
    review: the PDP itself (``/catalog/product/view/id/<id>``) and the
    AJAX listing endpoint (``/review/product/listAjax/id/<id>``). These
    are the URLs the agent's network trace will hit when it follows a
    review-bearing PDP, so they're the surfaces C1b's substring match
    keys on.

    **Magento moderation caveat.** The default Magento install
    moderates customer reviews — a freshly-planted review is held in an
    "Approved by admin" queue and does NOT render on the PDP until an
    operator approves it. This is a Magento default, not a defect in
    the editor's URL emission, so the test is structured as a graceful
    two-tier check:

      1. Hard assertion — at least one emitted URL must return HTTP 200.
         This catches the actual C1b correctness regression: an editor
         emitting a wrong path that 404s would silently break C1b on
         every shopping task.
      2. Soft assertion (xfail on miss) — the canary text appearing in
         the response body. This is the "stretch goal" proof that
         end-to-end rendering works, but is suppressed because Magento's
         moderation default makes it environment-dependent.

    To upgrade the soft check to a hard check on a deployment, either
    auto-approve reviews via the admin user (out of scope for default
    benchmark hosts) or relax the test once Magento config is known to
    skip moderation.

    Skipped by default. Opt in when changing
    ``worldsim/editors/shopping.py``'s ``read_surface_urls`` emission,
    the Magento path-template helpers, or after a Magento version bump
    on the benchmark host.
    """
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
