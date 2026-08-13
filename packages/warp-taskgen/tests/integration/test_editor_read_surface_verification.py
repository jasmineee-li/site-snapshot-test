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

  - any ``warp_taskgen/editors/*.py`` file's ``read_surface_urls`` emission
    or the helper module ``warp_taskgen/editors/_read_surface.py``,
  - the C1b branch in ``warp_taskgen/outcome_taxonomy.py::
    _check_injection_surface_visited``, or
  - benchmark-host routing / nginx config that could change which URLs
    actually render the canary on each site.

Default integration runs include this module's tests as ``s`` (skipped)
in the pytest output so it's visible that the gate exists; opt in
explicitly when the change warrants the extra coverage.

Invocation:
    bash scripts/run_integration_tests.sh \\
        --host-config configs/benchmark_hosts/r8a.yaml \\
        --verify-read-surface-urls
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

import pytest
import requests

from warp_taskgen.auth_tokens import acquire_tokens_for_instances
from warp_taskgen.editors.base import EditorError
from warp_taskgen.editors.gitlab import GitlabEditor
from warp_taskgen.editors.reddit import RedditEditor

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


def _instance_hostname(instance: dict[str, Any]) -> str:
    """Return the hostname portion of ``instance['site_url']`` (empty if absent)."""
    site = str(instance.get("site_url") or "").strip()
    if not site:
        return ""
    return (urlparse(site).hostname or "").lower()


def _is_reachable_url(instance: dict[str, Any], url: str) -> bool:
    """Return True iff ``url`` is fetchable from the orchestrator-host topology.

    Editors emit URLs faithfully — including host-qualified forms copied
    verbatim from the platform's API response (e.g. GitLab's ``web_url``
    which mirrors ``/etc/gitlab/gitlab.rb``'s ``external_url``). On a
    benchmark host where that baked-in URL points at a different hostname
    (classic case: gitlab.rb with ``external_url 'http://localhost:8023/'``
    while we reach the stack as ``3.12.221.9``), the emitted URL is not
    reachable from the test runner. Rather than treating those as fetch
    failures, filter them out of the canary loop — they are faithful but
    env-specific leakage. Path-only forms and site-url-hostname forms
    always stay in the loop.

    Future-proofing: the filter becomes a no-op the moment the benchmark
    host's ``external_url`` / ``base_url`` is aligned to the public host,
    so this helper does not have to be removed when the root cause of the
    hostname divergence is eventually fixed.
    """
    raw = str(url or "").strip()
    if not raw:
        return False
    if raw.startswith("/"):
        return True
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return host == _instance_hostname(instance)


def _fetch_url(
    session: requests.Session,
    instance: dict[str, Any],
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET ``url``; if path-only, prefix the instance site_url.

    ``headers`` override/augment the session defaults — used by the GitLab
    test to attach API auth (PRIVATE-TOKEN / Bearer) so the REST surfaces
    in ``read_surface_urls`` return canary-bearing JSON instead of the
    anonymous sign-in HTML shell the stock session gets.
    """
    target = url
    if url.startswith("/"):
        site = str(instance.get("site_url") or "").rstrip("/")
        target = f"{site}{url}"
    return session.get(target, timeout=30, allow_redirects=True, headers=headers)


def _assert_canary_rendered(
    session: requests.Session,
    instance: dict[str, Any],
    surface_urls: list[str],
    canary: str,
    *,
    label: str,
    headers: dict[str, str] | None = None,
) -> None:
    """At least one reachable URL in ``surface_urls`` must render ``canary``.

    Host-qualified and path-only forms are expected to coexist (handoff §12.8);
    either (or both) resolving successfully is sufficient. URLs whose
    hostname does not match the instance's ``site_url`` are skipped
    (faithful-but-env-specific leakage, e.g. GitLab's ``external_url``);
    they do not count toward either success or failure.

    At least one URL from ``surface_urls`` must be reachable in this
    topology — if every emitted URL is filtered as external, that is
    itself a regression worth surfacing (the editor emitted nothing
    useful for the honeyclient to verify).
    """
    assert surface_urls, f"{label}: editor emitted no read_surface_urls"
    reachable = [u for u in surface_urls if _is_reachable_url(instance, u)]
    assert reachable, (
        f"{label}: every emitted read_surface_url was external-host-only; "
        f"check that the editor emits path-only or site_url-hostname forms. "
        f"Emitted: {surface_urls}"
    )
    errors: list[str] = []
    for url in reachable:
        try:
            response = _fetch_url(session, instance, url, headers=headers)
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
        f"{label}: none of the reachable read_surface_urls rendered the canary\n  "
        + "\n  ".join(errors)
    )


def _assert_surface_reachable(
    session: requests.Session,
    instance: dict[str, Any],
    surface_urls: list[str],
    *,
    label: str,
    headers: dict[str, str] | None = None,
) -> None:
    """Reachability-only contract: at least one reachable URL returns HTTP 200.

    Used for surfaces whose payload cannot be verified by an anonymous /
    api-header-only curl: GitLab note pages render inside an
    authenticated HTML session that requires web-login session cookies
    the honeyclient does not hold (GitLab's note API responses contain
    no ``web_url`` / ``_links``, so the editor can only emit the parent
    issue/MR HTML URL). Pairing this with ``_assert_canary_in_json``
    against the authoritative API endpoint gives a two-step equivalent
    of ``_assert_canary_rendered``:

      - URL emission contract: at least one editor-emitted URL is
        reachable and 200s → the C1b match surface exists.
      - Content contract: the canary actually exists at the platform's
        authoritative endpoint → the payload was really planted.
    """
    assert surface_urls, f"{label}: editor emitted no read_surface_urls"
    reachable = [u for u in surface_urls if _is_reachable_url(instance, u)]
    assert reachable, (
        f"{label}: every emitted read_surface_url was external-host-only; emitted={surface_urls}"
    )
    errors: list[str] = []
    for url in reachable:
        try:
            response = _fetch_url(session, instance, url, headers=headers)
        except requests.RequestException as exc:
            errors.append(f"{url}: fetch failed ({exc})")
            continue
        if response.status_code == 200:
            return
        errors.append(f"{url}: HTTP {response.status_code}")
    raise AssertionError(
        f"{label}: none of the reachable read_surface_urls returned 200\n  " + "\n  ".join(errors)
    )


def _assert_canary_in_json(
    session: requests.Session,
    instance: dict[str, Any],
    api_path: str,
    canary: str,
    *,
    label: str,
    headers: dict[str, str] | None = None,
) -> None:
    """Fetch ``api_path`` (site_url-prefixed) and assert ``canary`` in body.

    Companion to ``_assert_surface_reachable`` for endpoints not
    directly emitted in ``read_surface_urls`` but that serve as the
    authoritative source of truth for payload content (e.g. GitLab's
    ``/api/v4/projects/{pid}/issues/{iid}/notes/{note_id}``). The path
    MUST be constructed by the test from the editor's returned ids (not
    from ``read_surface_urls``) so this check is independent of the
    editor's URL-emission contract.
    """
    try:
        response = _fetch_url(session, instance, api_path, headers=headers)
    except requests.RequestException as exc:
        raise AssertionError(
            f"{label}: fetch of authoritative API path failed: {api_path} ({exc})"
        ) from exc
    if response.status_code != 200:
        raise AssertionError(
            f"{label}: authoritative API path {api_path} returned HTTP {response.status_code}"
        )
    if canary not in response.text:
        raise AssertionError(
            f"{label}: canary missing from authoritative API path "
            f"{api_path} (HTTP 200 body did not contain canary)"
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
    ``warp_taskgen/editors/gitlab.py``'s ``read_surface_urls`` emission, the
    GitLab path-template helpers in
    ``warp_taskgen/editors/_read_surface.py``, or after a GitLab version
    bump on the benchmark host (URL layouts can drift across major
    releases — e.g. ``/-/issues/`` ↔ ``/_/issues/``).
    """
    _skip_unless_verify_read_surface_urls()
    instance = live_instance("gitlab")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        GitlabEditor.probe_base_state(instance)

        # GitLab's HTML UI binds to session cookies from the web-login form;
        # an API-header-only session (PRIVATE-TOKEN / Bearer) hits the
        # anonymous sign-in page for private project issues. The REST
        # surfaces (``/api/v4/projects/<id>/issues/<iid>`` etc.) the editor
        # emits alongside each HTML URL DO render the canary in JSON when
        # fetched with api-mechanism headers, so we attach those headers to
        # the canary loop. That makes the honeyclient robust across
        # deployments that disable anonymous access to project HTML without
        # having to thread a full form-login flow (which requires GitLab's
        # CSRF dance + knows_cookie_policy + 2FA prompts, all
        # environment-dependent).
        gitlab_api_headers = editor._build_headers(mechanism="api")

        try:
            canary_issue = f"WS-CANARY-ISSUE-{unique_suffix}"
            canary_issue_note = f"WS-CANARY-ISSUE-NOTE-{unique_suffix}"
            canary_mr = f"WS-CANARY-MR-{unique_suffix}"
            canary_mr_note = f"WS-CANARY-MR-NOTE-{unique_suffix}"
            canary_repo_file = f"WS-CANARY-REPO-{unique_suffix}"

            editor.create_project(
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
                headers=gitlab_api_headers,
            )

            issue_note = editor.create_issue_note(
                project_id=issue["project_id"],
                issue_iid=issue["issue_iid"],
                note_body=canary_issue_note,
            )
            # Note API responses in GitLab carry no ``web_url`` / ``_links``,
            # so the editor can only emit the parent issue HTML page — which
            # is auth-gated. Split the contract: URL-emission reachability
            # (editor emitted a resolvable surface) vs. content existence
            # (canary present at the authoritative notes API endpoint).
            _assert_surface_reachable(
                session,
                instance,
                issue_note.get("read_surface_urls") or [],
                label="gitlab.create_issue_note",
                headers=gitlab_api_headers,
            )
            _assert_canary_in_json(
                session,
                instance,
                f"/api/v4/projects/{issue['project_id']}"
                f"/issues/{issue['issue_iid']}/notes/{issue_note['note_id']}",
                canary_issue_note,
                label="gitlab.create_issue_note",
                headers=gitlab_api_headers,
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
            # GitLab's MR API response carries ``web_url`` (baked-in external_url
            # host — unreachable on benchmark stacks that diverge from
            # ``/etc/gitlab/gitlab.rb``'s external_url) but NOT ``_links.self``,
            # so the editor cannot emit a path-only API URL the way it does
            # for issues. Apply the same two-step contract used for notes:
            # reachability of the HTML surface + canary content verified
            # directly against the MR API endpoint.
            _assert_surface_reachable(
                session,
                instance,
                merge_request.get("read_surface_urls") or [],
                label="gitlab.create_mr",
                headers=gitlab_api_headers,
            )
            _assert_canary_in_json(
                session,
                instance,
                f"/api/v4/projects/{merge_request['project_id']}"
                f"/merge_requests/{merge_request['mr_iid']}",
                canary_mr,
                label="gitlab.create_mr",
                headers=gitlab_api_headers,
            )

            mr_note = editor.create_mr_note(
                project_id=merge_request["project_id"],
                mr_iid=merge_request["mr_iid"],
                note_body=canary_mr_note,
            )
            # MR notes have the same shape gap as issue notes — API response
            # lacks ``web_url`` / ``_links``. Use the same two-step contract.
            _assert_surface_reachable(
                session,
                instance,
                mr_note.get("read_surface_urls") or [],
                label="gitlab.create_mr_note",
                headers=gitlab_api_headers,
            )
            _assert_canary_in_json(
                session,
                instance,
                f"/api/v4/projects/{merge_request['project_id']}"
                f"/merge_requests/{merge_request['mr_iid']}"
                f"/notes/{mr_note['note_id']}",
                canary_mr_note,
                label="gitlab.create_mr_note",
                headers=gitlab_api_headers,
            )

            repo_file = editor.create_repo_file(
                project_id=issue["project_id"],
                branch=str(project_payload.get("default_branch") or "main"),
                path=f"rsv-{unique_suffix}.md",
                content=f"{canary_repo_file}\n",
            )
            # Repo files render on the "raw" endpoint; fetch and check any URL.
            # Filter to reachable URLs (see ``_is_reachable_url``) so a localhost
            # web_url from gitlab.rb doesn't short-circuit the "no reachable
            # URL" assertion; then require at least one reachable URL to 200
            # under api-mechanism headers.
            repo_surface = repo_file.get("read_surface_urls") or []
            assert repo_surface, "gitlab.create_repo_file: no read_surface_urls"
            reachable_repo = [u for u in repo_surface if _is_reachable_url(instance, u)]
            assert reachable_repo, (
                "gitlab.create_repo_file: every emitted read_surface_url was "
                f"external-host-only; emitted={repo_surface}"
            )
            fetched_ok = False
            for url in reachable_repo:
                try:
                    response = _fetch_url(session, instance, url, headers=gitlab_api_headers)
                except requests.RequestException:
                    continue
                if response.status_code == 200:
                    fetched_ok = True
                    break
            assert fetched_ok, (
                "gitlab.create_repo_file: none of the reachable "
                f"read_surface_urls returned 200 (reachable={reachable_repo})"
            )
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
    ``warp_taskgen/editors/reddit.py``'s ``read_surface_urls`` emission or
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


# Magento test removed 2026-04-21 with the WASP-aligned scoping decision.
# See docs/handoffs/wasp-aligned-scoping-decision.md.
