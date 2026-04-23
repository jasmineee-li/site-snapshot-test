"""Phase 2c preflight — HTTP-probe deterministic source-data surfaces before
the expensive seed/render/reachability verifier.

Bug I (2026-04-23). Three classes of tasks sit in
``adversarial_tasks.infeasible.json`` whose actual problem is **dataset
quality**, not a probe bug:

* Private-MR tasks where the benign user gets a 302 to
  ``/users/sign_in`` or a ~443-byte login-stub body — the seed writes,
  the render probe sees the shell, and the task is classified
  ``render_unverified``. Benign user genuinely cannot see the MR; the
  adversarial trial would run against a page the victim never reaches.
* Reddit tasks with stale L4 anchors
  (e.g. ``/f/headphones/4``, ``/f/news/6``) where the submission id
  embedded in the editor call was valid at dataset-generation time but
  the submission no longer exists. Editor GET → 404. Classified
  ``request_failed``.
* Occasional 403 / 410 / 401 variants of the above.

These are different from transient backend contention (5xx, 429,
ReadTimeout) which should *stay* in the infeasible bucket and retry on
the next 2c run. The preflight only quarantines when the server
DETERMINISTICALLY signals a broken benign surface.

Probe mechanism: a Playwright ``APIRequestContext`` initialized with the
same benign agent auth shape Phase 4 uses (storage_state, http_headers,
or http_basic). Each task probes both
``benign_target_resource.start_url_resolved`` and any read/attach surface
implied by editor-call anchors, e.g. Reddit ``/f/<forum>/<submission>``
or GitLab ``/<project>/-/merge_requests/<iid>``. GET with
``max_redirects=0`` and a 5 s timeout; read the first 8 KB of body.
Classifier table (see ``_classify_probe``) translates HTTP outcome to a
stable ``source_data_issue.kind`` label.

Safety valve: whole-run bailout if more than ``bailout_ratio`` (default
50 %) of probes classify as ``login_redirect``. That shape means the
storage_state cookie expired mid-run and mass-quarantining the dataset
is the wrong call — let the existing infeasible path catch it.

Output: tuple ``(tasks_to_probe, tasks_dropped)``. ``tasks_dropped``
records carry an added ``source_data_issue`` dict with ``kind``,
``http_status``, ``detail``, ``probed_at``, ``probed_url``. They are
written to ``logs/phase_2/adversarial_tasks.dropped_source_data.json``
by the caller.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from worldsim.phases.phase_2_reachability import resolve_start_url

logger = logging.getLogger(__name__)

# Default knob values. The concurrency sits safely inside the replica's
# 16-HTTP-slot puma budget; the timeout is shorter than the real probe's
# 30 s so a transient timeout here does NOT quarantine.
DEFAULT_PREFLIGHT_CONCURRENCY = 16
DEFAULT_PREFLIGHT_TIMEOUT_S = 5.0
DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO = 0.5

_AUTH_SELF_TEST_PATHS: dict[str, str] = {
    # Browser-facing session check: unauthenticated users redirect to
    # /users/sign_in, while a live cookie jar reaches the profile page.
    "gitlab": "/-/profile",
}

# Body markers indicating the 200-response is actually a login page
# (GitLab + Postmill share most of these; we keep substrings short and
# attribute-anchored so a README that mentions "sign in" does not
# false-trip).
_LOGIN_STUB_MARKERS: tuple[str, ...] = (
    'action="/users/sign_in"',
    'name="user[login]"',
    'name="user[email]"',
    "/users/sign_up",
)
_LOGIN_STUB_BODY_SIZE_LIMIT = 600

# Location headers that unambiguously route the benign user to a sign-in
# page.
_LOGIN_REDIRECT_LOCATION_MARKERS: tuple[str, ...] = (
    "/users/sign_in",
    "/users/sign_up",
    "/login",
    "/auth/sign_in",
)


@dataclass(frozen=True)
class PreflightClassification:
    """Outcome of a single URL probe. ``quarantine=False`` means the
    task goes to the normal infeasible / verified pipeline."""

    kind: str  # e.g. "login_redirect", "not_found", "probe_timeout"
    quarantine: bool
    http_status: int | None
    detail: str


@dataclass(frozen=True)
class ProbeTarget:
    url: str
    source: str


_BENIGN_TOKEN_RE = re.compile(r"\{benign_([A-Za-z0-9_]+)\}")


def _looks_like_login_stub(body: str) -> bool:
    if not body:
        return False
    lowered = body.lower()
    for marker in _LOGIN_STUB_MARKERS:
        if marker.lower() in lowered:
            return True
    # Short bodies matching "Sign in" are a strong signal but we keep
    # the body-size gate so a long legit page mentioning "sign in" in
    # prose does not trip.
    if len(body) <= _LOGIN_STUB_BODY_SIZE_LIMIT and "sign in" in lowered:
        return True
    return False


def _location_is_login(location: str | None) -> bool:
    if not location:
        return False
    lower = location.lower()
    return any(marker in lower for marker in _LOGIN_REDIRECT_LOCATION_MARKERS)


def _classify_probe(
    *,
    status: int | None,
    headers: dict[str, str] | None,
    body_snippet: str,
    exception_name: str | None,
) -> PreflightClassification:
    if exception_name:
        name = exception_name.lower()
        if "timeout" in name:
            return PreflightClassification(
                kind="probe_timeout",
                quarantine=False,
                http_status=None,
                detail=f"preflight probe timed out ({exception_name})",
            )
        return PreflightClassification(
            kind="host_unreachable",
            quarantine=False,
            http_status=None,
            detail=f"preflight probe raised {exception_name}",
        )
    if status is None:
        return PreflightClassification(
            kind="host_unreachable",
            quarantine=False,
            http_status=None,
            detail="preflight probe returned no status",
        )
    if status == 200:
        if _looks_like_login_stub(body_snippet):
            return PreflightClassification(
                kind="login_redirect",
                quarantine=True,
                http_status=200,
                detail=("200 with login-stub markers — benign user cannot reach this surface"),
            )
        return PreflightClassification(
            kind="reachable",
            quarantine=False,
            http_status=200,
            detail="200 OK",
        )
    if 300 <= status < 400:
        # Playwright lowercases all response header names, so ``location``
        # is the only key that ever appears. Keeping the fallback to the
        # raw spelling would look defensive but is dead in practice and
        # obscures the invariant — Playwright's header contract.
        location = headers.get("location") if headers else None
        if _location_is_login(location):
            return PreflightClassification(
                kind="login_redirect",
                quarantine=True,
                http_status=status,
                detail=f"{status} redirect to {location}",
            )
        return PreflightClassification(
            kind="redirect_noncritical",
            quarantine=False,
            http_status=status,
            detail=f"{status} redirect (non-login)",
        )
    if status == 401:
        return PreflightClassification(
            kind="auth_missing",
            quarantine=True,
            http_status=status,
            detail="401 Unauthorized — benign storage_state did not authenticate",
        )
    if status == 403:
        return PreflightClassification(
            kind="forbidden",
            quarantine=True,
            http_status=status,
            detail="403 Forbidden — benign user lacks permission for this surface",
        )
    if status == 404:
        return PreflightClassification(
            kind="not_found",
            quarantine=True,
            http_status=status,
            detail="404 Not Found — stale L4 anchor or deleted resource",
        )
    if status == 410:
        return PreflightClassification(
            kind="gone",
            quarantine=True,
            http_status=status,
            detail="410 Gone — resource permanently removed",
        )
    if status == 429:
        return PreflightClassification(
            kind="rate_limited",
            quarantine=False,
            http_status=status,
            detail="429 Too Many Requests — transient",
        )
    if 500 <= status < 600:
        return PreflightClassification(
            kind="server_error",
            quarantine=False,
            http_status=status,
            detail=f"{status} server error — transient",
        )
    return PreflightClassification(
        kind="unexpected_status",
        quarantine=False,
        http_status=status,
        detail=f"unexpected HTTP {status}",
    )


def _task_probe_url(task: dict[str, Any], instance_site_url: str) -> str | None:
    target = task.get("benign_target_resource")
    if not isinstance(target, dict):
        return None
    start_url = target.get("start_url_resolved")
    if not isinstance(start_url, str) or not start_url.strip():
        return None
    return resolve_start_url(start_url, instance_site_url)


def _render_anchor_tokens(value: Any, anchors: dict[str, Any]) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        replacement = anchors.get(key)
        return "" if replacement is None else str(replacement)

    rendered = _BENIGN_TOKEN_RE.sub(repl, text).strip()
    return rendered or None


def _first_value(
    args: dict[str, Any],
    anchors: dict[str, Any],
    *names: str,
) -> str | None:
    for name in names:
        if name in args:
            rendered = _render_anchor_tokens(args.get(name), anchors)
            if rendered:
                return rendered
        rendered = _render_anchor_tokens(anchors.get(name), anchors)
        if rendered:
            return rendered
    return None


def _editor_surface_path(
    *,
    site: str,
    method: str,
    args: dict[str, Any],
    anchors: dict[str, Any],
) -> str | None:
    if site == "reddit":
        if method == "create_comment":
            forum = _first_value(args, anchors, "forum_name")
            submission = _first_value(args, anchors, "submission_id")
            if forum and submission:
                return f"/f/{quote(forum, safe='')}/{quote(submission, safe='')}"
        if method == "create_submission":
            forum = _first_value(args, anchors, "forum_name")
            if forum:
                return f"/submit/{quote(forum, safe='')}"
        return None

    if site == "gitlab":
        project_path = _first_value(
            args,
            anchors,
            "project_path",
            "project_path_template",
        )
        if not project_path:
            return None
        project_path = project_path.strip("/")
        if method == "create_issue_note":
            issue_iid = _first_value(args, anchors, "issue_iid")
            if issue_iid:
                return f"/{project_path}/-/issues/{quote(issue_iid, safe='')}"
        if method == "create_mr_note":
            mr_iid = _first_value(args, anchors, "mr_iid")
            if mr_iid:
                return f"/{project_path}/-/merge_requests/{quote(mr_iid, safe='')}"
        if method == "create_repo_file":
            branch = _first_value(args, anchors, "branch") or "main"
            file_path = _first_value(args, anchors, "path")
            if file_path:
                quoted_file = "/".join(quote(part, safe="") for part in file_path.split("/"))
                return f"/{project_path}/-/blob/{quote(branch, safe='')}/{quoted_file}"
    return None


def _task_probe_targets(task: dict[str, Any], instance_site_url: str) -> list[ProbeTarget]:
    targets: list[ProbeTarget] = []
    start_url = _task_probe_url(task, instance_site_url)
    if start_url:
        targets.append(ProbeTarget(url=start_url, source="benign_start_url"))

    resource = task.get("benign_target_resource")
    anchors_raw = resource.get("anchors") if isinstance(resource, dict) else None
    anchors = anchors_raw if isinstance(anchors_raw, dict) else {}
    seed = task.get("adversarial_data_seed")
    calls = seed.get("editor_calls") if isinstance(seed, dict) else None
    if not isinstance(calls, list):
        return _dedupe_targets(targets)

    task_site = str(task.get("site") or "").strip().lower()
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        site = str(call.get("site") or task_site).strip().lower()
        method = str(call.get("method") or "").strip()
        args = call.get("args")
        if not method or not isinstance(args, dict):
            continue
        path = _editor_surface_path(site=site, method=method, args=args, anchors=anchors)
        if path:
            targets.append(
                ProbeTarget(
                    url=resolve_start_url(path, instance_site_url),
                    source=f"editor_call[{index}].{site}.{method}",
                )
            )
    return _dedupe_targets(targets)


def _dedupe_targets(targets: list[ProbeTarget]) -> list[ProbeTarget]:
    deduped: list[ProbeTarget] = []
    seen: set[str] = set()
    for target in targets:
        if target.url in seen:
            continue
        seen.add(target.url)
        deduped.append(target)
    return deduped


async def _probe_one(
    *,
    request_context: Any,
    url: str,
    timeout_s: float,
) -> PreflightClassification:
    try:
        response = await request_context.get(
            url,
            timeout=timeout_s * 1000.0,
            max_redirects=0,
        )
    except Exception as exc:
        return _classify_probe(
            status=None,
            headers=None,
            body_snippet="",
            exception_name=exc.__class__.__name__,
        )
    try:
        status = response.status
    except Exception:
        status = None
    headers: dict[str, str] = {}
    try:
        raw_headers = response.headers
        if isinstance(raw_headers, dict):
            # Normalize keys to lowercase so the classifier can rely on
            # one canonical form. Playwright already lowercases headers;
            # this covers drift + non-Playwright test fakes that pass
            # raw capitalized headers.
            headers = {str(k).lower(): str(v) for k, v in raw_headers.items()}
    except Exception:
        headers = {}
    body_snippet = ""
    if status == 200:
        try:
            text = await response.text()
            body_snippet = (text or "")[:8192]
        except Exception:
            body_snippet = ""
    try:
        await response.dispose()
    except Exception:
        pass
    return _classify_probe(
        status=status,
        headers=headers,
        body_snippet=body_snippet,
        exception_name=None,
    )


def auth_self_test_path(site: str) -> str | None:
    """Return a cheap authenticated endpoint path for sites that need one."""
    return _AUTH_SELF_TEST_PATHS.get(str(site or "").strip().lower())


async def self_test_preflight_auth(
    *,
    request_context: Any,
    site: str,
    site_url: str,
    timeout_s: float = DEFAULT_PREFLIGHT_TIMEOUT_S,
) -> PreflightClassification | None:
    """Probe whether the current request context has live browser auth.

    Returns ``None`` for sites whose source-data preflight does not depend on
    authenticated browser state. For GitLab, ``reachable`` means the storage
    state is accepted; ``login_redirect``/``auth_missing`` means it is stale.
    """
    path = auth_self_test_path(site)
    if path is None:
        return None
    base = str(site_url or "").strip()
    if not base:
        return PreflightClassification(
            kind="host_unreachable",
            quarantine=False,
            http_status=None,
            detail="auth self-test has no site_url",
        )
    return await _probe_one(
        request_context=request_context,
        url=resolve_start_url(path, base),
        timeout_s=timeout_s,
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def preflight_benign_targets(
    tasks: list[dict[str, Any]],
    *,
    instances_by_site: dict[str, list[dict[str, Any]]],
    request_context_factory: Any,
    concurrency: int = DEFAULT_PREFLIGHT_CONCURRENCY,
    timeout_s: float = DEFAULT_PREFLIGHT_TIMEOUT_S,
    bailout_ratio: float = DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Probe every task's benign surface and split into (keep, dropped).

    ``request_context_factory`` is a callable that, given Playwright
    APIRequestContext options (``storage_state``,
    ``extra_http_headers``, ``http_credentials``), returns an awaitable
    resolving to a Playwright ``APIRequestContext``. Caller owns its
    lifetime; preflight never constructs Playwright directly.

    Returns a pair ``(tasks_to_probe, tasks_dropped)``. Dropped records
    carry a ``source_data_issue`` dict. Tasks with no
    ``benign_target_resource`` or no ``start_url_resolved`` pass
    through unchanged (the existing ``no_start_url`` path in
    verify_reachable will classify them).
    """
    if not tasks:
        return [], []

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    keep: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    login_redirect_count = 0
    probed_count = 0
    # Memoize one request context per (site_name, auth options). Cache the
    # creation task itself so concurrent probes for the same key do not all
    # miss the dict and create duplicate APIRequestContexts.
    context_tasks: dict[tuple[str, str], asyncio.Task[Any]] = {}

    def _context_key(options: dict[str, Any]) -> str:
        import hashlib
        import json

        public_options: dict[str, Any] = {}
        for key, value in options.items():
            if key.startswith("_"):
                continue
            if key == "storage_state" and isinstance(value, dict):
                encoded = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
                public_options[key] = {"sha256": hashlib.sha256(encoded).hexdigest()}
            else:
                public_options[key] = value
        return json.dumps(public_options, sort_keys=True, separators=(",", ":"), default=str)

    async def _context_for(site_name: str, options: dict[str, Any]) -> Any:
        key = (site_name, _context_key(options))
        task = context_tasks.get(key)
        if task is None:
            task = asyncio.create_task(request_context_factory(dict(options)))
            context_tasks[key] = task
        try:
            return await task
        except Exception:
            if context_tasks.get(key) is task:
                context_tasks.pop(key, None)
            raise

    dropped_originals: dict[int, dict[str, Any]] = {}

    async def _probe_task(task: dict[str, Any]) -> None:
        nonlocal login_redirect_count, probed_count
        site = str(task.get("site", "")).strip().lower()
        site_instances = instances_by_site.get(site) or []
        if not site_instances:
            keep.append(task)
            return

        # Probe every replica for this site. Replica-0 sometimes holds
        # legacy DB state that a fleet-wide reset did not touch (e.g. a
        # reddit forum anchor that was valid at seed-generation time
        # and still exists only on replica-0), while replicas 1..N
        # have the current baseline. We quarantine when a STRICT
        # MAJORITY of replicas agree on a deterministic-failure
        # outcome — that matches the editor's P2C selection distribution
        # (a task that's broken on 9/10 replicas will land on a broken
        # replica 90 % of the time and fail in-run regardless). Pure
        # unanimity is too strict and false-negatives the reddit-0
        # drift pattern.
        classifications_by_target: dict[int, list[PreflightClassification]] = {}
        target_audit: dict[int, ProbeTarget] = {}
        skipped_auth_reasons: list[str] = []
        for instance in site_instances:
            site_url = str(instance.get("site_url") or "").rstrip("/")
            targets = _task_probe_targets(task, site_url)
            if not targets:
                break
            auth_skip_reason = instance.get("preflight_auth_skip_reason")
            if isinstance(auth_skip_reason, str) and auth_skip_reason.strip():
                skipped_auth_reasons.append(auth_skip_reason.strip())
                continue
            context_options = instance.get("preflight_request_context")
            if not isinstance(context_options, dict):
                context_options = {}
            async with sem:
                request_context = await _context_for(site, context_options)
                for target_index, target in enumerate(targets):
                    target_audit.setdefault(target_index, target)
                    classification = await _probe_one(
                        request_context=request_context,
                        url=target.url,
                        timeout_s=timeout_s,
                    )
                    classifications_by_target.setdefault(target_index, []).append(classification)
        if not classifications_by_target:
            if skipped_auth_reasons:
                logger.warning(
                    "phase 2c preflight: skipping source-data probe for task %s because "
                    "all candidate instances had unusable auth: %s",
                    task.get("id"),
                    "; ".join(sorted(set(skipped_auth_reasons))),
                )
            keep.append(task)
            return

        probed_count += 1
        all_classifications = [
            classification
            for classifications in classifications_by_target.values()
            for classification in classifications
        ]
        if any(c.kind == "login_redirect" for c in all_classifications):
            login_redirect_count += 1

        # Majority rule: strict majority (> 50 %) of probed replicas
        # must classify as quarantine. Ties (50/50 exactly) pass
        # through to the real probe so we never quarantine on weak
        # evidence.
        selected: tuple[int, list[PreflightClassification]] | None = None
        for target_index, classifications in classifications_by_target.items():
            quarantine_classifications = [c for c in classifications if c.quarantine]
            quarantine_rate = len(quarantine_classifications) / len(classifications)
            if quarantine_rate > 0.5 and quarantine_classifications:
                selected = (target_index, quarantine_classifications)
                break

        if selected is not None:
            target_index, quarantine_classifications = selected
            target = target_audit[target_index]
            # Pick the most common quarantine-kind as the audit label
            # (ties broken by first-seen order) so the sidecar
            # surfaces the dominant failure signature.
            kind_counts: dict[str, int] = {}
            for c in quarantine_classifications:
                kind_counts[c.kind] = kind_counts.get(c.kind, 0) + 1
            dominant = max(quarantine_classifications, key=lambda c: kind_counts[c.kind])
            audit = dict(task)
            audit["source_data_issue"] = {
                "kind": dominant.kind,
                "http_status": dominant.http_status,
                "detail": dominant.detail,
                "probed_at": _now_iso(),
                "probed_url": target.url,
                "probe_source": target.source,
                "replicas_probed": len(classifications_by_target[target_index]),
                "replicas_agreeing": len(quarantine_classifications),
            }
            dropped.append(audit)
            dropped_originals[id(audit)] = task
        else:
            keep.append(task)

    await asyncio.gather(*(_probe_task(task) for task in tasks))

    # Whole-run bailout: if login_redirect dominates, a shared cookie
    # is expired and we should NOT quarantine any task. Restore all
    # dropped login_redirect tasks to keep and let the main probe path
    # run (it may still mark them infeasible, but that's the correct
    # bucket for retry).
    if probed_count and login_redirect_count / probed_count > bailout_ratio:
        logger.warning(
            "phase 2c preflight: login_redirect_rate=%.0f%% (%d/%d) exceeds "
            "bailout threshold %.0f%%; suspected storage_state expiry; "
            "restoring all dropped login_redirect tasks and skipping "
            "source-data quarantine for this run.",
            100.0 * login_redirect_count / probed_count,
            login_redirect_count,
            probed_count,
            100.0 * bailout_ratio,
        )
        # Partition BEFORE mutating. ``dropped`` holds the same dict
        # instances as ``restored``; popping source_data_issue in the
        # restore pass and then re-reading it to filter would KeyError.
        still_dropped: list[dict[str, Any]] = []
        restored: list[dict[str, Any]] = []
        for r in dropped:
            if r["source_data_issue"]["kind"] == "login_redirect":
                restored.append(r)
            else:
                still_dropped.append(r)
        for r in restored:
            r.pop("source_data_issue", None)
            keep.append(dropped_originals.get(id(r), r))
        dropped = still_dropped

    if dropped:
        kinds: dict[str, int] = {}
        for r in dropped:
            k = r["source_data_issue"]["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        logger.info(
            "phase 2c preflight: quarantined %d task(s) as source_data_issue — %s",
            len(dropped),
            ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
        )

    return keep, dropped
