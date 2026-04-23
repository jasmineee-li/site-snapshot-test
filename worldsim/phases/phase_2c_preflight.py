"""Phase 2c preflight — HTTP-probe ``benign_target_resource.start_url_resolved``
against the benign user's storage_state and quarantine tasks whose benign
entry point is deterministically broken.

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

Probe mechanism: a Playwright ``APIRequestContext`` initialized from the
benign storage_state (reuses ``_resolve_benign_storage_state_path``).
GET with ``max_redirects=0`` and a 5 s timeout; read the first 8 KB of
body. Classifier table (see ``_classify_probe``) translates HTTP
outcome to a stable ``source_data_issue.kind`` label.

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
import time
from dataclasses import dataclass
from typing import Any

from worldsim.phases.phase_2_reachability import resolve_start_url

logger = logging.getLogger(__name__)

# Default knob values. The concurrency sits safely inside the replica's
# 16-HTTP-slot puma budget; the timeout is shorter than the real probe's
# 30 s so a transient timeout here does NOT quarantine.
DEFAULT_PREFLIGHT_CONCURRENCY = 16
DEFAULT_PREFLIGHT_TIMEOUT_S = 5.0
DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO = 0.5

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
        location = None
        if headers:
            location = headers.get("location") or headers.get("Location")
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
            headers = {str(k): str(v) for k, v in raw_headers.items()}
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

    ``request_context_factory`` is a callable that, given a
    ``storage_state_path`` string (or None), returns an awaitable
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
    # Memoize one request context per (site_name, storage_state_path).
    contexts: dict[tuple[str, str | None], Any] = {}

    async def _context_for(site_name: str, storage_state_path: str | None) -> Any:
        key = (site_name, storage_state_path)
        if key not in contexts:
            contexts[key] = await request_context_factory(storage_state_path)
        return contexts[key]

    async def _probe_task(task: dict[str, Any]) -> None:
        nonlocal login_redirect_count, probed_count
        site = str(task.get("site", "")).strip().lower()
        site_instances = instances_by_site.get(site) or []
        if not site_instances:
            keep.append(task)
            return
        instance = site_instances[0]
        site_url = str(instance.get("site_url") or "").rstrip("/")
        url = _task_probe_url(task, site_url)
        if not url:
            keep.append(task)
            return
        storage_state_path = instance.get("storage_state_path") or None
        async with sem:
            request_context = await _context_for(site, storage_state_path)
            classification = await _probe_one(
                request_context=request_context,
                url=url,
                timeout_s=timeout_s,
            )
        probed_count += 1
        if classification.kind == "login_redirect":
            login_redirect_count += 1
        if classification.quarantine:
            audit = dict(task)
            audit["source_data_issue"] = {
                "kind": classification.kind,
                "http_status": classification.http_status,
                "detail": classification.detail,
                "probed_at": _now_iso(),
                "probed_url": url,
            }
            dropped.append(audit)
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
            keep.append(r)
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
