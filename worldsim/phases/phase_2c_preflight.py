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
import hashlib
import json
import logging
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from worldsim.benchmark_capabilities import normalize_benchmark_name
from worldsim.phase_2c.policy import (
    PreflightClassification,
    ProbeTarget,
    get_feasibility_policy,
)
from worldsim.phase_2c.webarena import (
    DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO,
    WebArenaFeasibilityPolicy,
    classify_webarena_probe,
    clean_gitlab_project_path,
    dedupe_targets,
    editor_surface_path,
    first_value,
    location_is_login,
    looks_like_login_stub,
    render_anchor_tokens,
    task_probe_url,
)
from worldsim.phases.phase_2_reachability import resolve_start_url

logger = logging.getLogger(__name__)

# Default knob values. The concurrency sits safely inside the replica's
# 16-HTTP-slot puma budget; the timeout is shorter than the real probe's
# 30 s so a transient timeout here does NOT quarantine.
DEFAULT_PREFLIGHT_CONCURRENCY = 16
DEFAULT_PREFLIGHT_TIMEOUT_S = 5.0


def _looks_like_login_stub(body: str) -> bool:
    return looks_like_login_stub(body)


def _location_is_login(location: str | None) -> bool:
    return location_is_login(location)


def _classify_probe(
    *,
    status: int | None,
    headers: dict[str, str] | None,
    body_snippet: str,
    exception_name: str | None,
) -> PreflightClassification:
    return classify_webarena_probe(
        status=status,
        headers=headers,
        body_snippet=body_snippet,
        exception_name=exception_name,
    )


def _task_probe_url(task: dict[str, Any], instance_site_url: str) -> str | None:
    return task_probe_url(task, instance_site_url)


def _render_anchor_tokens(value: Any, anchors: dict[str, Any]) -> str | None:
    return render_anchor_tokens(value, anchors)


def _first_value(
    args: dict[str, Any],
    anchors: dict[str, Any],
    *names: str,
) -> str | None:
    return first_value(args, anchors, *names)


def _clean_gitlab_project_path(project_path: str) -> str:
    return clean_gitlab_project_path(project_path)


def _editor_surface_path(
    *,
    site: str,
    method: str,
    args: dict[str, Any],
    anchors: dict[str, Any],
) -> str | None:
    return editor_surface_path(site=site, method=method, args=args, anchors=anchors)


def _task_probe_targets(task: dict[str, Any], instance_site_url: str) -> list[ProbeTarget]:
    return WebArenaFeasibilityPolicy(site=str(task.get("site") or "")).probe_targets(
        task, instance_site_url
    )


def _dedupe_targets(targets: list[ProbeTarget]) -> list[ProbeTarget]:
    return dedupe_targets(targets)


async def _probe_one(
    *,
    request_context: Any,
    url: str,
    timeout_s: float,
    policy: Any | None = None,
) -> PreflightClassification:
    try:
        response = await request_context.get(
            url,
            timeout=timeout_s * 1000.0,
            max_redirects=0,
        )
    except Exception as exc:
        classifier = policy.classify_probe if policy is not None else _classify_probe
        return classifier(
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
    classifier = policy.classify_probe if policy is not None else _classify_probe
    return classifier(
        status=status,
        headers=headers,
        body_snippet=body_snippet,
        exception_name=None,
    )


def auth_self_test_path(site: str) -> str | None:
    """Return a cheap authenticated endpoint path for sites that need one."""
    policy = get_feasibility_policy("webarena_verified", str(site or "").strip().lower())
    return policy.auth_self_test_path() if policy is not None else None


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
        policy=get_feasibility_policy("webarena_verified", str(site or "").strip().lower()),
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _benchmark_for_preflight(task: dict[str, Any], site_instances: list[dict[str, Any]]) -> str:
    for key in ("benchmark", "benchmark_name", "benchmark_adapter"):
        raw = task.get(key)
        if isinstance(raw, str) and raw.strip():
            return normalize_benchmark_name(raw)
    seed = task.get("adversarial_data_seed")
    calls = seed.get("editor_calls") if isinstance(seed, dict) else None
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            raw = call.get("benchmark")
            if isinstance(raw, str) and raw.strip():
                return normalize_benchmark_name(raw)
    for instance in site_instances:
        for key in ("benchmark", "benchmark_name", "benchmark_adapter"):
            raw = instance.get(key)
            if isinstance(raw, str) and raw.strip():
                return normalize_benchmark_name(raw)
    # Compatibility for direct unit tests that predate benchmark metadata.
    return "webarena_verified"


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
    policies_by_key: dict[tuple[str, str], Any] = {}
    bailout_counts: dict[tuple[str, str], int] = {}
    probed_counts: dict[tuple[str, str], int] = {}

    async def _probe_task(task: dict[str, Any]) -> None:
        nonlocal login_redirect_count, probed_count
        site = str(task.get("site", "")).strip().lower()
        site_instances = instances_by_site.get(site) or []
        if not site_instances:
            keep.append(task)
            return
        benchmark = _benchmark_for_preflight(task, site_instances)
        policy = get_feasibility_policy(benchmark, site)
        if policy is None:
            keep.append(task)
            return
        policy_key = (benchmark, site)
        policies_by_key[policy_key] = policy

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
            targets = policy.probe_targets(task, site_url)
            if not targets:
                break
            auth_skip_reason = instance.get("preflight_auth_skip_reason")
            if isinstance(auth_skip_reason, str) and auth_skip_reason.strip():
                skipped_auth_reasons.append(auth_skip_reason.strip())
                continue
            context_options = instance.get("preflight_request_context")
            if (
                policy.requires_authenticated_preflight()
                and (not isinstance(context_options, dict) or not context_options)
            ):
                skipped_auth_reasons.append(
                    "authenticated source-data preflight required but no usable auth was configured"
                )
                continue
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
                        policy=policy,
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
        if any(policy.counts_toward_run_bailout(c) for c in all_classifications):
            login_redirect_count += 1
            bailout_counts[policy_key] = bailout_counts.get(policy_key, 0) + 1
        probed_counts[policy_key] = probed_counts.get(policy_key, 0) + 1

        decision = policy.decide_source_data(
            task=task,
            classifications_by_target=classifications_by_target,
            target_audit=target_audit,
            login_redirect_count=login_redirect_count,
            probed_count=probed_count,
            bailout_ratio=bailout_ratio,
        )
        if decision.action == "drop" and decision.classification is not None and decision.target:
            dominant = decision.classification
            target = decision.target
            audit = dict(task)
            audit["source_data_issue"] = {
                "kind": dominant.kind,
                "http_status": dominant.http_status,
                "detail": dominant.detail,
                "probed_at": _now_iso(),
                "probed_url": _redact_probe_url(target.url),
                "probe_source": target.source,
                **decision.evidence,
            }
            dropped.append(audit)
            dropped_originals[id(audit)] = task
        elif decision.action == "bailout":
            keep.append(task)
        else:
            keep.append(task)

    await asyncio.gather(*(_probe_task(task) for task in tasks))

    # Whole-run bailout remains policy-owned: a site policy decides which
    # classifications count and which dropped records should be restored.
    bailout_policy_keys = {
        key
        for key, policy in policies_by_key.items()
        if policy.should_bailout_source_data_run(
            bailout_count=bailout_counts.get(key, 0),
            probed_count=probed_counts.get(key, 0),
            bailout_ratio=bailout_ratio,
        )
    }
    if bailout_policy_keys:
        total_bailout = sum(bailout_counts.get(key, 0) for key in bailout_policy_keys)
        total_probed = sum(probed_counts.get(key, 0) for key in bailout_policy_keys)
        logger.warning(
            "phase 2c preflight: policy bailout rate=%.0f%% (%d/%d) exceeds "
            "bailout threshold %.0f%%; suspected storage_state expiry; "
            "restoring policy-selected source-data drops and skipping "
            "source-data quarantine for this run.",
            100.0 * total_bailout / total_probed if total_probed else 0.0,
            total_bailout,
            total_probed,
            100.0 * bailout_ratio,
        )
        # Partition BEFORE mutating. ``dropped`` holds the same dict
        # instances as ``restored``; popping source_data_issue in the
        # restore pass and then re-reading it to filter would KeyError.
        still_dropped: list[dict[str, Any]] = []
        restored: list[dict[str, Any]] = []
        for record in dropped:
            issue = record.get("source_data_issue") if isinstance(record, dict) else None
            record_site = str(record.get("site") or "").strip().lower()
            record_benchmark = _benchmark_for_preflight(
                record,
                instances_by_site.get(record_site) or [],
            )
            record_key = (record_benchmark, record_site)
            policy = policies_by_key.get(record_key)
            if (
                record_key in bailout_policy_keys
                and policy is not None
                and isinstance(issue, dict)
                and policy.restore_drop_on_run_bailout(issue)
            ):
                restored.append(record)
            else:
                still_dropped.append(record)
        for record in restored:
            record.pop("source_data_issue", None)
            keep.append(dropped_originals.get(id(record), record))
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


def _redact_probe_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    netloc = host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    raw_query = parse_qs(parsed.query, keep_blank_values=True)
    redacted_query = urlencode(
        {str(key): ["<redacted>"] * len(values) for key, values in raw_query.items()},
        doseq=True,
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, redacted_query, ""))
