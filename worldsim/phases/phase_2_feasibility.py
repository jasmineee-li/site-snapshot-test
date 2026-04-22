"""Phase 2c — adversarial-task feasibility verification.

Runs *after* Phase 2b (text fill) and *before* Phase 2's final ``complete``
checkpoint. Each adversarial task's ``adversarial_data_seed`` is POSTed
against a live dev instance via the existing editor layer. Tasks that 2xx
are tagged ``feasibility.status = "verified"`` with a fingerprint; tasks
that raise ``EditorError`` (or trip schema validation) are quarantined to
``adversarial_tasks.infeasible.json`` with ``feasibility.status =
"infeasible"`` and an error payload.

Design anchors: ``docs/handoffs/codex-handoff-phase-2c-feasibility-verification.md``
§3 (Design), §3.4 (per-task execution), §3.7 (idempotency truth table),
§4 (wiring).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldsim._async_utils import retrying
from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.editors import EDITOR_REGISTRY, EditorError
from worldsim.phases.phase_2_reachability import (
    ReachabilityOutcome,
    derive_second_witness,
    verify_reachable,
)
from worldsim.phases.phase_2_render_check import (
    RenderOutcome,
    render_signature,
    verify_seed_renders,
)
from worldsim.seeding import SeedCleanupHandle, UnboundTokenError, apply_data_seed_async

logger = logging.getLogger(__name__)

# Failpoint bases fired by ``write_json_atomic``. Callers wire these up so the
# crash-resume tests can interrupt each write.
FAILPOINT_DATASET = "phase_2.output.feasibility_dataset"
FAILPOINT_QUARANTINE = "phase_2.output.feasibility_quarantine"
FAILPOINT_REPORT = "phase_2.output.feasibility_report"

# Phase 2c render verification is a hard gate by default. The env-var opt-out
# is intended for unit tests that mock the seed flow and don't need a browser,
# and for development hosts without Playwright installed. Production runs MUST
# leave this unset — the 2026-04-21 Magento review-pending bug shipped 174
# tasks under the "verified == HTTP 2xx only" lie this gate now closes.
_SKIP_RENDER_CHECK_ENV = "WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK"


@dataclass(frozen=True)
class FeasibilityReport:
    """Aggregated outcome of a Phase 2c run.

    The caller in ``phase_2_injections`` is responsible for persisting the
    three artifacts; this dataclass is a pure value type.
    """

    verified: list[dict[str, Any]]
    infeasible: list[dict[str, Any]]
    skipped_already_verified: list[dict[str, Any]]
    cleanup_warnings: list[str]
    host_fingerprint: dict[str, str]
    elapsed_seconds: float
    per_site_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    phase_2_status: str | None = None


async def verify_feasibility(
    tasks_path: Path,
    *,
    instances: list[dict[str, Any]],
    instances_label: str = "instances.smoke.json",
    concurrency: int = 10,
    retry_count: int = 1,
    ttl_hours: float | None = None,
    force_reverify: bool = False,
    phase_2_status: str | None = None,
    stagger_delay: float = 0.0,
) -> FeasibilityReport:
    """Verify each adversarial task in ``tasks_path`` against a dev instance.

    Args:
        tasks_path: Path to the Phase 2 adversarial-tasks JSON array.
        instances: Per-site instance dicts (already extracted from the
            ``instances.smoke.json`` wrapper by the caller).
        instances_label: Basename of the instances file for fingerprinting.
        concurrency: Worker-pool size.
        retry_count: Per-task retry budget for transient EditorError kinds.
        ttl_hours: Skip re-verify when ``verified_at`` is newer than ``N``
            hours, even if the fingerprint drifts. Opt-in dev convenience.
        force_reverify: Re-verify every task regardless of fingerprint.
        phase_2_status: The ``status`` Phase 2b wrote into ``pipeline_state``
            (``"complete"`` or ``"partial_complete"``). Recorded in the report
            header so reviewers see the upstream qualifier.
        stagger_delay: Optional startup spread between workers.
    """
    raw = json.loads(tasks_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{tasks_path} must contain a JSON array of tasks; got {type(raw).__name__}"
        )

    fingerprint_base = _host_fingerprint(instances_label, instances)

    if not raw:
        logger.info("phase 2c: no tasks in %s; nothing to verify", tasks_path)
        return FeasibilityReport(
            verified=[],
            infeasible=[],
            skipped_already_verified=[],
            cleanup_warnings=[],
            host_fingerprint=fingerprint_base,
            elapsed_seconds=0.0,
            per_site_counts={},
            phase_2_status=phase_2_status,
        )

    token_errors = acquire_tokens_for_instances(instances)
    if token_errors:
        raise RuntimeError(
            "phase 2c pre-flight: token acquisition failed:\n  - " + "\n  - ".join(token_errors)
        )

    sites_in_tasks = {_resolve_seed_site(task) for task in raw}
    sites_in_tasks.discard("")
    sites_in_instances = {str(inst.get("site_name", "")).strip().lower() for inst in instances}
    missing_sites = sites_in_tasks - sites_in_instances
    if missing_sites:
        raise RuntimeError(
            "phase 2c pre-flight: tasks reference sites with no matching instance: "
            + ", ".join(sorted(missing_sites))
        )

    instance_by_site: dict[str, dict[str, Any]] = {}
    for inst in instances:
        name = str(inst.get("site_name", "")).strip().lower()
        if not name:
            continue
        instance_by_site[name] = inst

    for site, inst in instance_by_site.items():
        if site not in sites_in_tasks:
            continue
        benchmark = str(inst.get("benchmark") or "webarena_verified").strip().lower()
        editor_cls = EDITOR_REGISTRY.get((benchmark, site))
        if editor_cls is None:
            raise RuntimeError(
                f"phase 2c pre-flight: no editor registered for (benchmark={benchmark!r}, site={site!r})"
            )
        editor_cls.probe_base_state(inst)

    verified: list[dict[str, Any]] = []
    infeasible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    cleanup_warnings: list[str] = []
    per_site: dict[str, dict[str, int]] = {}

    semaphore = asyncio.Semaphore(max(1, concurrency))
    started = time.monotonic()
    now_iso = _now_iso()

    skip_render_check = os.getenv(_SKIP_RENDER_CHECK_ENV) == "1"
    async_playwright_factory: Any = None
    if skip_render_check:
        logger.warning(
            "%s=1 set; phase 2c render verification disabled. "
            "feasibility.status='verified' no longer guarantees the seeded "
            "payload renders on its read_surface_urls. Production runs MUST "
            "leave this unset.",
            _SKIP_RENDER_CHECK_ENV,
        )
    else:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "phase 2c render verification requires Playwright: install "
                "'playwright' and run 'playwright install chromium', or set "
                f"{_SKIP_RENDER_CHECK_ENV}=1 to opt out (development only). "
                f"Underlying import error: {exc!r}"
            ) from exc
        async_playwright_factory = async_playwright

    async def worker(task: dict[str, Any], index: int) -> dict[str, Any]:
        async with semaphore:
            if stagger_delay:
                await asyncio.sleep(min(stagger_delay * index, stagger_delay * 10))
            # Phase 4 binds the seed call to the *delivery* site (from
            # ``delivery_channel.delivery_site``, falling back to the first
            # editor_call's ``site``, falling back to ``task["site"]``).
            # Mirror that so cross-site adversarial seeds — e.g. a
            # shopping_admin task whose payload seeds a product review on
            # the shopping storefront — verify against the correct instance.
            seed_site = _resolve_seed_site(task)
            instance = instance_by_site.get(seed_site)
            if instance is None:
                return _infeasible_task(
                    task,
                    kind="unsupported_site",
                    detail=f"no instance for seed site {seed_site!r}",
                    fingerprint=fingerprint_base,
                    http_status=None,
                    response_snippet=None,
                    attempts=[],
                    timestamp=now_iso,
                )
            # Per-task browser: every worker gets its own Playwright +
            # Chromium pair, torn down before the next task runs. A render-
            # check crash in one task (renderer OOM, sandbox child death,
            # page crash that propagates to the browser) now stays inside
            # this worker — sibling tasks launch their own browsers and
            # are unaffected. This matches the pattern every peer
            # web-agent eval harness uses (WebArena, VisualWebArena,
            # BrowserGym, AgentLab, WASP) and the official pytest-
            # playwright idiom (browser scope = worker, context scope = task).
            # Launch cost is ~1.5-3s per task; at concurrency N that's
            # ~(launch_cost * total_tasks / N) added wall-clock, acceptable
            # vs the all-or-nothing shared-browser fragility.
            pw_handle: Any = None
            browser: Any = None
            try:
                if async_playwright_factory is not None:
                    pw_handle = await async_playwright_factory().start()
                    browser = await pw_handle.chromium.launch(headless=True)
                try:
                    return await _verify_one(
                        task,
                        instance,
                        retry_count=retry_count,
                        fingerprint_base=fingerprint_base,
                        ttl_hours=ttl_hours,
                        force_reverify=force_reverify,
                        cleanup_warnings=cleanup_warnings,
                        browser=browser,
                        render_semaphore=None,
                    )
                except Exception as exc:
                    task_id = str(task.get("id", "unknown"))
                    logger.exception("phase 2c verification crashed for task %s", task_id)
                    return _infeasible_task(
                        task,
                        kind="verification_crashed",
                        detail=f"{exc.__class__.__name__}: {exc}",
                        fingerprint=fingerprint_base,
                        http_status=None,
                        response_snippet=None,
                        attempts=[],
                        timestamp=now_iso,
                    )
            finally:
                if browser is not None:
                    try:
                        await browser.close()
                    except Exception:
                        logger.exception("phase 2c: failed to close per-task browser")
                if pw_handle is not None:
                    try:
                        await pw_handle.stop()
                    except Exception:
                        logger.exception("phase 2c: failed to stop per-task playwright handle")

    results = await asyncio.gather(
        *(worker(task, i) for i, task in enumerate(raw)),
        return_exceptions=False,
    )

    for result in results:
        feasibility = result.get("feasibility") or {}
        status = feasibility.get("status") if isinstance(feasibility, dict) else None
        site = str(result.get("site", "")).strip().lower() or "unknown"
        bucket = per_site.setdefault(
            site,
            {"verified": 0, "infeasible": 0, "skipped": 0},
        )
        if status == "verified":
            # Tasks reused via idempotency carry ``last_reverify_skipped_at``.
            # They stay in ``verified`` (that's the list persisted to disk and
            # admitted by Phase 4) but are *also* tracked in ``skipped`` for
            # the report breakdown.
            verified.append(result)
            if isinstance(feasibility, dict) and "last_reverify_skipped_at" in feasibility:
                skipped.append(result)
                bucket["skipped"] += 1
            else:
                bucket["verified"] += 1
        elif status == "infeasible":
            infeasible.append(result)
            bucket["infeasible"] += 1
        else:
            infeasible.append(result)
            bucket["infeasible"] += 1

    return FeasibilityReport(
        verified=verified,
        infeasible=infeasible,
        skipped_already_verified=skipped,
        cleanup_warnings=cleanup_warnings,
        host_fingerprint=fingerprint_base,
        elapsed_seconds=time.monotonic() - started,
        per_site_counts=per_site,
        phase_2_status=phase_2_status,
    )


async def _verify_one(
    task: dict[str, Any],
    instance: dict[str, Any],
    *,
    retry_count: int,
    fingerprint_base: dict[str, str],
    ttl_hours: float | None,
    force_reverify: bool,
    cleanup_warnings: list[str],
    browser: Any = None,
    render_semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    seed = task.get("adversarial_data_seed") or {}
    editor_calls = seed.get("editor_calls") if isinstance(seed, dict) else None

    content_hash = _task_content_hash(editor_calls if isinstance(editor_calls, list) else [])
    fingerprint = dict(fingerprint_base)
    fingerprint["task_content_hash"] = content_hash

    decision, skip_reason = _idempotency_decision(
        task.get("feasibility"),
        current_fingerprint=fingerprint,
        ttl_hours=ttl_hours,
        force_reverify=force_reverify,
    )
    if decision == "skip":
        # Preserve the prior ``status="verified"`` record verbatim — Phase 4's
        # strict admission gate only admits ``status == "verified"``, so
        # overwriting to ``"skipped"`` would silently take prior verifications
        # offline on every idempotent re-run. We record the skip fact on a
        # sibling field so the report bucket still picks it out.
        result = dict(task)
        prior = dict(task.get("feasibility") or {})
        prior["last_reverify_skipped_at"] = _now_iso()
        prior["last_reverify_skip_reason"] = skip_reason or "fingerprint_match"
        result["feasibility"] = prior
        return result

    if not isinstance(editor_calls, list):
        return _infeasible_task(
            task,
            kind="schema_mismatch",
            detail="adversarial_data_seed missing editor_calls list",
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=[],
            timestamp=_now_iso(),
        )

    bound_instance = dict(instance)
    bound_instance["seed_task"] = task

    attempts: list[dict[str, Any]] = []
    handle: SeedCleanupHandle | None = None
    metadata: dict[str, Any] = {}

    async def _apply_and_keep_metadata() -> tuple[SeedCleanupHandle | None, dict[str, Any]]:
        return await apply_data_seed_async(seed, bound_instance)

    try:
        handle, metadata = await retrying(
            _apply_and_keep_metadata,
            retries=retry_count,
            attempts_log=attempts,
        )
    except EditorError as exc:
        _safe_cleanup(handle, cleanup_warnings, task.get("id"))
        return _infeasible_task(
            task,
            kind=exc.kind,
            detail=exc.detail,
            fingerprint=fingerprint,
            http_status=exc.http_status,
            response_snippet=exc.response_snippet,
            attempts=attempts,
            timestamp=_now_iso(),
        )
    except UnboundTokenError as exc:
        # Phantom {benign_*} token — the seed referenced a token the
        # resolver's anchors don't support. Categorized separately from
        # schema_mismatch so dashboards can track the commit 4/6 fail-
        # loud contract hits distinct from shape violations.
        _safe_cleanup(handle, cleanup_warnings, task.get("id"))
        return _infeasible_task(
            task,
            kind="contract_violation",
            detail=str(exc),
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
        )
    except (ValueError, RuntimeError) as exc:
        # ValueError comes from validate_data_seed; RuntimeError comes from
        # ``_render_editor_seed_call`` when a template placeholder (e.g.
        # ``{submission_id}``) can't be resolved because the chain is
        # missing a producer call. Both are structural problems; neither is
        # a platform rejection.
        _safe_cleanup(handle, cleanup_warnings, task.get("id"))
        return _infeasible_task(
            task,
            kind="schema_mismatch",
            detail=str(exc),
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
        )

    if handle is None:
        # Empty seed never registered a cleanup handle, so no cleanup needed.
        return _infeasible_task(
            task,
            kind="empty_seed",
            detail="adversarial_data_seed produced no editor calls",
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
        )

    # Render check runs BEFORE cleanup because cleanup deletes the seeded
    # row. The 2026-04-21 Magento bug shipped because Phase 2c stamped
    # ``verified`` on HTTP 2xx alone — Layer 2 of the long-term fix closes
    # that contract gap. ``browser is None`` only when the operator opted
    # out via WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK=1; in that case the
    # ``verified`` stamp regresses to the pre-Layer-2 meaning ("API write
    # succeeded only").
    render_outcome: RenderOutcome | None = None
    reachability_outcome: ReachabilityOutcome | None = None
    try:
        if browser is not None:
            render_outcome = await _run_render_check(
                browser=browser,
                render_semaphore=render_semaphore,
                seed=seed,
                metadata=metadata,
                instance=instance,
            )
            if render_outcome is not None and render_outcome.ok:
                # Option A reachability only applies to tasks whose benign
                # target resource is known — legacy datasets without the
                # field are skipped so this commit doesn't regress them.
                resource = task.get("benign_target_resource")
                if isinstance(resource, dict) and resource.get("kind") is not None:
                    reachability_outcome = await _run_reachability_check(
                        browser=browser,
                        render_semaphore=render_semaphore,
                        task=task,
                        seed=seed,
                        metadata=metadata,
                        instance=instance,
                        render_outcome=render_outcome,
                    )
    finally:
        _safe_cleanup(handle, cleanup_warnings, task.get("id"))

    if render_outcome is not None and not render_outcome.ok:
        return _infeasible_task(
            task,
            kind=render_outcome.kind,
            detail=render_outcome.detail,
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
            render_evidence=render_outcome.evidence(),
        )

    if reachability_outcome is not None and reachability_outcome.reachability == "unreachable":
        return _infeasible_task(
            task,
            kind=f"reachability_{reachability_outcome.kind}" or "reachability_failed",
            detail=reachability_outcome.detail,
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
            render_evidence=(render_outcome.evidence() if render_outcome else None),
            reachability_evidence=reachability_outcome.evidence(),
        )

    result = dict(task)
    feasibility: dict[str, Any] = {
        "status": "verified",
        "verified_at": _now_iso(),
        "host_fingerprint": fingerprint,
        "attempts": attempts,
    }
    if render_outcome is not None:
        feasibility["render_verified"] = True
        feasibility["render_evidence"] = render_outcome.evidence()
    if reachability_outcome is not None:
        feasibility["reachability"] = reachability_outcome.reachability
        feasibility["reachability_evidence"] = reachability_outcome.evidence()
    result["feasibility"] = feasibility
    return result


async def _run_reachability_check(
    *,
    browser: Any,
    render_semaphore: asyncio.Semaphore | None,
    task: dict[str, Any],
    seed: dict[str, Any],
    metadata: dict[str, Any],
    instance: dict[str, Any],
    render_outcome: RenderOutcome,
) -> ReachabilityOutcome:
    """Run the Option A reachability probe guarded by the same semaphore.

    Signature is taken from the already-successful render outcome so
    the two checks grep for the same string. Second witness is derived
    from the seed body text via
    ``phase_2_reachability.derive_second_witness``.
    """
    benign_target_resource = task.get("benign_target_resource")
    site_url = str(instance.get("site_url") or "").strip()
    signature = render_outcome.matched_signature or render_signature(seed)
    rendered_payload = _first_rendered_payload(seed)
    second_witness = derive_second_witness(rendered_payload, signature)
    storage_state_path = _resolve_benign_storage_state_path(instance)

    async def _do() -> ReachabilityOutcome:
        try:
            return await verify_reachable(
                browser=browser,
                benign_target_resource=benign_target_resource,
                instance_site_url=site_url,
                signature=signature,
                second_witness=second_witness,
                storage_state_path=storage_state_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("phase 2c reachability probe crashed")
            return ReachabilityOutcome.unreachable(
                kind="probe_raised",
                detail=f"{exc.__class__.__name__}: {exc}",
                url=site_url,
            )

    if render_semaphore is None:
        return await _do()
    async with render_semaphore:
        return await _do()


def _first_rendered_payload(seed: dict[str, Any]) -> str | None:
    """Extract the first rendered payload string from a seed's editor_calls.

    Used as the source pool for the reachability probe's second witness.
    Falls through to None when the seed has no textual args.
    """
    if not isinstance(seed, dict):
        return None
    calls = seed.get("editor_calls")
    if not isinstance(calls, list):
        return None
    for call in calls:
        if not isinstance(call, dict):
            continue
        args = call.get("args")
        if not isinstance(args, dict):
            continue
        for value in args.values():
            if isinstance(value, str) and len(value) >= 20:
                return value
    return None


def _resolve_benign_storage_state_path(instance: dict[str, Any]) -> str | None:
    """Return the Phase-0d-bootstrapped storage_state.json path for this site.

    Under Option A α identity the seed writer and the reachability
    probe both act as the benign user, so threading those cookies into
    Playwright lets the probe reach private projects + authed-only
    pages. Falls back to ``None`` when no artifact is present (public
    content still works in an anonymous context).
    """
    explicit = instance.get("storage_state_path")
    if isinstance(explicit, str) and explicit.strip():
        path = Path(explicit.strip())
        return str(path) if path.exists() else None

    state_dir_env = os.environ.get("WORLDSIM_STATE_DIR") or "logs"
    site_name = str(instance.get("site_name") or "").strip()
    if not site_name:
        return None
    candidate = Path(state_dir_env) / "phase_0d" / site_name / "storage_state.json"
    return str(candidate) if candidate.exists() else None


async def _run_render_check(
    *,
    browser: Any,
    render_semaphore: asyncio.Semaphore | None,
    seed: dict[str, Any],
    metadata: dict[str, Any],
    instance: dict[str, Any],
) -> RenderOutcome:
    urls = metadata.get("read_surface_urls") if isinstance(metadata, dict) else None
    if not isinstance(urls, list):
        urls = []
    site_name = str(instance.get("site_name", "")).strip().lower()
    site_url = str(instance.get("site_url", "")).rstrip("/")
    signature = render_signature(seed, metadata)

    storage_state_path = _resolve_benign_storage_state_path(instance)

    async def _do() -> RenderOutcome:
        try:
            return await verify_seed_renders(
                browser=browser,
                urls=urls,
                site_name=site_name,
                site_url=site_url,
                signature=signature,
                storage_state_path=storage_state_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("phase 2c render check crashed")
            return RenderOutcome.failed(
                kind="render_check_error",
                detail=f"render check raised {exc.__class__.__name__}: {exc}",
                urls_tried=urls,
                per_url_errors={},
            )

    if render_semaphore is None:
        return await _do()
    async with render_semaphore:
        return await _do()


def _infeasible_task(
    task: dict[str, Any],
    *,
    kind: str,
    detail: str,
    fingerprint: dict[str, str],
    http_status: int | None,
    response_snippet: str | None,
    attempts: list[dict[str, Any]],
    timestamp: str,
    render_evidence: dict[str, Any] | None = None,
    reachability_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(task)
    error_entry: dict[str, Any] = {
        "call_index": 0,
        "method": _first_method(task),
        "kind": kind,
        "detail": detail,
    }
    if http_status is not None:
        error_entry["http_status"] = http_status
    if response_snippet is not None:
        error_entry["response_snippet"] = response_snippet
    if render_evidence is not None:
        error_entry["render_evidence"] = render_evidence
    if reachability_evidence is not None:
        error_entry["reachability_evidence"] = reachability_evidence
    result["feasibility"] = {
        "status": "infeasible",
        "host_fingerprint": fingerprint,
        "errors": [error_entry],
        "first_failed_at": timestamp,
        "attempts": attempts,
    }
    return result


def _safe_cleanup(
    handle: SeedCleanupHandle | None,
    cleanup_warnings: list[str],
    task_id: Any,
) -> None:
    if handle is None:
        return
    try:
        handle.cleanup()
    except EditorError as exc:
        cleanup_warnings.append(f"task={task_id!s} cleanup_failed: {exc.detail}")
    except Exception as exc:  # pragma: no cover - defensive
        cleanup_warnings.append(f"task={task_id!s} cleanup_raised: {exc.__class__.__name__}: {exc}")


def _idempotency_decision(
    existing: Any,
    *,
    current_fingerprint: dict[str, str],
    ttl_hours: float | None,
    force_reverify: bool,
) -> tuple[str, str | None]:
    """Return ``("skip", reason)`` to reuse the prior result or
    ``("verify", None)`` to re-run.

    Matches the truth table in §3.7:

    - Missing feasibility field → verify.
    - ``verified`` + fingerprint matches → skip (reason=fingerprint_match);
      force_reverify overrides.
    - ``verified`` + fingerprint drifts → re-verify unless TTL covers it
      (reason=ttl_hours).
    - ``infeasible`` (any fingerprint) → re-verify (platform may have
      changed its policy since).
    - ``unverified`` → verify.
    """
    if force_reverify:
        return ("verify", None)
    if not isinstance(existing, dict):
        return ("verify", None)
    status = existing.get("status")
    if status != "verified":
        return ("verify", None)
    prior_fp = existing.get("host_fingerprint") or {}
    if not isinstance(prior_fp, dict):
        return ("verify", None)
    if _fingerprints_match(prior_fp, current_fingerprint):
        return ("skip", "fingerprint_match")
    if ttl_hours is not None:
        verified_at = existing.get("verified_at")
        age = _hours_since(verified_at)
        if age is not None and age <= ttl_hours:
            return ("skip", "ttl_hours")
    return ("verify", None)


def _fingerprints_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = (
        "host_config",
        "instances_digest",
        "editor_commit",
        "dataset_commit",
        "task_content_hash",
    )
    return all(str(a.get(k, "")) == str(b.get(k, "")) for k in keys)


def _hours_since(timestamp: Any) -> float | None:
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    delta = datetime.now(tz=UTC) - parsed
    return delta.total_seconds() / 3600.0


def _task_content_hash(editor_calls: list[Any]) -> str:
    canonical = json.dumps(editor_calls, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _host_fingerprint(instances_label: str, instances: list[dict[str, Any]]) -> dict[str, str]:
    commit = _git_head_short()
    return {
        "host_config": instances_label,
        "instances_digest": _instances_digest(instances),
        "editor_commit": commit,
        "dataset_commit": commit,
    }


def _instances_digest(instances: list[dict[str, Any]]) -> str:
    canonical = json.dumps(instances, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _git_head_short() -> str:
    override = os.environ.get("WORLDSIM_EDITOR_COMMIT_OVERRIDE")
    if override:
        return override.strip()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


def _first_method(task: dict[str, Any]) -> str:
    seed = task.get("adversarial_data_seed") or {}
    if not isinstance(seed, dict):
        return ""
    calls = seed.get("editor_calls")
    if isinstance(calls, list) and calls and isinstance(calls[0], dict):
        return str(calls[0].get("method", ""))
    return ""


def _resolve_seed_site(task: dict[str, Any]) -> str:
    """Return the site the adversarial seed actually POSTs against.

    Phase 4 uses the same precedence: ``delivery_channel.delivery_site`` →
    first editor_call's ``site`` → ``task["site"]``. A shopping_admin task
    whose payload seeds a product review on the shopping storefront has
    ``delivery_site="shopping"`` and must bind to the shopping instance.
    """
    delivery = task.get("delivery_channel")
    if isinstance(delivery, dict):
        ds = delivery.get("delivery_site")
        if isinstance(ds, str) and ds.strip() and ds.strip().lower() != "none":
            return ds.strip().lower()
    seed = task.get("adversarial_data_seed") or {}
    if isinstance(seed, dict):
        calls = seed.get("editor_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    cs = call.get("site")
                    if isinstance(cs, str) and cs.strip():
                        return cs.strip().lower()
    return str(task.get("site", "")).strip().lower()


def _now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def skipped_task_stanza(
    task: dict[str, Any], *, reason: str = "skip_feasibility_flag"
) -> dict[str, Any]:
    """Tag ``task`` with an ``unverified`` feasibility stanza (used by
    ``--skip-feasibility``)."""
    result = dict(task)
    result["feasibility"] = {
        "status": "unverified",
        "skipped_at": _now_iso(),
        "reason": reason,
    }
    return result
