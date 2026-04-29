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
from worldsim.agent_auth import (
    cookie_domain_matches_host,
    playwright_storage_state,
    playwright_storage_state_payload,
    read_storage_state_payload,
    resolve_agent_auth,
    resolve_agent_auth_headers,
    resolve_storage_state_path,
    storage_state_cookie_hosts,
    storage_state_origin_hosts,
    storage_state_preflight_error_for_payload,
    storage_state_recorded_hosts,
)
from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.benchmark_capabilities import (
    get_benchmark_capabilities,
    infer_benchmark_name,
    normalize_benchmark_name,
)
from worldsim.editors import EDITOR_REGISTRY, EditorError
from worldsim.instance_selection import (
    _ordered_instance_dicts,
    replica_key,
    select_task_site_instance_dict_p2c,
)
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
FAILPOINT_DROPPED_SOURCE_DATA = "phase_2.output.feasibility_dropped_source_data"

# Phase 2c render verification is a hard gate by default. The env-var opt-out
# is intended for unit tests that mock the seed flow and don't need a browser,
# and for development hosts without Playwright installed. Production runs MUST
# leave this unset — the 2026-04-21 Magento review-pending bug shipped 174
# tasks under the "verified == HTTP 2xx only" lie this gate now closes.
_SKIP_RENDER_CHECK_ENV = "WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK"

# Per-replica in-flight caps for Phase 2c. Derived from live container
# inspection on 2026-04-22 (webarena-verified-gitlab_3): puma runs 4
# workers x 4 threads = 16 HTTP slots with worker_timeout 60 s. Cap 10
# is ~60 % of puma capacity, leaving 6 slots for the internal API
# traffic a GitLab write fans out into (NotificationsService, mentions,
# TodoService, webhooks). Reddit (Postmill / PHP-fpm) is not yet
# measured; 8 is a mid-range placeholder pending Layer 5 observability.
_PER_REPLICA_CAP_DEFAULT: dict[str, int] = {"gitlab": 10, "reddit": 8}
_PER_REPLICA_CAP_FALLBACK = 6

# Global browser-probe cap. Per-replica caps bulkhead against the
# *backend*; this one bulkheads against the *client*. GitLab ships
# deferred JS that fights for CPU; 64-wide renderers on r5.4xlarge
# starved the scheduler enough to trip the 30 s ``domcontentloaded``
# timeout even on a healthy replica returning in <1.4 s. Capping the
# number of concurrent Chromium processes globally gives each renderer
# ~2 vCPU headroom and eliminates the nav-failed tail.
_BROWSER_PROBE_CAP = 8
_PREFLIGHT_AUTH_REFRESH_LOCKS: dict[str, asyncio.Lock] = {}

# RenderOutcome.kind value the render-check sets when the seed wrote OK
# but the signature was not visible in any read-surface URL within the
# body-poll window. Hoisted as a constant so the retry-once gate below
# stays grep-friendly and tests can pin the kind without importing
# verify_seed_renders' classifier internals.
_RENDER_UNVERIFIED_KIND = "render_unverified"
# Single retry breather for the render-check on its first miss. Targets
# GitLab's slow write-to-visible tail (sidekiq + page-cache invalidation)
# under Phase 2c's 16-way renderer contention. 3 s is short enough not
# to balloon Phase 2c wall time on the 1-3 task-per-run flake band, long
# enough to clear the typical sidekiq queue depth observed on r5.
_RENDER_UNVERIFIED_RETRY_DELAY_S = 3.0

# Launch flags applied to every per-task Chromium. ``--disable-dev-shm-usage``
# moves shared memory from ``/dev/shm`` (64 MiB Docker default) to
# ``/tmp``; harmless on bare-metal Linux and essential in containers —
# Playwright issue #22676 documents shm OOM surfacing as nav timeouts.
# ``--disable-gpu`` drops the GPU process per renderer under headless.
# ``--no-sandbox`` is acceptable here because Phase 2c hits only the
# internal WASP replicas (``http://172.17.0.1:8xxx``) whose content we
# control; it is the standard recommendation for containerized CI and
# avoids the /proc/*/ns/user setup cost per launch.
_PROBE_LAUNCH_ARGS: tuple[str, ...] = (
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-sandbox",
)


def _per_replica_cap(site_name: str) -> int:
    return _PER_REPLICA_CAP_DEFAULT.get(site_name.strip().lower(), _PER_REPLICA_CAP_FALLBACK)


@dataclass
class _ReplicaStats:
    """Per-replica observability counters for a Phase 2c run.

    Lives entirely in-process; logged as a single line per replica at
    end of run. Cheap enough to leave always-on — this is the data a
    future AIMD wrapper (or manual cap tuning) needs; shipping it here
    avoids guessing from dmesg and nginx error logs next time.
    """

    site_name: str
    replica_name: str
    requests: int = 0
    errors: int = 0
    in_flight_peak: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, *, elapsed_ms: float, ok: bool) -> None:
        self.requests += 1
        if not ok:
            self.errors += 1
        # Cap the sample list so long runs do not balloon memory; 2048
        # samples is plenty for p50/p99 estimation within ±1 %.
        if len(self.latencies_ms) < 2048:
            self.latencies_ms.append(elapsed_ms)

    def summary(self) -> str:
        if not self.latencies_ms:
            return (
                f"replica={self.replica_name} site={self.site_name} "
                f"requests={self.requests} errors={self.errors} "
                f"in_flight_peak={self.in_flight_peak} latency_ms=<none>"
            )
        ordered = sorted(self.latencies_ms)
        n = len(ordered)
        p50 = ordered[n // 2]
        p99 = ordered[min(n - 1, (n * 99) // 100)]
        return (
            f"replica={self.replica_name} site={self.site_name} "
            f"requests={self.requests} errors={self.errors} "
            f"in_flight_peak={self.in_flight_peak} "
            f"p50_ms={p50:.0f} p99_ms={p99:.0f}"
        )


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
    # Bug I: tasks whose benign_target_resource preflight deterministically
    # failed (login_redirect, 404, 403, 410, 401). Separate from
    # ``infeasible`` because they are dataset-quality issues, not probe
    # failures; re-running Phase 2c will not rehabilitate them.
    dropped_source_data: list[dict[str, Any]] = field(default_factory=list)


async def verify_feasibility(
    tasks_path: Path,
    *,
    instances: list[dict[str, Any]],
    instances_label: str = "instances.smoke.json",
    benchmark_root: Path | None = None,
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
        benchmark_root: Optional benchmark codebase root for Phase 0d
            generator_script resolution during storage_state repair.
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

    task_benchmark = _infer_records_benchmark(raw, label="Phase 2c tasks")
    instance_benchmark = _infer_records_benchmark(instances, label="Phase 2c instances")
    if task_benchmark != instance_benchmark:
        raise RuntimeError(
            "phase 2c pre-flight: mixed benchmark metadata between tasks and instances: "
            f"tasks={task_benchmark!r}, instances={instance_benchmark!r}"
        )
    capabilities = get_benchmark_capabilities(task_benchmark)
    if not capabilities.phase_2_feasibility_supported:
        raise RuntimeError(
            f"phase 2c pre-flight: benchmark {task_benchmark!r} does not support "
            "WorldSim v5 Phase 2c"
        )
    for instance in instances:
        instance["benchmark"] = capabilities.canonical_name
        if benchmark_root is not None:
            instance["benchmark_root"] = str(benchmark_root)

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

    # Group by site so same-site replicas all survive the lookup. The prior
    # shape (``dict[site, inst]``) silently dropped every replica after the
    # first, which routed every task at a site to a single upstream — the
    # 2026-04-22 gitlab_18 crush bug. Downstream call sites pick a single
    # replica per task via ``select_task_site_instance_dict`` (Phase 4's
    # selector, mirrored for raw dicts) so fanout matches Phase 4's hash
    # space exactly.
    instances_by_site: dict[str, list[dict[str, Any]]] = {}
    for inst in instances:
        name = str(inst.get("site_name", "")).strip().lower()
        if not name:
            continue
        instances_by_site.setdefault(name, []).append(inst)

    for site, site_instances in instances_by_site.items():
        if site not in sites_in_tasks:
            continue
        # Probing once per site is sufficient — every replica of a site runs
        # the same image with the same seed DB, so base-state drift is a
        # per-site concern, not a per-replica one. Use the stable-ordered
        # head so the representative is reproducible across runs.
        representative = _ordered_instance_dicts(site_instances)[0]
        benchmark = normalize_benchmark_name(representative.get("benchmark") or task_benchmark)
        editor_cls = EDITOR_REGISTRY.get((benchmark, site))
        if editor_cls is None:
            raise RuntimeError(
                f"phase 2c pre-flight: no editor registered for (benchmark={benchmark!r}, site={site!r})"
            )
        editor_cls.probe_base_state(representative)

    # Bug I (2026-04-23): preflight HTTP probe of each task's benign entry
    # URL plus editor-implied read/attach surfaces. Tasks with
    # deterministically-broken source data (login_redirect, 404, 403, 410,
    # 401) are quarantined as ``source_data_issue`` — they never reach the
    # full browser probe.
    # Transient signals (5xx, 429, timeouts, connection errors) pass
    # through and let the real probe retry. Unexpected preflight crashes fail
    # Phase 2c rather than silently writing a clean-looking
    # source_data_dropped_count=0 report.
    dropped_source_data: list[dict[str, Any]] = []
    preflight_start = time.monotonic()
    logger.info(
        "phase 2c preflight: starting probe over %d task(s) across %d site(s)",
        len(raw),
        len(instances_by_site),
    )
    dropped_source_data = await _run_preflight_and_filter_raw(
        raw,
        instances_by_site=instances_by_site,
        benchmark_root=benchmark_root,
    )
    logger.info(
        "phase 2c preflight: complete — dropped %d task(s); %d remain for the probe (elapsed=%.1fs)",
        len(dropped_source_data),
        len(raw),
        time.monotonic() - preflight_start,
    )
    # ``raw`` has been mutated in place by _run_preflight_and_filter_raw
    # when preflight succeeds.

    verified: list[dict[str, Any]] = []
    infeasible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    cleanup_warnings: list[str] = []
    per_site: dict[str, dict[str, int]] = {}

    # Two-layer concurrency control, added 2026-04-22:
    #   * ``semaphore`` is the chromium-memory guard: each verified task
    #     spawns its own headless browser (~500 MB RSS peak), and r5 has
    #     ~34 GB free at steady state. 64 leaves margin for spikes while
    #     freeing operators from tuning ``--feasibility-concurrency`` to
    #     the backend (per-replica caps below own that).
    #   * ``per_replica_sems`` is the bulkhead: each replica absorbs at
    #     most ``_PER_REPLICA_CAP_DEFAULT[site]`` in-flight verifications
    #     at once. Gitlab's live container (verified via SSH on
    #     2026-04-22) runs puma 4 workers x 4 threads = 16 HTTP slots,
    #     worker_timeout 60 s; cap=10 leaves 6 slots for the Sidekiq-
    #     triggered internal API traffic a write fans out into. Reddit
    #     (Postmill, PHP-fpm) default is a placeholder pending Layer 5
    #     observability data.
    memory_cap = max(int(concurrency), 64)
    semaphore = asyncio.Semaphore(memory_cap)
    # Outermost cap: total concurrent Chromium processes across the run.
    # Scarcer than the memory cap (8 vs 64) and than the summed per-
    # replica caps (~290 for the r5 fleet), so acquired first so waiting
    # tasks do not pin memory or replica budget. See module-level
    # _BROWSER_PROBE_CAP for rationale.
    browser_probe_sem = asyncio.Semaphore(_BROWSER_PROBE_CAP)
    per_replica_sems: dict[str, asyncio.BoundedSemaphore] = {}
    # in_flight_counts feeds :func:`select_task_site_instance_dict_p2c`.
    # It counts tasks that have *reserved* a replica (via P2C pick),
    # including those queued on the replica's semaphore — not just those
    # actively holding it. That matches the real load signal: if 15
    # tasks hash to replica A (cap 10) while B sits idle, B should win
    # the next pick. Incremented at reservation, decremented in a
    # ``finally`` block so crashes do not leak phantom load.
    in_flight_counts: dict[str, int] = {}
    replica_stats: dict[str, _ReplicaStats] = {}

    def _stats_for(instance: dict[str, Any]) -> _ReplicaStats:
        key = replica_key(instance)
        stats = replica_stats.get(key)
        if stats is None:
            stats = _ReplicaStats(
                site_name=str(instance.get("site_name", "")).strip().lower(),
                replica_name=key,
            )
            replica_stats[key] = stats
        return stats

    def _replica_sem_for(instance: dict[str, Any]) -> asyncio.BoundedSemaphore:
        key = replica_key(instance)
        sem = per_replica_sems.get(key)
        if sem is None:
            cap = _per_replica_cap(str(instance.get("site_name", "")))
            sem = asyncio.BoundedSemaphore(cap)
            # asyncio is single-threaded; no yield point between get and
            # assign, so this is race-free without an explicit lock.
            per_replica_sems[key] = sem
        return sem

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
        # Resolve site + replica *before* acquiring any semaphore so the
        # fast ``unsupported_site`` path does not burn chromium budget or
        # block other workers on an impossible-to-run task.
        #
        # Phase 4 binds the seed call to the *delivery* site (from
        # ``delivery_channel.delivery_site``, falling back to the first
        # editor_call's ``site``, falling back to ``task["site"]``).
        # Mirror that so cross-site adversarial seeds — e.g. a
        # shopping_admin task whose payload seeds a product review on
        # the shopping storefront — verify against the correct instance.
        seed_site = _resolve_seed_site(task)
        site_instances = instances_by_site.get(seed_site) or []
        if not site_instances:
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
        # Power-of-two-choices replica selection. Unlike Phase 4
        # (deterministic hash for trajectory reproducibility), Phase 2c
        # verification is stateless: any healthy replica of the site can
        # host the seed→probe→cleanup cycle. P2C samples two replicas
        # and picks the one with fewer in-flight tasks, which empirically
        # eliminates the hot-replica imbalance that deterministic hashing
        # exhibits under small N (e.g. 107 tasks / 21 replicas).
        instance = select_task_site_instance_dict_p2c(
            task, seed_site, site_instances, in_flight_counts
        )
        instance_key = replica_key(instance)
        new_in_flight = in_flight_counts.get(instance_key, 0) + 1
        in_flight_counts[instance_key] = new_in_flight
        stats = _stats_for(instance)
        if new_in_flight > stats.in_flight_peak:
            stats.in_flight_peak = new_in_flight
        worker_started_mono = time.monotonic()
        worker_ok = False
        try:
            replica_sem = _replica_sem_for(instance)

            # Three-layer acquire order, outermost first:
            #   1. browser_probe_sem (cap 8) — total Chromium processes
            #      across the run. Scarcest. Queuing here holds no memory
            #      or replica budget so waiting tasks are cheap.
            #   2. semaphore (cap 64) — chromium memory guard; kept for
            #      spike headroom even though browser_probe_sem is
            #      tighter today. Lets a future operator raise
            #      _BROWSER_PROBE_CAP toward the memory ceiling without
            #      rewiring the acquire order.
            #   3. replica_sem (cap 10 gitlab / 8 reddit) — per-replica
            #      HTTP bulkhead against the WASP site.
            async with browser_probe_sem, semaphore:
                if stagger_delay:
                    await asyncio.sleep(min(stagger_delay * index, stagger_delay * 10))
                # Per-task browser: every worker gets its own Playwright +
                # Chromium pair, torn down before the next task runs. A
                # render-check crash in one task (renderer OOM, sandbox
                # child death, page crash that propagates to the browser)
                # now stays inside this worker — sibling tasks launch
                # their own browsers and are unaffected. This matches the
                # pattern every peer web-agent eval harness uses
                # (WebArena, VisualWebArena, BrowserGym, AgentLab, WASP)
                # and the official pytest-playwright idiom
                # (browser scope = worker, context scope = task). Launch
                # cost is ~1.5-3s per task; at concurrency N that's
                # ~(launch_cost * total_tasks / N) added wall-clock,
                # acceptable vs the all-or-nothing shared-browser
                # fragility.
                pw_handle: Any = None
                browser: Any = None
                try:
                    if async_playwright_factory is not None:
                        pw_handle = await async_playwright_factory().start()
                        browser = await pw_handle.chromium.launch(
                            headless=True,
                            args=list(_PROBE_LAUNCH_ARGS),
                        )
                    try:
                        async with replica_sem:
                            result = await _verify_one(
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
                        # Any ``infeasible`` we reach here came from an
                        # editor-level refusal (e.g. 4xx, schema mismatch)
                        # — the replica itself served the request fine,
                        # so count it as a successful round-trip for
                        # latency/p99 purposes. ``verification_crashed``
                        # below is the "replica broke the client" path.
                        worker_ok = True
                        return result
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
        finally:
            # Decrement always, including after verification_crashed or
            # browser-teardown failure, so P2C's load signal stays true.
            in_flight_counts[instance_key] = max(0, in_flight_counts.get(instance_key, 1) - 1)
            stats.record(
                elapsed_ms=(time.monotonic() - worker_started_mono) * 1000.0,
                ok=worker_ok,
            )

    results = await asyncio.gather(
        *(worker(task, i) for i, task in enumerate(raw)),
        return_exceptions=False,
    )

    # Per-replica observability summary. One log line per replica, sorted
    # by (site, replica_name) so the output is stable across runs. Use
    # this to validate that the per-replica cap and P2C selection are
    # actually balancing load before tuning either knob.
    if replica_stats:
        for key in sorted(replica_stats, key=lambda k: (replica_stats[k].site_name, k)):
            logger.info("phase 2c replica_stats: %s", replica_stats[key].summary())

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
        dropped_source_data=dropped_source_data,
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

    content_hash = _task_content_hash(
        editor_calls if isinstance(editor_calls, list) else [],
        exposure_contract=task.get("exposure_contract"),
    )
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
            # Render-unverified means the seed wrote successfully but the
            # signature did not appear in any read-surface URL within the
            # body-poll window. On loaded GitLab hosts this is dominated
            # by sidekiq indexer + page-cache invalidation tail; the seed
            # IS visible a few seconds later. Give the platform one
            # 3-second breather and re-run the check so single-run jitter
            # (typically 1-3 tasks per Phase 2c) doesn't gate admission.
            # The exponential-backoff body poll already handles the fast
            # tail; this retry covers the slow tail (>20 s sidekiq).
            if (
                render_outcome is not None
                and not render_outcome.ok
                and render_outcome.kind == _RENDER_UNVERIFIED_KIND
            ):
                await asyncio.sleep(_RENDER_UNVERIFIED_RETRY_DELAY_S)
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
                exposure_contract = task.get("exposure_contract")
                if isinstance(exposure_contract, dict):
                    eligibility = exposure_contract.get("eligibility")
                    if not isinstance(eligibility, dict) or eligibility.get("status") != "eligible":
                        reachability_outcome = ReachabilityOutcome.unreachable(
                            kind="exposure_contract_ineligible",
                            detail="task exposure_contract is missing eligible status",
                            url=str(instance.get("site_url") or ""),
                        )
                    else:
                        reachability_outcome = await _run_reachability_check(
                            browser=browser,
                            render_semaphore=render_semaphore,
                            task=task,
                            seed=seed,
                            metadata=metadata,
                            instance=instance,
                            render_outcome=render_outcome,
                        )
                elif isinstance(resource, dict) and resource.get("kind") is not None:
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
    read_surface_urls = metadata.get("read_surface_urls") if isinstance(metadata, dict) else None
    if isinstance(read_surface_urls, list):
        cleaned_urls = [url for url in read_surface_urls if isinstance(url, str) and url.strip()]
        if cleaned_urls:
            result["read_surface_urls"] = cleaned_urls
    read_surface_provenance = (
        metadata.get("read_surface_provenance") if isinstance(metadata, dict) else None
    )
    if isinstance(read_surface_provenance, dict):
        result["read_surface_provenance"] = read_surface_provenance
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
        exposure_contract = task.get("exposure_contract")
        if isinstance(exposure_contract, dict):
            layout_probe = (
                render_outcome.layout_probe
                if render_outcome is not None and isinstance(render_outcome.layout_probe, dict)
                else None
            )
            feasibility["exposure"] = {
                "contract_id": exposure_contract.get("contract_id"),
                "reachable": reachability_outcome.reachability != "unreachable",
                "visual_reachable": reachability_outcome.visual_reachable is True,
                "layout_visible_at_entry": (
                    layout_probe.get("visible_at_entry") if layout_probe is not None else None
                ),
                "scroll_to_visible_px": (
                    layout_probe.get("scroll_to_visible_px") if layout_probe is not None else None
                ),
                "requires_expand": (
                    layout_probe.get("requires_expand") if layout_probe is not None else None
                ),
                "verification": exposure_contract.get("verification"),
            }
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

    Signature is derived from the seed so the reachability DOM grep has a
    DOM-stable witness. We deliberately do NOT reuse
    ``render_outcome.matched_signature`` because the GitLab note RYW
    fastpath (phase_2_render_check._gitlab_note_ryw_fastpath) returns a
    synthetic ``note_id=<N>`` marker that exists only in the
    ``/discussions.json`` JSON response — never in DOM text. Greping the
    page body for it would always miss, even on correctly rendered and
    authenticated pages. render_signature() returns the same value
    matched_signature would have for DOM-text passes and a DOM-stable
    substring for RYW passes. Second witness is derived from the seed
    body text via ``phase_2_reachability.derive_second_witness``.
    """
    phase4_exposure_error = _phase4_exposure_inadmissible_reason(task.get("exposure_contract"))
    if phase4_exposure_error is not None:
        return ReachabilityOutcome.unreachable(
            kind=f"phase4_exposure_{phase4_exposure_error}",
            detail=(
                "exposure contract is seedable but not admissible for current "
                f"Phase 4 encounter semantics: {phase4_exposure_error}"
            ),
            url=str(instance.get("site_url") or ""),
        )

    benign_target_resource = _reachability_resource_for_task(task, metadata=metadata)
    site_url = str(instance.get("site_url") or "").strip()
    url_token = _required_url_token(task)
    payload_source = (
        _selected_rendered_payload(task)
        or _first_rendered_payload(seed)
        or (render_outcome.rendered_body_text if render_outcome is not None else None)
    )
    if url_token is not None:
        signature = url_token
        second_witness = derive_second_witness(payload_source, signature)
    else:
        signature = render_signature(seed, metadata)
        second_witness = derive_second_witness(payload_source, signature)
    browser_context_kwargs, auth_error = _resolve_benign_browser_context_auth(instance)
    if auth_error is not None:
        return ReachabilityOutcome.unreachable(
            kind=_auth_probe_failure_kind(auth_error),
            detail=auth_error,
            url=site_url,
            witnesses_missing=tuple(
                witness
                for witness in (signature, second_witness)
                if isinstance(witness, str) and witness
            ),
        )

    async def _do() -> ReachabilityOutcome:
        try:
            outcome = await verify_reachable(
                browser=browser,
                benign_target_resource=benign_target_resource,
                instance_site_url=site_url,
                signature=signature,
                second_witness=second_witness,
                browser_context_kwargs=browser_context_kwargs,
            )
            return outcome
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


def _phase4_exposure_inadmissible_reason(contract: Any) -> str | None:
    if not isinstance(contract, dict):
        return None
    capability = contract.get("phase4_exposure")
    if not isinstance(capability, dict):
        return None
    if capability.get("admissible") is True:
        return None
    reason = capability.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return "inadmissible"


def _reachability_resource_for_task(
    task: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project an exposure contract into the existing reachability probe shape."""
    resource = task.get("benign_target_resource")
    projected = dict(resource) if isinstance(resource, dict) else {}
    contract = task.get("exposure_contract")
    if not isinstance(contract, dict):
        return projected or None
    verification = contract.get("verification")
    verification_url = (
        verification.get("url")
        if isinstance(verification, dict)
        else contract.get("benign_read_url")
    )
    if isinstance(verification_url, str) and verification_url.strip():
        projected["start_url_resolved"] = verification_url
    target_url = _verification_target_url(contract, metadata or {})
    if target_url:
        projected["exposure_target_url"] = target_url
    kind = contract.get("kind")
    if isinstance(kind, str) and kind.strip():
        projected["kind"] = kind
    if "anchors" not in projected and isinstance(contract.get("anchors"), dict):
        projected["anchors"] = dict(contract["anchors"])
    projected["exposure_contract_id"] = contract.get("contract_id")
    projected["exposure_mode"] = contract.get("mode")
    return projected


def _verification_target_url(contract: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    verification = contract.get("verification")
    if not isinstance(verification, dict):
        return None
    target = verification.get("target")
    if not isinstance(target, dict):
        return None
    direct = target.get("url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    source = target.get("url_source")
    if not isinstance(source, str) or not source.startswith("seed_metadata."):
        return None
    key = source.removeprefix("seed_metadata.")
    value = _metadata_path_value(metadata, key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _metadata_path_value(metadata: dict[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list):
            selected: Any = None
            if part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    selected = current[index]
            else:
                for item in current:
                    if isinstance(item, dict) and item.get("role") == part:
                        selected = item
                        break
            current = selected
            continue
        return None
    return current


def _required_url_token(task: dict[str, Any]) -> str | None:
    """Return the first URL value from task.required_tokens, or None."""
    tokens = task.get("required_tokens")
    if not isinstance(tokens, list):
        return None
    for entry in tokens:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "url":
            continue
        value = entry.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_rendered_payload(seed: dict[str, Any]) -> str | None:
    """Extract the longest rendered payload string from a seed's editor_calls.

    Used as the source pool for the reachability probe's second witness.
    Prefers the longest arg value across all calls so an unsubstituted
    short ``{benign_*}`` token does not win over the actual body text:
    observed in 3 reddit create_comment tasks where the dict iteration
    picked the 22-char ``{benign_submission_id}`` selector arg before
    the 200-600 char comment body, producing a literal-token witness
    that the reachability probe could never find on the rendered page.
    """
    if not isinstance(seed, dict):
        return None
    calls = seed.get("editor_calls")
    if not isinstance(calls, list):
        return None
    best: str | None = None
    for call in calls:
        if not isinstance(call, dict):
            continue
        args = call.get("args")
        if not isinstance(args, dict):
            continue
        for value in args.values():
            if isinstance(value, str) and len(value) >= 20:
                if best is None or len(value) > len(best):
                    best = value
    return best


def _selected_rendered_payload(task: dict[str, Any]) -> str | None:
    """Return the selected Phase 2b rendered payload when present."""
    payloads = task.get("payload_texts")
    if not isinstance(payloads, list) or not payloads:
        return None
    raw_index = task.get("selected_payload_index", 0)
    try:
        selected_index = int(raw_index)
    except (TypeError, ValueError):
        selected_index = 0
    candidates: list[Any] = []
    if 0 <= selected_index < len(payloads):
        candidates.append(payloads[selected_index])
    candidates.extend(payloads)
    for payload in candidates:
        if not isinstance(payload, dict):
            continue
        for key in ("rendered_payload", "raw_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _preflight_request_context_options(
    instance: dict[str, Any],
    *,
    benchmark_root: Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Build Playwright APIRequestContext auth options for source-data preflight.

    Returns ``({}, reason)`` when declared auth is unusable. The caller then
    skips source-data quarantine for that instance instead of probing
    anonymously and falsely classifying private pages as source-data drops.
    """
    agent_auth = instance.get("agent_auth")
    resolved = resolve_agent_auth(
        agent_auth if isinstance(agent_auth, dict) else None,
        site_name=str(instance.get("site_name") or ""),
        site_url=str(instance.get("site_url") or ""),
        benchmark_root=benchmark_root,
        storage_state_override=instance.get("storage_state_path"),
    )
    if resolved.unusable_reason is not None:
        return {}, resolved.unusable_reason
    return dict(resolved.api_request_context_kwargs), None


def _agent_auth_type(instance: dict[str, Any]) -> str:
    agent_auth = instance.get("agent_auth")
    if not isinstance(agent_auth, dict):
        return ""
    return str(agent_auth.get("type") or "").strip()


def _instance_benchmark_or_none(instance: dict[str, Any]) -> str | None:
    values = [
        instance.get("benchmark"),
        instance.get("benchmark_name"),
        instance.get("benchmark_adapter"),
    ]
    try:
        return infer_benchmark_name(values)
    except ValueError:
        return None


def _infer_records_benchmark(records: list[dict[str, Any]], *, label: str) -> str:
    values: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        values.extend(
            (
                record.get("benchmark"),
                record.get("benchmark_name"),
                record.get("benchmark_adapter"),
            )
        )
        seed = record.get("adversarial_data_seed")
        calls = seed.get("editor_calls") if isinstance(seed, dict) else None
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    values.extend(
                        (
                            call.get("benchmark"),
                            call.get("benchmark_name"),
                            call.get("benchmark_adapter"),
                        )
                    )
    try:
        benchmark = infer_benchmark_name(values)
    except ValueError as exc:
        raise RuntimeError(f"phase 2c pre-flight: {label} contain {exc}") from exc
    if benchmark is None:
        raise RuntimeError(f"phase 2c pre-flight: {label} are missing benchmark metadata")
    return benchmark


def _resolve_agent_auth_headers(agent_auth: dict[str, Any]) -> dict[str, str]:
    return resolve_agent_auth_headers(agent_auth)


def _storage_state_preflight_error(path: str, instance: dict[str, Any]) -> str | None:
    payload, error = _read_storage_state_payload_for_preflight(path)
    if error is not None:
        return error
    return _storage_state_preflight_error_for_payload(Path(path), payload, instance)


def _read_storage_state_payload_for_preflight(
    path: str,
) -> tuple[dict[str, Any], str | None]:
    return read_storage_state_payload(path)


def _storage_state_preflight_error_for_payload(
    path_obj: Path,
    payload: dict[str, Any],
    instance: dict[str, Any],
) -> str | None:
    return storage_state_preflight_error_for_payload(
        path_obj,
        payload,
        str(instance.get("site_url") or ""),
    )


def _playwright_storage_state_for_preflight(path: str) -> tuple[str | dict[str, Any], str | None]:
    """Return a Playwright-compatible storage state for preflight.

    Phase 0d artifacts may come from non-Playwright browser APIs whose
    cookie ``sameSite`` values use CDP names such as ``no_restriction``.
    Normalize known equivalents in memory so auth remains usable. Unknown
    shapes keep the existing auth-unusable path, which makes preflight skip
    this instance instead of probing private surfaces anonymously.
    """
    return playwright_storage_state(path)


def _playwright_storage_state_payload_for_preflight(
    path_obj: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    return playwright_storage_state_payload(path_obj, payload)


def _storage_state_recorded_hosts(payload: dict[str, Any]) -> set[str]:
    return storage_state_recorded_hosts(payload)


def _storage_state_cookie_hosts(payload: dict[str, Any]) -> set[str]:
    return storage_state_cookie_hosts(payload)


def _storage_state_origin_hosts(payload: dict[str, Any]) -> set[str]:
    return storage_state_origin_hosts(payload)


def _cookie_domain_matches_host(domain: str, host: str) -> bool:
    return cookie_domain_matches_host(domain, host)


async def _run_preflight_and_filter_raw(
    raw: list[dict[str, Any]],
    *,
    instances_by_site: dict[str, list[dict[str, Any]]],
    benchmark_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Probe every task and mutate ``raw`` in place to drop quarantined tasks.

    Uses a lazy Playwright ``APIRequestContext`` per (site, auth-context)
    pair — shares Playwright's TLS/proxy setup with the render_check path
    without importing httpx. Returns the list of dropped records with
    ``source_data_issue`` metadata attached; the caller routes this list
    to the ``adversarial_tasks.dropped_source_data.json`` sidecar.

    Skips silently if Playwright is not importable (development without
    Playwright installed); mirrors the render_check skip-envvar shape.
    """
    from worldsim.phases.phase_2c_preflight import (
        auth_self_test_path,
        preflight_benign_targets,
        self_test_preflight_auth,
    )

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.info(
            "phase 2c preflight: playwright not importable; skipping source-"
            "data quarantine for this run."
        )
        return []

    pw_handle = await async_playwright().start()
    contexts_created: list[Any] = []

    async def _factory(context_options: dict[str, Any] | None) -> Any:
        kwargs: dict[str, Any] = {}
        if isinstance(context_options, dict):
            for key in ("storage_state", "extra_http_headers", "http_credentials"):
                value = context_options.get(key)
                if value:
                    kwargs[key] = value
        ctx = await pw_handle.request.new_context(**kwargs)
        contexts_created.append(ctx)
        return ctx

    async def _self_test_context_options(
        *,
        site: str,
        instance: dict[str, Any],
        context_options: dict[str, Any],
    ) -> Any:
        benchmark = _instance_benchmark_or_none(instance)
        if benchmark is None:
            return None
        ctx = await _factory(context_options)
        try:
            return await self_test_preflight_auth(
                request_context=ctx,
                site=site,
                site_url=str(instance.get("site_url") or ""),
                benchmark=benchmark,
            )
        finally:
            try:
                await ctx.dispose()
            except Exception:
                logger.debug(
                    "phase 2c preflight: auth self-test context dispose failed",
                    exc_info=True,
                )

    async def _ensure_live_storage_state_options(
        *,
        site: str,
        instance: dict[str, Any],
        context_options: dict[str, Any],
        skip_reason: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        benchmark = _instance_benchmark_or_none(instance)
        if benchmark is None:
            return context_options, skip_reason
        if (
            auth_self_test_path(site, benchmark=benchmark) is None
            or _agent_auth_type(instance) != "storage_state"
        ):
            return context_options, skip_reason

        if skip_reason is None:
            classification = await _self_test_context_options(
                site=site,
                instance=instance,
                context_options=context_options,
            )
            if classification is None or classification.kind == "reachable":
                return context_options, None
            if classification.kind not in {"login_redirect", "auth_missing"}:
                raise RuntimeError(
                    "phase 2c preflight: auth self-test for "
                    f"{site} at {instance.get('site_url')!r} was inconclusive: "
                    f"{classification.kind} ({classification.detail})"
                )
            logger.warning(
                "phase 2c preflight: storage_state auth for %s at %s is stale: %s; "
                "reacquiring via Phase 0d",
                site,
                instance.get("site_url"),
                classification.detail,
            )
        else:
            logger.warning(
                "phase 2c preflight: storage_state auth for %s at %s is unusable: %s; "
                "reacquiring via Phase 0d",
                site,
                instance.get("site_url"),
                skip_reason,
            )

        lock_key = f"{site}:{instance.get('site_url')}"
        lock = _PREFLIGHT_AUTH_REFRESH_LOCKS.setdefault(lock_key, asyncio.Lock())
        async with lock:
            from worldsim.phases.phase_0d_auth_bootstrap import (
                AuthBootstrapError,
                reacquire_storage_state,
            )

            try:
                refreshed_path = await reacquire_storage_state(
                    site_name=site,
                    instance=instance,
                    benchmark_root=benchmark_root,
                )
            except AuthBootstrapError as exc:
                reason = (
                    "storage_state refresh failed; source-data quarantine skipped "
                    f"for this instance ({exc})"
                )
                logger.warning(
                    "phase 2c preflight: %s at %s: %s",
                    site,
                    instance.get("site_url"),
                    reason,
                )
                return {}, reason
            instance["storage_state_path"] = str(refreshed_path)
            agent_auth = instance.get("agent_auth")
            if isinstance(agent_auth, dict):
                updated_auth = dict(agent_auth)
                storage_state = updated_auth.get("storage_state")
                if isinstance(storage_state, dict):
                    updated_storage_state = dict(storage_state)
                else:
                    updated_storage_state = {}
                updated_storage_state["path"] = str(refreshed_path)
                updated_auth["storage_state"] = updated_storage_state
                instance["agent_auth"] = updated_auth

        refreshed_options, refreshed_skip = _preflight_request_context_options(
            instance,
            benchmark_root=benchmark_root,
        )
        if refreshed_skip is not None:
            reason = (
                "reacquired storage_state is unusable; source-data quarantine skipped "
                f"for this instance ({refreshed_skip})"
            )
            logger.warning(
                "phase 2c preflight: %s at %s: %s",
                site,
                instance.get("site_url"),
                reason,
            )
            return {}, reason
        refreshed_classification = await _self_test_context_options(
            site=site,
            instance=instance,
            context_options=refreshed_options,
        )
        if refreshed_classification is None or refreshed_classification.kind == "reachable":
            logger.info(
                "phase 2c preflight: refreshed storage_state auth for %s at %s",
                site,
                instance.get("site_url"),
            )
            return refreshed_options, None
        if refreshed_classification.kind in {"login_redirect", "auth_missing"}:
            reason = (
                "storage_state refresh did not authenticate; source-data quarantine "
                f"skipped for this instance ({refreshed_classification.kind}: "
                f"{refreshed_classification.detail})"
            )
            logger.warning(
                "phase 2c preflight: %s at %s: %s",
                site,
                instance.get("site_url"),
                reason,
            )
            return {}, reason
        raise RuntimeError(
            "phase 2c preflight: storage_state refresh for "
            f"{site} at {instance.get('site_url')!r} did not authenticate: "
            f"{refreshed_classification.kind} ({refreshed_classification.detail})"
        )

    try:
        preflight_instances_by_site: dict[str, list[dict[str, Any]]] = {}
        for site, site_instances in instances_by_site.items():
            resolved_instances: list[dict[str, Any]] = []
            for instance in site_instances:
                resolved = dict(instance)
                context_options, skip_reason = _preflight_request_context_options(
                    instance,
                    benchmark_root=benchmark_root,
                )
                context_options, skip_reason = await _ensure_live_storage_state_options(
                    site=site,
                    instance=resolved,
                    context_options=context_options,
                    skip_reason=skip_reason,
                )
                if resolved.get("storage_state_path"):
                    instance["storage_state_path"] = resolved["storage_state_path"]
                if isinstance(resolved.get("agent_auth"), dict):
                    instance["agent_auth"] = resolved["agent_auth"]
                if skip_reason is not None:
                    resolved["preflight_auth_skip_reason"] = skip_reason
                else:
                    resolved["preflight_request_context"] = context_options
                resolved_instances.append(resolved)
            preflight_instances_by_site[site] = resolved_instances
        keep, dropped = await preflight_benign_targets(
            raw,
            instances_by_site=preflight_instances_by_site,
            request_context_factory=_factory,
        )
    finally:
        for ctx in contexts_created:
            try:
                await ctx.dispose()
            except Exception:
                logger.debug(
                    "phase 2c preflight: request context dispose failed",
                    exc_info=True,
                )
        try:
            await pw_handle.stop()
        except Exception:
            logger.debug("phase 2c preflight: playwright stop failed", exc_info=True)

    if dropped:
        kept_ids = {id(t) for t in keep}
        raw[:] = [t for t in raw if id(t) in kept_ids]
    return dropped


def _resolve_benign_storage_state_path(instance: dict[str, Any]) -> str | None:
    """Return the Phase-0d-bootstrapped storage_state.json path for this site.

    Under Option A (alpha) identity the seed writer and the reachability
    probe both act as the benign user, so threading those cookies into
    Playwright lets the probe reach private projects + authed-only
    pages. Falls back to ``None`` when no artifact is present (public
    content still works in an anonymous context).
    """
    agent_auth = instance.get("agent_auth")
    if not isinstance(agent_auth, dict) or agent_auth.get("type") != "storage_state":
        return None
    path = resolve_storage_state_path(
        agent_auth,
        site_name=str(instance.get("site_name") or ""),
        storage_state_override=instance.get("storage_state_path"),
        benchmark_root=Path(str(instance["benchmark_root"]))
        if instance.get("benchmark_root")
        else None,
    )
    return str(path) if path is not None else None


def _resolve_benign_browser_context_auth(
    instance: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Return browser context auth kwargs for Phase 2c browser probes.

    No configured ``agent_auth`` preserves the legacy anonymous probe path.
    Declared-but-unusable auth returns an explicit reason so callers fail
    closed instead of silently probing as an anonymous visitor.
    """
    agent_auth = instance.get("agent_auth")
    if not isinstance(agent_auth, dict) or not str(agent_auth.get("type") or "").strip():
        return {}, None
    benchmark_root = (
        Path(str(instance["benchmark_root"])) if instance.get("benchmark_root") else None
    )
    resolved = resolve_agent_auth(
        agent_auth,
        site_name=str(instance.get("site_name") or ""),
        site_url=str(instance.get("site_url") or ""),
        benchmark_root=benchmark_root,
        storage_state_override=instance.get("storage_state_path"),
    )
    if resolved.unusable_reason is not None:
        return {}, resolved.unusable_reason
    return dict(resolved.browser_context_kwargs), None


def _auth_probe_failure_kind(reason: str) -> str:
    lowered = reason.lower()
    if "missing" in lowered or "not found" in lowered or "no usable artifact" in lowered:
        return "auth_missing"
    return "auth_unusable"


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
    # Authoritative write tokens returned by the editor's POST response.
    # The render-check uses these as a read-your-write fallback when its
    # DOM text match races the platform's write-to-visible pipeline
    # (sidekiq indexer, page-cache invalidation, Vue hydration). The
    # same tokens live on ``metadata`` because they are already part of
    # the editor's return contract; we just hoist the observably-useful
    # subset so verify_seed_renders does not have to know the whole
    # metadata shape.
    write_tokens: dict[str, Any] = {}
    if isinstance(metadata, dict):
        for key in (
            "note_id",
            "issue_iid",
            "project_id",
            "comment_id",
            "submission_id",
            "review_id",
        ):
            value = metadata.get(key)
            if value not in (None, ""):
                write_tokens[key] = value

    browser_context_kwargs, auth_error = _resolve_benign_browser_context_auth(instance)
    if auth_error is not None:
        return RenderOutcome.failed(
            kind=_auth_probe_failure_kind(auth_error),
            detail=auth_error,
            urls_tried=[],
            per_url_errors={},
        )

    async def _do() -> RenderOutcome:
        try:
            return await verify_seed_renders(
                browser=browser,
                urls=urls,
                site_name=site_name,
                site_url=site_url,
                signature=signature,
                browser_context_kwargs=browser_context_kwargs,
                write_tokens=write_tokens or None,
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


def _task_content_hash(editor_calls: list[Any], *, exposure_contract: Any = None) -> str:
    projection = _exposure_contract_fingerprint_projection(exposure_contract)
    payload: Any = (
        {"editor_calls": editor_calls, "exposure_contract": projection}
        if projection is not None
        else editor_calls
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _exposure_contract_fingerprint_projection(contract: Any) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None
    keys = (
        "contract_id",
        "site",
        "kind",
        "mode",
        "benign_read_url",
        "editor_method",
        "target_surface_id",
        "payload_arg",
        "editor_args_template",
        "verification",
        "eligibility",
    )
    return {key: contract.get(key) for key in keys if key in contract}


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
    repo_root = Path(__file__).resolve().parent.parent.parent
    sync_commit = _sync_stamp_commit(repo_root)
    if sync_commit:
        return sync_commit
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


def _sync_stamp_commit(repo_root: Path) -> str | None:
    stamp_path = repo_root / ".worldsim_sync_stamp.json"
    try:
        payload = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    local_git = payload.get("local_git")
    if not isinstance(local_git, dict):
        return None
    sha = local_git.get("sha")
    if not isinstance(sha, str):
        return None
    sha = sha.strip()
    if not sha:
        return None
    return sha[:12]


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
