"""Phase 2c feasibility runner: source-data preflight, replica admission,
per-task fan-out, and the ``FeasibilityReport`` aggregate.

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
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from warp_taskgen.benchmark_capabilities import (
    get_benchmark_capabilities,
    infer_benchmark_name,
    normalize_benchmark_name,
)
from warp_taskgen.editors import EDITOR_REGISTRY
from warp_taskgen.instance_selection import (
    _ordered_instance_dicts,
    replica_key,
    select_task_site_instance_dict_p2c,
)
from warp_taskgen.phase_2.phase_2c import checkpoints as _checkpoints
from warp_taskgen.phase_2.phase_2c.checkpoints import (
    POLICY_CATALOG_VERSION,
    SITE_CATALOG_VERSION,
    VERIFIER_VERSION,
)
from warp_taskgen.phase_2.phase_2c.fingerprints import _task_content_hash
from warp_taskgen.phase_2.phase_2c.outcomes import _infeasible_task, _now_iso, _resolve_seed_site
from warp_taskgen.phase_2.phase_2c.pause_control import (
    assert_preflight_boundary,
    run_verification_units,
)
from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
from warp_taskgen.phase_2.phase_2c.probe_bundle import Phase2cProbeBundle
from warp_taskgen.phase_2.phase_2c.probes import (
    _BROWSER_PROBE_CAP,
    _PROBE_LAUNCH_ARGS,
    _SKIP_RENDER_CHECK_ENV,
)
from warp_taskgen.phase_2.phase_2c.types import FeasibilityReport, _ReplicaStats
from warp_taskgen.phase_2.phase_2c.verifier import _verify_one
from warp_taskgen.runtime_composition import RequiredSeedCleanupError, RuntimeComposition
from warp_taskgen.seeding import SeedSiteRegistry
from warp_taskgen.sites import SiteCatalog

logger = logging.getLogger(__name__)

# Per-replica in-flight caps for Phase 2c. Derived from live container
# inspection on 2026-04-22 (webarena-verified-gitlab_3): puma runs 4
# workers x 4 threads = 16 HTTP slots with worker_timeout 60 s. Cap 10
# is ~60 % of puma capacity, leaving 6 slots for the internal API
# traffic a GitLab write fans out into (NotificationsService, mentions,
# TodoService, webhooks). Reddit (Postmill / PHP-fpm) is not yet
# measured; 8 is a mid-range placeholder pending Layer 5 observability.
_PER_REPLICA_CAP_DEFAULT: dict[str, int] = {"gitlab": 10, "reddit": 8}
_PER_REPLICA_CAP_FALLBACK = 6


def _per_replica_cap(site_name: str) -> int:
    return _PER_REPLICA_CAP_DEFAULT.get(site_name.strip().lower(), _PER_REPLICA_CAP_FALLBACK)


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
    feasibility_policy_catalog: FeasibilityPolicyCatalog | None = None,
    seed_registry: SeedSiteRegistry | None = None,
    site_catalog: SiteCatalog | None = None,
    runtime_composition: RuntimeComposition | None = None,
    checkpoint_dir: Path | str | None = None,
    state_dir: Path | str | None = None,
    run_id: str | None = None,
    definition_digest: str | None = None,
    verifier_version: str = VERIFIER_VERSION,
    policy_version: str = POLICY_CATALOG_VERSION,
    catalog_version: str = SITE_CATALOG_VERSION,
    probes: Phase2cProbeBundle | None = None,
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
        feasibility_policy_catalog: Optional immutable per-run policy snapshot
            for source-data preflight. When omitted, the explicit WebArena
            default catalog is assembled for that run.
        seed_registry: Optional immutable per-run Site editor snapshot. When
            omitted, the historical editor registry remains the compatibility
            source.
        site_catalog: Optional immutable per-run Site capability catalog used
            to plan read-surface verification. When omitted, the production
            GitLab/Reddit catalog is assembled for each check.
        runtime_composition: Optional immutable per-run composition. When
            supplied, its Site, seed, and feasibility catalogs are used
            together; the default ``None`` path preserves existing behavior.
        checkpoint_dir: Optional directory for durable per-task checkpoints.
            When omitted, direct/legacy callers retain pre-checkpoint behavior;
            an identity without a directory never writes into the process cwd.
        state_dir: Optional authoritative Run state root used for cooperative
            pause admission. When omitted, direct/legacy callers do not poll
            lifecycle markers.
        run_id / definition_digest: Immutable Run identity. Checkpoint reuse
            is disabled when either is absent so legacy roots cannot acquire
            an invented identity.
        verifier_version / policy_version / catalog_version: Explicit
            compatibility versions bound into each checkpoint.
        probes: The injectable collaborators the loop calls (token
            acquisition, source-data preflight, seeding, render/reachability
            probes, retry sleep, host fingerprint, and browser readiness).
            ``None`` wires the real siblings via
            :meth:`Phase2cProbeBundle.default`.
    """
    if probes is None:
        probes = Phase2cProbeBundle.default()
    if runtime_composition is not None:
        if any(
            value is not None for value in (feasibility_policy_catalog, seed_registry, site_catalog)
        ):
            raise ValueError(
                "runtime_composition cannot be combined with explicit Phase 2c catalogs"
            )
        feasibility_policy_catalog = runtime_composition.feasibility_policy_catalog
        seed_registry = runtime_composition.seed_registry
        site_catalog = runtime_composition.site_catalog

    raw = json.loads(tasks_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{tasks_path} must contain a JSON array of tasks; got {type(raw).__name__}"
        )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
    if state_dir is not None:
        state_dir = Path(state_dir)
    fingerprint_base = probes.host_fingerprint(instances_label, instances)

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
            reused_checkpoints=0,
        )

    task_benchmark = _infer_records_benchmark(raw, label="Phase 2c tasks")
    instance_benchmark = _infer_records_benchmark(instances, label="Phase 2c instances")
    if task_benchmark != instance_benchmark:
        raise RuntimeError(
            "phase 2c pre-flight: mixed benchmark metadata between tasks and instances: "
            f"tasks={task_benchmark!r}, instances={instance_benchmark!r}"
        )
    benchmark_capabilities = get_benchmark_capabilities(task_benchmark)
    try:
        capabilities = benchmark_capabilities.require("phase_2_feasibility")
    except ValueError as exc:
        # A feature-local runtime composition may prove a bounded Phase 2
        # contract even while the global Benchmark catalog remains disabled.
        # The hook is explicit and fail-closed: it must admit the complete raw
        # task/instance set before TAC can cross this compatibility gate.
        admission_hook = (
            getattr(runtime_composition, "phase_2_admission", None)
            if runtime_composition is not None
            else None
        )
        if not callable(admission_hook):
            raise RuntimeError(
                f"phase 2c pre-flight: benchmark {task_benchmark!r} does not support "
                "WARP Taskgen Phase 2c"
            ) from exc
        try:
            admission = admission_hook(raw, instances)
        except Exception as admission_exc:
            raise RuntimeError(
                "phase 2c pre-flight: explicit runtime composition admission raised "
                f"{admission_exc.__class__.__name__}: {admission_exc}"
            ) from admission_exc
        # Only the feature-owned typed result may open this compatibility
        # gate.  A generic truthy callback must not become an undocumented
        # Benchmark-capability bypass.
        from warp_taskgen.runtime_composition import Phase2RuntimeAdmission

        if not isinstance(admission, Phase2RuntimeAdmission):
            raise RuntimeError(
                "phase 2c pre-flight: explicit runtime composition returned an "
                "unsupported admission result"
            ) from exc
        admitted = bool(getattr(admission, "admitted", getattr(admission, "ok", False)))
        if not admitted:
            reason = str(getattr(admission, "reason", "runtime composition rejected"))
            raise RuntimeError(
                "phase 2c pre-flight: explicit runtime composition admission rejected: " + reason
            ) from exc
        # Preserve the canonical identity for downstream fingerprints and
        # instance binding without pretending the global catalog supports TAC.
        capabilities = benchmark_capabilities
    for instance in instances:
        instance["benchmark"] = capabilities.canonical_name
        if benchmark_root is not None:
            instance["benchmark_root"] = str(benchmark_root)

    token_errors = probes.acquire_tokens(instances)
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
        if seed_registry is None:
            editor_cls = EDITOR_REGISTRY.get((benchmark, site))
        else:
            registration = seed_registry.get(benchmark, site)
            editor_cls = registration.editor_factory if registration is not None else None
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
    preflight_kwargs: dict[str, Any] = {
        "instances_by_site": instances_by_site,
        "benchmark_root": benchmark_root,
    }
    if feasibility_policy_catalog is not None:
        preflight_kwargs["feasibility_policy_catalog"] = feasibility_policy_catalog
    dropped_source_data = await probes.source_data_preflight(raw, **preflight_kwargs)
    logger.info(
        "phase 2c preflight: complete — dropped %d task(s); %d remain for the probe (elapsed=%.1fs)",
        len(dropped_source_data),
        len(raw),
        time.monotonic() - preflight_start,
    )
    # ``raw`` has been mutated in place by _run_preflight_and_filter_raw
    # when preflight succeeds.
    # Source-data preflight is one bounded setup operation. A request that
    # arrives while it drains is acknowledged only at this boundary, before
    # any verification task can claim an Atomic Work Unit.
    assert_preflight_boundary(state_dir)

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
    reused_checkpoint_count = 0
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
    # A render-disabled development run must never become production render
    # evidence merely because its other fields match.
    checkpoint_verifier_version = (
        f"{verifier_version}-render-disabled" if skip_render_check else verifier_version
    )
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
        async_playwright_factory = probes.playwright_factory
        if async_playwright_factory is None:
            raise RuntimeError(
                "phase 2c render verification requires Playwright: install "
                "'playwright' and run 'playwright install chromium', or set "
                f"{_SKIP_RENDER_CHECK_ENV}=1 to opt out (development only)."
            )
        await probes.ensure_chromium_ready(async_playwright_factory)

    async def worker(task: dict[str, Any], index: int) -> dict[str, Any]:
        nonlocal reused_checkpoint_count
        task_seed = task.get("adversarial_data_seed") or {}
        task_calls = task_seed.get("editor_calls") if isinstance(task_seed, dict) else None
        task_content_hash = _task_content_hash(
            task_calls if isinstance(task_calls, list) else [],
            exposure_contract=task.get("exposure_contract"),
        )
        task_checkpoint = (
            _checkpoints.checkpoint_context(
                run_id=run_id,
                definition_digest=definition_digest,
                task=task,
                task_content_hash=task_content_hash,
                topology_fingerprint=fingerprint_base,
                verifier_version=checkpoint_verifier_version,
                policy_version=policy_version,
                catalog_version=catalog_version,
            )
            if checkpoint_dir is not None
            else None
        )
        if task_checkpoint is not None and not force_reverify:
            loaded_checkpoint = _checkpoints.load_checkpoint(
                checkpoint_dir or Path("."),
                context=task_checkpoint,
            )
            if _checkpoints.checkpoint_is_fresh(loaded_checkpoint, ttl_hours=ttl_hours):
                reused_checkpoint_count += 1
                cleanup_warnings.extend(loaded_checkpoint.cleanup_warnings)
                return loaded_checkpoint.result or {}

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
        local_cleanup_warnings: list[str] = []
        checkpoint_work_unit = {
            "seed_applied": False,
            "render_completed": False,
            "reachability_completed": False,
        }
        result: dict[str, Any]
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
                            verify_kwargs: dict[str, Any] = {
                                "retry_count": retry_count,
                                "fingerprint_base": fingerprint_base,
                                "ttl_hours": ttl_hours,
                                "force_reverify": force_reverify,
                                "cleanup_warnings": local_cleanup_warnings,
                                "browser": browser,
                                "render_semaphore": None,
                                "checkpoint_context": task_checkpoint,
                                "checkpoint_work_unit": checkpoint_work_unit,
                                "probes": probes,
                            }
                            if seed_registry is not None:
                                verify_kwargs["seed_registry"] = seed_registry
                            if site_catalog is not None:
                                verify_kwargs["site_catalog"] = site_catalog
                            if runtime_composition is not None:
                                verify_kwargs["runtime_composition"] = runtime_composition
                            result = await _verify_one(
                                task,
                                instance,
                                **verify_kwargs,
                            )
                        # Any ``infeasible`` we reach here came from an
                        # editor-level refusal (e.g. 4xx, schema mismatch)
                        # — the replica itself served the request fine,
                        # so count it as a successful round-trip for
                        # latency/p99 purposes. ``verification_crashed``
                        # below is the "replica broke the client" path.
                        worker_ok = True
                    except RequiredSeedCleanupError:
                        # Cleanup is part of the named composition's Atomic
                        # Work Unit. Do not convert this into a reusable
                        # verification checkpoint or continue against dirty
                        # state.
                        raise
                    except Exception as exc:
                        task_id = str(task.get("id", "unknown"))
                        logger.exception("phase 2c verification crashed for task %s", task_id)
                        result = _infeasible_task(
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
            cleanup_warnings.extend(local_cleanup_warnings)
            if task_checkpoint is not None:
                checkpoint_work_unit["cleanup_completed"] = True
                _checkpoints.write_checkpoint(
                    checkpoint_dir or Path("."),
                    context=task_checkpoint,
                    result=result,
                    cleanup_warnings=local_cleanup_warnings,
                    seed_applied=checkpoint_work_unit["seed_applied"],
                    render_completed=checkpoint_work_unit["render_completed"],
                    reachability_completed=checkpoint_work_unit["reachability_completed"],
                )
            return result
        finally:
            # Decrement always, including after verification_crashed or
            # browser-teardown failure, so P2C's load signal stays true.
            in_flight_counts[instance_key] = max(0, in_flight_counts.get(instance_key, 1) - 1)
            stats.record(
                elapsed_ms=(time.monotonic() - worker_started_mono) * 1000.0,
                ok=worker_ok,
            )

    # The operation includes replica admission, seed effects, render/readback
    # and reachability evidence, cleanup, and the task-local checkpoint write.
    # The feature-owned scheduler serializes claims with the pause marker and
    # drains every already-admitted unit without cancellation.
    indexed_tasks = list(enumerate(raw))
    try:
        results = await run_verification_units(
            indexed_tasks,
            lambda indexed_task: worker(indexed_task[1], indexed_task[0]),
            concurrency=concurrency,
            state_dir=state_dir,
        )
    except RuntimeError as exc:
        # The scheduler preserves failed-unit identity as ``__cause__``.  A
        # required cleanup failure is a named terminal gate, not an ordinary
        # worker crash, so retain that identity for the caller/operator.
        if isinstance(exc.__cause__, RequiredSeedCleanupError):
            raise exc.__cause__ from exc
        raise

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
        reused_checkpoints=reused_checkpoint_count,
    )


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


__all__ = ["verify_feasibility"]
