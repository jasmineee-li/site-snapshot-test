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
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldsim._async_utils import retrying
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
from worldsim.phase_2.phase_2c import auth_preflight as _auth_preflight
from worldsim.phase_2.phase_2c import fingerprints as _fingerprints
from worldsim.phase_2.phase_2c import outcomes as _outcomes
from worldsim.phase_2.phase_2c import probes as _probes
from worldsim.phase_2.phase_2c import source_data_preflight as _source_data_preflight
from worldsim.phase_2.phase_2c.admission_guards import (
    _answer_target_collision_reason,
    _benign_agent_response_expected_strings,
    _flatten_string_values,
    _seed_has_appended_surface_attack,
    _seed_surface_values,
    _strings_overlap,
)
from worldsim.phase_2.phase_2c.exposure import _metadata_path_value, _verification_target_url
from worldsim.phase_2.phase_2c.fingerprints import (
    _exposure_contract_fingerprint_projection,
    _fingerprints_match,
    _first_method,
    _hours_since,
    _idempotency_decision,
    _instances_digest,
    _task_content_hash,
)
from worldsim.phase_2.phase_2c.outcomes import (
    _now_iso,
    _resolve_seed_site,
    skipped_task_stanza,
)
from worldsim.phase_2.phase_2c.reddit_attribution import (
    _attach_gitlab_issue_note_state_probe_anchors,
    _attach_reddit_comment_attribution_contract,
    _copy_gitlab_issue_note_anchor_fields,
    _gitlab_issue_note_anchors_from_seed_metadata,
    _iter_final_state_reward_configs,
    _patch_reddit_submit_comment_state_probes,
    _reddit_seed_comment_ids_from_seed_metadata,
    _task_has_gitlab_create_issue_note_reward,
    _task_has_reddit_submit_comment_reward,
)
from worldsim.phases.phase_2_reachability import (
    ReachabilityOutcome,
)
from worldsim.phases.phase_2_render_check import (
    RenderOutcome,
)
from worldsim.seeding import SeedCleanupHandle, UnboundTokenError, apply_data_seed_async

cookie_domain_matches_host = _auth_preflight.cookie_domain_matches_host
playwright_storage_state = _auth_preflight.playwright_storage_state
playwright_storage_state_payload = _auth_preflight.playwright_storage_state_payload
read_storage_state_payload = _auth_preflight.read_storage_state_payload
resolve_agent_auth = _auth_preflight.resolve_agent_auth
resolve_agent_auth_headers = _auth_preflight.resolve_agent_auth_headers
resolve_storage_state_path = _auth_preflight.resolve_storage_state_path
storage_state_cookie_hosts = _auth_preflight.storage_state_cookie_hosts
storage_state_origin_hosts = _auth_preflight.storage_state_origin_hosts
storage_state_preflight_error_for_payload = (
    _auth_preflight.storage_state_preflight_error_for_payload
)
storage_state_recorded_hosts = _auth_preflight.storage_state_recorded_hosts

logger = logging.getLogger(__name__)

_PHASE_2C_REEXPORTED_HELPERS = (
    _exposure_contract_fingerprint_projection,
    _fingerprints_match,
    _first_method,
    _hours_since,
    _instances_digest,
    _benign_agent_response_expected_strings,
    _flatten_string_values,
    _metadata_path_value,
    _copy_gitlab_issue_note_anchor_fields,
    _seed_has_appended_surface_attack,
    _seed_surface_values,
    _gitlab_issue_note_anchors_from_seed_metadata,
    _iter_final_state_reward_configs,
    _patch_reddit_submit_comment_state_probes,
    _reddit_seed_comment_ids_from_seed_metadata,
    skipped_task_stanza,
    _strings_overlap,
    _task_has_gitlab_create_issue_note_reward,
    _task_has_reddit_submit_comment_reward,
    _verification_target_url,
)

_ORIGINAL_OUTCOME_FUNCS = {
    "_first_method": _outcomes._first_method,
}


def _sync_outcome_patches() -> None:
    _outcomes.EditorError = EditorError
    _outcomes.SeedCleanupHandle = SeedCleanupHandle
    for name, original in _ORIGINAL_OUTCOME_FUNCS.items():
        current = globals().get(name)
        wrapper = _OUTCOME_WRAPPERS.get(name)
        if current is wrapper:
            setattr(_outcomes, name, original)
        elif current is not None and current is not original:
            setattr(_outcomes, name, current)


def _host_fingerprint(instances_label: str, instances: list[dict[str, Any]]) -> dict[str, str]:
    commit = _git_head_short()
    return {
        "host_config": instances_label,
        "instances_digest": _instances_digest(instances),
        "editor_commit": commit,
        "dataset_commit": commit,
    }


def _git_head_short() -> str:
    override = os.environ.get("WORLDSIM_EDITOR_COMMIT_OVERRIDE")
    if override:
        return override.strip()
    repo_root = Path(__file__).resolve().parent.parent.parent
    sync_commit = _sync_stamp_commit(repo_root)
    if sync_commit:
        return sync_commit
    try:
        out = _fingerprints.subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            stderr=_fingerprints.subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, _fingerprints.subprocess.SubprocessError):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


def _sync_stamp_commit(repo_root: Path) -> str | None:
    return _fingerprints._sync_stamp_commit(repo_root)


def _infeasible_task(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _sync_outcome_patches()
    return _outcomes._infeasible_task(*args, **kwargs)


def _safe_cleanup(*args: Any, **kwargs: Any) -> None:
    _sync_outcome_patches()
    return _outcomes._safe_cleanup(*args, **kwargs)


_OUTCOME_WRAPPERS = {
    "_first_method": _first_method,
}

# Failpoint bases fired by ``write_json_atomic``. Callers wire these up so the
# crash-resume tests can interrupt each write.
FAILPOINT_DATASET = "phase_2.output.feasibility_dataset"
FAILPOINT_QUARANTINE = "phase_2.output.feasibility_quarantine"
FAILPOINT_REPORT = "phase_2.output.feasibility_report"
FAILPOINT_DROPPED_SOURCE_DATA = "phase_2.output.feasibility_dropped_source_data"

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
_PREFLIGHT_AUTH_REFRESH_LOCKS = _source_data_preflight._PREFLIGHT_AUTH_REFRESH_LOCKS
_BROWSER_PROBE_CAP = _probes._BROWSER_PROBE_CAP
_PROBE_LAUNCH_ARGS = _probes._PROBE_LAUNCH_ARGS
_RENDER_UNVERIFIED_KIND = _probes._RENDER_UNVERIFIED_KIND
_RENDER_UNVERIFIED_RETRY_DELAY_S = _probes._RENDER_UNVERIFIED_RETRY_DELAY_S
_SKIP_RENDER_CHECK_ENV = _probes._SKIP_RENDER_CHECK_ENV
derive_second_witness = _probes.derive_second_witness
render_signature = _probes.render_signature
render_signature_selection = _probes.render_signature_selection
verify_reachable = _probes.verify_reachable
verify_seed_renders = _probes.verify_seed_renders
_first_rendered_payload = _probes._first_rendered_payload
_phase4_exposure_inadmissible_reason = _probes._phase4_exposure_inadmissible_reason
_reachability_resource_for_task = _probes._reachability_resource_for_task
_render_check_inputs_from_metadata = _probes._render_check_inputs_from_metadata
_required_url_token = _probes._required_url_token
_selected_rendered_payload = _probes._selected_rendered_payload
_ORIGINAL_PROBE_PATCHES = {
    name: getattr(_probes, name)
    for name in (
        "_auth_probe_failure_kind",
        "_instance_benchmark_or_none",
        "_resolve_benign_browser_context_auth",
    )
}


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
            "WARP Taskgen Phase 2c"
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
        await _ensure_playwright_chromium_ready(async_playwright_factory)

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

    answer_collision_reason = _answer_target_collision_reason(task)
    if answer_collision_reason is not None:
        return _infeasible_task(
            task,
            kind="answer_target_collision",
            detail=answer_collision_reason,
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
            diagnostics = (
                render_outcome.diagnostics
                if render_outcome is not None and isinstance(render_outcome.diagnostics, dict)
                else {}
            )
            reddit_comment_visibility = diagnostics.get("reddit_seed_comment_visibility")
            if isinstance(reddit_comment_visibility, dict):
                feasibility["exposure"]["seeded_comment_visibility"] = dict(
                    reddit_comment_visibility
                )
    _attach_gitlab_issue_note_state_probe_anchors(result, metadata)
    attribution_problem = _attach_reddit_comment_attribution_contract(result, metadata)
    if attribution_problem is not None:
        return _infeasible_task(
            task,
            kind="reddit_comment_attribution_unbound",
            detail=attribution_problem,
            fingerprint=fingerprint,
            http_status=None,
            response_snippet=None,
            attempts=attempts,
            timestamp=_now_iso(),
            render_evidence=(render_outcome.evidence() if render_outcome else None),
            reachability_evidence=(
                reachability_outcome.evidence() if reachability_outcome else None
            ),
        )
    result["feasibility"] = feasibility
    return result


def _sync_probe_patches() -> None:
    for name in _probes._PATCHABLE_GLOBAL_NAMES:
        current = globals().get(name)
        if current is not None:
            wrapper = _PROBE_WRAPPERS.get(name)
            if wrapper is not None and current is wrapper:
                current = _ORIGINAL_PROBE_PATCHES[name]
            setattr(_probes, name, current)


async def _ensure_playwright_chromium_ready(async_playwright_factory: Any) -> None:
    probe_state = _probes._patchable_globals()
    _sync_probe_patches()
    try:
        return await _probes._ensure_playwright_chromium_ready(async_playwright_factory)
    finally:
        _probes._restore_patchable_globals(probe_state)


async def _run_reachability_check(*args: Any, **kwargs: Any) -> ReachabilityOutcome:
    probe_state = _probes._patchable_globals()
    _sync_probe_patches()
    try:
        return await _probes._run_reachability_check(*args, **kwargs)
    finally:
        _probes._restore_patchable_globals(probe_state)


def _preflight_request_context_options(
    instance: dict[str, Any],
    *,
    benchmark_root: Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    return _auth_preflight._preflight_request_context_options(
        instance,
        benchmark_root=benchmark_root,
    )


def _agent_auth_type(instance: dict[str, Any]) -> str:
    return _auth_preflight._agent_auth_type(instance)


def _instance_benchmark_or_none(instance: dict[str, Any]) -> str | None:
    probe_state = _probes._patchable_globals()
    _sync_probe_patches()
    try:
        return _probes._instance_benchmark_or_none(instance)
    finally:
        _probes._restore_patchable_globals(probe_state)


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
    return _auth_preflight._resolve_agent_auth_headers(agent_auth)


def _storage_state_preflight_error(path: str, instance: dict[str, Any]) -> str | None:
    return _auth_preflight._storage_state_preflight_error(path, instance)


def _read_storage_state_payload_for_preflight(
    path: str,
) -> tuple[dict[str, Any], str | None]:
    return _auth_preflight._read_storage_state_payload_for_preflight(path)


def _storage_state_preflight_error_for_payload(
    path_obj: Path,
    payload: dict[str, Any],
    instance: dict[str, Any],
) -> str | None:
    return _auth_preflight._storage_state_preflight_error_for_payload(
        path_obj,
        payload,
        instance,
    )


def _playwright_storage_state_for_preflight(path: str) -> tuple[str | dict[str, Any], str | None]:
    return _auth_preflight._playwright_storage_state_for_preflight(path)


def _playwright_storage_state_payload_for_preflight(
    path_obj: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    return _auth_preflight._playwright_storage_state_payload_for_preflight(
        path_obj,
        payload,
    )


def _storage_state_recorded_hosts(payload: dict[str, Any]) -> set[str]:
    return _auth_preflight._storage_state_recorded_hosts(payload)


def _storage_state_cookie_hosts(payload: dict[str, Any]) -> set[str]:
    return _auth_preflight._storage_state_cookie_hosts(payload)


def _storage_state_origin_hosts(payload: dict[str, Any]) -> set[str]:
    return _auth_preflight._storage_state_origin_hosts(payload)


def _cookie_domain_matches_host(domain: str, host: str) -> bool:
    return _auth_preflight._cookie_domain_matches_host(domain, host)


def _sync_source_data_preflight_patches() -> None:
    for name in (
        "_PREFLIGHT_AUTH_REFRESH_LOCKS",
        "_agent_auth_type",
        "_preflight_request_context_options",
        "infer_benchmark_name",
        "logger",
    ):
        current = globals().get(name)
        wrapper = _SOURCE_DATA_PREFLIGHT_WRAPPERS.get(name)
        if wrapper is not None and current is wrapper:
            current = getattr(_auth_preflight, name)
        setattr(_source_data_preflight, name, current)


_SOURCE_DATA_PREFLIGHT_WRAPPERS = {
    "_agent_auth_type": _agent_auth_type,
    "_preflight_request_context_options": _preflight_request_context_options,
}


async def _run_preflight_and_filter_raw(
    raw: list[dict[str, Any]],
    *,
    instances_by_site: dict[str, list[dict[str, Any]]],
    benchmark_root: Path | None = None,
) -> list[dict[str, Any]]:
    source_data_state = _source_data_preflight._patchable_globals()
    _sync_source_data_preflight_patches()
    try:
        return await _source_data_preflight._run_preflight_and_filter_raw(
            raw,
            instances_by_site=instances_by_site,
            benchmark_root=benchmark_root,
        )
    finally:
        _source_data_preflight._restore_patchable_globals(source_data_state)


def _resolve_benign_storage_state_path(instance: dict[str, Any]) -> str | None:
    return _auth_preflight._resolve_benign_storage_state_path(instance)


def _resolve_benign_browser_context_auth(
    instance: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    probe_state = _probes._patchable_globals()
    _sync_probe_patches()
    try:
        return _probes._resolve_benign_browser_context_auth(instance)
    finally:
        _probes._restore_patchable_globals(probe_state)


def _auth_probe_failure_kind(reason: str) -> str:
    probe_state = _probes._patchable_globals()
    _sync_probe_patches()
    try:
        return _probes._auth_probe_failure_kind(reason)
    finally:
        _probes._restore_patchable_globals(probe_state)


async def _run_render_check(*args: Any, **kwargs: Any) -> RenderOutcome:
    probe_state = _probes._patchable_globals()
    _sync_probe_patches()
    try:
        return await _probes._run_render_check(*args, **kwargs)
    finally:
        _probes._restore_patchable_globals(probe_state)


_PROBE_WRAPPERS = {
    "_auth_probe_failure_kind": _auth_probe_failure_kind,
    "_instance_benchmark_or_none": _instance_benchmark_or_none,
    "_resolve_benign_browser_context_auth": _resolve_benign_browser_context_auth,
}
