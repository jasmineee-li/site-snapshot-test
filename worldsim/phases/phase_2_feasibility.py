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
from worldsim.seeding import SeedCleanupHandle, apply_data_seed_async

logger = logging.getLogger(__name__)

# Failpoint bases fired by ``write_json_atomic``. Callers wire these up so the
# crash-resume tests can interrupt each write.
FAILPOINT_DATASET = "phase_2.output.feasibility_dataset"
FAILPOINT_QUARANTINE = "phase_2.output.feasibility_quarantine"
FAILPOINT_REPORT = "phase_2.output.feasibility_report"


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

    fingerprint_base = _host_fingerprint(instances_label)

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

    sites_in_tasks = {str(task.get("site", "")).strip().lower() for task in raw if task.get("site")}
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

    async def worker(task: dict[str, Any], index: int) -> dict[str, Any]:
        async with semaphore:
            if stagger_delay:
                await asyncio.sleep(min(stagger_delay * index, stagger_delay * 10))
            site = str(task.get("site", "")).strip().lower()
            instance = instance_by_site.get(site)
            if instance is None:
                return _infeasible_task(
                    task,
                    kind="unsupported_site",
                    detail=f"no instance for site {site!r}",
                    fingerprint=fingerprint_base,
                    http_status=None,
                    response_snippet=None,
                    attempts=[],
                    timestamp=now_iso,
                )
            return await _verify_one(
                task,
                instance,
                retry_count=retry_count,
                fingerprint_base=fingerprint_base,
                ttl_hours=ttl_hours,
                force_reverify=force_reverify,
                cleanup_warnings=cleanup_warnings,
            )

    results = await asyncio.gather(
        *(worker(task, i) for i, task in enumerate(raw)),
        return_exceptions=False,
    )

    for result in results:
        status = result.get("feasibility", {}).get("status")
        site = str(result.get("site", "")).strip().lower() or "unknown"
        bucket = per_site.setdefault(
            site,
            {"verified": 0, "infeasible": 0, "skipped": 0},
        )
        if status == "verified":
            verified.append(result)
            bucket["verified"] += 1
        elif status == "infeasible":
            infeasible.append(result)
            bucket["infeasible"] += 1
        elif status == "skipped":
            skipped.append(result)
            bucket["skipped"] += 1
            verified.append(result)
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
) -> dict[str, Any]:
    seed = task.get("adversarial_data_seed") or {}
    editor_calls = seed.get("editor_calls") if isinstance(seed, dict) else None

    content_hash = _task_content_hash(editor_calls if isinstance(editor_calls, list) else [])
    fingerprint = dict(fingerprint_base)
    fingerprint["task_content_hash"] = content_hash

    decision = _idempotency_decision(
        task.get("feasibility"),
        current_fingerprint=fingerprint,
        ttl_hours=ttl_hours,
        force_reverify=force_reverify,
    )
    if decision == "skip":
        result = dict(task)
        prior = dict(task.get("feasibility") or {})
        prior["status"] = "skipped"
        prior["skipped_at"] = _now_iso()
        prior["reason"] = "fingerprint_match"
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

    try:
        handle = await retrying(
            lambda: apply_data_seed_async(seed, bound_instance),
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
    except ValueError as exc:
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

    try:
        if handle is None:
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
    finally:
        _safe_cleanup(handle, cleanup_warnings, task.get("id"))

    result = dict(task)
    result["feasibility"] = {
        "status": "verified",
        "verified_at": _now_iso(),
        "host_fingerprint": fingerprint,
        "attempts": attempts,
    }
    return result


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
) -> str:
    """Return ``"skip"`` to reuse the prior result, ``"verify"`` to re-run.

    Matches the truth table in §3.7:

    - Missing feasibility field → verify.
    - ``verified`` + fingerprint matches → skip (unless force_reverify).
    - ``verified`` + fingerprint drifts → re-verify unless TTL covers it.
    - ``infeasible`` (any fingerprint) → re-verify (platform may have
      changed its policy since).
    - ``unverified`` → verify.
    """
    if force_reverify:
        return "verify"
    if not isinstance(existing, dict):
        return "verify"
    status = existing.get("status")
    if status != "verified":
        return "verify"
    prior_fp = existing.get("host_fingerprint") or {}
    if not isinstance(prior_fp, dict):
        return "verify"
    if _fingerprints_match(prior_fp, current_fingerprint):
        return "skip"
    if ttl_hours is not None:
        verified_at = existing.get("verified_at")
        age = _hours_since(verified_at)
        if age is not None and age <= ttl_hours:
            return "skip"
    return "verify"


def _fingerprints_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("host_config", "editor_commit", "dataset_commit", "task_content_hash")
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


def _host_fingerprint(instances_label: str) -> dict[str, str]:
    commit = _git_head_short()
    return {
        "host_config": instances_label,
        "editor_commit": commit,
        "dataset_commit": commit,
    }


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
