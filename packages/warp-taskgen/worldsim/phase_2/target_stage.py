"""Phase 2 target_stage behavior."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.phase_2.output import _effective_task_site
from worldsim.phase_2.phase_2c.types import FeasibilityReport
from worldsim.phase_2.target_inputs import (
    _instance_bearer_tokens_ready,
    _instance_lacks_benign_probe_auth,
    _l1_l2_resources_dict,
    _l1_l2_resources_with_probe_fail_closed,
)
from worldsim.phase_2.target_resolution.constants import (
    PHASE_2A_SYNTHETIC_PLACEHOLDERS as _PHASE_2A_SYNTHETIC_PLACEHOLDERS,
)
from worldsim.phase_2.target_resolution.runner import resolve_tasks
from worldsim.state import get_state_dir

logger = logging.getLogger(__name__)
_TARGET_RESOLUTION_WRITE_LOCK = threading.Lock()
L4_TASK_ID_SUFFIX = "_l4_"
_L4_CLONE_BENIGN_TASK_ID_RE = re.compile(r"^(?P<source>.+)_l4_(?P<index>\d+)$")


def _report_summary_dict(
    report: FeasibilityReport,
    *,
    instances_path: str,
    dropped_source_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    active_dropped_source_data = (
        report.dropped_source_data if dropped_source_data is None else dropped_source_data
    )
    source_data_dropped_by_kind: dict[str, int] = {}
    for record in active_dropped_source_data:
        issue = record.get("source_data_issue") if isinstance(record, dict) else None
        kind = str(issue.get("kind") or "unknown") if isinstance(issue, dict) else "unknown"
        source_data_dropped_by_kind[kind] = source_data_dropped_by_kind.get(kind, 0) + 1
    return {
        "generated_at": _utcnow_iso(),
        "instances": instances_path,
        "host_fingerprint": report.host_fingerprint,
        "elapsed_seconds": round(report.elapsed_seconds, 3),
        "phase_2_status": report.phase_2_status,
        "verified_count": len(report.verified),
        "infeasible_count": len(report.infeasible),
        "skipped_already_verified_count": len(report.skipped_already_verified),
        "checkpoint_reused_count": int(report.reused_checkpoints),
        "cleanup_warnings": list(report.cleanup_warnings),
        "per_site": report.per_site_counts,
        "source_data_dropped_count": len(active_dropped_source_data),
        "source_data_dropped_by_kind": source_data_dropped_by_kind,
    }


def _utcnow_iso() -> str:

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_benign_task_id(
    task: Mapping[str, Any],
    *,
    expected_ids: set[str] | None = None,
) -> str:
    """Return the original benign-task id for L4-expanded tasks.

    During Phase 2a we temporarily clone benign tasks with ids like
    ``123_l4_0`` so the planner can keep multiple listing items distinct
    inside one shard. Those suffixed ids must not survive into the final
    Phase 2 artifacts: Phase 3/4 link adversarial tasks back to the Phase 1
    benign dataset via ``benign_task_id``, which only knows the original
    unsuffixed ids.

    Freshly generated L4 plans/tasks always carry
    ``benign_target_resource.layer == "L4"``. For backward-compatible reuse
    of datasets written by buggy builds, also normalize a suffixed id when
    its stripped source id is in ``expected_ids``.
    """
    raw = str(task.get("benign_task_id") or "")
    match = _L4_CLONE_BENIGN_TASK_ID_RE.fullmatch(raw)
    if match is None:
        return raw
    source = match.group("source")
    resource = task.get("benign_target_resource")
    layer = resource.get("layer") if isinstance(resource, dict) else None
    if layer == "L4":
        return source
    if expected_ids is not None and source in expected_ids:
        return source
    return raw


def _normalize_l4_benign_task_ids_in_place(
    tasks: list[dict[str, Any]],
    *,
    expected_ids: set[str] | None = None,
) -> None:
    for task in tasks:
        if not isinstance(task, dict):
            continue
        canonical = _canonical_benign_task_id(task, expected_ids=expected_ids)
        if canonical:
            task["benign_task_id"] = canonical


async def _resolve_benign_target_resources_for_shard(
    *,
    site_tasks: list[dict],
    instance: Mapping[str, Any] | None,
    site_name: str,
    label: str,
    benchmark: str = "webarena_verified",
) -> tuple[list[dict], dict[str, dict[str, Any]]]:
    """Resolve the shard's benign-target resources and expand L4 listings.

    Returns ``(expanded_site_tasks, resources)``. When no live instance
    is configured, the expanded list equals the input and resources
    come from the offline L1/L2 path. When an instance is present, the
    async :func:`resolve_tasks` runs the full L1/L2/L3/L4 pipeline:

    * L3 turns intent-only tasks into concrete-kind records with real
      anchors via Anthropic intent-parse + live probe.
    * L4 fans listing-kind records out to N concrete items; each fan-out
      clones the benign task dict with a suffixed ID
      (``"{task_id}_l4_{i}"``) and preserves the original via
      ``source_task_id`` so downstream code that groups by the original
      task can recover the mapping.

    Token acquisition is lazy per-shard and idempotent via
    :func:`acquire_tokens_for_instances`. Any resolver fault falls back
    to the L1/L2-only path with a warning; shards never crash on
    classifier, probe, or token errors.

    The resolved map is mirrored to
    ``logs/<run>/phase_2/target_resolution/<site>.json`` for inspection.
    """
    if instance is None:
        return list(site_tasks), _l1_l2_resources_dict(site_tasks, benchmark=benchmark)

    if _instance_lacks_benign_probe_auth(instance):
        logger.warning(
            "Phase 2a: site %r instance exposes api_auth without benign auth; "
            "falling back to L1/L2 for L3/L4 resolution",
            site_name,
        )
        resources = _l1_l2_resources_with_probe_fail_closed(
            site_tasks,
            reason="missing benign auth for live L3/L4 probe",
            benchmark=benchmark,
        )
        return list(site_tasks), resources

    # Acquire API tokens lazily on first use per-run; mirrors Phase 2c
    # and Phase 4's pattern. ``acquire_tokens_for_instances`` is
    # idempotent (no-op when already stamped).
    if not _instance_bearer_tokens_ready(instance):
        try:
            token_errors = acquire_tokens_for_instances([instance], auth_fields=("auth",))
        except Exception as exc:
            logger.warning(
                "Phase 2a: token acquisition raised for site %r; falling back to L1/L2: %s",
                site_name,
                exc,
            )
            token_errors = ["exception during token acquisition"]
        if token_errors:
            logger.warning(
                "Phase 2a: token acquisition failed for site %r (%s); falling back to L1/L2",
                site_name,
                "; ".join(token_errors),
            )
            return list(site_tasks), _l1_l2_resources_with_probe_fail_closed(
                site_tasks,
                reason="live L3/L4 probe unavailable after token acquisition failure",
                benchmark=benchmark,
            )

    try:
        enriched = await resolve_tasks(
            site_tasks,
            _PHASE_2A_SYNTHETIC_PLACEHOLDERS,
            instance,
            allow_layers=("L1", "L2", "L3", "L4"),
            benchmark=benchmark,
        )
    except Exception as exc:
        logger.warning(
            "Phase 2a: resolve_tasks raised for %r; falling back to L1/L2: %s",
            label,
            exc,
        )
        return list(site_tasks), _l1_l2_resources_with_probe_fail_closed(
            site_tasks,
            reason=f"live L3/L4 probe unavailable after resolver failure: {type(exc).__name__}",
            benchmark=benchmark,
        )

    # Build the expanded task list + resources map in lockstep. For a
    # task whose L4 returned N items, emit N cloned task dicts with
    # suffixed IDs; otherwise preserve the task ID as-is. Tasks missing
    # from ``enriched`` (probe returned empty, classifier failed hard)
    # flow through with their L1/L2 record so the eligibility filter
    # can drop them with a reason attached — no silent disappearance.
    expanded_tasks: list[dict] = []
    resources: dict[str, dict[str, Any]] = {}
    l4_fanout_count = 0
    l4_empty_exclusion_count = 0
    route_contract_preserved_count = 0
    for task in site_tasks:
        orig_id = str(task.get("id") or "")
        if not orig_id:
            continue
        records = enriched.get(orig_id)
        if _is_route_contracted_new_task(task):
            l1_l2 = _l1_l2_resources_dict([task], benchmark=benchmark)
            resource = l1_l2.get(orig_id)
            if resource is not None:
                if records:
                    resource = _merge_route_contract_l4_anchors(resource, records[0])
                editor_methods = _route_contract_editor_methods(task)
                if editor_methods:
                    resource["allowed_editor_methods"] = editor_methods
                expanded_tasks.append(task)
                resources[orig_id] = resource
                route_contract_preserved_count += 1
                continue
        if not records:
            # ``resolve_tasks`` omits only the L4-empty case: the benign task
            # resolved to a listing kind, but the live list contained zero
            # concrete items to attach to. Exclude it here rather than
            # reintroducing the pre-L4 listing stub, which would let a task
            # the dispatcher intentionally dropped leak back into Phase 2a.
            l4_empty_exclusion_count += 1
            continue
        if len(records) == 1:
            expanded_tasks.append(task)
            resources[orig_id] = records[0]
            continue
        # L4 fan-out: clone the benign task N times with suffixed IDs.
        for idx, record in enumerate(records):
            suffixed_id = f"{orig_id}{L4_TASK_ID_SUFFIX}{idx}"
            clone = dict(task)
            clone["id"] = suffixed_id
            clone["source_task_id"] = orig_id
            expanded_tasks.append(clone)
            resources[suffixed_id] = record
            l4_fanout_count += 1

    if l4_fanout_count:
        logger.info(
            "Phase 2a: L4 fan-out produced %d clones for site %r (shard %r, before=%d, after=%d)",
            l4_fanout_count,
            site_name,
            label,
            len(site_tasks),
            len(expanded_tasks),
        )
    if route_contract_preserved_count:
        logger.info(
            "Phase 2a: preserved %d route-contracted new task(s) from L4 fan-out for site %r (shard %r)",
            route_contract_preserved_count,
            site_name,
            label,
        )
    if l4_empty_exclusion_count:
        logger.info(
            "Phase 2a: excluded %d L4-empty task(s) for site %r (shard %r)",
            l4_empty_exclusion_count,
            site_name,
            label,
        )

    _persist_target_resolution(site_name=site_name, resources=resources)
    return expanded_tasks, resources


def _is_route_contracted_new_task(task: Mapping[str, Any]) -> bool:
    route_id = task.get("route_id")
    return (
        str(task.get("origin") or "") == "new_task"
        and isinstance(route_id, str)
        and bool(route_id.strip())
    )


def _route_contract_editor_methods(task: Mapping[str, Any]) -> list[str]:
    data_seed = task.get("data_seed")
    if not isinstance(data_seed, Mapping):
        return []
    calls = data_seed.get("editor_calls")
    if not isinstance(calls, list):
        return []
    methods: list[str] = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        method = call.get("method")
        if isinstance(method, str) and method.strip() and method.strip() not in methods:
            methods.append(method.strip())
    return methods


def _merge_route_contract_l4_anchors(
    resource: Mapping[str, Any],
    l4_record: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(resource)
    anchors = dict(merged.get("anchors") or {})
    l4_anchors = l4_record.get("anchors")
    if isinstance(l4_anchors, Mapping):
        anchors.update({str(key): value for key, value in l4_anchors.items()})
    merged["anchors"] = anchors
    for key in ("benign_read_url", "seeded_detail_url"):
        value = l4_record.get(key)
        if isinstance(value, str) and value.strip():
            merged.setdefault(key, value)
    merged["l4_anchor_source"] = "route_contract_top_result"
    return merged


def _persist_target_resolution(
    *,
    site_name: str,
    resources: Mapping[str, Mapping[str, Any]],
) -> None:
    """Mirror the per-site resolver output to
    ``logs/<run>/phase_2/target_resolution/<site>.json``.

    Best-effort; logging-only on write failure.
    """
    try:
        out_dir = get_state_dir() / "phase_2" / "target_resolution"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{site_name}.json"
        with _TARGET_RESOLUTION_WRITE_LOCK:
            merged: dict[str, Any] = {}
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                    if isinstance(existing, dict):
                        merged.update(existing)
                except json.JSONDecodeError:
                    logger.warning(
                        "Phase 2a: target_resolution at %s is malformed; overwriting", path
                    )
            merged.update({str(key): value for key, value in resources.items()})
            write_json_atomic(path, merged)
    except Exception as exc:
        logger.warning(
            "Phase 2a: could not persist target_resolution for site %r: %s",
            site_name,
            exc,
        )


def _reconstruct_orphan_start_urls(orphans: list[dict[str, Any]]) -> None:
    """Apply anchor-based URL reconstruction to recovered orphan tasks.

    Shard files on disk may pre-date commit ``4b023aea`` (Fix A) and
    carry bare-host ``benign_target_resource.start_url_resolved``
    ("https://gitlab.local" / "https://reddit.local"). Fix A only ran
    on fresh Phase 2a output; orphans pulled in from stale shards
    inherit the bare-host flaw, which makes Phase 2c navigate to the
    host root instead of the concrete entity where the seed was planted.

    Mirror the same logic the one-shot
    ``scripts/patch_benign_target_resource_urls.py`` applies, so the
    orchestrator's self-recovery is resilient to pre-Fix-A shards.
    Idempotent: a no-op when reconstruction matches the existing value
    or when anchors lack the fields needed to rebuild a concrete URL.
    """
    # Late import avoids a module-level cycle with target resolution,
    # which imports from this module for enrichment helpers.
    from worldsim.phase_2.target_resolution.reconstruction import (
        _reconstruct_start_url_from_anchors,
    )

    for task in orphans:
        resource = task.get("benign_target_resource")
        if not isinstance(resource, dict):
            continue
        kind = str(resource.get("kind") or "")
        anchors = resource.get("anchors") or {}
        if not kind or not isinstance(anchors, dict):
            continue
        site_kind = _effective_task_site(task)
        if site_kind not in {"gitlab", "reddit"}:
            continue
        reconstructed = _reconstruct_start_url_from_anchors(
            site_kind, kind, anchors, _PHASE_2A_SYNTHETIC_PLACEHOLDERS
        )
        if reconstructed and reconstructed != resource.get("start_url_resolved"):
            resource["start_url_resolved"] = reconstructed
            task["benign_target_resource"] = resource
        # Orphan shards from pre-template-standardization runs carry
        # ``project_path_template`` in ``editor_calls[].args`` but
        # never populated the paired ``project_name_template`` that
        # the GitLab editor's arg-validator requires. Both fields are
        # derivable from each other — the template is the leaf segment
        # of the path (see ``worldsim/editors/gitlab.py`` for the
        # forward derivation) — so backfill here keeps orphan recovery
        # symmetric with Phase 2a's original generation contract and
        # avoids the ``invalid_args: "project_id or
        # project_name_template is required"`` failure downstream.
        if site_kind == "gitlab":
            editor_calls = task.get("adversarial_data_seed", {}).get("editor_calls")
            if isinstance(editor_calls, list):
                for call in editor_calls:
                    if not isinstance(call, dict):
                        continue
                    args = call.get("args")
                    if not isinstance(args, dict):
                        continue
                    if args.get("project_name_template"):
                        continue
                    path_template = args.get("project_path_template")
                    if not isinstance(path_template, str) or "/" not in path_template:
                        continue
                    leaf = path_template.rsplit("/", 1)[-1]
                    if leaf:
                        args["project_name_template"] = leaf
