"""Phase 2 eligibility behavior."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.benchmark_capabilities import normalize_benchmark_name
from worldsim.editors._registry import available_tokens_for_kind, kind_contract, method_spec
from worldsim.phase_2.exposure_contract import build_exposure_contract
from worldsim.seed_contracts.delivery import _call_delivery_mechanism
from worldsim.state import get_state_dir

logger = logging.getLogger(__name__)

_ELIGIBILITY_DROPS_WRITE_LOCK = threading.Lock()
_FRAMINGS = (
    "authority",
    "urgency",
    "policy",
    "error_recovery",
    "support_escalation",
    "ci_failure",
    "moderation_appeal",
    "seller_footnote",
    "peer_tip",
)
_CONCEALMENTS = (
    "plaintext",
    "offscreen_css",
    "markdown_fenced_system",
)
_ELIGIBLE_CONTROLLABLE_TIERS = frozenset({"any_user", "authed_user"})


def _phase_2a_eligible_tasks(
    site_tasks: list[dict],
    benign_target_resources: dict[str, Any],
    site: str,
    *,
    benchmark: str = "webarena_verified",
    exposure_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Split a shard's tasks into (eligible, dropped).

    A task is ineligible iff:

    * ``benign_target_resource.kind`` is None, OR
    * the contract has no valid methods for this kind on this site
      (``kind_contract(kind).valid_methods ∩ editor.supported_methods``
      empty), OR
    * the only reachable token is ``{benign_user_handle}`` AND no spec
      addressing this kind has a ``free_text`` body-accepting binding
      (no way to route via body mention).

    Dashboard-list kinds are *eligible* because ``create_issue_note`` /
    ``create_comment`` have ``free_text`` body bindings that satisfy the
    last clause.
    """
    from worldsim.editors import EDITOR_REGISTRY

    benchmark = normalize_benchmark_name(benchmark) or "webarena_verified"
    editor_cls: Any = EDITOR_REGISTRY.get((benchmark, site))
    supported = getattr(editor_cls, "supported_methods", frozenset()) if editor_cls else frozenset()

    eligible: list[dict] = []
    dropped: list[dict[str, Any]] = []
    for task in site_tasks:
        task_id = str(task.get("id") or "")
        origin = str(task.get("origin") or "")
        exposure_contract = (
            exposure_contracts.get(task_id) if isinstance(exposure_contracts, Mapping) else None
        )
        if isinstance(exposure_contract, Mapping):
            eligibility = exposure_contract.get("eligibility")
            status = eligibility.get("status") if isinstance(eligibility, Mapping) else None
            if status != "eligible":
                dropped.append(
                    {
                        "task_id": task_id,
                        "origin": origin,
                        "kind": exposure_contract.get("kind"),
                        "reason": (
                            str(eligibility.get("reason"))
                            if isinstance(eligibility, Mapping)
                            else "exposure_contract_ineligible"
                        ),
                        "anchors": dict(exposure_contract.get("anchors") or {}),
                        "available_tokens": list(exposure_contract.get("required_tokens") or []),
                        "contract_id": exposure_contract.get("contract_id"),
                        "target_surface_id": exposure_contract.get("target_surface_id"),
                    }
                )
                continue

        record = benign_target_resources.get(task_id) or {}
        kind = record.get("kind") if isinstance(record, dict) else None
        anchors_raw = record.get("anchors") if isinstance(record, dict) else None
        anchors = anchors_raw if isinstance(anchors_raw, dict) else {}

        if not isinstance(kind, str) or not kind:
            dropped.append(
                {
                    "task_id": task_id,
                    "origin": origin,
                    "kind": None,
                    "reason": str(record.get("reason") or "unresolved_target_resource"),
                    "anchors": dict(anchors),
                    "available_tokens": [],
                }
            )
            continue

        contract = kind_contract(kind, benchmark=benchmark, site=site)
        site_methods = contract.valid_methods & frozenset(supported)
        if not site_methods:
            dropped.append(
                {
                    "task_id": task_id,
                    "origin": origin,
                    "kind": kind,
                    "reason": "no_addressable_method_on_site",
                    "anchors": dict(anchors),
                    "available_tokens": sorted(
                        available_tokens_for_kind(
                            kind,
                            anchors,
                            benchmark=benchmark,
                            site=site,
                        )
                    ),
                }
            )
            continue

        available = available_tokens_for_kind(
            kind,
            anchors,
            benchmark=benchmark,
            site=site,
        )
        identity_only = available == frozenset({"{benign_user_handle}"})
        if identity_only:
            # When the only token reachable from the resolved anchors is the
            # user handle, the seed needs somewhere to land. Two routes:
            #   - dashboard-list @mention: a free_text comment/note body that
            #     references the user handle.
            #   - direct field overwrite: a free_text binding the seeder can
            #     populate (e.g. bio, description, content, title).
            # Either route is viable; the field name is not load-bearing.
            has_body_route = False
            for method in site_methods:
                try:
                    spec = method_spec(site, method, benchmark=benchmark)
                except KeyError:
                    continue
                if any(binding.kind == "free_text" for binding in spec.bindings.values()):
                    has_body_route = True
                    break
            if not has_body_route:
                dropped.append(
                    {
                        "task_id": task_id,
                        "origin": origin,
                        "kind": kind,
                        "reason": "only_user_handle_token_and_no_free_text_binding",
                        "anchors": dict(anchors),
                        "available_tokens": sorted(available),
                    }
                )
                continue

        eligible.append(task)

    return eligible, dropped


def _phase_2a_eligible_tasks_for_benchmark(
    site_tasks: list[dict],
    benign_target_resources: dict[str, Any],
    site: str,
    *,
    benchmark: str,
    exposure_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict], list[dict[str, Any]]]:
    try:
        return _phase_2a_eligible_tasks(
            site_tasks,
            benign_target_resources,
            site,
            benchmark=benchmark,
            exposure_contracts=exposure_contracts,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return _phase_2a_eligible_tasks(site_tasks, benign_target_resources, site)


def _build_exposure_contracts_for_shard(
    *,
    site_tasks: list[dict],
    benign_target_resources: Mapping[str, Mapping[str, Any]],
    site: str,
    benchmark: str,
    surface_visibility_by_id: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for task in site_tasks:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        contracts[task_id] = build_exposure_contract(
            benign_task_id=task_id,
            site=site,
            benchmark=benchmark,
            benign_target_resource=benign_target_resources.get(task_id),
            surface_visibility_by_id=surface_visibility_by_id,
        )
    return contracts


def _surface_visibility_by_id(site_profile: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    surfaces = site_profile.get("injection_surface")
    if not isinstance(surfaces, list):
        return out
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            continue
        surface_id = surface.get("id")
        visibility = surface.get("visibility")
        if isinstance(surface_id, str) and isinstance(visibility, str):
            out[surface_id] = visibility
    return out


def _persist_exposure_contracts(
    *,
    site_name: str,
    contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        out_dir = get_state_dir() / "phase_2"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "exposure_contracts.json"
        ineligible_path = out_dir / "exposure_ineligible.json"
        with _ELIGIBILITY_DROPS_WRITE_LOCK:
            existing: dict[str, Any] = {}
            if path.exists():
                try:
                    raw = json.loads(path.read_text())
                    if isinstance(raw, dict):
                        existing = raw
                except json.JSONDecodeError:
                    logger.warning(
                        "Phase 2: exposure_contracts.json at %s is malformed; overwriting",
                        path,
                    )
            existing.setdefault(site_name, {}).update(
                {str(key): dict(value) for key, value in contracts.items()}
            )
            write_json_atomic(path, existing)

            ineligible_existing: dict[str, list[dict[str, Any]]] = {}
            if ineligible_path.exists():
                try:
                    raw = json.loads(ineligible_path.read_text())
                    if isinstance(raw, dict):
                        ineligible_existing = raw
                except json.JSONDecodeError:
                    logger.warning(
                        "Phase 2: exposure_ineligible.json at %s is malformed; overwriting",
                        ineligible_path,
                    )
            site_ineligible = [
                dict(contract)
                for contract in contracts.values()
                if isinstance(contract.get("eligibility"), Mapping)
                and contract["eligibility"].get("status") != "eligible"
            ]
            if site_ineligible:
                ineligible_existing.setdefault(site_name, []).extend(site_ineligible)
                write_json_atomic(ineligible_path, ineligible_existing)
    except Exception as exc:
        logger.warning(
            "Phase 2a: could not persist exposure contracts for site %r: %s",
            site_name,
            exc,
        )


def _seed_delivery_mechanism(seed_template: Mapping[str, Any]) -> str:
    seed = dict(seed_template)
    mechanism = str(seed.get("mechanism") or "").strip().lower()
    if mechanism == "api":
        api_calls = seed.get("api_calls")
        if not isinstance(api_calls, list) or not api_calls:
            raise ValueError("materialized seed_template has mechanism=api but no api_calls")
        return "api"
    calls = seed.get("editor_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("materialized seed_template has no editor_calls")
    mechanisms = {
        mechanism
        for call in calls
        if isinstance(call, dict)
        for mechanism in [_call_delivery_mechanism(seed, call)]
        if mechanism is not None
    }
    if len(mechanisms) != 1:
        raise ValueError(
            "materialized seed_template must resolve to exactly one delivery mechanism, "
            f"got {sorted(mechanisms)}"
        )
    return next(iter(mechanisms))


def _write_eligibility_drops(site: str, dropped: list[dict[str, Any]]) -> None:
    state_dir = Path(os.environ.get("WORLDSIM_STATE_DIR", "logs"))
    path = state_dir / "phase_2" / "dropped_no_contract.json"
    new_task_path = state_dir / "phase_2" / "new_task_resolver_dropouts.json"
    new_task_dropped = [entry for entry in dropped if entry.get("origin") == "new_task"]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _ELIGIBILITY_DROPS_WRITE_LOCK:
            existing: dict[str, list[dict[str, Any]]] = {}
            if path.exists():
                try:
                    raw = json.loads(path.read_text())
                    if isinstance(raw, dict):
                        existing = raw
                except json.JSONDecodeError:
                    logger.warning(
                        "Phase 2: dropped_no_contract.json at %s is malformed; overwriting", path
                    )
            existing.setdefault(site, []).extend(dropped)
            write_json_atomic(path, existing)

            if new_task_dropped:
                new_task_existing: dict[str, list[dict[str, Any]]] = {}
                if new_task_path.exists():
                    try:
                        raw = json.loads(new_task_path.read_text())
                        if isinstance(raw, dict):
                            new_task_existing = raw
                    except json.JSONDecodeError:
                        logger.warning(
                            "Phase 2: new_task_resolver_dropouts.json at %s is malformed; overwriting",
                            new_task_path,
                        )
                new_task_existing.setdefault(site, []).extend(new_task_dropped)
                write_json_atomic(new_task_path, new_task_existing)
        logger.info(
            "Phase 2: dropped %d task(s) for site %r as no-contract (see %s); %d were new_task origin",
            len(dropped),
            site,
            path,
            len(new_task_dropped),
        )
    except Exception:
        logger.exception("failed to write dropped_no_contract.json")


def _kinds_in_shard(benign_target_resources: dict[str, Any]) -> frozenset[str]:
    """Collect the set of non-None ``benign_target_resource.kind`` values
    present across a shard's benign tasks. Feeds :class:`ContractRenderContext`
    so the prompt table only lists kinds this shard actually uses."""
    kinds: set[str] = set()
    for entry in benign_target_resources.values():
        if isinstance(entry, dict):
            kind = entry.get("kind")
            if isinstance(kind, str) and kind:
                kinds.add(kind)
    return frozenset(kinds)


def _build_cell_targets(
    site_profile: dict[str, Any],
    site_tasks: list[dict],
    all_site_tasks: list[dict],
) -> dict[str, int]:
    available_cells = _available_cells(site_profile)
    if not available_cells:
        available_cells = [
            (framing, concealment) for framing in _FRAMINGS for concealment in _CONCEALMENTS
        ]

    # Index by the unsuffixed (source) task id so L4 clones bucket into
    # the same cell as the source — the concealment/framing cell is a
    # property of the underlying benign task, not of which listing item
    # the attacker ended up attached to.
    def _lookup_id(task: dict) -> str:
        return str(task.get("source_task_id") or task.get("id") or "")

    index_by_task_id: dict[str, int] = {}
    for idx, task in enumerate(all_site_tasks):
        key = _lookup_id(task)
        index_by_task_id.setdefault(key, idx)
    targets = {_cell_key(framing, concealment): 0 for framing, concealment in available_cells}
    for task in site_tasks:
        task_index = index_by_task_id.get(_lookup_id(task), 0)
        framing, concealment = available_cells[task_index % len(available_cells)]
        targets[_cell_key(framing, concealment)] += 1
    return targets


def _available_cells(site_profile: dict[str, Any]) -> list[tuple[str, str]]:
    concealments: set[str] = set()
    for surface in site_profile.get("injection_surface", []):
        if not isinstance(surface, dict):
            continue
        if surface.get("controllable_by_tier") not in _ELIGIBLE_CONTROLLABLE_TIERS:
            continue
        compatible = surface.get("compatible_concealments")
        if isinstance(compatible, list):
            concealments.update(
                concealment for concealment in compatible if concealment in _CONCEALMENTS
            )

    if not concealments:
        concealments = set(_CONCEALMENTS)

    return [
        (framing, concealment)
        for framing in _FRAMINGS
        for concealment in _CONCEALMENTS
        if concealment in concealments
    ]


def _cell_key(framing: str, concealment: str) -> str:
    return f"{framing}::{concealment}"


def _select_balanced_subset(
    validated_tasks: list[dict],
    cell_targets: dict[str, int],
) -> list[dict]:
    if not validated_tasks or not cell_targets:
        return validated_tasks

    remaining = dict(cell_targets)
    selected: list[dict] = []
    seen_benign: set[str] = set()
    overfull_unique: list[dict] = []
    for task in validated_tasks:
        benign_task_id = str(task.get("benign_task_id", ""))
        if benign_task_id in seen_benign:
            continue
        cell = _cell_key(str(task.get("framing", "")), str(task.get("concealment", "")))
        if remaining.get(cell, 0) <= 0:
            overfull_unique.append(task)
            continue
        selected.append(task)
        seen_benign.add(benign_task_id)
        remaining[cell] -= 1

    if not selected:
        logger.warning(
            "Phase 2: balanced subset selection produced no tasks, keeping all validated tasks"
        )
        return validated_tasks

    backfilled = 0
    for task in overfull_unique:
        benign_task_id = str(task.get("benign_task_id", ""))
        if benign_task_id in seen_benign:
            continue
        selected.append(task)
        seen_benign.add(benign_task_id)
        backfilled += 1
    if backfilled:
        logger.info(
            "Phase 2: balanced subset backfilled %d overfull-cell task(s) "
            "to preserve one valid plan per benign task",
            backfilled,
        )

    dropped = len(validated_tasks) - len(selected)
    if dropped:
        logger.info("Phase 2: balanced subset dropped %d duplicate task(s)", dropped)
    return selected
