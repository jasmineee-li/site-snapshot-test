"""Append-only task-bank helpers for admitted WorldSim tasks."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from warp_taskgen.phases.phase_2_core_surfaces import retired_carrier_reason
from warp_taskgen.state import get_state_dir

TASK_BANK_SCHEMA_VERSION = 1
DEFAULT_TASK_BANK_RELATIVE_PATH = Path("task_bank") / "events.jsonl"


class TaskBankError(ValueError):
    """Raised when task-bank records are malformed or non-appendable."""


def default_task_bank_path() -> Path:
    return get_state_dir() / DEFAULT_TASK_BANK_RELATIVE_PATH


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def short_digest(payload: Any, *, chars: int = 16) -> str:
    return stable_digest(payload)[:chars]


def load_task_bank(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_event_ids: dict[str, int] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskBankError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        validate_task_bank_event(record, line_number=line_number)
        event_id = str(record["event_id"])
        first_line = seen_event_ids.get(event_id)
        if first_line is not None:
            raise TaskBankError(
                f"{path}:{line_number}: duplicate event_id {event_id!r} first seen on line {first_line}"
            )
        seen_event_ids[event_id] = line_number
        records.append(record)
    return records


def validate_task_bank_event(event: Any, *, line_number: int | None = None) -> None:
    prefix = f"line {line_number}: " if line_number is not None else ""
    if not isinstance(event, dict):
        raise TaskBankError(f"{prefix}event must be an object")
    required = {
        "schema_version",
        "event_id",
        "event_type",
        "created_at",
        "task_id",
        "site",
        "task_signature",
    }
    missing = sorted(field for field in required if field not in event)
    if missing:
        raise TaskBankError(f"{prefix}event missing required fields {missing}")
    if event.get("schema_version") != TASK_BANK_SCHEMA_VERSION:
        raise TaskBankError(f"{prefix}unsupported schema_version {event.get('schema_version')!r}")
    if event.get("event_type") not in {"admit_task", "phase4_result"}:
        raise TaskBankError(f"{prefix}unsupported event_type {event.get('event_type')!r}")
    for field in ("event_id", "created_at", "task_id", "site", "task_signature"):
        if not isinstance(event.get(field), str) or not event[field].strip():
            raise TaskBankError(f"{prefix}{field} must be a non-empty string")


def append_task_bank_events(path: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []
    existing = load_task_bank(path)
    existing_event_ids = {str(event["event_id"]) for event in existing}
    existing_task_signatures = {
        str(event.get("task_signature"))
        for event in existing
        if event.get("event_type") == "admit_task"
    }
    appended: list[dict[str, Any]] = []
    for event in events:
        validate_task_bank_event(event)
        event_id = str(event["event_id"])
        if event_id in existing_event_ids:
            raise TaskBankError(f"duplicate event_id {event_id!r}")
        if event.get("event_type") == "admit_task":
            signature = str(event.get("task_signature"))
            if signature in existing_task_signatures:
                raise TaskBankError(f"duplicate admitted task_signature {signature!r}")
            existing_task_signatures.add(signature)
        existing_event_ids.add(event_id)
        appended.append(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in appended:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    return appended


def build_task_signature(task: dict[str, Any]) -> str:
    contract = carrier_contract_from_task(task)
    provenance = task_provenance(task)
    payload = {
        "benign_task_id": task.get("benign_task_id"),
        "instruction": task.get("instruction"),
        "site": task.get("site"),
        "origin": task.get("origin"),
        "route_id": task.get("route_id"),
        "start_urls": task.get("start_urls"),
        "benign_reward": (task.get("reward_function") or {}).get("benign_reward"),
        "carrier_contract": contract,
        "task_archetype": provenance.get("task_archetype"),
        "archetype_id": provenance.get("archetype_id"),
    }
    comparison_identity = _gitlab_compare_world_identity(task)
    if comparison_identity is not None:
        payload["gitlab_compare_world_identity"] = comparison_identity
    return short_digest(payload)


def build_archetype_signature(task: dict[str, Any]) -> str:
    provenance = task_provenance(task)
    payload = {
        "site": task.get("site"),
        "origin": task.get("origin"),
        "carrier_contract": carrier_contract_from_task(task),
        "task_archetype": provenance.get("task_archetype"),
        "archetype_id": provenance.get("archetype_id"),
        "answer_shape": provenance.get("answer_shape"),
    }
    return short_digest(payload)


def task_provenance(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("task_provenance")
    return dict(value) if isinstance(value, dict) else {}


def _gitlab_compare_world_identity(task: dict[str, Any]) -> Any | None:
    """Return full canonical comparison-world identity for task-bank signatures."""

    if task.get("site") != "gitlab":
        return None
    task_card_id = task.get("task_card_id")
    if not isinstance(task_card_id, str) or not task_card_id.strip():
        return None
    from warp_taskgen.phase_1.gitlab_compare_decide_generation import (
        gitlab_compare_world_identity,
    )

    candidate = task
    reward = task.get("reward_function")
    if isinstance(reward, Mapping) and isinstance(reward.get("benign_reward"), Mapping):
        candidate = dict(task)
        candidate["reward_function"] = reward["benign_reward"]
    return gitlab_compare_world_identity(
        candidate,
        task_card_id=task_card_id,
        act=isinstance(candidate.get("comparison_act_contract"), Mapping),
    )


def carrier_contract_from_task(task: dict[str, Any]) -> dict[str, Any]:
    contract = task.get("exposure_contract")
    surface_route: dict[str, Any] = {}
    if isinstance(contract, dict):
        raw_surface_route = contract.get("surface_route")
        if isinstance(raw_surface_route, dict):
            surface_route = raw_surface_route
    else:
        contract = {}
    return {
        "site": task.get("site"),
        "route_id": task.get("route_id"),
        "route_variant": task.get("route_variant")
        or contract.get("route_variant")
        or surface_route.get("route_variant"),
        "target_surface_id": task.get("target_surface_id") or contract.get("target_surface_id"),
        "editor_method": task.get("editor_method") or contract.get("editor_method"),
        "content_capacity": surface_route.get("content_capacity"),
        "transition_required": surface_route.get("transition_required"),
    }


def carrier_status_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    site = str(contract.get("site") or "").strip()
    surface = str(contract.get("target_surface_id") or "").strip()
    reason = retired_carrier_reason(site, surface)
    if reason is not None:
        return {"active": False, "reason": reason}
    return {"active": True}


def is_active_task_bank_event(event: dict[str, Any]) -> bool:
    if event.get("event_type") != "admit_task":
        return True
    status = event.get("carrier_status")
    if isinstance(status, dict) and status.get("active") is False:
        return False
    contract = event.get("carrier_contract")
    if isinstance(contract, dict):
        return (
            retired_carrier_reason(
                str(contract.get("site") or event.get("site") or ""),
                str(contract.get("target_surface_id") or ""),
            )
            is None
        )
    return True


def admitted_task_event(
    task: dict[str, Any],
    *,
    run_dir: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    task_id = str(task.get("id") or "").strip()
    if not task_id:
        raise TaskBankError("admitted task is missing id")
    signature = build_task_signature(task)
    archetype_signature = build_archetype_signature(task)
    provenance = task_provenance(task)
    carrier_contract = carrier_contract_from_task(task)
    event = {
        "schema_version": TASK_BANK_SCHEMA_VERSION,
        "event_type": "admit_task",
        "event_id": f"admit:{task_id}:{signature}",
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "task_id": task_id,
        "benign_task_id": str(task.get("benign_task_id") or ""),
        "site": str(task.get("site") or ""),
        "origin": str(task.get("origin") or ""),
        "carrier_contract": carrier_contract,
        "carrier_status": carrier_status_from_contract(carrier_contract),
        "task_archetype": provenance.get("task_archetype") or {},
        "archetype_id": str(provenance.get("archetype_id") or ""),
        "task_signature": signature,
        "archetype_signature": archetype_signature,
        "phase2c_status": str((task.get("feasibility") or {}).get("status") or ""),
    }
    validate_task_bank_event(event)
    return event


def load_phase2c_admitted_tasks(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "phase_2" / "adversarial_tasks.json"
    if not path.exists():
        raise TaskBankError(f"Phase 2c artifact not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskBankError(f"Phase 2c artifact is invalid JSON: {path}: {exc}") from exc
    if not isinstance(data, list):
        raise TaskBankError(f"Phase 2c artifact must be a JSON array: {path}")
    admitted = []
    for item in data:
        if not isinstance(item, dict):
            continue
        feasibility = item.get("feasibility")
        status = feasibility.get("status") if isinstance(feasibility, dict) else None
        if status in {None, "", "verified"}:
            admitted.append(item)
    return admitted


def admitted_events_from_phase2c_run(
    run_dir: Path,
    *,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    return [
        admitted_task_event(task, run_dir=run_dir, created_at=created_at)
        for task in load_phase2c_admitted_tasks(run_dir)
    ]


def summarize_task_bank(events: list[dict[str, Any]]) -> dict[str, Any]:
    admitted = [event for event in events if event.get("event_type") == "admit_task"]
    active_admitted = [event for event in admitted if is_active_task_bank_event(event)]
    retired_admitted = [event for event in admitted if not is_active_task_bank_event(event)]
    phase4 = [event for event in events if event.get("event_type") == "phase4_result"]
    return {
        "total_events": len(events),
        "admitted_tasks": len(admitted),
        "active_admitted_tasks": len(active_admitted),
        "retired_admitted_tasks": len(retired_admitted),
        "phase4_results": len(phase4),
        "by_site": dict(Counter(str(event.get("site") or "unknown") for event in admitted)),
        "by_origin": dict(Counter(str(event.get("origin") or "unknown") for event in admitted)),
        "by_surface": dict(
            Counter(
                str((event.get("carrier_contract") or {}).get("target_surface_id") or "unknown")
                for event in admitted
                if isinstance(event.get("carrier_contract"), dict)
            )
        ),
        "active_by_surface": dict(
            Counter(
                str((event.get("carrier_contract") or {}).get("target_surface_id") or "unknown")
                for event in active_admitted
                if isinstance(event.get("carrier_contract"), dict)
            )
        ),
        "retired_by_surface": dict(
            Counter(
                str((event.get("carrier_contract") or {}).get("target_surface_id") or "unknown")
                for event in retired_admitted
                if isinstance(event.get("carrier_contract"), dict)
            )
        ),
        "by_archetype": dict(
            Counter(str(event.get("archetype_id") or "unknown") for event in admitted)
        ),
        "latest_event_id": str(events[-1].get("event_id")) if events else None,
        "latest_created_at": str(events[-1].get("created_at")) if events else None,
    }
