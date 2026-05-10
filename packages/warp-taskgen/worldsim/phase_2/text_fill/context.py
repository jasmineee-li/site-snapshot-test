from __future__ import annotations

from typing import Any

from worldsim.phase_2.text_fill.voice import _classify_source_field


def _surface_route_context(task: dict[str, Any]) -> dict[str, str]:
    exposure_contract = task.get("exposure_contract")
    route = exposure_contract.get("surface_route") if isinstance(exposure_contract, dict) else None
    if not isinstance(route, dict):
        route = {}
    phase4 = (
        exposure_contract.get("phase4_exposure") if isinstance(exposure_contract, dict) else None
    )
    if not isinstance(phase4, dict):
        phase4 = {}
    evidence = route.get("route_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        evidence = _route_evidence_from_phase4(phase4)
    return {
        "entry_surface": _string_or_unknown(route.get("entry_surface")),
        "seed_surface": _string_or_unknown(
            route.get("seed_surface") or task.get("target_surface_id")
        ),
        "content_capacity": _string_or_unknown(
            route.get("content_capacity") or _content_capacity_for_surface(task)
        ),
        "route_evidence": evidence,
    }

def _route_evidence_from_phase4(phase4: dict[str, Any]) -> str:
    encounter_surface = str(phase4.get("encounter_surface") or "unknown")
    if phase4.get("requires_transition") is True:
        return f"{encounter_surface}; task-forced transition={phase4.get('transition_forced_by_task') is True}"
    return encounter_surface

def _content_capacity_for_surface(task: dict[str, Any]) -> str:
    source_field = task.get("source_field")
    if isinstance(source_field, str):
        return _classify_source_field(source_field)
    target_surface = str(task.get("target_surface_id") or "")
    if target_surface:
        return _classify_source_field(target_surface)
    return "unknown"

def _string_or_unknown(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"
