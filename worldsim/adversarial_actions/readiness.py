"""Deterministic action-readiness reports for Phase 2 artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from worldsim.adversarial_actions.catalog import describe_action_kind, get_action_spec
from worldsim.adversarial_actions.tier3 import option_marks_host_ready, tier3_action_readiness


def build_action_readiness_artifacts(
    *,
    site_name: str,
    contracts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Return action contracts, summary report, and ineligible rows for a site."""
    action_contracts: dict[str, dict[str, Any]] = {}
    ineligible: list[dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    by_tier: Counter[str] = Counter()
    by_signal: Counter[str] = Counter()
    by_status: Counter[str] = Counter()

    for task_id, contract in contracts.items():
        row = _action_contract_row(site_name=site_name, task_id=str(task_id), contract=contract)
        action_contracts[str(task_id)] = row
        readiness = row["readiness"]
        by_status[str(readiness["status"])] += 1
        if readiness["status"] != "ready":
            ineligible.append(row)
        for option in row["action_options"]:
            kind = str(option["kind"])
            by_kind[kind] += 1
            by_tier[f"tier_{option['impact_tier']}"] += 1
            by_signal[str(option["reward_signal"])] += 1

    report = {
        "site": site_name,
        "total_contracts": len(action_contracts),
        "ready_contracts": by_status.get("ready", 0),
        "ineligible_contracts": len(ineligible),
        "by_readiness_status": dict(sorted(by_status.items())),
        "by_action_kind": dict(sorted(by_kind.items())),
        "by_impact_tier": dict(sorted(by_tier.items())),
        "by_reward_signal": dict(sorted(by_signal.items())),
    }
    return action_contracts, report, ineligible


def _action_contract_row(
    *,
    site_name: str,
    task_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    options = _expanded_options(contract, site_name=site_name)
    unknown: list[str] = []
    disabled: list[str] = []
    for option in contract.get("adversarial_action_options") or []:
        if not isinstance(option, Mapping):
            continue
        kind = str(option.get("kind") or "")
        spec = get_action_spec(kind)
        if spec is None:
            unknown.append(kind)
        elif not spec.default_enabled and not _host_ready_option_recomputes(
            option,
            contract,
            site_name=site_name,
        ):
            disabled.append(kind)
    if unknown:
        readiness = {
            "status": "ineligible",
            "reason": "unknown_action_kind:" + ",".join(sorted(unknown)),
        }
    elif disabled:
        readiness = {
            "status": "ineligible",
            "reason": "disabled_action_kind:" + ",".join(sorted(disabled)),
        }
    elif not options:
        readiness = {
            "status": "ineligible",
            "reason": "no_host_compilable_action_options",
        }
    else:
        readiness = {
            "status": "ready",
            "reason": "host_compilable_action_options_present",
        }
    eligibility = contract.get("eligibility")
    eligibility_status = (
        str(eligibility.get("status") or "") if isinstance(eligibility, Mapping) else None
    )
    if eligibility_status and eligibility_status != "eligible":
        readiness = {
            "status": "ineligible",
            "reason": "exposure_contract_not_eligible:" + eligibility_status,
        }
    return {
        "task_id": task_id,
        "site": site_name,
        "contract_id": contract.get("contract_id") or contract.get("id"),
        "target_surface_id": contract.get("target_surface_id"),
        "editor_method": contract.get("editor_method"),
        "eligibility_status": eligibility_status,
        "adversarial_action_preference": contract.get("adversarial_action_preference"),
        "action_options": options,
        "readiness": readiness,
    }


def _expanded_options(contract: Mapping[str, Any], *, site_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    raw_options = contract.get("adversarial_action_options")
    if not isinstance(raw_options, list):
        return out
    for raw in raw_options:
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "")
        spec = get_action_spec(kind)
        host_ready = _host_ready_option_recomputes(raw, contract, site_name=site_name)
        if spec is None or (not spec.default_enabled and not host_ready):
            continue
        described = describe_action_kind(kind)
        if isinstance(raw.get("description"), str):
            described["description"] = raw["description"]
        if host_ready:
            described["host_ready"] = True
            for key in (
                "pilot_policy",
                "readiness_level",
                "readiness_reason",
                "fixture_kind",
                "fixture_scope",
                "setup_strategy",
                "cleanup_strategy",
                "readback_kind",
            ):
                if raw.get(key) is not None:
                    described[key] = raw[key]
        out.append(described)
    return out


def _host_ready_option_recomputes(
    option: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    site_name: str,
) -> bool:
    if not option_marks_host_ready(option):
        return False
    kind = str(option.get("kind") or "").strip()
    policy = str(option.get("pilot_policy") or "").strip()
    contract_with_site = dict(contract)
    contract_with_site.setdefault("site", site_name)
    readiness = tier3_action_readiness(
        kind,
        benign_task=None,
        exposure_contract=contract_with_site,
        policy=policy,
    )
    return readiness["status"] == "ready"
