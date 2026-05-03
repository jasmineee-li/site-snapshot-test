"""Audit Phase 2 artifacts for Tier 3 action-pilot readiness.

This is a read-only guardrail for high-impact action pilots. It answers two
separate questions:

1. Do the exposure contracts contain host-ready Tier 3 action options under the
   named pilot policy?
2. If Phase 2 has already planned tasks, did it select and compile those Tier 3
   actions with request evidence plus final-state readback?

The audit does not mutate tasks, rewards, feasibility, or Phase 4 outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from worldsim.adversarial_actions import (
    annotate_exposure_contracts_with_action_policy,
    compile_adversarial_final_state_check,
    compile_adversarial_reward,
    get_action_spec,
)
from worldsim.adversarial_actions.tier3 import (
    TIER3_PILOT_POLICY,
    option_marks_host_ready,
    tier3_action_readiness,
)

TIER3_KIND = "modify_repository_content"


def analyze_run_dir(
    run_dir: Path,
    *,
    action_kind: str = TIER3_KIND,
    max_samples: int = 50,
    require_benign_action_evidence: bool = False,
) -> dict[str, Any]:
    phase_1_tasks = _read_json_if_exists(run_dir / "phase_1" / "benign_tasks.json")
    phase_2_tasks = _read_json_if_exists(run_dir / "phase_2" / "adversarial_tasks.json")
    exposure_contracts = _read_json_if_exists(run_dir / "phase_2" / "exposure_contracts.json")

    benign_tasks = phase_1_tasks if isinstance(phase_1_tasks, list) else []
    contracts = exposure_contracts if isinstance(exposure_contracts, Mapping) else {}
    adversarial_tasks = phase_2_tasks if isinstance(phase_2_tasks, list) else []
    return analyze_artifacts(
        exposure_contracts=contracts,
        benign_tasks=[task for task in benign_tasks if isinstance(task, Mapping)],
        adversarial_tasks=[task for task in adversarial_tasks if isinstance(task, Mapping)],
        artifact_paths={
            "run_dir": str(run_dir),
            "benign_tasks": str(run_dir / "phase_1" / "benign_tasks.json"),
            "exposure_contracts": str(run_dir / "phase_2" / "exposure_contracts.json"),
            "adversarial_tasks": str(run_dir / "phase_2" / "adversarial_tasks.json"),
        },
        action_kind=action_kind,
        max_samples=max_samples,
        require_benign_action_evidence=require_benign_action_evidence,
    )


def analyze_artifacts(
    *,
    exposure_contracts: Mapping[str, Any],
    benign_tasks: list[Mapping[str, Any]],
    adversarial_tasks: list[Mapping[str, Any]],
    artifact_paths: Mapping[str, str] | None = None,
    action_kind: str = TIER3_KIND,
    max_samples: int = 50,
    require_benign_action_evidence: bool = False,
) -> dict[str, Any]:
    benign_by_id = {str(task.get("id") or ""): task for task in benign_tasks}
    flattened_contracts = _flatten_contracts(exposure_contracts)
    annotated_contracts = _annotate_contracts_for_tier3(flattened_contracts, benign_tasks)
    exposure_report = _analyze_exposure_contracts(
        flattened_contracts,
        annotated_contracts=annotated_contracts,
        benign_by_id=benign_by_id,
        action_kind=action_kind,
        max_samples=max_samples,
    )
    selected_report = _analyze_selected_tasks(
        adversarial_tasks,
        benign_by_id=benign_by_id,
        action_kind=action_kind,
        max_samples=max_samples,
        require_benign_action_evidence=require_benign_action_evidence,
    )
    return {
        "artifact_paths": dict(artifact_paths or {}),
        "policy": TIER3_PILOT_POLICY,
        "tier3_kind": action_kind,
        "exposure_contracts": exposure_report,
        "selected_tasks": selected_report,
        "methodology": (
            "Read-only artifact audit. Counts are readiness diagnostics only; "
            "they do not alter Phase 2c admission, rewards, or Phase 4 scoring."
        ),
    }


def _analyze_exposure_contracts(
    rows: list[dict[str, Any]],
    *,
    annotated_contracts: dict[str, dict[str, Any]],
    benign_by_id: Mapping[str, Mapping[str, Any]],
    action_kind: str,
    max_samples: int,
) -> dict[str, Any]:
    total = len(rows)
    ready: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    by_site: dict[str, Counter[str]] = {}
    by_surface: dict[str, Counter[str]] = {}
    by_editor_method: dict[str, Counter[str]] = {}
    reasons = Counter()

    for row in rows:
        task_id = str(row.get("task_id") or "")
        contract = row["contract"]
        annotated = annotated_contracts.get(task_id) or {}
        benign_task = benign_by_id.get(task_id)
        readiness = tier3_action_readiness(
            action_kind,
            benign_task=benign_task,
            exposure_contract=contract,
            policy=TIER3_PILOT_POLICY,
        )
        host_ready_options = _host_ready_tier3_options(annotated)
        status = "ready" if host_ready_options else "ineligible"
        reason = readiness.get("reason") or "unknown"
        item = {
            "task_id": task_id,
            "site": _site_for_row(row, benign_task, contract),
            "contract_id": contract.get("contract_id") or contract.get("id"),
            "target_surface_id": contract.get("target_surface_id"),
            "editor_method": contract.get("editor_method"),
            "eligibility_status": _eligibility_status(contract),
            "readiness": {"status": status, "reason": reason},
            "host_ready_options": host_ready_options,
        }
        if status == "ready":
            ready.append(item)
        else:
            ineligible.append(item)
            reasons[reason] += 1
        _increment_bucket(by_site, item["site"], status)
        _increment_bucket(by_surface, str(item["target_surface_id"] or "unknown"), status)
        _increment_bucket(by_editor_method, str(item["editor_method"] or "unknown"), status)

    return {
        "total_contracts": total,
        "ready_contracts": len(ready),
        "ineligible_contracts": len(ineligible),
        "ineligible_reasons": dict(sorted(reasons.items())),
        "by_site": _counter_map_to_dict(by_site),
        "by_surface": _counter_map_to_dict(by_surface),
        "by_editor_method": _counter_map_to_dict(by_editor_method),
        "ready_samples": ready[:max_samples],
        "ineligible_samples": ineligible[:max_samples],
    }


def _analyze_selected_tasks(
    tasks: list[Mapping[str, Any]],
    *,
    benign_by_id: Mapping[str, Mapping[str, Any]],
    action_kind: str,
    max_samples: int,
    require_benign_action_evidence: bool,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    non_selected_reasons = Counter()
    risk_counts = Counter()
    by_site: dict[str, Counter[str]] = {}
    by_surface: dict[str, Counter[str]] = {}
    by_editor_method: dict[str, Counter[str]] = {}

    for index, task in enumerate(tasks):
        kind = _selected_action_kind(task)
        if kind != action_kind:
            non_selected_reasons[f"selected_action_not_tier3:{kind or 'missing'}"] += 1
            continue
        benign_task = _benign_task_for_compile(task, benign_by_id)
        compiled_reward, reward_error = _compile_task_reward(task, benign_task)
        final_state, final_state_error = _compile_task_final_state(task)
        evidence = _network_reward_evidence(compiled_reward)
        benign_evidence = _benign_action_evidence(task, benign_task)
        risks = _task_risks(
            task,
            evidence,
            final_state,
            reward_error,
            final_state_error,
            action_kind=action_kind,
            benign_action_evidence=benign_evidence,
            require_benign_action_evidence=require_benign_action_evidence,
        )
        ok = reward_error is None and final_state_error is None and not risks
        row = {
            "index": index,
            "id": task.get("id"),
            "benign_task_id": task.get("benign_task_id"),
            "site": str(task.get("site") or "unknown"),
            "origin": task.get("origin"),
            "surface": _task_surface(task),
            "route_id": task.get("route_id"),
            "editor_method": task.get("editor_method")
            or _mapping(task.get("exposure_contract")).get("editor_method"),
            "feasibility_status": _mapping(task.get("feasibility")).get("status"),
            "selected_action_kind": kind,
            "compiled_reward_evidence": evidence,
            "benign_action_evidence": benign_evidence,
            "benign_action_evidence_required": require_benign_action_evidence,
            "final_state_check": _final_state_evidence(final_state),
            "ok": ok,
            "risks": risks,
            "compile_errors": [
                error for error in (reward_error, final_state_error) if error is not None
            ],
        }
        selected.append(row)
        if not ok:
            failures.append(row)
        for risk in risks:
            risk_counts[risk] += 1
        _increment_bucket(by_site, row["site"], "passed" if ok else "failed")
        _increment_bucket(by_surface, str(row["surface"] or "unknown"), "passed" if ok else "failed")
        _increment_bucket(
            by_editor_method,
            str(row["editor_method"] or "unknown"),
            "passed" if ok else "failed",
        )

    return {
        "total_tasks": len(tasks),
        "selected_tier3_tasks": len(selected),
        "compile_ready_tasks": sum(1 for row in selected if row["ok"]),
        "failed_selected_tasks": len(failures),
        "benign_action_evidence_required": require_benign_action_evidence,
        "benign_action_evidence": sum(
            1 for row in selected if row["benign_action_evidence"].get("ok") is True
        ),
        "non_selected_reasons": dict(sorted(non_selected_reasons.items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "by_site": _counter_map_to_dict(by_site),
        "by_surface": _counter_map_to_dict(by_surface),
        "by_editor_method": _counter_map_to_dict(by_editor_method),
        "selected_samples": selected[:max_samples],
        "failure_samples": failures[:max_samples],
    }


def _compile_task_reward(
    task: Mapping[str, Any],
    benign_task: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        reward = compile_adversarial_reward(task, benign_task)
    except ValueError as exc:
        return None, str(exc)
    return reward, None


def _compile_task_final_state(
    task: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    check = compile_adversarial_final_state_check(task)
    if check is None:
        return None, "missing host-compiled final-state check"
    return check, None


def _network_reward_evidence(reward: Mapping[str, Any] | None) -> dict[str, Any]:
    expected = reward.get("expected") if isinstance(reward, Mapping) else None
    if not isinstance(expected, Mapping):
        return {
            "type": reward.get("type") if isinstance(reward, Mapping) else None,
            "has_post": False,
            "has_response_status": False,
            "has_body_evidence": False,
            "has_path_specific_url_pattern": False,
            "url_patterns": [],
        }
    patterns = _as_list(expected.get("url"))
    methods = {
        str(method).upper()
        for method in _as_list(expected.get("http_method"))
        if str(method or "").strip()
    }
    return {
        "type": reward.get("type") or reward.get("evaluator"),
        "http_method": expected.get("http_method"),
        "methods": sorted(methods),
        "has_post": "POST" in methods,
        "has_mutation_method": bool(methods & {"POST", "PUT", "PATCH", "DELETE"}),
        "response_status": expected.get("response_status"),
        "has_response_status": bool(expected.get("response_status")),
        "has_body_evidence": bool(expected.get("post_data") or expected.get("post_data_contains")),
        "post_data_contains_count": len(
            expected.get("post_data_contains")
            if isinstance(expected.get("post_data_contains"), list)
            else []
        ),
        "has_path_specific_url_pattern": any(
            isinstance(pattern, str)
            and pattern.strip()
            and pattern.strip() not in {".*", "*"}
            for pattern in patterns
        ),
        "url_patterns": patterns,
        "post_data": expected.get("post_data"),
        "post_data_contains": expected.get("post_data_contains"),
    }


def _benign_action_evidence(
    task: Mapping[str, Any],
    benign_task: Mapping[str, Any],
) -> dict[str, Any]:
    reward = _mapping(task.get("reward_function"))
    benign_reward = _mapping(reward.get("benign_reward")) or _mapping(
        benign_task.get("reward_function")
    )
    evidences = [
        _network_reward_evidence(config)
        for config in _network_reward_configs(benign_reward)
    ]
    ok = any(_network_evidence_is_action_specific(evidence) for evidence in evidences)
    return {
        "ok": ok,
        "network_reward_count": len(evidences),
        "evidence": evidences,
    }


def _network_reward_configs(reward: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    evals = reward.get("eval")
    if isinstance(evals, list):
        return [
            item
            for item in evals
            if isinstance(item, Mapping)
            and str(item.get("evaluator") or item.get("type") or "")
            in {"NetworkEventEvaluator", "network_event"}
        ]
    if str(reward.get("evaluator") or reward.get("type") or "") in {
        "NetworkEventEvaluator",
        "network_event",
    }:
        return [reward]
    return []


def _network_evidence_is_action_specific(evidence: Mapping[str, Any]) -> bool:
    return (
        evidence.get("has_post") is True
        and evidence.get("has_response_status") is True
        and evidence.get("has_body_evidence") is True
        and int(evidence.get("post_data_contains_count") or 0) >= 2
        and evidence.get("has_path_specific_url_pattern") is True
    )


def _final_state_evidence(check: Mapping[str, Any] | None) -> dict[str, Any]:
    expected = check.get("expected") if isinstance(check, Mapping) else None
    if not isinstance(expected, Mapping):
        return {"present": False}
    state_probe = expected.get("state_probe")
    network_event = expected.get("network_event")
    return {
        "present": True,
        "action_kind": expected.get("action_kind"),
        "has_network_event": isinstance(network_event, Mapping),
        "state_probe_kind": (
            state_probe.get("kind") if isinstance(state_probe, Mapping) else None
        ),
        "state_probe_site": (
            state_probe.get("site") if isinstance(state_probe, Mapping) else None
        ),
        "has_project_anchor": bool(
            isinstance(state_probe, Mapping)
            and (state_probe.get("project_id") or state_probe.get("project_path"))
        ),
    }


def _task_risks(
    task: Mapping[str, Any],
    evidence: Mapping[str, Any],
    final_state: Mapping[str, Any] | None,
    reward_error: str | None,
    final_state_error: str | None,
    *,
    action_kind: str,
    benign_action_evidence: Mapping[str, Any],
    require_benign_action_evidence: bool,
) -> list[str]:
    risks: list[str] = []
    if reward_error is not None:
        risks.append("reward_compile_failed")
    if final_state_error is not None:
        risks.append("final_state_compile_failed")
    if not _has_required_url_token(task):
        risks.append("missing_required_url_token")
    if evidence.get("type") != "NetworkEventEvaluator":
        risks.append("not_network_event_evaluator")
    if evidence.get("has_mutation_method") is not True:
        risks.append("missing_mutation_method")
    if evidence.get("has_response_status") is not True:
        risks.append("missing_response_status")
    if action_kind == "modify_repository_content":
        if evidence.get("has_post") is not True:
            risks.append("missing_post_method")
        if evidence.get("has_body_evidence") is not True:
            risks.append("missing_body_evidence")
        elif int(evidence.get("post_data_contains_count") or 0) < 2:
            risks.append("missing_file_path_body_evidence")
    if evidence.get("has_path_specific_url_pattern") is not True:
        risks.append("broad_only_endpoint_pattern")
    final_state_evidence = _final_state_evidence(final_state)
    if final_state_evidence.get("present") is not True:
        risks.append("missing_final_state_readback")
    elif final_state_evidence.get("has_project_anchor") is not True:
        risks.append("missing_final_state_project_anchor")
    feasibility_status = _mapping(task.get("feasibility")).get("status")
    if feasibility_status not in {None, "valid", "verified", "passed", "feasible"}:
        risks.append("non_verified_feasibility_status")
    if require_benign_action_evidence and benign_action_evidence.get("ok") is not True:
        risks.append("missing_benign_action_network_reward")
    return sorted(set(risks))


def _annotate_contracts_for_tier3(
    rows: list[dict[str, Any]],
    benign_tasks: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    contracts = {str(row.get("task_id") or ""): row["contract"] for row in rows}
    return annotate_exposure_contracts_with_action_policy(
        contracts,
        benign_tasks,
        policy=TIER3_PILOT_POLICY,
    )


def _flatten_contracts(exposure_contracts: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_key, value in exposure_contracts.items():
        if isinstance(value, Mapping) and all(isinstance(v, Mapping) for v in value.values()):
            for task_id, contract in value.items():
                if isinstance(contract, Mapping):
                    copied = dict(contract)
                    copied.setdefault("site", str(outer_key))
                    rows.append(
                        {"site": str(outer_key), "task_id": str(task_id), "contract": copied}
                    )
        elif isinstance(value, Mapping):
            rows.append(
                {
                    "site": str(value.get("site") or "unknown"),
                    "task_id": str(outer_key),
                    "contract": dict(value),
                }
            )
    return rows


def _host_ready_tier3_options(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    options = contract.get("adversarial_action_options")
    if not isinstance(options, list):
        return []
    out: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, Mapping):
            continue
        kind = str(option.get("kind") or "").strip()
        spec = get_action_spec(kind)
        if spec is None or spec.impact_tier != 3 or not option_marks_host_ready(option):
            continue
        out.append(dict(option))
    return out


def _benign_task_for_compile(
    task: Mapping[str, Any],
    benign_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    benign_id = str(task.get("benign_task_id") or "")
    if benign_id in benign_by_id:
        return benign_by_id[benign_id]
    reward = task.get("reward_function")
    benign_reward = _mapping(reward).get("benign_reward")
    return {
        "id": benign_id or task.get("id"),
        "site": task.get("site"),
        "reward_function": benign_reward or {},
    }


def _site_for_row(
    row: Mapping[str, Any],
    benign_task: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
) -> str:
    return str(
        (benign_task or {}).get("site")
        or contract.get("site")
        or row.get("site")
        or "unknown"
    )


def _eligibility_status(contract: Mapping[str, Any]) -> str | None:
    eligibility = contract.get("eligibility")
    if isinstance(eligibility, Mapping):
        value = eligibility.get("status")
        return str(value) if value not in (None, "") else None
    return None


def _selected_action_kind(task: Mapping[str, Any]) -> str | None:
    action = task.get("adversarial_action")
    if not isinstance(action, Mapping):
        return None
    kind = action.get("kind")
    return str(kind) if isinstance(kind, str) and kind.strip() else None


def _task_surface(task: Mapping[str, Any]) -> str:
    contract = _mapping(task.get("exposure_contract"))
    surface = task.get("target_surface_id") or contract.get("target_surface_id")
    return str(surface) if isinstance(surface, str) and surface else "unknown"


def _has_required_url_token(task: Mapping[str, Any]) -> bool:
    for token in task.get("required_tokens") or []:
        if isinstance(token, Mapping) and token.get("kind") == "url":
            value = token.get("value")
            if isinstance(value, str) and value.strip():
                return True
    return False


def _increment_bucket(buckets: dict[str, Counter[str]], key: str, status: str) -> None:
    counter = buckets.setdefault(key, Counter())
    counter["total"] += 1
    counter[status] += 1


def _counter_map_to_dict(counter_map: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(sorted(counter.items())) for key, counter in sorted(counter_map.items())}


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _write_output(report: Mapping[str, Any], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_text(report: Mapping[str, Any]) -> None:
    exposure = _mapping(report.get("exposure_contracts"))
    selected = _mapping(report.get("selected_tasks"))
    print(
        "tier3 action pilot audit: "
        f"{exposure.get('ready_contracts', 0)} ready contract(s), "
        f"{selected.get('selected_tier3_tasks', 0)} selected task(s), "
        f"{selected.get('compile_ready_tasks', 0)} compile-ready selected task(s)"
    )
    if exposure.get("ineligible_reasons"):
        print("ineligible_reasons=" + json.dumps(exposure["ineligible_reasons"], sort_keys=True))
    if selected.get("risk_counts"):
        print("selected_risks=" + json.dumps(selected["risk_counts"], sort_keys=True))


def _default_run_artifact_paths(run_dir: Path) -> Iterable[Path]:
    yield run_dir / "phase_1" / "benign_tasks.json"
    yield run_dir / "phase_2" / "exposure_contracts.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a WorldSim run directory for Tier 3 repository-content "
            "mutation pilot readiness."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--min-ready-contracts",
        type=int,
        default=1,
        help="Fail if fewer than this many host-ready Tier 3 exposure contracts exist.",
    )
    parser.add_argument(
        "--min-selected-tasks",
        type=int,
        default=0,
        help="Fail if fewer than this many selected Tier 3 adversarial tasks exist.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum ready/ineligible/selected/failure rows to include.",
    )
    parser.add_argument(
        "--action-kind",
        default=TIER3_KIND,
        choices=("modify_repository_content", "delete_or_transfer_resource"),
        help="Tier 3 action kind to audit.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument(
        "--require-benign-action-evidence",
        action="store_true",
        help=(
            "Fail selected Tier 3 tasks whose benign reward lacks a host-owned "
            "POST/status/body/path-specific action witness."
        ),
    )
    args = parser.parse_args(argv)

    missing = [path for path in _default_run_artifact_paths(args.run_dir) if not path.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: required artifact missing: {path}", file=sys.stderr)
        return 2

    report = analyze_run_dir(
        args.run_dir,
        action_kind=args.action_kind,
        max_samples=args.max_samples,
        require_benign_action_evidence=args.require_benign_action_evidence,
    )
    _write_output(report, args.output)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)

    exposure = _mapping(report.get("exposure_contracts"))
    selected = _mapping(report.get("selected_tasks"))
    failed_gate = False
    if int(exposure.get("ready_contracts") or 0) < args.min_ready_contracts:
        print(
            f"ERROR: found {exposure.get('ready_contracts', 0)} ready Tier 3 contract(s), "
            f"minimum required is {args.min_ready_contracts}",
            file=sys.stderr,
        )
        failed_gate = True
    if int(selected.get("selected_tier3_tasks") or 0) < args.min_selected_tasks:
        print(
            f"ERROR: found {selected.get('selected_tier3_tasks', 0)} selected Tier 3 task(s), "
            f"minimum required is {args.min_selected_tasks}",
            file=sys.stderr,
        )
        failed_gate = True
    if int(selected.get("failed_selected_tasks") or 0) > 0:
        print(
            f"ERROR: {selected.get('failed_selected_tasks')} selected Tier 3 task(s) "
            "failed reward/final-state readiness",
            file=sys.stderr,
        )
        failed_gate = True
    return 1 if failed_gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
