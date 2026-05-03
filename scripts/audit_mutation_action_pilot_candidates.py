"""Audit Phase 2 artifacts for mutation-action pilot readiness.

This is the second-stage guardrail after the binary adversarial-action canary.
It does not change tasks or run browsers. It reads a Phase 2
``adversarial_tasks.json`` artifact, recomputes host-compatible mutation action
options from each task's exposure contract/editor method, and verifies that
each candidate compiles to a POST/status/body-witness network reward.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldsim.adversarial_actions import (
    allowed_action_options,
    compile_adversarial_reward,
    get_action_spec,
    reward_signal_for_task,
)

_MUTATION_KINDS = frozenset(
    {"submit_comment", "create_post", "create_issue_note", "create_issue"}
)


def analyze_adversarial_tasks(
    tasks: list[dict[str, Any]],
    *,
    artifact: str | None = None,
    max_samples: int = 50,
    require_benign_action_evidence: bool = False,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    failed_compilations: list[dict[str, Any]] = []
    non_candidate_reasons = Counter()
    risk_counts = Counter()
    by_site: dict[str, Counter[str]] = {}
    by_surface: dict[str, Counter[str]] = {}
    by_action_kind = Counter()

    for index, task in enumerate(tasks):
        benign_reward = _benign_reward(task)
        benign_task = {
            "id": task.get("benign_task_id") or task.get("id"),
            "reward_function": benign_reward or {},
        }
        exposure_contract = _exposure_contract_for_task(task)
        eligibility_status = _mapping(exposure_contract.get("eligibility")).get("status")
        if eligibility_status not in {None, "eligible"}:
            non_candidate_reasons[f"exposure_contract_not_eligible:{eligibility_status}"] += 1
            continue
        mutation_options = [
            option["kind"]
            for option in allowed_action_options(benign_task, exposure_contract)
            if option.get("kind") in _MUTATION_KINDS
        ]
        if not mutation_options:
            non_candidate_reasons[_non_candidate_reason(task, exposure_contract)] += 1
            continue

        action_rows: list[dict[str, Any]] = []
        task_failed = False
        task_risks = [
            *_task_risks(task),
            *_selected_action_risks(task, mutation_options),
        ]
        benign_action_evidence = _benign_action_reward_evidence(task)
        for kind in mutation_options:
            compiled, compile_error = _compile_mutation_reward(
                task,
                benign_task=benign_task,
                exposure_contract=exposure_contract,
                kind=kind,
            )
            if compile_error is not None:
                task_failed = True
                action_row = {
                    "kind": kind,
                    "ok": False,
                    "error": compile_error,
                    "evidence": {},
                    "risks": task_risks,
                }
                failed_compilations.append(
                    _candidate_row(
                        task,
                        index=index,
                        mutation_options=mutation_options,
                        action_rows=[action_row],
                        risks=task_risks,
                        benign_action_evidence=benign_action_evidence,
                    )
                )
                action_rows.append(action_row)
                continue

            evidence = _network_reward_evidence(compiled)
            action_risks = [*task_risks, *_network_reward_risks(evidence)]
            ok = not _network_reward_risks(evidence)
            if not ok:
                task_failed = True
                failed_compilations.append(
                    _candidate_row(
                        task,
                        index=index,
                        mutation_options=mutation_options,
                        action_rows=[
                            {
                                "kind": kind,
                                "ok": False,
                                "error": "compiled network reward lacks required evidence",
                                "evidence": evidence,
                                "risks": action_risks,
                            }
                        ],
                        risks=action_risks,
                        benign_action_evidence=benign_action_evidence,
                    )
                )
            by_action_kind[kind] += 1
            action_rows.append(
                {
                    "kind": kind,
                    "ok": ok,
                    "error": None,
                    "evidence": evidence,
                    "risks": action_risks,
                }
            )

        benign_action_risks = _benign_action_evidence_risks(
            benign_action_evidence,
            required=require_benign_action_evidence,
        )
        risks = sorted(
            {
                risk
                for action in action_rows
                for risk in action.get("risks", [])
            }
            | set(benign_action_risks)
        )
        for risk in risks:
            risk_counts[risk] += 1
        row = _candidate_row(
            task,
            index=index,
            mutation_options=mutation_options,
            action_rows=action_rows,
            risks=risks,
            benign_action_evidence=benign_action_evidence,
        )
        candidates.append(row)
        _increment_buckets(
            by_site,
            by_surface,
            row,
            failed=task_failed or bool(risks),
        )

    return {
        "artifact": artifact,
        "total_tasks": len(tasks),
        "candidate_tasks": len(candidates),
        "compiled_mutation_actions": sum(
            1 for row in candidates for action in row["compiled_actions"] if action["ok"]
        ),
        "failed_compilations": len(failed_compilations),
        "risk_failures": sum(1 for row in candidates if row.get("risks")),
        "benign_action_evidence_required": require_benign_action_evidence,
        "benign_action_evidence_failures": sum(
            1
            for row in candidates
            if "missing_benign_action_contract" in row.get("risks", [])
            or "missing_benign_action_network_reward" in row.get("risks", [])
            or "benign_action_network_reward_missing_witness" in row.get("risks", [])
        ),
        "non_candidate_reasons": dict(sorted(non_candidate_reasons.items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "by_site": _counter_map_to_dict(by_site),
        "by_surface": _counter_map_to_dict(by_surface),
        "by_action_kind": dict(sorted(by_action_kind.items())),
        "failures": failed_compilations[:max_samples],
        "candidates": candidates[:max_samples],
    }


def _compile_mutation_reward(
    task: Mapping[str, Any],
    *,
    benign_task: Mapping[str, Any],
    exposure_contract: Mapping[str, Any],
    kind: str,
) -> tuple[dict[str, Any] | None, str | None]:
    task_for_compile = json.loads(json.dumps(dict(task)))
    task_for_compile["exposure_contract"] = dict(exposure_contract)
    task_for_compile["adversarial_action"] = {
        "kind": kind,
        "description": "Mutation pilot audit recompilation.",
    }
    try:
        compiled = compile_adversarial_reward(task_for_compile, benign_task)
    except ValueError as exc:
        return None, str(exc)
    return compiled, None


def _candidate_row(
    task: Mapping[str, Any],
    *,
    index: int,
    mutation_options: list[str],
    action_rows: list[dict[str, Any]],
    risks: list[str],
    benign_action_evidence: dict[str, Any],
) -> dict[str, Any]:
    contract = _exposure_contract_for_task(task)
    return {
        "index": index,
        "id": task.get("id"),
        "benign_task_id": task.get("benign_task_id"),
        "site": str(task.get("site") or "unknown"),
        "origin": task.get("origin"),
        "surface": _task_surface(task, contract),
        "route_id": task.get("route_id"),
        "editor_method": task.get("editor_method") or contract.get("editor_method"),
        "selected_action_kind": _selected_action_kind(task),
        "selected_reward_signal": reward_signal_for_task(task),
        "feasibility_status": _mapping(task.get("feasibility")).get("status"),
        "mutation_options": mutation_options,
        "compiled_actions": action_rows,
        "benign_action_evidence": benign_action_evidence,
        "risks": risks,
    }


def _network_reward_evidence(reward: Mapping[str, Any] | None) -> dict[str, Any]:
    expected = reward.get("expected") if isinstance(reward, Mapping) else None
    if not isinstance(expected, Mapping):
        return {
            "type": reward.get("type") if isinstance(reward, Mapping) else None,
            "has_post": False,
            "has_response_status": False,
            "has_body_evidence": False,
            "has_path_specific_url_pattern": False,
            "post_data": None,
            "post_data_fields": [],
            "legacy_post_data_contains": None,
        }
    patterns = _as_list(expected.get("url"))
    post_data = expected.get("post_data")
    post_data_fields = sorted(post_data) if isinstance(post_data, Mapping) else []
    return {
        "type": reward.get("type"),
        "http_method": expected.get("http_method"),
        "has_post": str(expected.get("http_method") or "").upper() == "POST",
        "response_status": expected.get("response_status"),
        "has_response_status": bool(expected.get("response_status")),
        "has_body_evidence": bool(expected.get("post_data") or expected.get("post_data_contains")),
        "has_path_specific_url_pattern": any(
            isinstance(pattern, str)
            and pattern.strip()
            and pattern.strip() not in {".*", "*"}
            for pattern in patterns
        ),
        "url_patterns": patterns,
        "post_data": post_data,
        "post_data_fields": post_data_fields,
        "legacy_post_data_contains": expected.get("post_data_contains"),
    }


def _network_reward_risks(evidence: Mapping[str, Any]) -> list[str]:
    risks: list[str] = []
    if evidence.get("type") != "NetworkEventEvaluator":
        risks.append("not_network_event_evaluator")
    if evidence.get("has_post") is not True:
        risks.append("missing_post_method")
    if evidence.get("has_response_status") is not True:
        risks.append("missing_response_status")
    if evidence.get("has_body_evidence") is not True:
        risks.append("missing_body_evidence")
    if evidence.get("has_path_specific_url_pattern") is not True:
        risks.append("broad_only_endpoint_pattern")
    return risks


def _benign_action_reward_evidence(task: Mapping[str, Any]) -> dict[str, Any]:
    reward = _benign_reward(task)
    evals = reward.get("eval") if isinstance(reward, Mapping) else None
    provenance = task.get("task_provenance")
    action_contract = (
        provenance.get("benign_action_contract") if isinstance(provenance, Mapping) else None
    )
    witness = (
        str(action_contract.get("witness") or "").strip()
        if isinstance(action_contract, Mapping)
        else ""
    )
    network_evidence: list[dict[str, Any]] = []
    if isinstance(evals, list):
        for config in evals:
            if not isinstance(config, Mapping):
                continue
            if config.get("evaluator") != "NetworkEventEvaluator":
                continue
            evidence = _network_reward_evidence(config)
            evidence["contains_benign_witness"] = bool(witness and witness in str(config))
            network_evidence.append(evidence)
    matching = [
        evidence
        for evidence in network_evidence
        if evidence.get("contains_benign_witness")
        and evidence.get("has_post") is True
        and evidence.get("has_response_status") is True
        and evidence.get("has_body_evidence") is True
        and evidence.get("has_path_specific_url_pattern") is True
    ]
    return {
        "has_contract": isinstance(action_contract, Mapping),
        "action_kind": (
            action_contract.get("action_kind") if isinstance(action_contract, Mapping) else None
        ),
        "witness": witness or None,
        "network_reward_count": len(network_evidence),
        "matching_network_reward_count": len(matching),
        "ok": bool(matching),
        "network_rewards": network_evidence,
    }


def _benign_action_evidence_risks(
    evidence: Mapping[str, Any],
    *,
    required: bool,
) -> list[str]:
    if not required:
        return []
    if evidence.get("has_contract") is not True:
        return ["missing_benign_action_contract"]
    if not evidence.get("network_reward_count"):
        return ["missing_benign_action_network_reward"]
    if not evidence.get("matching_network_reward_count"):
        return ["benign_action_network_reward_missing_witness"]
    return []


def _task_risks(task: Mapping[str, Any]) -> list[str]:
    risks: list[str] = []
    if not _has_required_url_token(task):
        risks.append("missing_required_url_token")
    surface = _task_surface(task, _exposure_contract_for_task(task)).casefold()
    if "title" in surface:
        risks.append("retired_title_surface")
    feasibility_status = _mapping(task.get("feasibility")).get("status")
    if feasibility_status not in {None, "valid", "verified", "passed", "feasible"}:
        risks.append("non_verified_feasibility_status")
    selected_kind = _selected_action_kind(task)
    spec = get_action_spec(selected_kind or "")
    if spec is not None and not spec.default_enabled:
        risks.append(f"disabled_selected_action:{selected_kind}")
    return risks


def _selected_action_risks(
    task: Mapping[str, Any],
    mutation_options: list[str],
) -> list[str]:
    selected_kind = _selected_action_kind(task)
    if not selected_kind:
        return ["missing_selected_action"]
    if selected_kind not in mutation_options:
        return [f"selected_action_not_mutation_candidate:{selected_kind}"]
    return []


def _non_candidate_reason(
    task: Mapping[str, Any], exposure_contract: Mapping[str, Any]
) -> str:
    method = task.get("editor_method") or exposure_contract.get("editor_method")
    if not isinstance(method, str) or not method.strip():
        return "missing_editor_method"
    return f"editor_method_not_mutation_capable:{method}"


def _benign_reward(task: Mapping[str, Any]) -> Any:
    reward = task.get("reward_function")
    if not isinstance(reward, Mapping):
        return None
    if "benign_reward" in reward:
        return reward.get("benign_reward")
    if "adversarial_reward" not in reward:
        return reward
    return None


def _exposure_contract_for_task(task: Mapping[str, Any]) -> dict[str, Any]:
    contract = task.get("exposure_contract")
    copied = dict(contract) if isinstance(contract, Mapping) else {}
    for source_field, contract_field in (
        ("editor_method", "editor_method"),
        ("target_surface_id", "target_surface_id"),
    ):
        if copied.get(contract_field):
            continue
        value = task.get(source_field)
        if isinstance(value, str) and value.strip():
            copied[contract_field] = value.strip()
    return copied


def _selected_action_kind(task: Mapping[str, Any]) -> str | None:
    action = task.get("adversarial_action")
    if not isinstance(action, Mapping):
        return None
    kind = action.get("kind")
    return str(kind) if isinstance(kind, str) and kind else None


def _task_surface(task: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    surface = task.get("target_surface_id") or contract.get("target_surface_id")
    return str(surface) if isinstance(surface, str) and surface else "unknown"


def _has_required_url_token(task: Mapping[str, Any]) -> bool:
    for token in task.get("required_tokens") or []:
        if isinstance(token, Mapping) and token.get("kind") == "url":
            value = token.get("value")
            if isinstance(value, str) and value.strip():
                return True
    return False


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _increment_buckets(
    by_site: dict[str, Counter[str]],
    by_surface: dict[str, Counter[str]],
    row: Mapping[str, Any],
    *,
    failed: bool,
) -> None:
    for buckets, key in (
        (by_site, str(row.get("site") or "unknown")),
        (by_surface, str(row.get("surface") or "unknown")),
    ):
        counter = buckets.setdefault(key, Counter())
        counter["total"] += 1
        counter["failed" if failed else "passed"] += 1


def _counter_map_to_dict(counter_map: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(sorted(counter.items())) for key, counter in sorted(counter_map.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Phase 2 adversarial_tasks.json artifact for tasks whose "
            "exposure contracts can support DoomArena-style mutation rewards."
        )
    )
    parser.add_argument("adversarial_tasks", type=Path)
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=1,
        help="Fail if fewer than this many mutation candidate tasks are present.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum candidate/failure rows to include in JSON output.",
    )
    parser.add_argument(
        "--require-benign-action-evidence",
        action="store_true",
        help=(
            "Fail if mutation candidates lack host-compiled benign action "
            "NetworkEventEvaluator evidence."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args(argv)

    tasks = json.loads(args.adversarial_tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("adversarial_tasks must be a JSON array")
    report = analyze_adversarial_tasks(
        [task for task in tasks if isinstance(task, dict)],
        artifact=str(args.adversarial_tasks),
        max_samples=args.max_samples,
        require_benign_action_evidence=args.require_benign_action_evidence,
    )

    failed_gate = (
        report["failed_compilations"] > 0
        or report["risk_failures"] > 0
        or report["candidate_tasks"] < args.min_candidates
        or (
            args.require_benign_action_evidence
            and report["benign_action_evidence_failures"] > 0
        )
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "mutation action pilot audit: "
            f"{report['candidate_tasks']} candidate task(s), "
            f"{report['compiled_mutation_actions']} compiled mutation action(s), "
            f"{report['failed_compilations']} failed compilation(s), "
            f"{report['risk_failures']} risk failure(s), "
            f"{report['benign_action_evidence_failures']} benign-action evidence failure(s) "
            f"({report['total_tasks']} total tasks)"
        )
        if report["candidate_tasks"] < args.min_candidates:
            print(
                f"ERROR: found {report['candidate_tasks']} mutation candidate task(s), "
                f"minimum required is {args.min_candidates}",
                file=sys.stderr,
            )
        for failure in report["failures"][:10]:
            print(
                "FAIL "
                f"{failure.get('id')}: method={failure.get('editor_method')} "
                f"actions={failure.get('mutation_options')} risks={failure.get('risks')}",
                file=sys.stderr,
            )
        if args.require_benign_action_evidence:
            for candidate in report["candidates"][:10]:
                risks = set(candidate.get("risks") or [])
                if not risks.intersection(
                    {
                        "missing_benign_action_contract",
                        "missing_benign_action_network_reward",
                        "benign_action_network_reward_missing_witness",
                    }
                ):
                    continue
                print(
                    "BENIGN_ACTION_EVIDENCE_FAIL "
                    f"{candidate.get('id')}: risks={candidate.get('risks')}",
                    file=sys.stderr,
                )
    return 1 if failed_gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
