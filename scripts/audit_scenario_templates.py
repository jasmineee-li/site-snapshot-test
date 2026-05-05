#!/usr/bin/env python3
"""Audit scenario-template strength for generated WorldSim task cohorts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from worldsim.adversarial_actions.scenario_templates import (
    scenario_template_from_mapping,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run or artifact-export directory")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    tasks = _load_tasks(args.run_dir)
    summary = audit_tasks(tasks)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 1 if summary["error_count"] else 0


def audit_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    findings: list[dict[str, Any]] = []

    for raw_task in tasks:
        task = _task_payload(raw_task)
        task_id = str(task.get("id") or task.get("task_id") or "unknown")
        instruction = str(task.get("instruction") or "")
        template = _task_scenario_template(task)
        if template is None:
            scenario_counts["missing"] += 1
            findings.append(
                {
                    "severity": "warning",
                    "task_id": task_id,
                    "code": "SCENARIO_TEMPLATE_MISSING",
                    "message": "task has no host-owned scenario_template in task_provenance",
                }
            )
            continue

        try:
            scenario = scenario_template_from_mapping(template)
        except ValueError as exc:
            scenario_counts["invalid"] += 1
            findings.append(
                {
                    "severity": "error",
                    "task_id": task_id,
                    "code": "SCENARIO_TEMPLATE_INVALID",
                    "message": str(exc),
                }
            )
            continue

        scenario_counts[scenario.id] += 1
        missing_evidence = []
        for group in scenario.required_instruction_evidence:
            patterns = [
                pattern
                for pattern in group.get("any_regex") or []
                if isinstance(pattern, str) and pattern.strip()
            ]
            if patterns and not any(re.search(pattern, instruction) for pattern in patterns):
                missing_evidence.append({"name": group.get("name"), "any_regex": patterns})
        if missing_evidence:
            findings.append(
                {
                    "severity": "error",
                    "task_id": task_id,
                    "scenario_template": scenario.id,
                    "code": "SCENARIO_INSTRUCTION_EVIDENCE_MISSING",
                    "message": "instruction does not satisfy all scenario evidence groups",
                    "expected": missing_evidence,
                }
            )
        for pattern in scenario.forbidden_instruction_evidence:
            if re.search(pattern, instruction):
                findings.append(
                    {
                        "severity": "error",
                        "task_id": task_id,
                        "scenario_template": scenario.id,
                        "code": "SCENARIO_FORBIDDEN_INSTRUCTION_EVIDENCE",
                        "message": f"instruction matches forbidden scenario pattern {pattern!r}",
                    }
                )

        findings.extend(_audit_benign_action_evidence(task, scenario.id))
        findings.extend(_audit_adversarial_action_evidence(task, scenario.id))
        status_counts[_scenario_strength_status(task, scenario.id)] += 1

    error_count = sum(1 for finding in findings if finding["severity"] == "error")
    warning_count = sum(1 for finding in findings if finding["severity"] == "warning")
    return {
        "task_count": len(tasks),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "scenario_strength_status_counts": dict(sorted(status_counts.items())),
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": findings,
    }


def _task_payload(task: dict[str, Any]) -> dict[str, Any]:
    nested = task.get("task")
    return nested if isinstance(nested, dict) else task


def _task_scenario_template(task: dict[str, Any]) -> dict[str, Any] | None:
    provenance = task.get("task_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("scenario_template"), dict):
        return provenance["scenario_template"]
    card = task.get("task_card")
    if isinstance(card, dict) and isinstance(card.get("scenario_template"), dict):
        return card["scenario_template"]
    return None


def _audit_benign_action_evidence(
    task: dict[str, Any],
    scenario_id: str,
) -> list[dict[str, Any]]:
    if scenario_id != "gitlab_issue_public_followup":
        return []
    task_id = str(task.get("id") or task.get("task_id") or "unknown")
    reward_shape = str(
        task.get("benign_reward_shape")
        or (task.get("task_provenance") or {}).get("benign_reward_shape")
        or ""
    )
    if reward_shape != "agent_response_plus_host_action":
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_BENIGN_REWARD_SHAPE_MISMATCH",
                "message": (
                    "GitLab public follow-up tasks must use "
                    "agent_response_plus_host_action benign rewards"
                ),
                "actual": reward_shape or "missing",
            }
        ]

    provenance = task.get("task_provenance")
    contract = (
        provenance.get("benign_action_contract")
        if isinstance(provenance, dict)
        else None
    )
    if not isinstance(contract, dict):
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_BENIGN_ACTION_CONTRACT_MISSING",
                "message": "public follow-up task has no host-owned benign_action_contract",
            }
        ]
    action_kind = str(contract.get("action_kind") or "")
    editor_method = str(contract.get("editor_method") or "")
    witness = str(contract.get("witness") or "")
    if action_kind != "create_issue" or editor_method not in {
        "create_issue",
        "create_issue_description",
    }:
        return []
    if not witness:
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_BENIGN_ACTION_WITNESS_MISSING",
                "message": "GitLab public follow-up create_issue contract has no witness",
            }
        ]

    evaluator = contract.get("evaluator")
    if not _is_gitlab_issue_final_state_evaluator(evaluator, witness):
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_BENIGN_ACTION_EVIDENCE_STALE",
                "message": (
                    "GitLab public follow-up create_issue evidence must use "
                    "FinalStateEvaluator readback; REST-only NetworkEventEvaluator "
                    "cannot observe normal UI/GraphQL issue creation reliably"
                ),
            }
        ]

    if not any(
        _is_gitlab_issue_final_state_evaluator(item, witness)
        for item in _benign_reward_evals(task)
    ):
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_BENIGN_ACTION_REWARD_STALE",
                "message": (
                    "reward_function does not include the host-owned GitLab "
                    "issue final-state readback evaluator"
                ),
            }
        ]
    return []


def _benign_reward_evals(task: dict[str, Any]) -> list[dict[str, Any]]:
    reward = task.get("reward_function")
    if not isinstance(reward, dict):
        return []
    benign_reward = reward.get("benign_reward")
    if isinstance(benign_reward, dict):
        reward = benign_reward
    evals = reward.get("eval")
    if not isinstance(evals, list):
        return []
    return [item for item in evals if isinstance(item, dict)]


def _audit_adversarial_action_evidence(
    task: dict[str, Any],
    scenario_id: str,
) -> list[dict[str, Any]]:
    if scenario_id != "gitlab_issue_public_followup":
        return []
    action = task.get("adversarial_action")
    action_kind = str(action.get("kind") or "") if isinstance(action, dict) else ""
    if action_kind != "create_issue":
        return []
    task_id = str(task.get("id") or task.get("task_id") or "unknown")
    witness = _task_required_url(task)
    if not witness:
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_ADVERSARIAL_ACTION_WITNESS_MISSING",
                "message": "GitLab public follow-up create_issue action has no required URL witness",
            }
        ]
    reward = task.get("reward_function")
    if not isinstance(reward, dict):
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_ADVERSARIAL_REWARD_MISSING",
                "message": "task has no reward_function for adversarial action evidence",
            }
        ]
    adversarial_reward = reward.get("adversarial_reward")
    if not _is_gitlab_issue_final_state_evaluator(adversarial_reward, witness):
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_ADVERSARIAL_ACTION_REWARD_STALE",
                "message": (
                    "GitLab public follow-up adversarial create_issue reward must "
                    "use FinalStateEvaluator readback; REST-only NetworkEventEvaluator "
                    "can miss normal UI/GraphQL issue creation"
                ),
            }
        ]
    final_state_check = reward.get("adversarial_final_state_check")
    if not _is_gitlab_issue_final_state_evaluator(final_state_check, witness):
        return [
            {
                "severity": "error",
                "task_id": task_id,
                "scenario_template": scenario_id,
                "code": "SCENARIO_ADVERSARIAL_FINAL_STATE_CHECK_STALE",
                "message": (
                    "GitLab public follow-up adversarial create_issue final-state "
                    "check must match the readback-based adversarial reward"
                ),
            }
        ]
    return []


def _task_required_url(task: dict[str, Any]) -> str:
    for token in task.get("required_tokens") or []:
        if not isinstance(token, dict):
            continue
        if token.get("kind") != "url":
            continue
        value = token.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_gitlab_issue_final_state_evaluator(value: Any, witness: str) -> bool:
    if not isinstance(value, dict):
        return False
    evaluator = str(value.get("evaluator") or value.get("type") or "")
    if evaluator != "FinalStateEvaluator":
        return False
    expected = value.get("expected")
    if not isinstance(expected, dict):
        return False
    if str(expected.get("action_kind") or "") != "create_issue":
        return False
    if str(expected.get("contains") or "") != witness:
        return False
    state_probe = expected.get("state_probe")
    if not isinstance(state_probe, dict):
        return False
    return (
        state_probe.get("kind") == "issue_contains"
        and state_probe.get("site") == "gitlab"
    )


def _scenario_strength_status(task: dict[str, Any], scenario_id: str) -> str:
    reward_shape = str(
        task.get("benign_reward_shape")
        or (task.get("task_provenance") or {}).get("benign_reward_shape")
        or ""
    )
    action = task.get("adversarial_action")
    action_kind = str(action.get("kind") or "") if isinstance(action, dict) else ""
    if scenario_id == "gitlab_comment_first_delete_resource":
        return "known_weak_delete_resource_alignment"
    if reward_shape == "host_action_only" or action_kind in {
        "create_issue",
        "create_issue_note",
        "create_post",
        "submit_comment",
        "modify_repository_content",
    }:
        return "capability_aligned"
    return "scenario_declared"


def _load_tasks(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        raise SystemExit(f"run dir does not exist: {run_dir}")
    if run_dir.is_file():
        data = json.loads(run_dir.read_text(encoding="utf-8"))
        return _coerce_tasks(data)
    candidates = (
        run_dir / "adversarial_tasks.json",
        run_dir / "phase_2" / "adversarial_tasks.json",
        run_dir / "benign_tasks.json",
        run_dir / "phase_1" / "benign_tasks.json",
        run_dir / "tasks.json",
    )
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return _coerce_tasks(data)
    raise SystemExit(
        "could not find a task artifact under "
        f"{run_dir}: expected adversarial_tasks.json, phase_2/adversarial_tasks.json, "
        "benign_tasks.json, phase_1/benign_tasks.json, or tasks.json"
    )


def _coerce_tasks(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [_task_payload(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("tasks", "adversarial_tasks", "benign_tasks"):
            value = data.get(key)
            if isinstance(value, list):
                return [_task_payload(item) for item in value if isinstance(item, dict)]
    raise SystemExit("task file must be a list or object containing a task array")


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Tasks: {summary['task_count']}")
    print(f"Errors: {summary['error_count']}  Warnings: {summary['warning_count']}")
    print("Scenario counts:")
    for scenario, count in summary["scenario_counts"].items():
        print(f"  - {scenario}: {count}")
    print("Scenario strength:")
    for status, count in summary["scenario_strength_status_counts"].items():
        print(f"  - {status}: {count}")
    if summary["findings"]:
        print("Findings:")
        for finding in summary["findings"]:
            print(
                f"  - [{finding['severity']}] {finding['task_id']} "
                f"{finding['code']}: {finding['message']}"
            )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
