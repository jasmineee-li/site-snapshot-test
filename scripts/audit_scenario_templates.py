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

    for task in tasks:
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


def _task_scenario_template(task: dict[str, Any]) -> dict[str, Any] | None:
    provenance = task.get("task_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("scenario_template"), dict):
        return provenance["scenario_template"]
    card = task.get("task_card")
    if isinstance(card, dict) and isinstance(card.get("scenario_template"), dict):
        return card["scenario_template"]
    return None


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
    candidates = (
        run_dir / "adversarial_tasks.json",
        run_dir / "benign_tasks.json",
        run_dir / "tasks.json",
    )
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return _coerce_tasks(data)
    raise SystemExit(
        f"could not find adversarial_tasks.json, benign_tasks.json, or tasks.json under {run_dir}"
    )


def _coerce_tasks(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("tasks", "adversarial_tasks", "benign_tasks"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
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
