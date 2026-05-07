#!/usr/bin/env python3
"""Validate Phase 1 contract-bound action-task artifacts.

This is a pre-Phase-2 gate: it checks that generated benign tasks use the
host-compiled action-task shape and do not regress to legacy answer probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LEGACY_BINARY_LABELS = ("blank/populated", "link/no-link", "answer with exactly")


def analyze_phase1_contract_bound_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    by_site: Counter[str] = Counter()
    by_action: Counter[str] = Counter()
    ids: set[str] = set()
    for index, task in enumerate(tasks):
        row_failures = _task_failures(task)
        task_id = str(task.get("id") or "")
        if task_id in ids:
            row_failures.append("duplicate_task_id")
        if task_id:
            ids.add(task_id)
        by_site[str(task.get("site") or "unknown")] += 1
        by_action[_action_kind(task) or "missing"] += 1
        if row_failures:
            failures.append(
                {
                    "index": index,
                    "id": task.get("id"),
                    "site": task.get("site"),
                    "task_card_id": _task_card_id(task),
                    "action_kind": _action_kind(task),
                    "failures": row_failures,
                }
            )
    return {
        "total_tasks": len(tasks),
        "valid_tasks": len(tasks) - len(failures),
        "failed_tasks": len(failures),
        "by_site": dict(sorted(by_site.items())),
        "by_action_kind": dict(sorted(by_action.items())),
        "failures": failures,
    }


def _task_failures(task: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if str(task.get("origin") or "") != "new_task":
        failures.append("origin_not_new_task")
    if not _task_card_id(task):
        failures.append("missing_task_card_id")
    if not isinstance(task.get("route_id"), str) or not str(task.get("route_id")).strip():
        failures.append("missing_route_id")
    provenance = task.get("task_provenance")
    if not isinstance(provenance, Mapping):
        failures.append("missing_task_provenance")
    elif provenance.get("benign_reward_shape") != "host_action_only":
        failures.append("benign_reward_shape_not_host_action_only")
    if not _benign_action_contract(task):
        failures.append("missing_benign_action_contract")
    if not _reward_is_host_action_only(task):
        failures.append("reward_not_host_action_only")
    instruction = str(task.get("instruction") or "").casefold()
    for label in LEGACY_BINARY_LABELS:
        if label in instruction:
            failures.append(f"legacy_probe_instruction:{label}")
    if _contains_agent_response_evaluator(task):
        failures.append("agent_response_evaluator_present")
    action = _action_kind(task)
    if action not in {"create_issue", "create_issue_note", "create_post", "submit_comment"}:
        failures.append(f"unsupported_or_missing_action_kind:{action or 'missing'}")
    if action == "submit_comment" and not _has_submit_comment_visibility_gate(task):
        failures.append("submit_comment_requires_explicit_visibility_gate")
    return failures


def _task_card_id(task: Mapping[str, Any]) -> str | None:
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("task_card_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = task.get("task_card_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _action_kind(task: Mapping[str, Any]) -> str | None:
    provenance = task.get("task_provenance")
    contract = provenance.get("benign_action_contract") if isinstance(provenance, Mapping) else None
    if isinstance(contract, Mapping):
        value = contract.get("action_kind")
        if isinstance(value, str) and value.strip():
            return value.strip()
    values = task.get("compatible_action_kinds")
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], str):
        return values[0].strip()
    return None


def _benign_action_contract(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    provenance = task.get("task_provenance")
    contract = provenance.get("benign_action_contract") if isinstance(provenance, Mapping) else None
    return contract if isinstance(contract, Mapping) else None


def _reward_is_host_action_only(task: Mapping[str, Any]) -> bool:
    reward = task.get("reward_function")
    evals = reward.get("eval") if isinstance(reward, Mapping) else None
    if not isinstance(evals, list) or not evals:
        return False
    for item in evals:
        if not isinstance(item, Mapping):
            return False
        evaluator = str(item.get("evaluator") or item.get("type") or "")
        if evaluator not in {"FinalStateEvaluator", "NetworkEventEvaluator"}:
            return False
    return True


def _contains_agent_response_evaluator(task: Mapping[str, Any]) -> bool:
    reward = task.get("reward_function")
    evals = reward.get("eval") if isinstance(reward, Mapping) else None
    if not isinstance(evals, list):
        return False
    return any(
        isinstance(item, Mapping)
        and "AgentResponseEvaluator" in str(item.get("evaluator") or item.get("type") or "")
        for item in evals
    )


def _has_submit_comment_visibility_gate(task: Mapping[str, Any]) -> bool:
    generation = task.get("contract_bound_generation")
    if not isinstance(generation, Mapping):
        return False
    evidence = generation.get("reddit_seed_comment_visibility_anchor_evidence")
    if not isinstance(evidence, Mapping):
        return False
    if evidence.get("source") != "phase0c_empty_submission_inventory":
        return False
    existing = _nonnegative_int(evidence.get("existing_comment_count"))
    limit = _nonnegative_int(evidence.get("max_existing_comments_for_comment_seed"))
    return existing is not None and limit is not None and existing <= limit


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _load_tasks(run_dir: Path) -> list[dict[str, Any]]:
    phase_1 = run_dir / "phase_1"
    files = sorted(
        path
        for path in phase_1.glob("novel_tasks_*.json")
        if not path.name.endswith(".metadata.json")
    )
    if not files:
        combined = phase_1 / "benign_tasks.json"
        files = [combined] if combined.exists() else []
    tasks: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array")
        tasks.extend(item for item in payload if isinstance(item, dict))
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-tasks", type=int, default=1)
    args = parser.parse_args(argv)

    tasks = _load_tasks(args.run_dir)
    report = analyze_phase1_contract_bound_tasks(tasks)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["total_tasks"] < args.min_tasks:
        print(
            f"error: only {report['total_tasks']} task(s), expected at least {args.min_tasks}",
            file=sys.stderr,
        )
        return 1
    if report["failed_tasks"]:
        print(f"error: {report['failed_tasks']} task(s) failed validation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
