"""Canary-check host compilation for existing binary adversarial rewards.

This is an artifact-level guardrail for the first validation stage after
introducing host-owned ``adversarial_action`` rewards. It does not run Phase 2
or Phase 4. Instead, it reads a Phase 2 ``adversarial_tasks.json`` artifact,
finds binary semantic tasks, recompiles them as
``answer_opposite_binary_label``, and verifies the compiled reward matches the
artifact's existing adversarial label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldsim.adversarial_actions import compile_adversarial_reward

_BINARY_LABEL_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"blank", "populated"}),
    frozenset({"link", "no-link"}),
)


def analyze_adversarial_tasks(
    tasks: list[dict[str, Any]],
    *,
    artifact: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = Counter()
    by_site: dict[str, Counter[str]] = {}
    by_surface: dict[str, Counter[str]] = {}

    for index, task in enumerate(tasks):
        reward_function = task.get("reward_function")
        if not isinstance(reward_function, Mapping):
            skipped["missing_reward_function"] += 1
            continue
        benign_reward = reward_function.get("benign_reward")
        benign_binary = _single_binary_label_expected(benign_reward)
        if benign_binary is None:
            skipped["non_binary_benign_reward"] += 1
            continue

        benign_label, benign_family = benign_binary
        expected_label = sorted(label for label in benign_family if label != benign_label)[0]
        existing_binary = _single_binary_label_expected(reward_function.get("adversarial_reward"))
        existing_label = existing_binary[0] if existing_binary is not None else None

        task_for_compile = dict(task)
        task_for_compile["adversarial_action"] = {
            "kind": "answer_opposite_binary_label",
            "description": "Canary recompilation of the existing binary semantic reward.",
        }
        benign_task = {
            "id": task.get("benign_task_id"),
            "reward_function": benign_reward,
        }
        try:
            compiled = compile_adversarial_reward(task_for_compile, benign_task)
        except ValueError as exc:
            row = _row(
                task,
                index=index,
                benign_label=benign_label,
                expected_label=expected_label,
                existing_label=existing_label,
                compiled_label=None,
                ok=False,
                reason=f"compile_error: {exc}",
            )
            rows.append(row)
            failures.append(row)
            _increment_buckets(by_site, by_surface, row, "failed")
            continue

        compiled_binary = _single_binary_label_expected(compiled)
        compiled_label = compiled_binary[0] if compiled_binary is not None else None
        ok = compiled_label == existing_label == expected_label
        reason = "ok" if ok else "compiled label does not match existing opposite-label reward"
        row = _row(
            task,
            index=index,
            benign_label=benign_label,
            expected_label=expected_label,
            existing_label=existing_label,
            compiled_label=compiled_label,
            ok=ok,
            reason=reason,
        )
        rows.append(row)
        if not ok:
            failures.append(row)
        _increment_buckets(by_site, by_surface, row, "passed" if ok else "failed")

    return {
        "artifact": artifact,
        "total_tasks": len(tasks),
        "binary_tasks": len(rows),
        "passed": sum(1 for row in rows if row["ok"]),
        "failed": len(failures),
        "skipped": dict(sorted(skipped.items())),
        "by_site": _counter_map_to_dict(by_site),
        "by_surface": _counter_map_to_dict(by_surface),
        "failures": failures[:50],
        "samples": rows[:5],
    }


def _row(
    task: Mapping[str, Any],
    *,
    index: int,
    benign_label: str,
    expected_label: str,
    existing_label: str | None,
    compiled_label: str | None,
    ok: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "id": task.get("id"),
        "benign_task_id": task.get("benign_task_id"),
        "site": str(task.get("site") or "unknown"),
        "surface": _task_surface(task),
        "editor_method": task.get("editor_method")
        or _mapping(task.get("exposure_contract")).get("editor_method"),
        "feasibility": _mapping(task.get("feasibility")).get("status"),
        "benign_label": benign_label,
        "expected_label": expected_label,
        "existing_label": existing_label,
        "compiled_label": compiled_label,
        "ok": ok,
        "reason": reason,
    }


def _increment_buckets(
    by_site: dict[str, Counter[str]],
    by_surface: dict[str, Counter[str]],
    row: Mapping[str, Any],
    status: str,
) -> None:
    site = str(row.get("site") or "unknown")
    surface = str(row.get("surface") or "unknown")
    for buckets, key in ((by_site, site), (by_surface, surface)):
        counter = buckets.setdefault(key, Counter())
        counter["total"] += 1
        counter[status] += 1


def _counter_map_to_dict(counter_map: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(sorted(counter.items())) for key, counter in sorted(counter_map.items())}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_surface(task: Mapping[str, Any]) -> str:
    surface = task.get("target_surface_id")
    if isinstance(surface, str) and surface:
        return surface
    contract = _mapping(task.get("exposure_contract"))
    surface = contract.get("target_surface_id")
    if isinstance(surface, str) and surface:
        return surface
    return "unknown"


def _single_binary_label_expected(reward: Any) -> tuple[str, frozenset[str]] | None:
    strings = [
        value.casefold().strip()
        for value in _reward_agent_response_expected_strings(reward)
        if isinstance(value, str) and value.strip()
    ]
    if len(strings) != 1:
        return None
    label = strings[0]
    for family in _BINARY_LABEL_FAMILIES:
        if label in family:
            return label, family
    return None


def _reward_agent_response_expected_strings(reward: Any) -> list[str]:
    if not isinstance(reward, Mapping):
        return []
    out: list[str] = []
    for config in _agent_response_reward_configs(reward):
        expected = config.get("expected")
        if isinstance(expected, Mapping):
            out.extend(_flatten_strings(expected.get("retrieved_data")))
    return out


def _agent_response_reward_configs(reward: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    reward_type = str(reward.get("type") or reward.get("evaluator") or "")
    if reward_type == "AgentResponseEvaluator":
        return [reward]
    evals = reward.get("eval")
    if not isinstance(evals, list):
        return []
    return [
        item
        for item in evals
        if isinstance(item, Mapping) and item.get("evaluator") == "AgentResponseEvaluator"
    ]


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompile binary semantic adversarial rewards from a Phase 2 artifact "
            "and verify the host compiler preserves the existing opposite label."
        )
    )
    parser.add_argument("adversarial_tasks", type=Path)
    parser.add_argument(
        "--min-binary",
        type=int,
        default=1,
        help="Fail if fewer than this many binary tasks are present.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args(argv)

    tasks = json.loads(args.adversarial_tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("adversarial_tasks must be a JSON array")
    report = analyze_adversarial_tasks(
        [task for task in tasks if isinstance(task, dict)],
        artifact=str(args.adversarial_tasks),
    )

    failed_gate = report["failed"] > 0 or report["binary_tasks"] < args.min_binary
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "binary action compiler canary: "
            f"{report['passed']}/{report['binary_tasks']} passed "
            f"({report['total_tasks']} total tasks, skipped={report['skipped']})"
        )
        if report["binary_tasks"] < args.min_binary:
            print(
                f"ERROR: found {report['binary_tasks']} binary task(s), "
                f"minimum required is {args.min_binary}",
                file=sys.stderr,
            )
        for failure in report["failures"][:10]:
            print(
                "FAIL "
                f"{failure.get('id')}: benign={failure.get('benign_label')} "
                f"expected={failure.get('expected_label')} "
                f"existing={failure.get('existing_label')} "
                f"compiled={failure.get('compiled_label')} "
                f"reason={failure.get('reason')}",
                file=sys.stderr,
            )
    return 1 if failed_gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
