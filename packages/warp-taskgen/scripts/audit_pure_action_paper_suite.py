#!/usr/bin/env python3
"""Audit a pure action Tier 2 paper-suite artifact.

The audit is intentionally stricter than the mutation-pilot audit. It is meant
to run before Phase 4 on a candidate paper cohort and fail closed when rows look
like legacy hybrid answer-probe tasks, request-only action tasks, unsupported
action families, or Reddit/Postmill comment rows whose visibility gate is not
paper-ready.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PAPER_PROFILE = "tier2_pure_action_paper"
PAPER_ACTION_KINDS = frozenset(
    {"create_issue", "create_issue_note", "create_post", "submit_comment"}
)
PAPER_TASK_CARD_IDS = frozenset(
    {
        "gitlab_issue_description_public_followup_issue_paper",
        "gitlab_issue_description_public_followup_comment_paper",
        "reddit_comment_body_public_followup_comment_paper",
        "reddit_submission_body_public_followup_post_paper",
    }
)
LEGACY_BINARY_LABELS = frozenset({"blank", "populated", "link", "no-link"})


def analyze_pure_action_paper_suite(
    tasks: list[dict[str, Any]],
    *,
    artifact: str | None = None,
    run_dir: Path | None = None,
    expected_profile: str = PAPER_PROFILE,
    max_failures: int = 50,
    allow_missing_profile_metadata: bool = False,
    expected_action_counts: Mapping[str, int] | None = None,
    min_purpose_tags: int | None = None,
    min_style_tags: int | None = None,
    reject_duplicate_instructions: bool = False,
) -> dict[str, Any]:
    profile = _load_task_capability_profile(run_dir)
    suite_failures: list[str] = []
    if profile is None:
        if not allow_missing_profile_metadata:
            suite_failures.append("missing_task_capability_profile_metadata")
    elif profile != expected_profile:
        suite_failures.append(f"unexpected_task_capability_profile:{profile}")

    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    site_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    purpose_tags: Counter[str] = Counter()
    style_tags: Counter[str] = Counter()
    normalized_instructions: Counter[str] = Counter()
    for index, task in enumerate(tasks):
        failures = _task_failures(task)
        for failure in failures:
            failure_counts[failure] += 1
        action_kind = _selected_action_kind(task) or "missing"
        action_counts[action_kind] += 1
        site_counts[str(task.get("site") or "unknown")] += 1
        generation = task.get("contract_bound_generation")
        if isinstance(generation, Mapping):
            purpose_tags[str(generation.get("abstract_purpose_tag") or "missing")] += 1
            style_tags[str(generation.get("style_tag") or "missing")] += 1
        normalized_instruction = _normalize_instruction(task.get("instruction"))
        if normalized_instruction:
            normalized_instructions[normalized_instruction] += 1
        rows.append(
            {
                "index": index,
                "id": task.get("id"),
                "site": task.get("site"),
                "task_card_id": _task_card_id(task),
                "action_kind": action_kind,
                "feasibility_status": _feasibility_status(task),
                "benign_reward_shape": _benign_reward_shape(task),
                "failures": failures,
            }
        )

    if expected_action_counts is not None:
        actual = {kind: action_counts.get(kind, 0) for kind in expected_action_counts}
        extra = {
            kind: count
            for kind, count in action_counts.items()
            if count and kind not in expected_action_counts
        }
        if actual != dict(expected_action_counts) or extra:
            suite_failures.append(
                "action_count_mismatch:"
                + json.dumps(
                    {"expected": dict(expected_action_counts), "actual": dict(action_counts)},
                    sort_keys=True,
                )
            )
    if (
        min_purpose_tags is not None
        and len([k for k in purpose_tags if k != "missing"]) < min_purpose_tags
    ):
        suite_failures.append(
            f"insufficient_purpose_tag_diversity:{len(purpose_tags)}<{min_purpose_tags}"
        )
    if (
        min_style_tags is not None
        and len([k for k in style_tags if k != "missing"]) < min_style_tags
    ):
        suite_failures.append(
            f"insufficient_style_tag_diversity:{len(style_tags)}<{min_style_tags}"
        )
    duplicate_instructions = {
        instruction: count for instruction, count in normalized_instructions.items() if count > 1
    }
    if reject_duplicate_instructions and duplicate_instructions:
        suite_failures.append(f"duplicate_normalized_instructions:{len(duplicate_instructions)}")

    failed_rows = [row for row in rows if row["failures"]]
    return {
        "artifact": artifact,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "expected_profile": expected_profile,
        "task_capability_profile": profile,
        "total_tasks": len(tasks),
        "valid_tasks": len(tasks) - len(failed_rows),
        "failed_tasks": len(failed_rows),
        "suite_failures": suite_failures,
        "failure_counts": dict(sorted(failure_counts.items())),
        "by_action_kind": dict(sorted(action_counts.items())),
        "by_site": dict(sorted(site_counts.items())),
        "diversity": {
            "by_abstract_purpose_tag": dict(sorted(purpose_tags.items())),
            "by_style_tag": dict(sorted(style_tags.items())),
            "duplicate_normalized_instructions": duplicate_instructions,
            "unique_normalized_instructions": len(normalized_instructions),
        },
        "failures": failed_rows[:max_failures],
    }


def _task_failures(task: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if _task_card_id(task) not in PAPER_TASK_CARD_IDS:
        failures.append("not_paper_task_card")
    action_kind = _selected_action_kind(task)
    if action_kind not in PAPER_ACTION_KINDS:
        failures.append(f"unsupported_action_kind:{action_kind or 'missing'}")
    if action_kind == "submit_comment" and not _has_reddit_seed_comment_visibility_proof(task):
        failures.append("reddit_submit_comment_missing_seeded_painted_visibility")
    if _benign_reward_shape(task) != "host_action_only":
        failures.append("benign_reward_shape_not_host_action_only")
    if not isinstance(_benign_action_contract(task), Mapping):
        failures.append("missing_benign_action_contract")
    if not _benign_reward_is_host_action_only(task):
        failures.append("benign_reward_not_host_action_only")
    if _contains_binary_agent_response_utility(task):
        failures.append("legacy_binary_answer_probe_utility")
    if _feasibility_status(task) != "verified":
        failures.append("phase2c_not_verified")
    if not _has_seeded_payload_text(task):
        failures.append("missing_seeded_payload_text")
    if not _has_required_url_token(task):
        failures.append("missing_required_url_token")
    if not _has_state_confirmed_adversarial_reward(task):
        failures.append("missing_state_confirmed_adversarial_reward")
    elif not _final_state_contains_required_url(task):
        failures.append("adversarial_state_reward_not_tied_to_required_url")
    return failures


def _has_reddit_seed_comment_visibility_proof(task: Mapping[str, Any]) -> bool:
    feasibility = task.get("feasibility")
    if not isinstance(feasibility, Mapping):
        return False
    render = feasibility.get("render_evidence")
    if not isinstance(render, Mapping):
        return False
    diagnostics = render.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    probe = diagnostics.get("reddit_seed_comment_visibility")
    if not isinstance(probe, Mapping):
        return False
    return (
        probe.get("ok") is True
        and bool(str(probe.get("comment_id") or "").strip())
        and probe.get("comment_id") == probe.get("first_visible_comment_id")
        and probe.get("visible_at_entry") is True
        and not probe.get("requires_expand")
        and int(probe.get("visible_nonspace_chars") or 0) > 0
    )


def _load_task_capability_profile(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    path = run_dir / "phase_1" / "generate_new_tasks_resume_metadata.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = payload.get("task_capability_profile") if isinstance(payload, Mapping) else None
    return str(value) if isinstance(value, str) and value.strip() else None


def _task_card_id(task: Mapping[str, Any]) -> str | None:
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("task_card_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = task.get("task_card_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _selected_action_kind(task: Mapping[str, Any]) -> str | None:
    action = task.get("adversarial_action")
    if not isinstance(action, Mapping):
        return None
    kind = action.get("kind")
    return kind.strip() if isinstance(kind, str) and kind.strip() else None


def _benign_reward_shape(task: Mapping[str, Any]) -> str | None:
    for container in (
        task.get("task_provenance"),
        task,
        task.get("task_archetype"),
    ):
        if not isinstance(container, Mapping):
            continue
        value = container.get("benign_reward_shape")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _benign_action_contract(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    provenance = task.get("task_provenance")
    contract = provenance.get("benign_action_contract") if isinstance(provenance, Mapping) else None
    return contract if isinstance(contract, Mapping) else None


def _benign_reward_is_host_action_only(task: Mapping[str, Any]) -> bool:
    evals = _benign_reward_evaluators(task)
    if not evals:
        return False
    for evaluator in evals:
        if not isinstance(evaluator, Mapping):
            return False
        evaluator_type = str(evaluator.get("type") or evaluator.get("evaluator") or "")
        if evaluator_type not in {"NetworkEventEvaluator", "FinalStateEvaluator"}:
            return False
    return True


def _contains_binary_agent_response_utility(task: Mapping[str, Any]) -> bool:
    for evaluator in _benign_reward_evaluators(task):
        if not isinstance(evaluator, Mapping):
            continue
        evaluator_type = str(evaluator.get("type") or evaluator.get("evaluator") or "")
        if "AgentResponseEvaluator" not in evaluator_type:
            continue
        expected = evaluator.get("expected")
        expected_values = {
            value.casefold().strip()
            for value in _flatten_strings(expected)
            if value.casefold().strip()
        }
        if not expected_values or expected_values <= LEGACY_BINARY_LABELS:
            return True
    return False


def _benign_reward_evaluators(task: Mapping[str, Any]) -> list[Any]:
    reward = task.get("reward_function")
    if not isinstance(reward, Mapping):
        return []
    evals = reward.get("eval")
    if isinstance(evals, list):
        return evals
    benign_reward = reward.get("benign_reward")
    if isinstance(benign_reward, Mapping):
        benign_evals = benign_reward.get("eval")
        if isinstance(benign_evals, list):
            return benign_evals
    return []


def _feasibility_status(task: Mapping[str, Any]) -> str | None:
    feasibility = task.get("feasibility")
    if isinstance(feasibility, Mapping):
        value = feasibility.get("status")
        return value.strip() if isinstance(value, str) and value.strip() else None
    return None


def _has_seeded_payload_text(task: Mapping[str, Any]) -> bool:
    payloads = task.get("payload_texts")
    if not isinstance(payloads, list):
        return False
    for payload in payloads:
        if isinstance(payload, Mapping):
            text = payload.get("rendered_payload") or payload.get("raw_text")
            if isinstance(text, str) and text.strip():
                return True
        elif isinstance(payload, str) and payload.strip():
            return True
    return False


def _has_required_url_token(task: Mapping[str, Any]) -> bool:
    return _required_url_token(task) is not None


def _required_url_token(task: Mapping[str, Any]) -> str | None:
    for token in task.get("required_tokens") or []:
        if not isinstance(token, Mapping) or token.get("kind") != "url":
            continue
        value = token.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _has_state_confirmed_adversarial_reward(task: Mapping[str, Any]) -> bool:
    final_state = _adversarial_final_state_reward(task)
    if not isinstance(final_state, Mapping):
        return False
    expected = final_state.get("expected")
    if not isinstance(expected, Mapping):
        return False
    return final_state.get("type") == "FinalStateEvaluator" and isinstance(
        expected.get("state_probe"), Mapping
    )


def _final_state_contains_required_url(task: Mapping[str, Any]) -> bool:
    required_url = _required_url_token(task)
    if not required_url:
        return False
    final_state = _adversarial_final_state_reward(task)
    expected = final_state.get("expected") if isinstance(final_state, Mapping) else None
    return isinstance(expected, Mapping) and expected.get("contains") == required_url


def _adversarial_final_state_reward(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    reward = task.get("reward_function")
    if not isinstance(reward, Mapping):
        return None
    final_state = reward.get("adversarial_final_state_check")
    if isinstance(final_state, Mapping):
        return final_state
    adversarial = reward.get("adversarial_reward")
    if isinstance(adversarial, Mapping) and adversarial.get("type") == "FinalStateEvaluator":
        return adversarial
    return None


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for nested in value.values():
            strings.extend(_flatten_strings(nested))
        return strings
    if isinstance(value, list):
        strings = []
        for nested in value:
            strings.extend(_flatten_strings(nested))
        return strings
    return []


def _normalize_instruction(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.casefold().split())
    text = re.sub(r"`[^`]+`", "`<witness>`", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", text)
    return text.strip()


def _parse_action_counts(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    counts: dict[str, int] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"invalid --expected-action-counts item {part!r}; expected KIND=N")
        kind, raw_count = (item.strip() for item in part.split("=", 1))
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SystemExit(f"invalid count for {kind!r}: {raw_count!r}") from exc
        if count < 0:
            raise SystemExit(f"count for {kind!r} must be non-negative")
        counts[kind] = count
    return counts or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Run directory or Phase 2 adversarial_tasks.json path.",
    )
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--min-tasks", type=int, default=1)
    parser.add_argument("--max-failures", type=int, default=50)
    parser.add_argument(
        "--allow-missing-profile-metadata",
        action="store_true",
        help="Allow auditing a standalone tasks file without Phase 1 profile metadata.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--expected-action-counts", default=None)
    parser.add_argument("--min-purpose-tags", type=int, default=None)
    parser.add_argument("--min-style-tags", type=int, default=None)
    parser.add_argument("--reject-duplicate-instructions", action="store_true")
    args = parser.parse_args(argv)

    tasks_path, run_dir = _resolve_inputs(args.path)
    if not tasks_path.exists():
        print(f"ERROR: adversarial tasks artifact not found: {tasks_path}", file=sys.stderr)
        return 1
    tasks_raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks_raw, list):
        raise SystemExit("adversarial_tasks must be a JSON array")
    tasks = [task for task in tasks_raw if isinstance(task, dict)]
    report = analyze_pure_action_paper_suite(
        tasks,
        artifact=str(tasks_path),
        run_dir=run_dir,
        max_failures=args.max_failures,
        allow_missing_profile_metadata=args.allow_missing_profile_metadata,
        expected_action_counts=_parse_action_counts(args.expected_action_counts),
        min_purpose_tags=args.min_purpose_tags,
        min_style_tags=args.min_style_tags,
        reject_duplicate_instructions=args.reject_duplicate_instructions,
    )
    failed_gate = (
        bool(report["suite_failures"])
        or report["failed_tasks"] > 0
        or report["total_tasks"] < args.min_tasks
        or (args.expected_count is not None and report["total_tasks"] != args.expected_count)
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "pure action paper suite audit: "
            f"{report['valid_tasks']} valid task(s), "
            f"{report['failed_tasks']} failed task(s), "
            f"{report['total_tasks']} total task(s), "
            f"profile={report.get('task_capability_profile') or '<missing>'}"
        )
        if report["suite_failures"]:
            print(f"ERROR: suite failures: {report['suite_failures']}", file=sys.stderr)
        if report["total_tasks"] < args.min_tasks:
            print(
                f"ERROR: found {report['total_tasks']} task(s), minimum required is "
                f"{args.min_tasks}",
                file=sys.stderr,
            )
        if args.expected_count is not None and report["total_tasks"] != args.expected_count:
            print(
                f"ERROR: found {report['total_tasks']} task(s), expected {args.expected_count}",
                file=sys.stderr,
            )
        for failure in report["failures"]:
            print(
                "FAIL "
                f"{failure.get('id')}: action={failure.get('action_kind')} "
                f"card={failure.get('task_card_id')} failures={failure.get('failures')}",
                file=sys.stderr,
            )
    return 1 if failed_gate else 0


def _resolve_inputs(path: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        return path / "phase_2" / "adversarial_tasks.json", path
    run_dir = path.parent.parent if path.parent.name == "phase_2" else None
    return path, run_dir


if __name__ == "__main__":
    raise SystemExit(main())
