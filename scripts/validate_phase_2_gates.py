#!/usr/bin/env python3
"""Validate Phase 2 Option A placement contract on an adversarial_tasks.json.

Two gates (docs/handoffs/phase-2-placement-systemic-gap.md §Validation gate):

  Gate 1 — no-dangling-mechanism.
    No editor_call creates a dangling parent artifact
    (create_project / create_group / create_forum), and every
    create_issue / create_submission / create_comment / create_issue_note /
    create_mr_note references the benign anchor via a {benign_*} template
    token.

  Gate 2 — DOM reachability.
    Every task carries feasibility.reachability in
    {reachable_direct, reachable_transitively}. Zero unreachable. Zero
    unverified.

Exit 0 iff both gates green. Non-zero with a structured reason table
otherwise. Must not crash on empty files or pre-existing artifacts that
predate Option A — both show up as "Gate 1: 0/0 ok" or similar.

Usage:
  uv run python scripts/validate_phase_2_gates.py \\
      --artifact logs/phase_2/adversarial_tasks.json

  # Dry-run: reports what would fail without exiting non-zero.
  uv run python scripts/validate_phase_2_gates.py \\
      --artifact logs/phase_2/adversarial_tasks.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BAD_METHODS = {"create_project", "create_group", "create_forum"}
_CHILD_CREATE_ARG_TOKENS: dict[str, tuple[str, str]] = {
    "create_issue": ("project_id", "{benign_project_id"),
    "create_issue_note": ("issue_iid", "{benign_issue_iid"),
    "create_mr_note": ("mr_iid", "{benign_mr_iid"),
    "create_submission": ("forum_name", "{benign_forum_name"),
    "create_comment": ("submission_id", "{benign_submission_id"),
}


def _is_wasp_task(task: dict[str, Any]) -> bool:
    for key in ("sites", "site"):
        raw = task.get(key)
        if isinstance(raw, str):
            if raw.strip().lower() in ("gitlab", "reddit"):
                return True
        elif isinstance(raw, list):
            if any(isinstance(s, str) and s.strip().lower() in ("gitlab", "reddit") for s in raw):
                return True
    return False


def _gate_1_violations(task: dict[str, Any]) -> list[str]:
    """Return a list of Gate 1 rejection reasons (empty == pass)."""
    reasons: list[str] = []
    seed = task.get("adversarial_data_seed") or {}
    if not isinstance(seed, dict):
        return []  # malformed seeds are caught by upstream validators
    calls = seed.get("editor_calls") or []
    if not isinstance(calls, list):
        return []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        method = str(call.get("method") or "")
        if method in _BAD_METHODS:
            reasons.append(f"editor_calls[{idx}].method={method} (dangling parent)")
            continue
        if method in _CHILD_CREATE_ARG_TOKENS:
            required_arg, token_prefix = _CHILD_CREATE_ARG_TOKENS[method]
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            value = str(args.get(required_arg, ""))
            if not value.startswith(token_prefix):
                reasons.append(
                    f"editor_calls[{idx}].{method}: args.{required_arg}={value!r} "
                    f"missing {token_prefix}...}}"
                )
    return reasons


def _gate_2_verdict(task: dict[str, Any]) -> tuple[bool, str]:
    """Return (passes_gate_2, reason)."""
    feasibility = task.get("feasibility") or {}
    if not isinstance(feasibility, dict):
        return False, "feasibility missing or malformed"
    if feasibility.get("status") != "verified":
        return False, f"feasibility.status={feasibility.get('status')!r} != 'verified'"
    reachability = feasibility.get("reachability")
    if reachability in ("reachable_direct", "reachable_transitively"):
        return True, reachability
    return False, f"reachability={reachability!r} (expected reachable_direct/_transitively)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifact",
        default="logs/phase_2/adversarial_tasks.json",
        help="Path to adversarial_tasks.json to validate",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report but exit 0 regardless of gate failures.",
    )
    ap.add_argument(
        "--skip-gate-2",
        action="store_true",
        help=(
            "Skip Gate 2 (reachability). Use during rollout before 2c has "
            "been re-run with the reachability probe enabled."
        ),
    )
    args = ap.parse_args()

    path = Path(args.artifact)
    if not path.exists():
        print(f"[gates] artifact not found: {path}", file=sys.stderr)
        print("[gates] (treating missing artifact as Gate 1: 0/0 ok, Gate 2: N/A)")
        return 0 if args.dry_run else 1

    try:
        tasks = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[gates] invalid JSON at {path}: {exc}", file=sys.stderr)
        return 0 if args.dry_run else 2

    if not isinstance(tasks, list):
        print(f"[gates] artifact root must be a list, got {type(tasks).__name__}", file=sys.stderr)
        return 0 if args.dry_run else 2

    gate_1_failures: list[tuple[str, list[str]]] = []
    gate_2_failures: list[tuple[str, str]] = []
    wasp_total = 0
    total = len(tasks)

    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "?")
        if _is_wasp_task(task):
            wasp_total += 1
            reasons = _gate_1_violations(task)
            if reasons:
                gate_1_failures.append((task_id, reasons))
        if not args.skip_gate_2:
            passes, reason = _gate_2_verdict(task)
            if not passes:
                gate_2_failures.append((task_id, reason))

    print(f"[gates] artifact: {path}")
    print(f"[gates] total tasks: {total} (WASP-scoped: {wasp_total})")
    print()

    if not gate_1_failures:
        print(
            f"[gates] Gate 1 (no dangling mechanism): "
            f"PASS ({wasp_total - len(gate_1_failures)}/{wasp_total} WASP tasks ok)"
        )
    else:
        print(
            f"[gates] Gate 1 (no dangling mechanism): "
            f"FAIL ({len(gate_1_failures)}/{wasp_total} WASP tasks rejected)"
        )
        for task_id, reasons in gate_1_failures[:20]:
            for reason in reasons:
                print(f"  - {task_id}: {reason}")
        if len(gate_1_failures) > 20:
            print(f"  ... and {len(gate_1_failures) - 20} more")

    print()

    if args.skip_gate_2:
        print("[gates] Gate 2 (reachability): SKIPPED (--skip-gate-2)")
        gate_2_failures = []
    elif not gate_2_failures:
        print(
            f"[gates] Gate 2 (reachability): "
            f"PASS ({total - len(gate_2_failures)}/{total} tasks verified reachable)"
        )
    else:
        print(
            f"[gates] Gate 2 (reachability): "
            f"FAIL ({len(gate_2_failures)}/{total} tasks not reachable_*)"
        )
        for task_id, reason in gate_2_failures[:20]:
            print(f"  - {task_id}: {reason}")
        if len(gate_2_failures) > 20:
            print(f"  ... and {len(gate_2_failures) - 20} more")

    if gate_1_failures or gate_2_failures:
        print()
        print("[gates] RESULT: one or more gates FAILED")
        return 0 if args.dry_run else 3

    print()
    print("[gates] RESULT: all gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
