"""Task-bank CLI commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _task_bank_path_from_args(args: argparse.Namespace) -> Path:
    from warp_taskgen.task_bank import default_task_bank_path

    return Path(getattr(args, "path", None) or default_task_bank_path())


def _format_task_bank_summary(summary: dict[str, Any], *, path: Path) -> str:
    def fmt_counts(values: Any) -> str:
        if not isinstance(values, dict) or not values:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))

    return "\n".join(
        [
            f"Task bank: {path}",
            (
                "Events: "
                f"total={summary.get('total_events', 0)} "
                f"admitted={summary.get('admitted_tasks', 0)} "
                f"active_admitted={summary.get('active_admitted_tasks', 0)} "
                f"retired_admitted={summary.get('retired_admitted_tasks', 0)} "
                f"phase4_results={summary.get('phase4_results', 0)}"
            ),
            f"By site: {fmt_counts(summary.get('by_site'))}",
            f"By origin: {fmt_counts(summary.get('by_origin'))}",
            f"By surface: {fmt_counts(summary.get('by_surface'))}",
            f"Active by surface: {fmt_counts(summary.get('active_by_surface'))}",
            f"Retired by surface: {fmt_counts(summary.get('retired_by_surface'))}",
            f"By archetype: {fmt_counts(summary.get('by_archetype'))}",
            (
                "Latest: "
                f"{summary.get('latest_created_at') or 'none'} "
                f"{summary.get('latest_event_id') or 'none'}"
            ),
        ]
    )


def _dispatch_task_bank(args: argparse.Namespace) -> int:
    import json

    from warp_taskgen.task_bank import (
        TaskBankError,
        admitted_events_from_phase2c_run,
        append_task_bank_events,
        is_active_task_bank_event,
        load_task_bank,
        summarize_task_bank,
    )

    path = _task_bank_path_from_args(args)
    try:
        if args.task_bank_command == "append":
            events = admitted_events_from_phase2c_run(Path(args.run_dir))
            appended = append_task_bank_events(path, events)
            payload = {
                "task_bank_path": str(path),
                "run_dir": str(args.run_dir),
                "source": args.source,
                "appended": len(appended),
                "summary": summarize_task_bank(load_task_bank(path)),
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(
                    f"Task bank append: appended={payload['appended']} "
                    f"path={payload['task_bank_path']}"
                )
                print(_format_task_bank_summary(payload["summary"], path=path))
            return 0
        if args.task_bank_command == "status":
            summary = summarize_task_bank(load_task_bank(path))
            if args.json:
                print(
                    json.dumps(
                        {"task_bank_path": str(path), "summary": summary}, indent=2, sort_keys=True
                    )
                )
            else:
                print(_format_task_bank_summary(summary, path=path))
            return 0
        if args.task_bank_command == "export":
            events = load_task_bank(path)
            if args.summary:
                payload: Any = summarize_task_bank(events)
            else:
                payload = (
                    events
                    if args.include_retired_carriers
                    else [event for event in events if is_active_task_bank_event(event)]
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Task bank export: wrote {args.output}")
            return 0
    except TaskBankError as exc:
        print(f"task-bank failed: {exc}", file=sys.stderr)
        return 2
    return 2


__all__ = ["_dispatch_task_bank", "_format_task_bank_summary", "_task_bank_path_from_args"]
