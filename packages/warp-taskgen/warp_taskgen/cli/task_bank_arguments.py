"""Arguments for the WARP Taskgen ``task-bank`` command and its subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_task_bank_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``task-bank`` command with its ``append``, ``status``, and ``export`` subcommands."""
    task_bank_cmd = subparsers.add_parser(
        "task-bank",
        help="Manage the append-only admitted-task bank.",
    )
    task_bank_cmd.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Task-bank JSONL path. Defaults to "
            "WARP_TASKGEN_STATE_DIR/task_bank/events.jsonl, with "
            "WORLDSIM_STATE_DIR accepted as a legacy alias."
        ),
    )
    task_bank_sub = task_bank_cmd.add_subparsers(dest="task_bank_command", required=True)

    task_bank_append = task_bank_sub.add_parser(
        "append",
        help="Append admitted tasks from a verified Phase 2c run.",
    )
    task_bank_append.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="WARP Taskgen run dir containing phase_2/adversarial_tasks.json.",
    )
    task_bank_append.add_argument(
        "--source",
        choices=("phase2c",),
        default="phase2c",
        help="Source artifact to append from.",
    )
    task_bank_append.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )

    task_bank_status = task_bank_sub.add_parser("status", help="Show task-bank coverage counts.")
    task_bank_status.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )

    task_bank_export = task_bank_sub.add_parser(
        "export", help="Export task-bank events or summary."
    )
    task_bank_export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    task_bank_export.add_argument(
        "--summary",
        action="store_true",
        help="Export summary JSON instead of raw event records.",
    )
    task_bank_export.add_argument(
        "--include-retired-carriers",
        action="store_true",
        help="Include retired title-carrier events in raw exports. Summary exports always report both counts.",
    )
