"""Output renderers for Phase 4 scenario funnel evidence exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_scenario_funnel_json(export: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_scenario_funnel_csv(export: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "commit_sha",
        "source_run_dir",
        "task_id",
        "scenario_template_id",
        "action_kind",
        "surface",
        "editor_method",
        "created_issue_iid",
        "created_issue_url",
        "created_issue_title",
        "created_issue_body_excerpt",
        "expected_adversarial_witness",
        "expected_witness_present_in_recovered_text",
        "benign_passed",
        "benign_utility_miss",
        "pvpo_observation",
        "pvpo_max_coverage",
        "final_status",
        "action_attempted",
        "request_action_success",
        "state_confirmed_success",
        "awareness_note",
        "final_response",
        "trace_path",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in export.get("rows", []):
            writer.writerow(_csv_row(export, row))


def write_scenario_funnel_markdown(export: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown(export), encoding="utf-8")


def _csv_row(export: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    created = row.get("created_issue") if isinstance(row.get("created_issue"), dict) else {}
    return {
        "run_dir": row.get("run_dir"),
        "commit_sha": export.get("commit_sha"),
        "source_run_dir": export.get("source_run_dir"),
        "task_id": row.get("task_id"),
        "scenario_template_id": row.get("scenario_template_id"),
        "action_kind": row.get("action_kind"),
        "surface": row.get("surface"),
        "editor_method": row.get("editor_method"),
        "created_issue_iid": created.get("iid"),
        "created_issue_url": created.get("url"),
        "created_issue_title": created.get("title"),
        "created_issue_body_excerpt": created.get("body"),
        "expected_adversarial_witness": row.get("expected_adversarial_witness"),
        "expected_witness_present_in_recovered_text": created.get(
            "expected_witness_present_in_recovered_text"
        ),
        "benign_passed": metrics.get("benign_passed"),
        "benign_utility_miss": metrics.get("benign_utility_miss"),
        "pvpo_observation": metrics.get("pvpo_observation"),
        "pvpo_max_coverage": metrics.get("pvpo_max_coverage"),
        "final_status": metrics.get("final_status"),
        "action_attempted": metrics.get("action_attempted"),
        "request_action_success": metrics.get("request_action_success"),
        "state_confirmed_success": metrics.get("state_confirmed_success"),
        "awareness_note": metrics.get("awareness_note"),
        "final_response": row.get("final_response"),
        "trace_path": row.get("trace_path"),
        "warnings": ";".join(str(warning) for warning in row.get("warnings", [])),
    }


def _render_markdown(export: dict[str, Any]) -> str:
    summary = export.get("summary") if isinstance(export.get("summary"), dict) else {}
    lines = [
        "# Phase 4 Scenario Funnel Evidence",
        "",
        f"- Results: `{export.get('results_path')}`",
        f"- Commit: `{export.get('commit_sha') or 'unknown'}`",
        f"- Source run: `{export.get('source_run_dir') or 'unknown'}`",
        f"- Generated/admitted/infeasible: `{summary.get('generated_tasks')}` / `{summary.get('admitted_tasks')}` / `{summary.get('infeasible_tasks')}`",
        f"- Result rows: `{summary.get('result_rows')}`",
        f"- Final status counts: `{summary.get('final_status_counts')}`",
        f"- PVPO counts: `{summary.get('pvpo_counts')}`",
        f"- Action attempts/request/state-confirmed: `{summary.get('action_attempt_count')}` / `{summary.get('request_success_count')}` / `{summary.get('state_confirmed_success_count')}`",
        "",
        "| Task | Created Issue | Witness | Benign | PVPO | Status | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in export.get("rows", []):
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        created = row.get("created_issue") if isinstance(row.get("created_issue"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                _md_cell(value)
                for value in (
                    row.get("task_id"),
                    created.get("url") or created.get("iid") or "not recovered",
                    "present"
                    if created.get("expected_witness_present_in_recovered_text") is True
                    else "not recovered/present in readback only",
                    metrics.get("benign_utility_miss"),
                    metrics.get("pvpo_observation"),
                    metrics.get("final_status"),
                    f"attempt={metrics.get('action_attempted')} request={metrics.get('request_action_success')} state={metrics.get('state_confirmed_success')}",
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")
