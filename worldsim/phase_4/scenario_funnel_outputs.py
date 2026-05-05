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
        "created_issue_body_source",
        "created_issue_reconstruction_method",
        "created_issue_artifact_role",
        "target_artifact_kind",
        "target_artifact_state_probe_kind",
        "target_artifact_site",
        "target_artifact_forum_name",
        "target_artifact_submission_id",
        "target_artifact_project_path",
        "target_artifact_file_path",
        "target_artifact_state_confirmed",
        "target_artifact_strict_success_source",
        "strict_success_source",
        "expected_adversarial_witness",
        "expected_witness_present_in_recovered_text",
        "benign_passed",
        "benign_utility_miss",
        "pvpo_observation",
        "pvpo_max_coverage",
        "final_status",
        "direct_action_attempted",
        "action_attempted",
        "direct_request_action_success",
        "request_action_success",
        "direct_state_confirmed_success",
        "state_confirmed_success",
        "direct_strict_benchmark_success",
        "strict_benchmark_success",
        "followup_loop_outcome",
        "selected_variation_strategy",
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
    target = row.get("target_artifact") if isinstance(row.get("target_artifact"), dict) else {}
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
        "created_issue_body_source": created.get("body_source"),
        "created_issue_reconstruction_method": created.get("reconstruction_method"),
        "created_issue_artifact_role": created.get("artifact_role"),
        "target_artifact_kind": target.get("kind"),
        "target_artifact_state_probe_kind": target.get("state_probe_kind"),
        "target_artifact_site": target.get("site"),
        "target_artifact_forum_name": target.get("forum_name"),
        "target_artifact_submission_id": target.get("submission_id"),
        "target_artifact_project_path": target.get("project_path"),
        "target_artifact_file_path": target.get("file_path"),
        "target_artifact_state_confirmed": target.get("state_confirmed"),
        "target_artifact_strict_success_source": target.get("strict_success_source"),
        "strict_success_source": created.get("strict_success_source"),
        "expected_adversarial_witness": row.get("expected_adversarial_witness"),
        "expected_witness_present_in_recovered_text": created.get(
            "expected_witness_present_in_recovered_text"
        ),
        "benign_passed": metrics.get("benign_passed"),
        "benign_utility_miss": metrics.get("benign_utility_miss"),
        "pvpo_observation": metrics.get("pvpo_observation"),
        "pvpo_max_coverage": metrics.get("pvpo_max_coverage"),
        "final_status": metrics.get("final_status"),
        "direct_action_attempted": metrics.get("direct_action_attempted"),
        "action_attempted": metrics.get("action_attempted"),
        "direct_request_action_success": metrics.get("direct_request_action_success"),
        "request_action_success": metrics.get("request_action_success"),
        "direct_state_confirmed_success": metrics.get(
            "direct_state_confirmed_success"
        ),
        "state_confirmed_success": metrics.get("state_confirmed_success"),
        "direct_strict_benchmark_success": metrics.get(
            "direct_strict_benchmark_success"
        ),
        "strict_benchmark_success": metrics.get("strict_benchmark_success"),
        "followup_loop_outcome": metrics.get("followup_loop_outcome"),
        "selected_variation_strategy": metrics.get("selected_variation_strategy"),
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
        f"- Final action attempts/request/state-confirmed: `{summary.get('action_attempt_count')}` / `{summary.get('request_success_count')}` / `{summary.get('state_confirmed_success_count')}`",
        f"- Direct request/state-confirmed/strict success: `{summary.get('direct_request_success_count')}` / `{summary.get('direct_state_confirmed_success_count')}` / `{summary.get('direct_strict_success_count')}`",
        f"- Final strict success/follow-up-loop success: `{summary.get('strict_success_count')}` / `{summary.get('followup_loop_success_count')}`",
        "- Created issue title/body snippets are heuristic Browser Use history reconstructions unless the row explicitly cites `strict_success_source=final_state_readback` for witness-only state evidence.",
        "- For non-`create_issue` actions, the issue URL is the carrier/navigation issue; the adversarial target is shown separately as `target_artifact`.",
        "",
        "| Task | Carrier/Created Issue | Target | Witness | Benign | PVPO | Status | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in export.get("rows", []):
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        created = row.get("created_issue") if isinstance(row.get("created_issue"), dict) else {}
        target = row.get("target_artifact") if isinstance(row.get("target_artifact"), dict) else {}
        target_label = (
            target.get("file_path")
            or target.get("project_path")
            or _reddit_target_label(target)
            or target.get("kind")
        )
        lines.append(
            "| "
            + " | ".join(
                _md_cell(value)
                for value in (
                    row.get("task_id"),
                    created.get("url") or created.get("iid") or "not recovered",
                    target_label or "",
                    _witness_label(created, target),
                    metrics.get("benign_utility_miss"),
                    metrics.get("pvpo_observation"),
                    metrics.get("final_status"),
                    f"attempt={metrics.get('action_attempted')} request={metrics.get('request_action_success')} state={metrics.get('state_confirmed_success')} strict={metrics.get('strict_benchmark_success')} loop={metrics.get('followup_loop_outcome')}",
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")


def _witness_label(created: dict[str, Any], target: dict[str, Any]) -> str:
    if created.get("expected_witness_present_in_recovered_text") is True:
        return "present in recovered browser text"
    if target.get("state_confirmed") is True:
        return "present in target readback"
    return "not recovered"


def _reddit_target_label(target: dict[str, Any]) -> str | None:
    site = target.get("site")
    forum = target.get("forum_name")
    submission = target.get("submission_id")
    if site != "reddit" or not forum:
        return None
    if submission:
        return f"reddit:{forum}/{submission}"
    return f"reddit:{forum}"
