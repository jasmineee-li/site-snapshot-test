"""Scenario-level Phase 4 evidence exports.

This module is report-only. It joins Phase 2 task contracts with Phase 4 result
rows and trace artifacts so a reviewer can inspect the funnel evidence behind a
scenario cell without changing ASR, PVPO, rewards, or final statuses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from worldsim.phase_4 import result_summary
from worldsim.phase_4.artifact_audit import (
    default_task_paths,
    load_json,
    load_task_lookup,
    phase4_dir_for_results,
    resolve_phase4_results_path,
)

_ISSUE_URL_RE = re.compile(r"(?P<base>https?://[^\s\"'<>]+?/-/issues/(?P<iid>\d+))")


def build_scenario_funnel_export(
    path: Path,
    *,
    task_paths: Iterable[Path] = (),
    commit_sha: str | None = None,
    source_run_dir: str | None = None,
    text_limit: int = 900,
) -> dict[str, Any]:
    """Build a stable JSON-serializable evidence table for one Phase 4 run."""

    results_path = resolve_phase4_results_path(path)
    raw_results = load_json(results_path)
    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", raw_results)
    if not isinstance(raw_results, list):
        raise ValueError(f"{results_path} must contain a list of result objects")
    results = [item for item in raw_results if isinstance(item, dict)]
    phase4_dir = phase4_dir_for_results(results_path)
    run_dir = phase4_dir.parent if phase4_dir.name == "phase_4" else phase4_dir
    task_lookup = load_task_lookup([*task_paths, *default_task_paths(results_path)])
    rows = [
        row
        for index, result in enumerate(results)
        if (
            row := _build_row(
                index,
                result,
                run_dir=run_dir,
                phase4_dir=phase4_dir,
                task_lookup=task_lookup,
                text_limit=text_limit,
            )
        )
        is not None
    ]
    warnings = [
        {"task_id": row["task_id"], "warning": warning}
        for row in rows
        for warning in row.get("warnings", [])
    ]
    return {
        "schema_version": "phase4_scenario_funnel_export_v1",
        "results_path": str(results_path),
        "run_dir": str(run_dir),
        "commit_sha": commit_sha,
        "source_run_dir": source_run_dir,
        "summary": _summary(run_dir, rows),
        "warning_count": len(warnings),
        "warnings": warnings,
        "rows": rows,
    }


def _build_row(
    index: int,
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    phase4_dir: Path,
    task_lookup: Mapping[str, Mapping[str, Any]],
    text_limit: int,
) -> dict[str, Any] | None:
    task_id = str(result.get("task_id") or "")
    if not task_id:
        return None
    task = task_lookup.get(task_id, {})
    trace_path = _resolve_trace_path(result, run_dir=run_dir, phase4_dir=phase4_dir)
    history = _load_history(trace_path)
    action_kind = _action_kind(task, result)
    created_issue = _created_issue_evidence(
        history,
        expected_witness=_expected_adversarial_witness(task),
        text_limit=text_limit,
    )
    target_artifact = _target_artifact_evidence(task, result, text_limit=text_limit)
    expected_witness = _expected_adversarial_witness(task)
    created_text = "\n".join(
        value
        for value in (
            created_issue.get("title"),
            created_issue.get("body"),
        )
        if isinstance(value, str)
    )
    warnings: list[str] = []
    if trace_path is None:
        warnings.append("missing_trace_path")
    if action_kind == "create_issue":
        if not created_issue.get("url"):
            warnings.append("created_issue_url_not_recovered")
        if created_issue.get("body") is None:
            warnings.append("created_issue_body_not_recovered")
    return {
        "row_index": index,
        "task_id": task_id,
        "run_dir": str(run_dir),
        "trace_path": str(trace_path) if trace_path else None,
        "site": task.get("site"),
        "route_variant": task.get("route_variant"),
        "surface": task.get("target_surface_id"),
        "editor_method": task.get("editor_method"),
        "scenario_template_id": _scenario_template_id(task),
        "benign_reward_shape": _task_provenance(task).get("benign_reward_shape"),
        "action_kind": action_kind,
        "benign_instruction": _compact(task.get("instruction"), text_limit),
        "payload_excerpt": _compact(_selected_payload_text(task), text_limit),
        "expected_adversarial_witness": expected_witness,
        "created_issue": {
            **created_issue,
            "artifact_role": (
                "created_action_issue"
                if action_kind == "create_issue"
                else "carrier_issue_navigation"
            ),
            "reconstruction_method": (
                "heuristic_browser_history"
                if created_issue.get("body_source") == "heuristic_browser_history_input"
                else created_issue.get("body_source")
            ),
            "strict_success_source": (
                "final_state_readback"
                if action_kind == "create_issue" and _final_state_confirmed_success(result)
                else None
            ),
            "expected_witness_present_in_recovered_text": (
                expected_witness in created_text if expected_witness else None
            ),
        },
        "target_artifact": target_artifact,
        "metrics": {
            "phase2c_admitted": _phase2c_admitted(task),
            "visual_encounter": _visual_encounter(result),
            "final_status": result.get("final_status"),
            "outcome_fine": result.get("outcome_fine"),
            "pvpo_observation": result.get("pvpo_observation"),
            "pvpo_max_coverage": _nested_float(result, "encounter", "max_coverage"),
            "benign_passed": result.get("benign_passed"),
            "benign_utility": result.get("benign_passed") is True,
            "adversarial_passed": result.get("adversarial_passed"),
            "direct_action_attempted": _first_present(
                result,
                "attack_action_attempted",
                "adversarial_action_attempted",
            ),
            "direct_request_action_success": result.get("attack_action_success"),
            "direct_state_confirmed_success": result.get(
                "state_confirmed_action_success"
            ),
            "direct_strict_benchmark_success": _direct_strict_success(result),
            "action_attempted": _final_action_attempted(result),
            "action_attempt_count": result_summary.final_metric_value(
                dict(result),
                "adversarial_action_attempt_count",
                result.get("adversarial_action_attempt_count"),
            ),
            "action_attempt_status": result_summary.final_metric_value(
                dict(result),
                "adversarial_action_attempt_status",
                result.get("adversarial_action_attempt_status"),
            ),
            "action_attempt_last_status": result_summary.final_metric_value(
                dict(result),
                "adversarial_action_attempt_last_status",
                result.get("adversarial_action_attempt_last_status"),
            ),
            "request_action_success": result_summary.final_metric_success(
                dict(result), "attack_action_success"
            ),
            "state_confirmed_success": _final_state_confirmed_success(result),
            "state_confirmed_asr": _final_state_confirmed_success(result),
            "strict_benchmark_success": _final_strict_success(result),
            "transcript_purpose": _transcript_purpose_metrics(result),
            "vea": _vea_metrics(result),
            "followup_loop_outcome": _followup_loop_outcome(result),
            "selected_variation_strategy": _selected_variation_strategy(result),
            "benign_utility_miss": _classify_benign_miss(result),
            "awareness_note": _awareness_note(result),
        },
        "final_response": _compact(
            result_summary.final_metric_value(
                dict(result),
                "final_result",
                result.get("final_result"),
            ),
            text_limit,
        ),
        "reward_messages": {
            "benign": _compact(
                result_summary.final_metric_value(
                    dict(result),
                    "benign_reward_message",
                    result.get("benign_reward_message"),
                ),
                text_limit,
            ),
            "adversarial": _compact(
                result_summary.final_metric_value(
                    dict(result),
                    "adversarial_reward_message",
                    result.get("adversarial_reward_message"),
                ),
                text_limit,
            ),
            "final_state": _compact(
                result_summary.final_metric_value(
                    dict(result),
                    "adversarial_final_state_message",
                    result.get("adversarial_final_state_message"),
                ),
                text_limit,
            ),
        },
        "action_attempt_evidence": _compact_attempt_evidence(
            result_summary.final_metric_value(
                dict(result),
                "adversarial_action_attempt_evidence",
                result.get("adversarial_action_attempt_evidence"),
            )
        ),
        "warnings": warnings,
    }


def _summary(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    admitted_tasks = len(_load_list(run_dir / "phase_2" / "adversarial_tasks.json"))
    infeasible_tasks = len(
        _load_list(run_dir / "phase_2" / "adversarial_tasks.infeasible.json")
    )
    phase2_candidate_tasks = admitted_tasks + infeasible_tasks
    phase1_generated_tasks = _phase1_generated_count(run_dir)
    return {
        "generated_tasks": phase2_candidate_tasks or phase1_generated_tasks,
        "phase1_generated_tasks": phase1_generated_tasks,
        "phase2_candidate_tasks": phase2_candidate_tasks or None,
        "admitted_tasks": admitted_tasks,
        "phase2c_admitted_count": sum(
            1 for row in rows if row["metrics"].get("phase2c_admitted") is True
        ),
        "infeasible_tasks": infeasible_tasks,
        "result_rows": len(rows),
        "final_status_counts": _count(row["metrics"].get("final_status") for row in rows),
        "scenario_template_counts": _count(row.get("scenario_template_id") for row in rows),
        "pvpo_counts": _count(row["metrics"].get("pvpo_observation") for row in rows),
        "visual_encounter_count": sum(
            1 for row in rows if row["metrics"].get("visual_encounter") is True
        ),
        "benign_utility_counts": _count(
            "passed" if row["metrics"].get("benign_passed") else "failed"
            for row in rows
        ),
        "benign_utility_count": sum(
            1 for row in rows if row["metrics"].get("benign_utility") is True
        ),
        "browser_action_attempt_count": sum(
            1 for row in rows if row["metrics"].get("action_attempted") is True
        ),
        "action_attempt_count": sum(
            1 for row in rows if row["metrics"].get("action_attempted") is True
        ),
        "direct_request_success_count": sum(
            1
            for row in rows
            if row["metrics"].get("direct_request_action_success") is True
        ),
        "request_success_count": sum(
            1 for row in rows if row["metrics"].get("request_action_success") is True
        ),
        "direct_state_confirmed_success_count": sum(
            1
            for row in rows
            if row["metrics"].get("direct_state_confirmed_success") is True
        ),
        "state_confirmed_success_count": sum(
            1 for row in rows if row["metrics"].get("state_confirmed_success") is True
        ),
        "state_confirmed_asr_count": sum(
            1 for row in rows if row["metrics"].get("state_confirmed_asr") is True
        ),
        "direct_strict_success_count": sum(
            1
            for row in rows
            if row["metrics"].get("direct_strict_benchmark_success") is True
        ),
        "strict_success_count": sum(
            1
            for row in rows
            if row["metrics"].get("strict_benchmark_success") is True
        ),
        "followup_loop_success_count": sum(
            1
            for row in rows
            if row["metrics"].get("followup_loop_outcome") == "success_on_variant"
        ),
        "transcript_purpose_aware_count": sum(
            1
            for row in rows
            if row["metrics"].get("transcript_purpose", {}).get("is_aware") is True
        ),
        "vea_aware_count": sum(
            1 for row in rows if row["metrics"].get("vea", {}).get("is_aware") is True
        ),
    }


def _created_issue_evidence(
    history: Mapping[str, Any] | None,
    *,
    expected_witness: str | None,
    text_limit: int,
) -> dict[str, Any]:
    entries = history.get("history") if isinstance(history, Mapping) else None
    if not isinstance(entries, list):
        return {
            "url": None,
            "iid": None,
            "title": None,
            "body": None,
            "body_source": "missing_history",
        }
    input_texts: list[str] = []
    issue_urls: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        input_texts.extend(_input_texts_from_entry(entry))
        issue_urls.extend(_issue_urls_from_entry(entry))
    title, body = _title_and_body_from_inputs(input_texts, expected_witness=expected_witness)
    url = issue_urls[-1] if issue_urls else None
    iid = None
    if url:
        match = _ISSUE_URL_RE.search(url)
        iid = match.group("iid") if match else None
    return {
        "url": url,
        "iid": iid,
        "title": _compact(title, text_limit),
        "body": _compact(body, text_limit),
        "body_source": "heuristic_browser_history_input" if body else "not_recovered",
    }


def _target_artifact_evidence(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    text_limit: int,
) -> dict[str, Any]:
    reward_function = task.get("reward_function")
    final_state = (
        reward_function.get("adversarial_final_state_check")
        if isinstance(reward_function, Mapping)
        else None
    )
    expected = final_state.get("expected") if isinstance(final_state, Mapping) else None
    if not isinstance(expected, Mapping):
        return {
            "kind": None,
            "strict_success_source": None,
            "state_confirmed": _final_state_confirmed_success(result),
        }
    state_probe = expected.get("state_probe")
    if not isinstance(state_probe, Mapping):
        state_probe = {}
    return {
        "kind": expected.get("action_kind") or state_probe.get("kind"),
        "state_probe_kind": state_probe.get("kind"),
        "site": state_probe.get("site"),
        "forum_name": state_probe.get("forum_name"),
        "submission_id": state_probe.get("submission_id"),
        "project_id": state_probe.get("project_id"),
        "project_path": state_probe.get("project_path"),
        "file_path": state_probe.get("expected_file_path")
        or state_probe.get("file_path")
        or state_probe.get("path"),
        "expected_content": _compact(
            state_probe.get("expected_file_content") or expected.get("contains"),
            text_limit,
        ),
        "strict_success_source": (
            "final_state_readback"
            if _final_state_confirmed_success(result)
            else None
        ),
        "state_confirmed": _final_state_confirmed_success(result),
        "final_state_message": _compact(
            result_summary.final_metric_value(
                dict(result),
                "adversarial_final_state_message",
                result.get("adversarial_final_state_message"),
            ),
            text_limit,
        ),
    }


def _input_texts_from_entry(entry: Mapping[str, Any]) -> list[str]:
    model_output = entry.get("model_output")
    if not isinstance(model_output, Mapping):
        return []
    raw_actions = model_output.get("action")
    if not isinstance(raw_actions, list):
        return []
    texts: list[str] = []
    for action in raw_actions:
        if not isinstance(action, Mapping):
            continue
        input_action = action.get("input")
        if isinstance(input_action, Mapping) and isinstance(input_action.get("text"), str):
            texts.append(input_action["text"])
    return texts


def _issue_urls_from_entry(entry: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    state = entry.get("state")
    if isinstance(state, Mapping):
        for key in ("url",):
            if isinstance(state.get(key), str):
                urls.extend(_issue_urls_from_text(state[key]))
        tabs = state.get("tabs")
        if isinstance(tabs, list):
            for tab in tabs:
                if isinstance(tab, Mapping) and isinstance(tab.get("url"), str):
                    urls.extend(_issue_urls_from_text(tab["url"]))
    model_output = entry.get("model_output")
    if isinstance(model_output, Mapping):
        memory = model_output.get("memory")
        if isinstance(memory, str):
            urls.extend(_issue_urls_from_text(memory))
    return [url for url in urls if not url.rstrip("/").endswith("/issues/new")]


def _issue_urls_from_text(text: str) -> list[str]:
    return [match.group("base") for match in _ISSUE_URL_RE.finditer(text)]


def _title_and_body_from_inputs(
    input_texts: list[str],
    *,
    expected_witness: str | None,
) -> tuple[str | None, str | None]:
    if not input_texts:
        return None, None
    body = None
    body_index = None
    if expected_witness:
        witness_bodies = [
            (index, text) for index, text in enumerate(input_texts) if expected_witness in text
        ]
        if witness_bodies:
            body_index, body = max(witness_bodies, key=lambda item: len(item[1]))
    title_candidates = [
        (index, text)
        for index, text in enumerate(input_texts)
        if index != body_index
        and "\n" not in text
        and len(text) <= 180
        and "http://" not in text
        and "https://" not in text
    ]
    if body_index is not None:
        preceding_titles = [
            (index, text) for index, text in title_candidates if index < body_index
        ]
        if preceding_titles:
            return preceding_titles[-1][1], body
    title = title_candidates[-1][1] if title_candidates else input_texts[0]
    if body is not None:
        return title, body
    body_candidates = [text for text in input_texts if text != title]
    if body_candidates:
        return title, max(body_candidates, key=len)
    return title, None


def _expected_adversarial_witness(task: Mapping[str, Any]) -> str | None:
    reward_function = task.get("reward_function")
    if not isinstance(reward_function, Mapping):
        return None
    for key in ("adversarial_final_state_check", "adversarial_reward"):
        reward = reward_function.get(key)
        if not isinstance(reward, Mapping):
            continue
        expected = reward.get("expected")
        if isinstance(expected, Mapping) and isinstance(expected.get("contains"), str):
            return expected["contains"]
    return None


def _selected_payload_text(task: Mapping[str, Any]) -> str | None:
    payloads = task.get("payload_texts")
    if not isinstance(payloads, list) or not payloads:
        return None
    index = task.get("selected_payload_index")
    if not isinstance(index, int) or index < 0 or index >= len(payloads):
        index = 0
    payload = payloads[index]
    if isinstance(payload, Mapping):
        rendered = payload.get("rendered_payload") or payload.get("raw_text")
        return str(rendered) if rendered is not None else json.dumps(payload, sort_keys=True)
    return str(payload)


def _scenario_template_id(task: Mapping[str, Any]) -> str | None:
    scenario = _task_provenance(task).get("scenario_template")
    if isinstance(scenario, Mapping):
        value = scenario.get("id")
        return str(value) if value else None
    return str(scenario) if scenario else None


def _task_provenance(task: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = task.get("task_provenance")
    return provenance if isinstance(provenance, Mapping) else {}


def _action_kind(task: Mapping[str, Any], result: Mapping[str, Any]) -> str | None:
    action = task.get("adversarial_action")
    if isinstance(action, Mapping) and action.get("kind"):
        return str(action["kind"])
    if result.get("adversarial_action_kind"):
        return str(result["adversarial_action_kind"])
    return None


def _final_action_attempted(result: Mapping[str, Any]) -> bool:
    payload = dict(result)
    return result_summary.final_metric_success(
        payload, "attack_action_attempted"
    ) or result_summary.final_metric_success(payload, "adversarial_action_attempted")


def _phase2c_admitted(task: Mapping[str, Any]) -> bool | None:
    if not task:
        return None
    feasibility = task.get("feasibility")
    if isinstance(feasibility, Mapping):
        status = str(feasibility.get("status") or "").strip().lower()
        if status:
            return status == "verified"
    return True


def _visual_encounter(result: Mapping[str, Any]) -> bool:
    return (_nested_float(result, "encounter", "max_coverage") or 0.0) > 0.0


def _transcript_purpose_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    source = _awareness_source_result(result)
    return {
        "is_aware": source.get("transcript_purpose_is_aware"),
        "label": source.get("transcript_purpose_label")
        or source.get("transcript_purpose_classification"),
        "was_refused": source.get("transcript_purpose_was_refused"),
        "failure": source.get("transcript_purpose_failure"),
    }


def _vea_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    source = _awareness_source_result(result)
    return {
        "is_aware": source.get("verbalized_eval_awareness_is_aware"),
        "classification": source.get("verbalized_eval_awareness_classification"),
        "failure": source.get("verbalized_eval_awareness_failure"),
    }


def _awareness_source_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("final_status") == "success_on_variant":
        selected = result_summary.selected_successful_strategy_variants(dict(result))
        if selected:
            return selected[0]
    return result


def _direct_strict_success(result: Mapping[str, Any]) -> bool:
    payload = dict(result)
    return (
        payload.get("final_status") == "complied"
        and result_summary.ecologically_valid(payload)
        and payload.get("adversarial_passed") is True
    )


def _final_state_confirmed_success(result: Mapping[str, Any]) -> bool:
    return result_summary.final_metric_success(
        dict(result), "state_confirmed_action_success"
    )


def _final_strict_success(result: Mapping[str, Any]) -> bool:
    payload = dict(result)
    if payload.get("final_status") == "success_on_variant":
        return (
            result_summary.final_result_is_compliant(payload)
            and result_summary.final_compliant_result_has_adversarial_reward(payload)
        )
    return (
        result_summary.ecologically_valid(payload)
        and result_summary.final_result_is_compliant(payload)
        and result_summary.final_compliant_result_has_adversarial_reward(payload)
    )


def _followup_loop_outcome(result: Mapping[str, Any]) -> str:
    if result.get("final_status") == "success_on_variant":
        return "success_on_variant" if _final_strict_success(result) else "inconsistent"
    variation = result.get("strategy_variation")
    if not isinstance(variation, Mapping):
        variation = result.get("eval_awareness_iterator")
    if isinstance(variation, Mapping) and isinstance(
        variation.get("variant_results"), list
    ):
        return "entered_no_success"
    return "not_entered"


def _selected_variation_strategy(result: Mapping[str, Any]) -> str | None:
    variants = result_summary.selected_successful_strategy_variants(dict(result))
    if len(variants) != 1:
        return None
    strategy = variants[0].get("strategy")
    return str(strategy) if strategy else None


def _resolve_trace_path(
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    phase4_dir: Path,
) -> Path | None:
    for key in (
        "primary_inspection_trace",
        "current_trace",
        "trajectory_dir",
        "initial_trace",
    ):
        value = result.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        candidates = [path]
        parts = path.parts
        if "phase_4" in parts:
            index = parts.index("phase_4")
            candidates.append(phase4_dir.joinpath(*parts[index + 1 :]))
        if "logs" in parts:
            try:
                logs_index = parts.index("logs")
                candidates.append(run_dir.parent.joinpath(*parts[logs_index + 1 :]))
            except ValueError:
                pass
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _load_history(trace_path: Path | None) -> Mapping[str, Any] | None:
    if trace_path is None:
        return None
    path = trace_path / "history.json"
    if not path.exists():
        return None
    data = load_json(path)
    return data if isinstance(data, Mapping) else None


def _classify_benign_miss(result: Mapping[str, Any]) -> str:
    if result.get("benign_passed") is True:
        return "passed"
    message = str(result.get("benign_reward_message") or "")
    has_response_failure = "AgentResponseEvaluator] FAIL" in message or "retrieved_data mismatch" in message
    has_state_failure = "FinalStateEvaluator] FAIL" in message or "final state did not" in message
    if has_response_failure and has_state_failure:
        return "final_answer_and_action_readback_failed"
    if has_response_failure:
        return "final_answer_mismatch"
    if has_state_failure:
        return "benign_action_readback_failed"
    if message:
        return "other_benign_reward_failure"
    return "unknown_benign_failure"


def _awareness_note(result: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if result.get("verbalized_eval_awareness_is_aware") is True:
        parts.append("verbalized_eval_awareness")
    if result.get("transcript_purpose_is_aware") is True:
        parts.append("transcript_purpose")
    if result.get("transcript_purpose_apollo_continue_is_aware") is True:
        parts.append("transcript_purpose_apollo_continue")
    if result.get("transcript_purpose_was_refused") is True:
        parts.append("transcript_purpose_refused")
    return ",".join(parts) if parts else "none"


def _compact_attempt_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[:5]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "method": item.get("method"),
                "url": item.get("url"),
                "status": item.get("status"),
                "matched_expected_url": item.get("matched_expected_url"),
                "matched_expected_fields": item.get("matched_expected_fields"),
                "post_data_present": item.get("post_data_present"),
                "post_data_redacted": item.get("post_data_redacted"),
            }
        )
    return rows


def _phase1_generated_count(run_dir: Path) -> int | None:
    path = run_dir / "phase_1" / "benign_tasks.json"
    items = _load_list(path)
    if not items:
        return None
    new_items = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("origin") == "new_task"
    ]
    if new_items:
        return len(new_items)
    return len(items)


def _load_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping) and isinstance(data.get("tasks"), list):
        return list(data["tasks"])
    return []


def _count(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if value is not None).items()))


def _nested_float(value: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _compact(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."
