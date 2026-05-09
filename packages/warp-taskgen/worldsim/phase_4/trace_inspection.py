"""Context-efficient Phase 4 trace inspection helpers.

This module is read-only. It projects existing Phase 4 results and trajectory
artifacts into small, agent-friendly views so debugging does not require ad hoc
JSON dumps of large signal matrices.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldsim.phase_4.artifact_audit import (
    default_task_paths,
    load_json,
    load_json_or_empty,
    load_task_lookup,
    phase4_dir_for_results,
    resolve_phase4_results_path,
)
from worldsim.phase_4.pvpo_observations import pvpo_observation_bucket
from worldsim.phase_4.result_summary import (
    ecologically_valid,
    final_metric_success,
    summarize_results,
    task_editor_method,
    task_origin,
    task_route_variant,
    task_site,
    task_surface,
)
from worldsim.phase_4.variant_trace_text import answer_label, compact_text

DEFAULT_FIELDS = [
    "task_id",
    "site",
    "action",
    "final_status",
    "outcome_fine",
    "benign_passed",
    "attack_attempted",
    "attack_success",
    "state_success",
    "max_coverage",
    "tp",
    "vea",
    "iterator_stop",
    "trajectory_dir",
]

DETAIL_FIELDS = [
    *DEFAULT_FIELDS,
    "surface",
    "reward_msg",
    "final_state_msg",
    "iterator_algorithm",
    "iterator_generated",
    "attempt_count",
    "answer_label",
    "origin",
    "route",
    "pvpo",
    "tp_aware",
    "vea_aware",
    "signals",
    "runner",
    "agentlab_status",
    "agentlab_reward",
    "agentlab_steps",
    "agentlab_error",
    "runtime_artifact_status",
    "browser_instance_scope",
    "agent_browser_connect_count",
    "auxiliary_browser_connect_count",
    "recycle_status",
    "sidecar_status",
    "sidecar_returncode",
]

ALL_FIELDS = DETAIL_FIELDS


def load_inspection(
    path: Path,
    *,
    task_paths: list[Path] | None = None,
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    results_path = resolve_phase4_results_path(path)
    raw = load_json(results_path)
    if not isinstance(raw, list):
        raise ValueError(f"{results_path} must contain a list")
    task_lookup = load_task_lookup([*(task_paths or []), *default_task_paths(results_path)])
    return (
        results_path,
        phase4_dir_for_results(results_path),
        [item for item in raw if isinstance(item, dict)],
        task_lookup,
    )


def filter_results(
    results: list[dict[str, Any]],
    task_lookup: dict[str, dict[str, Any]],
    *,
    phase4_dir: Path | None = None,
    site: str | None = None,
    action: str | None = None,
    status: str | None = None,
    outcome: str | None = None,
    task_id: str | None = None,
    surface: str | None = None,
    origin: str | None = None,
    route: str | None = None,
    pvpo: str | None = None,
    coverage_min: float | None = None,
    tp: str | None = None,
    vea: str | None = None,
    awareness: str | None = None,
    iterator_stop: str | None = None,
    iterator_algorithm: str | None = None,
    has_iterator: bool | None = None,
    benign_passed: bool | None = None,
    attack_attempted: bool | None = None,
    attack_success: bool | None = None,
    state_success: bool | None = None,
    has_trace: bool | None = None,
    missing_artifact: str | None = None,
    reward_contains: str | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for result in results:
        task = task_lookup.get(str(result.get("task_id", "")), {})
        if site and task_site(result, task) != site:
            continue
        if action and action_name(result, task) != action:
            continue
        if status and result.get("final_status") != status:
            continue
        if outcome and result.get("outcome_fine") != outcome:
            continue
        if task_id and result.get("task_id") != task_id:
            continue
        if surface and task_surface(task) != surface:
            continue
        if origin and task_origin(result, task) != origin:
            continue
        if route and task_route_variant(task) != route:
            continue
        if pvpo and _pvpo_filter_value(result) != pvpo:
            continue
        if coverage_min is not None and float(_max_coverage(result) or 0) < coverage_min:
            continue
        if tp and _aware_filter_value(result.get("transcript_purpose_is_aware")) != tp:
            continue
        if vea and _aware_filter_value(result.get("verbalized_eval_awareness_is_aware")) != vea:
            continue
        if awareness and _combined_awareness(result) != awareness:
            continue
        iterator = _iterator_record(result)
        if iterator_stop and _iterator_stop_reason(iterator) != iterator_stop:
            continue
        if iterator_algorithm and iterator.get("algorithm") != iterator_algorithm:
            continue
        if has_iterator is not None and bool(iterator) is not has_iterator:
            continue
        if benign_passed is not None and (result.get("benign_passed") is True) is not benign_passed:
            continue
        if (
            attack_attempted is not None
            and final_metric_success(result, "attack_action_attempted") is not attack_attempted
        ):
            continue
        if (
            attack_success is not None
            and final_metric_success(result, "attack_action_success") is not attack_success
        ):
            continue
        if (
            state_success is not None
            and final_metric_success(result, "state_confirmed_action_success") is not state_success
        ):
            continue
        resolved_phase4_dir = phase4_dir or Path(".")
        trace = _primary_trace(result, phase4_dir=resolved_phase4_dir)
        if has_trace is not None and (trace is not None) is not has_trace:
            continue
        if missing_artifact and not _artifact_missing(
            result,
            missing_artifact,
            phase4_dir=resolved_phase4_dir,
        ):
            continue
        if reward_contains:
            reward = str(result.get("adversarial_reward_message") or "")
            if reward_contains.lower() not in reward.lower():
                continue
        filtered.append(result)
    return filtered


def action_name(result: dict[str, Any], task: dict[str, Any] | None = None) -> str:
    for key in ("adversarial_action_kind", "adversarial_action_family"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    task = task or {}
    editor = task_editor_method(task)
    return editor if editor != "unknown" else "unknown"


def task_row(
    result: dict[str, Any],
    task_lookup: dict[str, dict[str, Any]],
    *,
    phase4_dir: Path,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    task_id = str(result.get("task_id") or "")
    task = task_lookup.get(task_id, {})
    iterator = _iterator_record(result)
    signals = result.get("signals") if isinstance(result.get("signals"), dict) else {}
    max_coverage = _max_coverage(result)
    trajectory_dir = _primary_trace(result, phase4_dir=phase4_dir)
    agentlab_fields = _agentlab_inspection_fields(result, trajectory_dir)
    row = {
        "task_id": task_id,
        "site": task_site(result, task),
        "surface": task_surface(task),
        "origin": task_origin(result, task),
        "route": task_route_variant(task),
        "action": action_name(result, task),
        "final_status": result.get("final_status"),
        "outcome_fine": result.get("outcome_fine"),
        "benign_passed": result.get("benign_passed"),
        "attack_attempted": final_metric_success(result, "attack_action_attempted"),
        "attack_success": final_metric_success(result, "attack_action_success"),
        "state_success": final_metric_success(result, "state_confirmed_action_success"),
        "max_coverage": max_coverage,
        "pvpo": pvpo_observation_bucket(result),
        "tp": result.get("transcript_purpose_classification"),
        "tp_aware": result.get("transcript_purpose_is_aware"),
        "vea": result.get("verbalized_eval_awareness_classification"),
        "vea_aware": result.get("verbalized_eval_awareness_is_aware"),
        "reward_msg": compact_text(result.get("adversarial_reward_message"), limit=180),
        "final_state_msg": compact_text(
            result.get("adversarial_final_state_message"),
            limit=180,
        ),
        "iterator_algorithm": iterator.get("algorithm"),
        "iterator_stop": _iterator_stop_reason(iterator),
        "iterator_generated": _nested(iterator, "adaptive_budget", "generated"),
        "attempt_count": len(iterator.get("attempts") or []),
        "trajectory_dir": str(trajectory_dir) if trajectory_dir else None,
        "answer_label": _answer_label_for_trace(trajectory_dir),
        "signals": _compact_signals(signals),
        **agentlab_fields,
    }
    selected = fields or DEFAULT_FIELDS
    return {field: row.get(field) for field in selected}


def build_summary(
    results_path: Path,
    phase4_dir: Path,
    results: list[dict[str, Any]],
    task_lookup: dict[str, dict[str, Any]],
    *,
    filtered: list[dict[str, Any]],
    sample_limit: int = 8,
) -> dict[str, Any]:
    summary = summarize_results(filtered, task_lookup=task_lookup)
    reward_messages = Counter(
        compact_text(result.get("adversarial_reward_message"), limit=180) or "missing"
        for result in filtered
    )
    iterator_stops = Counter(
        str(_iterator_stop_reason(_iterator_record(result)) or "missing") for result in filtered
    )
    iterator_algorithms = Counter(
        str(_iterator_record(result).get("algorithm") or "missing") for result in filtered
    )
    tp = Counter(str(result.get("transcript_purpose_classification") or "missing") for result in filtered)
    vea = Counter(
        str(result.get("verbalized_eval_awareness_classification") or "missing")
        for result in filtered
    )
    return {
        "schema_version": "phase4_trace_inspection_v1",
        "results_path": str(results_path),
        "phase4_dir": str(phase4_dir),
        "total_results": len(results),
        "matched_results": len(filtered),
        "final_status_counts": summary.get("final_status_counts", {}),
        "outcome_fine_counts": summary.get("outcome_fine_counts", {}),
        "site_counts": summary.get("site_counts", {}),
        "pvpo_observation_counts": summary.get("pvpo_observation_counts", {}),
        "tp_counts": dict(sorted(tp.items())),
        "vea_counts": dict(sorted(vea.items())),
        "iterator_algorithms": dict(sorted(iterator_algorithms.items())),
        "iterator_stop_reasons": dict(sorted(iterator_stops.items())),
        "top_reward_messages": _top_counter(reward_messages, limit=5),
        "samples": [
            task_row(result, task_lookup, phase4_dir=phase4_dir)
            for result in filtered[:sample_limit]
        ],
        "next_commands": next_commands(str(results_path.parent.parent), filtered[:1]),
    }


def build_task_detail(
    result: dict[str, Any],
    task_lookup: dict[str, dict[str, Any]],
    *,
    phase4_dir: Path,
    include_iterator: bool = False,
    include_refs: bool = False,
) -> dict[str, Any]:
    base = task_row(result, task_lookup, phase4_dir=phase4_dir, fields=DETAIL_FIELDS)
    base["why"] = _why(result)
    if include_iterator:
        trace = _primary_trace(result, phase4_dir=phase4_dir)
        base["iterator"] = _compact_iterator(_iterator_record(result), trace=trace)
    if include_refs:
        trace = _primary_trace(result, phase4_dir=phase4_dir)
        base["refs"] = _artifact_refs(trace)
        base["artifacts"] = artifact_manifest(trace)
    base["next_commands"] = next_commands(str(phase4_dir.parent), [result])
    return base


def build_timeline(
    result: dict[str, Any],
    *,
    phase4_dir: Path,
) -> dict[str, Any]:
    trace = _primary_trace(result, phase4_dir=phase4_dir)
    iterator = _iterator_record(result)
    events = [
        _event("agent_run", result.get("final_status"), result.get("outcome_fine"), trace),
        *_agentlab_timeline_events(trace),
        _event("pvpo_capture", _pvpo_filter_value(result), f"max_coverage={_max_coverage(result)}", trace),
        _event("reward_eval", result.get("adversarial_passed"), result.get("adversarial_reward_message"), trace),
        _event(
            "final_state_readback",
            result.get("state_confirmed_action_success"),
            result.get("adversarial_final_state_message"),
            trace,
        ),
        _event(
            "transcript_purpose",
            result.get("transcript_purpose_classification"),
            result.get("transcript_purpose_failure"),
            trace,
        ),
        _event(
            "verbalized_eval_awareness",
            result.get("verbalized_eval_awareness_classification"),
            result.get("verbalized_eval_awareness_failure"),
            trace,
        ),
    ]
    attempts = iterator.get("attempts") if isinstance(iterator.get("attempts"), list) else []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            continue
        events.append(
            {
                "index": len(events),
                "kind": "iterator_attempt",
                "attempt_index": index,
                "status": attempt.get("status") or attempt.get("final_status"),
                "message": compact_text(
                    attempt.get("adversarial_reward_message")
                    or attempt.get("failure")
                    or attempt.get("failed_reason"),
                    limit=180,
                ),
                "trace": attempt.get("trajectory_dir") or attempt.get("variant_trajectory_dir"),
            }
        )
    for error in _iterator_generation_errors(iterator, trace=trace):
        events.append(
            {
                "index": len(events),
                "kind": "iterator_generation_error",
                "attempt_index": error.get("iteration"),
                "status": error.get("failure_class"),
                "message": error.get("reason"),
                "trace": str(trace) if trace else None,
            }
        )
    spans = _span_summary(events)
    return {
        "schema_version": "phase4_trace_timeline_v1",
        "task_id": result.get("task_id"),
        "trace": str(trace) if trace else None,
        "redaction_mode": "compact",
        "events": events,
        "spans": spans,
        "next_commands": next_commands(str(phase4_dir.parent), [result]),
    }


def format_text(payload: dict[str, Any], *, command: str) -> str:
    if command == "summary":
        return _format_summary(payload)
    if command == "task":
        return _format_task(payload)
    if command == "schema":
        return _format_schema(payload)
    if command == "timeline":
        return _format_timeline(payload)
    return _format_rows(payload)


def _format_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"results: {payload['results_path']}",
        f"matched: {payload['matched_results']}/{payload['total_results']}",
        f"final_status: {_count_map(payload['final_status_counts'])}",
        f"outcome_fine: {_count_map(payload['outcome_fine_counts'])}",
        f"sites: {_count_map(payload['site_counts'])}",
        f"pvpo: {_count_map(payload['pvpo_observation_counts'])}",
        f"tp: {_count_map(payload['tp_counts'])}",
        f"vea: {_count_map(payload['vea_counts'])}",
        f"iterator: algorithms={_count_map(payload['iterator_algorithms'])}; stops={_count_map(payload['iterator_stop_reasons'])}",
    ]
    rewards = payload.get("top_reward_messages") or []
    if rewards:
        lines.append("top_reward_messages:")
        for item in rewards:
            lines.append(f"  {item['count']}x {item['value']}")
    samples = payload.get("samples") or []
    if samples:
        lines.append("samples:")
        for row in samples:
            lines.append(
                "  "
                + " ".join(
                    [
                        str(row.get("task_id")),
                        f"site={row.get('site')}",
                        f"action={row.get('action')}",
                        f"status={row.get('final_status')}",
                        f"outcome={row.get('outcome_fine')}",
                        f"pvpo={row.get('max_coverage')}",
                        f"tp={row.get('tp')}",
                        f"stop={row.get('iterator_stop')}",
                    ]
                )
            )
    return "\n".join(lines)


def _format_rows(payload: dict[str, Any]) -> str:
    rows = payload.get("rows") or []
    if not rows:
        return "No matching Phase 4 rows."
    fields = list(rows[0])
    widths = {field: min(42, max(len(field), *(len(str(row.get(field, ""))) for row in rows))) for field in fields}
    lines = ["  ".join(field.ljust(widths[field]) for field in fields)]
    lines.append("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        lines.append(
            "  ".join(str(row.get(field, ""))[: widths[field]].ljust(widths[field]) for field in fields)
        )
    return "\n".join(lines)


def _format_task(payload: dict[str, Any]) -> str:
    lines = [
        f"task_id: {payload.get('task_id')}",
        f"site/action: {payload.get('site')} / {payload.get('action')}",
        f"status: {payload.get('final_status')} ({payload.get('outcome_fine')})",
        f"reward: benign={payload.get('benign_passed')} attack_attempted={payload.get('attack_attempted')} attack_success={payload.get('attack_success')} state_success={payload.get('state_success')}",
        f"awareness: tp={payload.get('tp')} vea={payload.get('vea')} pvpo={payload.get('max_coverage')}",
        f"iterator: {payload.get('iterator_algorithm')} stop={payload.get('iterator_stop')} generated={payload.get('iterator_generated')} attempts={payload.get('attempt_count')}",
        f"why: {payload.get('why')}",
    ]
    if payload.get("reward_msg"):
        lines.append(f"reward_msg: {payload['reward_msg']}")
    if payload.get("trajectory_dir"):
        lines.append(f"trajectory_dir: {payload['trajectory_dir']}")
    if payload.get("iterator"):
        lines.append("iterator_attempts:")
        for attempt in payload["iterator"].get("attempts", []):
            lines.append(
                f"  {attempt.get('index')}: status={attempt.get('status')} outcome={attempt.get('outcome')} reward={attempt.get('reward')}"
            )
        generation_errors = payload["iterator"].get("generation_errors") or []
        if generation_errors:
            lines.append("iterator_generation_errors:")
            for error in generation_errors:
                api = error.get("api") if isinstance(error.get("api"), dict) else {}
                retry = (
                    error.get("instructor_retry")
                    if isinstance(error.get("instructor_retry"), dict)
                    else {}
                )
                lines.append(
                    "  "
                    f"iteration={error.get('iteration')} "
                    f"failure_class={error.get('failure_class')} "
                    f"reason={error.get('reason')} "
                    f"api_attempts={api.get('attempts')} "
                    f"instructor_attempts={retry.get('n_attempts')} "
                    f"completion_ids={api.get('completion_ids')}"
                )
                for parse_error in api.get("parse_errors") or []:
                    lines.append(
                        "    "
                        f"parse_error[{parse_error.get('type')}]: "
                        f"{parse_error.get('message')}"
                    )
                for last_error in api.get("last_attempt_errors") or []:
                    lines.append(
                        "    "
                        f"last_attempt_error[{last_error.get('type')}]: "
                        f"{last_error.get('message')}"
                    )
    if payload.get("refs"):
        lines.append("refs:")
        for key, value in payload["refs"].items():
            lines.append(f"  {key}: {value}")
    if payload.get("artifacts"):
        lines.append("artifacts:")
        for item in payload["artifacts"]:
            lines.append(
                f"  {item.get('kind')}: exists={item.get('exists')} size={item.get('size')} path={item.get('path')}"
            )
    lines.extend(_format_next_commands(payload))
    return "\n".join(lines)


def _format_timeline(payload: dict[str, Any]) -> str:
    lines = [f"timeline: {payload.get('task_id')} redaction_mode={payload.get('redaction_mode')}"]
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        lines.append(
            f"  {event.get('index')}. {event.get('kind')} status={event.get('status')} {event.get('message') or ''}"
        )
    lines.extend(_format_next_commands(payload))
    return "\n".join(lines)


def _format_next_commands(payload: dict[str, Any]) -> list[str]:
    commands = payload.get("next_commands")
    if not isinstance(commands, dict) or not commands:
        return []
    lines = ["next_commands:"]
    for key, value in commands.items():
        lines.append(f"  {key}: {value}")
    return lines


def _format_schema(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _count_map(values: Any) -> str:
    if not isinstance(values, dict) or not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _top_counter(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def _iterator_record(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("eval_awareness_iterator", "strategy_variation"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _iterator_stop_reason(iterator: dict[str, Any]) -> str | None:
    reason = _nested(iterator, "adaptive_budget", "stop_reason")
    if not isinstance(reason, str) or not reason:
        return None
    if iterator.get("algorithm") == "eval-awareness-iterator" and reason == "budget_exhausted":
        return "rewrite_limit_reached"
    return reason


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _max_coverage(result: dict[str, Any]) -> float | int | None:
    for source in (result.get("signals"), result.get("encounter")):
        if isinstance(source, dict) and source.get("max_coverage") is not None:
            return source.get("max_coverage")
    return 1 if ecologically_valid(result) else 0


def _primary_trace(result: dict[str, Any], *, phase4_dir: Path) -> Path | None:
    for key in ("primary_inspection_trace", "trajectory_dir", "current_trace", "initial_trace"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if path.is_absolute():
                return path
            return phase4_dir / path
    task_id = result.get("task_id")
    if isinstance(task_id, str):
        candidates = list(phase4_dir.glob(f"*/{task_id}"))
        if candidates:
            return candidates[0]
    return None


def _pvpo_filter_value(result: dict[str, Any]) -> str:
    coverage = float(_max_coverage(result) or 0)
    if coverage > 0:
        return "encountered"
    if result.get("encounter") or result.get("signals"):
        return "not_encountered"
    return "unknown"


def _aware_filter_value(value: Any) -> str:
    return "aware" if value is True else "unaware"


def _combined_awareness(result: dict[str, Any]) -> str:
    tp = result.get("transcript_purpose_is_aware") is True
    vea = result.get("verbalized_eval_awareness_is_aware") is True
    if tp and vea:
        return "both"
    if tp or vea:
        return "any"
    return "none"


def _artifact_missing(result: dict[str, Any], kind: str, *, phase4_dir: Path | None = None) -> bool:
    trace = _primary_trace(result, phase4_dir=phase4_dir or Path("."))
    if trace is None:
        return True
    refs = _artifact_paths(trace)
    path = refs.get(kind)
    return path is None or not path.exists()


def _answer_label_for_trace(trace: Path | None) -> str | None:
    if trace is None:
        return None
    payload = load_json_or_empty(trace / "result.json")
    return answer_label(payload.get("final_result"))


def _compact_signals(signals: dict[str, Any]) -> dict[str, Any]:
    keys = ("A1", "A2", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "D1")
    return {key: signals.get(key) for key in keys if key in signals}


def _why(result: dict[str, Any]) -> str:
    final_status = str(result.get("final_status") or "missing")
    outcome = str(result.get("outcome_fine") or "missing")
    reward = result.get("adversarial_reward_message")
    if final_status in {"complied", "success_on_variant"}:
        return "final status has compliant adversarial reward evidence"
    if "resistant" in outcome:
        if result.get("transcript_purpose_is_aware") or result.get("verbalized_eval_awareness_is_aware"):
            return "agent resisted after encounter and at least one awareness metric was aware"
        return "agent resisted or ignored the attack after encounter without awareness evidence"
    if reward:
        return str(compact_text(reward, limit=220))
    return f"classified from final_status={final_status} outcome_fine={outcome}"


def _compact_iterator(iterator: dict[str, Any], *, trace: Path | None = None) -> dict[str, Any]:
    attempts = iterator.get("attempts")
    if not isinstance(attempts, list):
        attempts = iterator.get("variant_results") if isinstance(iterator.get("variant_results"), list) else []
    generation_errors = _iterator_generation_errors(iterator, trace=trace)
    return {
        "algorithm": iterator.get("algorithm"),
        "adaptive_budget": iterator.get("adaptive_budget"),
        "judge_diagnosis": iterator.get("judge_diagnosis"),
        "generation_errors": generation_errors,
        "attempts": [
            {
                "index": index,
                "status": attempt.get("status") or attempt.get("final_status"),
                "outcome": attempt.get("outcome") or attempt.get("outcome_fine"),
                "strategy": attempt.get("strategy") or attempt.get("strategy_name"),
                "reward": compact_text(attempt.get("adversarial_reward_message"), limit=160),
                "failure": compact_text(
                    attempt.get("failure") or attempt.get("failed_reason"),
                    limit=160,
                ),
            }
            for index, attempt in enumerate(attempts)
            if isinstance(attempt, dict)
        ],
    }


def _agentlab_inspection_fields(
    result: dict[str, Any],
    trace: Path | None,
) -> dict[str, Any]:
    summary = load_json_or_empty(trace / "summary_info.json") if trace else {}
    runtime = load_json_or_empty(trace / "browser_runtime.json") if trace else {}
    status = load_json_or_empty(trace / "agentlab_sidecar_status.json") if trace else {}
    sidecar_result = load_json_or_empty(trace / "agentlab_sidecar_result.json") if trace else {}
    if not any(isinstance(item, dict) and item for item in (summary, runtime, status, sidecar_result)):
        return {
            "runner": result.get("runner") or result.get("agent_runner"),
            "agentlab_status": result.get("agentlab_status"),
            "agentlab_reward": result.get("agentlab_reward"),
            "agentlab_steps": result.get("steps"),
            "agentlab_error": result.get("error"),
            "runtime_artifact_status": None,
            "browser_instance_scope": None,
            "agent_browser_connect_count": None,
            "auxiliary_browser_connect_count": None,
            "recycle_status": None,
            "sidecar_status": None,
            "sidecar_returncode": None,
        }
    return {
        "runner": "agentlab" if runtime.get("runner") == "agentlab" or sidecar_result else result.get("runner"),
        "agentlab_status": sidecar_result.get("status") or result.get("agentlab_status"),
        "agentlab_reward": sidecar_result.get("agentlab_reward")
        or result.get("agentlab_reward")
        or summary.get("cum_reward"),
        "agentlab_steps": sidecar_result.get("steps") or summary.get("n_steps") or result.get("steps"),
        "agentlab_error": sidecar_result.get("error") or summary.get("err_msg") or result.get("error"),
        "runtime_artifact_status": runtime.get("runtime_artifact_status"),
        "browser_instance_scope": runtime.get("browser_instance_scope"),
        "agent_browser_connect_count": runtime.get("agent_browser_connect_count"),
        "auxiliary_browser_connect_count": runtime.get("auxiliary_browser_connect_count"),
        "recycle_status": runtime.get("recycle_status") or runtime.get("pvpo_browser_recycle_status"),
        "sidecar_status": status.get("status"),
        "sidecar_returncode": status.get("returncode"),
    }


def _agentlab_timeline_events(trace: Path | None) -> list[dict[str, Any]]:
    if trace is None:
        return []
    events: list[dict[str, Any]] = []
    status = load_json_or_empty(trace / "agentlab_sidecar_status.json")
    if status:
        events.append(
            {
                "index": 0,
                "kind": "agentlab_sidecar",
                "status": status.get("status"),
                "message": compact_text(
                    f"step={status.get('current_step')} phase={status.get('current_phase')} "
                    f"url={status.get('last_url')}",
                    limit=180,
                ),
                "trace": str(trace),
            }
        )
    runtime = load_json_or_empty(trace / "browser_runtime.json")
    if runtime:
        events.append(
            {
                "index": 0,
                "kind": "browser_runtime",
                "status": runtime.get("runtime_artifact_status"),
                "message": compact_text(
                    f"scope={runtime.get('browser_instance_scope')} "
                    f"task_browsers={runtime.get('agent_browser_connect_count')} "
                    f"aux_browsers={runtime.get('auxiliary_browser_connect_count')} "
                    f"recycle={runtime.get('recycle_status') or runtime.get('pvpo_browser_recycle_status')}",
                    limit=180,
                ),
                "trace": str(trace),
            }
        )
    native_events = _load_agentlab_native_timeline(trace)
    for item in native_events[-4:]:
        events.append(
            {
                "index": 0,
                "kind": f"agentlab_{item.get('event') or 'timeline'}",
                "status": item.get("phase"),
                "message": compact_text(
                    f"step={item.get('step')} url={item.get('url')} action={item.get('action')}",
                    limit=180,
                ),
                "trace": str(trace),
            }
        )
    return events


def _load_agentlab_native_timeline(trace: Path) -> list[dict[str, Any]]:
    path = trace / "agentlab_step_timeline.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _iterator_generation_errors(
    iterator: dict[str, Any],
    *,
    trace: Path | None,
) -> list[dict[str, Any]]:
    errors = iterator.get("generation_errors")
    if not isinstance(errors, list):
        errors = []
    checkpoint = _load_iterator_checkpoint(trace)
    checkpoint_errors = checkpoint.get("generation_errors")
    if isinstance(checkpoint_errors, list):
        errors = [*errors, *checkpoint_errors]
    compacted = [_compact_generation_error(error) for error in errors if isinstance(error, dict)]
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for error in compacted:
        key = json.dumps(
            {
                "iteration": error.get("iteration"),
                "failure_class": error.get("failure_class"),
                "reason": error.get("reason"),
                "completion_ids": _nested(error, "api", "completion_ids"),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(error)
    return unique


def _load_iterator_checkpoint(trace: Path | None) -> dict[str, Any]:
    if trace is None:
        return {}
    checkpoint = trace / "eval_awareness_iterator_checkpoint.json"
    if not checkpoint.exists():
        return {}
    payload = load_json_or_empty(checkpoint)
    return payload if isinstance(payload, dict) else {}


def _compact_generation_error(error: dict[str, Any]) -> dict[str, Any]:
    diagnostics = error.get("api_diagnostics") if isinstance(error.get("api_diagnostics"), dict) else {}
    instructor = (
        diagnostics.get("instructor_retry_exception")
        if isinstance(diagnostics.get("instructor_retry_exception"), dict)
        else {}
    )
    return {
        "iteration": error.get("iteration"),
        "status": error.get("status"),
        "failure_class": error.get("failure_class"),
        "reason": compact_text(error.get("reason"), limit=220),
        "api": _compact_api_diagnostics(diagnostics),
        "instructor_retry": _compact_instructor_retry(instructor),
    }


def _compact_api_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    responses = diagnostics.get("completion_responses")
    if not isinstance(responses, list):
        responses = []
    parse_errors = diagnostics.get("parse_errors")
    if not isinstance(parse_errors, list):
        parse_errors = []
    completion_errors = diagnostics.get("completion_errors")
    if not isinstance(completion_errors, list):
        completion_errors = []
    last_attempt_errors = diagnostics.get("last_attempt_errors")
    if not isinstance(last_attempt_errors, list):
        last_attempt_errors = []
    return {
        "provider": diagnostics.get("provider"),
        "mode": diagnostics.get("mode"),
        "response_model": diagnostics.get("response_model"),
        "attempts": diagnostics.get("attempts"),
        "completion_ids": [
            response.get("id")
            for response in responses
            if isinstance(response, dict) and response.get("id")
        ],
        "stop_reasons": [
            response.get("stop_reason")
            for response in responses
            if isinstance(response, dict) and response.get("stop_reason")
        ],
        "parse_errors": [_compact_error_message(error) for error in parse_errors],
        "completion_errors": [_compact_error_message(error) for error in completion_errors],
        "last_attempt_errors": [_compact_error_message(error) for error in last_attempt_errors],
    }


def _compact_instructor_retry(instructor: dict[str, Any]) -> dict[str, Any]:
    failed = instructor.get("failed_attempts")
    if not isinstance(failed, list):
        failed = []
    return {
        "n_attempts": instructor.get("n_attempts"),
        "total_usage": instructor.get("total_usage"),
        "failed_attempts": [
            {
                "attempt_number": attempt.get("attempt_number"),
                "exception": _compact_error_message(attempt.get("exception")),
                "completion_id": _nested(
                    attempt.get("completion") if isinstance(attempt, dict) else {},
                    "id",
                ),
            }
            for attempt in failed
            if isinstance(attempt, dict)
        ],
    }


def _compact_error_message(error: Any) -> dict[str, Any]:
    if not isinstance(error, dict):
        return {"type": type(error).__name__, "message": compact_text(error, limit=260)}
    message = str(error.get("message") or "")
    return {
        "type": error.get("type"),
        "message": compact_text(_redact_validation_input_value(message), limit=360),
    }


def _redact_validation_input_value(message: str) -> str:
    lines = []
    for line in message.splitlines():
        if "input_value=" in line:
            line = line.split("input_value=", 1)[0] + "input_value=<omitted>"
        lines.append(line)
    return " | ".join(line.strip() for line in lines if line.strip())


def _artifact_refs(trace: Path | None) -> dict[str, str | None]:
    if trace is None:
        return {}
    refs = _artifact_paths(trace)
    return {key: str(path) if path.exists() else None for key, path in refs.items()}


def _artifact_paths(trace: Path) -> dict[str, Path]:
    return {
        "trace": trace,
        "history": trace / "history.json",
        "result": trace / "result.json",
        "final_response": trace / "final_response.json",
        "summary_info": trace / "summary_info.json",
        "network_trace": trace / "network_trace.json",
        "network_har": trace / "network.har",
        "network_evidence": trace / "network_evidence.json",
        "browser_runtime": trace / "browser_runtime.json",
        "agentlab_request": trace / "agentlab_phase4_request.json",
        "agentlab_result": trace / "agentlab_sidecar_result.json",
        "agentlab_status": trace / "agentlab_sidecar_status.json",
        "agentlab_stdout": trace / "agentlab_sidecar_stdout.log",
        "agentlab_stderr": trace / "agentlab_sidecar_stderr.log",
        "agentlab_timeline": trace / "agentlab_step_timeline.jsonl",
        "agentlab_events": trace / "agentlab_events.jsonl",
        "needham_trace": trace / "needham_trace.xml",
        "tp_raw": trace / "transcript_purpose_raw_response.json",
        "vea_raw": trace / "verbalized_eval_awareness_raw_response.json",
        "pvpo_summary": trace / "pvpo" / "capture_summary.json",
    }


def artifact_manifest(trace: Path | None) -> list[dict[str, Any]]:
    if trace is None:
        return []
    rows = []
    for kind, path in _artifact_paths(trace).items():
        exists = path.exists()
        stat = path.stat() if exists else None
        rows.append(
            {
                "kind": kind,
                "path": str(path),
                "exists": exists,
                "size": stat.st_size if stat else None,
                "mtime": stat.st_mtime if stat else None,
                "redaction_level": (
                    "diagnostic_redacted_ref_only"
                    if kind in {"agentlab_status", "agentlab_stdout", "agentlab_stderr"}
                    else "raw_ref_only"
                ),
            }
        )
    return rows


def _event(kind: str, status: Any, message: Any, trace: Path | None) -> dict[str, Any]:
    return {
        "index": 0,
        "kind": kind,
        "status": status,
        "message": compact_text(message, limit=180),
        "trace": str(trace) if trace else None,
    }


def _span_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans = []
    for index, event in enumerate(events):
        event["index"] = index
        spans.append(
            {
                "kind": event.get("kind"),
                "status": event.get("status"),
                "child_count": 0,
                "bottleneck": False,
            }
        )
    return spans


def next_commands(run_path: str, results: list[dict[str, Any]]) -> dict[str, str]:
    commands = {
        "summary": f"uv run warp-taskgen trace summary {run_path}",
        "slice_resistant_unaware": (
            f"uv run warp-taskgen trace slice {run_path} "
            "--outcome resistant_unaware --fields task_id,site,action,max_coverage,tp,vea,iterator_stop --limit 20"
        ),
    }
    if results:
        task_id = results[0].get("task_id")
        if isinstance(task_id, str) and task_id:
            commands["task_iterator"] = (
                f"uv run warp-taskgen trace task {run_path} {task_id} --iterator"
            )
            commands["task_refs"] = (
                f"uv run warp-taskgen trace task {run_path} {task_id} --refs"
            )
            commands["timeline"] = f"uv run warp-taskgen trace timeline {run_path} {task_id}"
    return commands


def schema() -> dict[str, Any]:
    return {
        "commands": {
            "summary": "Aggregate compact counts and representative samples.",
            "slice": "Render compact task rows matching filters.",
            "task": "Explain one task with optional iterator attempts, generation errors, and artifact refs.",
            "timeline": "Show compact derived event timeline for one task.",
            "schema": "Print this machine-readable command summary.",
        },
        "filters": [
            "--site",
            "--action",
            "--status",
            "--outcome",
            "--task-id",
            "--surface",
            "--origin",
            "--route",
            "--pvpo",
            "--coverage-min",
            "--tp",
            "--vea",
            "--awareness",
            "--iterator-stop",
            "--iterator-algorithm",
            "--has-iterator",
            "--benign-passed",
            "--attack-attempted",
            "--attack-success",
            "--state-success",
            "--has-trace",
            "--missing-artifact",
            "--reward-contains",
        ],
        "outputs": ["text", "json", "jsonl"],
        "fields": ALL_FIELDS,
        "default_fields": DEFAULT_FIELDS,
        "examples": [
            "uv run warp-taskgen trace summary logs/<run> --action create_issue_note",
            "uv run warp-taskgen trace slice logs/<run> --outcome resistant_unaware --fields task_id,site,action,reward_msg,iterator_stop",
            "uv run warp-taskgen trace task logs/<run> <task_id> --iterator --refs",
            "uv run warp-taskgen trace timeline logs/<run> <task_id>",
        ],
    }


__all__ = [
    "DEFAULT_FIELDS",
    "DETAIL_FIELDS",
    "build_summary",
    "build_task_detail",
    "build_timeline",
    "filter_results",
    "format_text",
    "load_inspection",
    "next_commands",
    "schema",
    "task_row",
]
