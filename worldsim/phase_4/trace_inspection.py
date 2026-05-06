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
]


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
    site: str | None = None,
    action: str | None = None,
    status: str | None = None,
    outcome: str | None = None,
    task_id: str | None = None,
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
    row = {
        "task_id": task_id,
        "site": task_site(result, task),
        "surface": task_surface(task),
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
        "iterator_stop": _nested(iterator, "adaptive_budget", "stop_reason"),
        "iterator_generated": _nested(iterator, "adaptive_budget", "generated"),
        "attempt_count": len(iterator.get("attempts") or []),
        "trajectory_dir": str(trajectory_dir) if trajectory_dir else None,
        "answer_label": _answer_label_for_trace(trajectory_dir),
        "signals": _compact_signals(signals),
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
        str(_nested(_iterator_record(result), "adaptive_budget", "stop_reason") or "missing")
        for result in filtered
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
        base["iterator"] = _compact_iterator(_iterator_record(result))
    if include_refs:
        trace = _primary_trace(result, phase4_dir=phase4_dir)
        base["refs"] = _artifact_refs(trace)
    return base


def format_text(payload: dict[str, Any], *, command: str) -> str:
    if command == "summary":
        return _format_summary(payload)
    if command == "task":
        return _format_task(payload)
    if command == "schema":
        return _format_schema(payload)
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
    if payload.get("refs"):
        lines.append("refs:")
        for key, value in payload["refs"].items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


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


def _compact_iterator(iterator: dict[str, Any]) -> dict[str, Any]:
    attempts = iterator.get("attempts")
    if not isinstance(attempts, list):
        attempts = iterator.get("variant_results") if isinstance(iterator.get("variant_results"), list) else []
    return {
        "algorithm": iterator.get("algorithm"),
        "adaptive_budget": iterator.get("adaptive_budget"),
        "judge_diagnosis": iterator.get("judge_diagnosis"),
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


def _artifact_refs(trace: Path | None) -> dict[str, str | None]:
    if trace is None:
        return {}
    refs = {
        "trace": trace,
        "history": trace / "history.json",
        "result": trace / "result.json",
        "final_response": trace / "final_response.json",
        "needham_trace": trace / "needham_trace.xml",
        "tp_raw": trace / "transcript_purpose_raw_response.json",
        "vea_raw": trace / "verbalized_eval_awareness_raw_response.json",
        "pvpo_summary": trace / "pvpo" / "capture_summary.json",
    }
    return {key: str(path) if path.exists() else None for key, path in refs.items()}


def schema() -> dict[str, Any]:
    return {
        "commands": {
            "summary": "Aggregate compact counts and representative samples.",
            "slice": "Render compact task rows matching filters.",
            "task": "Explain one task with optional iterator attempts and artifact refs.",
            "schema": "Print this machine-readable command summary.",
        },
        "filters": ["--site", "--action", "--status", "--outcome", "--task-id"],
        "outputs": ["text", "json"],
        "fields": [*DETAIL_FIELDS, "pvpo", "tp_aware", "vea_aware", "signals"],
        "default_fields": DEFAULT_FIELDS,
    }


__all__ = [
    "DEFAULT_FIELDS",
    "DETAIL_FIELDS",
    "build_summary",
    "build_task_detail",
    "filter_results",
    "format_text",
    "load_inspection",
    "schema",
    "task_row",
]
