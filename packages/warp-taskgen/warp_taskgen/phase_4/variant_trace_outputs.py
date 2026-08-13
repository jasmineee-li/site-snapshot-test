"""Output renderers for Phase 4 variant trace exports."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from warp_taskgen.phase_4.variant_trace_text import compact_text


def write_variant_trace_json(export: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_variant_trace_csv(export: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "result_index",
        "task_id",
        "final_status",
        "outcome_fine",
        "site",
        "surface",
        "instruction",
        "base_seeded_ipi_present",
        "post_resistance_variants_entered",
        "variant_loop_reason",
        "initial_answer_label",
        "initial_agentlab_timeline_path",
        "initial_agentlab_timeline_event_count",
        "initial_agentlab_last_event",
        "initial_agentlab_last_step",
        "initial_agentlab_last_network_event_count",
        "initial_agentlab_last_network_delta_count",
        "final_action_attempted",
        "final_action_attempt_status",
        "final_action_attempt_count",
        "final_action_attempt_last_status",
        "final_action_attempt_failed_reason",
        "final_attack_action_success",
        "final_state_confirmed_action_success",
        "variant_index",
        "global_variant_index",
        "round_index",
        "round_variant_index",
        "parent_global_variant_index",
        "root_attempt_id",
        "parent_attempt_id",
        "strategy",
        "generation_status",
        "host_status",
        "contract_qa_status",
        "original_chars",
        "revised_chars",
        "char_delta",
        "attack_witness_offset",
        "variant_outcome",
        "variant_answer_label",
        "variant_agentlab_timeline_path",
        "variant_agentlab_timeline_event_count",
        "variant_agentlab_last_event",
        "variant_agentlab_last_step",
        "variant_agentlab_last_network_event_count",
        "variant_agentlab_last_network_delta_count",
        "variant_action_attempted",
        "variant_action_attempt_status",
        "variant_action_attempt_count",
        "variant_action_attempt_last_status",
        "variant_action_attempt_failed_reason",
        "variant_attack_action_success",
        "variant_state_confirmed_action_success",
        "worked",
        "selected_success",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in export.get("rows", []):
            variants = row.get("variants") or [None]
            for variant in variants:
                writer.writerow(_csv_row(row, variant))


def write_variant_trace_html(
    export: dict[str, Any], path: Path, *, title: str | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_title = title or "Phase 4 Variant Trace Table"
    path.write_text(_render_html(export, title=page_title), encoding="utf-8")


def _csv_row(row: dict[str, Any], variant: dict[str, Any] | None) -> dict[str, Any]:
    generation = variant.get("generation", {}) if isinstance(variant, dict) else {}
    delta = variant.get("delta", {}) if isinstance(variant, dict) else {}
    evaluation = variant.get("evaluation", {}) if isinstance(variant, dict) else {}
    final_action = row.get("action_metrics")
    final_action = final_action if isinstance(final_action, dict) else {}
    variant_action = variant.get("action_metrics", {}) if isinstance(variant, dict) else {}
    initial_timeline = _nested_get(row, "initial", "agentlab_timeline")
    initial_timeline = initial_timeline if isinstance(initial_timeline, dict) else {}
    variant_timeline = _nested_get(variant, "evaluation", "agentlab_timeline")
    variant_timeline = variant_timeline if isinstance(variant_timeline, dict) else {}
    warnings = [
        *row.get("warnings", []),
        *(variant.get("warnings", []) if isinstance(variant, dict) else []),
    ]
    variant_loop = row.get("variant_loop")
    variant_loop = variant_loop if isinstance(variant_loop, dict) else {}
    return {
        "result_index": row.get("result_index"),
        "task_id": row.get("task_id"),
        "final_status": row.get("final_status"),
        "outcome_fine": row.get("outcome_fine"),
        "site": row.get("site"),
        "surface": row.get("surface"),
        "instruction": row.get("instruction"),
        "base_seeded_ipi_present": variant_loop.get("base_seeded_ipi_present"),
        "post_resistance_variants_entered": variant_loop.get("post_resistance_variants_entered"),
        "variant_loop_reason": variant_loop.get("reason"),
        "initial_answer_label": _nested_get(row, "initial", "answer_label"),
        "initial_agentlab_timeline_path": initial_timeline.get("path"),
        "initial_agentlab_timeline_event_count": initial_timeline.get("event_count"),
        "initial_agentlab_last_event": initial_timeline.get("last_event"),
        "initial_agentlab_last_step": initial_timeline.get("last_step"),
        "initial_agentlab_last_network_event_count": initial_timeline.get(
            "last_network_event_count"
        ),
        "initial_agentlab_last_network_delta_count": initial_timeline.get(
            "last_network_delta_count"
        ),
        "final_action_attempted": final_action.get("attempted"),
        "final_action_attempt_status": final_action.get("attempt_status"),
        "final_action_attempt_count": final_action.get("attempt_count"),
        "final_action_attempt_last_status": final_action.get("attempt_last_status"),
        "final_action_attempt_failed_reason": final_action.get("attempt_failed_reason"),
        "final_attack_action_success": final_action.get("attack_action_success"),
        "final_state_confirmed_action_success": final_action.get("state_confirmed_action_success"),
        "variant_index": variant.get("variant_index") if isinstance(variant, dict) else None,
        "global_variant_index": (
            variant.get("global_variant_index") if isinstance(variant, dict) else None
        ),
        "round_index": variant.get("round_index") if isinstance(variant, dict) else None,
        "round_variant_index": (
            variant.get("round_variant_index") if isinstance(variant, dict) else None
        ),
        "parent_global_variant_index": (
            variant.get("parent_global_variant_index") if isinstance(variant, dict) else None
        ),
        "root_attempt_id": variant.get("root_attempt_id") if isinstance(variant, dict) else None,
        "parent_attempt_id": (
            variant.get("parent_attempt_id") if isinstance(variant, dict) else None
        ),
        "strategy": variant.get("strategy") if isinstance(variant, dict) else None,
        "generation_status": generation.get("status"),
        "host_status": generation.get("host_status"),
        "contract_qa_status": generation.get("contract_qa_status"),
        "original_chars": delta.get("original_chars"),
        "revised_chars": delta.get("revised_chars"),
        "char_delta": delta.get("char_delta"),
        "attack_witness_offset": delta.get("attack_witness_offset"),
        "variant_outcome": evaluation.get("outcome"),
        "variant_answer_label": evaluation.get("answer_label"),
        "variant_agentlab_timeline_path": variant_timeline.get("path"),
        "variant_agentlab_timeline_event_count": variant_timeline.get("event_count"),
        "variant_agentlab_last_event": variant_timeline.get("last_event"),
        "variant_agentlab_last_step": variant_timeline.get("last_step"),
        "variant_agentlab_last_network_event_count": variant_timeline.get(
            "last_network_event_count"
        ),
        "variant_agentlab_last_network_delta_count": variant_timeline.get(
            "last_network_delta_count"
        ),
        "variant_action_attempted": variant_action.get("attempted"),
        "variant_action_attempt_status": variant_action.get("attempt_status"),
        "variant_action_attempt_count": variant_action.get("attempt_count"),
        "variant_action_attempt_last_status": variant_action.get("attempt_last_status"),
        "variant_action_attempt_failed_reason": variant_action.get("attempt_failed_reason"),
        "variant_attack_action_success": variant_action.get("attack_action_success"),
        "variant_state_confirmed_action_success": variant_action.get(
            "state_confirmed_action_success"
        ),
        "worked": evaluation.get("worked"),
        "selected_success": variant.get("selected_success") if isinstance(variant, dict) else None,
        "warnings": ";".join(warnings),
    }


def _render_html(export: dict[str, Any], *, title: str) -> str:
    rows = "\n".join(_render_task_row(row) for row in export.get("rows", []))
    escaped_title = html.escape(title)
    export_rows = list(export.get("rows", []))
    entered_count = sum(1 for row in export_rows if _variant_loop_entered(row))
    direct_stop_count = sum(
        1
        for row in export_rows
        if _nested_get(row, "variant_loop", "reason") == "stopped_after_initial_compliance"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; background: #f7f8fa; color: #1d2430; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1680px; margin: 0 auto; padding: 32px 28px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .subtitle {{ color: #647083; margin: 0 0 20px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
    .pill {{ background: #fff; border: 1px solid #d9dee7; border-radius: 999px; padding: 6px 10px; font-size: 13px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #d9dee7; border-radius: 8px; background: #fff; }}
    table {{ width: 100%; min-width: 1500px; border-collapse: separate; border-spacing: 0; }}
    th, td {{ vertical-align: top; text-align: left; border-right: 1px solid #d9dee7; border-bottom: 1px solid #d9dee7; padding: 11px; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #eef2f7; color: #334155; text-transform: uppercase; font-size: 12px; letter-spacing: .04em; }}
    th:last-child, td:last-child {{ border-right: 0; }}
    tr:last-child td {{ border-bottom: 0; }}
    code, .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; }}
    .answer {{ display: inline-block; margin: 0 0 6px; border-radius: 6px; padding: 2px 7px; background: #f1f5f9; }}
    .variant {{ min-width: 260px; }}
    .worked {{ background: #edf7f5; border-left: 4px solid #0f766e; }}
    .missed {{ background: #fff7ed; border-left: 4px solid #9a3412; }}
    .none {{ background: #f8fafc; }}
    .muted {{ color: #647083; }}
    .payload {{ color: #334155; }}
    .delta {{ color: #647083; margin-top: 8px; }}
    .warnings {{ color: #9a3412; margin-top: 8px; }}
    .stage {{ margin-top: 8px; color: #475569; font-size: 12px; }}
  </style>
</head>
<body>
  <main>
    <h1>{escaped_title}</h1>
    <p class="subtitle"><code>{html.escape(str(export.get("results_path") or ""))}</code></p>
    <section class="summary">
      <span class="pill">Rows: {len(export.get("rows", []))}</span>
      <span class="pill">Total results: {html.escape(str(export.get("total_results")))}</span>
      <span class="pill">Warnings: {html.escape(str(export.get("warning_count")))}</span>
      <span class="pill">Base seeded IPI shown for every row</span>
      <span class="pill">Post-resistance variant loop entered: {entered_count}</span>
      <span class="pill">Stopped after direct compliance: {direct_stop_count}</span>
    </section>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Task</th>
            <th>Base Seeded IPI / Initial Response</th>
            <th>Post-Resistance Variants</th>
            <th>Warnings</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""


def _render_task_row(row: dict[str, Any]) -> str:
    variants = list(row.get("variants") or [])
    variant_cell = (
        "<td>" + "".join(_render_variant_card(variant) for variant in variants) + "</td>"
        if variants
        else '<td class="variant none muted">No post-resistance variant executed</td>'
    )
    warnings = "<br>".join(html.escape(warning) for warning in row.get("warnings", []))
    loop_note = _variant_loop_note(row)
    action_note = _action_note(row.get("action_metrics"))
    agentlab_note = _agentlab_timeline_note(_nested_get(row, "initial", "agentlab_timeline"))
    return f"""
          <tr>
            <td class="mono">{html.escape(str(row.get("result_index")))}</td>
            <td><div class="mono">{html.escape(str(row.get("task_id")))}</div><div>{html.escape(str(row.get("instruction") or ""))}</div><div class="muted">{html.escape(str(row.get("final_status") or ""))} / {html.escape(str(row.get("outcome_fine") or ""))}</div><div class="stage">{html.escape(loop_note)}</div><div class="stage">{html.escape(action_note)}</div></td>
            <td><span class="answer mono">{html.escape(str(_nested_get(row, "initial", "answer_label") or ""))}</span><div class="payload">{html.escape(str(compact_text(_nested_get(row, "initial", "final_result"), limit=220) or ""))}</div><div class="stage">{html.escape(agentlab_note)}</div>{_render_agentlab_tail(_nested_get(row, "initial", "agentlab_timeline"))}</td>
            {variant_cell}
            <td class="warnings">{warnings}</td>
          </tr>"""


def _render_variant_card(variant: dict[str, Any]) -> str:
    evaluation = variant.get("evaluation", {})
    delta = variant.get("delta", {})
    generation = variant.get("generation", {})
    action = variant.get("action_metrics")
    agentlab_note = _agentlab_timeline_note(_nested_get(variant, "evaluation", "agentlab_timeline"))
    css = "worked" if evaluation.get("worked") else "missed"
    if variant.get("selected_success"):
        css = "worked"
    warnings = " ".join(variant.get("warnings", []))
    label = (
        f"Variant {variant.get('global_variant_index')} "
        f"(round {variant.get('round_index')}, slot {variant.get('round_variant_index')})"
    )
    lineage = (
        f"parent={variant.get('parent_global_variant_index')} "
        f"root={variant.get('root_attempt_id')} attempt={variant.get('parent_attempt_id')}"
    )
    return f"""<div class="variant {css}">
      <div class="mono">{html.escape(label)} · {html.escape(str(variant.get("strategy") or ""))}</div>
      <div class="muted mono">{html.escape(lineage)}</div>
      <span class="answer mono">{html.escape(str(evaluation.get("answer_label") or ""))}</span>
      <div class="muted">{html.escape(str(evaluation.get("outcome") or ""))} | gen={html.escape(str(generation.get("status") or ""))} host={html.escape(str(generation.get("host_status") or ""))}</div>
      <div class="stage">{html.escape(_action_note(action))}</div>
      <div class="stage">{html.escape(agentlab_note)}</div>
      {_render_agentlab_tail(_nested_get(variant, "evaluation", "agentlab_timeline"))}
      <div class="payload">{html.escape(str(compact_text(delta.get("revised_payload"), limit=260) or ""))}</div>
      <div class="delta">chars {html.escape(str(delta.get("original_chars")))} → {html.escape(str(delta.get("revised_chars")))}; witness {html.escape(str(delta.get("attack_witness_offset")))}/{html.escape(str(delta.get("max_attack_witness_offset")))}</div>
      <div class="warnings">{html.escape(warnings)}</div>
    </div>"""


def _agentlab_timeline_note(timeline: Any) -> str:
    if not isinstance(timeline, dict) or not timeline.get("exists"):
        return "AgentLab timeline: unavailable"
    return (
        "AgentLab timeline: "
        f"events={timeline.get('event_count')} "
        f"last={timeline.get('last_event')} "
        f"step={timeline.get('last_step')} "
        f"network={timeline.get('last_network_event_count')} "
        f"delta={timeline.get('last_network_delta_count')} "
        f"path={timeline.get('path')}"
    )


def _render_agentlab_tail(timeline: Any) -> str:
    if not isinstance(timeline, dict) or not timeline.get("exists"):
        return ""
    rows = timeline.get("tail")
    if not isinstance(rows, list) or not rows:
        return ""
    items = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        items.append(
            '<div class="stage mono">'
            f"{html.escape(str(row.get('event') or ''))} "
            f"step={html.escape(str(row.get('step') or ''))} "
            f"delta={html.escape(str(row.get('network_delta_count') or ''))} "
            f"url={html.escape(str(compact_text(row.get('url'), limit=90) or ''))}"
            "</div>"
        )
    return "".join(items)


def _action_note(action: Any) -> str:
    if not isinstance(action, dict) or not action:
        return "action attempt: unknown"
    attempted = action.get("attempted")
    status = action.get("attempt_status") or "unknown"
    count = action.get("attempt_count")
    last_status = action.get("attempt_last_status")
    success = action.get("attack_action_success")
    state_success = action.get("state_confirmed_action_success")
    reason = action.get("attempt_failed_reason")
    parts = [
        f"action attempt: {attempted}",
        f"status={status}",
        f"count={count}",
        f"http={last_status}",
        f"request_success={success}",
        f"state_success={state_success}",
    ]
    if reason:
        parts.append(f"reason={reason}")
    return "; ".join(parts)


def _nested_get(payload: Any, *keys: str, default: Any = None) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _variant_loop_entered(row: dict[str, Any]) -> bool:
    return bool(_nested_get(row, "variant_loop", "post_resistance_variants_entered"))


def _variant_loop_note(row: dict[str, Any]) -> str:
    reason = str(_nested_get(row, "variant_loop", "reason") or "unknown")
    if reason == "entered_after_initial_refusal_or_ignore":
        return (
            "Post-resistance variant loop entered after the initial attack was resisted or ignored."
        )
    if reason == "stopped_after_initial_compliance":
        return "Stopped after direct compliance; no post-resistance variants were generated."
    if reason == "stopped_before_strategy_variation_no_pvpo_encounter":
        return "Stopped before strategy variation because strict PVPO did not confirm encounter."
    if reason == "stopped_before_strategy_variation_task_broke":
        return "Stopped before strategy variation because the task broke."
    return f"Post-resistance variant loop did not enter: {reason}."


__all__ = [
    "write_variant_trace_csv",
    "write_variant_trace_html",
    "write_variant_trace_json",
]
