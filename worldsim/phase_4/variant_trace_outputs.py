"""Output renderers for Phase 4 variant trace exports."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from worldsim.phase_4.variant_trace_text import compact_text


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
        "variant_index",
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
        "post_resistance_variants_entered": variant_loop.get(
            "post_resistance_variants_entered"
        ),
        "variant_loop_reason": variant_loop.get("reason"),
        "initial_answer_label": _nested_get(row, "initial", "answer_label"),
        "variant_index": variant.get("variant_index") if isinstance(variant, dict) else None,
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
            <th>Post-Resistance Variant 1</th>
            <th>Post-Resistance Variant 2</th>
            <th>Post-Resistance Variant 3</th>
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
    variant_cells = [_render_variant_cell(variant) for variant in variants[:3]]
    while len(variant_cells) < 3:
        variant_cells.append(
            '<td class="variant none muted">No post-resistance variant executed</td>'
        )
    warnings = "<br>".join(html.escape(warning) for warning in row.get("warnings", []))
    loop_note = _variant_loop_note(row)
    return f"""
          <tr>
            <td class="mono">{html.escape(str(row.get("result_index")))}</td>
            <td><div class="mono">{html.escape(str(row.get("task_id")))}</div><div>{html.escape(str(row.get("instruction") or ""))}</div><div class="muted">{html.escape(str(row.get("final_status") or ""))} / {html.escape(str(row.get("outcome_fine") or ""))}</div><div class="stage">{html.escape(loop_note)}</div></td>
            <td><span class="answer mono">{html.escape(str(_nested_get(row, "initial", "answer_label") or ""))}</span><div class="payload">{html.escape(str(compact_text(_nested_get(row, "initial", "final_result"), limit=220) or ""))}</div></td>
            {"".join(variant_cells)}
            <td class="warnings">{warnings}</td>
          </tr>"""


def _render_variant_cell(variant: dict[str, Any]) -> str:
    evaluation = variant.get("evaluation", {})
    delta = variant.get("delta", {})
    generation = variant.get("generation", {})
    css = "worked" if evaluation.get("worked") else "missed"
    if variant.get("selected_success"):
        css = "worked"
    warnings = " ".join(variant.get("warnings", []))
    return f"""<td class="variant {css}">
      <div class="mono">{html.escape(str(variant.get("strategy") or ""))}</div>
      <span class="answer mono">{html.escape(str(evaluation.get("answer_label") or ""))}</span>
      <div class="muted">{html.escape(str(evaluation.get("outcome") or ""))} | gen={html.escape(str(generation.get("status") or ""))} host={html.escape(str(generation.get("host_status") or ""))}</div>
      <div class="payload">{html.escape(str(compact_text(delta.get("revised_payload"), limit=260) or ""))}</div>
      <div class="delta">chars {html.escape(str(delta.get("original_chars")))} → {html.escape(str(delta.get("revised_chars")))}; witness {html.escape(str(delta.get("attack_witness_offset")))}/{html.escape(str(delta.get("max_attack_witness_offset")))}</div>
      <div class="warnings">{html.escape(warnings)}</div>
    </td>"""


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
        return "Post-resistance variant loop entered after the initial attack was resisted or ignored."
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
