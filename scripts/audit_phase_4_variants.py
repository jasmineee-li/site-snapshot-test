"""Audit Phase 4 variant-generation artifacts.

This is an observability tool only: it reports how judge diagnoses, variant
generation attempts, host validation, and PVPO evaluations line up. It does
not alter admission, rewards, or Phase 4 final statuses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldsim.phase_4.artifact_audit import (
    build_variant_artifact_audit,
    default_task_paths,
    load_json,
    load_task_lookup,
    phase4_dir_for_results,
    resolve_phase4_results_path,
)


def _fmt_count_map(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _format_text_report(
    report: dict[str, Any],
    *,
    results_path: Path,
    task_limit: int,
) -> str:
    summary = report["summary"]
    lines = [f"Phase 4 variant QA: {results_path}"]
    lines.append(
        "Method: audit-only artifact reconciliation; accepted variants are still "
        "defined by host finalization plus PVPO/reward evidence in results.json."
    )
    lines.append(
        "Final Gate-1 ASR: "
        f"{summary['asr_valid_numerator']} / {summary['asr_valid_denominator']} = "
        f"{summary['asr_valid']:.2f}"
        if summary["asr_valid"] is not None
        else "Final Gate-1 ASR: n/a"
    )
    lines.append(
        "Variant flow: "
        f"{summary['variant_regeneration_audit'].get('tasks_entered', 0)} task(s) entered; "
        f"{summary['variant_regeneration_audit'].get('planned_attempts', 0)} planned; "
        f"{summary['variant_regeneration_audit'].get('generated_attempts', 0)} generated; "
        f"{summary['variant_regeneration_audit'].get('evaluated_attempts', 0)} evaluated; "
        f"{summary['variant_regeneration_audit'].get('gate1_valid_evaluations', 0)} PVPO-valid; "
        f"{summary['variant_regeneration_audit'].get('compliant_evaluations', 0)} complied."
    )
    lines.append(
        "Artifacts: "
        f"{report['artifact_attempts']} attempt dir(s); "
        f"generation={_fmt_count_map(report['artifact_generation_status_counts'])}; "
        f"host={_fmt_count_map(report['artifact_host_status_counts'])}; "
        f"failure_context={report['attempts_with_failure_context']}; "
        f"payload_diff={report['attempts_with_payload_diff']}; "
        f"contract_qa={report.get('attempts_with_contract_qa', 0)}."
    )
    if report.get("contract_qa_failure_class_counts"):
        lines.append(
            "Contract QA failures: "
            f"{_fmt_count_map(report['contract_qa_failure_class_counts'])}."
        )
    if report["quality_flag_counts"]:
        lines.append(f"Quality flags: {_fmt_count_map(report['quality_flag_counts'])}.")
    else:
        lines.append("Quality flags: none.")
    failure_buckets = report.get("host_failure_buckets") or []
    if failure_buckets:
        lines.append("Terminal host-finalization failure buckets:")
        for row in failure_buckets[:10]:
            lines.append(
                "  "
                f"{row['count']} {row['failure_class']} {row['site']} {row['surface']} "
                f"route={row['route_variant']} strategy={row['strategy']}: "
                f"{row['sample_reason']}"
            )
    repaired_buckets = report.get("repaired_host_failure_buckets") or []
    if repaired_buckets:
        lines.append("Repaired host-feedback buckets:")
        for row in repaired_buckets[:10]:
            lines.append(
                "  "
                f"{row['count']} {row['failure_class']} {row['site']} {row['surface']} "
                f"route={row['route_variant']} strategy={row['strategy']}: "
                f"{row['sample_reason']}"
            )

    rows = report["task_rows"]
    if rows and task_limit != 0:
        lines.append("Per-task variant QA:")
        for row in rows[: max(task_limit, 0)]:
            flags = ", ".join(row["quality_flags"]) if row["quality_flags"] else "none"
            lines.append(
                "  "
                f"{row['task_id']} {row['site']} {row['surface']} "
                f"{row['editor_method']} route={row['route_variant']} "
                f"{row['final_status']} trigger={row['refusal_trigger']} "
                f"generated={row['generated_records']} rejected={row['rejected_records']} "
                f"evaluated={row['evaluated_variants']} pvpo={row['gate1_valid_variants']} "
                f"complied={row['compliant_variants']} artifacts={row['artifact_attempts']} "
                f"host={_fmt_count_map(row['artifact_host_status_counts'])} "
                f"contract_qa={_fmt_count_map(row.get('contract_qa_failure_class_counts', {}))} "
                f"terminal_failures={_fmt_count_map(row['terminal_failure_class_counts'])} "
                f"repaired_failures={_fmt_count_map(row['repaired_failure_class_counts'])} "
                f"flags={flags}"
            )
            terminal_rejection = row.get("first_terminal_rejection")
            repaired_rejection = row.get("first_rejection")
            if isinstance(terminal_rejection, dict):
                lines.append(
                    "     first_terminal_rejection="
                    f"{terminal_rejection.get('strategy')}:{terminal_rejection.get('attempt')} "
                    f"{terminal_rejection.get('generation_status')} "
                    f"host={terminal_rejection.get('host_status')} "
                    "reason="
                    f"{terminal_rejection.get('generation_reason') or terminal_rejection.get('host_reason') or ''}"
                )
            elif isinstance(repaired_rejection, dict):
                lines.append(
                    "     first_repaired_rejection="
                    f"{repaired_rejection.get('strategy')}:{repaired_rejection.get('attempt')} "
                    f"{repaired_rejection.get('generation_status')} "
                    f"host={repaired_rejection.get('host_status')} "
                    "reason="
                    f"{repaired_rejection.get('generation_reason') or repaired_rejection.get('host_reason') or ''}"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Phase 4 variant-generation artifacts against results.json."
    )
    parser.add_argument("path", type=Path, help="State dir, phase_4 dir, or results.json path.")
    parser.add_argument(
        "--tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional adversarial task JSON file(s) used for site/surface/origin breakdowns.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--task-limit",
        type=int,
        default=16,
        help="Number of per-task rows to print in text mode (0 disables).",
    )
    args = parser.parse_args(argv)

    try:
        results_path = resolve_phase4_results_path(args.path)
        results = load_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"{results_path} must contain a list of result objects")
        task_lookup = load_task_lookup([*args.tasks, *default_task_paths(results_path)])
        report = build_variant_artifact_audit(
            results,
            task_lookup=task_lookup,
            phase4_dir=phase4_dir_for_results(results_path),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {"results_path": str(results_path), **report},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_format_text_report(report, results_path=results_path, task_limit=args.task_limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
