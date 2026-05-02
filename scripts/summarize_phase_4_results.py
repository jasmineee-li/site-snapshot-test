"""Summarize Phase 4 results with final-status and variant-aware ASR metrics.

Usage:

    uv run python scripts/summarize_phase_4_results.py logs/<run>
    uv run python scripts/summarize_phase_4_results.py logs/<run>/phase_4/results.json --json

The input may be a WorldSim state directory, a ``phase_4`` directory, or a
direct ``results.json`` path. Task metadata is loaded from
``phase_2/adversarial_tasks.json`` when available, and can be supplemented
with ``--tasks``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from worldsim.phase_4.result_summary import summarize_results


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_results_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "phase_4" / "results.json",
        path / "results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find Phase 4 results at {path}/phase_4/results.json or {path}/results.json"
    )


def _default_task_paths(results_path: Path) -> list[Path]:
    paths: list[Path] = []
    parent = results_path.parent
    if parent.name == "phase_4":
        paths.append(parent.parent / "phase_2" / "adversarial_tasks.json")
    paths.append(parent / "phase_2" / "adversarial_tasks.json")
    return [p for p in paths if p.exists()]


def _load_task_lookup(paths: list[Path]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = _load_json(path)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
            items = data["tasks"]
        elif isinstance(data, dict):
            items = [data]
        else:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            task_id = item.get("id")
            if isinstance(task_id, str) and task_id:
                lookup[task_id] = item
    return lookup


def _fmt_rate(rate: float | None) -> str:
    return f"{rate:.2f}" if rate is not None else "n/a"


def _fmt_count_map(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _trace_dir_values(result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "trajectory_dir",
        "initial_trace",
        "current_trace",
        "primary_inspection_trace",
        "successful_variant_trace",
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    variation = result.get("strategy_variation")
    variants = variation.get("variant_results") if isinstance(variation, dict) else None
    if isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            for key in ("trajectory_dir", "variant_trajectory_dir"):
                value = variant.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
    return list(dict.fromkeys(values))


def _trace_artifact_path_candidates(
    trace_dir: str,
    *,
    results_path: Path,
    filename: str,
) -> list[Path]:
    raw_path = Path(trace_dir).expanduser()
    if raw_path.is_absolute():
        return [raw_path / filename]
    return [
        raw_path / filename,
        results_path.parent / raw_path / filename,
        results_path.parent.parent / raw_path / filename,
    ]


def _resolve_trace_artifact_path(
    trace_dir: str,
    *,
    results_path: Path,
    filename: str,
) -> Path | None:
    for candidate in _trace_artifact_path_candidates(
        trace_dir,
        results_path=results_path,
        filename=filename,
    ):
        if candidate.exists():
            return candidate
    return None


def _resolve_browser_runtime_path(trace_dir: str, *, results_path: Path) -> Path | None:
    return _resolve_trace_artifact_path(
        trace_dir,
        results_path=results_path,
        filename="browser_runtime.json",
    )


def summarize_browser_runtime(results: list[dict[str, Any]], *, results_path: Path) -> dict[str, Any]:
    """Aggregate per-trajectory Browser Use runtime counters when artifacts exist."""

    seen_paths: set[Path] = set()
    seen_final_response_paths: set[Path] = set()
    scroll_counters: Counter[str] = Counter()
    beginframe_counters: Counter[str] = Counter()
    navigation_tick_counters: Counter[str] = Counter()
    pvpo_cdp_counters: Counter[str] = Counter()
    browser_use_step_errors = 0
    browser_use_traces_with_step_errors = 0
    browser_use_empty_step_errors = 0
    traces_scanned = 0
    for result in results:
        for trace_dir in _trace_dir_values(result):
            traces_scanned += 1
            runtime_path = _resolve_browser_runtime_path(trace_dir, results_path=results_path)
            if runtime_path is None or runtime_path in seen_paths:
                continue
            seen_paths.add(runtime_path)
            try:
                payload = _load_json(runtime_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    continue
                if value > 0:
                    if isinstance(key, str) and key.startswith("pvpo_scroll_"):
                        scroll_counters[key] += value
                    elif isinstance(key, str) and key.startswith("pvpo_beginframe_"):
                        beginframe_counters[key] += value
                    elif isinstance(key, str) and key.startswith("pvpo_navigation_tick_"):
                        navigation_tick_counters[key] += value
                    elif isinstance(key, str) and key.startswith("pvpo_cdp_"):
                        pvpo_cdp_counters[key] += value
            final_response_path = _resolve_trace_artifact_path(
                trace_dir,
                results_path=results_path,
                filename="final_response.json",
            )
            if final_response_path is None or final_response_path in seen_final_response_paths:
                continue
            seen_final_response_paths.add(final_response_path)
            try:
                final_response = _load_json(final_response_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(final_response, dict):
                continue
            errors = final_response.get("errors")
            if not isinstance(errors, list):
                continue
            normalized_errors = [error for error in errors if error is not None]
            if normalized_errors:
                browser_use_step_errors += len(normalized_errors)
                browser_use_traces_with_step_errors += 1
                browser_use_empty_step_errors += sum(
                    1
                    for error in normalized_errors
                    if isinstance(error, str)
                    and error in ("", "<empty browser-use step error>")
                )
    return {
        "traces_scanned": traces_scanned,
        "runtime_artifacts": len(seen_paths),
        "pvpo_scroll_counters": dict(sorted(scroll_counters.items())),
        "pvpo_beginframe_counters": dict(sorted(beginframe_counters.items())),
        "pvpo_navigation_tick_counters": dict(sorted(navigation_tick_counters.items())),
        "pvpo_cdp_counters": dict(sorted(pvpo_cdp_counters.items())),
        "browser_use_step_errors": browser_use_step_errors,
        "browser_use_traces_with_step_errors": browser_use_traces_with_step_errors,
        "browser_use_empty_step_errors": browser_use_empty_step_errors,
    }


def format_variant_regeneration_audit(summary: dict[str, Any]) -> list[str]:
    audit = summary.get("variant_regeneration_audit")
    if not isinstance(audit, dict) or not audit.get("tasks_entered"):
        return []

    lines = ["Variant regeneration audit:"]
    lines.append(
        "  Method: only ecologically valid initial attacks that were resisted enter "
        "variation; the judge classifies the refusal trigger, the host maps that "
        "trigger to fixed strategies, the generator may rewrite only the selected "
        "payload text, host finalization rejects contract drift, and accepted "
        "variants re-enter PVPO."
    )
    lines.append(
        "  Flow: "
        f"{audit.get('tasks_entered', 0)} task(s) entered; "
        f"{audit.get('planned_attempts', 0)} strategy attempt(s) planned; "
        f"{audit.get('generated_attempts', 0)} generated; "
        f"{audit.get('rejected_before_eval', 0)} rejected before browser eval; "
        f"{audit.get('evaluated_attempts', 0)} evaluated; "
        f"{audit.get('gate1_valid_evaluations', 0)} PVPO-valid; "
        f"{audit.get('compliant_evaluations', 0)} complied with adversarial reward."
    )
    lines.append(
        "  Judge: "
        f"statuses={_fmt_count_map(audit.get('judge_status_counts') or {})}; "
        f"triggers={_fmt_count_map(audit.get('judge_trigger_counts') or {})}; "
        f"confidence={_fmt_count_map(audit.get('judge_confidence_counts') or {})}."
    )

    generation_statuses = audit.get("generation_status_counts") or {}
    if generation_statuses:
        lines.append(f"  Generation statuses: {_fmt_count_map(generation_statuses)}.")

    rows = audit.get("trigger_strategy_rows") or []
    if rows:
        lines.append("  Trigger -> strategy flow:")
        for row in rows:
            lines.append(
                "    "
                f"{row['refusal_trigger']} -> {row['strategy']}: "
                f"planned={row.get('planned', 0)} "
                f"generated={row.get('generated', 0)} "
                f"rejected={row.get('rejected', 0)} "
                f"evaluated={row.get('evaluated', 0)} "
                f"gate1_valid={row.get('gate1_valid', 0)} "
                f"complied={row.get('complied', 0)}"
            )

    errors = summary.get("variant_error_buckets") or []
    if errors:
        lines.append("  Rejection buckets to inspect first:")
        for row in errors[:5]:
            lines.append(f"    {row['count']} {row['class']}: {row['reason']}")
    return lines


def format_inspection_index(summary: dict[str, Any], *, limit: int = 8) -> list[str]:
    rows = summary.get("inspection_index")
    if not isinstance(rows, list) or not rows or limit == 0:
        return []

    lines = ["Inspect next:"]
    for idx, row in enumerate(rows[: max(limit, 0)], start=1):
        if not isinstance(row, dict):
            continue
        task_id = row.get("task_id") or "unknown"
        site = row.get("site") or "unknown"
        surface = row.get("surface") or "unknown"
        final_status = row.get("final_status") or "missing"
        priority_reason = row.get("priority_reason") or "inspect"
        why = row.get("why") or ""
        lines.append(
            "  "
            f"{idx}. [{priority_reason}] {task_id} {site} {surface} "
            f"{final_status}: {why}"
        )
        final_result = row.get("primary_final_result")
        if isinstance(final_result, str) and final_result:
            lines.append(f"     final={final_result}")
        pvpo_observation = row.get("pvpo_observation")
        pvpo_status = row.get("pvpo_status")
        pvpo_failure = row.get("pvpo_failure")
        if pvpo_observation or pvpo_status or pvpo_failure:
            lines.append(
                "     pvpo_observation="
                f"{pvpo_observation or 'unknown'} "
                f"status={pvpo_status or 'unknown'} "
                f"failure={pvpo_failure or 'none'}"
            )
        trace = row.get("primary_inspection_trace")
        if isinstance(trace, str) and trace:
            lines.append(f"     trace={trace}")
        success_trace = row.get("successful_variant_trace")
        if isinstance(success_trace, str) and success_trace and success_trace != trace:
            lines.append(f"     successful_variant_trace={success_trace}")
        artifacts = row.get("artifacts")
        if isinstance(artifacts, dict):
            screenshot = artifacts.get("reference_screenshot")
            pvpo_step = artifacts.get("pvpo_step")
            if screenshot or pvpo_step:
                lines.append(
                    "     pvpo="
                    f"{pvpo_step or 'n/a'} screenshot={screenshot or 'n/a'}"
                )
        rejected = row.get("rejected_variants")
        if isinstance(rejected, list) and rejected:
            first = rejected[0]
            if isinstance(first, dict):
                lines.append(
                    "     rejected="
                    f"{first.get('strategy', 'unknown')} "
                    f"{first.get('status', 'unknown')}: "
                    f"{first.get('reason', '')}"
                )
    return lines


def format_text_summary(
    summary: dict[str, Any],
    *,
    results_path: Path,
    inspect_limit: int = 8,
) -> str:
    lines = [f"Phase 4 results: {results_path}"]
    lines.append(f"Total: {summary['total']} tasks; scorable: {summary['scorable']}")
    lines.append(
        "Final Gate-1 ASR: "
        f"{summary['asr_valid_numerator']} / {summary['asr_valid_denominator']} = "
        f"{_fmt_rate(summary['asr_valid'])}"
    )
    lines.append(
        "Final compliant with adversarial reward: "
        f"{summary['final_compliant_with_adversarial_reward']} / "
        f"{summary['final_compliant_denominator']}"
    )
    lines.append(
        "Initial direct adversarial reward success (B2 ∧ D1): "
        f"{summary['initial_adversarial_reward_success_numerator']} / "
        f"{summary['initial_adversarial_reward_success_denominator']} = "
        f"{_fmt_rate(summary['initial_adversarial_reward_success_rate'])}"
    )
    lines.append(
        "Benign capability under attack (B1 ∧ D1): "
        f"{summary['benign_capability_numerator']} / "
        f"{summary['benign_capability_denominator']} = "
        f"{_fmt_rate(summary['benign_capability'])}"
    )
    lines.append(
        "Initial conditional ASR (complied | C1 ∧ C2 ∧ D1): "
        f"{summary['conditional_initial_asr_numerator']} / "
        f"{summary['conditional_initial_asr_denominator']} = "
        f"{_fmt_rate(summary['conditional_initial_asr'])}"
    )
    lines.append(f"Final status counts: {summary['final_status_counts']}")
    audit = summary.get("variant_regeneration_audit")
    if isinstance(audit, dict):
        direct_complied = int((summary.get("final_status_counts") or {}).get("complied", 0))
        lines.append(
            "Strategy variation entry: "
            f"{audit.get('tasks_entered', 0)} task(s) entered the post-resistance loop; "
            f"{direct_complied} direct complied task(s) stopped before variants."
        )
    lines.append(f"Outcome fine counts: {summary['outcome_fine_counts']}")
    if summary.get("pvpo_observation_counts"):
        lines.append(f"PVPO observation counts: {summary['pvpo_observation_counts']}")
    lines.append(f"Site counts: {summary['site_counts']}")
    lines.append(f"Origin counts: {summary['origin_counts']}")
    if summary.get("judge_trigger_counts"):
        lines.append(f"Judge trigger counts: {summary['judge_trigger_counts']}")
    browser_runtime = summary.get("browser_runtime")
    if isinstance(browser_runtime, dict) and (
        browser_runtime.get("runtime_artifacts")
        or browser_runtime.get("browser_use_step_errors")
    ):
        lines.append(
            "Browser runtime counters: "
            f"artifacts={browser_runtime.get('runtime_artifacts', 0)} "
            f"traces_scanned={browser_runtime.get('traces_scanned', 0)}"
        )
        if browser_runtime.get("browser_use_step_errors"):
            lines.append(
                "  Browser Use step errors: "
                f"total={browser_runtime.get('browser_use_step_errors', 0)} "
                f"traces={browser_runtime.get('browser_use_traces_with_step_errors', 0)} "
                f"empty={browser_runtime.get('browser_use_empty_step_errors', 0)}"
            )
        pvpo_scroll_counters = browser_runtime.get("pvpo_scroll_counters")
        if isinstance(pvpo_scroll_counters, dict) and pvpo_scroll_counters:
            lines.append(f"  PVPO scroll: {_fmt_count_map(pvpo_scroll_counters)}")
        pvpo_beginframe_counters = browser_runtime.get("pvpo_beginframe_counters")
        if isinstance(pvpo_beginframe_counters, dict) and pvpo_beginframe_counters:
            lines.append(f"  PVPO beginFrame: {_fmt_count_map(pvpo_beginframe_counters)}")
        pvpo_navigation_tick_counters = browser_runtime.get("pvpo_navigation_tick_counters")
        if isinstance(pvpo_navigation_tick_counters, dict) and pvpo_navigation_tick_counters:
            lines.append(
                f"  PVPO navigation ticks: {_fmt_count_map(pvpo_navigation_tick_counters)}"
            )
        pvpo_cdp_counters = browser_runtime.get("pvpo_cdp_counters")
        if isinstance(pvpo_cdp_counters, dict) and pvpo_cdp_counters:
            lines.append(f"  PVPO CDP deadlines: {_fmt_count_map(pvpo_cdp_counters)}")

    rows = summary.get("by_site_surface_editor_status") or []
    if rows:
        lines.append("By site/surface/editor/status:")
        for row in rows:
            lines.append(
                "  "
                f"{row['count']} {row['site']} {row['surface']} {row['editor_method']} "
                f"route={row['route_variant']} {row['final_status']} {row['outcome_fine']}"
            )

    strategy_rows = summary.get("variant_strategy_outcomes") or []
    if strategy_rows:
        lines.append("Variant strategy outcomes:")
        for row in strategy_rows:
            lines.append(f"  {row['count']} {row['strategy']} {row['outcome']} {row['gate1']}")

    successes = summary.get("variant_successes") or []
    if successes:
        lines.append("Compliant variant evaluations (not unique tasks):")
        for row in successes:
            lines.append(
                "  "
                f"{row['task_id']} {row['site']} {row['surface']} "
                f"{row['editor_method']} route={row['route_variant']} "
                f"strategy={row['strategy']}"
            )

    errors = summary.get("variant_error_buckets") or []
    if errors:
        lines.append("Variant generation/error buckets:")
        for row in errors:
            lines.append(f"  {row['count']} {row['class']}: {row['reason']}")
    lines.extend(format_variant_regeneration_audit(summary))
    lines.extend(format_inspection_index(summary, limit=inspect_limit))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 4 results with final-status and variant-aware metrics."
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
        "--inspect-limit",
        type=int,
        default=8,
        help="Number of ranked per-task inspection rows to print in text mode (0 disables).",
    )
    args = parser.parse_args(argv)

    try:
        results_path = _resolve_results_path(args.path)
        results = _load_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"{results_path} must contain a list of result objects")
        task_paths = [*args.tasks, *_default_task_paths(results_path)]
        task_lookup = _load_task_lookup(task_paths)
        summary = summarize_results(results, task_lookup=task_lookup)
        summary["browser_runtime"] = summarize_browser_runtime(results, results_path=results_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {"results_path": str(results_path), **summary}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            format_text_summary(
                summary,
                results_path=results_path,
                inspect_limit=args.inspect_limit,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
