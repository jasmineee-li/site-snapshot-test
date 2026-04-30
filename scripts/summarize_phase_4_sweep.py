#!/usr/bin/env python3
"""Summarize multiple Phase 4 runs as a model/task-bank sweep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldsim.phase_4.result_summary import summarize_results


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_task_lookup(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "phase_2" / "adversarial_tasks.json"
    if not path.exists():
        return {}
    data = _load_json(path)
    if not isinstance(data, list):
        return {}
    return {str(task.get("id")): task for task in data if isinstance(task, dict)}


def _phase4_results_path(run_dir: Path) -> Path:
    candidates = [run_dir / "phase_4" / "results.json", run_dir / "results.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no Phase 4 results.json found under {run_dir}")


def _state_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "pipeline_state.json"
    if not path.exists():
        return {}
    data = _load_json(path)
    if not isinstance(data, dict):
        return {}
    return data


def summarize_sweep(run_dirs: list[Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    paired_task_ids: set[str] | None = None
    by_model_status: dict[str, Counter[str]] = {}
    for run_dir in run_dirs:
        results_path = _phase4_results_path(run_dir)
        results = _load_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"{results_path} must contain a JSON array")
        task_lookup = _load_task_lookup(run_dir)
        summary = summarize_results(results, task_lookup=task_lookup)
        state = _state_metadata(run_dir)
        model_key = "|".join(
            str(state.get(key) or "unknown")
            for key in ("agent_provider", "agent_model", "agent_service_tier", "sandbox_model")
        )
        task_ids = {str(result.get("task_id")) for result in results if result.get("task_id")}
        paired_task_ids = task_ids if paired_task_ids is None else paired_task_ids.intersection(task_ids)
        by_model_status[model_key] = Counter(summary.get("final_status_counts") or {})
        runs.append(
            {
                "run_dir": str(run_dir),
                "results_path": str(results_path),
                "model_key": model_key,
                "agent_provider": state.get("agent_provider"),
                "agent_model": state.get("agent_model"),
                "agent_service_tier": state.get("agent_service_tier"),
                "sandbox_model": state.get("sandbox_model"),
                "total": summary.get("total", 0),
                "asr_valid": summary.get("asr_valid"),
                "asr_valid_numerator": summary.get("asr_valid_numerator", 0),
                "asr_valid_denominator": summary.get("asr_valid_denominator", 0),
                "final_status_counts": summary.get("final_status_counts") or {},
                "site_counts": summary.get("site_counts") or {},
                "origin_counts": summary.get("origin_counts") or {},
            }
        )
    return {
        "run_count": len(runs),
        "paired_task_count": len(paired_task_ids or set()),
        "runs": runs,
        "by_model_status": {
            model: dict(sorted(counter.items())) for model, counter in sorted(by_model_status.items())
        },
    }


def format_sweep_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Phase 4 sweep: runs={summary.get('run_count', 0)} paired_tasks={summary.get('paired_task_count', 0)}"
    ]
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        lines.append(
            "  "
            f"{run.get('model_key')}: "
            f"ASR={run.get('asr_valid_numerator', 0)}/{run.get('asr_valid_denominator', 0)} "
            f"statuses={run.get('final_status_counts', {})} "
            f"run={run.get('run_dir')}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Phase 4 run directories.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    summary = summarize_sweep(args.run_dirs)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_sweep_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
