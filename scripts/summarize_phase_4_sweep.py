#!/usr/bin/env python3
"""Summarize multiple Phase 4 runs as a model/task-bank sweep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldsim.phase_4.pvpo_observations import pvpo_observation_bucket
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


def _is_json_text(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _fenced_body(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or not lines[0].strip().startswith("```"):
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "```":
            return "\n".join(lines[1:index]).strip()
    return None


def final_result_shape(value: Any) -> str:
    """Classify agent final answers for reporting without changing scoring."""

    if value is None:
        return "missing"
    if isinstance(value, dict | list):
        return "structured_json_value"
    text = str(value).strip()
    if not text:
        return "empty_string"
    if _is_json_text(text):
        return "json_string"
    fenced = _fenced_body(text)
    if fenced is not None and _is_json_text(fenced):
        return "fenced_json_string"
    if "retrieved_data" in text and "{" in text and "}" in text:
        return "prose_with_json_fragment"
    return "plain_text"


def _route_variant(task: dict[str, Any]) -> str | None:
    if isinstance(task.get("route_variant"), str):
        return task["route_variant"]
    exposure = task.get("exposure_contract")
    if not isinstance(exposure, dict):
        return None
    surface_route = exposure.get("surface_route")
    if isinstance(surface_route, dict) and isinstance(surface_route.get("route_variant"), str):
        return surface_route["route_variant"]
    if isinstance(exposure.get("mode"), str):
        return exposure["mode"]
    return None


def _task_metadata(task: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {
            "site": None,
            "origin": None,
            "route_id": None,
            "route_variant": None,
            "target_surface_id": None,
            "editor_method": None,
        }
    exposure = task.get("exposure_contract")
    exposure = exposure if isinstance(exposure, dict) else {}
    return {
        "site": task.get("site"),
        "origin": task.get("origin"),
        "route_id": task.get("route_id"),
        "route_variant": _route_variant(task),
        "target_surface_id": task.get("target_surface_id") or exposure.get("target_surface_id"),
        "editor_method": task.get("editor_method") or exposure.get("editor_method"),
    }


def summarize_sweep(run_dirs: list[Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    paired_task_ids: set[str] | None = None
    by_model_status: dict[str, Counter[str]] = {}
    results_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    task_lookup_merged: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        results_path = _phase4_results_path(run_dir)
        results = _load_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"{results_path} must contain a JSON array")
        task_lookup = _load_task_lookup(run_dir)
        task_lookup_merged.update({key: value for key, value in task_lookup.items() if key not in task_lookup_merged})
        summary = summarize_results(results, task_lookup=task_lookup)
        state = _state_metadata(run_dir)
        final_result_shapes = Counter(final_result_shape(result.get("final_result")) for result in results)
        model_key = "|".join(
            str(state.get(key) or "unknown")
            for key in ("agent_provider", "agent_model", "agent_service_tier", "sandbox_model")
        )
        task_ids = {str(result.get("task_id")) for result in results if result.get("task_id")}
        paired_task_ids = task_ids if paired_task_ids is None else paired_task_ids.intersection(task_ids)
        by_model_status[model_key] = Counter(summary.get("final_status_counts") or {})
        results_by_model[model_key] = {
            str(result.get("task_id")): result
            for result in results
            if isinstance(result, dict) and result.get("task_id")
        }
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
                "final_result_shape_counts": dict(sorted(final_result_shapes.items())),
                "pvpo_observation_counts": summary.get("pvpo_observation_counts") or {},
                "site_counts": summary.get("site_counts") or {},
                "origin_counts": summary.get("origin_counts") or {},
            }
        )
    paired = sorted(paired_task_ids or set())
    task_rows = []
    for task_id in paired:
        row: dict[str, Any] = {
            "task_id": task_id,
            **_task_metadata(task_lookup_merged.get(task_id)),
            "models": {},
        }
        for model_key, model_results in sorted(results_by_model.items()):
            result = model_results.get(task_id) or {}
            row["models"][model_key] = {
                "final_status": result.get("final_status"),
                "outcome_fine": result.get("outcome_fine"),
                "benign_passed": result.get("benign_passed"),
                "adversarial_passed": result.get("adversarial_passed"),
                "max_coverage": (result.get("encounter") or {}).get("max_coverage")
                if isinstance(result.get("encounter"), dict)
                else None,
                "final_result_shape": final_result_shape(result.get("final_result")),
                "pvpo_observation": pvpo_observation_bucket(result),
                "steps": result.get("steps"),
            }
        task_rows.append(row)
    return {
        "run_count": len(runs),
        "paired_task_count": len(paired),
        "runs": runs,
        "by_model_status": {
            model: dict(sorted(counter.items())) for model, counter in sorted(by_model_status.items())
        },
        "task_rows": task_rows,
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
            f"final_result_shapes={run.get('final_result_shape_counts', {})} "
            f"pvpo_observations={run.get('pvpo_observation_counts', {})} "
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
