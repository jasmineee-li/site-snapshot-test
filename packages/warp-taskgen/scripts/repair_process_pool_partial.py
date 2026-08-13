#!/usr/bin/env python3
"""Repair a failed process-pool Phase 4 run with successful one-task retries.

This intentionally writes a new run directory. It does not mutate the failed
partial run or the retry run in place.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from warp_taskgen.atomic_io import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge successful one-task retry results into a process-pool partial run."
    )
    parser.add_argument("--partial-run", required=True, type=Path)
    parser.add_argument("--retry-run", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    repair_process_pool_partial(
        partial_run=args.partial_run,
        retry_runs=args.retry_run,
        out_dir=args.out_dir,
    )
    return 0


def repair_process_pool_partial(
    *,
    partial_run: Path,
    retry_runs: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    partial_run = partial_run.resolve()
    retry_runs = [path.resolve() for path in retry_runs]
    out_dir = out_dir.resolve()

    if out_dir.exists():
        raise SystemExit(f"out dir already exists: {out_dir}")
    _require_file(partial_run / "phase_4" / "results.partial.json")
    partial_manifest_path = _require_file(partial_run / "phase_4" / "partial_manifest.json")
    for retry_run in retry_runs:
        _require_file(retry_run / "phase_4" / "results.json")

    shutil.copytree(partial_run, out_dir, symlinks=True)

    phase4_dir = out_dir / "phase_4"
    partial_results = _read_json(phase4_dir / "results.partial.json")
    partial_manifest = _read_json(partial_manifest_path)
    if not isinstance(partial_results, list):
        raise SystemExit("partial results must be a JSON list")
    if not isinstance(partial_manifest, dict):
        raise SystemExit("partial manifest must be a JSON object")

    results_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for result in partial_results:
        if not isinstance(result, dict):
            raise SystemExit("partial result rows must be JSON objects")
        task_id = str(result.get("task_id") or "")
        if not task_id:
            raise SystemExit("partial result row missing task_id")
        if task_id in results_by_id:
            raise SystemExit(f"duplicate partial task_id: {task_id}")
        results_by_id[task_id] = result
        order.append(task_id)

    replacements: dict[str, dict[str, Any]] = {}
    replacement_sources: dict[str, Path] = {}
    for retry_run in retry_runs:
        retry_results = _read_json(retry_run / "phase_4" / "results.json")
        if not isinstance(retry_results, list) or len(retry_results) != 1:
            raise SystemExit(f"retry run must contain exactly one result: {retry_run}")
        retry_result = retry_results[0]
        if not isinstance(retry_result, dict):
            raise SystemExit(f"retry result must be a JSON object: {retry_run}")
        task_id = str(retry_result.get("task_id") or "")
        if not task_id:
            raise SystemExit(f"retry result missing task_id: {retry_run}")
        if task_id not in results_by_id:
            raise SystemExit(f"retry task_id {task_id!r} not present in partial results")
        if task_id in replacements:
            raise SystemExit(f"duplicate retry task_id: {task_id}")
        replacements[task_id] = retry_result
        replacement_sources[task_id] = retry_run

    copied_artifacts: dict[str, list[dict[str, str]]] = {}
    task_root = phase4_dir / "process_pool_tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    for task_id, retry_result in replacements.items():
        retry_run = replacement_sources[task_id]
        copied, path_replacements = _copy_retry_task_artifacts(
            retry_run=retry_run,
            task_id=task_id,
            task_root=task_root,
        )
        replacement = _rewrite_string_values(retry_result, path_replacements)
        replacement["process_pool_repair"] = {
            "source_retry_run": str(retry_run),
            "source_partial_run": str(partial_run),
            "repair_reason": "successful_one_task_retry_replaced_failed_worker_result",
        }
        results_by_id[task_id] = replacement
        copied_artifacts[task_id] = copied

    final_results = [results_by_id[task_id] for task_id in order]
    write_json_atomic(phase4_dir / "results.json", final_results)

    repair_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_partial_run": str(partial_run),
        "source_partial_manifest": str(partial_manifest_path),
        "retry_runs": [str(path) for path in retry_runs],
        "out_dir": str(out_dir),
        "expected_tasks": partial_manifest.get("expected_tasks"),
        "result_count": len(final_results),
        "replaced_task_ids": sorted(replacements),
        "source_errors": partial_manifest.get("errors") or [],
        "copied_artifacts": copied_artifacts,
        "canonical_results_written": True,
        "paper_eligible": "operator_review_required",
    }
    write_json_atomic(phase4_dir / "process_pool_repair_manifest.json", repair_manifest)

    progress_path = phase4_dir / "progress.json"
    if progress_path.exists():
        progress = _read_json(progress_path)
        if isinstance(progress, dict):
            progress.update(
                {
                    "status": "complete",
                    "stage": "complete_repaired",
                    "results_path": str(phase4_dir / "results.json"),
                    "postprocess_failed_tasks": 0,
                    "process_pool_repaired": True,
                    "process_pool_repair_manifest": str(
                        phase4_dir / "process_pool_repair_manifest.json"
                    ),
                    "updated_at": repair_manifest["created_at"],
                }
            )
            write_json_atomic(progress_path, progress)

    return repair_manifest


def _copy_retry_task_artifacts(
    *,
    retry_run: Path,
    task_id: str,
    task_root: Path,
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    copied: list[dict[str, str]] = []
    replacements: list[tuple[str, str]] = []
    phase4_dir = retry_run / "phase_4"
    for run_child in phase4_dir.iterdir() if phase4_dir.exists() else []:
        if not run_child.is_dir():
            continue
        for artifact_dir in run_child.iterdir():
            if not artifact_dir.is_dir() or not artifact_dir.name.startswith(task_id):
                continue
            target = _unique_target(task_root, f"retry__{artifact_dir.name}")
            shutil.copytree(artifact_dir, target, symlinks=True)
            copied.append({"source": str(artifact_dir), "target": str(target)})
            replacements.extend(
                [
                    (str(artifact_dir), str(target)),
                    (str(artifact_dir.resolve()), str(target.resolve())),
                ]
            )
    if not copied:
        raise SystemExit(f"retry run has no task artifacts for {task_id}: {retry_run}")
    return copied, replacements


def _unique_target(parent: Path, name: str) -> Path:
    target = parent / name
    if not target.exists():
        return target
    suffix = 1
    while True:
        candidate = parent / f"{name}__{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _rewrite_string_values(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        rewritten = value
        for source, target in replacements:
            if rewritten.startswith(source):
                rewritten = target + rewritten[len(source) :]
        return rewritten
    if isinstance(value, list):
        return [_rewrite_string_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_string_values(item, replacements) for key, item in value.items()}
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
