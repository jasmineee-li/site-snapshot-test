"""Rerun errored trajectory-awareness judge rows in-place.

This is intentionally narrower than `run_safety_pipeline --stage judge-only`:
it scans existing `trajectory_awareness_results.jsonl` files, reruns only rows
whose metadata contains an error, and replaces those rows without rewriting the
rest of the split.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from eval_awareness_experiments.experiments.trajectory_awareness import (
    TrajectoryAwarenessExperiment,
)
from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.run_safety_pipeline import _discover_browser
from eval_awareness_experiments.types import WebsiteExperimentResult, WebsiteSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairTarget:
    result_file: Path
    line_index: int
    row: dict
    benchmark: str
    split: str
    arm: str
    model_slug: str
    judge_name: str
    task_id: str
    trajectory_dir: Path


def _row_error(row: dict) -> str | None:
    metadata = row.get("metadata") or {}
    err = metadata.get("error") or row.get("error") or row.get("err_msg")
    return str(err) if err else None


def _task_id_from_sample(sample_id: str, benchmark: str, split: str) -> str:
    prefix = f"{benchmark}_{split}_"
    if sample_id.startswith(prefix):
        return sample_id.removeprefix(prefix)
    return sample_id.rsplit("_", 1)[-1]


def _result_to_row(result: WebsiteExperimentResult) -> dict:
    return {
        "sample_id": result.sample_id,
        "experiment_type": result.experiment_type,
        "model": result.model,
        "format_used": result.format_used,
        "score": result.score,
        "classification": result.classification,
        "raw_response": result.raw_response,
        "metadata": result.metadata,
    }


def _load_targets(results_root: Path, judges: set[str] | None) -> list[RepairTarget]:
    targets: list[RepairTarget] = []
    trajectory_cache: dict[Path, dict[str, dict]] = {}

    for result_file in sorted(results_root.glob("*/*/*/*/trajectory_awareness_results.jsonl")):
        rel = result_file.relative_to(results_root).parts
        if len(rel) < 5:
            continue
        arm, model_slug, benchmark, split = rel[:4]
        browser_root = results_root / arm / model_slug / "_browser_runs" / split
        if not browser_root.exists():
            logger.warning("No browser root for %s", result_file)
            continue

        if browser_root not in trajectory_cache:
            trajectory_cache[browser_root] = {
                str(t["task_id"]): t for t in _discover_browser(browser_root)
            }
        trajectory_by_id = trajectory_cache[browser_root]

        rows = [json.loads(line) for line in result_file.read_text().splitlines() if line.strip()]
        for line_index, row in enumerate(rows):
            if not _row_error(row):
                continue
            metadata = row.get("metadata") or {}
            judge_name = metadata.get("judge_name")
            if not judge_name:
                experiment_type = row.get("experiment_type") or ""
                judge_name = experiment_type.removeprefix("trajectory_") or "unknown"
            if judges is not None and judge_name not in judges:
                continue

            sample_id = row.get("sample_id") or ""
            task_id = _task_id_from_sample(sample_id, benchmark, split)
            trajectory = trajectory_by_id.get(task_id)
            if trajectory is None:
                logger.warning("No trajectory for %s in %s", task_id, browser_root)
                continue

            targets.append(
                RepairTarget(
                    result_file=result_file,
                    line_index=line_index,
                    row=row,
                    benchmark=benchmark,
                    split=split,
                    arm=arm,
                    model_slug=model_slug,
                    judge_name=judge_name,
                    task_id=task_id,
                    trajectory_dir=Path(trajectory["task_dir"]),
                )
            )
    return targets


async def _repair_target(
    target: RepairTarget,
    *,
    model: LLM,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        exp = TrajectoryAwarenessExperiment(
            model=model,
            output_dir=target.result_file.parent,
            judge_names=[target.judge_name],
        )
        sample = WebsiteSample(
            id=target.row["sample_id"],
            source=target.benchmark,
            website_type=target.split,
            metadata={
                "trajectory_dir": str(target.trajectory_dir),
                "benchmark": target.benchmark,
                "agent": target.model_slug,
                "task_id": target.task_id,
                "is_trajectory": True,
                "condition": target.arm,
                "extra_instructions_preset": "none",
                "system_prompt_frame": target.arm,
            },
        )
        results = await exp.run_sample(sample, "trajectory")

    if len(results) != 1:
        return {
            **target.row,
            "metadata": {
                **(target.row.get("metadata") or {}),
                "error": f"repair returned {len(results)} rows",
                "judge_name": target.judge_name,
            },
        }
    return _result_to_row(results[0])


def _rewrite_files(replacements: dict[tuple[Path, int], dict]) -> None:
    by_file: dict[Path, dict[int, dict]] = {}
    for (path, line_index), row in replacements.items():
        by_file.setdefault(path, {})[line_index] = row

    for path, rows_by_index in by_file.items():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for line_index, row in rows_by_index.items():
            rows[line_index] = row
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        logger.info("Rewrote %s (%d repaired rows)", path, len(rows_by_index))


async def _main_async(args: argparse.Namespace) -> int:
    judges = set(args.judges) if args.judges else None
    targets = _load_targets(args.results_root, judges)
    if args.limit is not None:
        targets = targets[: args.limit]

    print(f"Found {len(targets)} errored judge rows to repair")
    for target in targets:
        print(
            f"  {target.arm}/{target.model_slug}/{target.split} "
            f"{target.task_id} judge={target.judge_name}"
        )

    if args.dry_run or not targets:
        return 0

    model = LLM(args.judge_model, concurrency=args.judge_concurrency)
    semaphore = asyncio.Semaphore(args.task_concurrency)
    repaired = await asyncio.gather(
        *[_repair_target(t, model=model, semaphore=semaphore) for t in targets]
    )

    replacements = {
        (target.result_file, target.line_index): row
        for target, row in zip(targets, repaired, strict=True)
    }
    _rewrite_files(replacements)

    remaining = sum(1 for row in repaired if _row_error(row))
    print(
        f"Repaired {len(repaired) - remaining}/{len(repaired)} rows; remaining_errors={remaining}"
    )
    return 0 if remaining == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.7:thinking")
    parser.add_argument("--judges", nargs="*", default=None)
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--task-concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
