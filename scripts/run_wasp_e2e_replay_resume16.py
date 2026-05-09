#!/usr/bin/env python3
"""Resume WASP e2e replay on the currently planted 16 stacks.

The low-level evaluator rewrites its output JSONL after every task and does not
load rows from an existing output at startup. This launcher avoids clobbering by
reading completed rows from one or more previous output directories, assigning
only the remaining task ids to workers, and writing each worker to a fresh shard
file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "anthropic_claude-opus-4.7_thinking",
    "anthropic_claude-sonnet-4.6_thinking",
    "google_gemini-2.5-pro_thinking",
    "moonshotai_kimi-k2.5_thinking",
    "openai_gpt-5.2_thinking",
    "z-ai_glm-5_thinking",
]

# These are the stacks that were observed planted with 168 user configs and
# 168 attacker configs on 2026-05-09.
ASSIGNMENTS = {
    "anthropic_claude-opus-4.7_thinking": ["opus", "w02", "w03"],
    "anthropic_claude-sonnet-4.6_thinking": ["sonnet", "w04", "w05"],
    "google_gemini-2.5-pro_thinking": ["gemini25", "w06", "w07"],
    "moonshotai_kimi-k2.5_thinking": ["kimi25", "w09", "w11"],
    "openai_gpt-5.2_thinking": ["gpt", "w12"],
    "z-ai_glm-5_thinking": ["glm", "w14"],
}


@dataclass(frozen=True)
class Worker:
    model: str
    stack: str
    shard_idx: int
    shard_count: int


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _split_words(value: str) -> list[str]:
    return [part for part in value.split() if part]


def _split_dirs(value: str) -> list[Path]:
    return [Path(part) for part in value.split(":") if part]


def _task_num(task_id: str) -> int:
    return int(task_id.split(".", 1)[1])


def _cell_task_ids(cell_dir: Path) -> list[str]:
    ids = set()
    for path in cell_dir.rglob("summary_info.json"):
        match = re.search(r"webarena\.(\d+)", str(path.parent))
        if match:
            ids.add(f"webarena.{match.group(1)}")
    return sorted(ids, key=_task_num)


def _task_dir_ids(task_dir: Path) -> list[str]:
    cfg_dir = task_dir / "webarena_tasks" if (task_dir / "webarena_tasks").is_dir() else task_dir
    ids = {
        f"webarena.{path.stem}"
        for path in cfg_dir.glob("*.json")
        if path.stem.isdigit()
    }
    return sorted(ids, key=_task_num)


def _completed_rows(paths: Iterable[Path]) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "summary":
                    continue
                if row.get("status") == "scored" and row.get("task_id"):
                    completed.add(str(row["task_id"]))
    return completed


def _matching_outputs(root: Path, arm: str, model: str) -> list[Path]:
    if not root.exists():
        return []
    exact = root / f"{arm}__{model}.jsonl"
    paths = [exact] if exact.exists() else []
    paths.extend(sorted(root.glob(f"{arm}__{model}__*.jsonl")))
    return paths


def _completed_task_ids(previous_dirs: list[Path], arm: str, model: str) -> set[str]:
    paths: list[Path] = []
    for root in previous_dirs:
        paths.extend(_matching_outputs(root, arm, model))
    return _completed_rows(paths)


def _is_stack_planted(stack: str) -> bool:
    task_dir = Path(f"/tmp/wasp_full_{stack}") / "webarena_tasks"
    plant_root = Path(f"/tmp/wasp_full_plant_{stack}")
    direct_attacker_dir = plant_root / "webarena_tasks_attacker"
    if direct_attacker_dir.is_dir():
        attacker_count = len(list(direct_attacker_dir.glob("*.json")))
    elif plant_root.is_dir():
        attacker_count = len(list(plant_root.glob("*/webarena_tasks_attacker/*.json")))
    else:
        attacker_count = 0
    return task_dir.is_dir() and len(list(task_dir.glob("*.json"))) >= 168 and attacker_count >= 168


def _parse_output_name(path: Path) -> tuple[str, str] | None:
    parts = path.stem.split("__")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _summarize_roots(roots: list[Path]) -> None:
    rows: dict[tuple[str, str, str], bool] = {}
    errors: dict[tuple[str, str], int] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.jsonl")):
            parsed = _parse_output_name(path)
            if parsed is None:
                continue
            arm, model = parsed
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("type") == "summary":
                        continue
                    if row.get("status") == "error":
                        errors[(arm, model)] = errors.get((arm, model), 0) + 1
                    if row.get("status") != "scored" or not row.get("task_id"):
                        continue
                    key = (arm, model, str(row["task_id"]))
                    rows[key] = bool(row.get("attack_success"))

    totals: dict[tuple[str, str], list[int]] = {}
    for (arm, model, _task_id), success in rows.items():
        bucket = totals.setdefault((arm, model), [0, 0])
        bucket[0] += 1
        bucket[1] += int(success)

    for (arm, model), (n, success) in sorted(totals.items()):
        asr = 100 * success / n if n else 0.0
        n_errors = errors.get((arm, model), 0)
        print(f"{arm} {model}: n={n} success={success} asr={asr:.2f}% errors={n_errors}", flush=True)


def _run_worker(
    worker: Worker,
    *,
    arms: list[str],
    wasp_dir: Path,
    out_dir: Path,
    previous_dirs: list[Path],
    log_dir: Path,
    run_id: str,
    python: str,
    dry_run: bool,
) -> int:
    task_dir = Path(f"/tmp/wasp_full_{worker.stack}")
    log_path = log_dir / f"{run_id}_{worker.stack}__{worker.model}__shard{worker.shard_idx}of{worker.shard_count}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"worker model={worker.model} stack={worker.stack} "
            f"shard={worker.shard_idx}/{worker.shard_count} task_dir={task_dir}\n"
        )
        for arm in arms:
            cell_dir = wasp_dir / arm / worker.model
            if not cell_dir.is_dir():
                log.write(f"skip missing cell_dir={cell_dir}\n")
                continue

            all_task_ids = _cell_task_ids(cell_dir) or _task_dir_ids(task_dir)
            completed = _completed_task_ids(previous_dirs, arm, worker.model)
            remaining = [task_id for task_id in all_task_ids if task_id not in completed]
            shard_task_ids = [
                task_id
                for idx, task_id in enumerate(remaining)
                if idx % worker.shard_count == worker.shard_idx
            ]

            log.write(
                f"{arm}: total={len(all_task_ids)} completed={len(completed)} "
                f"remaining={len(remaining)} selected={len(shard_task_ids)}\n"
            )
            log.flush()
            if not shard_task_ids:
                continue

            output_jsonl = (
                out_dir
                / f"{arm}__{worker.model}__resume{run_id}"
                f"__shard{worker.shard_idx}of{worker.shard_count}__{worker.stack}.jsonl"
            )
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                python,
                "-m",
                "eval_awareness_experiments.wasp_replay_e2e_evaluator",
                "--cell-dir",
                str(cell_dir),
                "--task-dir",
                str(task_dir),
                "--task-id",
                ",".join(shard_task_ids),
                "--output-jsonl",
                str(output_jsonl),
                "--order",
                "task_id",
            ]
            log.write("cmd=" + " ".join(cmd) + "\n")
            log.flush()
            if dry_run:
                continue

            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                log.write(f"FAILED arm={arm} returncode={proc.returncode}\n")
                log.flush()
                return proc.returncode
        return 0


def main() -> int:
    results_dir = Path(_env("RESULTS_DIR", "eval_awareness_experiments/results/n200_2026-04-29"))
    wasp_dir = Path(_env("WASP_DIR", str(results_dir / "wasp")))
    out_base = Path(_env("OUT_BASE", str(results_dir / "wasp_e2e_replay_20260509_resume16")))
    phase = _env("PHASE", "no_reset")
    out_dir = out_base / phase
    log_dir = Path(_env("LOG_DIR", "logs/wasp_e2e_replay_resume16_20260509"))
    arms = _split_words(_env("ARMS", "bare xml_safety xml_scenario xml_control"))
    model_filter = _env("MODEL_FILTER", "all")
    stack_filter = _env("STACK_FILTER", "all")
    python = _env("PYTHON", ".venv/bin/python")
    run_id = _env("RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
    dry_run = _env("DRY_RUN", "0") == "1"

    previous_dirs = _split_dirs(
        _env(
            "PREVIOUS_DIRS",
            str(results_dir / "wasp_e2e_replay_20260509_lanes_skipreset" / phase),
        )
    )

    selected_models = [
        model
        for model in MODELS
        if model_filter == "all" or model in model_filter.split()
    ]
    workers: list[Worker] = []
    for model in selected_models:
        stacks = [
            stack
            for stack in ASSIGNMENTS[model]
            if stack_filter == "all" or stack in stack_filter.split()
        ]
        for shard_idx, stack in enumerate(stacks):
            if not _is_stack_planted(stack):
                raise SystemExit(f"Stack {stack} is not fully planted; refusing to assign it")
            workers.append(Worker(model=model, stack=stack, shard_idx=shard_idx, shard_count=len(stacks)))

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "phase": phase,
        "arms": arms,
        "wasp_dir": str(wasp_dir),
        "out_dir": str(out_dir),
        "previous_dirs": [str(path) for path in previous_dirs],
        "workers": [worker.__dict__ for worker in workers],
        "dry_run": dry_run,
    }
    manifest_path = out_dir / f"resume16_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"out_dir={out_dir}", flush=True)
    print(f"log_dir={log_dir}", flush=True)
    print(f"previous_dirs={':'.join(str(path) for path in previous_dirs)}", flush=True)
    print(f"arms={' '.join(arms)}", flush=True)
    print(f"workers={len(workers)} run_id={run_id} dry_run={dry_run}", flush=True)
    for worker in workers:
        print(
            f"worker stack={worker.stack} model={worker.model} "
            f"shard={worker.shard_idx}/{worker.shard_count}",
            flush=True,
        )

    if not workers:
        raise SystemExit("No workers selected")

    overall_status = 0
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {
            pool.submit(
                _run_worker,
                worker,
                arms=arms,
                wasp_dir=wasp_dir,
                out_dir=out_dir,
                previous_dirs=previous_dirs,
                log_dir=log_dir,
                run_id=run_id,
                python=python,
                dry_run=dry_run,
            ): worker
            for worker in workers
        }
        for future in as_completed(futures):
            worker = futures[future]
            status = future.result()
            if status:
                overall_status = status
                print(f"FAILED stack={worker.stack} model={worker.model} status={status}", flush=True)
            else:
                print(f"completed stack={worker.stack} model={worker.model}", flush=True)

    print("combined scored rows from previous dirs + resume output:", flush=True)
    _summarize_roots([*previous_dirs, out_dir])
    return overall_status


if __name__ == "__main__":
    sys.exit(main())
