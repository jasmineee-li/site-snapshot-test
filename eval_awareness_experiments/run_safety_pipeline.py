"""Unified two-stage safety-benchmark pipeline.

Stage 1 (optional): dispatch to the per-benchmark runner to produce trajectories.
  - doomarena, wasp:       browser runs → results/browsergym/study_*/...
  - agentdojo, injecagent: tool-calling → results/toolcalling/{bench}/.../{task}.json
  - eia:                   import-only; stage 1 is a no-op, reads data/eia/manifest.json

Stage 2 (always): discover the produced trajectories, build WebsiteSamples, and
run TrajectoryAwarenessExperiment (unchanged) with the configured judges.

Usage:
    # Run WASP gitlab, then judge.
    python -m eval_awareness_experiments.run_safety_pipeline \
        --benchmark wasp --splits gitlab --tasks-per-split 20 \
        --model-name anthropic/claude-opus-4.7:thinking \
        --condition baseline --extra-instructions-preset motive \
        --judge-model anthropic/claude-opus-4.6:thinking

    # Judge EIA imported trajectories only.
    python -m eval_awareness_experiments.run_safety_pipeline \
        --benchmark eia --stage judge-only \
        --judge-model anthropic/claude-opus-4.6:thinking
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_awareness_experiments.extra_instructions import PRESETS as EXTRA_INSTRUCTION_PRESETS
from eval_awareness_experiments.injection_modifier import ALL_CONDITIONS
from eval_awareness_experiments.system_prompt_frame import ALL_FRAMES

logger = logging.getLogger(__name__)

BROWSER_BENCHMARKS = ("doomarena", "wasp")
TOOLCALLING_BENCHMARKS = ("agentdojo", "injecagent")
IMPORT_ONLY_BENCHMARKS = ("eia",)
ALL_BENCHMARKS = BROWSER_BENCHMARKS + TOOLCALLING_BENCHMARKS + IMPORT_ONLY_BENCHMARKS

BENCHMARK_RUNNERS = {
    "doomarena": "eval_awareness_experiments.run_doomarena_glm5_popup",
    "wasp": "eval_awareness_experiments.run_wasp",
    "agentdojo": "eval_awareness_experiments.run_agentdojo",
    "injecagent": "eval_awareness_experiments.run_injecagent",
}

DEFAULT_SPLITS = {
    "doomarena": ["reddit", "shopping", "shopping_admin", "gitlab"],
    "wasp": ["gitlab", "reddit"],
    "agentdojo": ["workspace"],
    "injecagent": ["dh_base"],
    "eia": ["baseline"],
}

MODEL_SLUG_TO_NAME = {
    "anthropic_claude-opus-4.7_thinking": "anthropic/claude-opus-4.7:thinking",
    "anthropic_claude-sonnet-4.6_thinking": "anthropic/claude-sonnet-4.6:thinking",
    "google_gemini-2.5-pro_thinking": "google/gemini-2.5-pro:thinking",
    "moonshotai_kimi-k2.5_thinking": "moonshotai/kimi-k2.5:thinking",
    "openai_gpt-5.2_thinking": "openai/gpt-5.2:thinking",
    "z-ai_glm-5_thinking": "z-ai/glm-5:thinking",
}


def _model_name_from_slug(slug: str) -> str | None:
    if slug in MODEL_SLUG_TO_NAME:
        return MODEL_SLUG_TO_NAME[slug]
    if slug.startswith("local_"):
        return "local/" + slug[len("local_") :]
    return None


# Stable per-split offsets so each parallel split inside one cell gets a
# unique --report-port. The cell's port-base is unique-per-process (PID-derived
# unless --report-port-base is set explicitly), and these offsets fan out
# across the cell's 4 site splits. Old default `1234` was shared across every
# concurrent split + every concurrent cell — see DOOMARENA_ROOT_CAUSE_HANDOFF.md
# (root cause 3).
_DOOMARENA_SPLIT_PORT_OFFSETS = {
    "reddit": 0,
    "shopping": 1,
    "shopping_admin": 2,
    "gitlab": 3,
}


def _resolve_report_port_base(args: argparse.Namespace) -> int:
    """Pick a unique-per-process port base for a DoomArena cell.

    If --report-port-base is set explicitly, use it. Otherwise derive from
    PID, which makes every `run_safety_pipeline` invocation get its own
    base without coordination. The 16-stream launcher invokes us 16×6=96
    times across the matrix; PIDs collide only after wraparound (~4M apart)
    so this is reliable for one run.
    """
    if getattr(args, "report_port_base", None):
        return int(args.report_port_base)
    # Map PID into [12000, 62000) for stable, well-known-port-avoiding range.
    return 12000 + (os.getpid() % 50000)


def _browser_stage1_timeout(args: argparse.Namespace) -> int:
    """Compute the per-split browser subprocess timeout.

    AgentLab still enforces the tighter per-task episode timeout
    (`max_steps * avg_step_timeout`). This is the coarse process-level
    guard for setup, teardown, and browser child-process hangs.
    """
    if getattr(args, "browser_stage1_timeout", None) is not None:
        return int(args.browser_stage1_timeout)
    return int(args.tasks_per_split) * int(args.max_steps) * int(args.avg_step_timeout) + int(
        args.browser_stage1_overhead
    )


def _browser_stage1_idle_timeout(args: argparse.Namespace) -> int:
    """Maximum quiet period for a browser split before killing its process tree.

    AgentLab still enforces the tighter per-task timeout. If a split has not
    written logs, step files, summaries, or aggregate results for an hour, it
    is almost certainly stuck in Playwright/Chrome teardown or task cleanup.
    """
    value = getattr(args, "browser_stage1_idle_timeout", None)
    if value is None:
        return 3600
    return int(value)


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict | None = None,
    timeout_sec: int | None = None,
) -> bool:
    """Run a subprocess and return whether it succeeded.

    If `timeout_sec` is set, wraps the command with Linux `timeout(1)` so a
    hung subprocess (e.g. browser-track Playwright env.close() deadlock — see
    py-spy stack trace in commit history) gets force-killed instead of
    soaking up the whole stream's wallclock. We use the `timeout` binary
    rather than subprocess.run's `timeout=` kwarg because the latter only
    kills the immediate child — Playwright's headless-Chromium child tree
    needs the whole process-group cleanup that `timeout(1)` does.
    """
    env = env or os.environ.copy()
    if timeout_sec is not None:
        # --kill-after=10: if SIGTERM doesn't work within 10s, SIGKILL.
        cmd = ["timeout", "--kill-after=10", str(timeout_sec), *cmd]
    logger.info("  $ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    for line in (result.stdout or "").splitlines()[-30:]:
        logger.info(f"  [stdout] {line}")
    for line in (result.stderr or "").splitlines()[-15:]:
        logger.warning(f"  [stderr] {line}")
    if result.returncode != 0:
        # `timeout` exits 124 on timeout, 137 if SIGKILL was needed.
        if result.returncode in (124, 137):
            logger.error(
                f"  subprocess TIMED OUT after {timeout_sec}s "
                f"(exit={result.returncode}). The subprocess may be stuck in "
                f"backend setup, task reset/login, agent execution, or browser "
                f"teardown — check the subprocess logs for the last completed phase."
            )
        else:
            logger.error(f"  subprocess failed: exit={result.returncode}")
    return result.returncode == 0


def _tail_lines(path: Path, n: int) -> list[str]:
    """Return the last n lines from a text log without loading huge files."""
    if n <= 0 or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=n))
    except OSError as e:
        logger.warning(f"  failed to read log tail from {path}: {e!r}")
        return []


def _latest_browser_activity_time(
    split_root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> float:
    """Return latest mtime for files that prove a browser split is moving."""
    paths = [
        stdout_path,
        stderr_path,
        split_root / "full_results.csv",
        split_root / "short_results.csv",
        split_root / "attack_results_v2.csv",
        split_root / "attack_df_deduplicated.csv",
        split_root / "attack_df_legacy.csv",
    ]
    for pattern in (
        "summary_info.json",
        "attack_summary_info.json",
        "*.log",
        "step_*.pkl.gz",
    ):
        paths.extend(split_root.rglob(pattern))

    latest = 0.0
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            pass
    return latest


def _is_hidden_browser_path(split_root: Path, path: Path) -> bool:
    """Skip AgentLab dirs hidden by relaunch (`_old_dir`) or manual archives."""
    try:
        rel_parts = path.relative_to(split_root).parts
    except ValueError:
        return False
    return any(part.startswith(("_", ".")) for part in rel_parts)


def _browser_task_id(task_dir: Path) -> str:
    task_name = task_dir.name
    parts = task_name.rsplit("_on_", 1)
    return parts[-1].rsplit("_", 1)[0] if len(parts) > 1 else task_name


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _browser_task_progress(split_root: Path) -> dict[str, dict]:
    """Summarize per-task state from AgentLab files under one split root."""
    task_dirs: set[Path] = set()
    for marker_name in ("exp_args.pkl", "summary_info.json"):
        for marker in split_root.rglob(marker_name):
            task_dir = marker.parent
            if _is_hidden_browser_path(split_root, task_dir):
                continue
            task_dirs.add(task_dir)

    progress: dict[str, dict] = {}
    for task_dir in sorted(task_dirs):
        summary_path = task_dir / "summary_info.json"
        attack_summary_path = task_dir / "attack_summary_info.json"
        experiment_log = task_dir / "experiment.log"
        step_count = sum(1 for _ in task_dir.glob("step_*.pkl.gz"))
        log_started = False
        try:
            log_started = experiment_log.stat().st_size > 0
        except OSError:
            pass
        started = step_count > 0 or log_started or summary_path.exists()

        summary = _read_json_file(summary_path) if summary_path.exists() else {}
        attack_summary = (
            _read_json_file(attack_summary_path) if attack_summary_path.exists() else {}
        )
        progress[str(task_dir)] = {
            "task_id": _browser_task_id(task_dir),
            "started": started,
            "completed": summary_path.exists(),
            "err_msg": summary.get("err_msg"),
            "n_steps": summary.get("n_steps"),
            "cum_reward": summary.get("cum_reward"),
            "terminated": summary.get("terminated"),
            "truncated": summary.get("truncated"),
            "attack_successful": attack_summary.get("attack_successful"),
            "step_count": step_count,
        }
    return progress


def _browser_split_completion(
    split_root: Path,
    *,
    expected_total: int,
) -> dict[str, int | bool]:
    """Return unique-task completion counts for one browser split root."""
    progress = _browser_task_progress(split_root)
    completed_ids = {info["task_id"] for info in progress.values() if info["completed"]}
    error_ids = {
        info["task_id"] for info in progress.values() if info["completed"] and info.get("err_msg")
    }
    incomplete_ids = {
        info["task_id"] for info in progress.values() if info["started"] and not info["completed"]
    }
    done = expected_total > 0 and len(completed_ids) >= expected_total
    return {
        "completed": len(completed_ids),
        "expected": expected_total,
        "errors": len(error_ids),
        "incomplete": len(incomplete_ids),
        "done": done,
    }


def _log_browser_task_progress(item: dict, *, expected_total: int) -> None:
    """Emit one parent-log line for every newly-started/completed browser task."""
    split = item["split"]
    progress = _browser_task_progress(item["split_root"])
    started_keys = {k for k, v in progress.items() if v["started"]}
    completed_keys = {k for k, v in progress.items() if v["completed"]}

    new_started = sorted(
        started_keys - item["seen_started_tasks"],
        key=lambda k: (progress[k]["task_id"], k),
    )
    for key in new_started:
        info = progress[key]
        logger.info(
            f"  [{split}] task START {info['task_id']} "
            f"(started={len(item['seen_started_tasks']) + 1}/{expected_total})"
        )
        item["seen_started_tasks"].add(key)

    new_completed = sorted(
        completed_keys - item["seen_completed_tasks"],
        key=lambda k: (progress[k]["task_id"], k),
    )
    for key in new_completed:
        info = progress[key]
        status = "ERROR" if info.get("err_msg") else "DONE"
        message = (
            f"  [{split}] task {status} {info['task_id']} "
            f"(completed={len(item['seen_completed_tasks']) + 1}/{expected_total}, "
            f"steps={info.get('n_steps')}, reward={info.get('cum_reward')}, "
            f"attack_success={info.get('attack_successful')})"
        )
        if info.get("err_msg"):
            logger.warning(message)
        else:
            logger.info(message)
        item["seen_completed_tasks"].add(key)


def _log_existing_browser_task_progress(
    split: str,
    progress: dict[str, dict],
    *,
    expected_total: int,
) -> None:
    """Emit one line per task that already exists before a resume subprocess."""
    completed = [(k, v) for k, v in progress.items() if v["completed"]]
    incomplete = [(k, v) for k, v in progress.items() if v["started"] and not v["completed"]]
    for idx, (_, info) in enumerate(sorted(completed, key=lambda kv: (kv[1]["task_id"], kv[0])), 1):
        status = "EXISTING_ERROR" if info.get("err_msg") else "EXISTING_DONE"
        logger.info(
            f"  [{split}] task {status} {info['task_id']} "
            f"(completed={idx}/{expected_total}, steps={info.get('n_steps')}, "
            f"reward={info.get('cum_reward')}, "
            f"attack_success={info.get('attack_successful')})"
        )
    for idx, (_, info) in enumerate(
        sorted(incomplete, key=lambda kv: (kv[1]["task_id"], kv[0])), 1
    ):
        logger.info(
            f"  [{split}] task EXISTING_INCOMPLETE {info['task_id']} "
            f"(started={idx}/{expected_total}, steps_written={info.get('step_count')})"
        )


def _terminate_process_group(
    proc: subprocess.Popen,
    *,
    split: str,
    reason: str,
    grace_sec: int = 10,
) -> bool:
    """Terminate a split process group, including Playwright/Chrome children."""
    if proc.poll() is not None:
        return True
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return True

    logger.error(f"  [{split}] terminating process group {pgid}: {reason}")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError as e:
        logger.error(f"  [{split}] failed to SIGTERM process group {pgid}: {e!r}")
        return False

    try:
        proc.wait(timeout=grace_sec)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"  [{split}] process group {pgid} ignored SIGTERM; sending SIGKILL")

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as e:
        logger.error(f"  [{split}] failed to SIGKILL process group {pgid}: {e!r}")
    try:
        proc.wait(timeout=grace_sec)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"  [{split}] process group {pgid} still did not exit")
        return False


def _write_run_meta(run_dir: Path, args, benchmark: str, split: str) -> None:
    """Persist the original-run config alongside trajectories.

    Read by judge-only re-runs (`_load_run_meta`) to recover model_name /
    condition / extra_instructions_preset, which would otherwise default to
    None / 'none' in the re-judge's pipeline manifest.
    """
    meta = {
        "benchmark": benchmark,
        "split": split,
        "model_name": args.model_name,
        "condition": args.condition,
        "extra_instructions_preset": args.extra_instructions_preset,
        "system_prompt_frame": getattr(args, "system_prompt_frame", "none"),
        "shuffle": args.shuffle,
        "tasks_per_split": args.tasks_per_split,
        "max_steps": getattr(args, "max_steps", None),
        "toolcalling_max_tokens": getattr(args, "toolcalling_max_tokens", None),
        "avg_step_timeout": getattr(args, "avg_step_timeout", None),
        "browser_stage1_timeout": getattr(args, "browser_stage1_timeout", None),
        "browser_stage1_overhead": getattr(args, "browser_stage1_overhead", None),
        "browser_stage1_idle_timeout": getattr(args, "browser_stage1_idle_timeout", None),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    except OSError as e:
        logger.warning(f"  Failed to write run_meta.json to {run_dir}: {e}")


_KNOWN_CONDITIONS = ("baseline", *(c for c in ALL_CONDITIONS if c != "baseline"))


def _load_run_meta(run_dir: Path) -> dict | None:
    """Recover original-run config from a trajectory dir.

    Tries `run_dir/run_meta.json` first (newer runs). Falls back to parsing
    the dir name `{condition}_{preset}_{model_slug}` for tool-calling dirs
    (older runs that pre-date `run_meta.json`). Returns None if neither
    works — caller should fall back to CLI args.
    """
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"  Failed to read {meta_path}: {e}")

    name = run_dir.name
    for cond in _KNOWN_CONDITIONS:
        cond_prefix = f"{cond}_"
        if not name.startswith(cond_prefix):
            continue
        rest = name[len(cond_prefix) :]
        # Match the longest preset name first so `scratchpad_bare` wins over `scratchpad`.
        for preset in sorted(EXTRA_INSTRUCTION_PRESETS.keys(), key=len, reverse=True):
            preset_prefix = f"{preset}_"
            if rest.startswith(preset_prefix):
                model_slug = rest[len(preset_prefix) :]
                return {
                    "condition": cond,
                    "extra_instructions_preset": preset,
                    # Reverse of `model_name.replace("/", "_").replace(":", "_")`
                    # is lossy — return slug as-is, marked.
                    "model_name": model_slug,
                    "_recovered_from": "run_dir_name",
                }

    parts = run_dir.parts
    if "_browser_runs" in parts:
        idx = parts.index("_browser_runs")
        if idx > 0:
            model_name = _model_name_from_slug(parts[idx - 1])
            if model_name:
                condition = parts[idx - 2] if idx > 1 else None
                return {
                    "model_name": model_name,
                    "condition": (condition if condition in _KNOWN_CONDITIONS else None),
                    "_recovered_from": "browser_results_path",
                }
    return None


def _build_browser_cmd(
    benchmark: str,
    split: str,
    args,
    *,
    report_port: int | None = None,
    cell_results_dir: Path | None = None,
    force_relaunch_incomplete: bool = False,
) -> list[str]:
    """Build the runner command for one browser benchmark + split.

    `report_port` overrides the default and is the per-split unique value
    derived in `_stage1_browser_parallel_splits`. `cell_results_dir` (when
    set) tells the runner to write its AgentLab study output under a
    cell-owned namespace instead of the global `results/browsergym/` pool —
    avoids the cross-cell study_dir race documented in
    DOOMARENA_ROOT_CAUSE_HANDOFF.md (root cause 4).
    """
    runner = BENCHMARK_RUNNERS[benchmark]
    cmd = [
        sys.executable,
        "-m",
        runner,
        "--single-site",
        split,
        "--max-tasks",
        str(args.tasks_per_split),
        "--model-name",
        args.model_name,
        "--extra-instructions-preset",
        args.extra_instructions_preset,
        "--system-prompt-frame",
        args.system_prompt_frame,
        "--max-steps",
        str(args.max_steps),
        "--avg-step-timeout",
        str(args.avg_step_timeout),
        "--shuffle",
        str(args.shuffle),
    ]
    # --condition (causal injection wrapping: baseline/eval_cue/scenario_cue) is
    # only implemented in WASP's runner. DoomArena uses runtime AttackConfigs
    # instead, so it doesn't accept the flag.
    if benchmark == "wasp":
        cmd += ["--condition", args.condition]
        wasp_task_dir = getattr(args, "wasp_task_dir", None) or "/tmp/wasp_full"
        cmd += ["--task-dir", wasp_task_dir]
    if benchmark == "doomarena":
        port = report_port if report_port is not None else getattr(args, "report_port", None)
        if port is not None:
            cmd += ["--report-port", str(port)]
        # Pin reachability check to the actual site we're exercising — the old
        # default `--online-sites reddit` left non-reddit splits silently
        # mis-checking. See root cause 9 in the handoff.
        cmd += ["--online-sites", split]
    if cell_results_dir is not None:
        cmd += ["--results-dir", str(cell_results_dir / split)]
    if force_relaunch_incomplete or getattr(args, "browser_relaunch_incomplete", False):
        cmd += ["--relaunch-incomplete"]
    return cmd


def _match_study_dirs_to_splits(new_dirs: list[Path], splits: list[str]) -> dict[str, Path]:
    """Match each newly-created AgentLab study_dir to its split by inspecting
    the inner agent-run subdir name (which always contains the split string,
    e.g., `..._on-webarena-reddit-single-site-...`).
    """
    out: dict[str, Path] = {}
    for study_dir in new_dirs:
        try:
            children = list(study_dir.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            for split in splits:
                # Match `-<split>-` or `-<split>$` in the agent run subdir.
                if f"-{split}-" in child.name or child.name.endswith(f"-{split}"):
                    if split not in out:
                        out[split] = study_dir
                    break
    return out


def _cell_results_dir(args: argparse.Namespace, benchmark: str) -> Path | None:
    """Where this cell's browser-runner subprocesses should write their
    AgentLab study_*. Cell-namespaced under `args.output_dir` so two
    concurrent cells can never alias each other's trajectories. Returns
    None for benchmarks that don't yet support the override."""
    if benchmark not in {"doomarena", "wasp"}:
        return None
    return Path(args.output_dir) / "_browser_runs"


def _browser_judge_only_root(args: argparse.Namespace, split: str) -> Path | None:
    """Default trajectory root for browser benchmark judge-only re-runs.

    Stage 1 writes browser trajectories to `<cell output>/_browser_runs/<split>`.
    That path is known from `--output-dir`, so matrix-level judge-only runs do
    not need to pass verbose `--existing-dirs split:path ...` arguments.
    """
    cell_dir = _cell_results_dir(args, args.benchmark)
    if cell_dir is None:
        return None
    root = cell_dir / split
    return root if root.exists() else None


def _stage1_browser(benchmark: str, split: str, args) -> Path | None:
    """Run DoomArena or WASP for ONE split (sequential / single-split path).
    Kept for callers that want per-split control. Most callers should prefer
    `_stage1_browser_parallel_splits` to run all splits concurrently.
    """
    report_port = None
    cell_dir = _cell_results_dir(args, benchmark)
    if benchmark == "doomarena":
        base = _resolve_report_port_base(args)
        offset = _DOOMARENA_SPLIT_PORT_OFFSETS.get(split, 0)
        report_port = base + offset
        _log_cell_env(args, benchmark, split, report_port, cell_dir)
    before = set((REPO_ROOT / "results" / "browsergym").glob("study_*"))
    timeout_sec = _browser_stage1_timeout(args)
    expected_total = int(getattr(args, "tasks_per_split", 0) or 0)
    max_attempts = max(1, int(getattr(args, "browser_stage1_relaunch_attempts", 3) or 1))

    last_study_dir: Path | None = None
    for attempt in range(1, max_attempts + 1):
        cmd = _build_browser_cmd(
            benchmark,
            split,
            args,
            report_port=report_port,
            cell_results_dir=cell_dir,
            force_relaunch_incomplete=attempt > 1,
        )
        ok = _run_subprocess(cmd, cwd=REPO_ROOT, timeout_sec=timeout_sec)
        study_dir = _resolve_study_dir(benchmark, split, cell_dir, before)
        if study_dir is not None:
            last_study_dir = study_dir
            _write_run_meta(study_dir, args, benchmark, split)

        if study_dir is None:
            if attempt < max_attempts:
                logger.warning(
                    f"  [{split}] no study_dir after attempt {attempt}/{max_attempts}; "
                    "relaunching split"
                )
                continue
            return last_study_dir

        completion = _browser_split_completion(
            study_dir,
            expected_total=expected_total,
        )
        if ok and completion["done"]:
            return study_dir

        if attempt < max_attempts:
            logger.warning(
                f"  [{split}] partial after attempt {attempt}/{max_attempts}: "
                f"{completion['completed']}/{completion['expected']} unique tasks "
                f"({completion['errors']} error rows, "
                f"{completion['incomplete']} incomplete started); relaunching split"
            )
            continue

        logger.error(
            f"  [{split}] exhausted {max_attempts} attempt(s): "
            f"{completion['completed']}/{completion['expected']} unique tasks "
            f"({completion['errors']} error rows, "
            f"{completion['incomplete']} incomplete started)"
        )
        return study_dir

    return last_study_dir


def _resolve_study_dir(
    benchmark: str,
    split: str,
    cell_dir: Path | None,
    before: set[Path],
) -> Path | None:
    """Return the study_dir produced by a runner.

    For DoomArena with a cell-owned output dir, the runner is launched
    with `--results-dir <cell_dir>/<split>` which becomes AgentLab's
    `exp_root` directly — so the study contents (agent-run subdirs +
    attack_config.json + CSVs) land in `<cell_dir>/<split>/` itself,
    NOT under a `study_*` subdir. Treat the split dir AS the study
    when it has trajectory subdirs in it.

    Falls back to the global `results/browsergym/` pool for benchmarks
    that haven't been migrated to cell namespacing.
    """
    if cell_dir is not None:
        split_dir = cell_dir / split
        if split_dir.exists():
            # Did the runner actually write trajectories here?
            if any(split_dir.rglob("summary_info.json")):
                return split_dir
            # Sometimes a wrapper layer creates a `study_*` subdir; tolerate it.
            new = sorted(split_dir.glob("study_*"), key=lambda p: p.stat().st_mtime)
            if new and any(new[-1].rglob("summary_info.json")):
                return new[-1]
    after = set((REPO_ROOT / "results" / "browsergym").glob("study_*"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    return new[-1] if new else None


def _log_cell_env(
    args: argparse.Namespace,
    benchmark: str,
    split: str,
    report_port: int,
    cell_dir: Path | None,
) -> None:
    """Single-line per-split breadcrumb. Lets us grep logs for the exact
    (arm, model, split, site_url, report_port) tuple — root cause 9."""
    arm_key = f"preset={args.extra_instructions_preset}/" f"frame={args.system_prompt_frame}"
    site_env_var = {
        "reddit": "REDDIT",
        "shopping": "SHOPPING",
        "shopping_admin": "SHOPPING_ADMIN",
        "gitlab": "GITLAB",
    }.get(split, "")
    site_url = os.environ.get(site_env_var, "<unset>") if site_env_var else "<unset>"
    logger.info(
        f"  [cell] benchmark={benchmark} arm={arm_key} model={args.model_name} "
        f"split={split} {site_env_var}={site_url} report_port={report_port} "
        f"results_dir={cell_dir}"
    )


def _stage1_browser_parallel_splits(
    benchmark: str,
    splits: list[str],
    args,
) -> dict[str, Path]:
    """Run all browser splits in parallel via Popen, return {split: study_dir}.

    Splits hit different docker services (reddit→forum, gitlab→gitlab,
    shopping→shopping, etc.) so they don't collide. This 4×'s the per-cell
    speed for DoomArena (4 splits) compared to sequential. WASP has 2 splits
    so 2×.

    Quirks handled:
    - AgentLab study_dir uses second-resolution timestamps. Two concurrent
      launches in the same wallclock second produce dirs at different paths
      (AgentLab disambiguates internally), but to avoid edge cases we
      stagger Popen launches by `_LAUNCH_STAGGER_SEC` seconds.
    - Match study_dirs to splits by inspecting agent-run subdir names
      (always contain the `--single-site` string), since the global
      glob-diff before/after pattern can't tell which dir is for which
      split when launches are concurrent.
    - Each subprocess wrapped in `timeout(1)` and launched in its own process
      group so idle watchdog kills clean up Playwright/Chrome descendants.
    """
    timeout_sec = _browser_stage1_timeout(args)
    idle_timeout_sec = _browser_stage1_idle_timeout(args)
    before = set((REPO_ROOT / "results" / "browsergym").glob("study_*"))
    cell_dir = _cell_results_dir(args, benchmark)
    if cell_dir is not None:
        cell_dir.mkdir(parents=True, exist_ok=True)

    # Per-split unique report port, derived from a per-process base.
    if benchmark == "doomarena":
        port_base = _resolve_report_port_base(args)
    else:
        port_base = None

    results: dict[str, bool] = {}
    expected_total = int(getattr(args, "tasks_per_split", 0) or 0)
    max_attempts = max(1, int(getattr(args, "browser_stage1_relaunch_attempts", 3) or 1))
    splits_to_run = list(splits)

    for attempt in range(1, max_attempts + 1):
        procs: list[dict] = []
        for i, split in enumerate(splits_to_run):
            report_port = None
            split_index = splits.index(split)
            if port_base is not None:
                report_port = port_base + _DOOMARENA_SPLIT_PORT_OFFSETS.get(split, split_index)
                _log_cell_env(args, benchmark, split, report_port, cell_dir)
            cmd = _build_browser_cmd(
                benchmark,
                split,
                args,
                report_port=report_port,
                cell_results_dir=cell_dir,
                force_relaunch_incomplete=attempt > 1,
            )
            cmd = ["timeout", "--kill-after=10", str(timeout_sec), *cmd]
            if i > 0:
                time.sleep(_LAUNCH_STAGGER_SEC)
            logger.info(
                f"  [parallel split {split} attempt {attempt}/{max_attempts}] " f"$ {' '.join(cmd)}"
            )
            if cell_dir is not None:
                log_dir = cell_dir / split
            else:
                log_dir = (
                    REPO_ROOT
                    / "results"
                    / "browsergym"
                    / "_split_logs"
                    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
                )
            log_dir.mkdir(parents=True, exist_ok=True)
            log_prefix = "subprocess" if attempt == 1 else f"subprocess.attempt{attempt}"
            stdout_path = log_dir / f"{log_prefix}.stdout.log"
            stderr_path = log_dir / f"{log_prefix}.stderr.log"
            baseline_progress = _browser_task_progress(log_dir)
            _log_existing_browser_task_progress(
                split,
                baseline_progress,
                expected_total=expected_total,
            )
            stdout_file = stdout_path.open("w", encoding="utf-8")
            stderr_file = stderr_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=os.environ.copy(),
                start_new_session=True,
            )
            now = time.time()
            procs.append(
                {
                    "split": split,
                    "proc": proc,
                    "cmd": cmd,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "stdout_file": stdout_file,
                    "stderr_file": stderr_file,
                    "split_root": log_dir,
                    "started_at": now,
                    "last_activity": now,
                    "kill_reason": None,
                    "closed": False,
                    "seen_started_tasks": {k for k, v in baseline_progress.items() if v["started"]},
                    "seen_completed_tasks": {
                        k for k, v in baseline_progress.items() if v["completed"]
                    },
                }
            )

        # Wait for this attempt. Child stdout/stderr go to files instead of pipes:
        # this avoids pipe-buffer backpressure when one split logs heavily while
        # another split is still running. Poll all splits together so a hung first
        # split cannot prevent us from noticing later splits have completed.
        pending = {item["split"]: item for item in procs}

        def _finalize_split(item: dict) -> None:
            split = item["split"]
            proc = item["proc"]
            stdout_path = item["stdout_path"]
            stderr_path = item["stderr_path"]
            if not item["closed"]:
                item["stdout_file"].close()
                item["stderr_file"].close()
                item["closed"] = True
            logger.info(f"  [{split}] stdout log → {stdout_path}")
            logger.info(f"  [{split}] stderr log → {stderr_path}")
            for line in _tail_lines(stdout_path, 30):
                logger.info(f"  [{split} stdout] {line.rstrip()}")
            for line in _tail_lines(stderr_path, 15):
                logger.warning(f"  [{split} stderr] {line.rstrip()}")
            ok = proc.returncode == 0
            if item["kill_reason"]:
                logger.error(
                    f"  [{split}] subprocess killed: {item['kill_reason']} "
                    f"(exit={proc.returncode})"
                )
            elif proc.returncode in (124, 137):
                logger.error(
                    f"  [{split}] subprocess TIMED OUT after {timeout_sec}s "
                    f"(exit={proc.returncode}). The split may be stuck in backend "
                    f"setup, task reset/login, agent execution, or browser teardown; "
                    f"see {stdout_path} and {stderr_path}."
                )
            elif not ok:
                logger.error(f"  [{split}] subprocess failed: exit={proc.returncode}")
            results[split] = ok

        while pending:
            now = time.time()
            for split, item in list(pending.items()):
                proc = item["proc"]
                activity = _latest_browser_activity_time(
                    item["split_root"],
                    item["stdout_path"],
                    item["stderr_path"],
                )
                if activity > item["last_activity"]:
                    item["last_activity"] = activity
                _log_browser_task_progress(item, expected_total=expected_total)

                if proc.poll() is not None:
                    _finalize_split(item)
                    del pending[split]
                    continue

                elapsed = now - item["started_at"]
                idle_for = now - item["last_activity"]
                if elapsed > timeout_sec + 30:
                    item["kill_reason"] = f"hard timeout after {timeout_sec}s process budget"
                    terminated = _terminate_process_group(
                        proc,
                        split=split,
                        reason=item["kill_reason"],
                    )
                    if not terminated:
                        continue
                    _finalize_split(item)
                    del pending[split]
                    continue
                if idle_timeout_sec > 0 and idle_for > idle_timeout_sec:
                    item["kill_reason"] = (
                        f"idle timeout after {idle_timeout_sec}s without browser "
                        f"logs/results activity"
                    )
                    terminated = _terminate_process_group(
                        proc,
                        split=split,
                        reason=item["kill_reason"],
                    )
                    if not terminated:
                        continue
                    _finalize_split(item)
                    del pending[split]

            if pending:
                time.sleep(15)

        retry_splits: list[str] = []
        for split in splits_to_run:
            split_root = cell_dir / split if cell_dir is not None else None
            if split_root is not None and split_root.exists():
                completion = _browser_split_completion(
                    split_root,
                    expected_total=expected_total,
                )
                split_done = (
                    bool(completion["done"]) if expected_total > 0 else bool(results.get(split))
                )
            else:
                completion = {
                    "completed": 0,
                    "expected": expected_total,
                    "errors": 0,
                    "incomplete": 0,
                    "done": False,
                }
                split_done = False

            if split_done:
                logger.info(
                    f"  [{split}] complete after attempt {attempt}/{max_attempts}: "
                    f"{completion['completed']}/{completion['expected']} unique tasks "
                    f"({completion['errors']} error rows)"
                )
                continue

            if attempt < max_attempts:
                logger.warning(
                    f"  [{split}] partial after attempt {attempt}/{max_attempts}: "
                    f"{completion['completed']}/{completion['expected']} unique tasks "
                    f"({completion['errors']} error rows, "
                    f"{completion['incomplete']} incomplete started); relaunching split"
                )
                retry_splits.append(split)
            else:
                logger.error(
                    f"  [{split}] exhausted {max_attempts} attempt(s): "
                    f"{completion['completed']}/{completion['expected']} unique tasks "
                    f"({completion['errors']} error rows, "
                    f"{completion['incomplete']} incomplete started)"
                )

        splits_to_run = retry_splits
        if not splits_to_run:
            break

    # Match new study_dirs to splits. Two paths:
    # 1. cell-namespaced: each split's runner wrote to <cell_dir>/<split>/study_*,
    #    so we just glob there. No race with sibling streams.
    # 2. legacy global: scan results/browsergym/ for newly-created study_*
    #    and match by split substring (race-prone — see root cause 4).
    split_to_study: dict[str, Path] = {}
    if cell_dir is not None:
        for split in splits:
            split_root = cell_dir / split
            if not split_root.exists():
                continue
            # cell_dir/split IS the study dir (--results-dir == AgentLab exp_root).
            # Treat it as a study iff it actually contains trajectories.
            if any(split_root.rglob("summary_info.json")):
                split_to_study[split] = split_root
                continue
            # Tolerate older runner layouts that nested study_* inside.
            studies = sorted(split_root.glob("study_*"), key=lambda p: p.stat().st_mtime)
            if studies and any(studies[-1].rglob("summary_info.json")):
                split_to_study[split] = studies[-1]

    # Fallback to legacy discovery for any splits that didn't land in the
    # cell-owned namespace (e.g. older runners that ignore --results-dir).
    missing = [s for s in splits if s not in split_to_study]
    if missing:
        after = set((REPO_ROOT / "results" / "browsergym").glob("study_*"))
        new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
        legacy = _match_study_dirs_to_splits(new_dirs, missing)
        split_to_study.update(legacy)

    for split, study_dir in split_to_study.items():
        _write_run_meta(study_dir, args, benchmark, split)

    # Log any splits that ran but couldn't be matched to a study_dir
    for split in splits:
        if results.get(split) and split not in split_to_study:
            logger.warning(
                f"  [{split}] succeeded but no matching study_dir found; "
                f"checked {cell_dir}/{split}/ and {REPO_ROOT}/results/browsergym/."
            )

    return split_to_study


# Stagger Popen launches in parallel-splits mode by this many seconds. Avoids
# AgentLab's second-resolution `study_<TS>` timestamp colliding when 4
# subprocesses spawn within the same wallclock second.
_LAUNCH_STAGGER_SEC = 1.5


def _stage1_toolcalling(benchmark: str, split: str, args) -> Path | None:
    """Run AgentDojo or InjecAgent for one split. Returns the output dir."""
    runner = BENCHMARK_RUNNERS[benchmark]
    model_slug = args.model_name.replace("/", "_").replace(":", "_")
    frame_suffix = f"_{args.system_prompt_frame}" if args.system_prompt_frame != "none" else ""
    steer_suffix = (
        f"_alpha{args.steering_alpha:+.2f}_layer{args.steering_layer or 'best'}"
        if args.steering_alpha != 0.0
        else ""
    )
    run_name = (
        f"{args.condition}_{args.extra_instructions_preset}"
        f"{frame_suffix}{steer_suffix}_{model_slug}"
    )

    common_steer_flags: list[str] = []
    if args.backend != "auto":
        common_steer_flags += ["--backend", args.backend]
    if args.probe_dir:
        common_steer_flags += ["--probe-dir", args.probe_dir]
    if args.steering_layer is not None:
        common_steer_flags += ["--steering-layer", str(args.steering_layer)]
    if args.steering_alpha != 0.0:
        common_steer_flags += ["--steering-alpha", str(args.steering_alpha)]

    common_toolcalling_flags = [
        "--max-tokens",
        str(args.toolcalling_max_tokens),
    ]

    if benchmark == "agentdojo":
        cmd = [
            sys.executable,
            "-m",
            runner,
            "--suite",
            split,
            "--max-tasks",
            str(args.tasks_per_split),
            "--model-name",
            args.model_name,
            "--condition",
            args.condition,
            "--extra-instructions-preset",
            args.extra_instructions_preset,
            "--system-prompt-frame",
            args.system_prompt_frame,
            "--shuffle",
            str(args.shuffle),
            "--concurrency",
            str(args.concurrency),
            "--run-name",
            run_name,
            *common_toolcalling_flags,
            *common_steer_flags,
        ]
        out_dir = REPO_ROOT / "results" / "toolcalling" / "agentdojo" / split / run_name
    elif benchmark == "injecagent":
        attack_type, setting = split.split("_", 1)
        cmd = [
            sys.executable,
            "-m",
            runner,
            "--attack-type",
            attack_type,
            "--setting",
            setting,
            "--max-tasks",
            str(args.tasks_per_split),
            "--model-name",
            args.model_name,
            "--condition",
            args.condition,
            "--extra-instructions-preset",
            args.extra_instructions_preset,
            "--system-prompt-frame",
            args.system_prompt_frame,
            "--shuffle",
            str(args.shuffle),
            "--concurrency",
            str(args.concurrency),
            "--run-name",
            run_name,
            *common_toolcalling_flags,
            *common_steer_flags,
        ]
        out_dir = REPO_ROOT / "results" / "toolcalling" / "injecagent" / split / run_name
    else:
        raise ValueError(f"Not a tool-calling benchmark: {benchmark}")

    ok = _run_subprocess(cmd, cwd=REPO_ROOT)
    if not ok or not out_dir.exists():
        return None
    _write_run_meta(out_dir, args, benchmark, split)
    return out_dir


def _discover_browser(study_dir: Path) -> list[dict]:
    """Same logic as run_doomarena_pipeline.discover_trajectory_dirs but reusable."""
    trajectories: list[dict] = []
    for summary in sorted(study_dir.rglob("summary_info.json")):
        task_dir = summary.parent
        if _is_hidden_browser_path(study_dir, task_dir):
            continue
        steps = list(task_dir.glob("step_*.pkl.gz"))
        task_name = task_dir.name
        parts = task_name.rsplit("_on_", 1)
        task_id = parts[-1].rsplit("_", 1)[0] if len(parts) > 1 else task_name

        attack_success = None
        attack_info_path = task_dir / "attack_summary_info.json"
        if attack_info_path.exists():
            try:
                attack_success = json.loads(attack_info_path.read_text()).get("attack_successful")
            except (json.JSONDecodeError, OSError):
                pass

        task_reward = None
        try:
            task_reward = json.loads(summary.read_text()).get("cum_reward")
        except (json.JSONDecodeError, OSError):
            pass

        trajectories.append(
            {
                "task_dir": str(task_dir),
                "task_id": task_id,
                "n_steps": len(steps),
                "attack_success": attack_success,
                "task_reward": task_reward,
            }
        )
    return trajectories


def _attach_wasp_asr_to_trajectories(
    trajectories: list[dict],
    asr_by_task: dict[str, bool],
) -> int:
    """Attach post-hoc WASP ASR labels before judging."""
    n_attached = 0
    for t in trajectories:
        task_id = str(t.get("task_id", ""))
        if task_id not in asr_by_task:
            continue
        t["attack_success"] = asr_by_task[task_id]
        t["attack_success_source"] = "wasp_post_hoc_action_classifier"
        n_attached += 1
    return n_attached


def _result_task_id(row: dict) -> str:
    metadata = row.get("metadata") or {}
    task_id = metadata.get("task_id")
    if task_id:
        return str(task_id)
    sample_id = str(row.get("sample_id") or "")
    return sample_id.rsplit("_", 1)[-1] if sample_id else ""


def _annotate_wasp_judge_results(
    *,
    output_dir: Path,
    split: str,
    asr_by_task: dict[str, bool],
    asr_summary: dict | None = None,
) -> dict:
    """Backfill WASP ASR into judge JSONL rows without rerunning judges."""
    if not asr_by_task:
        return {"updated_rows": 0, "path": None}

    result_path = output_dir / "wasp" / split / "trajectory_awareness_results.jsonl"
    if not result_path.exists():
        return {"updated_rows": 0, "path": str(result_path), "missing": True}

    asr_summary = asr_summary or {}
    classifier_model = asr_summary.get("classifier_model")
    asr_path = asr_summary.get("path")

    changed = False
    updated_rows = 0
    out_lines: list[str] = []
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                out_lines.append(line)
                continue
            row = json.loads(line)
            task_id = _result_task_id(row)
            if task_id in asr_by_task:
                metadata = row.setdefault("metadata", {})
                before = dict(metadata)
                metadata["task_id"] = task_id
                metadata["attack_success"] = asr_by_task[task_id]
                metadata["attack_success_source"] = "wasp_post_hoc_action_classifier"
                if classifier_model:
                    metadata["asr_classifier_model"] = classifier_model
                if asr_path:
                    metadata["wasp_asr_path"] = asr_path
                if metadata != before:
                    changed = True
                updated_rows += 1
            out_lines.append(json.dumps(row) + "\n")

    if changed:
        tmp_path = result_path.with_name(result_path.name + ".tmp")
        tmp_path.write_text("".join(out_lines), encoding="utf-8")
        tmp_path.replace(result_path)

    return {"updated_rows": updated_rows, "path": str(result_path), "changed": changed}


def _discover_toolcalling(run_dir: Path) -> list[dict]:
    """Each per-task JSON is one trajectory.

    Skips metadata files (run_meta.json, *_manifest.json) — those live next
    to trajectories but aren't trajectories themselves. Without this filter
    the judges would treat run_meta.json as a fake trajectory and produce
    garbage classification rows in the output JSONL.
    """
    # Files written by the pipeline that share the dir but aren't trajectories.
    METADATA_FILES = {"run_meta.json"}
    trajectories: list[dict] = []
    for fp in sorted(run_dir.glob("*.json")):
        if fp.name in METADATA_FILES or fp.name.endswith("_manifest.json"):
            continue
        try:
            data = json.loads(fp.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed {fp}")
            continue
        # Defensive: trajectories must have a `steps` field. Skip anything
        # that doesn't look like a trajectory (e.g. an unknown future
        # metadata file we forgot to add to METADATA_FILES).
        if "steps" not in data:
            logger.debug(f"Skipping non-trajectory JSON {fp.name}")
            continue
        trajectories.append(
            {
                "task_dir": str(fp),  # parse_toolcalling_trajectory accepts file or dir
                "task_id": data.get("task_id", fp.stem),
                "n_steps": len(data.get("steps", [])),
                "attack_success": data.get("attack_success"),
                "task_reward": data.get("passed"),
            }
        )
    return trajectories


def _discover_eia(data_dir: Path) -> list[dict]:
    """Read the manifest produced by import_eia_trajectories.py."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run "
            "`python -m eval_awareness_experiments.import_eia_trajectories` first."
        )
    manifest = json.loads(manifest_path.read_text())
    return [
        {
            "task_dir": str(data_dir / entry["path"]),
            "task_id": entry["task_id"],
            "n_steps": entry.get("n_steps", 0),
            "attack_success": entry.get("attack_success"),
            "task_reward": None,
        }
        for entry in manifest.get("tasks", [])
    ]


async def _judge_split(
    *,
    benchmark: str,
    split: str,
    trajectories: list[dict],
    judge_model: str,
    judge_names: list[str],
    agent_model: str,
    condition: str,
    extra_instructions_preset: str,
    system_prompt_frame: str,
    output_dir: Path,
    purpose_continuation_model_source: str,
    judge_concurrency: int,
    purpose_continuation_concurrency: int,
    judge_retries: int,
    purpose_continuation_retries: int,
) -> None:
    from eval_awareness_experiments.experiments.trajectory_awareness import (
        TrajectoryAwarenessExperiment,
    )
    from eval_awareness_experiments.llm import LLM
    from eval_awareness_experiments.types import WebsiteSample

    if not trajectories:
        logger.warning(f"  No trajectories for {benchmark}/{split}")
        return

    samples = [
        WebsiteSample(
            id=f"{benchmark}_{split}_{t['task_id']}",
            # `benchmark` reaches here from `--benchmark`, which argparse
            # restricts to ALL_BENCHMARKS; every member of that tuple is one of
            # WebsiteSample.source's literals, so this is valid at runtime and
            # mypy simply cannot see the argparse constraint. Note for
            # maintainers: adding a benchmark to ALL_BENCHMARKS without adding
            # it to that Literal would make this suppression hide a real
            # provenance mismatch.
            source=benchmark,  # type: ignore[arg-type]
            website_type=split,
            metadata={
                "trajectory_dir": t["task_dir"],
                "benchmark": benchmark,
                "agent": agent_model,
                "task_id": t["task_id"],
                "is_trajectory": True,
                "attack_success": t.get("attack_success"),
                "attack_success_source": t.get("attack_success_source"),
                "task_reward": t.get("task_reward"),
                "condition": condition,
                "extra_instructions_preset": extra_instructions_preset,
                "system_prompt_frame": system_prompt_frame,
                "task_info": {
                    "task_id": t["task_id"],
                    "attack_success": t.get("attack_success"),
                    "attack_success_source": t.get("attack_success_source"),
                },
            },
        )
        for t in trajectories
    ]
    logger.info(f"  Judging {len(samples)} {benchmark}/{split} trajectories with {judge_names}")

    model = LLM(
        judge_model,
        temperature=0.0,
        concurrency=judge_concurrency,
        retries=judge_retries,
    )
    exp = TrajectoryAwarenessExperiment(
        model=model,
        output_dir=output_dir / benchmark / split,
        judge_names=judge_names,
        purpose_continuation_model_source=purpose_continuation_model_source,
        purpose_continuation_concurrency=purpose_continuation_concurrency,
        purpose_continuation_retries=purpose_continuation_retries,
    )
    await exp.run(samples, ["trajectory"])
    logger.info(f"  Judge results saved → {output_dir / benchmark / split}")

    # Sidecar: human-readable .md per trajectory, pairing transcript + judge rows.
    # Only fires for tool-calling benchmarks (agentdojo / injecagent) where each
    # trajectory is a single JSON file. Browser benchmarks use a different
    # layout (agentlab pickle dirs) — skip them for now.
    try:
        from eval_awareness_experiments.markdown_export import write_markdown_for_split

        first_task = Path(trajectories[0]["task_dir"]) if trajectories else None
        if first_task and first_task.is_file() and first_task.suffix == ".json":
            md_dir = output_dir / benchmark / split / "markdown"
            write_markdown_for_split(
                transcript_dir=first_task.parent,
                judge_jsonl=output_dir / benchmark / split / "trajectory_awareness_results.jsonl",
                out_dir=md_dir,
            )
            logger.info(f"  Markdown sidecars → {md_dir}")
    except Exception as e:
        logger.warning(f"  [md] markdown export failed (non-fatal): {e!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified safety-benchmark pipeline.")
    parser.add_argument("--benchmark", required=True, choices=ALL_BENCHMARKS)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Per-benchmark split list. " f"Defaults: {DEFAULT_SPLITS}",
    )
    parser.add_argument("--stage", choices=["all", "run-only", "judge-only"], default="all")
    parser.add_argument("--tasks-per-split", type=int, default=20)
    parser.add_argument(
        "--wasp-task-dir",
        default=None,
        help="WASP planted-task pool directory (default: /tmp/wasp_full, "
        "matching scripts/wasp_n100_run.sh). Only used when "
        "--benchmark wasp.",
    )
    parser.add_argument(
        "--skip-wasp-asr",
        action="store_true",
        help="WASP only: do not run the post-hoc ASR classifier "
        "before trajectory-awareness judging.",
    )
    parser.add_argument(
        "--wasp-asr-classifier-model",
        default="anthropic/claude-sonnet-4.6",
        help="WASP only: OpenRouter model for the post-hoc " "action-only ASR classifier.",
    )
    parser.add_argument(
        "--wasp-asr-concurrency",
        type=int,
        default=32,
        help="WASP only: concurrent ASR classifier calls.",
    )
    parser.add_argument(
        "--wasp-asr-max-steps-per-task",
        type=int,
        default=10,
        help="WASP only: max steps classified per task.",
    )
    parser.add_argument(
        "--wasp-asr-force",
        action="store_true",
        help="WASP only: recompute wasp_asr_<split>.jsonl even " "when it already exists.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Per-task parallelism for the agent stage (tool-calling benchmarks). "
        "Forwarded as --concurrency to run_agentdojo / run_injecagent. "
        "Bottleneck is target-model TPM via OpenRouter, not local CPU.",
    )
    parser.add_argument(
        "--avg-step-timeout",
        type=int,
        default=60,
        help="Forwarded to browser runners. AgentLab's per-task "
        "episode timeout is max_steps × avg_step_timeout.",
    )
    parser.add_argument(
        "--browser-stage1-timeout",
        type=int,
        default=None,
        help="Per-(model × split) timeout in seconds for browser benchmarks "
        "(DoomArena, WASP). When omitted, defaults to "
        "tasks_per_split × max_steps × avg_step_timeout + "
        "browser_stage1_overhead. Tool-calling benchmarks ignore this flag.",
    )
    parser.add_argument(
        "--browser-stage1-overhead",
        type=int,
        default=1800,
        help="Extra seconds added to the computed browser split timeout "
        "when --browser-stage1-timeout is omitted. Default 1800s.",
    )
    parser.add_argument(
        "--browser-stage1-idle-timeout",
        type=int,
        default=3600,
        help="Kill a browser split if no subprocess log, step, "
        "summary, or aggregate result file changes for this "
        "many seconds. Use 0 to disable. Default 3600s.",
    )
    parser.add_argument(
        "--browser-stage1-relaunch-attempts",
        type=int,
        default=3,
        help="Total attempts per browser split, including the "
        "first launch. If a split is killed or remains "
        "partial, relaunch it with --relaunch-incomplete "
        "before judging. Use 1 for legacy behavior. "
        "Default 3.",
    )
    parser.add_argument(
        "--browser-splits-sequential",
        action="store_true",
        help="Force browser-benchmark splits to run sequentially within a "
        "cell (legacy behavior). Default is parallel — splits hit "
        "different docker services so they don't contend, giving ~4× "
        "speedup on DoomArena (4 splits) and ~2× on WASP (2 splits).",
    )
    parser.add_argument(
        "--browser-relaunch-incomplete",
        action="store_true",
        help="Browser benchmarks only: resume the latest AgentLab study "
        "under each split's --results-dir, rerunning incomplete "
        "or errored tasks instead of creating a fresh study.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Agent model for stage 1. Required unless --stage judge-only.",
    )
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default="baseline")
    parser.add_argument(
        "--extra-instructions-preset",
        default="none",
        choices=list(EXTRA_INSTRUCTION_PRESETS.keys()),
    )
    parser.add_argument(
        "--system-prompt-frame",
        default="none",
        choices=ALL_FRAMES,
        help="Wrap the entire system prompt in XML tags. "
        "Tool-calling benchmarks only (browser track not yet supported).",
    )
    # Steering flags — only meaningful for the tool-calling benchmarks
    # (agentdojo, injecagent). Forwarded verbatim to the underlying runner.
    # Browser benchmarks (wasp, doomarena) ignore them since the AgentLab
    # browser path runs through vLLM/OpenRouter and has no HF-hook harness.
    parser.add_argument(
        "--backend",
        choices=("auto", "openai", "hf"),
        default="auto",
        help="auto picks 'hf' iff --steering-alpha != 0; otherwise 'openai'. "
        "Tool-calling benchmarks only.",
    )
    parser.add_argument(
        "--probe-dir",
        default=None,
        help="Path to a saved probe (probes/trained/<model>/). Required "
        "when --steering-alpha != 0. Tool-calling benchmarks only.",
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="Decoder layer for steering. Defaults to probe.best_layer. "
        "Tool-calling benchmarks only.",
    )
    parser.add_argument(
        "--steering-alpha",
        type=float,
        default=0.0,
        help="Steering strength. Negative = away from eval-aware. "
        "0 disables steering. Tool-calling benchmarks only.",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--toolcalling-max-tokens",
        type=int,
        default=8192,
        help="Maximum model output tokens per turn for "
        "AgentDojo/InjecAgent. Browser benchmarks ignore "
        "this flag.",
    )
    parser.add_argument("--shuffle", type=int, default=42)
    parser.add_argument(
        "--report-port",
        type=int,
        default=1234,
        help="Only used by DoomArena runner. Legacy single-port "
        "default; ignored when --report-port-base is given "
        "or PID-derived auto-base is computed.",
    )
    parser.add_argument(
        "--report-port-base",
        type=int,
        default=None,
        help="DoomArena: base port for the cell. Each split "
        "gets base+offset (reddit=0, shopping=1, "
        "shopping_admin=2, gitlab=3). Default: PID-derived "
        "(unique per process).",
    )
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.6:thinking")
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=8,
        help="Max concurrent classifier/judge model calls during " "judge-only and judge stages.",
    )
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=3,
        help="Transport retries for classifier/judge model calls.",
    )
    parser.add_argument(
        "--purpose-continuation-model-source",
        choices=["agent", "judge"],
        default="agent",
        help="For purpose_continue(_5q), which model answers "
        "the appended follow-up questions. 'agent' uses "
        "the original trajectory model and --judge-model "
        "only classifies the answers; 'judge' preserves "
        "the legacy behavior where the judge model also "
        "answers the questions.",
    )
    parser.add_argument(
        "--purpose-continuation-concurrency",
        type=int,
        default=8,
        help="Max concurrent original-model calls for " "purpose_continue(_5q) follow-up answers.",
    )
    parser.add_argument(
        "--purpose-continuation-retries",
        type=int,
        default=3,
        help="Transport retries for original-model " "purpose_continue(_5q) follow-up answers.",
    )
    parser.add_argument(
        "--judges", nargs="+", default=["test_awareness_steering", "verbalized_awareness"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval_awareness_experiments" / "results" / "safety_pipeline",
    )
    parser.add_argument(
        "--existing-dirs",
        nargs="+",
        default=None,
        help="For --stage judge-only: explicit trajectory roots to judge, "
        "one per split. Format: 'split:path'.",
    )
    return parser.parse_args()


def _resolve_splits(benchmark: str, splits: list[str] | None) -> list[str]:
    if splits:
        return splits
    return list(DEFAULT_SPLITS[benchmark])


def _validate_args(args: argparse.Namespace) -> None:
    if args.benchmark == "eia":
        if args.stage == "run-only":
            raise SystemExit("EIA has no run stage — use --stage judge-only or all.")
        if args.condition != "baseline":
            logger.warning("EIA: forcing condition=baseline (imported trajectories).")
            args.condition = "baseline"
        if args.system_prompt_frame != "none":
            logger.warning("EIA: forcing system_prompt_frame=none (imported trajectories).")
            args.system_prompt_frame = "none"
    if args.stage != "judge-only" and args.benchmark != "eia" and not args.model_name:
        raise SystemExit("--model-name is required unless --stage judge-only or benchmark=eia.")
    # Browser-track frame support is now wired through agent_factory:
    # `build_openrouter_agent` wraps `extra_instructions` with the XML frame
    # (rendered in AgentLab's "## Extra instructions:" section). Less literal
    # than the tool-calling track's full-system-prompt wrap, but the model
    # sees the frame tag clearly. See agent_factory._build_framed_extra_instructions.


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    _validate_args(args)
    splits = _resolve_splits(args.benchmark, args.splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pipeline: benchmark={args.benchmark} splits={splits} stage={args.stage}")

    manifest: dict = {
        "started": datetime.now(timezone.utc).isoformat(),
        "config": {**vars(args), "output_dir": str(args.output_dir)},
        "splits": {},
    }

    # Map split → trajectory root (either study_dir, run_dir, or data_dir).
    split_to_root: dict[str, Path] = {}

    # Parse --existing-dirs (judge-only path).
    if args.stage == "judge-only" and args.existing_dirs:
        for entry in args.existing_dirs:
            split_name, _, path_str = entry.partition(":")
            if not path_str:
                raise SystemExit(f"--existing-dirs entry {entry!r} must be 'split:path'")
            split_to_root[split_name] = Path(path_str)

    # Stage 1.
    if args.stage in ("all", "run-only"):
        # Browser benchmarks: run all splits in PARALLEL via Popen by default
        # (4 splits per cell for DoomArena, 2 for WASP). Splits hit different
        # docker services so no contention; ~4× speedup. Sequential mode
        # available via --browser-splits-sequential for debugging.
        if args.benchmark in BROWSER_BENCHMARKS and not getattr(
            args, "browser_splits_sequential", False
        ):
            logger.info(f"=== Stage 1: {args.benchmark}/{splits} (PARALLEL splits) ===")
            split_to_study = _stage1_browser_parallel_splits(args.benchmark, splits, args)
            for split in splits:
                study = split_to_study.get(split)
                if study:
                    split_to_root[split] = study
                    manifest["splits"].setdefault(split, {})["study_dir"] = str(study)
                else:
                    manifest["splits"].setdefault(split, {})["status"] = "run_failed"
        else:
            for split in splits:
                logger.info(f"=== Stage 1: {args.benchmark}/{split} ===")
                if args.benchmark in BROWSER_BENCHMARKS:
                    study = _stage1_browser(args.benchmark, split, args)
                    if study:
                        split_to_root[split] = study
                        manifest["splits"].setdefault(split, {})["study_dir"] = str(study)
                    else:
                        manifest["splits"].setdefault(split, {})["status"] = "run_failed"
                elif args.benchmark in TOOLCALLING_BENCHMARKS:
                    run = _stage1_toolcalling(args.benchmark, split, args)
                    if run:
                        split_to_root[split] = run
                        manifest["splits"].setdefault(split, {})["run_dir"] = str(run)
                    else:
                        manifest["splits"].setdefault(split, {})["status"] = "run_failed"
                elif args.benchmark in IMPORT_ONLY_BENCHMARKS:
                    split_to_root[split] = Path(__file__).parent / "data" / "eia"
                    manifest["splits"].setdefault(split, {})["run_dir"] = str(split_to_root[split])
                else:
                    raise ValueError(f"Unknown benchmark {args.benchmark}")

    if args.stage == "run-only":
        _write_manifest(manifest, args.output_dir)
        return

    # For judge-only on EIA/missing roots, fall back to default locations.
    # Browser benchmark cells keep their AgentLab trajectory roots under the
    # cell output dir, so `run_causal_experiment --stage judge-only` can
    # rejudge a whole matrix without manually spelling out --existing-dirs.
    for split in splits:
        if split in split_to_root:
            continue
        if args.benchmark == "eia":
            split_to_root[split] = Path(__file__).parent / "data" / "eia"
        elif args.benchmark in BROWSER_BENCHMARKS:
            root = _browser_judge_only_root(args, split)
            if root is not None:
                split_to_root[split] = root
                manifest["splits"].setdefault(split, {})["study_dir"] = str(root)
            else:
                logger.warning(
                    f"No trajectory root known for {args.benchmark}/{split} "
                    f"in judge-only mode. Expected {args.output_dir / '_browser_runs' / split} "
                    f"or pass --existing-dirs {split}:PATH."
                )
        elif args.benchmark in TOOLCALLING_BENCHMARKS:
            model_slug = (args.model_name or "").replace("/", "_").replace(":", "_")
            frame_suffix = (
                f"_{args.system_prompt_frame}" if args.system_prompt_frame != "none" else ""
            )
            run_name = (
                f"{args.condition}_{args.extra_instructions_preset}{frame_suffix}_{model_slug}"
            )
            if args.benchmark == "agentdojo":
                split_to_root[split] = (
                    REPO_ROOT / "results" / "toolcalling" / "agentdojo" / split / run_name
                )
            else:
                split_to_root[split] = (
                    REPO_ROOT / "results" / "toolcalling" / "injecagent" / split / run_name
                )
        else:
            logger.warning(
                f"No trajectory root known for {args.benchmark}/{split} in judge-only mode."
            )

    # Recover original-run config per split. For stage=all/run-only we just
    # wrote run_meta.json so this round-trips to args; for stage=judge-only on
    # existing dirs this is what avoids the "model=None preset=none" leak.
    split_to_meta: dict[str, dict] = {}
    for split in splits:
        root = split_to_root.get(split)
        if root is None:
            continue
        recovered = _load_run_meta(Path(root))
        if recovered is not None:
            split_to_meta[split] = recovered
        else:
            split_to_meta[split] = {
                "model_name": args.model_name,
                "condition": args.condition,
                "extra_instructions_preset": args.extra_instructions_preset,
                "_recovered_from": "cli_args_fallback",
            }
        manifest["splits"].setdefault(split, {})["original_run_meta"] = split_to_meta[split]

    # Stage 2.
    wasp_asr_llm = None

    async def judge_all():
        nonlocal wasp_asr_llm
        for split in splits:
            root = split_to_root.get(split)
            if root is None or not Path(root).exists():
                logger.warning(f"Skipping {split}: no trajectory root ({root})")
                continue
            logger.info(f"=== Stage 2: {args.benchmark}/{split} ← {root} ===")

            if args.benchmark in BROWSER_BENCHMARKS:
                trajectories = _discover_browser(root)
            elif args.benchmark in TOOLCALLING_BENCHMARKS:
                trajectories = _discover_toolcalling(root)
            else:  # eia
                trajectories = _discover_eia(root)

            meta = split_to_meta.get(split, {})
            manifest["splits"].setdefault(split, {})["n_trajectories"] = len(trajectories)

            wasp_asr_by_task: dict[str, bool] = {}
            wasp_asr_summary: dict | None = None
            if args.benchmark == "wasp" and not args.skip_wasp_asr:
                from eval_awareness_experiments.llm import LLM
                from eval_awareness_experiments.wasp_post_hoc_evaluator import (
                    evaluate_wasp_cell_split,
                    load_wasp_asr_map,
                )

                if wasp_asr_llm is None:
                    wasp_asr_llm = LLM(
                        model=args.wasp_asr_classifier_model,
                        thinking="thinking" in args.wasp_asr_classifier_model,
                        concurrency=args.wasp_asr_concurrency,
                        temperature=0.0,
                    )

                cell = {
                    "arm": args.output_dir.parent.name,
                    "model_slug": args.output_dir.name,
                    "split": split,
                    "study_dir": str(root),
                    "cell_dir": str(args.output_dir),
                }
                logger.info(
                    f"=== WASP ASR: {split} ← {root} " f"({args.wasp_asr_classifier_model}) ==="
                )
                wasp_asr_summary = await evaluate_wasp_cell_split(
                    cell=cell,
                    llm=wasp_asr_llm,
                    classifier_model=args.wasp_asr_classifier_model,
                    max_steps_per_task=args.wasp_asr_max_steps_per_task,
                    max_tasks_per_cell=None,
                    force=args.wasp_asr_force,
                )
                wasp_asr_by_task = load_wasp_asr_map(args.output_dir, split)
                n_attached = _attach_wasp_asr_to_trajectories(trajectories, wasp_asr_by_task)
                manifest["splits"][split]["wasp_asr"] = {
                    "path": wasp_asr_summary.get("path"),
                    "classifier_model": wasp_asr_summary.get("classifier_model"),
                    "n_tasks": wasp_asr_summary.get("n_tasks"),
                    "n_compromised": wasp_asr_summary.get("n_compromised"),
                    "asr": wasp_asr_summary.get("asr"),
                    "n_attached_to_trajectories": n_attached,
                }

            await _judge_split(
                benchmark=args.benchmark,
                split=split,
                trajectories=trajectories,
                judge_model=args.judge_model,
                judge_names=args.judges,
                agent_model=meta.get("model_name") or args.model_name or "imported",
                condition=meta.get("condition") or args.condition,
                extra_instructions_preset=(
                    meta.get("extra_instructions_preset") or args.extra_instructions_preset
                ),
                system_prompt_frame=(meta.get("system_prompt_frame") or args.system_prompt_frame),
                output_dir=args.output_dir,
                purpose_continuation_model_source=args.purpose_continuation_model_source,
                judge_concurrency=args.judge_concurrency,
                purpose_continuation_concurrency=args.purpose_continuation_concurrency,
                judge_retries=args.judge_retries,
                purpose_continuation_retries=args.purpose_continuation_retries,
            )
            if args.benchmark == "wasp" and wasp_asr_by_task:
                annotation = _annotate_wasp_judge_results(
                    output_dir=args.output_dir,
                    split=split,
                    asr_by_task=wasp_asr_by_task,
                    asr_summary=wasp_asr_summary,
                )
                manifest["splits"][split]["wasp_asr_annotation"] = annotation
                logger.info(
                    f"  WASP ASR annotations → {annotation.get('updated_rows')} rows "
                    f"in {annotation.get('path')}"
                )
            expected_total = int(getattr(args, "tasks_per_split", 0) or 0)
            if (
                args.benchmark in TOOLCALLING_BENCHMARKS
                and expected_total
                and len(trajectories) < expected_total
            ):
                manifest["splits"][split][
                    "status"
                ] = f"partial_judges_complete ({len(trajectories)}/{expected_total})"
                logger.warning(
                    f"  {args.benchmark}/{split} judged only "
                    f"{len(trajectories)}/{expected_total} trajectories"
                )
            else:
                manifest["splits"][split]["status"] = "judges_complete"

    asyncio.run(judge_all())
    _write_manifest(manifest, args.output_dir)


def _write_manifest(manifest: dict, output_dir: Path) -> None:
    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    path = output_dir / f"pipeline_manifest_{manifest['config'].get('benchmark')}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info(f"Pipeline manifest → {path}")


if __name__ == "__main__":
    main()
