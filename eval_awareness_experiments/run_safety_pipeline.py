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
import subprocess
import sys
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


def _run_subprocess(cmd: list[str], *, cwd: Path, env: dict | None = None) -> bool:
    env = env or os.environ.copy()
    logger.info("  $ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    for line in (result.stdout or "").splitlines()[-30:]:
        logger.info(f"  [stdout] {line}")
    for line in (result.stderr or "").splitlines()[-15:]:
        logger.warning(f"  [stderr] {line}")
    if result.returncode != 0:
        logger.error(f"  subprocess failed: exit={result.returncode}")
    return result.returncode == 0


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
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    except OSError as e:
        logger.warning(f"  Failed to write run_meta.json to {run_dir}: {e}")


_KNOWN_CONDITIONS = ("baseline",) + tuple(c for c in ALL_CONDITIONS if c != "baseline")


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
        rest = name[len(cond_prefix):]
        # Match the longest preset name first so `scratchpad_bare` wins over `scratchpad`.
        for preset in sorted(EXTRA_INSTRUCTION_PRESETS.keys(), key=len, reverse=True):
            preset_prefix = f"{preset}_"
            if rest.startswith(preset_prefix):
                model_slug = rest[len(preset_prefix):]
                return {
                    "condition": cond,
                    "extra_instructions_preset": preset,
                    # Reverse of `model_name.replace("/", "_").replace(":", "_")`
                    # is lossy — return slug as-is, marked.
                    "model_name": model_slug,
                    "_recovered_from": "run_dir_name",
                }
    return None


def _stage1_browser(benchmark: str, split: str, args) -> Path | None:
    """Run DoomArena or WASP for one split. Returns study dir."""
    runner = BENCHMARK_RUNNERS[benchmark]
    cmd = [
        sys.executable, "-m", runner,
        "--single-site", split,
        "--max-tasks", str(args.tasks_per_split),
        "--model-name", args.model_name,
        "--extra-instructions-preset", args.extra_instructions_preset,
        "--max-steps", str(args.max_steps),
        "--shuffle", str(args.shuffle),
    ]
    # --condition (causal injection wrapping: baseline/eval_cue/scenario_cue) is
    # only implemented in WASP's runner. DoomArena uses runtime AttackConfigs
    # instead, so it doesn't accept the flag.
    if benchmark == "wasp":
        cmd += ["--condition", args.condition]
        # WASP needs the planted-task pool dir. Default matches
        # scripts/wasp_n100_run.sh (the per-benchmark launcher).
        wasp_task_dir = getattr(args, "wasp_task_dir", None) or "/tmp/wasp_full"
        cmd += ["--task-dir", wasp_task_dir]
    if benchmark == "doomarena" and getattr(args, "report_port", None):
        cmd += ["--report-port", str(args.report_port)]

    before = set((REPO_ROOT / "results" / "browsergym").glob("study_*"))
    ok = _run_subprocess(cmd, cwd=REPO_ROOT)
    if not ok:
        return None
    after = set((REPO_ROOT / "results" / "browsergym").glob("study_*"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new:
        return None
    study_dir = new[-1]
    _write_run_meta(study_dir, args, benchmark, split)
    return study_dir


def _stage1_toolcalling(benchmark: str, split: str, args) -> Path | None:
    """Run AgentDojo or InjecAgent for one split. Returns the output dir."""
    runner = BENCHMARK_RUNNERS[benchmark]
    model_slug = args.model_name.replace("/", "_").replace(":", "_")
    frame_suffix = (
        f"_{args.system_prompt_frame}"
        if args.system_prompt_frame != "none"
        else ""
    )
    run_name = f"{args.condition}_{args.extra_instructions_preset}{frame_suffix}_{model_slug}"

    if benchmark == "agentdojo":
        cmd = [
            sys.executable, "-m", runner,
            "--suite", split,
            "--max-tasks", str(args.tasks_per_split),
            "--model-name", args.model_name,
            "--condition", args.condition,
            "--extra-instructions-preset", args.extra_instructions_preset,
            "--system-prompt-frame", args.system_prompt_frame,
            "--shuffle", str(args.shuffle),
            "--concurrency", str(args.concurrency),
            "--run-name", run_name,
        ]
        out_dir = REPO_ROOT / "results" / "toolcalling" / "agentdojo" / split / run_name
    elif benchmark == "injecagent":
        attack_type, setting = split.split("_", 1)
        cmd = [
            sys.executable, "-m", runner,
            "--attack-type", attack_type,
            "--setting", setting,
            "--max-tasks", str(args.tasks_per_split),
            "--model-name", args.model_name,
            "--condition", args.condition,
            "--extra-instructions-preset", args.extra_instructions_preset,
            "--system-prompt-frame", args.system_prompt_frame,
            "--shuffle", str(args.shuffle),
            "--concurrency", str(args.concurrency),
            "--run-name", run_name,
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

        trajectories.append({
            "task_dir": str(task_dir),
            "task_id": task_id,
            "n_steps": len(steps),
            "attack_success": attack_success,
            "task_reward": task_reward,
        })
    return trajectories


def _discover_toolcalling(run_dir: Path) -> list[dict]:
    """Each per-task JSON is one trajectory."""
    trajectories: list[dict] = []
    for fp in sorted(run_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed {fp}")
            continue
        trajectories.append({
            "task_dir": str(fp),  # parse_toolcalling_trajectory accepts file or dir
            "task_id": data.get("task_id", fp.stem),
            "n_steps": len(data.get("steps", [])),
            "attack_success": data.get("attack_success"),
            "task_reward": data.get("passed"),
        })
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
) -> None:
    from eval_awareness_experiments.llm import LLM
    from eval_awareness_experiments.experiments.trajectory_awareness import TrajectoryAwarenessExperiment
    from eval_awareness_experiments.types import WebsiteSample

    if not trajectories:
        logger.warning(f"  No trajectories for {benchmark}/{split}")
        return

    samples = [
        WebsiteSample(
            id=f"{benchmark}_{split}_{t['task_id']}",
            source=benchmark,
            website_type=split,
            metadata={
                "trajectory_dir": t["task_dir"],
                "benchmark": benchmark,
                "agent": agent_model,
                "task_id": t["task_id"],
                "is_trajectory": True,
                "attack_success": t.get("attack_success"),
                "task_reward": t.get("task_reward"),
                "condition": condition,
                "extra_instructions_preset": extra_instructions_preset,
                "system_prompt_frame": system_prompt_frame,
            },
        )
        for t in trajectories
    ]
    logger.info(f"  Judging {len(samples)} {benchmark}/{split} trajectories with {judge_names}")

    model = LLM(judge_model)
    exp = TrajectoryAwarenessExperiment(
        model=model,
        output_dir=output_dir / benchmark / split,
        judge_names=judge_names,
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
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Per-benchmark split list. "
                             f"Defaults: {DEFAULT_SPLITS}")
    parser.add_argument("--stage", choices=["all", "run-only", "judge-only"], default="all")
    parser.add_argument("--tasks-per-split", type=int, default=20)
    parser.add_argument("--wasp-task-dir", default=None,
                        help="WASP planted-task pool directory (default: /tmp/wasp_full, "
                             "matching scripts/wasp_n100_run.sh). Only used when "
                             "--benchmark wasp.")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Per-task parallelism for the agent stage (tool-calling benchmarks). "
                             "Forwarded as --concurrency to run_agentdojo / run_injecagent. "
                             "Bottleneck is target-model TPM via OpenRouter, not local CPU.")
    parser.add_argument("--model-name", default=None,
                        help="Agent model for stage 1. Required unless --stage judge-only.")
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default="baseline")
    parser.add_argument("--extra-instructions-preset", default="none",
                        choices=list(EXTRA_INSTRUCTION_PRESETS.keys()))
    parser.add_argument("--system-prompt-frame", default="none", choices=ALL_FRAMES,
                        help="Wrap the entire system prompt in XML tags. "
                             "Tool-calling benchmarks only (browser track not yet supported).")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--shuffle", type=int, default=42)
    parser.add_argument("--report-port", type=int, default=1234,
                        help="Only used by DoomArena runner.")
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.6:thinking")
    parser.add_argument("--judges", nargs="+",
                        default=["test_awareness_steering", "verbalized_awareness"])
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "eval_awareness_experiments" / "results" /
                                "safety_pipeline")
    parser.add_argument("--existing-dirs", nargs="+", default=None,
                        help="For --stage judge-only: explicit trajectory roots to judge, "
                             "one per split. Format: 'split:path'.")
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
                # Stage 1 is a no-op for EIA.
                split_to_root[split] = Path(__file__).parent / "data" / "eia"
                manifest["splits"].setdefault(split, {})["run_dir"] = str(split_to_root[split])
            else:
                raise ValueError(f"Unknown benchmark {args.benchmark}")

    if args.stage == "run-only":
        _write_manifest(manifest, args.output_dir)
        return

    # For judge-only on EIA/missing roots, fall back to default locations.
    for split in splits:
        if split in split_to_root:
            continue
        if args.benchmark == "eia":
            split_to_root[split] = Path(__file__).parent / "data" / "eia"
        elif args.benchmark in TOOLCALLING_BENCHMARKS:
            model_slug = (args.model_name or "").replace("/", "_").replace(":", "_")
            frame_suffix = (
                f"_{args.system_prompt_frame}"
                if args.system_prompt_frame != "none"
                else ""
            )
            run_name = f"{args.condition}_{args.extra_instructions_preset}{frame_suffix}_{model_slug}"
            if args.benchmark == "agentdojo":
                split_to_root[split] = REPO_ROOT / "results" / "toolcalling" / "agentdojo" / split / run_name
            else:
                split_to_root[split] = REPO_ROOT / "results" / "toolcalling" / "injecagent" / split / run_name
        else:
            logger.warning(f"No trajectory root known for {args.benchmark}/{split} in judge-only mode.")

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
    async def judge_all():
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
                system_prompt_frame=(
                    meta.get("system_prompt_frame") or args.system_prompt_frame
                ),
                output_dir=args.output_dir,
            )
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
