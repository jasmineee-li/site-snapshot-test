"""Orchestrate causal-injection experiments across benchmarks × conditions × models.

Thin matrix wrapper around `run_safety_pipeline`. For each combo it shells out
to one pipeline invocation — the pipeline itself knows how to dispatch to the
right runner (browser / tool-calling / import-only) and then apply judges.

Usage:
    # Full pilot: 3 models × 3 benchmarks × 3 conditions = 27 combos
    python -m eval_awareness_experiments.run_causal_experiment \
        --models anthropic/claude-opus-4.7:thinking openai/gpt-5.4:thinking \
        --benchmarks doomarena wasp agentdojo injecagent \
        --conditions baseline eval_cue scenario_cue \
        --extra-instructions-presets motive \
        --tasks-per-split 20

    # System-prompt-frame ceiling sweep on tool-calling: wrap the whole
    # system prompt in <llm_safety_test> / <hypothetical_scenario> tags.
    python -m eval_awareness_experiments.run_causal_experiment \
        --benchmarks agentdojo \
        --models z-ai/glm-5:thinking \
        --conditions baseline \
        --extra-instructions-presets none \
        --system-prompt-frames none safety_test hypothetical_scenario \
        --tasks-per-split 20

    # EIA is import-only, condition is always baseline (pipeline enforces).
    python -m eval_awareness_experiments.run_causal_experiment \
        --benchmarks eia --stage judge-only \
        --models anthropic/claude-opus-4.6:thinking
"""

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

from eval_awareness_experiments.extra_instructions import PRESETS
from eval_awareness_experiments.injection_modifier import ALL_CONDITIONS
from eval_awareness_experiments.system_prompt_frame import ALL_FRAMES

DEFAULT_MODELS = ["anthropic/claude-opus-4.7:thinking"]
ALL_BENCHMARKS = ["doomarena", "wasp", "agentdojo", "injecagent", "eia"]
ALL_PRESETS = sorted(PRESETS.keys())
ALL_STAGES = ["all", "run-only", "judge-only"]
BROWSER_BENCHMARKS = {"doomarena", "wasp"}


def _arm_name(preset: str, frame: str) -> str:
    """Map a (preset, frame) pair to a short arm name for output-dir layout.

    The four canonical arms in the causal eval-awareness experiment:
      bare:         preset=none, frame=none
      xml_safety:   preset=none, frame=safety_test
      xml_scenario: preset=none, frame=hypothetical_scenario
      native:       preset=scratchpad_native, frame=none

    Other (preset, frame) combinations fall back to a deterministic
    `<preset>__<frame>` string so they don't collide with the canonical
    arm names but stay machine-readable.
    """
    canonical = {
        ("none", "none"): "bare",
        ("none", "safety_test"): "xml_safety",
        ("none", "hypothetical_scenario"): "xml_scenario",
        ("scratchpad_native", "none"): "native",
    }
    arm = canonical.get((preset, frame))
    if arm is not None:
        return arm
    if frame == "none":
        return preset
    if preset == "none":
        return f"frame_{frame}"
    return f"{preset}__{frame}"


def _parse_benchmark_splits(values: list[str] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Invalid --benchmark-splits value {value!r}; expected benchmark=split,split"
            )
        benchmark, raw_splits = value.split("=", 1)
        benchmark = benchmark.strip()
        if benchmark not in ALL_BENCHMARKS:
            raise ValueError(f"Unknown benchmark in --benchmark-splits: {benchmark!r}")
        splits = [s.strip() for s in raw_splits.split(",") if s.strip()]
        if not splits:
            raise ValueError(f"No splits provided for benchmark {benchmark!r}")
        out[benchmark] = splits
    return out


def run_causal_experiment(
    models: list[str],
    benchmarks: list[str],
    conditions: list[str],
    presets: list[str],
    frames: list[str],
    tasks_per_split: int | None,
    stage: str,
    judge_model: str | None,
    judges: list[str] | None,
    output_base: Path,
    dry_run: bool,
    skip_existing: bool = False,
    splits: list[str] | None = None,
    benchmark_splits: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    avg_step_timeout: int | None = None,
    browser_stage1_timeout: int | None = None,
    browser_stage1_overhead: int | None = None,
    browser_splits_sequential: bool = False,
) -> None:
    combos = list(product(benchmarks, conditions, models, presets, frames))
    print(f"Causal experiment: {len(combos)} combinations")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Conditions: {conditions}")
    print(f"  Models:     {models}")
    print(f"  Presets:    {presets}")
    print(f"  Frames:     {frames}")
    print(f"  Stage:      {stage}")
    if tasks_per_split:
        print(f"  Tasks/split: {tasks_per_split}")
    if splits:
        print(f"  Splits:      {splits}")
    if benchmark_splits:
        print(f"  Benchmark splits: {benchmark_splits}")
    if max_steps is not None:
        print(f"  Max steps:   {max_steps}")
    if avg_step_timeout is not None:
        print(f"  Avg step timeout: {avg_step_timeout}s")
    if browser_stage1_timeout is not None:
        print(f"  Browser split timeout: {browser_stage1_timeout}s")
    elif browser_stage1_overhead is not None:
        print(f"  Browser split timeout overhead: {browser_stage1_overhead}s")
    if skip_existing:
        print(f"  Skip-existing: ON (cells already complete will be skipped)")
    print()

    # If skip_existing, scan the output base to find already-complete cells
    # so we can short-circuit re-runs.
    #
    # Note for parallel launches (16 streams via launch_pilot.sh): each stream
    # owns a disjoint slice of the matrix (one arm × one benchmark × all
    # 6 models, run sequentially). Slices never overlap, so there's no
    # cross-stream coordination needed for skip_existing — each stream's
    # view of "completed" only needs to include its own slice.
    completed_cells: set[tuple[str, str, str]] = set()
    if skip_existing and output_base.exists():
        from eval_awareness_experiments.run_manifest import scan as scan_manifest
        m = scan_manifest(output_base)
        for c in m["cells"]:
            if c["status"].startswith("complete"):
                completed_cells.add((c["benchmark"], c["arm"], c["model_slug"]))
        if completed_cells:
            print(f"  Found {len(completed_cells)} already-complete cells; will skip them.\n")

    run_log = []
    for i, (benchmark, condition, model, preset, frame) in enumerate(combos, 1):
        # EIA trajectories predate the causal framework — pipeline rejects
        # non-baseline conditions there, so skip cleanly rather than fail.
        if benchmark == "eia" and condition != "baseline":
            print(f"[{i}/{len(combos)}] SKIP eia/{condition} (import-only, baseline only)")
            continue
        # EIA also can't take a system prompt frame (no agent run).
        if benchmark == "eia" and frame != "none":
            print(f"[{i}/{len(combos)}] SKIP eia/{frame} (import-only, no agent runs)")
            continue

        # Output layout: <output_base>/<benchmark>/<arm>/<model>/
        # arm encodes (preset, frame) — see _arm_name. `condition` is omitted
        # because this experiment always uses condition=baseline; the XML
        # manipulation lives in `frame`, not `condition`.
        arm = _arm_name(preset, frame)
        model_slug = model.replace("/", "_").replace(":", "_")
        combo_out = (
            output_base
            / benchmark
            / arm
            / model_slug
        )

        if skip_existing and (benchmark, arm, model_slug) in completed_cells:
            print(f"[{i}/{len(combos)}] SKIP {benchmark}/{arm}/{model} (already complete)")
            run_log.append({
                "benchmark": benchmark, "arm": arm, "condition": condition,
                "model": model, "preset": preset, "system_prompt_frame": frame,
                "stage": stage, "status": "skipped (already complete)",
                "output_dir": str(combo_out), "cmd": None,
            })
            continue
        cmd = [
            sys.executable, "-m", "eval_awareness_experiments.run_safety_pipeline",
            "--benchmark", benchmark,
            "--model-name", model,
            "--condition", condition,
            "--extra-instructions-preset", preset,
            "--system-prompt-frame", frame,
            "--stage", stage,
            "--output-dir", str(combo_out),
        ]
        if judge_model:
            cmd.extend(["--judge-model", judge_model])
        if judges:
            cmd.extend(["--judges", *judges])
        if tasks_per_split is not None:
            cmd.extend(["--tasks-per-split", str(tasks_per_split)])
        effective_splits = (benchmark_splits or {}).get(benchmark, splits)
        if effective_splits:
            cmd.extend(["--splits", *effective_splits])
        if max_steps is not None:
            cmd.extend(["--max-steps", str(max_steps)])
        if avg_step_timeout is not None:
            cmd.extend(["--avg-step-timeout", str(avg_step_timeout)])
        if browser_stage1_timeout is not None:
            cmd.extend(["--browser-stage1-timeout", str(browser_stage1_timeout)])
        if browser_stage1_overhead is not None:
            cmd.extend(["--browser-stage1-overhead", str(browser_stage1_overhead)])
        if browser_splits_sequential:
            cmd.append("--browser-splits-sequential")

        label = f"{benchmark}/{arm}/{model} (preset={preset}, frame={frame}, condition={condition})"
        print(f"[{i}/{len(combos)}] {label}")

        if dry_run:
            print(f"  DRY RUN: {' '.join(cmd)}")
            status = "dry_run"
        else:
            print(f"  Running: {' '.join(cmd)}")
            # Stream stdout/stderr live — pipeline runs can be hours long.
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  FAILED ({result.returncode}) — see output above")
                status = f"failed:{result.returncode}"
            else:
                print("  OK")
                status = "ok"

        run_log.append({
            "benchmark": benchmark,
            "arm": arm,
            "condition": condition,
            "model": model,
            "preset": preset,
            "system_prompt_frame": frame,
            "stage": stage,
            "status": status,
            "output_dir": str(combo_out),
            "cmd": " ".join(cmd),
        })

    output_base.mkdir(parents=True, exist_ok=True)
    # Per-process log filename — when 16 parallel streams share the same
    # output_base (launch_pilot.sh), every stream needs its own log file
    # or the last writer clobbers the rest. Timestamp + PID guarantees
    # uniqueness without coordination.
    import os
    from datetime import datetime
    suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    log_path = output_base / f"causal_experiment_log__{suffix}.json"
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"\nExperiment log saved to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matrix runner: benchmarks × conditions × models × presets via run_safety_pipeline"
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="OpenRouter model ids (use :thinking for reasoning).")
    parser.add_argument("--benchmarks", nargs="+", default=["doomarena"],
                        choices=ALL_BENCHMARKS)
    parser.add_argument("--conditions", nargs="+", default=["baseline"],
                        choices=ALL_CONDITIONS)
    parser.add_argument("--extra-instructions-presets", nargs="+", default=["none"],
                        choices=ALL_PRESETS, dest="presets")
    parser.add_argument("--system-prompt-frames", nargs="+", default=["none"],
                        choices=ALL_FRAMES, dest="frames",
                        help="Wrap the entire system prompt in XML tags. "
                             "Tool-calling benchmarks only — browser benchmarks "
                             "skip non-`none` frames.")
    parser.add_argument("--tasks-per-split", type=int, default=None,
                        help="Forwarded to run_safety_pipeline (e.g. WASP has 2 splits, so 20 → 40 tasks/combo).")
    parser.add_argument("--stage", default="all", choices=ALL_STAGES)
    parser.add_argument("--judge-model", default=None,
                        help="Judge model for stage 2 (defaults to pipeline default).")
    parser.add_argument("--judges", nargs="+", default=None,
                        help="Judge names (e.g. test_awareness_steering verbalized_awareness "
                             "purpose_continue_5q). Omit to use pipeline defaults.")
    parser.add_argument("--output-base", type=Path,
                        default=Path("eval_awareness_experiments/results/new_results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Before each combo, scan --output-base for cells "
                             "already marked complete (via run_manifest) and "
                             "skip them. Lets re-runs of the same matrix pick "
                             "up where they left off.")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Forwarded to run_safety_pipeline. For DoomArena, "
                             "use 'reddit gitlab' to skip the heavy/flaky "
                             "shopping splits.")
    parser.add_argument("--benchmark-splits", nargs="+", default=None,
                        help="Benchmark-specific split overrides, comma-separated. "
                             "Example: doomarena=reddit,shopping,shopping_admin,gitlab "
                             "wasp=reddit,gitlab. Overrides --splits for matching benchmarks.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Forwarded to run_safety_pipeline/browser runners.")
    parser.add_argument("--avg-step-timeout", type=int, default=None,
                        help="Forwarded to run_safety_pipeline/browser runners. "
                             "Per-task timeout is max_steps × avg_step_timeout.")
    parser.add_argument("--browser-stage1-timeout", type=int, default=None,
                        help="Forwarded to run_safety_pipeline.")
    parser.add_argument("--browser-stage1-overhead", type=int, default=None,
                        help="Forwarded to run_safety_pipeline. Used only when "
                             "--browser-stage1-timeout is omitted.")
    parser.add_argument("--browser-splits-sequential", action="store_true",
                        help="Forwarded to run_safety_pipeline for debugging "
                             "site/container contention.")
    args = parser.parse_args()
    benchmark_splits = _parse_benchmark_splits(args.benchmark_splits)

    run_causal_experiment(
        models=args.models,
        benchmarks=args.benchmarks,
        conditions=args.conditions,
        presets=args.presets,
        frames=args.frames,
        tasks_per_split=args.tasks_per_split,
        stage=args.stage,
        judge_model=args.judge_model,
        judges=args.judges,
        output_base=args.output_base,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        splits=args.splits,
        benchmark_splits=benchmark_splits,
        max_steps=args.max_steps,
        avg_step_timeout=args.avg_step_timeout,
        browser_stage1_timeout=args.browser_stage1_timeout,
        browser_stage1_overhead=args.browser_stage1_overhead,
        browser_splits_sequential=args.browser_splits_sequential,
    )


if __name__ == "__main__":
    main()
