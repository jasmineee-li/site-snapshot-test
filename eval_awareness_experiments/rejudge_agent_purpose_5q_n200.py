"""Re-run n200 purpose_continue_5q with the original trajectory model.

This is a thin launcher around run_safety_pipeline.py for the contaminated
2026-04-29 n200 result tree. It writes corrected 5Q rows into a separate
output tree and leaves the original results in place.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "eval_awareness_experiments" / "results" / "n200_2026-04-29"
OUTPUT_ROOT = (
    REPO_ROOT
    / "eval_awareness_experiments"
    / "results"
    / "n200_2026-04-29_agent5q_rejudge_20260506"
)

ARMS = ("bare", "xml_safety", "xml_scenario")
MODEL_SLUGS = (
    "moonshotai_kimi-k2.5_thinking",
    "anthropic_claude-sonnet-4.6_thinking",
    "openai_gpt-5.2_thinking",
    "anthropic_claude-opus-4.7_thinking",
    "z-ai_glm-5_thinking",
    "google_gemini-2.5-pro_thinking",
)
BENCHMARK_SPLITS = {
    "wasp": ("gitlab", "reddit"),
    "doomarena": ("reddit", "shopping", "shopping_admin", "gitlab"),
    "agentdojo": ("workspace",),
    "injecagent": ("dh_base",),
}

_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Cell:
    benchmark: str
    arm: str
    model_slug: str
    source_cell: Path
    output_cell: Path
    split_roots: dict[str, Path]

    @property
    def label(self) -> str:
        return f"{self.benchmark}/{self.arm}/{self.model_slug}"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_roots_for_toolcalling(
    *, benchmark: str, source_cell: Path, splits: tuple[str, ...]
) -> dict[str, Path]:
    manifest_path = source_cell / f"pipeline_manifest_{benchmark}.json"
    manifest = _read_json(manifest_path)
    config = manifest.get("config") or {}
    roots: dict[str, Path] = {}
    for split in splits:
        split_meta = (manifest.get("splits") or {}).get(split) or {}
        run_dir = split_meta.get("run_dir")
        if run_dir:
            roots[split] = Path(run_dir)
            continue

        # Some old n200 tool-calling cells were rejudged in place, leaving a
        # judge-only manifest that lacks run_dir. Recover the original
        # trajectory root from the recorded run metadata.
        meta = split_meta.get("original_run_meta") or {}
        model_name = meta.get("model_name") or config.get("model_name")
        condition = meta.get("condition") or config.get("condition") or "baseline"
        preset = (
            meta.get("extra_instructions_preset")
            or config.get("extra_instructions_preset")
            or "none"
        )
        frame = meta.get("system_prompt_frame") or config.get("system_prompt_frame") or "none"
        if model_name:
            model_slug = str(model_name).replace("/", "_").replace(":", "_")
            frame_suffix = f"_{frame}" if frame != "none" else ""
            run_name = f"{condition}_{preset}{frame_suffix}_{model_slug}"
            recovered = REPO_ROOT / "results" / "toolcalling" / benchmark / split / run_name
            if recovered.exists():
                roots[split] = recovered
                continue

        raise FileNotFoundError(f"{manifest_path} has no run_dir for split {split}")
    return roots


def _build_cell(
    *,
    benchmark: str,
    arm: str,
    model_slug: str,
    source_root: Path,
    output_root: Path,
) -> Cell | None:
    source_cell = source_root / benchmark / arm / model_slug
    output_cell = output_root / benchmark / arm / model_slug
    splits = BENCHMARK_SPLITS[benchmark]

    if benchmark in {"wasp", "doomarena"}:
        split_roots = {split: source_cell / "_browser_runs" / split for split in splits}
    else:
        split_roots = _split_roots_for_toolcalling(
            benchmark=benchmark,
            source_cell=source_cell,
            splits=splits,
        )

    missing = [str(path) for path in split_roots.values() if not path.exists()]
    if missing:
        with _PRINT_LOCK:
            print(f"[skip missing] {benchmark}/{arm}/{model_slug}: {missing}", flush=True)
        return None

    return Cell(
        benchmark=benchmark,
        arm=arm,
        model_slug=model_slug,
        source_cell=source_cell,
        output_cell=output_cell,
        split_roots=split_roots,
    )


def _result_file_for(cell: Cell, split: str) -> Path:
    return cell.output_cell / cell.benchmark / split / "trajectory_awareness_results.jsonl"


def _nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _cell_complete(cell: Cell) -> bool:
    return all(_nonempty(_result_file_for(cell, split)) for split in cell.split_roots)


def _command_for(
    cell: Cell,
    *,
    judges: list[str],
    judge_model: str,
    judge_concurrency: int,
    continuation_concurrency: int,
    judge_retries: int,
    continuation_retries: int,
    skip_wasp_asr: bool,
) -> list[str]:
    existing_dirs = [f"{split}:{path}" for split, path in cell.split_roots.items()]
    cmd = [
        sys.executable,
        "-m",
        "eval_awareness_experiments.run_safety_pipeline",
        "--benchmark",
        cell.benchmark,
        "--stage",
        "judge-only",
        "--splits",
        *cell.split_roots.keys(),
        "--existing-dirs",
        *existing_dirs,
        "--output-dir",
        str(cell.output_cell),
        "--judge-model",
        judge_model,
        "--judge-concurrency",
        str(judge_concurrency),
        "--judge-retries",
        str(judge_retries),
        "--purpose-continuation-model-source",
        "agent",
        "--purpose-continuation-concurrency",
        str(continuation_concurrency),
        "--purpose-continuation-retries",
        str(continuation_retries),
        "--judges",
        *judges,
    ]
    if cell.benchmark == "wasp" and skip_wasp_asr:
        cmd.append("--skip-wasp-asr")
    return cmd


def _run_cell(
    cell: Cell,
    *,
    judge_model: str,
    judges: list[str],
    judge_concurrency: int,
    continuation_concurrency: int,
    judge_retries: int,
    continuation_retries: int,
    skip_existing: bool,
    skip_wasp_asr: bool,
    log_dir: Path,
    dry_run: bool,
) -> int:
    if skip_existing and _cell_complete(cell):
        with _PRINT_LOCK:
            print(f"[complete] {cell.label}", flush=True)
        return 0

    cmd = _command_for(
        cell,
        judges=judges,
        judge_model=judge_model,
        judge_concurrency=judge_concurrency,
        continuation_concurrency=continuation_concurrency,
        judge_retries=judge_retries,
        continuation_retries=continuation_retries,
        skip_wasp_asr=skip_wasp_asr,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cell.benchmark}__{cell.arm}__{cell.model_slug}.log"

    with _PRINT_LOCK:
        print(f"[start] {cell.label}", flush=True)
        print(f"[cmd] {' '.join(cmd)}", flush=True)
        print(f"[log] {log_path}", flush=True)

    if dry_run:
        return 0

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n[start] {cell.label}\n")
        log_f.write(f"[cmd] {' '.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            with _PRINT_LOCK:
                print(f"[{cell.label}] {line}", end="", flush=True)
        rc = proc.wait()
        log_f.write(f"[finish] {cell.label} rc={rc}\n")

    with _PRINT_LOCK:
        print(f"[finish] {cell.label} rc={rc}", flush=True)
    return rc


def _run_model_cells(
    cells: list[Cell],
    *,
    judge_model: str,
    judges: list[str],
    judge_concurrency: int,
    continuation_concurrency: int,
    judge_retries: int,
    continuation_retries: int,
    skip_existing: bool,
    skip_wasp_asr: bool,
    log_dir: Path,
    dry_run: bool,
    stop_on_error: bool,
) -> tuple[str, list[tuple[str, int]]]:
    model_slug = cells[0].model_slug if cells else "unknown"
    failures: list[tuple[str, int]] = []
    for cell in cells:
        rc = _run_cell(
            cell,
            judge_model=judge_model,
            judges=judges,
            judge_concurrency=judge_concurrency,
            continuation_concurrency=continuation_concurrency,
            judge_retries=judge_retries,
            continuation_retries=continuation_retries,
            skip_existing=skip_existing,
            skip_wasp_asr=skip_wasp_asr,
            log_dir=log_dir,
            dry_run=dry_run,
        )
        if rc != 0:
            failures.append((cell.label, rc))
            if stop_on_error:
                break
    return model_slug, failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=tuple(BENCHMARK_SPLITS),
        default=["wasp", "doomarena", "agentdojo", "injecagent"],
        help="Benchmarks to rejudge, in order. All models finish one benchmark before the next starts.",
    )
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--model-slugs", nargs="+", default=list(MODEL_SLUGS))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--judges", nargs="+", default=["purpose_continue_5q"])
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.7:thinking")
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--judge-retries", type=int, default=7)
    parser.add_argument("--purpose-continuation-concurrency", type=int, default=8)
    parser.add_argument("--purpose-continuation-retries", type=int, default=7)
    parser.add_argument(
        "--model-workers",
        type=int,
        default=6,
        help="Number of model cells to run in parallel within each benchmark.",
    )
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after a failed cell. By default failures are recorded and the launcher continues.",
    )
    parser.add_argument(
        "--rerun-wasp-asr",
        action="store_true",
        help="Recompute WASP post-hoc ASR in the new output tree instead of using task attack summaries.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"

    print("Config:", flush=True)
    print(f"  source_root={args.source_root}", flush=True)
    print(f"  output_root={args.output_root}", flush=True)
    print(f"  benchmarks={args.benchmarks}", flush=True)
    print(f"  arms={args.arms}", flush=True)
    print(f"  model_slugs={args.model_slugs}", flush=True)
    print(f"  judges={args.judges}", flush=True)
    print(f"  model_workers={args.model_workers}", flush=True)
    print(f"  judge_model={args.judge_model}", flush=True)
    print(f"  judge_concurrency={args.judge_concurrency}", flush=True)
    print(f"  judge_retries={args.judge_retries}", flush=True)
    print(
        f"  purpose_continuation_concurrency={args.purpose_continuation_concurrency}",
        flush=True,
    )
    print(
        f"  purpose_continuation_retries={args.purpose_continuation_retries}",
        flush=True,
    )
    print(f"  skip_existing={not args.no_skip_existing}", flush=True)
    print(f"  skip_wasp_asr={not args.rerun_wasp_asr}", flush=True)

    failures: list[tuple[str, int]] = []
    for benchmark in args.benchmarks:
        print(f"\n=== benchmark {benchmark} ===", flush=True)
        benchmark_failures: list[tuple[str, int]] = []
        cells_by_model: dict[str, list[Cell]] = {}
        for model_slug in args.model_slugs:
            for arm in args.arms:
                cell = _build_cell(
                    benchmark=benchmark,
                    arm=arm,
                    model_slug=model_slug,
                    source_root=args.source_root,
                    output_root=args.output_root,
                )
                if cell is not None:
                    cells_by_model.setdefault(model_slug, []).append(cell)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.model_workers)) as pool:
            future_to_model = {
                pool.submit(
                    _run_model_cells,
                    cells,
                    judge_model=args.judge_model,
                    judges=args.judges,
                    judge_concurrency=args.judge_concurrency,
                    continuation_concurrency=args.purpose_continuation_concurrency,
                    judge_retries=args.judge_retries,
                    continuation_retries=args.purpose_continuation_retries,
                    skip_existing=not args.no_skip_existing,
                    skip_wasp_asr=not args.rerun_wasp_asr,
                    log_dir=log_dir,
                    dry_run=args.dry_run,
                    stop_on_error=args.stop_on_error,
                ): model_slug
                for model_slug, cells in cells_by_model.items()
            }
            for future in concurrent.futures.as_completed(future_to_model):
                model_slug = future_to_model[future]
                try:
                    _, model_failures = future.result()
                except Exception as exc:  # pragma: no cover - launcher guard
                    model_failures = [(f"{benchmark}/{model_slug}", 1)]
                    with _PRINT_LOCK:
                        print(f"[error] {benchmark}/{model_slug}: {exc}", flush=True)
                failures.extend(model_failures)
                benchmark_failures.extend(model_failures)

        if benchmark_failures:
            print(
                f"[continue] failures after {benchmark}: {benchmark_failures}",
                flush=True,
            )
            if args.stop_on_error:
                print(f"[stop] stop-on-error requested: {failures}", flush=True)
                return 1

    if failures:
        print(f"\nCompleted with failures: {failures}", flush=True)
        return 1
    print("\nAll requested rejudges completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
