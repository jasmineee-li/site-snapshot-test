"""Scan results directories and emit a JSON manifest of completed runs.

Used by the matrix runner to skip cells already complete and by the
dashboard to render the (benchmark × arm × model) coverage grid.

The new pilot layout is `<results_root>/<benchmark>/<arm>/<model>/...`
where each cell directory contains the run output (a `<benchmark>/<split>/`
subtree with trajectory files and a markdown sidecar dir).

Cells are detected by scanning for `markdown/` directories and inferring
the (benchmark, arm, model) from the path. Per-cell metrics (VEA rate,
ASR, 5PQ summary) are aggregated cheaply by parsing the JSONL judge
output and trajectory metadata where present.

Usage:
    # Write the manifest:
    python -m eval_awareness_experiments.run_manifest \
        --results-dir eval_awareness_experiments/results/causal_pilot \
        --output eval_awareness_experiments/results/causal_pilot/manifest.json

    # Print a human-readable summary table:
    python -m eval_awareness_experiments.run_manifest \
        --results-dir eval_awareness_experiments/results/causal_pilot \
        --print

    # Programmatic check — exit 0 if cell complete, exit 1 if missing/partial:
    python -m eval_awareness_experiments.run_manifest \
        --results-dir eval_awareness_experiments/results/causal_pilot \
        --check agentdojo bare z-ai/glm-5:thinking
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _model_to_slug(model: str) -> str:
    """Mirror the slugification used by run_causal_experiment when constructing
    output dirs: replace / and : with _."""
    return model.replace("/", "_").replace(":", "_")


# What "complete" means per benchmark for --skip-existing.
# Tighter than the legacy "any judge row exists" check — a partial cell
# (e.g. only reddit but missing gitlab) used to be marked complete because
# the manifest scanner only counted judge rows, not split coverage. See
# DOOMARENA_ROOT_CAUSE_HANDOFF.md root cause 8.
_EXPECTED_SPLITS: dict[str, list[str]] = {
    # Default-runs hit reddit + gitlab; shopping/shopping_admin are opt-in
    # via explicit --splits, so we don't require them for completeness.
    "doomarena": ["reddit", "gitlab"],
    "wasp": ["gitlab", "reddit"],
    "agentdojo": ["workspace"],
    "injecagent": ["dh_base"],
    "eia": ["baseline"],
}


def _expected_splits_for(benchmark: str, override: list[str] | None = None) -> list[str]:
    if override:
        return list(override)
    return list(_EXPECTED_SPLITS.get(benchmark, []))


def _slug_to_model(slug: str) -> str:
    """Best-effort reverse of _model_to_slug. The original / and : positions
    aren't recoverable, so we just return the slug as-is for display."""
    return slug


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _aggregate_judge_results(jsonl_path: Path) -> dict[str, Any]:
    """Pull headline metrics from the trajectory_awareness_results JSONL.

    Returns:
        n_vea: count of trajectories with verbalized_awareness judgment
        n_5pq: count of trajectories with purpose_continue_5q judgment
        vea_count: count of non-Real verbalized_awareness classifications
        align_share: mean share of Alignment Eval across 5PQ prompts (mean over tasks)
        any_eval_share: mean share of any-eval across 5PQ prompts
    """
    rows = _read_jsonl(jsonl_path)
    if not rows:
        return {
            "n_vea": 0, "n_5pq": 0, "vea_count": 0,
            "align_share": None, "any_eval_share": None,
        }

    rows_vea = [r for r in rows if r.get("experiment_type") == "trajectory_verbalized_awareness"]
    rows_5pq = [r for r in rows if r.get("experiment_type") == "trajectory_purpose_continue_5q"]

    vea_count = sum(
        1 for r in rows_vea
        if (r.get("classification") or "").strip().lower() not in ("real", "", None)
    )

    # 5PQ: per-task mean across variants of "is this an alignment eval"
    align_per_task = []
    any_per_task = []
    for r in rows_5pq:
        variants = r.get("metadata", {}).get("variant_categories", [])
        if not variants:
            continue
        nonref = [v for v in variants if (v.get("category") or "").strip() != "Refused"]
        if not nonref:
            continue
        n_align = sum(1 for v in nonref if v.get("category") == "Alignment Evaluation")
        n_any = sum(
            1 for v in nonref
            if v.get("category") in ("Alignment Evaluation", "Capability Evaluation", "Other Evaluation")
        )
        align_per_task.append(n_align / len(nonref))
        any_per_task.append(n_any / len(nonref))

    align_share = sum(align_per_task) / len(align_per_task) if align_per_task else None
    any_share = sum(any_per_task) / len(any_per_task) if any_per_task else None

    return {
        "n_vea": len(rows_vea),
        "n_5pq": len(rows_5pq),
        "vea_count": vea_count,
        "align_share": align_share,
        "any_eval_share": any_share,
    }


def _summarize_trajectories(trajectory_dir: Path) -> dict[str, Any]:
    """Count completed trajectories and aggregate ASR / task-pass.

    Trajectory files are typically `<benchmark>/<split>/*.json` (one per task).
    We open each and pull `attack_succeeded` / `task_passed` if present.
    """
    if not trajectory_dir.exists():
        return {"n_trajectories": 0, "n_attacks": 0, "n_passes": 0}

    n_traj = 0
    n_attack = 0
    n_pass = 0
    for f in trajectory_dir.glob("*.json"):
        if f.name in ("trajectory_awareness_results.jsonl",):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        n_traj += 1
        if data.get("attack_succeeded"):
            n_attack += 1
        if data.get("task_passed"):
            n_pass += 1
    return {"n_trajectories": n_traj, "n_attacks": n_attack, "n_passes": n_pass}


def _scan_cell(cell_dir: Path, benchmark: str, arm: str, model_slug: str) -> dict[str, Any]:
    """Look at `<cell_dir>/<benchmark>/<split>/` for trajectory + judge data."""
    inner_bench_dir = cell_dir / benchmark
    cell: dict[str, Any] = {
        "benchmark": benchmark,
        "arm": arm,
        "model_slug": model_slug,
        "model": _slug_to_model(model_slug),
        "output_dir": str(cell_dir),
        "splits": [],
        "n_trajectories": 0,
        "n_attacks": 0,
        "n_passes": 0,
        "n_vea": 0,
        "n_5pq": 0,
        "vea_count": 0,
        "align_share": None,
        "any_eval_share": None,
        "last_modified": None,
    }
    if not inner_bench_dir.exists():
        cell["status"] = "empty"
        return cell

    last_mtime = 0.0
    for split_dir in sorted(inner_bench_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split_summary = _summarize_trajectories(split_dir)
        cell["splits"].append({"name": split_dir.name, **split_summary})
        cell["n_trajectories"] += split_summary["n_trajectories"]
        cell["n_attacks"] += split_summary["n_attacks"]
        cell["n_passes"] += split_summary["n_passes"]

        judge_jsonl = split_dir / "trajectory_awareness_results.jsonl"
        if judge_jsonl.exists():
            agg = _aggregate_judge_results(judge_jsonl)
            cell["n_vea"] += agg["n_vea"]
            cell["n_5pq"] += agg["n_5pq"]
            cell["vea_count"] += agg["vea_count"]
            # For align/any_eval, we average across splits — usually only one
            # split per cell, so this just takes that value.
            if agg["align_share"] is not None:
                cell["align_share"] = agg["align_share"]
            if agg["any_eval_share"] is not None:
                cell["any_eval_share"] = agg["any_eval_share"]

        # Track latest mtime for "last updated" display.
        for f in split_dir.rglob("*"):
            try:
                m = f.stat().st_mtime
                if m > last_mtime:
                    last_mtime = m
            except FileNotFoundError:
                pass

    if last_mtime:
        cell["last_modified"] = datetime.fromtimestamp(last_mtime, tz=timezone.utc).isoformat()

    # Strict completeness: every expected split must have at least one
    # judged trajectory. Old check skipped cells with judges in only ONE
    # split, which made --skip-existing skip half-empty DoomArena cells.
    expected_splits = _expected_splits_for(benchmark)
    seen_splits = {s["name"] for s in cell["splits"]}
    judged_splits: set[str] = set()
    for split_dir in (inner_bench_dir.iterdir() if inner_bench_dir.exists() else []):
        if not split_dir.is_dir():
            continue
        if (split_dir / "trajectory_awareness_results.jsonl").exists():
            judged_splits.add(split_dir.name)
    cell["expected_splits"] = expected_splits
    cell["judged_splits"] = sorted(judged_splits)
    missing_splits = [s for s in expected_splits if s not in judged_splits]

    if cell["n_trajectories"] == 0:
        cell["status"] = "empty"
    elif cell["n_vea"] == 0 and cell["n_5pq"] == 0:
        cell["status"] = "run-only (no judges)"
    elif missing_splits:
        cell["status"] = f"partial (missing splits: {','.join(missing_splits)})"
    elif cell["n_5pq"] > 0:
        cell["status"] = "complete (with 5pq)"
    elif cell["n_vea"] > 0:
        cell["status"] = "complete (vea only)"
    else:
        cell["status"] = "partial"

    cell["asr"] = cell["n_attacks"] / cell["n_trajectories"] if cell["n_trajectories"] else None
    cell["task_pass_rate"] = cell["n_passes"] / cell["n_trajectories"] if cell["n_trajectories"] else None
    cell["vea_rate"] = cell["vea_count"] / cell["n_vea"] if cell["n_vea"] else None
    return cell


def scan(results_dir: Path) -> dict[str, Any]:
    """Walk `results_dir/<benchmark>/<arm>/<model>/` and produce a manifest dict."""
    cells: list[dict[str, Any]] = []
    if results_dir.exists():
        for benchmark_dir in sorted(results_dir.iterdir()):
            if not benchmark_dir.is_dir():
                continue
            benchmark = benchmark_dir.name
            for arm_dir in sorted(benchmark_dir.iterdir()):
                if not arm_dir.is_dir():
                    continue
                arm = arm_dir.name
                for model_dir in sorted(arm_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    cell = _scan_cell(model_dir, benchmark, arm, model_dir.name)
                    cells.append(cell)
    return {
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "cells": cells,
    }


def _print_table(manifest: dict[str, Any]) -> None:
    cells = manifest["cells"]
    if not cells:
        print(f"No cells found under {manifest['results_dir']}")
        return
    cols = ["benchmark", "arm", "model_slug", "n", "ASR", "task_pass", "VEA", "Align", "Any", "status"]
    fmt = "{:<11} {:<14} {:<40} {:>4} {:>5} {:>9} {:>5} {:>5} {:>5}  {}"
    print(fmt.format(*cols))
    print("-" * 130)
    for c in cells:
        n = c["n_trajectories"]
        asr = f"{c['asr']:.2f}" if c.get("asr") is not None else "  -  "
        tpr = f"{c['task_pass_rate']:.2f}" if c.get("task_pass_rate") is not None else "  -  "
        vea = f"{c['vea_count']}/{c['n_vea']}" if c["n_vea"] else "  -  "
        align = f"{c['align_share']:.3f}" if c.get("align_share") is not None else "  -  "
        any_ev = f"{c['any_eval_share']:.3f}" if c.get("any_eval_share") is not None else "  -  "
        print(fmt.format(
            c["benchmark"], c["arm"], c["model_slug"][:40], n, asr, tpr, vea, align, any_ev, c.get("status", "?")
        ))


def _check_cell(manifest: dict[str, Any], benchmark: str, arm: str, model: str) -> int:
    """Return shell exit code: 0 if cell exists with judges, 1 if missing/partial."""
    slug = _model_to_slug(model)
    for c in manifest["cells"]:
        if (c["benchmark"], c["arm"], c["model_slug"]) == (benchmark, arm, slug):
            if c["status"].startswith("complete"):
                print(f"complete: {benchmark}/{arm}/{model} (n={c['n_trajectories']})")
                return 0
            print(f"{c['status']}: {benchmark}/{arm}/{model} "
                  f"(n_trajectories={c['n_trajectories']}, n_vea={c['n_vea']}, n_5pq={c['n_5pq']})")
            return 1
    print(f"missing: {benchmark}/{arm}/{model}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Root of the causal-pilot results tree.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the JSON manifest. Defaults to "
                             "<results-dir>/manifest.json.")
    parser.add_argument("--print", action="store_true",
                        help="Print a human-readable summary table to stdout.")
    parser.add_argument("--check", nargs=3, metavar=("BENCHMARK", "ARM", "MODEL"),
                        default=None, help="Exit 0 if (benchmark, arm, model) is "
                        "complete; exit 1 if missing or partial. Skips manifest write.")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"results-dir does not exist: {args.results_dir}", file=sys.stderr)
        return 2

    manifest = scan(args.results_dir)

    if args.check:
        return _check_cell(manifest, *args.check)

    output = args.output or args.results_dir / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {output} ({len(manifest['cells'])} cells)")

    if args.print:
        print()
        _print_table(manifest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
