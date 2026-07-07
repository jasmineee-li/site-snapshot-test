"""Run VerbalizedAwarenessJudge on completed AgentDojo sweep cells.

Standalone analog of the LLM-judge step inside agentdojo_steering_sweep.sh,
runnable on already-complete cell directories without waiting for an
entire shard to finish. Cross-probe activation scoring is intentionally
omitted here so this can run anywhere (no GPU required).

Usage:
    .venv/bin/python scripts/judge_agentdojo_cells.py \\
        --suite workspace \\
        --output-csv /tmp/partial_sweep.csv \\
        results/toolcalling/agentdojo/sweep/guiowl_framing_*/cells/*

Each positional arg is a cell directory like
`results/.../cells/N6_layers9_8_10_12_15_11_alpha0.6/`. Cells are
identified by their dir name; per-task JSONs are read from
`<cell_dir>/<suite>/cell/*.json`.

Output CSV columns:
    cell, n_layers, layers, alpha, n, tsr, n_passed,
    eval_aware_rate, n_aware, n_judged

Eval-aware = LLM judge categorizes the transcript as ALIGNMENT_EVAL,
CAPABILITY_EVAL, or OTHER_EVAL (vs REAL/REFUSED/OTHER). Same convention
as the harness end-step.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_awareness_experiments.judges.verbalized_awareness import VerbalizedAwarenessJudge
from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import Message, ReasoningCategory, Transcript


EVAL_AWARE_CATS = {
    ReasoningCategory.ALIGNMENT_EVAL,
    ReasoningCategory.CAPABILITY_EVAL,
    ReasoningCategory.OTHER_EVAL,
}


def to_transcript(d: dict) -> Transcript:
    msgs: list[Message] = []
    if d.get("system_prompt"):
        msgs.append(Message(role="system", content=d["system_prompt"]))
    msgs.append(Message(role="user", content=d.get("task_instruction", "")))
    for step in d.get("steps", []):
        parts = []
        if step.get("preamble"):
            parts.append(step["preamble"])
        if step.get("thought"):
            parts.append(step["thought"])
        if step.get("action"):
            parts.append(step["action"])
        msgs.append(Message(role="assistant", content="\n".join(parts) or ""))
        if step.get("observation"):
            msgs.append(Message(role="tool", content=step["observation"]))
    if d.get("final_answer"):
        msgs.append(Message(role="assistant", content=d["final_answer"]))
    return Transcript(
        messages=msgs,
        system_instructions=d.get("system_prompt", ""),
        instruction=d.get("task_instruction", ""),
        metadata={"task_id": d.get("task_id")},
    )


def parse_cell_id(cell_id: str) -> tuple[int, str, float]:
    """Recover (n_layers, layers_csv, alpha) from a sweep cell directory name.

    Supports two formats:
      - baseline cells start with 'baseline'
      - steering cells: 'N{n}_layers{L1_L2_..}_alpha{a}'
    """
    if cell_id.startswith("baseline"):
        return 0, "", 0.0
    m = re.match(r"N(\d+)_layers([\d_]+)_alpha([\d.]+)", cell_id)
    if not m:
        return -1, cell_id, float("nan")
    n_layers = int(m.group(1))
    layers_csv = m.group(2).replace("_", ",")
    alpha = float(m.group(3))
    return n_layers, layers_csv, alpha


def cell_jsons(cell_dir: Path, suite: str) -> list[Path]:
    suite_dir = cell_dir / suite / "cell"
    if not suite_dir.exists():
        return []
    return sorted(p for p in suite_dir.glob("*.json") if p.name != "manifest.json")


async def judge_cell(cell_dir: Path, suite: str, judge: VerbalizedAwarenessJudge):
    docs = []
    for fp in cell_jsons(cell_dir, suite):
        try:
            docs.append(json.loads(fp.read_text()))
        except Exception:
            pass

    async def judge_one(d):
        try:
            return await judge.judge(to_transcript(d))
        except Exception as e:
            print(f"[judge warn] task={d.get('task_id')}: {e!r}", file=sys.stderr)
            return None

    results = await asyncio.gather(*[judge_one(d) for d in docs])
    return docs, results


async def amain(args: argparse.Namespace) -> None:
    cell_dirs = [Path(p) for p in args.cell_dirs]
    cell_dirs = [d for d in cell_dirs if d.is_dir()]
    if not cell_dirs:
        raise SystemExit("No valid cell directories provided.")

    judge_llm = LLM(
        model=args.judge_model,
        thinking="thinking" in args.judge_model,
        concurrency=args.judge_concurrency,
        temperature=0.0,
    )
    judge = VerbalizedAwarenessJudge(
        grader_model=judge_llm,
        classifier_model=judge_llm,
        max_concurrency=args.judge_concurrency,
    )

    rows = []
    for cell_dir in cell_dirs:
        cell_id = cell_dir.name
        n_layers, layers_csv, alpha = parse_cell_id(cell_id)
        docs, judge_results = await judge_cell(cell_dir, args.suite, judge)
        n = len(docs)
        n_passed = sum(1 for d in docs if d.get("passed") is True)
        n_aware = sum(
            1 for r in judge_results
            if r is not None and r.category in EVAL_AWARE_CATS
        )
        n_judged = sum(1 for r in judge_results if r is not None)
        row = {
            "cell": cell_id,
            "n_layers": n_layers,
            "layers": layers_csv,
            "alpha": alpha,
            "n": n,
            "tsr": (n_passed / n) if n else None,
            "n_passed": n_passed,
            "eval_aware_rate": (n_aware / n_judged) if n_judged else None,
            "n_aware": n_aware,
            "n_judged": n_judged,
        }
        rows.append(row)
        tsr_s = f"{row['tsr']:.4f}" if row['tsr'] is not None else "-"
        aware_s = f"{row['eval_aware_rate']:.4f}" if row['eval_aware_rate'] is not None else "-"
        print(f"{cell_id}: n={n} tsr={tsr_s} aware={aware_s} ({n_aware}/{n_judged} judged)")

    rows.sort(key=lambda r: (r["n_layers"], r["alpha"]))

    fields = [
        "cell", "n_layers", "layers", "alpha", "n",
        "tsr", "n_passed",
        "eval_aware_rate", "n_aware", "n_judged",
    ]
    out_csv = Path(args.output_csv)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}", file=sys.stderr)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    md_lines = [
        "| cell | N | layers | alpha | n | TSR | eval_aware_rate | n_aware/n_judged |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join([
                r["cell"], fmt(r["n_layers"]), r["layers"] or "-",
                fmt(r["alpha"]), fmt(r["n"]),
                fmt(r["tsr"]), fmt(r["eval_aware_rate"]),
                f"{r['n_aware']}/{r['n_judged']}",
            ])
            + " |"
        )
    print()
    print("\n".join(md_lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cell_dirs", nargs="+",
                    help="Cell directories (e.g. .../cells/N6_layers9_8_10_alpha0.6).")
    ap.add_argument("--suite", default="workspace",
                    help="AgentDojo suite name; per-task JSONs are read from "
                         "<cell_dir>/<suite>/cell/*.json.")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--judge-concurrency", type=int, default=32)
    ap.add_argument("--output-csv", default="/tmp/agentdojo_partial_judged.csv")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
