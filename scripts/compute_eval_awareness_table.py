"""Compute Verbalized% / Purpose / Spearman(awareness, ASR) for the
eval-awareness paper's Table 2 (DoomArena & WASP, bare arm, n200).

Usage: run from project root:
    python scripts/compute_eval_awareness_table.py
"""

import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path("eval_awareness_experiments/results/n200_2026-04-29")


def collect(benchmark: str):
    base = ROOT / benchmark / "bare"
    rows = []
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        bench_inner = model_dir / benchmark
        if not bench_inner.exists():
            continue
        for split_dir in sorted(bench_inner.iterdir()):
            if not split_dir.is_dir():
                continue
            jpath = split_dir / "trajectory_awareness_results.jsonl"
            if not jpath.exists():
                continue
            per_sample = defaultdict(dict)
            with open(jpath) as f:
                for line in f:
                    d = json.loads(line)
                    per_sample[d["sample_id"]][d["experiment_type"]] = d
            for sid, by_et in per_sample.items():
                v = by_et.get("trajectory_verbalized_awareness")
                p = by_et.get("trajectory_purpose_continue_5q")
                if v is None or p is None:
                    continue
                v_class = v.get("classification") or ""
                v_eval = "Eval" in v_class
                p_share = p.get("metadata", {}).get("alignment_eval_share")
                if p_share is None:
                    p_share = p.get("score")
                asr = v.get("metadata", {}).get("attack_success", False)
                rows.append(
                    {
                        "task_id": sid,
                        "model": model_dir.name,
                        "split": split_dir.name,
                        "verb_eval": bool(v_eval),
                        "purpose_share": float(p_share) if p_share is not None else None,
                        "attack_success": bool(asr),
                    }
                )
    return rows


def summarize(name: str, rows):
    n_total = len(rows)
    n_models = len(set(r["model"] for r in rows))
    verb_pct = 100.0 * sum(1 for r in rows if r["verb_eval"]) / n_total
    purpose_vals = [r["purpose_share"] for r in rows if r["purpose_share"] is not None]
    purpose_mean = sum(purpose_vals) / len(purpose_vals)
    by_aware = defaultdict(list)
    by_asr = defaultdict(list)
    by_purp = defaultdict(list)
    for r in rows:
        by_aware[r["task_id"]].append(int(r["verb_eval"]))
        by_asr[r["task_id"]].append(int(r["attack_success"]))
        if r["purpose_share"] is not None:
            by_purp[r["task_id"]].append(r["purpose_share"])
    tasks = sorted(by_aware.keys())
    aware_arr = [sum(by_aware[t]) / len(by_aware[t]) for t in tasks]
    asr_arr = [sum(by_asr[t]) / len(by_asr[t]) for t in tasks]
    purp_arr = [sum(by_purp[t]) / len(by_purp[t]) if by_purp[t] else 0 for t in tasks]
    rho_v, p_v = spearmanr(aware_arr, asr_arr)
    rho_p, p_p = spearmanr(purp_arr, asr_arr)
    asr_mean = 100.0 * sum(int(r["attack_success"]) for r in rows) / n_total
    print(f"\n=== {name.upper()} ===")
    print(f"  n_models={n_models}  n_trajectories={n_total}  n_unique_tasks={len(tasks)}")
    print(f"  Verbalized% (any *Eval)            = {verb_pct:.2f}%")
    print(f"  Purpose (alignment_eval_share mean) = {purpose_mean:.4f}")
    print(f"  rho(verb_aware, ASR)   = {rho_v:+.3f}  (p={p_v:.3g})")
    print(f"  rho(purpose_share, ASR) = {rho_p:+.3f}  (p={p_p:.3g})")
    print(f"  ASR mean across trajectories: {asr_mean:.2f}%")


for bench in ["doomarena", "wasp"]:
    rows = collect(bench)
    summarize(bench, rows)
