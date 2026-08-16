"""Results aggregation and plotting for eval awareness experiments.

Usage:
    python -m eval_awareness_experiments.analysis --results-dir results/ --output figures/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir: Path) -> list[dict]:
    """Load all JSONL result files from a directory tree."""
    results = []
    for jsonl_file in results_dir.rglob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def aggregate_p_eval(results: list[dict]) -> dict:
    """Aggregate P(eval) scores by source, format, and website type.

    Returns dict with structure:
        {source: {format: {website_type: {"scores": [...], "mean": float, "se": float}}}}
    """
    # Filter to probability results
    prob_results = [r for r in results if r.get("experiment_type") == "p_eval_probability"]

    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in prob_results:
        source = r.get("metadata", {}).get("source", "unknown")
        fmt = r.get("format_used", "unknown")
        wtype = r.get("metadata", {}).get("website_type", "unknown")
        score = r.get("score")
        if score is not None:
            agg[source][fmt][wtype].append(score)

    # Compute means and SEs
    summary = {}
    for source, formats in agg.items():
        summary[source] = {}
        for fmt, wtypes in formats.items():
            summary[source][fmt] = {}
            for wtype, scores in wtypes.items():
                arr = np.array(scores)
                summary[source][fmt][wtype] = {
                    "scores": scores,
                    "mean": float(arr.mean()),
                    "se": float(arr.std() / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                    "n": len(scores),
                }

    return summary


def aggregate_comparative(results: list[dict]) -> dict:
    """Aggregate comparative results into a win-rate matrix.

    Returns dict with structure:
        {(source_a, source_b): {"a_wins": int, "b_wins": int, "total": int}}
    """
    comp_results = [r for r in results if r.get("experiment_type") == "comparative"]

    wins = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "total": 0})

    for r in comp_results:
        meta = r.get("metadata", {})
        src_a = meta.get("sample_a_source")
        src_b = meta.get("sample_b_source")
        choice = meta.get("choice")

        if not (src_a and src_b and choice):
            continue

        key = tuple(sorted([src_a, src_b]))
        wins[key]["total"] += 1

        if choice == "A":
            # The model chose A as more eval-like
            if meta.get("sample_a_source") == key[0]:
                wins[key]["a_wins"] += 1
            else:
                wins[key]["b_wins"] += 1
        elif choice == "B":
            if meta.get("sample_b_source") == key[0]:
                wins[key]["a_wins"] += 1
            else:
                wins[key]["b_wins"] += 1

    return dict(wins)


def print_p_eval_summary(summary: dict) -> None:
    """Print P(eval) summary table."""
    print("\n=== P(eval) Probability Scores ===")
    print(f"{'Source':<15} {'Format':<12} {'Website':<10} {'Mean':>6} {'SE':>6} {'N':>4}")
    print("-" * 60)

    for source in sorted(summary.keys()):
        for fmt in sorted(summary[source].keys()):
            for wtype in sorted(summary[source][fmt].keys()):
                s = summary[source][fmt][wtype]
                print(f"{source:<15} {fmt:<12} {wtype:<10} {s['mean']:>6.3f} {s['se']:>6.3f} {s['n']:>4d}")


def print_comparative_summary(wins: dict) -> None:
    """Print comparative win-rate summary."""
    print("\n=== Comparative Win Rates ===")
    print(f"{'Pair':<30} {'Source A Wins':>14} {'Source B Wins':>14} {'Total':>6}")
    print("-" * 70)

    for (src_a, src_b), w in sorted(wins.items()):
        pair_str = f"{src_a} vs {src_b}"
        print(f"{pair_str:<30} {w['a_wins']:>14d} {w['b_wins']:>14d} {w['total']:>6d}")


def plot_p_eval_bars(summary: dict, output_path: Path) -> None:
    """Create bar chart of P(eval) scores by source and format."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Aggregate across website types for each source × format
    source_format_scores = defaultdict(lambda: defaultdict(list))
    for source, formats in summary.items():
        for fmt, wtypes in formats.items():
            for _wtype, stats in wtypes.items():
                source_format_scores[source][fmt].extend(stats["scores"])

    sources = sorted(source_format_scores.keys())
    all_formats = sorted(set(f for s in source_format_scores.values() for f in s.keys()))

    _fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sources))
    width = 0.25

    for i, fmt in enumerate(all_formats):
        means = []
        ses = []
        for src in sources:
            scores = source_format_scores[src].get(fmt, [])
            if scores:
                arr = np.array(scores)
                means.append(arr.mean())
                ses.append(arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else 0)
            else:
                means.append(0)
                ses.append(0)

        ax.bar(x + i * width, means, width, yerr=ses, label=fmt, capsize=3)

    ax.set_ylabel("P(eval)")
    ax.set_title("P(eval) Scores by Source and Format\n(error bars = ±1 SE)")
    ax.set_xticks(x + width * (len(all_formats) - 1) / 2)
    ax.set_xticklabels(sources)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / "p_eval_by_source_format.png", dpi=150)
    plt.close()
    print(f"  Plot saved to {output_path / 'p_eval_by_source_format.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze eval awareness experiment results")
    parser.add_argument("--results-dir", type=str, default="eval_awareness_experiments/results",
                        help="Directory containing result JSONL files")
    parser.add_argument("--output", type=str, default="eval_awareness_experiments/figures",
                        help="Output directory for plots")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export aggregated results to CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} results")

    # P(eval) summary
    p_eval_summary = aggregate_p_eval(results)
    if p_eval_summary:
        print_p_eval_summary(p_eval_summary)
        plot_p_eval_bars(p_eval_summary, Path(args.output))

    # Comparative summary
    comp_summary = aggregate_comparative(results)
    if comp_summary:
        print_comparative_summary(comp_summary)

    # Export CSV if requested
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sample_id", "experiment_type", "model", "format_used",
                "score", "classification", "source", "website_type",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "sample_id": r.get("sample_id", ""),
                    "experiment_type": r.get("experiment_type", ""),
                    "model": r.get("model", ""),
                    "format_used": r.get("format_used", ""),
                    "score": r.get("score", ""),
                    "classification": r.get("classification", ""),
                    "source": r.get("metadata", {}).get("source", ""),
                    "website_type": r.get("metadata", {}).get("website_type", ""),
                })
        print(f"\nCSV exported to {args.csv}")


if __name__ == "__main__":
    main()
