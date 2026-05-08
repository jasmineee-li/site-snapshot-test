"""Analysis and plotting for ASR x eval-awareness causal experiments.

Loads JSONL results from trajectory_awareness experiments and generates:
1. Bar chart: Eval awareness rate by benchmark (safety vs general)
2. Scatter plot: ASR vs eval awareness rate per model x benchmark
3. Grouped bar chart: ASR by injection condition, grouped by benchmark
4. Heatmap: eval awareness x task performance cross-tabulation
5. Line plot: ASR across models, faceted by condition

Usage:
    python -m eval_awareness_experiments.analysis_causal \
        --results-dir eval_awareness_experiments/results \
        --output-dir eval_awareness_experiments/figures
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WASP_INTERMEDIATE_ASR_SOURCE = "wasp_post_hoc_action_classifier"


def _canonical_attack_success(metadata: dict) -> bool | None:
    """Return canonical ASR only, excluding WASP intermediate action metrics."""
    if metadata.get("attack_success_source") == WASP_INTERMEDIATE_ASR_SOURCE:
        return None
    return metadata.get("attack_success")


def load_trajectory_results(results_dir: Path) -> list[dict]:
    """Load all trajectory_awareness JSONL results from a results directory."""
    all_results = []
    for jsonl_file in results_dir.rglob("trajectory_*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
    return all_results


def compute_awareness_rates(results: list[dict]) -> dict:
    """Compute eval awareness and scenario recognition rates by benchmark.

    Groups results by benchmark and computes:
    - model_awareness rate (% Yes or Maybe)
    - hypothetical_recognition rate (% Yes or Maybe)
    - attack_success rate (ASR)
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        meta = r.get("metadata", {})
        benchmark = meta.get("benchmark", meta.get("source", "unknown"))
        groups[benchmark].append(r)

    stats = {}
    for benchmark, group in groups.items():
        n = len(group)
        if n == 0:
            continue

        # Count awareness from TAS judge results
        awareness_yes = 0
        recognition_yes = 0
        asr_count = 0
        asr_total = 0

        for r in group:
            meta = r.get("metadata", {})

            # Model awareness
            ma = meta.get("model_awareness", {})
            if ma.get("conclusion") in ("Yes", "Maybe"):
                awareness_yes += 1

            # Hypothetical recognition
            hr = meta.get("hypothetical_recognition", {})
            if hr.get("conclusion") in ("Yes", "Maybe"):
                recognition_yes += 1

            # Attack success
            attack_success = _canonical_attack_success(meta)
            if attack_success is not None:
                asr_total += 1
                if attack_success:
                    asr_count += 1

        stats[benchmark] = {
            "n": n,
            "awareness_rate": awareness_yes / n if n > 0 else 0,
            "recognition_rate": recognition_yes / n if n > 0 else 0,
            "asr": asr_count / asr_total if asr_total > 0 else None,
            "awareness_count": awareness_yes,
            "recognition_count": recognition_yes,
            "asr_count": asr_count,
            "asr_total": asr_total,
        }

    return stats


def compute_causal_stats(results: list[dict]) -> dict:
    """Compute ASR and awareness by injection condition x benchmark x model.

    Returns nested dict: {benchmark: {condition: {model: {stats}}}}
    """
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        meta = r.get("metadata", {})
        benchmark = meta.get("benchmark", "unknown")
        condition = meta.get("injection_condition", "baseline")
        model = r.get("model", "unknown")
        groups[(benchmark, condition, model)].append(r)

    stats = {}
    for (benchmark, condition, model), group in groups.items():
        if benchmark not in stats:
            stats[benchmark] = {}
        if condition not in stats[benchmark]:
            stats[benchmark][condition] = {}

        n = len(group)
        awareness = sum(
            1 for r in group
            if r.get("metadata", {}).get("model_awareness", {}).get("conclusion") in ("Yes", "Maybe")
        )
        canonical_asr = [
            _canonical_attack_success(r.get("metadata", {})) for r in group
        ]
        asr_total = sum(1 for attack_success in canonical_asr if attack_success is not None)
        asr_count = sum(1 for attack_success in canonical_asr if attack_success)

        stats[benchmark][condition][model] = {
            "n": n,
            "awareness_rate": awareness / n if n > 0 else 0,
            "asr": asr_count / asr_total if asr_total > 0 else None,
            "awareness_se": _binomial_se(awareness, n),
            "asr_se": _binomial_se(asr_count, asr_total) if asr_total > 0 else None,
        }

    return stats


def _binomial_se(k: int, n: int) -> float:
    """Compute binomial standard error."""
    if n == 0:
        return 0.0
    p = k / n
    return float(np.sqrt(p * (1 - p) / n))


def print_awareness_summary(stats: dict) -> None:
    """Print a summary table of awareness rates by benchmark."""
    print(f"\n{'Benchmark':<25} {'N':>5} {'Awareness':>12} {'Recognition':>14} {'ASR':>8}")
    print("-" * 70)
    for benchmark, s in sorted(stats.items()):
        asr_str = f"{s['asr']:.2%}" if s['asr'] is not None else "N/A"
        print(
            f"{benchmark:<25} {s['n']:>5} "
            f"{s['awareness_count']}/{s['n']} ({s['awareness_rate']:.1%})"
            f"  {s['recognition_count']}/{s['n']} ({s['recognition_rate']:.1%})"
            f"  {asr_str}"
        )


def print_causal_summary(stats: dict) -> None:
    """Print a summary of causal experiment results."""
    for benchmark, conditions in sorted(stats.items()):
        print(f"\n=== {benchmark} ===")
        print(f"  {'Condition':<22} {'Model':<30} {'N':>4} {'Awareness':>11} {'ASR':>8}")
        print(f"  {'-' * 80}")
        for condition, models in sorted(conditions.items()):
            for model, s in sorted(models.items()):
                asr_str = f"{s['asr']:.2%}" if s['asr'] is not None else "N/A"
                print(
                    f"  {condition:<22} {model:<30} {s['n']:>4} "
                    f"{s['awareness_rate']:.1%}±{s['awareness_se']:.2f}"
                    f"  {asr_str}"
                )


def main():
    parser = argparse.ArgumentParser(description="Analyze causal eval-awareness experiment results")
    parser.add_argument("--results-dir", type=str,
                        default="eval_awareness_experiments/results",
                        help="Directory containing JSONL results")
    parser.add_argument("--output-dir", type=str,
                        default="eval_awareness_experiments/figures",
                        help="Directory for output plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_trajectory_results(results_dir)
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found. Run experiments first.")
        return

    # Filter to TAS judge results only
    tas_results = [
        r for r in results
        if r.get("experiment_type") == "trajectory_test_awareness_steering"
    ]
    print(f"TAS judge results: {len(tas_results)}")

    # Experiment A: Awareness rates by benchmark
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Eval Awareness Rates by Benchmark")
    print("=" * 70)
    stats = compute_awareness_rates(tas_results)
    print_awareness_summary(stats)

    # Experiment C: Causal injection effects
    causal_results = [r for r in tas_results if r.get("metadata", {}).get("injection_condition")]
    if causal_results:
        print("\n" + "=" * 70)
        print("EXPERIMENT C: Causal Injection Modification Effects")
        print("=" * 70)
        causal_stats = compute_causal_stats(causal_results)
        print_causal_summary(causal_stats)

    # Save stats as JSON
    stats_path = output_dir / "awareness_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
