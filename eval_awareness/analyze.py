"""Aggregate results and compute metrics.

Usage:
    python -m eval_awareness.analyze results/pilot_run/
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _accuracy(items: list[dict]) -> dict:
    """Compute accuracy from a list of classify result dicts."""
    scorable = [r for r in items if r.get("correct") is not None]
    if not scorable:
        return {"n": 0, "accuracy": None}
    correct = sum(1 for r in scorable if r["correct"])
    return {"n": len(scorable), "accuracy": round(correct / len(scorable), 4)}


def _confusion_matrix(items: list[dict]) -> dict:
    """Build a confusion matrix from classify results."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in items:
        if r.get("parsed_answer") is None or r.get("ground_truth") is None:
            continue
        gt = r["ground_truth"]
        pred = r["parsed_answer"]
        matrix[gt][pred] += 1
    return {gt: dict(preds) for gt, preds in matrix.items()}


def _pairwise_stats(items: list[dict]) -> dict:
    """Compute pairwise preference rates and position bias."""
    if not items:
        return {}

    total = len(items)
    parseable = [r for r in items if r.get("parsed_answer") is not None]

    # Position bias: how often A vs B was chosen
    position_counts = Counter(r["parsed_answer"] for r in parseable)

    # Source preference: how often each source was chosen as "more real"
    source_chosen = Counter()
    for r in parseable:
        chosen_id = r.get("chosen_id")
        if not chosen_id:
            continue
        # Figure out which source was chosen
        pos_map = r["position_map"]
        for pos, item_id in pos_map.items():
            if item_id == chosen_id:
                # Determine source from the result
                if pos == "A":
                    # Check if A maps to source_a or source_b based on position_map
                    pass
                break
        # Use source_a/source_b from the result
        if chosen_id == pos_map.get("A"):
            # item at position A — could be from either source
            # We need to check which source this item belongs to
            if r.get("source_a") and r.get("source_b"):
                # The original item_a is always source_a, item_b always source_b
                # But position_map tells us which position each was assigned
                if pos_map["A"] == r.get("item_a_id"):
                    source_chosen[r["source_a"]] += 1
                else:
                    source_chosen[r["source_b"]] += 1
        elif chosen_id == pos_map.get("B"):
            if pos_map["B"] == r.get("item_a_id"):
                source_chosen[r["source_a"]] += 1
            else:
                source_chosen[r["source_b"]] += 1

    return {
        "total": total,
        "parseable": len(parseable),
        "position_bias": {
            k: round(v / len(parseable), 4) if parseable else 0
            for k, v in position_counts.items()
        },
        "source_preference": {
            k: round(v / len(parseable), 4) if parseable else 0
            for k, v in source_chosen.items()
        },
    }


def analyze(results_dir: Path):
    """Analyze experiment results and produce summary."""
    raw_path = results_dir / "raw_responses.jsonl"
    responses = load_jsonl(raw_path)

    # Separate by type
    classify = [r for r in responses if r.get("type") == "classify"]
    compare = [r for r in responses if r.get("type") == "compare"]
    open_ended = [r for r in responses if r.get("type") == "open_ended"]
    errors = [r for r in responses if "error" in r]

    summary = {
        "total_responses": len(responses),
        "errors": len(errors),
    }

    # --- Classify metrics ---
    if classify:
        summary["classify"] = {}
        summary["classify"]["overall"] = _accuracy(classify)
        summary["classify"]["confusion_matrix"] = _confusion_matrix(classify)

        # By source
        by_source: dict[str, list] = defaultdict(list)
        for r in classify:
            by_source[r["source"]].append(r)
        summary["classify"]["by_source"] = {
            src: _accuracy(items) for src, items in sorted(by_source.items())
        }

        # By prompt
        by_prompt: dict[str, list] = defaultdict(list)
        for r in classify:
            by_prompt[r["prompt_id"]].append(r)
        summary["classify"]["by_prompt"] = {
            pid: _accuracy(items) for pid, items in sorted(by_prompt.items())
        }

        # By model
        by_model: dict[str, list] = defaultdict(list)
        for r in classify:
            by_model[r["model"]].append(r)
        summary["classify"]["by_model"] = {
            model: _accuracy(items) for model, items in sorted(by_model.items())
        }

    # --- Pairwise metrics ---
    if compare:
        summary["pairwise"] = _pairwise_stats(compare)

        # Pairwise by model
        by_model: dict[str, list] = defaultdict(list)
        for r in compare:
            by_model[r["model"]].append(r)
        summary["pairwise"]["by_model"] = {
            model: _pairwise_stats(items) for model, items in sorted(by_model.items())
        }

    # --- Open-ended ---
    if open_ended:
        summary["open_ended"] = {"count": len(open_ended)}

    # Write summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Experiment: {results_dir.name}")
    print(f"{'='*60}")
    print(f"Total responses: {summary['total_responses']}")
    print(f"Errors: {summary['errors']}")

    if classify:
        c = summary["classify"]
        print("\n--- Classify ---")
        ov = c["overall"]
        print(f"Overall accuracy: {ov['accuracy']} (n={ov['n']})")

        print("\nBy source:")
        for src, acc in c["by_source"].items():
            print(f"  {src}: {acc['accuracy']} (n={acc['n']})")

        print("\nBy prompt:")
        for pid, acc in c["by_prompt"].items():
            print(f"  {pid}: {acc['accuracy']} (n={acc['n']})")

        print("\nBy model:")
        for model, acc in c["by_model"].items():
            print(f"  {model}: {acc['accuracy']} (n={acc['n']})")

        if c.get("confusion_matrix"):
            print("\nConfusion matrix (ground_truth -> parsed_answer):")
            for gt, preds in c["confusion_matrix"].items():
                print(f"  {gt}: {preds}")

    if compare:
        pw = summary["pairwise"]
        print("\n--- Pairwise ---")
        print(f"Total: {pw['total']}, Parseable: {pw['parseable']}")
        print(f"Position bias: {pw['position_bias']}")
        print(f"Source preference: {pw['source_preference']}")

    if open_ended:
        print("\n--- Open-Ended ---")
        print(f"Count: {summary['open_ended']['count']}")

    print(f"\nSummary written to {summary_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m eval_awareness.analyze <results_dir>")
        sys.exit(1)
    results_dir = Path(sys.argv[1])
    analyze(results_dir)


if __name__ == "__main__":
    main()
