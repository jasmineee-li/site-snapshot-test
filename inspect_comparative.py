"""Inspect comparative results: rebuild prompts and print alongside model responses.

Usage:
    python inspect_comparative.py --format html --pair real webarena-infinity
    python inspect_comparative.py --format html --pair real worldsim
    python inspect_comparative.py --format html --pair webarena-infinity worldsim
    python inspect_comparative.py --format screenshot --pair real webarena-infinity
"""

import argparse
import json
from pathlib import Path

from eval_awareness_experiments.experiments.prompts import (
    JUDGE_PROMPTS,
    build_comparative_prompt,
    build_comparative_prompt_screenshot,
)
from eval_awareness_experiments.run import load_manifest

FORMAT_LABELS = {
    "html": "HTML source code",
    "axtree": "accessibility tree",
    "screenshot": "screenshot",
}

RESULT_FILES = {
    "html": {
        "opus-4.6": "eval_awareness_experiments/results/overnight_comparative_worldsim/anthropic_claude-opus-4.6:thinking/comparative_results.jsonl",
        "gpt-5.4-mini": "eval_awareness_experiments/results/overnight_comparative_worldsim/openai_gpt-5.4-mini:thinking/comparative_results.jsonl",
    },
    "screenshot": {
        "opus-4.6": "eval_awareness_experiments/results/overnight_comparative_worldsim_screenshot/anthropic_claude-opus-4.6:thinking/comparative_results.jsonl",
        "gpt-5.4-mini": "eval_awareness_experiments/results/overnight_comparative_worldsim_screenshot/openai_gpt-5.4-mini:thinking/comparative_results.jsonl",
    },
}


def trunc(s: str, n: int = 800) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated, total {len(s)} chars] ..."


def load_content(sample, format_type: str) -> str | None:
    if format_type == "html" and sample.html_path:
        p = Path(sample.html_path)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    elif format_type == "screenshot" and sample.screenshot_path:
        return str(sample.screenshot_path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True, choices=["html", "screenshot"])
    parser.add_argument(
        "--pair", nargs=2, required=True, help="Two sources to pair, e.g. real webarena-infinity"
    )
    parser.add_argument("--model", default="opus-4.6", choices=["opus-4.6", "gpt-5.4-mini"])
    parser.add_argument("--n-examples", type=int, default=1, help="How many pair-instances to show")
    parser.add_argument(
        "--n-prompts", type=int, default=2, help="How many of the 8 judge prompts to show per pair"
    )
    parser.add_argument(
        "--content-chars", type=int, default=800, help="Max chars of content to show (html only)"
    )
    args = parser.parse_args()

    src_a_filter, src_b_filter = args.pair
    wanted = {src_a_filter, src_b_filter}

    # Load manifest to get sample path lookup
    samples = load_manifest(Path("eval_awareness_experiments/data/manifest.json"))
    sample_by_id = {s.id: s for s in samples}

    # Load results
    path = RESULT_FILES[args.format][args.model]
    results = [json.loads(l) for l in open(path)]

    # Filter to requested source pair
    matching = []
    for r in results:
        md = r["metadata"]
        srcs = {md["sample_a_source"], md["sample_b_source"]}
        if srcs == wanted:
            matching.append(r)

    if not matching:
        print(f"No pairs found for {args.pair}")
        return

    print(f"Found {len(matching)} pair-runs for {args.pair} in {args.format} ({args.model})")
    print("=" * 100)

    format_label = FORMAT_LABELS[args.format]
    shown = 0
    for r in matching:
        if shown >= args.n_examples:
            break
        md = r["metadata"]
        if md.get("order") != "AB":
            continue  # only show AB to avoid showing same pair twice
        shown += 1

        a_id = md["sample_a_id"]
        b_id = md["sample_b_id"]
        a_src = md["sample_a_source"]
        b_src = md["sample_b_source"]

        print(f"\n\n{'#' * 100}")
        print(f"# PAIR {shown}: A={a_src}:{a_id}")
        print(f"#         B={b_src}:{b_id}")
        print(f"#         website_type={md.get('website_type')}")
        print(
            f"#         score (A-as-eval fraction)={r.get('score')}  modal_choice={r.get('classification')}"
        )
        print(f"{'#' * 100}")

        sample_a = sample_by_id.get(a_id)
        sample_b = sample_by_id.get(b_id)
        if not sample_a or not sample_b:
            print("WARN: missing sample in manifest")
            continue

        content_a = load_content(sample_a, args.format)
        content_b = load_content(sample_b, args.format)

        if args.format == "html":
            print(f"\n--- Content A ({a_src}, {a_id}) ---")
            print(trunc(content_a or "", args.content_chars))
            print(f"\n--- Content B ({b_src}, {b_id}) ---")
            print(trunc(content_b or "", args.content_chars))
        else:
            print(f"\n--- Screenshot A: {content_a}")
            print(f"--- Screenshot B: {content_b}")

        # Show a few of the 8 judge prompts with their responses
        prompt_results = md.get("prompt_results", [])
        # strict=False: a partial result file legitimately has fewer entries
        # than JUDGE_PROMPTS, and the loop below stops early anyway.
        for i, (pt, pr) in enumerate(zip(JUDGE_PROMPTS, prompt_results, strict=False)):
            if i >= args.n_prompts:
                break
            label, json_key, preamble, question = pt

            if args.format == "html":
                full_prompt = build_comparative_prompt(
                    json_key,
                    preamble,
                    question,
                    format_label,
                    content_a or "",
                    content_b or "",
                )
                # Don't print the full prompt (has content), just the framing
                preamble_filled = preamble.format(format_type=format_label)
                prompt_header = f"{preamble_filled}\n\nEnvironment A:\n[content above]\n\nEnvironment B:\n[content above]\n\n{question}"
            else:
                full_prompt = build_comparative_prompt_screenshot(
                    json_key,
                    preamble,
                    question,
                    format_label,
                )
                prompt_header = full_prompt

            print(f"\n\n>>> Judge prompt #{i+1} ({label}-framed, key='{json_key}')")
            print("-" * 80)
            print(prompt_header)
            print("-" * 80)
            print("MODEL RESPONSE:")
            print(pr.get("raw_response", "[no response]"))
            print(f"\nParsed choice: {pr.get('choice')}  confidence: {pr.get('confidence')}")
            print(f"chosen_eval_source: {pr.get('chosen_eval_source')}")
            if pr.get("error"):
                print(f"ERROR: {pr['error']}")


if __name__ == "__main__":
    main()
