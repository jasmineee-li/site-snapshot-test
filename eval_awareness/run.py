"""Main experiment runner for eval awareness pipeline.

Usage:
    python -m eval_awareness.run configs/pilot.yaml
"""

import asyncio
import json
import random
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .llm import LLMClient, LLMConfig
from .prompts import (
    PromptTemplate,
    load_all_prompts,
    render_classify,
    render_compare,
    render_open_ended,
)
from .sampler import DataItem, load_manifest, make_pairs, sample_items


def load_config(config_path: Path) -> dict:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_response(raw: str, template: PromptTemplate) -> str | None:
    """Extract the choice from a free-text response.

    For classify: find which choice appears in the response.
    For compare: find "A" or "B".
    For open_ended: return raw text.
    """
    if template.type == "open_ended":
        return raw.strip()

    if not template.choices:
        return None

    raw_upper = raw.strip().upper()

    # First check if the response starts with a choice
    for choice in template.choices:
        if raw_upper.startswith(choice.upper()):
            return choice

    # Then check if exactly one choice appears (word boundary match)
    found = [
        c for c in template.choices if re.search(r"\b" + re.escape(c.upper()) + r"\b", raw_upper)
    ]
    if len(found) == 1:
        return found[0]

    return None


async def _run_single_classify(
    client: LLMClient,
    template: PromptTemplate,
    item: DataItem,
    modality: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single classify evaluation and return result dict."""
    item_dict = item.to_dict()
    messages, images = render_classify(template, item_dict, modality)

    async with semaphore:
        t0 = time.monotonic()
        raw_response = await client.call(messages, images or None)
        elapsed = time.monotonic() - t0

    parsed = parse_response(raw_response, template)

    # Determine correctness for classify prompts
    correct = None
    if template.label_mapping and parsed:
        predicted_label = template.label_mapping.get(parsed)
        correct = predicted_label == item.label

    return {
        "type": "classify",
        "item_id": item.id,
        "source": item.source,
        "ground_truth": item.label,
        "prompt_id": template.id,
        "model": client.config.model,
        "provider": client.config.provider,
        "modality": modality,
        "raw_response": raw_response,
        "parsed_answer": parsed,
        "correct": correct,
        "latency_s": round(elapsed, 2),
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def _run_single_compare(
    client: LLMClient,
    template: PromptTemplate,
    item_a: DataItem,
    item_b: DataItem,
    modality: str,
    semaphore: asyncio.Semaphore,
    rng: random.Random,
) -> dict:
    """Run a single pairwise comparison and return result dict."""
    item_a_dict = item_a.to_dict()
    item_b_dict = item_b.to_dict()
    messages, images, position_map = render_compare(
        template, item_a_dict, item_b_dict, modality, rng
    )

    async with semaphore:
        t0 = time.monotonic()
        raw_response = await client.call(messages, images or None)
        elapsed = time.monotonic() - t0

    parsed = parse_response(raw_response, template)

    # Resolve which item was chosen
    chosen_id = position_map.get(parsed) if parsed else None

    return {
        "type": "compare",
        "item_a_id": item_a.id,
        "item_b_id": item_b.id,
        "source_a": item_a.source,
        "source_b": item_b.source,
        "position_map": position_map,
        "prompt_id": template.id,
        "model": client.config.model,
        "provider": client.config.provider,
        "modality": modality,
        "raw_response": raw_response,
        "parsed_answer": parsed,
        "chosen_id": chosen_id,
        "latency_s": round(elapsed, 2),
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def _run_single_open_ended(
    client: LLMClient,
    template: PromptTemplate,
    item: DataItem,
    modality: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single open-ended evaluation and return result dict."""
    item_dict = item.to_dict()
    messages, images = render_open_ended(template, item_dict, modality)

    async with semaphore:
        t0 = time.monotonic()
        raw_response = await client.call(messages, images or None)
        elapsed = time.monotonic() - t0

    return {
        "type": "open_ended",
        "item_id": item.id,
        "source": item.source,
        "ground_truth": item.label,
        "prompt_id": template.id,
        "model": client.config.model,
        "provider": client.config.provider,
        "modality": modality,
        "raw_response": raw_response,
        "latency_s": round(elapsed, 2),
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def run_experiment(config_path: Path):
    """Run a full experiment from a config file."""
    config = load_config(config_path)
    base_dir = Path(__file__).parent

    # Resolve paths relative to eval_awareness/
    manifest_path = base_dir / config["data"]["manifest"]
    prompts_dir = base_dir / "prompts"
    results_dir = base_dir / "results"

    exp_name = config["name"]
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    shutil.copy(config_path, exp_dir / "config.yaml")

    # Load prompts
    all_prompts = load_all_prompts(prompts_dir)
    prompt_ids = config["prompts"]
    templates = [all_prompts[pid] for pid in prompt_ids]

    # Load models
    clients = []
    for m in config["models"]:
        cfg = LLMConfig(
            provider=m["provider"],
            model=m["model"],
            temperature=m.get("temperature", 0.0),
            max_tokens=m.get("max_tokens", 1024),
        )
        clients.append(LLMClient(cfg))

    modality = config.get("modality", "html")
    concurrency = config.get("concurrency", 10)
    semaphore = asyncio.Semaphore(concurrency)

    # Load manifest
    manifest = load_manifest(manifest_path)

    is_pairwise = "pairwise" in config
    tasks = []

    if is_pairwise:
        pw_cfg = config["pairwise"]
        pairs = make_pairs(
            manifest,
            pair_sources=(pw_cfg["source_a"], pw_cfg["source_b"]),
            n_pairs=pw_cfg["n_pairs"],
            seed=pw_cfg.get("seed", 42),
        )
        compare_templates = [t for t in templates if t.type == "compare"]
        for template in compare_templates:
            for client in clients:
                for item_a, item_b in pairs:
                    rng = random.Random(f"{item_a.id}-{item_b.id}-{template.id}")
                    tasks.append(
                        _run_single_compare(
                            client, template, item_a, item_b, modality, semaphore, rng
                        )
                    )
    else:
        sampling_cfg = config["sampling"]
        items = sample_items(
            manifest,
            per_source=sampling_cfg["per_source"],
            modality_filter=[modality] if modality else None,
            seed=sampling_cfg.get("seed", 42),
        )
        for template in templates:
            for client in clients:
                if template.type == "compare":
                    continue  # skip compare templates in classify mode
                for item in items:
                    if template.type == "open_ended":
                        tasks.append(
                            _run_single_open_ended(client, template, item, modality, semaphore)
                        )
                    else:
                        tasks.append(
                            _run_single_classify(client, template, item, modality, semaphore)
                        )

    print(f"Running {len(tasks)} evaluations with concurrency={concurrency}...")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Write results
    output_path = exp_dir / "raw_responses.jsonl"
    n_success = 0
    n_errors = 0
    with open(output_path, "w") as f:
        for r in results:
            if isinstance(r, Exception):
                n_errors += 1
                f.write(json.dumps({"error": str(r)}) + "\n")
            else:
                n_success += 1
                f.write(json.dumps(r) + "\n")

    print(f"Done. {n_success} successes, {n_errors} errors.")
    print(f"Results written to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m eval_awareness.run <config.yaml>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    asyncio.run(run_experiment(config_path))


if __name__ == "__main__":
    main()
