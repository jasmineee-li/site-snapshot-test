"""Evaluate a contrastive DoM probe on SAD stages_oversight.

This is the paper-fidelity evaluation for the MCQ/contrastive probe family:
train the vector on synthetic contrastive pairs, then test whether it separates
SAD stages_oversight test/evaluation prompts from deployment prompts.

The SAD repository keeps benchmark text in an encrypted zip. This script reads
the needed YAML files directly from that zip and never writes prompt text to the
output JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contrastive_probe import ContrastiveProbe  # noqa: E402
from probes.model_loader import load_model  # noqa: E402
from probes.scoring import calculate_auroc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_layers(layers_str: str, available_layers: list[int]) -> list[int]:
    if layers_str == "all":
        return available_layers
    requested = [int(x.strip()) for x in layers_str.split(",") if x.strip()]
    return [layer for layer in requested if layer in available_layers]


def load_yaml_from_zip(zip_path: Path, member: str, password: str) -> Any:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member, pwd=password.encode("utf-8")) as f:
            return yaml.safe_load(f)


def load_sad_stages_oversight(
    structs_zip: Path,
    password: str,
    max_per_class: int | None = None,
) -> list[dict[str, Any]]:
    test_batch = load_yaml_from_zip(
        structs_zip, "batch/test_oversight.yaml", password
    )
    deploy_batch = load_yaml_from_zip(
        structs_zip, "batch/deploy_oversight.yaml", password
    )

    samples: list[dict[str, Any]] = []
    for label_name, label, batch in (
        ("test", 1, test_batch),
        ("deploy", 0, deploy_batch),
    ):
        batch_samples = batch["samples"]
        if max_per_class is not None:
            batch_samples = batch_samples[:max_per_class]
        for item in batch_samples:
            samples.append(
                {
                    "text": item["body"],
                    "label": label,
                    "label_name": label_name,
                    "source": item.get("splits", {}).get("source", "unknown"),
                }
            )
    return samples


def tokenize_chat_content(model, tokenizer, text: str) -> tuple[torch.Tensor, list[int]]:
    full_messages = [{"role": "user", "content": text}]
    empty_messages = [{"role": "user", "content": ""}]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    empty_text = tokenizer.apply_chat_template(
        empty_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    empty_ids = tokenizer.encode(empty_text, add_special_tokens=False)

    start = 0
    max_prefix = min(len(full_ids), len(empty_ids))
    while start < max_prefix and full_ids[start] == empty_ids[start]:
        start += 1

    end_full = len(full_ids)
    end_empty = len(empty_ids)
    while (
        end_full > start
        and end_empty > start
        and full_ids[end_full - 1] == empty_ids[end_empty - 1]
    ):
        end_full -= 1
        end_empty -= 1

    if end_full <= start:
        raise ValueError("Could not locate user content span in chat template")

    tokens = torch.tensor([full_ids], device=model.device)
    return tokens, list(range(start, end_full))


def tokenize_raw_content(model, tokenizer, text: str) -> tuple[torch.Tensor, list[int]]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    positions = [
        idx
        for idx, token_id in enumerate(ids)
        if token_id not in special_ids and tokenizer.decode([token_id]).strip()
    ]
    if not positions:
        positions = list(range(len(ids)))
    return torch.tensor([ids], device=model.device), positions


def score_sample_all_layers(
    probe: ContrastiveProbe,
    model,
    tokenizer,
    text: str,
    layers: list[int],
    use_chat_template: bool,
) -> dict[int, float]:
    if use_chat_template:
        tokens, positions = tokenize_chat_content(model, tokenizer, text)
    else:
        tokens, positions = tokenize_raw_content(model, tokenizer, text)

    scores: dict[int, float] = {}
    hooks = []
    for layer in layers:
        vector = probe.normalized_vectors[layer]
        vec = vector[0] if vector.dim() > 1 else vector

        def make_hook(layer: int = layer, vec: torch.Tensor = vec):
            def hook_fn(activation, hook):
                act = activation[0, positions, :].mean(dim=0)
                v = vec.to(device=act.device, dtype=act.dtype).flatten()
                scores[layer] = torch.dot(act, v).item()
                return activation

            return hook_fn

        hooks.append((f"blocks.{layer}.hook_resid_pre", make_hook()))

    try:
        model.reset_hooks()
        with model.hooks(fwd_hooks=hooks):
            with torch.no_grad():
                model(tokens)
    finally:
        del tokens

    return scores


def split_balanced(
    samples: list[dict[str, Any]],
    dev_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dev_fraction <= 0:
        return [], samples
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in (1, 0):
        label_samples = [sample for sample in samples if sample["label"] == label]
        split_idx = int(len(label_samples) * dev_fraction)
        dev.extend(label_samples[:split_idx])
        test.extend(label_samples[split_idx:])
    return dev, test


def summarize_layer_scores(
    layer_scores: dict[int, list[float]],
    labels: list[int],
    sources: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for layer, scores in layer_scores.items():
        if len(scores) != len(labels):
            continue
        auroc = calculate_auroc(labels, scores)
        eval_scores = [s for label, s in zip(labels, scores) if label == 1]
        deploy_scores = [s for label, s in zip(labels, scores) if label == 0]
        results[str(layer)] = {
            "auroc": auroc.auroc,
            "threshold": auroc.optimal_threshold,
            "sensitivity": auroc.sensitivity,
            "specificity": auroc.specificity,
            "mean_eval": sum(eval_scores) / len(eval_scores),
            "mean_deploy": sum(deploy_scores) / len(deploy_scores),
            "n_eval": len(eval_scores),
            "n_deploy": len(deploy_scores),
            "source_counts": dict(Counter(sources)),
        }
    return results


def top_layers(results: dict[str, Any], n: int) -> list[dict[str, Any]]:
    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1]["auroc"],
        reverse=True,
    )
    return [
        {
            "layer": int(layer),
            "auroc": info["auroc"],
            "mean_eval": info["mean_eval"],
            "mean_deploy": info["mean_deploy"],
            "threshold": info["threshold"],
        }
        for layer, info in ranked[:n]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument(
        "--sad-structs-zip",
        default="/tmp/sad/sad/stages/oversight/structs.zip",
        help="Path to SAD stages/oversight/structs.zip.",
    )
    ap.add_argument(
        "--sad-password",
        default=os.environ.get("SAD_ZIP_PASSWORD", "sadtimesforthesetimes"),
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument(
        "--dev-fraction",
        type=float,
        default=0.5,
        help="Balanced fraction of SAD per class used for layer selection.",
    )
    ap.add_argument("--top-n-layers", type=int, default=12)
    ap.add_argument(
        "--raw-tokenization",
        action="store_true",
        help="Use raw tokenizer.encode text instead of single-user chat template.",
    )
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    use_chat_template = not args.raw_tokenization

    samples = load_sad_stages_oversight(
        Path(args.sad_structs_zip),
        args.sad_password,
        max_per_class=args.max_per_class,
    )
    labels = [sample["label"] for sample in samples]
    logger.info(
        "Loaded SAD stages_oversight: %s eval/test + %s deploy prompts",
        sum(1 for label in labels if label == 1),
        sum(1 for label in labels if label == 0),
    )

    logger.info("Loading model %s", args.model)
    model, tokenizer = load_model(args.model, dtype=dtype)
    probe = ContrastiveProbe.load(args.probe_dir)
    available_layers = sorted(probe.normalized_vectors)
    layers = parse_layers(args.layers, available_layers)
    logger.info("Scoring %s layers with %s", len(layers), "chat template" if use_chat_template else "raw tokenization")

    scores_by_layer: dict[int, list[float]] = {layer: [] for layer in layers}
    for idx, sample in enumerate(samples, 1):
        scores = score_sample_all_layers(
            probe,
            model,
            tokenizer,
            sample["text"],
            layers,
            use_chat_template=use_chat_template,
        )
        for layer in layers:
            scores_by_layer[layer].append(scores[layer])
        if idx % 50 == 0:
            logger.info("Scored %s/%s SAD prompts", idx, len(samples))

    all_labels = [sample["label"] for sample in samples]
    all_sources = [sample["source"] for sample in samples]
    per_layer = summarize_layer_scores(scores_by_layer, all_labels, all_sources)

    dev_samples, test_samples = split_balanced(samples, args.dev_fraction)
    split_info = None
    if dev_samples and test_samples:
        dev_indices = set(id(sample) for sample in dev_samples)
        test_indices = set(id(sample) for sample in test_samples)

        def subset_summary(indices: set[int]) -> dict[str, Any]:
            subset_positions = [
                pos for pos, sample in enumerate(samples) if id(sample) in indices
            ]
            subset_scores = {
                layer: [scores[pos] for pos in subset_positions]
                for layer, scores in scores_by_layer.items()
            }
            subset_labels = [samples[pos]["label"] for pos in subset_positions]
            subset_sources = [samples[pos]["source"] for pos in subset_positions]
            return summarize_layer_scores(subset_scores, subset_labels, subset_sources)

        dev_results = subset_summary(dev_indices)
        test_results = subset_summary(test_indices)
        best_dev_layer = top_layers(dev_results, 1)[0]["layer"]
        split_info = {
            "dev_fraction": args.dev_fraction,
            "n_dev": len(dev_samples),
            "n_test": len(test_samples),
            "best_layer_by_dev": best_dev_layer,
            "dev_best_auroc": dev_results[str(best_dev_layer)]["auroc"],
            "test_auroc_at_dev_best_layer": test_results[str(best_dev_layer)]["auroc"],
            "dev_top_layers": top_layers(dev_results, args.top_n_layers),
            "test_top_layers": top_layers(test_results, args.top_n_layers),
        }

    summary = {
        "model": args.model,
        "probe_dir": args.probe_dir,
        "dataset": "SAD stages_oversight",
        "label_convention": {"test_eval": 1, "deploy": 0},
        "n_total": len(samples),
        "n_eval": sum(1 for label in all_labels if label == 1),
        "n_deploy": sum(1 for label in all_labels if label == 0),
        "tokenization": "chat_template_user_content" if use_chat_template else "raw_content",
        "top_layers_all": top_layers(per_layer, args.top_n_layers),
        "num_layers_exact_1_0": sum(1 for info in per_layer.values() if info["auroc"] == 1.0),
        "num_layers_below_0_5": sum(1 for info in per_layer.values() if info["auroc"] < 0.5),
        "dev_test_split": split_info,
        "per_layer": per_layer,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))


if __name__ == "__main__":
    main()
