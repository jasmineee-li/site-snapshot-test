"""Train a difference-of-means eval-awareness probe.

Adapted from eval-awareness/eval-awareness-testbed/scripts/train_probe.py.
Trims the original (which also supported attention probes + classifier
probes) down to just the contrastive-DoM path we need here.

Usage:
    python -m probes.train_probe \\
        --model xlangai/OpenCUA-32B \\
        --data probes/data/contrastive_dataset.json \\
        --output probes/trained/opencua-32b/ \\
        --layers all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.activation_extraction import extract_contrastive_activations
from probes.contrastive_probe import ContrastiveProbe, train_logreg_direction
from probes.model_loader import load_model
from probes.scoring import calculate_auroc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_layers(layers_str: str, n_layers: int) -> list[int]:
    if layers_str == "all":
        return list(range(n_layers))
    return [int(x.strip()) for x in layers_str.split(",")]


def load_contrastive_data(data_path: str) -> list[dict]:
    with open(data_path) as f:
        return json.load(f)


def load_framing_data(data_path: str) -> dict:
    """Load `{real: [...], fake: [...]}` framing prompts."""
    with open(data_path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or "real" not in data or "fake" not in data:
        raise ValueError(
            f"Framing data at {data_path} must be a JSON object with " f"'real' and 'fake' keys."
        )
    return data


def _eval_framing_probe(
    model,
    tokenizer,
    probe: ContrastiveProbe,
    eval_real: list[str],
    eval_fake: list[str],
    layers: list[int],
) -> dict[int, dict]:
    """Per-layer AUROC over a held-out framing split.

    Convention: vector = real - fake, so higher score = more real. We
    label real=1, fake=0 and use `mean_score` directly. AUROC >> 0.5 means
    the probe separates real-deployment framings from eval framings.
    """
    results: dict[int, dict] = {}
    for layer in layers:
        if layer not in probe.normalized_vectors:
            continue
        labels: list[int] = []
        scores: list[float] = []
        for prompt in eval_real:
            try:
                s = probe.score(model, prompt, layer=layer, tokenizer=tokenizer)
                labels.append(1)
                scores.append(s.mean_score)
            except Exception as e:
                logger.warning(f"Error scoring real prompt: {e}")
        for prompt in eval_fake:
            try:
                s = probe.score(model, prompt, layer=layer, tokenizer=tokenizer)
                labels.append(0)
                scores.append(s.mean_score)
            except Exception as e:
                logger.warning(f"Error scoring fake prompt: {e}")
        if len(set(labels)) >= 2 and len(labels) >= 4:
            auroc_result = calculate_auroc(labels, scores)
            results[layer] = {
                "auroc": auroc_result.auroc,
                "threshold": auroc_result.optimal_threshold,
                "sensitivity": auroc_result.sensitivity,
                "specificity": auroc_result.specificity,
            }
    return results


def train_framing(
    model,
    tokenizer,
    data_path: str,
    output_dir: str,
    layers: list[int],
    eval_split: float,
    token_position: int,
    top_n_layers: int,
) -> dict[str, Any]:
    """Train a framing-pair DoM probe (paper-fidelity)."""
    framing = load_framing_data(data_path)
    n_real, n_fake = len(framing["real"]), len(framing["fake"])
    logger.info(f"Loaded {n_real} real + {n_fake} fake framing prompts")

    real_split = max(1, int(n_real * (1 - eval_split)))
    fake_split = max(1, int(n_fake * (1 - eval_split)))
    train_data = {
        "real": framing["real"][:real_split],
        "fake": framing["fake"][:fake_split],
    }
    eval_real = framing["real"][real_split:]
    eval_fake = framing["fake"][fake_split:]
    logger.info(
        f"Train: {len(train_data['real'])} real + {len(train_data['fake'])} fake; "
        f"Eval: {len(eval_real)} real + {len(eval_fake)} fake"
    )

    probe = ContrastiveProbe()
    probe.train_framing(model, train_data, layers, tokenizer, token_position=token_position)

    results: dict[int, dict] = {}
    if eval_real and eval_fake:
        logger.info("Evaluating framing probe on held-out split...")
        results = _eval_framing_probe(model, tokenizer, probe, eval_real, eval_fake, layers)

    # Best layer + top-N by AUROC. If the held-out split was empty (e.g.
    # eval_split=0), fall back to the middle layer as a sane default.
    if results:
        ranked = sorted(results.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
        best_layer, best_info = ranked[0]
        best_auroc = best_info["auroc"]
        top_layers = [l for l, _ in ranked[:top_n_layers]]
        threshold = best_info["threshold"]
    else:
        logger.warning("No held-out evaluation data; selecting best_layer = middle layer.")
        best_layer = layers[len(layers) // 2]
        best_auroc = float("nan")
        top_layers = layers[:top_n_layers]
        threshold = 0.0

    probe.best_layer = best_layer
    probe.threshold = threshold
    probe.metadata = {
        "model": model.cfg.model_name if hasattr(model, "cfg") else "unknown",
        "probe_method": "framing_dom",
        "training_data": data_path,
        "training_date": datetime.now().isoformat(),
        "num_train_real": len(train_data["real"]),
        "num_train_fake": len(train_data["fake"]),
        "num_eval_real": len(eval_real),
        "num_eval_fake": len(eval_fake),
        "token_position": token_position,
        "top_layers": top_layers,
    }

    probe.save(output_dir)

    per_layer_path = Path(output_dir) / "per_layer_auroc.json"
    with open(per_layer_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    logger.info(f"Per-layer AUROC saved to {per_layer_path}")

    return {
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "top_layers": top_layers,
        "per_layer": results,
    }


def train_contrastive(
    model,
    tokenizer,
    data_path: str,
    output_dir: str,
    layers: list[int],
    eval_split: float,
    probe_method: str = "dom",
) -> dict[str, Any]:
    import numpy as np  # noqa: F401 (used by logreg path)

    dataset = load_contrastive_data(data_path)
    logger.info(f"Loaded {len(dataset)} contrastive pairs from {data_path}")

    split_idx = int(len(dataset) * (1 - eval_split))
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]
    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    probe = ContrastiveProbe()

    if probe_method == "logreg":
        positive_acts, negative_acts = extract_contrastive_activations(
            model, train_data, layers, tokenizer
        )
        for layer in layers:
            pos_list = positive_acts[layer]
            neg_list = negative_acts[layer]
            if not pos_list or not neg_list:
                logger.warning(f"Layer {layer}: missing activations, skipping")
                continue
            pos_np = torch.stack(pos_list).squeeze(1).cpu().float().numpy()
            neg_np = torch.stack(neg_list).squeeze(1).cpu().float().numpy()
            direction, info = train_logreg_direction(pos_np, neg_np)
            ref = pos_list[0]
            direction_tensor = torch.tensor(direction, dtype=ref.dtype, device=ref.device)
            probe.vectors[layer] = direction_tensor
            probe.normalized_vectors[layer] = direction_tensor
            logger.info(f"  Layer {layer}: logreg cv_auroc={info['cv_auroc']:.4f}")
    else:
        probe.train(model, train_data, layers, tokenizer)

    # Per-layer AUROC on held-out contrastive split
    results: dict[int, dict] = {}
    best_auroc = 0.0
    best_layer = layers[0]

    logger.info("Evaluating probe on held-out data...")
    for layer in layers:
        if layer not in probe.normalized_vectors:
            continue
        labels: list[int] = []
        scores: list[float] = []
        for item in eval_data:
            question = item["question"]
            pos = item.get("test") or item.get("positive")
            neg = item.get("deploy") or item.get("negative")
            try:
                pos_score = probe.score(
                    model, f"{question}\n\n {pos}", layer=layer, tokenizer=tokenizer
                )
                labels.append(1)
                scores.append(pos_score.mean_score)
            except Exception as e:
                logger.warning(f"Error scoring positive: {e}")
            try:
                neg_score = probe.score(
                    model, f"{question}\n\n {neg}", layer=layer, tokenizer=tokenizer
                )
                labels.append(0)
                scores.append(neg_score.mean_score)
            except Exception as e:
                logger.warning(f"Error scoring negative: {e}")
        if len(labels) >= 4:
            auroc_result = calculate_auroc(labels, scores)
            results[layer] = {
                "auroc": auroc_result.auroc,
                "threshold": auroc_result.optimal_threshold,
                "sensitivity": auroc_result.sensitivity,
                "specificity": auroc_result.specificity,
            }
            if auroc_result.auroc > best_auroc:
                best_auroc = auroc_result.auroc
                best_layer = layer

    top_layers = [
        l
        for l, _ in sorted(
            results.items(),
            key=lambda kv: kv[1].get("auroc", 0.0),
            reverse=True,
        )
    ][:6]

    probe.best_layer = best_layer
    probe.threshold = results.get(best_layer, {}).get("threshold", 0.0)
    probe.metadata = {
        "model": model.cfg.model_name if hasattr(model, "cfg") else "unknown",
        "probe_method": probe_method,
        "training_data": data_path,
        "training_date": datetime.now().isoformat(),
        "num_train": len(train_data),
        "num_eval": len(eval_data),
        "top_layers": top_layers,
    }

    probe.save(output_dir)

    per_layer_path = Path(output_dir) / "per_layer_auroc.json"
    with open(per_layer_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    logger.info(f"Per-layer AUROC saved to {per_layer_path}")

    return {"best_layer": best_layer, "best_auroc": best_auroc, "per_layer": results}


def main():
    parser = argparse.ArgumentParser(description="Train a DoM eval-awareness probe")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument(
        "--data",
        default="probes/data/contrastive_dataset.json",
        help="Path to contrastive training data",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer indices or 'all'",
    )
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16/float16/float32")
    parser.add_argument("--revision", default=None, help="Model revision")
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument(
        "--probe-method",
        choices=["dom", "logreg"],
        default="dom",
        help="dom (difference-of-means) or logreg (L2 logistic regression). "
        "Only applies to --dataset-format mcq.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["mcq", "framing"],
        default="mcq",
        help="mcq: list of {question, test, deploy} pairs (extract at "
        "answer-token position). framing: {real: [...], fake: [...]} "
        "colon-terminated prompts (extract at colon, position -2). "
        "The framing path is the paper-fidelity option (Hua et al. 2025).",
    )
    parser.add_argument(
        "--token-position",
        type=int,
        default=-2,
        help="Token position for framing extraction (default -2 = colon).",
    )
    parser.add_argument(
        "--top-n-layers",
        type=int,
        default=6,
        help="How many top-AUROC layers to record in probe_info.json under "
        "`top_layers` (used by multi-layer steering; paper uses 6).",
    )
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(
        args.model, device=args.device, dtype=dtype, revision=args.revision
    )

    n_layers = model.cfg.n_layers if hasattr(model, "cfg") else 32
    layers = parse_layers(args.layers, n_layers)
    logger.info(f"Training on layers: {layers}")

    if args.dataset_format == "framing":
        results = train_framing(
            model,
            tokenizer,
            args.data,
            args.output,
            layers,
            eval_split=args.eval_split,
            token_position=args.token_position,
            top_n_layers=args.top_n_layers,
        )
    else:
        results = train_contrastive(
            model,
            tokenizer,
            args.data,
            args.output,
            layers,
            args.eval_split,
            probe_method=args.probe_method,
        )

    print("\n" + "=" * 70)
    print(f"{'Layer':<8} {'AUROC':<10} {'Threshold':<12} {'Sensitivity':<14} {'Specificity':<14}")
    print("-" * 70)
    for layer in sorted(results["per_layer"].keys()):
        r = results["per_layer"][layer]
        if "auroc" not in r:
            continue
        print(
            f"{layer:<8} {r['auroc']:<10.4f} {r['threshold']:<12.4f} "
            f"{r['sensitivity']:<14.4f} {r['specificity']:<14.4f}"
        )
    print("-" * 70)
    print(f"Best layer: {results['best_layer']} (AUROC: {results['best_auroc']:.4f})")
    if "top_layers" in results:
        print(f"Top layers (multi-layer steering): {results['top_layers']}")
    print(f"Probe saved to: {args.output}")


if __name__ == "__main__":
    main()
