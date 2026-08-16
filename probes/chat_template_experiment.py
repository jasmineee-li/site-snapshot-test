"""Train and cross-validate DoM probes using the model's chat template.

This mirrors the GUI-Owl handoff experiment, but renders MCQ prompts as
`user -> assistant` chat turns and renders framing prompts as single user
messages before extracting activations. It intentionally lives beside the
original training code so raw-token and chat-template results remain directly
comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.activation_extraction import (  # noqa: E402
    TokenSelector,
    extract_activations_hooked,
)
from probes.contrastive_probe import ContrastiveProbe  # noqa: E402
from probes.eval_sad_stages_oversight import (  # noqa: E402
    load_sad_stages_oversight,
    score_sample_all_layers as score_sad_sample_all_layers,
    split_balanced,
    summarize_layer_scores as summarize_sad_layer_scores,
    top_layers as sad_top_layers,
)
from probes.model_loader import load_model  # noqa: E402
from probes.scoring import calculate_auroc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_layers(layers_str: str, n_layers: int) -> list[int]:
    if layers_str == "all":
        return list(range(n_layers))
    return [int(x.strip()) for x in layers_str.split(",") if x.strip()]


def tokenize_chat(model, tokenizer, messages: list[dict[str, str]]) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor([ids], device=model.device)


def tokenize_chat_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def inserted_assistant_content_span(
    tokenizer,
    question: str,
    answer: str,
) -> tuple[list[int], tuple[int, int]]:
    """Return full chat token ids and the span inserted by assistant content."""
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    empty_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    full_ids = tokenize_chat_ids(tokenizer, full_messages)
    empty_ids = tokenize_chat_ids(tokenizer, empty_messages)

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
        raise ValueError("Could not locate assistant answer content in chat template")
    return full_ids, (start, end_full)


def extract_chat_answer_activations(
    model,
    tokenizer,
    question: str,
    answer: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    ids, answer_span = inserted_assistant_content_span(tokenizer, question, answer)
    tokens = torch.tensor([ids], device=model.device)
    try:
        return extract_activations_hooked(
            model,
            tokens,
            layers,
            token_selector=TokenSelector.EVIDENCE_TOKENS,
            tokenizer=tokenizer,
            evidence_indices=[answer_span],
        )
    finally:
        del tokens


def extract_chat_prompt_position(
    model,
    tokenizer,
    prompt: str,
    layers: list[int],
    token_position: int,
) -> dict[int, torch.Tensor]:
    tokens = tokenize_chat(model, tokenizer, [{"role": "user", "content": prompt}])
    try:
        return extract_activations_hooked(
            model,
            tokens,
            layers,
            token_selector=TokenSelector.POSITION,
            tokenizer=tokenizer,
            position_index=token_position,
        )
    finally:
        del tokens


def train_vectors_from_pairs(
    pos_acts: dict[int, list[torch.Tensor]],
    neg_acts: dict[int, list[torch.Tensor]],
    layers: list[int],
) -> ContrastiveProbe:
    probe = ContrastiveProbe()
    for layer in layers:
        if not pos_acts[layer] or not neg_acts[layer]:
            logger.warning("Layer %s missing activations; skipping", layer)
            continue
        pos_mean = torch.stack(pos_acts[layer]).mean(dim=0)
        neg_mean = torch.stack(neg_acts[layer]).mean(dim=0)
        vector = pos_mean - neg_mean
        probe.vectors[layer] = vector
        norm = vector.norm()
        probe.normalized_vectors[layer] = vector / norm if norm > 0 else vector
    return probe


def score_activation(probe: ContrastiveProbe, act: torch.Tensor, layer: int) -> float:
    vec = probe.normalized_vectors[layer]
    vec = vec[0] if vec.dim() > 1 else vec
    act = act.squeeze(0) if act.dim() > 1 else act
    vec = vec.to(device=act.device, dtype=act.dtype)
    return torch.dot(act, vec).item()


def score_chat_mean(
    probe: ContrastiveProbe,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    layer: int,
) -> float:
    tokens = tokenize_chat(model, tokenizer, messages)
    vector = probe.normalized_vectors[layer]
    vec = vector[0] if vector.dim() > 1 else vector
    scores: list[float] = []
    # Decode once, up here: `hook_fn` must not close over `tokens`, because the
    # `finally` below deletes that binding. A hook that outlived the `with`
    # block and read `tokens` would raise NameError instead of scoring.
    token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]

    def hook_fn(activation, hook):
        v = vec.to(device=activation.device, dtype=activation.dtype)
        for pos, tok in enumerate(token_strings):
            if not tok.strip() or tok.startswith("<|") or tok in {"<s>", "</s>"}:
                continue
            scores.append(torch.dot(activation[0, pos, :], v).item())
        return activation

    try:
        model.reset_hooks()
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]):
            with torch.no_grad():
                model(tokens)
    finally:
        del tokens
    return sum(scores) / len(scores) if scores else 0.0


def eval_mcq_probe_answer_position(
    probe: ContrastiveProbe,
    model,
    tokenizer,
    data: list[dict[str, str]],
    layers: list[int],
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for layer in layers:
        if layer not in probe.normalized_vectors:
            continue
        labels: list[int] = []
        scores: list[float] = []
        for item in data:
            q = item["question"]
            pos = item.get("positive") or item.get("test")
            neg = item.get("negative") or item.get("deploy")
            pos_act = extract_chat_answer_activations(model, tokenizer, q, pos, [layer])[layer]
            neg_act = extract_chat_answer_activations(model, tokenizer, q, neg, [layer])[layer]
            labels += [1, 0]
            scores += [
                score_activation(probe, pos_act, layer),
                score_activation(probe, neg_act, layer),
            ]
        if len(set(labels)) >= 2:
            auroc = calculate_auroc(labels, scores)
            results[layer] = {
                "auroc": auroc.auroc,
                "threshold": auroc.optimal_threshold,
                "sensitivity": auroc.sensitivity,
                "specificity": auroc.specificity,
            }
    return results


def train_mcq_chat(
    model,
    tokenizer,
    data: list[dict[str, str]],
    layers: list[int],
    eval_split: float,
    output_dir: Path,
    top_n_layers: int,
) -> tuple[ContrastiveProbe, dict[str, Any]]:
    if not 0 <= eval_split < 1:
        raise ValueError("--mcq-eval-split must be in [0, 1).")

    split_idx = len(data) if eval_split == 0 else max(1, int(len(data) * (1 - eval_split)))
    train_data, eval_data = data[:split_idx], data[split_idx:]
    logger.info("MCQ chat split: train=%s eval=%s", len(train_data), len(eval_data))

    pos_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    neg_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for i, item in enumerate(train_data, 1):
        q = item["question"]
        pos = item.get("positive") or item.get("test")
        neg = item.get("negative") or item.get("deploy")
        pos_cached = extract_chat_answer_activations(model, tokenizer, q, pos, layers)
        neg_cached = extract_chat_answer_activations(model, tokenizer, q, neg, layers)
        for layer in layers:
            if layer in pos_cached:
                pos_acts[layer].append(pos_cached[layer])
            if layer in neg_cached:
                neg_acts[layer].append(neg_cached[layer])
        if i % 25 == 0:
            logger.info("MCQ train activations: %s/%s pairs", i, len(train_data))

    probe = train_vectors_from_pairs(pos_acts, neg_acts, layers)
    results: dict[int, dict[str, float]] = {}
    ranked: list[tuple[int, dict[str, float]]] = []
    if eval_data:
        results = eval_mcq_probe_answer_position(probe, model, tokenizer, eval_data, layers)
        ranked = sorted(results.items(), key=lambda kv: kv[1]["auroc"], reverse=True)

    best_layer = ranked[0][0] if ranked else None
    best_info = ranked[0][1] if ranked else None
    top_layers = [l for l, _ in ranked[:top_n_layers]] if ranked else []
    probe.best_layer = best_layer
    probe.threshold = best_info["threshold"] if best_info else 0.0
    probe.metadata = {
        "model": getattr(model.cfg, "model_name", "unknown"),
        "probe_method": "chat_template_mcq_dom",
        "training_data": "probes/data/contrastive_dataset.json",
        "training_date": datetime.now().isoformat(),
        "num_train": len(train_data),
        "num_eval": len(eval_data),
        "top_layers": top_layers,
        "layer_selection": "mcq_heldout" if ranked else "pending",
        "chat_template": True,
        "mcq_format": "user question + assistant answer",
        "extraction_position": "assistant answer content span",
    }
    probe.save(output_dir)
    (output_dir / "per_layer_auroc.json").write_text(
        json.dumps({str(k): v for k, v in results.items()}, indent=2)
    )
    return probe, {
        "best_layer": best_layer,
        "best_auroc": best_info["auroc"] if best_info else None,
        "top_layers": top_layers,
        "selection_source": "mcq_heldout" if ranked else "pending",
        "per_layer": results,
    }


def evaluate_mcq_probe_on_sad(
    probe: ContrastiveProbe,
    model,
    tokenizer,
    layers: list[int],
    structs_zip: Path,
    password: str,
    output_path: Path,
    dev_fraction: float,
    max_per_class: int | None,
    top_n_layers: int,
) -> dict[str, Any]:
    """Evaluate an in-memory MCQ probe on SAD and choose the default layer."""
    samples = load_sad_stages_oversight(
        structs_zip,
        password,
        max_per_class=max_per_class,
    )
    labels = [sample["label"] for sample in samples]
    eval_count = sum(1 for label in labels if label == 1)
    deploy_count = sum(1 for label in labels if label == 0)
    logger.info(
        "Loaded SAD stages_oversight: %s eval/test + %s deploy prompts",
        eval_count,
        deploy_count,
    )

    scored_layers = [layer for layer in layers if layer in probe.normalized_vectors]
    if not scored_layers:
        raise ValueError("No requested layers have trained MCQ vectors.")

    scores_by_layer: dict[int, list[float]] = {layer: [] for layer in scored_layers}
    for idx, sample in enumerate(samples, 1):
        scores = score_sad_sample_all_layers(
            probe,
            model,
            tokenizer,
            sample["text"],
            scored_layers,
            use_chat_template=True,
        )
        for layer in scored_layers:
            scores_by_layer[layer].append(scores[layer])
        if idx % 50 == 0:
            logger.info("Scored %s/%s SAD prompts", idx, len(samples))

    all_sources = [sample["source"] for sample in samples]
    per_layer = summarize_sad_layer_scores(scores_by_layer, labels, all_sources)

    dev_samples, test_samples = split_balanced(samples, dev_fraction)
    split_info = None
    selected_layer: int
    selected_threshold: float
    selected_auroc: float
    selection_source: str

    if dev_samples and test_samples:
        dev_indices = set(id(sample) for sample in dev_samples)
        test_indices = set(id(sample) for sample in test_samples)

        def subset_summary(indices: set[int]) -> dict[str, Any]:
            subset_positions = [pos for pos, sample in enumerate(samples) if id(sample) in indices]
            subset_scores = {
                layer: [scores[pos] for pos in subset_positions]
                for layer, scores in scores_by_layer.items()
            }
            subset_labels = [samples[pos]["label"] for pos in subset_positions]
            subset_sources = [samples[pos]["source"] for pos in subset_positions]
            return summarize_sad_layer_scores(
                subset_scores,
                subset_labels,
                subset_sources,
            )

        dev_results = subset_summary(dev_indices)
        test_results = subset_summary(test_indices)
        selected_layer = sad_top_layers(dev_results, 1)[0]["layer"]
        selected_threshold = dev_results[str(selected_layer)]["threshold"]
        selected_auroc = dev_results[str(selected_layer)]["auroc"]
        selection_source = "sad_stages_oversight_dev"
        split_info = {
            "dev_fraction": dev_fraction,
            "n_dev": len(dev_samples),
            "n_test": len(test_samples),
            "best_layer_by_dev": selected_layer,
            "dev_best_auroc": selected_auroc,
            "test_auroc_at_dev_best_layer": test_results[str(selected_layer)]["auroc"],
            "dev_top_layers": sad_top_layers(dev_results, top_n_layers),
            "test_top_layers": sad_top_layers(test_results, top_n_layers),
        }
    else:
        top_all = sad_top_layers(per_layer, 1)[0]
        selected_layer = top_all["layer"]
        selected_threshold = top_all["threshold"]
        selected_auroc = top_all["auroc"]
        selection_source = "sad_stages_oversight_all"

    summary = {
        "model": getattr(model.cfg, "model_name", "unknown"),
        "probe_dir": str(output_path.parent / "mcq"),
        "dataset": "SAD stages_oversight",
        "label_convention": {"test_eval": 1, "deploy": 0},
        "n_total": len(samples),
        "n_eval": eval_count,
        "n_deploy": deploy_count,
        "tokenization": "chat_template_user_content",
        "selected_layer": selected_layer,
        "selected_threshold": selected_threshold,
        "selected_auroc": selected_auroc,
        "selection_source": selection_source,
        "top_layers_all": sad_top_layers(per_layer, top_n_layers),
        "num_layers_exact_1_0": sum(1 for info in per_layer.values() if info["auroc"] == 1.0),
        "num_layers_below_0_5": sum(1 for info in per_layer.values() if info["auroc"] < 0.5),
        "dev_test_split": split_info,
        "per_layer": per_layer,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def eval_framing_probe_position(
    probe: ContrastiveProbe,
    model,
    tokenizer,
    real_prompts: list[str],
    fake_prompts: list[str],
    layers: list[int],
    token_position: int,
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for layer in layers:
        if layer not in probe.normalized_vectors:
            continue
        labels: list[int] = []
        scores: list[float] = []
        for prompt in real_prompts:
            act = extract_chat_prompt_position(model, tokenizer, prompt, [layer], token_position)[
                layer
            ]
            labels.append(1)
            scores.append(score_activation(probe, act, layer))
        for prompt in fake_prompts:
            act = extract_chat_prompt_position(model, tokenizer, prompt, [layer], token_position)[
                layer
            ]
            labels.append(0)
            scores.append(score_activation(probe, act, layer))
        if len(set(labels)) >= 2:
            auroc = calculate_auroc(labels, scores)
            results[layer] = {
                "auroc": auroc.auroc,
                "threshold": auroc.optimal_threshold,
                "sensitivity": auroc.sensitivity,
                "specificity": auroc.specificity,
            }
    return results


def train_framing_chat(
    model,
    tokenizer,
    framing: dict[str, list[str]],
    layers: list[int],
    eval_split: float,
    output_dir: Path,
    top_n_layers: int,
    token_position: int,
) -> tuple[ContrastiveProbe, dict[str, Any]]:
    real = framing["real"]
    fake = framing["fake"]
    real_split = max(1, int(len(real) * (1 - eval_split)))
    fake_split = max(1, int(len(fake) * (1 - eval_split)))
    train_real, eval_real = real[:real_split], real[real_split:]
    train_fake, eval_fake = fake[:fake_split], fake[fake_split:]
    logger.info(
        "Framing chat split: train=%s real + %s fake; eval=%s real + %s fake",
        len(train_real),
        len(train_fake),
        len(eval_real),
        len(eval_fake),
    )

    real_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    fake_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for prompt in train_real:
        cached = extract_chat_prompt_position(model, tokenizer, prompt, layers, token_position)
        for layer in layers:
            if layer in cached:
                real_acts[layer].append(cached[layer])
    for prompt in train_fake:
        cached = extract_chat_prompt_position(model, tokenizer, prompt, layers, token_position)
        for layer in layers:
            if layer in cached:
                fake_acts[layer].append(cached[layer])

    probe = train_vectors_from_pairs(real_acts, fake_acts, layers)
    results = eval_framing_probe_position(
        probe, model, tokenizer, eval_real, eval_fake, layers, token_position
    )
    ranked = sorted(results.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    best_layer, best_info = ranked[0]
    probe.best_layer = best_layer
    probe.threshold = best_info["threshold"]
    probe.metadata = {
        "model": getattr(model.cfg, "model_name", "unknown"),
        "probe_method": "chat_template_framing_dom",
        "training_data": "probes/data/framing_pairs.json",
        "training_date": datetime.now().isoformat(),
        "num_train_real": len(train_real),
        "num_train_fake": len(train_fake),
        "num_eval_real": len(eval_real),
        "num_eval_fake": len(eval_fake),
        "top_layers": [l for l, _ in ranked[:top_n_layers]],
        "chat_template": True,
        "framing_format": "single user message",
        "extraction_position": token_position,
    }
    probe.save(output_dir)
    (output_dir / "per_layer_auroc.json").write_text(
        json.dumps({str(k): v for k, v in results.items()}, indent=2)
    )
    return probe, {"best_layer": best_layer, "best_auroc": best_info["auroc"], "per_layer": results}


def cross_validate_chat(
    mcq_probe: ContrastiveProbe,
    framing_probe: ContrastiveProbe,
    model,
    tokenizer,
    mcq_data: list[dict[str, str]],
    framing: dict[str, list[str]],
) -> dict[str, Any]:
    mcq_layer = mcq_probe.best_layer
    framing_layer = framing_probe.best_layer
    if mcq_layer is None or framing_layer is None:
        raise ValueError("Both probes must have best_layer set.")

    labels_a: list[int] = []
    scores_a: list[float] = []
    for prompt in framing["real"]:
        scores_a.append(
            score_chat_mean(
                mcq_probe, model, tokenizer, [{"role": "user", "content": prompt}], mcq_layer
            )
        )
        labels_a.append(0)
    for prompt in framing["fake"]:
        scores_a.append(
            score_chat_mean(
                mcq_probe, model, tokenizer, [{"role": "user", "content": prompt}], mcq_layer
            )
        )
        labels_a.append(1)
    auroc_a = roc_auc_score(labels_a, scores_a)

    labels_b: list[int] = []
    scores_b: list[float] = []
    for i, item in enumerate(mcq_data, 1):
        q = item["question"]
        pos = item.get("positive") or item.get("test")
        neg = item.get("negative") or item.get("deploy")
        scores_b.append(
            score_chat_mean(
                framing_probe,
                model,
                tokenizer,
                [{"role": "user", "content": q}, {"role": "assistant", "content": pos}],
                framing_layer,
            )
        )
        labels_b.append(1)
        scores_b.append(
            score_chat_mean(
                framing_probe,
                model,
                tokenizer,
                [{"role": "user", "content": q}, {"role": "assistant", "content": neg}],
                framing_layer,
            )
        )
        labels_b.append(0)
        if i % 25 == 0:
            logger.info("Cross-val framing probe on MCQ: %s/%s pairs", i, len(mcq_data))
    # Framing vector is real - fake, so lower scores are more eval-like.
    auroc_b = roc_auc_score(labels_b, [-s for s in scores_b])

    def mean_for(labels: list[int], scores: list[float], label: int) -> float:
        # strict=False: lengths come from the caller, so keep the historical
        # truncate-to-shortest behavior rather than raising mid-experiment.
        vals = [s for l, s in zip(labels, scores, strict=False) if l == label]
        return sum(vals) / len(vals)

    cosine_at_mcq = None
    cosine_at_framing = None
    if mcq_layer in framing_probe.normalized_vectors:
        cosine_at_mcq = torch.nn.functional.cosine_similarity(
            mcq_probe.normalized_vectors[mcq_layer].flatten().float(),
            framing_probe.normalized_vectors[mcq_layer].flatten().float(),
            dim=0,
        ).item()
    if framing_layer in mcq_probe.normalized_vectors:
        cosine_at_framing = torch.nn.functional.cosine_similarity(
            mcq_probe.normalized_vectors[framing_layer].flatten().float(),
            framing_probe.normalized_vectors[framing_layer].flatten().float(),
            dim=0,
        ).item()

    return {
        "mcq_probe_to_framing": {
            "auroc": auroc_a,
            "mean_real": mean_for(labels_a, scores_a, 0),
            "mean_fake": mean_for(labels_a, scores_a, 1),
        },
        "framing_probe_to_mcq": {
            "auroc": auroc_b,
            "mean_eval_positive": mean_for(labels_b, scores_b, 1),
            "mean_deploy_negative": mean_for(labels_b, scores_b, 0),
        },
        "cosine": {
            "at_mcq_best_layer": cosine_at_mcq,
            "at_framing_best_layer": cosine_at_framing,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mPLUG/GUI-Owl-1.5-32B-Think")
    ap.add_argument("--mcq-data", default="probes/data/contrastive_dataset.json")
    ap.add_argument("--framing-data", default="probes/data/framing_pairs.json")
    ap.add_argument("--output-root", default="probes/trained/chat_template_gui_owl")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--mcq-layer-selection",
        choices=["sad", "heldout"],
        default="sad",
        help="Default is SAD stages_oversight dev selection; heldout is opt-in.",
    )
    ap.add_argument(
        "--mcq-eval-split",
        type=float,
        default=0.0,
        help="Synthetic contrastive held-out split. Default 0 trains on all MCQ pairs.",
    )
    ap.add_argument("--framing-eval-split", type=float, default=0.25)
    ap.add_argument("--framing-token-position", type=int, default=-2)
    ap.add_argument("--top-n-layers", type=int, default=6)
    ap.add_argument(
        "--sad-structs-zip",
        default="/tmp/sad/sad/stages/oversight/structs.zip",
        help="Path to SAD stages/oversight/structs.zip for MCQ layer selection.",
    )
    ap.add_argument(
        "--sad-password",
        default=os.environ.get("SAD_ZIP_PASSWORD", "sadtimesforthesetimes"),
    )
    ap.add_argument(
        "--sad-dev-fraction",
        type=float,
        default=0.5,
        help="Balanced fraction of SAD per class used to select the MCQ layer.",
    )
    ap.add_argument("--sad-max-per-class", type=int, default=None)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    mcq_data = json.loads(Path(args.mcq_data).read_text())
    framing = json.loads(Path(args.framing_data).read_text())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s", args.model)
    model, tokenizer = load_model(args.model, dtype=dtype)
    layers = parse_layers(args.layers, model.cfg.n_layers)
    logger.info("Using %s layers", len(layers))

    mcq_probe, mcq_results = train_mcq_chat(
        model,
        tokenizer,
        mcq_data,
        layers,
        args.mcq_eval_split,
        output_root / "mcq",
        args.top_n_layers,
    )

    sad_results = None
    if args.mcq_layer_selection == "sad":
        sad_path = output_root / "sad_stages_oversight_eval.json"
        sad_results = evaluate_mcq_probe_on_sad(
            mcq_probe,
            model,
            tokenizer,
            layers,
            Path(args.sad_structs_zip),
            args.sad_password,
            sad_path,
            args.sad_dev_fraction,
            args.sad_max_per_class,
            args.top_n_layers,
        )
        sad_top = (
            sad_results["dev_test_split"]["dev_top_layers"]
            if sad_results["dev_test_split"]
            else sad_results["top_layers_all"]
        )
        mcq_probe.best_layer = sad_results["selected_layer"]
        mcq_probe.threshold = sad_results["selected_threshold"]
        mcq_probe.metadata.update(
            {
                "layer_selection": sad_results["selection_source"],
                "sad_eval_output": str(sad_path),
                "sad_selected_auroc": sad_results["selected_auroc"],
                "sad_dev_fraction": args.sad_dev_fraction,
                "top_layers": [item["layer"] for item in sad_top],
            }
        )
        if sad_results["dev_test_split"]:
            mcq_probe.metadata["sad_test_auroc_at_selected_layer"] = sad_results["dev_test_split"][
                "test_auroc_at_dev_best_layer"
            ]
        mcq_probe.save(output_root / "mcq")
        mcq_results.update(
            {
                "best_layer": sad_results["selected_layer"],
                "best_auroc": sad_results["selected_auroc"],
                "top_layers": [item["layer"] for item in sad_top],
                "selection_source": sad_results["selection_source"],
            }
        )
    elif mcq_probe.best_layer is None:
        raise ValueError("--mcq-layer-selection heldout requires --mcq-eval-split > 0.")

    framing_probe, framing_results = train_framing_chat(
        model,
        tokenizer,
        framing,
        layers,
        args.framing_eval_split,
        output_root / "framing",
        args.top_n_layers,
        args.framing_token_position,
    )
    cross = cross_validate_chat(mcq_probe, framing_probe, model, tokenizer, mcq_data, framing)

    summary = {
        "model": args.model,
        "created_at": datetime.now().isoformat(),
        "mcq": {
            "best_layer": mcq_results["best_layer"],
            "best_auroc": mcq_results["best_auroc"],
            "layer_selection": mcq_results["selection_source"],
            "top_layers": mcq_results["top_layers"],
        },
        "framing": {
            "best_layer": framing_results["best_layer"],
            "best_auroc": framing_results["best_auroc"],
            "top_layers": framing_probe.metadata["top_layers"],
        },
        "cross_validation": cross,
    }
    if mcq_results["per_layer"]:
        heldout_ranked = sorted(
            mcq_results["per_layer"].items(),
            key=lambda kv: kv[1]["auroc"],
            reverse=True,
        )
        heldout_layer, heldout_info = heldout_ranked[0]
        summary["mcq"]["heldout"] = {
            "enabled": True,
            "best_layer": heldout_layer,
            "best_auroc": heldout_info["auroc"],
            "top_layers": [l for l, _ in heldout_ranked[: args.top_n_layers]],
        }
    else:
        summary["mcq"]["heldout"] = {"enabled": False}
    if sad_results is not None:
        summary["mcq"]["sad"] = {
            "output": str(output_root / "sad_stages_oversight_eval.json"),
            "selected_layer": sad_results["selected_layer"],
            "selected_auroc": sad_results["selected_auroc"],
            "selection_source": sad_results["selection_source"],
            "dev_test_split": sad_results["dev_test_split"],
            "top_layers_all": sad_results["top_layers_all"],
            "num_layers_exact_1_0": sad_results["num_layers_exact_1_0"],
            "num_layers_below_0_5": sad_results["num_layers_below_0_5"],
        }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
