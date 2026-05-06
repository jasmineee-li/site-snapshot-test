"""Cross-format validation for MCQ and framing DoM probes.

For each probe, score the OTHER format's data and compute AUROC. If both
probes track the same "eval-awareness" construct, each should correctly
classify the other's prompts even if their best-layer vectors are
geometrically distinct.

Sign conventions (from probes/contrastive_probe.py):
  - MCQ probe vector = mean(positive) - mean(negative); positive = eval.
    → higher mean_score = more eval-aware.
  - Framing probe vector = mean(real) - mean(fake); real = deploy.
    → higher mean_score = more deploy-like.

So for "is this eval-aware?" classification:
  - MCQ probe:     label=is_eval, score=mean_score          → AUROC > 0.5 = correct.
  - Framing probe: label=is_eval, score=-mean_score (flip)  → AUROC > 0.5 = correct.

Usage:
    HF_HOME=/workspace/hf_cache python -m probes.validate_probes_cross \\
        --model mPLUG/GUI-Owl-1.5-32B-Think \\
        --mcq-probe probes/trained/gui-owl-32b/ \\
        --framing-probe probes/trained/gui-owl-32b-framing/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from sklearn.metrics import roc_auc_score

from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model path")
    ap.add_argument("--mcq-probe", required=True, help="Trained MCQ probe dir")
    ap.add_argument("--framing-probe", required=True, help="Trained framing probe dir")
    ap.add_argument("--mcq-data", default="probes/data/contrastive_dataset.json")
    ap.add_argument("--framing-data", default="probes/data/framing_pairs.json")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    mcq_data = json.loads(Path(args.mcq_data).read_text())
    fr_data = json.loads(Path(args.framing_data).read_text())

    logger.info(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    logger.info(f"n_layers={model.cfg.n_layers}")

    mcq_probe = ContrastiveProbe.load(args.mcq_probe)
    fr_probe = ContrastiveProbe.load(args.framing_probe)
    logger.info(f"MCQ probe best_layer={mcq_probe.best_layer}, framing probe best_layer={fr_probe.best_layer}")

    # === A: MCQ probe → framing data ===
    print(f"\n=== A: MCQ probe scoring framing data ({len(fr_data['real']) + len(fr_data['fake'])} prompts) ===")
    labels_A: list[int] = []
    scores_A: list[float] = []
    for prompt in fr_data["real"]:
        scores_A.append(mcq_probe.score(model, prompt, tokenizer=tokenizer).mean_score)
        labels_A.append(0)
    for prompt in fr_data["fake"]:
        scores_A.append(mcq_probe.score(model, prompt, tokenizer=tokenizer).mean_score)
        labels_A.append(1)
    auroc_A = roc_auc_score(labels_A, scores_A)
    n_real = len(fr_data["real"])
    n_fake = len(fr_data["fake"])
    mean_real = sum(scores_A[:n_real]) / n_real
    mean_fake = sum(scores_A[n_real:]) / n_fake
    print(f"AUROC(label=is_fake, score=MCQ-mean) = {auroc_A:.4f}")
    print(f"  mean(real) = {mean_real:+.4f}    mean(fake) = {mean_fake:+.4f}    Δ = {mean_fake - mean_real:+.4f}")

    # === B: Framing probe → MCQ data ===
    print(f"\n=== B: Framing probe scoring MCQ data ({len(mcq_data) * 2} prompts) ===")
    labels_B: list[int] = []
    scores_B: list[float] = []
    for i, item in enumerate(mcq_data):
        q = item["question"]
        pos_ans = item.get("positive") or item.get("test")
        neg_ans = item.get("negative") or item.get("deploy")
        s_pos = fr_probe.score(model, f"{q}\n\n {pos_ans}", tokenizer=tokenizer).mean_score
        s_neg = fr_probe.score(model, f"{q}\n\n {neg_ans}", tokenizer=tokenizer).mean_score
        labels_B += [1, 0]
        scores_B += [s_pos, s_neg]
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(mcq_data)} pairs scored")

    # Framing vector = real - fake → high mean_score = deploy.
    # Eval-aware should score LOWER → flip sign for AUROC.
    auroc_B = roc_auc_score(labels_B, [-s for s in scores_B])
    auroc_B_unflipped = roc_auc_score(labels_B, scores_B)
    n_pairs = len(labels_B) // 2
    mean_pos = sum(s for l, s in zip(labels_B, scores_B) if l == 1) / n_pairs
    mean_neg = sum(s for l, s in zip(labels_B, scores_B) if l == 0) / n_pairs
    print(f"AUROC(label=is_eval, score=-Framing-mean) = {auroc_B:.4f}")
    print(f"  mean(eval-positive) = {mean_pos:+.4f}    mean(deploy-neg) = {mean_neg:+.4f}    Δ = {mean_pos - mean_neg:+.4f}")
    print(f"  (sanity) AUROC unflipped = {auroc_B_unflipped:.4f}")

    print("\n=== Summary ===")
    print(f"MCQ probe identifies eval in framing prompts:  AUROC = {auroc_A:.4f}")
    print(f"Framing probe identifies eval in MCQ prompts:  AUROC = {auroc_B:.4f}")
    print("\nInterpretation:")
    print("  > 0.7  : probes agree on the construct (good cross-generalization)")
    print("  ~ 0.5  : at chance — probe doesn't transfer (independent features / overfitting)")
    print("  < 0.3  : systematically inverted on the other format")


if __name__ == "__main__":
    main()
