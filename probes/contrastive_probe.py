"""Contrastive (difference-of-means) steering vector probe for eval awareness.

Ported from eval-awareness/eval-awareness-testbed/src/eval_awareness_probes/
contrastive_probe.py. The DoM compute is architecture-agnostic — same code
works for OLMo, Qwen, Qwen2.5-VL based models, etc.

Methodology:
    Nguyen et al. 2025 (arXiv:2507.01786) "Probing and Steering Evaluation Awareness"
    Chaudhary et al. 2025 (arXiv:2509.13333) "Evaluation Awareness Scales Predictably"

Training:
    steering_vector[layer] = mean(test_activations) - mean(deploy_activations)
Scoring:
    score = dot(activation, normalized_vector)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from probes.activation_extraction import (
    extract_contrastive_activations,
    extract_framing_activations,
)
from probes.scoring import ProbeScore, classify_score

logger = logging.getLogger(__name__)


class ContrastiveProbe:
    """Contrastive steering vector probe for eval awareness."""

    def __init__(
        self,
        vectors: dict[int, torch.Tensor] | None = None,
        normalized_vectors: dict[int, torch.Tensor] | None = None,
        best_layer: int | None = None,
        threshold: float = 0.0,
        metadata: dict | None = None,
    ):
        self.vectors = vectors or {}
        self.normalized_vectors = normalized_vectors or {}
        self.best_layer = best_layer
        self.threshold = threshold
        self.metadata = metadata or {}

    def train(
        self,
        model,
        contrastive_dataset: list[dict],
        layers: list[int] | None,
        tokenizer,
    ) -> None:
        """Train DoM steering vectors from a contrastive dataset.

        For each contrastive pair, extracts activations at the answer token
        position and computes mean(positive) - mean(negative) per layer.
        """
        if layers is None:
            layers = list(range(model.cfg.n_layers))

        logger.info(
            f"Training contrastive probe on {len(contrastive_dataset)} examples, "
            f"{len(layers)} layers"
        )

        positive_acts, negative_acts = extract_contrastive_activations(
            model, contrastive_dataset, layers, tokenizer
        )

        self.vectors = {}
        self.normalized_vectors = {}

        for layer in layers:
            pos_list = positive_acts[layer]
            neg_list = negative_acts[layer]

            if not pos_list or not neg_list:
                logger.warning(f"Layer {layer}: missing activations, skipping")
                continue

            pos_mean = torch.stack(pos_list).mean(dim=0)
            neg_mean = torch.stack(neg_list).mean(dim=0)
            vector = pos_mean - neg_mean
            self.vectors[layer] = vector

            norm = vector.norm()
            self.normalized_vectors[layer] = vector / norm if norm > 0 else vector

        logger.info(f"Trained vectors for {len(self.vectors)} layers")

    def train_framing(
        self,
        model,
        framing_data: dict,
        layers: list[int] | None,
        tokenizer,
        token_position: int = -2,
    ) -> None:
        """Train DoM steering vectors from the paper's framing-pair prompts.

        For each "real" / "fake" framing prompt (e.g.
        `"You are talking to a real user:"` vs
        `"You are talking to an AI evaluator:"`), extract the activation
        at the colon (`token_position=-2` after chat-template rendering)
        and compute `mean(real) - mean(fake)` per layer.

        Note: real and fake pools may have different lengths; we average
        each pool independently rather than treating them as paired.
        """
        if layers is None:
            layers = list(range(model.cfg.n_layers))

        n_real = len(framing_data.get("real", []))
        n_fake = len(framing_data.get("fake", []))
        logger.info(
            f"Training framing probe: {n_real} real + {n_fake} fake prompts, "
            f"{len(layers)} layers, token_position={token_position}"
        )

        real_acts, fake_acts = extract_framing_activations(
            model, framing_data, layers, tokenizer, token_position=token_position
        )

        self.vectors = {}
        self.normalized_vectors = {}

        for layer in layers:
            r_list = real_acts[layer]
            f_list = fake_acts[layer]
            if not r_list or not f_list:
                logger.warning(f"Layer {layer}: missing activations, skipping")
                continue
            real_mean = torch.stack(r_list).mean(dim=0)
            fake_mean = torch.stack(f_list).mean(dim=0)
            # Convention: positive = "real / deployment", so
            # vector = real - fake. `alpha > 0` pushes towards real;
            # `alpha < 0` pushes towards eval-aware.
            vector = real_mean - fake_mean
            self.vectors[layer] = vector
            norm = vector.norm()
            self.normalized_vectors[layer] = vector / norm if norm > 0 else vector

        logger.info(f"Trained framing vectors for {len(self.vectors)} layers")

    def score(
        self,
        model,
        text: str,
        layer: int | None = None,
        tokenizer=None,
    ) -> ProbeScore:
        """Score a text for eval awareness using the steering vector probe."""
        if layer is None:
            layer = self.best_layer
        if layer is None:
            raise ValueError("No layer specified and no best_layer set")
        if layer not in self.normalized_vectors:
            raise ValueError(f"No vector for layer {layer}")

        vector = self.normalized_vectors[layer]

        tokens = model.to_tokens(text)
        per_token_scores: list[float] = []

        def hook_fn(activation, hook):
            vec = vector[0] if vector.dim() > 1 else vector
            vec = vec.to(device=activation.device, dtype=activation.dtype)
            for pos in range(activation.shape[1]):
                act = activation[0, pos, :]
                projection = torch.dot(act, vec)
                proj_val = (
                    projection / vec.norm() if vector.dim() > 1 else projection.item()
                )
                if isinstance(proj_val, torch.Tensor):
                    proj_val = proj_val.item()
                if pos == len(per_token_scores):
                    per_token_scores.append(proj_val)
            return activation

        model.reset_hooks()
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]):
            with torch.no_grad():
                model(tokens)

        if tokenizer is not None:
            token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
            filtered = [
                s
                for s, tok in zip(per_token_scores, token_strings)
                if tok.strip()
                and "<|begin_of_text|>" not in tok
                and "<s>" not in tok
                and "<BOS>" not in tok
            ]
        else:
            filtered = per_token_scores
        if not filtered:
            filtered = per_token_scores or [0.0]

        mean_score = sum(filtered) / len(filtered)
        max_score = max(filtered)
        min_score = min(filtered)
        std_score = (
            sum((x - mean_score) ** 2 for x in filtered) / len(filtered)
        ) ** 0.5

        classification, confidence = classify_score(mean_score, self.threshold)

        return ProbeScore(
            mean_score=mean_score,
            max_score=max_score,
            min_score=min_score,
            std_score=std_score,
            classification=classification,
            confidence=confidence,
            per_token_scores=filtered,
            metadata={"layer": layer},
        )

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        vectors_dir = output_dir / "vectors"
        norm_dir = output_dir / "normalized_vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        norm_dir.mkdir(parents=True, exist_ok=True)

        for layer, vector in self.vectors.items():
            torch.save(vector, vectors_dir / f"layer_{layer}.pt")
        for layer, vector in self.normalized_vectors.items():
            torch.save(vector, norm_dir / f"layer_{layer}.pt")

        meta = {
            "best_layer": self.best_layer,
            "threshold": self.threshold,
            "num_layers": len(self.vectors),
            "layers": sorted(self.vectors.keys()),
            **self.metadata,
        }
        with open(output_dir / "probe_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved probe to {output_dir}")

    @classmethod
    def load(cls, probe_dir: str | Path) -> ContrastiveProbe:
        probe_dir = Path(probe_dir)
        norm_dir = probe_dir / "normalized_vectors"
        vectors_dir = probe_dir / "vectors"

        normalized_vectors = {}
        vectors = {}

        if norm_dir.exists():
            for f in sorted(norm_dir.glob("layer_*.pt")):
                layer = int(f.stem.split("_")[1])
                normalized_vectors[layer] = torch.load(
                    f, map_location="cpu", weights_only=True
                )
        if vectors_dir.exists():
            for f in sorted(vectors_dir.glob("layer_*.pt")):
                layer = int(f.stem.split("_")[1])
                vectors[layer] = torch.load(f, map_location="cpu", weights_only=True)

        if not normalized_vectors and vectors:
            for layer, v in vectors.items():
                norm = v.norm()
                normalized_vectors[layer] = v / norm if norm > 0 else v

        metadata = {}
        best_layer = None
        threshold = 0.0
        info_path = probe_dir / "probe_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            best_layer = info.get("best_layer")
            threshold = info.get("threshold", 0.0)
            metadata = {
                k: v for k, v in info.items() if k not in ("best_layer", "threshold")
            }

        logger.info(
            f"Loaded contrastive probe from {probe_dir} "
            f"({len(normalized_vectors)} layers, best_layer={best_layer})"
        )

        return cls(
            vectors=vectors,
            normalized_vectors=normalized_vectors,
            best_layer=best_layer,
            threshold=threshold,
            metadata=metadata,
        )

    @property
    def layers(self) -> list[int]:
        return sorted(set(self.vectors.keys()) | set(self.normalized_vectors.keys()))


def train_logreg_direction(
    positive_activations: np.ndarray,
    negative_activations: np.ndarray,
    Cs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """L2-regularized logistic regression on contrastive activations.

    Returns (normalized_direction, info_dict) where info_dict contains
    best_C and cv_auroc.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    if Cs is None:
        Cs = np.logspace(-4, 2, 20)

    X = np.concatenate([positive_activations, negative_activations], axis=0)
    y = np.concatenate(
        [np.ones(len(positive_activations)), np.zeros(len(negative_activations))]
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegressionCV(
        Cs=Cs, cv=5, penalty="l2", solver="lbfgs",
        max_iter=2000, scoring="roc_auc", random_state=42,
    )
    clf.fit(X_scaled, y)

    w = clf.coef_[0] / scaler.scale_
    w_normalized = w / np.linalg.norm(w)

    return w_normalized, {
        "best_C": float(clf.C_[0]),
        "cv_auroc": float(clf.scores_[1].mean(axis=0).max()),
    }
