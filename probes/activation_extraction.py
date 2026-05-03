"""Modular activation extraction for probe evaluation.

Ported from eval-awareness/eval-awareness-testbed/src/eval_awareness_probes/
activation_extraction.py. Original supports text-only causal LMs; for
multimodal (Qwen2.5-VL) models we feed text-only contrastive prompts
(the dataset is text), so the same code path works — `tokenizer.encode`
on Qwen2.5-VL's processor.tokenizer returns plain text token ids.
"""

from __future__ import annotations

import gc
import logging
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class TokenSelector(str, Enum):
    """Strategy for selecting which token positions to extract activations from."""

    ANSWER_TOKEN = "answer_token"
    LAST_TOKEN = "last_token"
    MEAN_ALL = "mean_all"
    EVIDENCE_TOKENS = "evidence_tokens"
    ALL_TOKENS = "all_tokens"


def extract_activations_hooked(
    model,
    tokens: torch.Tensor,
    layers: list[int],
    token_selector: TokenSelector = TokenSelector.MEAN_ALL,
    tokenizer=None,
    answer_token: str | None = None,
    evidence_indices: list[tuple[int, int]] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract activations at `blocks.{layer}.hook_resid_pre` via hooks."""
    cached: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook_fn(activation, hook):
            act = _select_token_activation(
                activation, tokens, token_selector, tokenizer,
                answer_token, evidence_indices,
            )
            cached[layer] = act.clone().detach()
            return activation
        return hook_fn

    fwd_hooks = [
        (f"blocks.{layer}.hook_resid_pre", make_hook(layer)) for layer in layers
    ]

    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        with torch.no_grad():
            model(tokens)

    return cached


def extract_contrastive_activations(
    model,
    dataset: list[dict],
    layers: list[int],
    tokenizer,
) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
    """For each contrastive pair, extract activations at the answer-token
    position for both the 'positive' (test/eval-aware) and 'negative'
    (deploy/not-aware) responses. Returns per-layer lists of tensors.
    """
    positive_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    negative_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for item in dataset:
        question = item["question"]
        pos_answer = item.get("test") or item.get("positive")
        neg_answer = item.get("deploy") or item.get("negative")

        pos_prompt = f"{question}\n\n {pos_answer}"
        neg_prompt = f"{question}\n\n {neg_answer}"

        # "(A)" -> "A"
        pos_token = pos_answer[1] if len(pos_answer) > 1 else pos_answer
        neg_token = neg_answer[1] if len(neg_answer) > 1 else neg_answer

        pos_tokens = model.to_tokens(pos_prompt)
        pos_cached = extract_activations_hooked(
            model, pos_tokens, layers, TokenSelector.ANSWER_TOKEN,
            tokenizer, answer_token=pos_token,
        )

        neg_tokens = model.to_tokens(neg_prompt)
        neg_cached = extract_activations_hooked(
            model, neg_tokens, layers, TokenSelector.ANSWER_TOKEN,
            tokenizer, answer_token=neg_token,
        )

        for layer in layers:
            if layer in pos_cached:
                positive_acts[layer].append(pos_cached[layer])
            if layer in neg_cached:
                negative_acts[layer].append(neg_cached[layer])

        del pos_tokens, neg_tokens, pos_cached, neg_cached
        gc.collect()

    return positive_acts, negative_acts


def _select_token_activation(
    activation: torch.Tensor,
    tokens: torch.Tensor,
    token_selector: TokenSelector,
    tokenizer=None,
    answer_token: str | None = None,
    evidence_indices: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    if token_selector == TokenSelector.ANSWER_TOKEN:
        if tokenizer is None or answer_token is None:
            raise ValueError(
                "tokenizer and answer_token required for ANSWER_TOKEN selection"
            )
        token_id = tokenizer.convert_tokens_to_ids(answer_token)
        positions = (tokens == token_id).nonzero()
        if len(positions) == 0:
            logger.warning(
                f"Answer token '{answer_token}' (id={token_id}) not found, "
                f"falling back to last token"
            )
            return activation[:, -1:, :].mean(dim=1, keepdim=False)
        last_pos = positions[-1][-1]
        return activation[:, last_pos, :]

    if token_selector == TokenSelector.LAST_TOKEN:
        return activation[:, -1, :]

    if token_selector == TokenSelector.MEAN_ALL:
        return activation.mean(dim=1)

    if token_selector == TokenSelector.EVIDENCE_TOKENS:
        if not evidence_indices:
            logger.warning("No evidence indices provided, falling back to mean_all")
            return activation.mean(dim=1)
        chunks = [activation[:, s:e, :].mean(dim=1) for s, e in evidence_indices]
        return torch.stack(chunks).mean(dim=0)

    if token_selector == TokenSelector.ALL_TOKENS:
        return activation

    raise ValueError(f"Unknown token selector: {token_selector}")
