"""Difference-of-means (DoM) eval-awareness probes for CUA models.

Ported from `jasmineee-li/eval-awareness` (eval-awareness-testbed/src/
eval_awareness_probes/) and adapted for the multimodal Qwen-VL models we
run via browser-sim: OpenCUA-32B/72B are Qwen2.5-VL, GUI-Owl-1.5-32B-Think
is Qwen3-VL.

For training and steering we use HuggingFace `transformers` with forward
hooks on the LM decoder's residual stream. For benchmark generation
(unsteered) we use vLLM via the OpenAI-compatible endpoint — see
`models/<model>/serve.sh`.

Modules:
    model_loader        — HookedTransformerShim for HF + Qwen2.5-VL / Qwen3-VL handling
    activation_extraction — Token-position selectors and contrastive extraction
    contrastive_probe   — DoM probe (train/save/load/score)
    scoring             — ProbeScore / AUROC helpers
    steering            — Forward-pre-hook context manager for residual-stream steering
    train_probe         — CLI to train a DoM probe from contrastive_dataset.json
    score_probe         — CLI to score transcripts with a saved probe
"""
