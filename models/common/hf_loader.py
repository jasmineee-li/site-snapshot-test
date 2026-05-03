"""HF transformers loader for local CUA models — used for probe extraction
and steered benchmark runs.

For unsteered benchmark generation we use vLLM (much faster). vLLM doesn't
expose hidden states or accept residual-stream hooks trivially, so the
HF path is the one that uses `probes.model_loader.HookedTransformerShim`
and `probes.steering.steering_hook`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_hf_model(
    short_id_or_repo: str,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
):
    """Load `(HookedTransformerShim, tokenizer)` for a registered local model.

    Accepts either a registry short id (`opencua-32b`) or a raw HF repo
    path (`xlangai/OpenCUA-32B`).
    """
    from probes.model_loader import load_model
    from models.common.registry import LOCAL_MODELS

    if short_id_or_repo in LOCAL_MODELS:
        hf_repo = LOCAL_MODELS[short_id_or_repo].hf_repo
    else:
        hf_repo = short_id_or_repo

    logger.info(f"Loading HF model {hf_repo} (device={device}, dtype={dtype})")
    return load_model(hf_repo, device=device, dtype=dtype, revision=revision)
