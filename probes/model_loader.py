"""Model loading utilities for DoM-probe extraction on CUA models.

Wraps any HuggingFace causal-LM (or Qwen-VL conditional-generation
model) in a `HookedTransformerShim` providing a TransformerLens-style
hook API for residual-stream activation extraction.

Adapted from eval-awareness/eval-awareness-testbed/src/eval_awareness_probes/
model_loader.py — original supported text-only causal LMs (OLMo, Qwen,
Llama, etc.); this version additionally handles the Qwen-VL based models we
run — OpenCUA-32B/72B (`qwen2_5_vl`) and GUI-Owl-1.5-32B-Think (`qwen3_vl`)
— by:
  1. Trying `AutoModelForCausalLM` first (works for many VL checkpoints
     because the LM head is registered).
  2. Falling back to `_load_vl()` when CausalLM loading fails, which
     dispatches on the config's `model_type` to
     `Qwen2_5_VLForConditionalGeneration` or
     `Qwen3VLForConditionalGeneration`, then exposing the underlying text
     decoder via `model.model.layers` (skipping the vision tower) for
     hook attachment.

We feed text-only prompts at probe-extraction time (the contrastive
dataset is text), so the vision tower is never exercised — but the model
weights still load on GPU. For OpenCUA-72B you need ≥2 H100s in bf16;
OpenCUA-32B / GUI-Owl-1.5-32B fit on one H100 with `device_map="auto"`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class HookedTransformerShim:
    """Minimal API-compatible shim for TransformerLens HookedTransformer.

    Wraps any HuggingFace model to provide activation extraction via
    forward hooks on `output_hidden_states`, matching the HookedTransformer
    API used in the probe pipeline.
    """

    def __init__(self, hf_model, tokenizer, device: str, cfg):
        self.model = hf_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.cfg = cfg
        self._hooks: dict[str, Callable] = {}
        self._hook_handles: list = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        revision: str | None = None,
    ) -> HookedTransformerShim:
        """Load a pretrained model from HuggingFace hub.

        Args:
            model_path: HuggingFace model path (e.g. 'xlangai/OpenCUA-32B').
            device: 'cuda' or 'cpu'.
            dtype: e.g. torch.bfloat16.
            revision: git revision / checkpoint.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        cfg_hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        cfg_hf.output_hidden_states = True
        cfg_hf.return_dict = True

        load_kwargs: dict[str, Any] = {
            "config": cfg_hf,
            "trust_remote_code": True,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        if device != "cpu":
            load_kwargs["device_map"] = "auto"
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        model_type = getattr(cfg_hf, "model_type", "")
        try:
            model_hf = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except (ValueError, KeyError) as e:
            logger.info(
                f"AutoModelForCausalLM failed for {model_path} ({e}); "
                f"falling back to architecture-specific loader."
            )
            model_hf = _load_vl(model_path, load_kwargs, model_type)

        if device == "cpu":
            # Third-party typing defect, not a bug here: transformers declares
            # `PreTrainedModel.to` as `@wraps(torch.nn.Module.to) def to(...)`,
            # and typeshed models `functools.wraps` as returning a `_Wrapped`
            # object that implements no descriptor protocol. mypy therefore sees
            # the unbound wrapper and reads `"cpu"` as the `self` argument.
            model_hf.to("cpu")  # type: ignore[arg-type]

        n_layers = _get_config_value(
            cfg_hf,
            model_hf,
            ["num_hidden_layers", "n_layers", "num_layers", "n_layer"],
        )
        d_model = _get_config_value(cfg_hf, None, ["hidden_size", "d_model", "n_embd", "dim"])
        n_heads = _get_config_value(
            cfg_hf, None, ["num_attention_heads", "n_heads", "num_heads", "n_head"]
        )

        if n_layers is None:
            raise ValueError(f"Could not determine number of layers for model {model_path}.")

        cfg = SimpleNamespace(
            model_name=model_path,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_model // n_heads if (d_model and n_heads) else None,
            d_vocab=getattr(cfg_hf, "vocab_size", None),
            n_ctx=getattr(cfg_hf, "max_position_embeddings", None),
        )

        logger.info(
            f"Loaded {model_path} via HookedTransformerShim "
            f"(n_layers={n_layers}, d_model={d_model})"
        )

        return cls(model_hf, tokenizer, device, cfg)

    def eval(self):
        self.model.eval()
        return self

    def reset_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._hooks = {}

    @contextmanager
    def hooks(self, fwd_hooks: list[tuple[str, Callable]] | None = None, **kwargs):
        """Temporarily install forward hooks (TransformerLens-style names)."""
        if fwd_hooks is None:
            fwd_hooks = []

        old_hooks = self._hooks.copy()
        for name, fn in fwd_hooks:
            self._hooks[name] = fn

        try:
            yield
        finally:
            self._hooks = old_hooks

    def to_tokens(self, prompt: str | list[str], prepend_bos: bool = True) -> torch.Tensor:
        if isinstance(prompt, (list, tuple)):
            ids = [self.tokenizer.encode(p, add_special_tokens=prepend_bos) for p in prompt]
            max_len = max(len(seq) for seq in ids)
            padded = [seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in ids]
            return torch.tensor(padded).to(self.device)
        ids = self.tokenizer.encode(prompt, add_special_tokens=prepend_bos)
        return torch.tensor([ids]).to(self.device)

    def __call__(self, tokens: torch.Tensor, **kwargs) -> Any:
        inputs: dict[str, Any]
        if isinstance(tokens, dict):
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in tokens.items()
            }
        else:
            inputs = {"input_ids": tokens.to(self.device)}
        inputs.update(kwargs)

        if self._hooks:
            inputs["output_hidden_states"] = True

        outputs = self.model(**inputs)

        if self._hooks and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states
            for name, fn in self._hooks.items():
                activation = _extract_activation(name, hidden_states)
                if activation is not None:
                    hook_point = SimpleNamespace(name=name)
                    try:
                        fn(activation, hook_point)
                    except TypeError:
                        fn(activation)

        return outputs

    def run_with_cache(
        self,
        tokens: torch.Tensor | str,
        names_filter: list[str] | None = None,
        device: str | None = None,
        remove_batch_dim: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(tokens, str):
            tokens = self.to_tokens(tokens)
        if device is not None:
            tokens = tokens.to(device)

        cache: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            outputs = self.model(tokens, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is not None:
            for layer_idx in range(len(hidden_states) - 1):
                pre_name = f"blocks.{layer_idx}.hook_resid_pre"
                post_name = f"blocks.{layer_idx}.hook_resid_post"

                if names_filter is None or pre_name in names_filter:
                    cache[pre_name] = hidden_states[layer_idx]
                if names_filter is None or post_name in names_filter:
                    cache[post_name] = hidden_states[layer_idx + 1]

        if remove_batch_dim and tokens.shape[0] == 1:
            cache = {k: v.squeeze(0) if v.shape[0] == 1 else v for k, v in cache.items()}

        logits = outputs.logits
        if remove_batch_dim and logits.shape[0] == 1:
            logits = logits.squeeze(0)

        return logits, cache

    def get_decoder_layers(self):
        """Return the list of decoder layer modules for hook attachment.

        Used by `steering.steering_hook` to register a forward-pre-hook on
        the residual stream input to a specific layer. Handles both plain
        causal-LM (`model.model.layers`) and Qwen-VL conditional-
        generation (`model.model.layers` after stripping the vision tower
        wrapper).
        """
        m = self.model
        # Plain causal LM: model.model.layers
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        # Some VL configurations: model.language_model.layers
        if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
            return m.language_model.layers
        if (
            hasattr(m, "language_model")
            and hasattr(m.language_model, "model")
            and hasattr(m.language_model.model, "layers")
        ):
            return m.language_model.model.layers
        raise AttributeError(
            f"Could not locate decoder layers on {type(m).__name__}. "
            f"Inspect the model and extend HookedTransformerShim.get_decoder_layers."
        )


def _load_vl(model_path: str, load_kwargs: dict, model_type: str = "") -> Any:
    """Architecture-specific loader for Qwen-VL based models.

    Dispatches on `model_type` from the HF config:
      - `qwen3_vl`   → Qwen3VLForConditionalGeneration (GUI-Owl-1.5)
      - `qwen2_5_vl` → Qwen2_5_VLForConditionalGeneration (OpenCUA)
    Falls back to `AutoModelForVision2Seq` / `AutoModel` for unknown types.
    """
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
    except (ImportError, ValueError):
        pass

    from transformers import AutoModel

    return AutoModel.from_pretrained(model_path, **load_kwargs)


def _get_config_value(cfg_hf, model_hf, attr_names: list[str]):
    for name in attr_names:
        val = getattr(cfg_hf, name, None)
        if val is not None:
            return val

    if model_hf is not None and hasattr(model_hf, "model"):
        base = model_hf.model
        if hasattr(base, "layers"):
            return len(base.layers)
        if hasattr(base, "h"):
            return len(base.h)
    if model_hf is not None and hasattr(model_hf, "language_model"):
        lm = model_hf.language_model
        if hasattr(lm, "layers"):
            return len(lm.layers)
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return len(lm.model.layers)

    return None


def _extract_activation(
    hook_name: str, hidden_states: tuple[torch.Tensor, ...]
) -> torch.Tensor | None:
    match_pre = re.match(r"blocks\.(\d+)\.hook_resid_pre", hook_name)
    if match_pre:
        idx = int(match_pre.group(1))
        if idx < len(hidden_states):
            return hidden_states[idx]

    match_post = re.match(r"blocks\.(\d+)\.hook_resid_post", hook_name)
    if match_post:
        idx = int(match_post.group(1))
        if idx + 1 < len(hidden_states):
            return hidden_states[idx + 1]

    return None


def load_model(
    model_path: str,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
) -> tuple:
    """Load a HuggingFace model + tokenizer for probe extraction or steering.

    Returns (HookedTransformerShim, tokenizer). The shim exposes both the
    `hooks(...)` context manager (for cached-activation extraction) and
    `get_decoder_layers()` (for steering pre-hooks).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformerShim.from_pretrained(
        model_path, device=device, dtype=dtype, revision=revision
    )
    model.eval()
    return model, model.tokenizer
