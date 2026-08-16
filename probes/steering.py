"""Residual-stream steering via forward pre-hooks.

NEW (not in eval-awareness): the upstream repo only does post-hoc
projection / classification. We need to actually intervene on the
residual stream during generation to estimate the *causal* effect of
eval awareness on benchmark behavior.

Usage:
    from probes.contrastive_probe import ContrastiveProbe
    from probes.model_loader import load_model
    from probes.steering import steering_hook

    model, tok = load_model("xlangai/OpenCUA-32B")
    probe = ContrastiveProbe.load("probes/trained/opencua-32b/")

    with steering_hook(model, probe, layer=probe.best_layer, alpha=-3.0):
        out = model.model.generate(input_ids, max_new_tokens=128)

The hook is registered on the input to `model.get_decoder_layers()[layer]`
(equivalent to TransformerLens `blocks.{layer}.hook_resid_pre`) and adds
`alpha * normalized_vector` to the residual stream at every forward pass
inside the context. `alpha < 0` steers AWAY from eval-aware; `alpha > 0`
steers TOWARDS eval-aware.

The vector is broadcast across the sequence dimension (every token gets
the same intervention). If you need token-conditional steering, modify
`_make_pre_hook` to gate on positions (e.g. only post-prompt).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)


@contextmanager
def steering_hook(
    model,
    probe,
    layer: int | None = None,
    layers: list[int] | None = None,
    alpha: float = 0.0,
    alpha_per_layer: float | None = None,
    use_normalized: bool = True,
):
    """Install residual-stream pre-hook(s) that add `alpha * vector`.

    Supports both single-layer and multi-layer steering (Hua et al. 2025
    apply α=0.6 at 6 layers simultaneously, effective α≈3.6).

    Args:
        model: HookedTransformerShim from probes.model_loader.
        probe: ContrastiveProbe with `.normalized_vectors` populated. The
            probe's `metadata['top_layers']` is used as the default for
            multi-layer steering when `layers` is omitted but the user
            asked for it via `alpha_per_layer`.
        layer: Single layer to intervene at. Mutually exclusive with
            `layers`. Defaults to `probe.best_layer` when both are None.
        layers: Multiple layers to intervene at simultaneously.
        alpha: Steering strength for single-layer mode. Negative ≈ steer
            towards "fake" (eval-aware) when probe is framing
            (vector = real − fake); flip the sign for MCQ probes per
            their convention. `alpha=0` is a no-op.
        alpha_per_layer: Strength applied to each layer in `layers`.
            Required when `layers` is set.
        use_normalized: Use unit-norm vectors (recommended).
    """
    if layer is not None and layers is not None:
        raise ValueError("Pass either `layer` or `layers`, not both.")

    if layers is None and alpha_per_layer is not None and layer is None:
        # User asked for multi-layer mode but didn't specify layers —
        # fall back to probe metadata's `top_layers`.
        meta_top = (probe.metadata or {}).get("top_layers")
        if meta_top:
            layers = list(meta_top)

    multi_layer = layers is not None
    # Tested directly rather than through `multi_layer` so that `layers` is
    # narrowed to non-None for `list(layers)` below. The two conditions are
    # identical -- `layers` is not rebound between here and the assignment above.
    if layers is not None:
        if alpha_per_layer is None:
            raise ValueError("`alpha_per_layer` is required when `layers` is set.")
        target_layers = list(layers)
        target_alphas = [alpha_per_layer] * len(target_layers)
    else:
        if layer is None:
            layer = probe.best_layer
        if layer is None:
            raise ValueError("No layer specified and no best_layer set on probe")
        target_layers = [layer]
        target_alphas = [alpha]

    vec_dict = probe.normalized_vectors if use_normalized else probe.vectors
    decoder_layers = _decoder_layers(model)
    n_layers_total = len(decoder_layers)

    handles: list = []
    installed: list[tuple[int, float, float]] = []

    try:
        for tgt_layer, tgt_alpha in zip(target_layers, target_alphas, strict=True):
            if tgt_layer not in vec_dict:
                raise ValueError(f"No probe vector for layer {tgt_layer}")
            if not (0 <= tgt_layer < n_layers_total):
                raise IndexError(
                    f"Layer {tgt_layer} out of range for model with " f"{n_layers_total} layers"
                )
            vec = vec_dict[tgt_layer]
            if vec.dim() > 1:
                vec = vec.squeeze(0)
            handle = decoder_layers[tgt_layer].register_forward_pre_hook(
                _make_pre_hook(vec, tgt_alpha), with_kwargs=True
            )
            handles.append(handle)
            installed.append((tgt_layer, tgt_alpha, vec.norm().item()))

        if multi_layer:
            logger.info(
                f"Installed multi-layer steering on {len(installed)} layers "
                f"(layers={[l for l, _, _ in installed]}, "
                f"alpha_per_layer={alpha_per_layer}, "
                # `alpha_per_layer` cannot be None here: `multi_layer` is True
                # only when `layers` was set, and that branch above raises
                # ValueError when `alpha_per_layer` is None. mypy cannot carry
                # that narrowing across the two separate `if` blocks, and
                # restructuring the control flow of a steering-install path to
                # suit the checker is not worth the behaviour risk.
                f"effective_alpha={alpha_per_layer * len(installed):.3f})"  # type: ignore[operator]
            )
        else:
            l, a, n = installed[0]
            logger.info(f"Installed steering hook on layer {l} (alpha={a}, |v|={n:.3f})")
        yield
    finally:
        for h in handles:
            h.remove()


def _make_pre_hook(vector: torch.Tensor, alpha: float):
    """Build a pre-hook that adds `alpha * vector` to the residual stream.

    Decoder layers are typically called as `layer(hidden_states, ...)`
    where `hidden_states` is the first positional arg with shape
    `(batch, seq_len, d_model)`. We modify it in-place via tensor
    addition (returning a new tuple of args).
    """

    def hook(module, args, kwargs):
        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if alpha == 0.0:
            return None  # no-op
        v = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        # Broadcast across batch and seq dims.
        new_hidden = hidden_states + alpha * v
        if args:
            return (new_hidden, *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = new_hidden
        return args, kwargs

    return hook


@dataclass
class TokenGatedSteeringState:
    """Mutable state shared between token-gated steering hooks.

    `prompt_positions` are token indices in the rendered prompt that should be
    steered during the prefill pass. During generation, hooks only steer the
    one-token cached forward passes after `generation_enabled` is set by the
    outer decoding loop.
    """

    prompt_len: int
    prompt_positions: set[int]
    generation_enabled: bool = False
    first_sentence_end_token_index: int | None = None


@contextmanager
def token_gated_steering_hook(
    model,
    probe,
    *,
    state: TokenGatedSteeringState,
    layers: Sequence[int],
    alpha_per_layer: float = 0.6,
    use_normalized: bool = True,
):
    """Install multi-layer steering with prompt/generation token gates.

    This matches the intervention pattern used in the Hazra/Goodfire-style
    setup we are testing:
      - prefill: steer only user-message content tokens;
      - generation: steer cached generation-token forwards only after the
        caller flips `state.generation_enabled`;
      - never steer system prompt or chat-template scaffolding tokens.
    """
    if not layers:
        raise ValueError("At least one layer is required for token-gated steering.")

    vec_dict = probe.normalized_vectors if use_normalized else probe.vectors
    decoder_layers = _decoder_layers(model)
    n_layers_total = len(decoder_layers)

    handles: list[Any] = []
    installed: list[tuple[int, float, float]] = []
    try:
        for tgt_layer in layers:
            if tgt_layer not in vec_dict:
                raise ValueError(f"No probe vector for layer {tgt_layer}")
            if not (0 <= tgt_layer < n_layers_total):
                raise IndexError(
                    f"Layer {tgt_layer} out of range for model with " f"{n_layers_total} layers"
                )
            vec = vec_dict[tgt_layer]
            if vec.dim() > 1:
                vec = vec.squeeze(0)
            handle = decoder_layers[tgt_layer].register_forward_pre_hook(
                _make_token_gated_pre_hook(vec, alpha_per_layer, state),
                with_kwargs=True,
            )
            handles.append(handle)
            installed.append((tgt_layer, alpha_per_layer, vec.norm().item()))
        logger.info(
            "Installed token-gated steering on layers=%s alpha_per_layer=%s " "prompt_positions=%s",
            [layer for layer, _, _ in installed],
            alpha_per_layer,
            len(state.prompt_positions),
        )
        yield state
    finally:
        for handle in handles:
            handle.remove()


def _make_token_gated_pre_hook(
    vector: torch.Tensor,
    alpha: float,
    state: TokenGatedSteeringState,
):
    def hook(module, args, kwargs):
        if alpha == 0.0:
            return None
        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None

        seq_len = int(hidden_states.shape[1])
        if seq_len == state.prompt_len:
            positions = [pos for pos in sorted(state.prompt_positions) if 0 <= pos < seq_len]
            if not positions:
                return None
            v = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            new_hidden = hidden_states.clone()
            new_hidden[:, positions, :] = new_hidden[:, positions, :] + alpha * v
        elif seq_len == 1 and state.generation_enabled:
            v = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            new_hidden = hidden_states + alpha * v
        else:
            return None

        if args:
            return (new_hidden, *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = new_hidden
        return args, kwargs

    return hook


def chat_role_content_token_positions(
    tokenizer,
    rendered_prompt: str,
    messages: list[dict[str, Any]],
    *,
    roles: set[str] | None = None,
) -> list[int]:
    """Return content-token positions for selected chat roles.

    The positions are computed against the fully rendered chat template, but
    only content substrings from the requested roles are selected. Role markers,
    special tokens, whitespace-only tokens, system content, and other template
    scaffolding are omitted.
    """
    roles = roles or {"user"}
    ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    positions: set[int] = set()
    search_from = 0
    for message in messages:
        role = str(message.get("role", ""))
        content = _flatten_content(message.get("content", ""))
        if role not in roles or not content:
            continue
        start = rendered_prompt.find(content, search_from)
        if start == -1:
            start = rendered_prompt.find(content)
        if start == -1:
            continue
        end = start + len(content)
        prefix_ids = tokenizer.encode(rendered_prompt[:start], add_special_tokens=False)
        through_ids = tokenizer.encode(rendered_prompt[:end], add_special_tokens=False)
        for pos in range(len(prefix_ids), len(through_ids)):
            if pos < 0 or pos >= len(ids):
                continue
            token_id = ids[pos]
            token_text = tokenizer.decode([token_id])
            if token_id in special_ids:
                continue
            if not token_text.strip():
                continue
            if token_text.startswith("<|") or token_text in {"<s>", "</s>"}:
                continue
            positions.add(pos)
        search_from = end
    return sorted(positions)


def generate_with_token_gated_steering_generate(
    model,
    processor,
    messages: list[dict[str, Any]],
    probe,
    *,
    layers: Sequence[int],
    alpha_per_layer: float = 0.6,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    eos_token_id: int | Sequence[int] | None = None,
    pad_token_id: int | None = None,
    stop_token_ids: Sequence[int] | None = None,
    rendered_prompt: str | None = None,
) -> dict[str, Any]:
    """Generate via HF `model.generate` with token-gated steering.

    This is the preferred helper for OpenCUA. It keeps the model's own
    generation stack and uses a stopping-criteria callback to enable
    generation-token steering after the first generated sentence has appeared.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    hf_model = _hf_model(model)
    rendered = rendered_prompt
    if rendered is None:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if callable(processor):
        inputs = processor(text=rendered, return_tensors="pt")
    else:
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(inputs["input_ids"].shape[-1])
    input_device = _model_input_device(hf_model)
    inputs = {
        key: value.to(input_device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    prompt_positions = set(
        chat_role_content_token_positions(
            tokenizer,
            rendered,
            messages,
            roles={"user"},
        )
    )
    state = TokenGatedSteeringState(
        prompt_len=prompt_len,
        prompt_positions=prompt_positions,
        generation_enabled=False,
    )

    stop_ids = _collect_stop_token_ids(
        tokenizer,
        eos_token_id=eos_token_id,
        stop_token_ids=stop_token_ids,
    )
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if eos_token_id is None and stop_ids:
        eos_token_id = sorted(stop_ids)

    do_sample = bool(temperature and temperature > 0)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "stopping_criteria": StoppingCriteriaList(
            [
                _EnableAfterFirstSentenceCriteria(tokenizer, state),
            ]
        ),
    }
    if do_sample:
        generation_kwargs["temperature"] = max(float(temperature), 1e-5)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    with token_gated_steering_hook(
        model,
        probe,
        state=state,
        layers=layers,
        alpha_per_layer=alpha_per_layer,
    ):
        with torch.inference_mode():
            output_ids = hf_model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "text": text,
        "generated_ids": generated_ids,
        "prompt_len": state.prompt_len,
        "n_user_prompt_positions": len(prompt_positions),
        "first_sentence_end_token_index": state.first_sentence_end_token_index,
        "generation_steering_enabled": state.generation_enabled,
        "layers": list(layers),
        "alpha_per_layer": alpha_per_layer,
        "stop_token_ids": sorted(stop_ids),
    }


def generate_with_token_gated_steering(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    probe,
    *,
    layers: Sequence[int],
    alpha_per_layer: float = 0.6,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    stop_token_ids: Sequence[int] | None = None,
    rendered_prompt: str | None = None,
    input_ids_list: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Greedy/sampled generation with user-token + post-first-sentence steering.

    This helper is intentionally simple and primarily meant for smoke tests and
    local HF-backed runs. It uses a manual cached decoding loop so we can turn
    generation-token steering on only after the first sentence has appeared.
    """
    rendered = rendered_prompt
    if rendered is None:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if input_ids_list is None:
        input_ids_list = tokenizer.encode(rendered, add_special_tokens=False)
    input_ids_list = list(input_ids_list)
    if not input_ids_list:
        raise ValueError("Rendered prompt tokenized to zero tokens.")

    input_device = _model_input_device(model)
    input_ids = torch.tensor([input_ids_list], device=input_device)
    prompt_positions = set(
        chat_role_content_token_positions(
            tokenizer,
            rendered,
            messages,
            roles={"user"},
        )
    )
    state = TokenGatedSteeringState(
        prompt_len=int(input_ids.shape[1]),
        prompt_positions=prompt_positions,
        generation_enabled=False,
    )
    stop_ids: set[int] = set(int(x) for x in (stop_token_ids or []) if x is not None)
    if eos_token_id is not None:
        stop_ids.add(int(eos_token_id))
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if tok_eos is not None:
        stop_ids.add(int(tok_eos))
    for marker in ("<|im_end|>", "<|endoftext|>", "[EOS]"):
        try:
            marker_id = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            marker_id = None
        if isinstance(marker_id, int) and marker_id >= 0:
            stop_ids.add(marker_id)

    generated_ids: list[int] = []
    first_sentence_end_token_index: int | None = None

    with token_gated_steering_hook(
        model,
        probe,
        state=state,
        layers=layers,
        alpha_per_layer=alpha_per_layer,
    ):
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            for _ in range(max_new_tokens):
                next_id = _sample_next_token(logits, temperature=temperature)
                token_id = int(next_id.item())
                generated_ids.append(token_id)

                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if first_sentence_end_token_index is None and _has_first_sentence(generated_text):
                    first_sentence_end_token_index = len(generated_ids) - 1
                    state.generation_enabled = True

                if token_id in stop_ids:
                    break

                outputs = model.model(
                    input_ids=next_id.reshape(1, 1),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "text": text,
        "generated_ids": generated_ids,
        "prompt_len": state.prompt_len,
        "n_user_prompt_positions": len(prompt_positions),
        "first_sentence_end_token_index": first_sentence_end_token_index,
        "generation_steering_enabled": state.generation_enabled,
        "layers": list(layers),
        "alpha_per_layer": alpha_per_layer,
        "stop_token_ids": sorted(stop_ids),
    }


def _sample_next_token(logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if temperature and temperature > 0:
        probs = torch.softmax(logits / max(float(temperature), 1e-5), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.argmax(logits, dim=-1)


def _has_first_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(
        # Keep this conservative: terminal punctuation followed by whitespace,
        # a quote/bracket, or end-of-string.
        __import__("re").search(r"[.!?](?:[\"')\]]+)?(?:\s|$)", stripped)
    )


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


class _EnableAfterFirstSentenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer, state: TokenGatedSteeringState) -> None:
        self.tokenizer = tokenizer
        self.state = state

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if self.state.generation_enabled:
            return False
        generated = input_ids[0, self.state.prompt_len :].detach().cpu().tolist()
        if not generated:
            return False
        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if _has_first_sentence(text):
            self.state.generation_enabled = True
            self.state.first_sentence_end_token_index = len(generated) - 1
        return False


def _collect_stop_token_ids(
    tokenizer,
    *,
    eos_token_id: int | Sequence[int] | None = None,
    stop_token_ids: Sequence[int] | None = None,
) -> set[int]:
    stop_ids: set[int] = set(int(x) for x in (stop_token_ids or []) if x is not None)
    if eos_token_id is not None:
        if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, (str, bytes)):
            stop_ids.update(int(x) for x in eos_token_id if x is not None)
        else:
            stop_ids.add(int(eos_token_id))
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if tok_eos is not None:
        stop_ids.add(int(tok_eos))
    for marker in ("<|im_end|>", "<|endoftext|>", "[EOS]"):
        try:
            marker_id = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            marker_id = None
        if isinstance(marker_id, int) and marker_id >= 0:
            stop_ids.add(marker_id)
    return stop_ids


def _hf_model(model):
    if hasattr(model, "get_decoder_layers") and hasattr(model, "model"):
        return model.model
    return model


def _decoder_layers(model):
    if hasattr(model, "get_decoder_layers"):
        return model.get_decoder_layers()
    m = _hf_model(model)
    if hasattr(m, "layers"):
        return m.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
        return m.language_model.layers
    if (
        hasattr(m, "language_model")
        and hasattr(m.language_model, "model")
        and hasattr(m.language_model.model, "layers")
    ):
        return m.language_model.model.layers
    raise AttributeError(f"Could not locate decoder layers on {type(m).__name__}.")


def _model_input_device(hf_model) -> torch.device:
    hf_model = _hf_model(hf_model)
    try:
        return hf_model.get_input_embeddings().weight.device
    except Exception:
        return next(hf_model.parameters()).device
