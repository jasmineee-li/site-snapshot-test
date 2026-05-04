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
from contextlib import contextmanager

import torch

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
    if multi_layer:
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
    decoder_layers = model.get_decoder_layers()
    n_layers_total = len(decoder_layers)

    handles: list = []
    installed: list[tuple[int, float, float]] = []

    try:
        for tgt_layer, tgt_alpha in zip(target_layers, target_alphas):
            if tgt_layer not in vec_dict:
                raise ValueError(f"No probe vector for layer {tgt_layer}")
            if not (0 <= tgt_layer < n_layers_total):
                raise IndexError(
                    f"Layer {tgt_layer} out of range for model with "
                    f"{n_layers_total} layers"
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
                f"effective_alpha={alpha_per_layer * len(installed):.3f})"
            )
        else:
            l, a, n = installed[0]
            logger.info(
                f"Installed steering hook on layer {l} (alpha={a}, |v|={n:.3f})"
            )
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
        if not args:
            return None  # nothing to modify
        hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if alpha == 0.0:
            return None  # no-op
        v = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        # Broadcast across batch and seq dims.
        new_hidden = hidden_states + alpha * v
        return (new_hidden, *args[1:]), kwargs

    return hook
