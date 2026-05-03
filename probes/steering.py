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
    alpha: float = 0.0,
    use_normalized: bool = True,
):
    """Install a residual-stream pre-hook that adds `alpha * vector`.

    Args:
        model: HookedTransformerShim from probes.model_loader.
        probe: ContrastiveProbe with `.normalized_vectors` populated.
        layer: Layer to intervene at (default: probe.best_layer).
        alpha: Steering strength. Negative = away from eval-aware.
            `alpha=0` is a no-op identity hook (for sanity testing
            that the hook itself doesn't perturb generation).
        use_normalized: If True, use unit-norm vector (recommended).
            If False, use raw mean-difference vector.
    """
    if layer is None:
        layer = probe.best_layer
    if layer is None:
        raise ValueError("No layer specified and no best_layer set on probe")

    vec_dict = probe.normalized_vectors if use_normalized else probe.vectors
    if layer not in vec_dict:
        raise ValueError(f"No probe vector for layer {layer}")

    vector = vec_dict[layer]
    # Vector may be (1, d_model) — squeeze to (d_model,)
    if vector.dim() > 1:
        vector = vector.squeeze(0)

    decoder_layers = model.get_decoder_layers()
    if not (0 <= layer < len(decoder_layers)):
        raise IndexError(
            f"Layer {layer} out of range for model with {len(decoder_layers)} layers"
        )
    target_module = decoder_layers[layer]

    handle = target_module.register_forward_pre_hook(
        _make_pre_hook(vector, alpha), with_kwargs=True
    )
    logger.info(
        f"Installed steering hook on layer {layer} (alpha={alpha}, "
        f"|v|={vector.norm().item():.3f})"
    )
    try:
        yield
    finally:
        handle.remove()


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
