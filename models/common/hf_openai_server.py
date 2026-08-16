"""Tiny OpenAI-compatible chat server backed by HF Transformers.

This is a fallback for local CUA checkpoints that load correctly through
Transformers but are not accepted by vLLM's model registry. It implements the
small subset of `/v1/chat/completions` that AgentLab's `VLLMChatModel` uses.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor

from probes.contrastive_probe import ContrastiveProbe
from probes.steering import generate_with_token_gated_steering_generate


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = 0.0
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    n: int = 1
    stop: str | list[str] | None = None
    stream: bool = False
    logprobs: bool | None = None
    extra_body: dict[str, Any] | None = None
    steering_alpha: float | None = None
    steering_alpha_per_layer: float | None = None
    steering_layers: str | list[int] | list[str] | None = None
    disable_steering: bool | str | None = None


class ServerState:
    def __init__(self) -> None:
        # Populated by the startup path below; annotated so the `None` seed does
        # not fix these attributes' types as `None`.
        self.model: Any = None
        self.processor: Any = None
        self.served_model_name = ""
        self.max_new_tokens_cap = 2048
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.steering_probe: ContrastiveProbe | None = None
        self.steering_layers: list[int] = []
        self.steering_alpha_per_layer = 0.0


state = ServerState()
app = FastAPI(title="HF OpenAI-compatible server")


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _normalize_content_part(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}
    if part.get("type") == "text" or "text" in part:
        return {"type": "text", "text": str(part.get("text", ""))}
    # The WASP/AgentLab text-only path should not send images. Keep a visible
    # placeholder if one appears instead of failing the whole browser task.
    if part.get("type") == "image_url" or "image_url" in part or "image" in part:
        return {"type": "text", "text": "[image omitted]"}
    return {"type": "text", "text": str(part)}


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            content = [_normalize_content_part(part) for part in content]
        elif content is None:
            content = ""
        else:
            content = str(content)
        normalized.append({"role": role, "content": content})
    return normalized


def _strip_stop(text: str, stop: str | list[str] | None) -> str:
    stops = [stop] if isinstance(stop, str) else (stop or [])
    cut = len(text)
    for marker in stops:
        if not marker:
            continue
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def _strip_turn_markers(text: str) -> str:
    for marker in ("<|im_end|>", "[EOS]"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _parse_layers(value: Any, default: list[int] | None = None) -> list[int]:
    if value is None or value == "":
        return list(default or [])
    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw_parts = list(value)
    else:
        raw_parts = [value]
    return [int(part) for part in raw_parts]


def _parse_max_memory(value: str | None) -> dict[int | str, str] | None:
    if not value:
        return None
    max_memory: dict[int | str, str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, memory = part.partition(":")
        if not sep:
            raise ValueError(f"Invalid --max-memory entry {part!r}; expected '<device>:<memory>'")
        key = key.strip()
        memory = memory.strip()
        max_memory[int(key) if key.isdigit() else key] = memory
    return max_memory


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _request_extra(request: ChatCompletionRequest) -> dict[str, Any]:
    extra = dict(request.extra_body or {})
    for key in (
        "steering_alpha",
        "steering_alpha_per_layer",
        "steering_layers",
        "disable_steering",
    ):
        value = getattr(request, key, None)
        if value is not None:
            extra[key] = value
    return extra


def _request_steering_config(
    request: ChatCompletionRequest,
) -> tuple[list[int], float] | None:
    if state.steering_probe is None:
        return None

    extra = _request_extra(request)
    if _as_bool(extra.get("disable_steering")):
        return None

    layers = _parse_layers(extra.get("steering_layers"), state.steering_layers)
    if not layers:
        raise HTTPException(
            status_code=400,
            detail="steering is enabled, but no steering layers are configured",
        )

    alpha = extra.get("steering_alpha_per_layer", extra.get("steering_alpha"))
    if alpha is None:
        alpha = state.steering_alpha_per_layer
    return layers, float(alpha)


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": state.served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=True is not supported")
    if state.model is None or state.processor is None:
        raise HTTPException(status_code=503, detail="model is still loading")

    n = max(1, int(request.n or 1))
    max_new = request.max_completion_tokens or request.max_tokens or 1024
    max_new = max(1, min(int(max_new), state.max_new_tokens_cap))
    temperature = 0.0 if request.temperature is None else float(request.temperature)
    do_sample = temperature > 0.0

    messages = _normalize_messages(request.messages)
    steering_config = _request_steering_config(request)

    choices = []
    total_completion_tokens = 0
    prompt_tokens = 0

    if steering_config is None:
        prompt = state.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = state.processor(text=prompt, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[-1])
        inputs = _to_device(inputs, _input_device(state.model))
        prompt_tokens = input_len

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": state.processor.tokenizer.pad_token_id,
            "eos_token_id": state.processor.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)

        with torch.inference_mode():
            for idx in range(n):
                output_ids = state.model.generate(**inputs, **gen_kwargs)
                new_ids = output_ids[0, input_len:]
                total_completion_tokens += int(new_ids.numel())
                text = state.processor.tokenizer.decode(
                    new_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                text = _strip_stop(_strip_turn_markers(text), request.stop)
                choices.append(
                    {
                        "index": idx,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                )
    else:
        layers, alpha_per_layer = steering_config
        for idx in range(n):
            result = generate_with_token_gated_steering_generate(
                state.model,
                state.processor,
                messages,
                state.steering_probe,
                layers=layers,
                alpha_per_layer=alpha_per_layer,
                max_new_tokens=max_new,
                temperature=temperature,
                pad_token_id=state.processor.tokenizer.pad_token_id,
                eos_token_id=state.processor.tokenizer.eos_token_id,
            )
            if idx == 0:
                prompt_tokens = int(result["prompt_len"])
            total_completion_tokens += len(result["generated_ids"])
            text = _strip_stop(_strip_turn_markers(result["text"]), request.stop)
            choices.append(
                {
                    "index": idx,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": state.served_model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF repo id or local snapshot path")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-new-tokens-cap",
        type=int,
        default=int(os.environ.get("HF_OPENAI_MAX_NEW_TOKENS_CAP", "2048")),
    )
    parser.add_argument("--max-memory", default=os.environ.get("HF_OPENAI_MAX_MEMORY"))
    parser.add_argument(
        "--steering-probe-dir", default=os.environ.get("HF_OPENAI_STEERING_PROBE_DIR")
    )
    parser.add_argument("--steering-layers", default=os.environ.get("HF_OPENAI_STEERING_LAYERS"))
    parser.add_argument(
        "--steering-alpha-per-layer",
        "--steering-alpha",
        dest="steering_alpha_per_layer",
        type=float,
        default=float(os.environ.get("HF_OPENAI_STEERING_ALPHA_PER_LAYER", "0.0")),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import uvicorn

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    state.served_model_name = args.served_model_name or args.model
    state.max_new_tokens_cap = int(args.max_new_tokens_cap)
    state.processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }
    max_memory = _parse_max_memory(args.max_memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
        print(f"Using max_memory={max_memory}", flush=True)
    state.model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    state.model.eval()

    if args.steering_probe_dir:
        state.steering_probe = ContrastiveProbe.load(args.steering_probe_dir)
        state.steering_layers = _parse_layers(args.steering_layers)
        if not state.steering_layers:
            if state.steering_probe.best_layer is None:
                raise ValueError(
                    "--steering-probe-dir was provided, but no --steering-layers "
                    "were configured and the probe has no best_layer"
                )
            state.steering_layers = [int(state.steering_probe.best_layer)]
        state.steering_alpha_per_layer = float(args.steering_alpha_per_layer)
        print(
            "Steering probe loaded: "
            f"probe_dir={args.steering_probe_dir} "
            f"layers={state.steering_layers} "
            f"alpha_per_layer={state.steering_alpha_per_layer}",
            flush=True,
        )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
