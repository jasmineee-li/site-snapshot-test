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
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoProcessor


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


class ServerState:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.served_model_name = ""
        self.max_new_tokens_cap = 2048
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    prompt = state.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = state.processor(text=prompt, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])
    inputs = _to_device(inputs, _input_device(state.model))

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "do_sample": do_sample,
        "pad_token_id": state.processor.tokenizer.pad_token_id,
        "eos_token_id": state.processor.tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-5)

    choices = []
    total_completion_tokens = 0
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
            choices.append({
                "index": idx,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            })

    prompt_tokens = input_len
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
    parser.add_argument("--max-new-tokens-cap", type=int, default=int(os.environ.get("HF_OPENAI_MAX_NEW_TOKENS_CAP", "2048")))
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
    state.model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    state.model.eval()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
