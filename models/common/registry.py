"""Local-model registry.

The benchmark runners use the convention `local/<short-id>` for local
(non-OpenRouter) models. This module maps each short-id to a
`LocalModelSpec` containing:
    hf_repo:        HuggingFace repo id (used by HF loader for probes)
    served_name:    Name vLLM serves the model under (defaults to hf_repo)
    default_url:    OpenAI-compatible base URL where vLLM is listening
                    (overridable via LOCAL_OPENAI_BASE_URL env)
    vision:         Whether the model accepts image inputs
    served_max_total_tokens / max_input / max_new: token budgets

Usage in the runners:

    from models.common import is_local_model_id, resolve_local_model
    if is_local_model_id(model_name):
        spec = resolve_local_model(model_name)
        client = build_local_chat_client(spec)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

LOCAL_PREFIX = "local/"


@dataclass(frozen=True)
class LocalModelSpec:
    """One local model served via vLLM's OpenAI-compatible endpoint."""

    short_id: str            # e.g. "opencua-32b"
    hf_repo: str             # e.g. "xlangai/OpenCUA-32B"
    default_url: str         # e.g. "http://localhost:8001/v1"
    vision: bool = True
    served_name: str | None = None  # None => same as hf_repo
    max_total_tokens: int = 131072
    max_input_tokens: int = 100000
    max_new_tokens: int = 8192
    notes: str = ""

    def resolve_url(self) -> str:
        """Allow runtime override via env var (matches one server-per-port setup)."""
        env_var = f"LOCAL_OPENAI_BASE_URL_{self.short_id.upper().replace('-', '_')}"
        return os.environ.get(env_var) or os.environ.get("LOCAL_OPENAI_BASE_URL", self.default_url)

    def resolve_served_name(self) -> str:
        return self.served_name or self.hf_repo


# Default ports: opencua-32b -> 8001, opencua-72b -> 8002, gui-owl-32b -> 8003.
# Override per server with LOCAL_OPENAI_BASE_URL_<SHORT_ID> env var.
LOCAL_MODELS: dict[str, LocalModelSpec] = {
    "opencua-32b": LocalModelSpec(
        short_id="opencua-32b",
        hf_repo="xlangai/OpenCUA-32B",
        default_url="http://localhost:8001/v1",
        max_total_tokens=131072,
    ),
    "opencua-72b": LocalModelSpec(
        short_id="opencua-72b",
        hf_repo="xlangai/OpenCUA-72B",
        default_url="http://localhost:8002/v1",
        max_total_tokens=131072,
        notes="vLLM-only — too large for HF steering on a single host without TP>=2.",
    ),
    "gui-owl-32b-think": LocalModelSpec(
        short_id="gui-owl-32b-think",
        hf_repo="xlangai/GUI-Owl-1.5-32B-Think",
        default_url="http://localhost:8003/v1",
        max_total_tokens=131072,
        notes="Think variant emits reasoning traces — use for thought-trace probes.",
    ),
}


def is_local_model_id(model_name: str) -> bool:
    return model_name.startswith(LOCAL_PREFIX)


def strip_local_prefix(model_name: str) -> str:
    if not is_local_model_id(model_name):
        return model_name
    # also strip a possible trailing :thinking suffix (none of the local
    # models route through OpenRouter's thinking path)
    return model_name[len(LOCAL_PREFIX):].split(":")[0]


def resolve_local_model(model_name: str) -> LocalModelSpec:
    short = strip_local_prefix(model_name)
    if short not in LOCAL_MODELS:
        raise KeyError(
            f"Unknown local model {short!r}. "
            f"Registered: {sorted(LOCAL_MODELS)}. "
            f"Add a LocalModelSpec to models/common/registry.py."
        )
    return LOCAL_MODELS[short]
