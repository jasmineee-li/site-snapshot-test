"""AgentLab model args that honour a local server's URL.

AgentLab's `SelfHostedModelArgs(backend="vllm")` cannot reach a server on a
port other than 8000. Two things drop the URL on the way through:

* `SelfHostedModelArgs.make_model()` builds `VLLMChatModel` from four fields
  and never passes `model_url`.
* `VLLMChatModel.__init__` hardcodes `client_args={"base_url":
  "http://0.0.0.0:8000/v1"}`.

`models/common/registry.py` assigns port 8001 to `opencua-32b`, 8002 to
`opencua-72b` and 8003 to `gui-owl-32b-think`, so every local model in this
tree is unreachable through that path.

`LocalVLLMModelArgs` below overrides `make_model()` and builds the same
`ChatModel` with `base_url` taken from `model_url`. AgentLab stays read-only,
per the root `CLAUDE.md`.

This module imports AgentLab at import time, which is why it is separate from
`models/common/vllm_client.py`: that module is also imported by the
tool-calling path, which needs no AgentLab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agentlab.llm.chat_api import ChatModel, SelfHostedModelArgs
from openai import OpenAI

# vLLM accepts any non-empty token on its default served auth.
_DUMMY_API_KEY = "EMPTY"


@dataclass
class LocalVLLMModelArgs(SelfHostedModelArgs):
    """`SelfHostedModelArgs` that sends requests to `model_url`."""

    def make_model(self) -> Any:
        if not self.model_url:
            raise ValueError(
                "LocalVLLMModelArgs needs model_url. Set it to the server's "
                "OpenAI base URL, for example http://localhost:8003/v1."
            )
        return ChatModel(
            model_name=self.model_name,
            api_key=os.environ.get("LOCAL_OPENAI_API_KEY", _DUMMY_API_KEY),
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            max_retry=self.n_retry_server,
            client_class=OpenAI,
            client_args={"base_url": self.model_url},
            pricing_func=None,
        )
