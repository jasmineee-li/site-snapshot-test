"""WorldSim chat-model adapter for AgentLab's GenericAgent.

AgentLab's ``GenericAgent`` uses the older chat-model protocol: call the model
with a ``Discussion`` and return an AgentLab ``AIMessage`` containing text that
the prompt parser turns into browser actions. The newer AgentLab LiteLLM model
path is tool/API-payload oriented, and the older built-in LiteLLM chat wrapper
does not forward OpenRouter routing controls. This module supplies the small
protocol object WorldSim needs without patching AgentLab.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorldSimChatModelArgs:
    """Pickleable AgentLab-compatible model args."""

    model_name: str
    transport: str = "openrouter"
    provider: str = "openrouter"
    required_env_var: str = ""
    max_total_tokens: int | None = 128_000
    max_input_tokens: int | None = 126_000
    max_new_tokens: int | None = 4_096
    temperature: float | None = None
    vision_support: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_retry: int = 4
    min_retry_wait_time: float = 20.0
    display_name: str = ""
    profile_key: str = ""
    metadata_path: str | None = None

    def make_model(self) -> WorldSimChatModel:
        return WorldSimChatModel(
            model_name=self.model_name,
            transport=self.transport,
            provider=self.provider,
            required_env_var=self.required_env_var,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_body=self.extra_body,
            max_retry=self.max_retry,
            min_retry_wait_time=self.min_retry_wait_time,
            display_name=self.display_name,
            profile_key=self.profile_key,
            metadata_path=self.metadata_path,
        )

    def prepare_server(self) -> None:
        return None

    def close_server(self) -> None:
        return None


class WorldSimChatModel:
    """Chat model callable matching AgentLab ``AbstractChatModel`` behavior."""

    def __init__(
        self,
        *,
        model_name: str,
        transport: str,
        provider: str,
        required_env_var: str,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any],
        max_retry: int,
        min_retry_wait_time: float,
        display_name: str = "",
        profile_key: str = "",
        metadata_path: str | None = None,
    ):
        self.model_name = model_name
        self.transport = transport
        self.provider = provider
        self.required_env_var = required_env_var
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.display_name = display_name
        self.profile_key = profile_key
        self.metadata_path = metadata_path
        self.retries = 0
        self.success = False
        self.error_types: list[str] = []
        self.last_usage: dict[str, int] = {}
        self.last_response_model: str | None = None

    def __call__(self, messages: Any, n_samples: int = 1, temperature: float | None = None) -> dict:
        self.retries = 0
        self.success = False
        self.error_types = []
        self.last_usage = {}
        self.last_response_model = None
        response: Any = None
        last_error: Exception | None = None

        for attempt in range(self.max_retry):
            self.retries += 1
            try:
                params = self._build_api_params(
                    messages,
                    n_samples=n_samples,
                    temperature=temperature,
                )
                response = self._call_api(params)
                self.success = True
                break
            except Exception as exc:
                last_error = exc
                self.error_types.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 >= self.max_retry:
                    raise
                time.sleep(self.min_retry_wait_time)

        if response is None:
            raise RuntimeError(f"model call failed without response: {last_error}")

        self._record_usage(response)
        self._record_call_metadata(response)
        content = self._extract_content(response)
        from agentlab.llm.llm_utils import AIMessage

        return AIMessage(content)

    def _call_api(self, params: dict[str, Any]) -> Any:
        if self.required_env_var:
            import os

            if not os.environ.get(self.required_env_var, "").strip():
                raise RuntimeError(
                    f"{self.required_env_var} must be set for AgentLab model "
                    f"{self.model_name!r}"
                )

        if self.transport == "openrouter":
            from openai import OpenAI

            client = OpenAI(
                api_key=_env_value(self.required_env_var),
                base_url="https://openrouter.ai/api/v1",
            )
            return client.chat.completions.create(**params)

        if self.transport == "litellm":
            from litellm import completion

            return completion(**params, num_retries=5)

        raise ValueError(f"unknown WorldSim AgentLab model transport {self.transport!r}")

    def _build_api_params(
        self,
        messages: Any,
        *,
        n_samples: int = 1,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._coerce_messages(messages),
            "n": n_samples,
        }
        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature is not None:
            params["temperature"] = effective_temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.extra_body:
            params["extra_body"] = dict(self.extra_body)
        return params

    @staticmethod
    def _coerce_messages(messages: Any) -> list[dict[str, Any]]:
        if hasattr(messages, "to_openai"):
            messages = messages.to_openai()
        return [WorldSimChatModel._coerce_message(message) for message in messages]

    @staticmethod
    def _coerce_message(message: Any) -> dict[str, Any]:
        raw = dict(message)
        return {
            "role": str(raw.get("role", "user")),
            "content": WorldSimChatModel._coerce_content(raw.get("content", "")),
        }

    @staticmethod
    def _coerce_content(content: Any) -> Any:
        if not isinstance(content, list):
            return "" if content is None else content
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                converted.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                converted.append({"type": "text", "text": str(item.get("text", ""))})
            elif item_type in {"image_url", "input_image"}:
                image_url = item.get("image_url") or item.get("image")
                if isinstance(image_url, str):
                    image_url = {"url": image_url}
                converted.append({"type": "image_url", "image_url": image_url})
            else:
                converted.append(dict(item))
        return converted

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.last_response_model = getattr(response, "model", None)
        if usage is None:
            return
        input_tokens = _int_attr(usage, "prompt_tokens")
        output_tokens = _int_attr(usage, "completion_tokens")
        self.last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": _int_attr(usage, "total_tokens"),
        }
        try:
            import agentlab.llm.tracking as tracking

            if hasattr(tracking.TRACKER, "instance") and isinstance(
                tracking.TRACKER.instance,
                tracking.LLMTracker,
            ):
                tracking.TRACKER.instance(input_tokens, output_tokens, 0.0)
        except Exception:
            return

    def _record_call_metadata(self, response: Any) -> None:
        if not self.metadata_path:
            return
        payload = {
            "profile_key": self.profile_key,
            "display_name": self.display_name,
            "transport": self.transport,
            "provider": self.provider,
            "request_model": self.model_name,
            "response_model": getattr(response, "model", None),
            "response_provider": getattr(response, "provider", None),
            "usage": self.last_usage,
        }
        path = Path(self.metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", {})
        if not isinstance(message, dict):
            to_dict = getattr(message, "to_dict", None)
            message = to_dict() if callable(to_dict) else vars(message)
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(part for part in parts if part)
        if content is None:
            content = message.get("text", "")
        return str(content or "").removesuffix("<|end|>").strip()

    def get_stats(self) -> dict[str, Any]:
        # AgentLab aggregates stats with numpy sum/max, so only numeric values
        # may be returned here. Route/model metadata is persisted separately in
        # the sidecar result payload.
        stats: dict[str, Any] = {"n_retry_llm": self.retries}
        stats.update({f"worldsim_{key}": value for key, value in self.last_usage.items()})
        return stats


def model_args_from_request(request: dict[str, Any]) -> WorldSimChatModelArgs:
    profile = request.get("model_profile")
    if isinstance(profile, dict):
        transport = str(profile.get("transport") or "openrouter")
        if transport not in {"openrouter", "litellm"}:
            raise ValueError(f"model_profile has invalid transport {transport!r}")
        vision_support = profile.get("vision_support", False)
        if not isinstance(vision_support, bool):
            raise ValueError("model_profile field 'vision_support' must be a boolean")
        return WorldSimChatModelArgs(
            model_name=_required_profile_str(profile, "transport_model"),
            transport=transport,
            provider=_optional_profile_str(profile, "provider")
            or str(request.get("provider") or "openrouter"),
            required_env_var=_optional_profile_str(profile, "required_env_var") or "",
            max_total_tokens=_optional_int(profile.get("max_total_tokens"), 128_000),
            max_input_tokens=_optional_int(profile.get("max_input_tokens"), 126_000),
            max_new_tokens=_optional_int(profile.get("max_new_tokens"), 4_096),
            temperature=_optional_float(profile.get("temperature")),
            vision_support=vision_support,
            extra_body=_required_dict(profile.get("extra_body")),
            display_name=_optional_profile_str(profile, "display_name") or "",
            profile_key=_optional_profile_str(profile, "key") or "",
            metadata_path=_optional_profile_str(request, "model_metadata_path"),
        )

    model = str(request.get("model") or "").strip()
    provider = str(request.get("provider") or "").strip().lower()
    if provider == "openrouter":
        transport_model = model.removeprefix("openrouter/")
        transport = "openrouter"
        required_env_var = "OPENROUTER_API_KEY"
    else:
        transport_model = model
        transport = "litellm"
        required_env_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(provider, "")
    return WorldSimChatModelArgs(
        model_name=transport_model,
        transport=transport,
        provider=provider or "auto",
        required_env_var=required_env_var,
        temperature=_default_temperature(model),
        vision_support=bool(request.get("vision_support", True)),
        metadata_path=_optional_profile_str(request, "model_metadata_path"),
    )


def _required_profile_str(profile: dict[str, Any], key: str) -> str:
    value = profile.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model_profile missing required string field {key!r}")
    return value


def _optional_profile_str(profile: dict[str, Any], key: str) -> str | None:
    value = profile.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"model_profile field {key!r} must be a string")
    return value


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _required_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("model_profile field 'extra_body' must be an object")
    return dict(value)


def _default_temperature(model: str) -> float | None:
    normalized = model.lower()
    if normalized.startswith("openai/gpt-5") or normalized.startswith("gpt-5"):
        return None
    return 0


def _env_value(name: str) -> str | None:
    if not name:
        return None
    import os

    return os.environ.get(name)


def _int_attr(value: Any, attr: str) -> int:
    if isinstance(value, dict):
        raw = value.get(attr, 0)
    else:
        raw = getattr(value, attr, 0)
    return int(raw or 0)
