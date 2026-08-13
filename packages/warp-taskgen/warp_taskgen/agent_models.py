"""Browser-agent model routing profiles shared by runner adapters.

The profiles here are configuration data only. They intentionally avoid
provider SDK imports so they can be used by the core WorldSim environment and
serialized across the isolated AgentLab sidecar boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentModelTransport = Literal["openrouter", "litellm"]


@dataclass(frozen=True)
class AgentModelProfile:
    """Resolved browser-agent model route for comparison/adversarial runners."""

    key: str
    display_name: str
    provider: str
    model: str
    transport: AgentModelTransport
    transport_model: str
    required_env_var: str
    temperature: float | None
    max_total_tokens: int = 128_000
    max_input_tokens: int = 126_000
    max_new_tokens: int = 4_096
    vision_support: bool = False
    aliases: tuple[str, ...] = ()
    extra_body: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_sidecar_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for the AgentLab sidecar."""

        return {
            "key": self.key,
            "display_name": self.display_name,
            "provider": self.provider,
            "model": self.model,
            "transport": self.transport,
            "transport_model": self.transport_model,
            "required_env_var": self.required_env_var,
            "temperature": self.temperature,
            "max_total_tokens": self.max_total_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "vision_support": self.vision_support,
            "extra_body": self.extra_body,
            "notes": self.notes,
        }


_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def _model_key(value: object) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _openrouter_extra_body(model: str, service_tier: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    provider_only = _openrouter_provider_pins(model)
    if provider_only:
        body["provider"] = {
            "only": list(provider_only),
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if model.startswith("openai/"):
        body["reasoning"] = {"effort": "none", "exclude": True}
        if service_tier:
            body["service_tier"] = service_tier
    return body


def _openrouter_provider_pins(model: str) -> tuple[str, ...]:
    """Return OpenRouter provider slugs for named research profiles."""

    if model.startswith("openai/"):
        return ("openai",)
    if model.startswith("anthropic/"):
        return ("anthropic",)
    if model.startswith("google/"):
        return ("google-vertex",)
    if model == "moonshotai/kimi-k2.5":
        return ("moonshotai",)
    if model.startswith("z-ai/"):
        return ("z-ai",)
    return ()


def _temperature_for_model(model: str) -> float | None:
    normalized = model.strip().lower().replace("_", "-")
    if "claude-opus-4.7" in normalized or "claude-opus-4-7" in normalized:
        return None
    if normalized.startswith("openai/gpt-5") or normalized.startswith("gpt-5"):
        return None
    if normalized == "moonshotai/kimi-k2.5":
        return None
    return 0


def _openrouter_profile(
    *,
    key: str,
    display_name: str,
    model: str,
    aliases: tuple[str, ...],
    vision_support: bool,
    max_new_tokens: int = 4_096,
    notes: str = "",
    service_tier: str | None = None,
) -> AgentModelProfile:
    return AgentModelProfile(
        key=key,
        display_name=display_name,
        provider="openrouter",
        model=model,
        transport="openrouter",
        transport_model=model,
        required_env_var="OPENROUTER_API_KEY",
        temperature=_temperature_for_model(model),
        max_new_tokens=max_new_tokens,
        vision_support=vision_support,
        aliases=aliases,
        extra_body=_openrouter_extra_body(model, service_tier=service_tier),
        notes=notes,
    )


def _canonical_openrouter_profiles(service_tier: str | None = None) -> dict[str, AgentModelProfile]:
    profiles = [
        _openrouter_profile(
            key="opus47",
            display_name="Claude Opus 4.7",
            model="anthropic/claude-opus-4.7",
            aliases=(
                "opus47",
                "opus-4.7",
                "opus 4.7",
                "claude-opus-4.7",
                "claude-opus-4-7",
                "anthropic/claude-opus-4.7",
            ),
            vision_support=True,
            notes="Omit temperature; Opus 4.7 rejects explicit sampling temperature.",
            service_tier=service_tier,
        ),
        _openrouter_profile(
            key="sonnet46",
            display_name="Claude Sonnet 4.6",
            model="anthropic/claude-sonnet-4.6",
            aliases=(
                "sonnet46",
                "sonnet-4.6",
                "sonnet 4.6",
                "claude-sonnet-4.6",
                "claude-sonnet-4-6",
                "anthropic/claude-sonnet-4.6",
            ),
            vision_support=True,
            service_tier=service_tier,
        ),
        _openrouter_profile(
            key="gemini25pro",
            display_name="Gemini 2.5 Pro",
            model="google/gemini-2.5-pro",
            aliases=(
                "gemini25pro",
                "gemini-2.5-pro",
                "gemini 2.5 pro",
                "google/gemini-2.5-pro",
            ),
            vision_support=True,
            service_tier=service_tier,
        ),
        _openrouter_profile(
            key="kimik25",
            display_name="Kimi K2.5",
            model="moonshotai/kimi-k2.5",
            aliases=(
                "kimik25",
                "kimi-k25",
                "kimi-k2.5",
                "kimi k2.5",
                "moonshotai/kimi-k2.5",
            ),
            vision_support=False,
            notes=(
                "Keep explicit reasoning disabled/omitted until the runner preserves "
                "OpenRouter reasoning details across turns."
            ),
            service_tier=service_tier,
        ),
        _openrouter_profile(
            key="gpt52",
            display_name="GPT-5.2",
            model="openai/gpt-5.2",
            aliases=("gpt52", "gpt-5.2", "openai/gpt-5.2"),
            vision_support=True,
            max_new_tokens=4_096,
            notes="Pin OpenRouter upstream to OpenAI with no fallbacks.",
            service_tier=service_tier,
        ),
        _openrouter_profile(
            key="glm5",
            display_name="GLM-5",
            model="z-ai/glm-5",
            aliases=("glm5", "glm-5", "z-ai/glm-5"),
            vision_support=False,
            max_new_tokens=4_096,
            service_tier=service_tier,
        ),
    ]
    return {profile.key: profile for profile in profiles}


def supported_agentlab_model_profiles() -> list[AgentModelProfile]:
    """Return the named AgentLab comparison profiles in display order."""

    return list(_canonical_openrouter_profiles().values())


def _alias_map(service_tier: str | None = None) -> dict[str, AgentModelProfile]:
    aliases: dict[str, AgentModelProfile] = {}
    for profile in _canonical_openrouter_profiles(service_tier=service_tier).values():
        aliases[_model_key(profile.key)] = profile
        aliases[_model_key(profile.model)] = profile
        for alias in profile.aliases:
            aliases[_model_key(alias)] = profile
    return aliases


def _openrouter_slug(model: str) -> str:
    normalized = model.strip()
    if normalized.startswith("openrouter/"):
        return normalized.removeprefix("openrouter/")
    if "/" in normalized:
        return normalized
    if normalized.startswith("gpt-"):
        return f"openai/{normalized}"
    return normalized


def _direct_litellm_model(model: str, provider: str) -> str:
    model = model.strip()
    if provider == "google" and model.startswith("google/"):
        return f"gemini/{model.split('/', 1)[1]}"
    if "/" in model:
        return model
    prefixes = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "gemini",
    }
    prefix = prefixes.get(provider)
    return f"{prefix}/{model}" if prefix else model


def _required_env_for_provider(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, "")


def _direct_profile(
    *,
    model: str,
    provider: str,
    service_tier: str | None = None,
    vision_support: bool = True,
) -> AgentModelProfile:
    transport_model = _direct_litellm_model(model, provider)
    extra_body: dict[str, Any] = {}
    if provider == "openai" and service_tier:
        extra_body["service_tier"] = service_tier
    return AgentModelProfile(
        key=f"{provider}:{model}",
        display_name=model,
        provider=provider,
        model=model,
        transport="litellm",
        transport_model=transport_model,
        required_env_var=_required_env_for_provider(provider),
        temperature=_temperature_for_model(transport_model),
        max_new_tokens=4_096,
        vision_support=vision_support,
        extra_body=extra_body,
        notes="Native provider route through the AgentLab sidecar LiteLLM transport.",
    )


def resolve_agent_model_profile(
    model: str,
    provider: str | None = None,
    *,
    service_tier: str | None = None,
    default_provider: str = "openrouter",
) -> AgentModelProfile:
    """Resolve a user model/provider pair into an explicit sidecar route.

    Named aliases use the OpenRouter-smoked matrix by default. Passing
    ``provider=openai|anthropic|google`` asks for a native LiteLLM route when
    that provider can plausibly serve the requested model.
    """

    requested_model = str(model or "").strip()
    if not requested_model:
        raise ValueError("agent model must be a non-empty string")
    provider_name = str(provider or default_provider or "openrouter").strip().lower()
    if provider_name == "auto":
        provider_name = default_provider

    if provider_name == "openrouter":
        alias = _alias_map(service_tier=service_tier).get(_model_key(requested_model))
        if alias is not None:
            return alias
        slug = _openrouter_slug(requested_model)
        return _openrouter_profile(
            key=f"openrouter:{slug}",
            display_name=slug,
            model=slug,
            aliases=(requested_model,),
            vision_support=False,
            service_tier=service_tier,
        )

    if provider_name not in {"openai", "anthropic", "google"}:
        raise ValueError(f"unknown agent provider {provider!r}")

    alias = _alias_map(service_tier=service_tier).get(_model_key(requested_model))
    if alias is not None:
        if provider_name == "openai" and alias.model.startswith("openai/"):
            return _direct_profile(
                model=alias.model.split("/", 1)[1],
                provider=provider_name,
                service_tier=service_tier,
                vision_support=alias.vision_support,
            )
        if provider_name == "anthropic" and alias.model.startswith("anthropic/"):
            return _direct_profile(
                model=alias.model.split("/", 1)[1],
                provider=provider_name,
                service_tier=service_tier,
                vision_support=alias.vision_support,
            )
        if provider_name == "google" and alias.model.startswith("google/"):
            return _direct_profile(
                model=alias.model.split("/", 1)[1],
                provider=provider_name,
                service_tier=service_tier,
                vision_support=alias.vision_support,
            )
        raise ValueError(
            f"model alias {requested_model!r} resolves to {alias.model!r}, "
            f"which is not served by provider {provider_name!r}; use provider='openrouter'"
        )

    return _direct_profile(
        model=requested_model,
        provider=provider_name,
        service_tier=service_tier,
        vision_support=True,
    )
