"""Shared agent LLM configuration and factory utilities.

Provides a single ``make_llm`` function that returns the correct Browser Use
chat model for a given provider/model pair, plus shared helpers that were
previously duplicated between Phase 3 and Phase 4: ``make_agent_factory``
and ``run_tasks_by_site``.

Supported providers:

- ``google``      -- ``browser_use.llm.google``      (``ChatGoogle``)
- ``openai``      -- native OpenAI Responses API via ``ChatOpenAIResponses`` (Arm A)
- ``anthropic``   -- ``browser_use.llm.anthropic``   (``ChatAnthropic``)
- ``openrouter``  -- ``browser_use.llm.openrouter``  (``ChatOpenRouter``, Arm B)

A/B study design. Both arms use the same model (gpt-5.4-mini), the same
``AgentOutput`` Pydantic schema, and strict structured output. They differ
only in API surface:

  Arm A (``--agent-provider openai``): native Responses API.
    Uses ``OPENAI_API_KEY``.  Calls ``client.responses.parse`` with
    ``text_format=AgentOutput`` and ``reasoning={"effort": "none"}``.
    Reasoning is separated into its own output item; content arrives as
    clean JSON with no ``<think>`` leakage.

  Arm B (``--agent-provider openrouter``): OpenRouter pinned to OpenAI.
    Uses ``OPENROUTER_API_KEY``.  Calls chat completions with
    ``response_format=json_schema`` (strict) and
    ``extra_body.reasoning.exclude=true`` plus
    ``extra_body.provider.only=["openai"]``.  No fallbacks, no silent
    dropping of the reasoning flag (``require_parameters=true``).

Auth env vars: ``GOOGLE_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from worldsim.browser_use_agent import AgentRunner, BrowserUseAgent
from worldsim.config import BenchmarkInstance
from worldsim.eval_worker_pool import run_eval
from worldsim.instance_selection import select_task_site_instance
from worldsim.placeholders import (
    apply_placeholders,
    merge_placeholder_maps,
    normalize_site_name,
    normalize_task_sites,
    placeholder_for_site,
    placeholders_for_site_urls,
)

logger = logging.getLogger(__name__)

RUNTIME_METADATA_KEY = "_worldsim_runtime"

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-6"
SUPPORTED_PROVIDERS = ("google", "openai", "anthropic", "openrouter")

# Provider auto-detection: prefix -> provider name.
_PREFIX_MAP: list[tuple[str, str]] = [
    ("gemini", "google"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
]


def detect_provider(model: str) -> str | None:
    """Best-effort provider detection from the model name."""
    if "/" in model:
        return "openrouter"
    lower = model.lower()
    for prefix, provider in _PREFIX_MAP:
        if lower.startswith(prefix):
            return provider
    return None


def resolve_provider(model: str, provider: str | None) -> str:
    """Return the explicit or inferred provider for ``model``.

    Unknown model families should fail fast instead of silently defaulting to
    a provider that may not match the user's credentials.
    """
    if provider is not None:
        normalized = provider.lower()
        if normalized not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(f"Unknown provider {provider!r}. Supported: {supported}")
        return normalized

    detected = detect_provider(model)
    if detected is None:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Could not infer a provider for model {model!r}. "
            f"Pass provider explicitly ({supported}) or use a model name with a "
            "known prefix such as gemini, gpt-, o1/o3/o4, or claude."
        )
    return detected


# ── LLM construction ─────────────────────────────────────────────────────


def _anthropic_proxy_env() -> tuple[str, str] | None:
    """Return (base_url, auth_token) if an Anthropic-compatible proxy is configured."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return None
    base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    auth = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    return (base, auth) if base and auth else None


def _normalize_openrouter_model(model: str) -> str:
    """Normalize known OpenAI-family slugs for explicit OpenRouter usage."""
    if "/" in model:
        return model
    if model.startswith("gpt-"):
        return f"openai/{model}"
    return model


def _openrouter_overrides(model: str, service_tier: str | None = None) -> dict[str, Any]:
    """Return OpenRouter-specific overrides for Arm B of the A/B study.

    For gpt-5.4-mini (and the ``openai/`` prefixed variant):
    - ``reasoning.effort="none"`` suppresses reasoning token generation.
    - ``reasoning.exclude=true`` keeps reasoning out of the response body
      entirely (analogous to Responses API separating it into its own item).
    - ``provider.only=["openai"]`` pins the OpenAI upstream, no fallbacks.
    - ``require_parameters=true`` ensures the reasoning flag is not silently
      dropped if the upstream does not support it.
    - ``temperature=None`` lets OpenRouter use the model's default (required
      for reasoning models that reject an explicit temperature).

    ``service_tier`` (optional) is forwarded inside the same ``extra_body``
    envelope so OpenRouter passes it to the pinned OpenAI upstream.

    Note on ``extra_body`` double-nesting. Browser Use's ``ChatOpenRouter``
    splats its ``extra_body`` field as top-level kwargs to
    ``client.chat.completions.create``. The openai SDK treats unknown
    top-level kwargs as an error. To land these fields inside the JSON
    request body (where OpenRouter expects them), we nest them under a
    literal ``extra_body`` key so that after the splat the SDK sees
    ``extra_body={...}`` and forwards the contents as extra body fields.
    """
    if model in {"gpt-5.4-mini", "openai/gpt-5.4-mini"}:
        inner: dict[str, Any] = {
            "reasoning": {"effort": "none", "exclude": True},
            "provider": {
                "only": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if service_tier:
            inner["service_tier"] = service_tier
        return {
            "temperature": None,
            "extra_body": {"extra_body": inner},
        }
    if service_tier:
        return {"extra_body": {"extra_body": {"service_tier": service_tier}}}
    return {}


def make_llm(
    model: str = DEFAULT_MODEL,
    provider: str | None = None,
    temperature: float = 0,
    service_tier: str | None = None,
) -> Any:
    """Return a Browser Use ``BaseChatModel`` for the requested provider.

    Args:
        model: Model name string (e.g. ``gemini-3-flash-preview``,
            ``gpt-5.4``, ``claude-sonnet-4-6``).
        provider: One of ``google``, ``openai``, ``anthropic``,
            ``openrouter``.  When ``None`` the provider is auto-detected
            from *model*.
        temperature: Sampling temperature.
        service_tier: Optional OpenAI service tier (``"auto"``, ``"default"``,
            ``"flex"``, ``"priority"``). Only applied for ``openai`` and
            ``openrouter`` providers; ignored with a warning for others.

    Raises:
        RuntimeError: If the required browser-use LLM module is missing.
        ValueError: If *provider* is not recognised.
    """
    provider = resolve_provider(model, provider)

    if service_tier and provider not in ("openai", "openrouter"):
        logger.warning(
            "service_tier=%r is OpenAI-only; ignored for provider=%r",
            service_tier,
            provider,
        )
        service_tier = None

    # ``openai`` uses the native Responses API (Arm A of the A/B study).
    # It requires ``OPENAI_API_KEY`` directly -- no redirect to OpenRouter.
    # Arm B is explicitly selected with ``provider="openrouter"``.

    _PROVIDER_ENV_VARS = {
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        # openai intentionally absent: OPENAI_API_KEY is read by ChatOpenAI
        # automatically from the environment; we do not redirect it.
    }
    provider_key_var = _PROVIDER_ENV_VARS.get(provider)
    anthropic_proxy_ready = provider == "anthropic" and _anthropic_proxy_env() is not None
    if (
        provider not in ("openrouter", "openai")
        and not anthropic_proxy_ready
        and provider_key_var
        and not os.environ.get(provider_key_var, "").strip()
        and os.environ.get("OPENROUTER_API_KEY", "").strip()
    ):
        logger.info(
            "Provider %r requires %s (not set), falling back to OpenRouter",
            provider,
            provider_key_var,
        )
        provider = "openrouter"
        prefix_map = {"google": "google/", "anthropic": "anthropic/"}
        prefix = prefix_map.get(resolve_provider(model, None) or "", "")
        if prefix and not model.startswith(prefix):
            model = prefix + model
    elif provider == "openrouter":
        model = _normalize_openrouter_model(model)

    if provider == "google":
        try:
            from browser_use.llm.google.chat import ChatGoogle
        except ImportError as exc:
            raise RuntimeError(
                "browser-use[google] is required for the Google provider. "
                "Install it with: uv pip install browser-use"
            ) from exc
        return ChatGoogle(model=model, temperature=temperature)

    if provider == "openai":
        try:
            from browser_use.llm.openai.chat import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "browser-use is required for the OpenAI provider. "
                "Install it with: uv pip install browser-use"
            ) from exc
        from worldsim.llm_wrapper import ChatOpenAIResponses, is_reasoning_model

        if is_reasoning_model(model):
            # Arm A: native Responses API with reasoning={"effort": "none"}.
            # ChatOpenAIResponses.create reads OPENAI_API_KEY from the
            # environment automatically via the underlying AsyncOpenAI client.
            return ChatOpenAIResponses.create(model=model, service_tier=service_tier)
        # Non-reasoning OpenAI models: plain ChatOpenAI with temperature.
        openai_kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if service_tier:
            openai_kwargs["extra_body"] = {"service_tier": service_tier}
        return ChatOpenAI(**openai_kwargs)

    if provider == "anthropic":
        try:
            from browser_use.llm.anthropic.chat import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "browser-use is required for the Anthropic provider. "
                "Install it with: uv pip install browser-use"
            ) from exc
        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        proxy = _anthropic_proxy_env()
        if proxy is not None:
            # Route via ChatAnthropic's tool-calling path: it ignores `minimum`/`maximum`
            # in schemas, while the OpenAI-style response_format path (ChatOpenRouter) rejects them.
            base_url, auth_token = proxy
            kwargs["base_url"] = base_url
            kwargs["auth_token"] = auth_token
            logger.info("Anthropic provider: using custom base_url %s with auth_token", base_url)
        return ChatAnthropic(**kwargs)

    if provider == "openrouter":
        try:
            from browser_use.llm.openrouter.chat import ChatOpenRouter
        except ImportError as exc:
            raise RuntimeError(
                "browser-use is required for the OpenRouter provider. "
                "Install it with: uv pip install browser-use"
            ) from exc
        # Arm B: ``_openrouter_overrides`` sets reasoning.exclude=true and
        # pins provider.only=["openai"] with no fallbacks so the request
        # always goes to OpenAI and reasoning never appears in content.
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        }
        kwargs.update(_openrouter_overrides(model, service_tier=service_tier))
        return ChatOpenRouter(**kwargs)

    raise ValueError(f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")


# ── Agent factory ─────────────────────────────────────────────────────────


def make_agent_factory(
    model: str = DEFAULT_MODEL,
    provider: str | None = None,
    service_tier: str | None = None,
    llm_timeout: int | None = None,
    step_timeout: int | None = None,
) -> Callable[[], BrowserUseAgent]:
    """Return a zero-arg callable that produces a fresh ``BrowserUseAgent``.

    The LLM is built lazily on first call so import errors surface at
    runtime inside the worker, not at factory-creation time.
    ``llm_timeout`` and ``step_timeout`` are Browser Use agent deadlines. When
    omitted, Browser Use's provider/model defaults are preserved.
    """

    def factory() -> BrowserUseAgent:
        llm = make_llm(model=model, provider=provider, service_tier=service_tier)
        return BrowserUseAgent(
            llm=llm,
            headless=True,
            llm_timeout=llm_timeout,
            step_timeout=step_timeout,
        )

    return factory


# ── Shared task routing ───────────────────────────────────────────────────


async def run_tasks_by_site(
    tasks: list[dict[str, Any]],
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], AgentRunner],
    task_runner: Callable[[dict[str, Any], AgentRunner, BenchmarkInstance, Path], Any],
    task_dir_root: Path,
    config_url_placeholders: dict[str, str] | None = None,
    resume: bool = False,
    resume_fingerprint_builder: Callable[[dict[str, Any]], str] | None = None,
    result_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Run tasks only against instances for the same site.

    Shared between Phase 3 (benign) and Phase 4 (adversarial).
    """
    tasks_by_site: dict[str, list[dict[str, Any]]] = {}
    prepared_tasks, preparation_errors = prepare_tasks_for_execution(
        tasks,
        instances,
        config_url_placeholders=config_url_placeholders,
    )

    for task in prepared_tasks:
        runtime = task[RUNTIME_METADATA_KEY]
        tasks_by_site.setdefault(runtime["primary_site"], []).append(task)

    results: list[dict[str, Any]] = []
    results.extend(preparation_errors)
    batches = []
    for site_name, site_tasks in tasks_by_site.items():
        site_instances = instances_for_site(instances, site_name)
        if not site_instances:
            logger.error("No instances configured for site %r", site_name)
            results.extend(
                {
                    "task_id": task.get("id", "unknown"),
                    "passed": False,
                    "message": f"no instances configured for site {site_name!r}",
                }
                for task in site_tasks
            )
            continue
        batches.append(
            run_eval(
                tasks=site_tasks,
                instances=site_instances,
                agent_factory=agent_factory,
                task_runner=task_runner,
                task_dir_root=task_dir_root,
                task_binder=lambda task, instance, *, _instances=instances: bind_task_to_instance(
                    task,
                    instance,
                    _instances,
                ),
                resume=resume,
                expected_result_fingerprints=(
                    {
                        str(task.get("id", "unknown")): resume_fingerprint_builder(task)
                        for task in site_tasks
                    }
                    if resume_fingerprint_builder is not None
                    else None
                ),
                result_callback=result_callback,
            )
        )

    if batches:
        grouped_results = await asyncio.gather(*batches, return_exceptions=True)
        for batch in grouped_results:
            if isinstance(batch, BaseException):
                logger.error("Site batch failed: %s", batch)
                continue
            results.extend(batch)
    return results


def prepare_tasks_for_execution(
    tasks: list[dict[str, Any]],
    instances: list[BenchmarkInstance],
    *,
    config_url_placeholders: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare tasks for routing and return any static configuration errors."""
    all_placeholders = merge_placeholder_maps(
        config_url_placeholders,
        placeholders_for_site_urls(
            (instance.site_name, instance.site_url) for instance in instances
        ),
    )
    prepared_tasks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for task in tasks:
        prepared_task, missing_sites = prepare_task_for_execution(
            task,
            instances,
            config_url_placeholders=all_placeholders,
        )
        if missing_sites:
            missing = ", ".join(sorted(missing_sites))
            logger.error(
                "Task %s requires sites with no configured instances: %s",
                task.get("id", "unknown"),
                missing,
            )
            errors.append(
                {
                    "task_id": str(task.get("id", "unknown")),
                    "passed": False,
                    "outcome": "error",
                    "message": f"missing configured instances for sites: {missing}",
                }
            )
            continue
        prepared_tasks.append(prepared_task)

    return prepared_tasks, errors


def instances_for_site(
    instances: list[BenchmarkInstance],
    site_name: str,
) -> list[BenchmarkInstance]:
    """Filter instances matching *site_name* (case-insensitive)."""
    normalized = normalize_site_name(site_name)
    return [
        instance for instance in instances if normalize_site_name(instance.site_name) == normalized
    ]


def prepare_task_for_execution(
    task: dict[str, Any],
    instances: list[BenchmarkInstance],
    *,
    config_url_placeholders: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Attach runtime metadata to a task for execution-time resolution."""
    task_sites = normalize_task_sites(task)
    if not task_sites:
        task_sites = [normalize_site_name(task.get("site", ""))]

    primary_site = normalize_site_name(task.get("site", "")) or (
        task_sites[0] if task_sites else ""
    )
    missing_sites: list[str] = []

    for site_name in task_sites:
        site_instances = instances_for_site(instances, site_name)
        if not site_instances:
            missing_sites.append(site_name)

    prepared = json.loads(json.dumps(task))
    prepared["site"] = primary_site
    prepared["sites"] = task_sites
    existing_runtime = prepared.get(RUNTIME_METADATA_KEY, {})
    prepared[RUNTIME_METADATA_KEY] = {
        "primary_site": primary_site,
        "sites": task_sites,
        "base_url_placeholders": merge_placeholder_maps(
            existing_runtime.get("base_url_placeholders"),
            config_url_placeholders,
        ),
        "reset_endpoints": [],
        "url_placeholders": {},
        "bound_instance": None,
        "bound_instances": {},
    }
    return prepared, missing_sites


def bind_task_to_instance(
    task: dict[str, Any],
    instance: BenchmarkInstance,
    instances: list[BenchmarkInstance],
) -> dict[str, Any]:
    """Bind a prepared task to the concrete instance chosen for execution."""
    prepared, missing_sites = prepare_task_for_execution(task, instances)
    if missing_sites:
        raise ValueError(
            "missing configured instances for sites: " + ", ".join(sorted(missing_sites))
        )

    runtime = prepared[RUNTIME_METADATA_KEY]
    primary_site = runtime["primary_site"]
    instance_site = normalize_site_name(instance.site_name)
    if primary_site and instance_site != primary_site:
        raise ValueError(
            f"cannot bind task for primary site {primary_site!r} to instance {instance.site_name!r}"
        )

    placeholders = merge_placeholder_maps(runtime.get("base_url_placeholders"))
    reset_endpoints: list[str] = []
    bound_instances: dict[str, dict[str, Any]] = {}
    for site_name in runtime["sites"]:
        site_instance = (
            instance
            if site_name == primary_site
            else select_task_site_instance(task, site_name, instances)
        )
        bound_instances[site_name] = site_instance.model_dump()
        placeholders = merge_placeholder_maps(
            placeholders,
            site_instance.url_placeholders,
            placeholders_for_site_urls([(site_instance.site_name, site_instance.site_url)]),
        )
        if site_instance.reset_endpoint and site_instance.reset_endpoint not in reset_endpoints:
            reset_endpoints.append(site_instance.reset_endpoint)

    bound_task = json.loads(json.dumps(prepared))
    bound_task[RUNTIME_METADATA_KEY] = {
        **runtime,
        "reset_endpoints": reset_endpoints,
        "url_placeholders": placeholders,
        "url_origin_rewrites": _url_origin_rewrites_for_bound_sites(
            bound_instances=bound_instances,
            instances=instances,
            base_url_placeholders=runtime.get("base_url_placeholders"),
        ),
        "bound_instance": instance.model_dump(),
        "bound_instances": bound_instances,
    }
    return _rewrite_task_bound_origins(
        bound_task, bound_instances=bound_instances, instances=instances
    )


def _rewrite_task_bound_origins(
    task: dict[str, Any],
    *,
    bound_instances: dict[str, dict[str, Any]],
    instances: list[BenchmarkInstance],
) -> dict[str, Any]:
    origin_mapping: dict[str, str] = {}
    for site_name, bound_payload in bound_instances.items():
        new_site_url = bound_payload.get("site_url")
        if not isinstance(new_site_url, str) or not new_site_url.strip():
            continue
        new_origin = _origin_for_url(new_site_url)
        for candidate in instances_for_site(instances, site_name):
            old_origin = _origin_for_url(candidate.site_url)
            if old_origin and old_origin != new_origin:
                origin_mapping[old_origin] = new_origin
    if not origin_mapping:
        return task
    return _rewrite_nested_origins(task, origin_mapping)


def _rewrite_nested_origins(value: Any, origin_mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_nested_origins(item, origin_mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_nested_origins(item, origin_mapping) for item in value]
    if isinstance(value, str):
        return _rewrite_string_origin(value, origin_mapping)
    return value


def _rewrite_string_origin(value: str, origin_mapping: dict[str, str]) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        parsed = None
    if parsed is None:
        rewritten = value
        for old_origin, new_origin in origin_mapping.items():
            if old_origin in rewritten:
                rewritten = rewritten.replace(old_origin, new_origin)
        return rewritten
    if parsed.scheme and parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        replacement = origin_mapping.get(origin)
        if replacement:
            replacement_parts = urlsplit(replacement)
            return urlunsplit(
                (
                    replacement_parts.scheme,
                    replacement_parts.netloc,
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
    rewritten = value
    for old_origin, new_origin in origin_mapping.items():
        if old_origin in rewritten:
            rewritten = rewritten.replace(old_origin, new_origin)
    return rewritten


def _origin_for_url(url: str) -> str:
    parsed = urlsplit(str(url))
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _url_origin_rewrites_for_bound_sites(
    *,
    bound_instances: dict[str, dict[str, Any]],
    instances: list[BenchmarkInstance],
    base_url_placeholders: dict[str, str] | None,
) -> dict[str, str]:
    """Return same-site browser-origin aliases that should stay on the bound replica.

    GitLab and some WebArena services emit absolute links using their
    configured canonical origin. In scaled runs that canonical origin can be
    the first replica or a loopback alias such as ``localhost:8023`` even when
    the task was seeded on another replica. These rewrites are limited to
    known benchmark origins for the same logical site.
    """
    rewrites: dict[str, str] = {}
    for site_name, bound_payload in bound_instances.items():
        bound_origin = _origin_for_url(str(bound_payload.get("site_url") or ""))
        if not bound_origin:
            continue
        aliases: set[str] = set()
        site_token = placeholder_for_site(site_name)
        if site_token and isinstance(base_url_placeholders, dict):
            aliases.update(_origin_aliases_for_url(base_url_placeholders.get(site_token, "")))
        for candidate in instances_for_site(instances, site_name):
            aliases.update(_origin_aliases_for_url(candidate.site_url))
            if site_token:
                aliases.update(
                    _origin_aliases_for_url(candidate.url_placeholders.get(site_token, ""))
                )
        if site_token:
            placeholders = bound_payload.get("url_placeholders")
            if isinstance(placeholders, dict):
                aliases.update(_origin_aliases_for_url(placeholders.get(site_token, "")))
        for alias in aliases:
            if alias and alias != bound_origin:
                rewrites[alias] = bound_origin
    return rewrites


def _origin_aliases_for_url(url: str) -> set[str]:
    parsed = urlsplit(str(url or ""))
    if not parsed.scheme or not parsed.netloc:
        return set()
    aliases = {f"{parsed.scheme}://{parsed.netloc}"}
    if parsed.port is not None:
        aliases.add(f"{parsed.scheme}://localhost:{parsed.port}")
        aliases.add(f"{parsed.scheme}://127.0.0.1:{parsed.port}")
    return aliases


def execution_instance_dict(
    instance: BenchmarkInstance,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Merge execution-time task metadata into the primary instance dict."""
    runtime = task.get(RUNTIME_METADATA_KEY, {})
    bound_instance = runtime.get("bound_instance")
    instance_dict = (
        dict(bound_instance) if isinstance(bound_instance, dict) else instance.model_dump()
    )
    instance_dict["url_placeholders"] = merge_placeholder_maps(
        instance_dict.get("url_placeholders"),
        runtime.get("url_placeholders"),
    )
    origin_rewrites = runtime.get("url_origin_rewrites")
    if isinstance(origin_rewrites, dict):
        instance_dict["url_origin_rewrites"] = {
            str(key): str(value)
            for key, value in origin_rewrites.items()
            if isinstance(key, str) and isinstance(value, str) and key and value
        }
    instance_dict["site_name"] = normalize_site_name(
        instance_dict.get("site_name", instance.site_name)
    )
    return instance_dict


def execution_site_instance_dict(
    instance: BenchmarkInstance,
    task: dict[str, Any],
    *,
    site_name: str,
) -> dict[str, Any]:
    """Return an execution instance dict for a specific bound site."""
    normalized_site = normalize_site_name(site_name)
    runtime = task.get(RUNTIME_METADATA_KEY, {})
    bound_instances = runtime.get("bound_instances")
    if isinstance(bound_instances, dict):
        bound_instance = bound_instances.get(normalized_site)
        if isinstance(bound_instance, dict):
            instance_dict = dict(bound_instance)
            instance_dict["url_placeholders"] = merge_placeholder_maps(
                instance_dict.get("url_placeholders"),
                runtime.get("url_placeholders"),
            )
            origin_rewrites = runtime.get("url_origin_rewrites")
            if isinstance(origin_rewrites, dict):
                instance_dict["url_origin_rewrites"] = {
                    str(key): str(value)
                    for key, value in origin_rewrites.items()
                    if isinstance(key, str) and isinstance(value, str) and key and value
                }
            instance_dict["site_name"] = normalize_site_name(
                instance_dict.get("site_name", normalized_site)
            )
            return instance_dict

    if normalize_site_name(instance.site_name) == normalized_site:
        return execution_instance_dict(instance, task)

    raise ValueError(f"task is not bound to site {site_name!r}")


def resolve_task_inputs(
    task: dict[str, Any],
    instance_dict: dict[str, Any],
) -> tuple[str, list[str]]:
    """Resolve placeholders in task instruction and start URLs."""
    placeholders = instance_dict.get("url_placeholders", {})
    instruction = apply_placeholders(
        str(task.get("instruction", "")),
        placeholders,
        strict=True,
    )
    start_urls = [
        apply_placeholders(str(url), placeholders, strict=True)
        for url in task.get("start_urls", [])
    ]
    return instruction, start_urls


def task_reset_endpoints(task: dict[str, Any]) -> list[str]:
    """Return the reset endpoints required for a task."""
    runtime = task.get(RUNTIME_METADATA_KEY, {})
    return list(runtime.get("reset_endpoints", []))


# ── Per-site task cap ────────────────────────────────────────────────────

_CAP_SEED = 42


def _cap_primary_site(task: dict[str, Any]) -> str:
    """Return the normalized primary site used for per-site capping."""
    return normalize_site_name(str(task.get("site", ""))) or "unknown"


def _cap_task_identity(task: dict[str, Any]) -> str:
    """Return a stable identity string for deterministic capped sampling."""
    task_id = task.get("id")
    if task_id not in (None, ""):
        return str(task_id)
    return json.dumps(task, sort_keys=True, separators=(",", ":"))


def cap_tasks_per_site(
    tasks: list[dict[str, Any]],
    max_per_site: int,
) -> list[dict[str, Any]]:
    """Return at most *max_per_site* tasks per primary site.

    Selection uses a fixed random seed so the same cap always produces the
    same subset for a given site — deterministic, normalized, and independent
    of input ordering or unrelated sites. Tasks are grouped by normalized
    primary site, which matches how ``run_tasks_by_site`` routes them.

    The returned list preserves the original task order (stable selection).
    """
    if max_per_site <= 0:
        raise ValueError("max_per_site must be a positive integer")

    # Group task indices by primary site.
    site_indices: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for i, task in enumerate(tasks):
        site_indices[_cap_primary_site(task)].append((_cap_task_identity(task), i))

    # Sample up to max_per_site indices per site. Use a site-local RNG and
    # identity-sorted candidates so each site's selection stays stable even if
    # other sites or task ordering change.
    selected_indices: set[int] = set()
    for site in sorted(site_indices):  # sorted for determinism
        entries = sorted(site_indices[site])
        if len(entries) <= max_per_site:
            selected_indices.update(index for _, index in entries)
        else:
            rng = random.Random(f"{_CAP_SEED}:{site}")
            selected_indices.update(index for _, index in rng.sample(entries, max_per_site))

    # Return in original order
    return [tasks[i] for i in sorted(selected_indices)]
